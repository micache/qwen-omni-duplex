"""Qwen2.5-Omni Thinker wrapper for the fixed duplex event timeline."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from .timeline import IGNORE_LABEL, ControlTokenIds, TimelineSpec


MODEL_ID = "Qwen/Qwen2.5-Omni-3B"
MODEL_COMPONENT = "thinker"
FUSION = "additive_audio_text_control"
OUTPUT = "text_only"

_GROUPS = ("text", "idle", "start", "stop")


def _config_value(container: object, name: str, path: str) -> Any:
    if isinstance(container, Mapping):
        if name not in container:
            raise ValueError(f"Qwen config is missing {path}.")
        return container[name]
    if not hasattr(container, name):
        raise ValueError(f"Qwen config is missing {path}.")
    return getattr(container, name)


@dataclass(frozen=True)
class NextEventLossWeights:
    """Per-target weights for the direct, already-shifted event objective."""

    text: float = 1.0
    idle: float = 1.0
    start: float = 1.0
    stop: float = 1.0

    def __post_init__(self) -> None:
        for name in _GROUPS:
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise ValueError(f"{name} loss weight must be finite and non-negative.")
        if not any(getattr(self, name) > 0 for name in _GROUPS):
            raise ValueError("At least one next-event loss weight must be positive.")

    def as_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in _GROUPS}


@dataclass
class DuplexThinkerOutput:
    """Text prediction output plus tensors needed for training diagnostics."""

    loss: torch.Tensor | None
    logits: torch.Tensor
    past_key_values: object | None
    lexical_hidden_states: torch.Tensor
    hidden_states: object | None
    attentions: object | None
    group_losses: dict[str, torch.Tensor]
    target_counts: dict[str, torch.Tensor]
    prediction_counts: dict[str, torch.Tensor]
    loss_weight_sum: torch.Tensor | None


class QwenDuplexThinker(nn.Module):
    """Add audio, text, and control embeddings before the original Thinker LM.

    ``qwen_config`` is the root Qwen2.5-Omni config, not only its Thinker
    sub-config: the three control IDs live under ``talker_config`` even though
    no Talker module is constructed or used.
    """

    def __init__(
        self,
        thinker: nn.Module,
        *,
        qwen_config: object,
        loss_weights: NextEventLossWeights | Mapping[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.thinker = thinker
        self.control_tokens = ControlTokenIds.from_qwen_config(qwen_config)

        thinker_config = _config_value(qwen_config, "thinker_config", "thinker_config")
        seconds_per_chunk = _config_value(
            thinker_config,
            "seconds_per_chunk",
            "thinker_config.seconds_per_chunk",
        )
        positions_per_second = _config_value(
            thinker_config,
            "position_id_per_seconds",
            "thinker_config.position_id_per_seconds",
        )
        self.timeline = TimelineSpec(
            chunk_seconds=seconds_per_chunk,
            frame_rate_hz=positions_per_second,
        )

        if loss_weights is None:
            self.loss_weights = NextEventLossWeights()
        elif isinstance(loss_weights, NextEventLossWeights):
            self.loss_weights = loss_weights
        elif isinstance(loss_weights, Mapping):
            unknown = set(loss_weights) - set(_GROUPS)
            if unknown:
                raise ValueError(f"Unknown next-event loss weight groups: {sorted(unknown)}.")
            self.loss_weights = NextEventLossWeights(**loss_weights)
        else:
            raise TypeError("loss_weights must be NextEventLossWeights or a mapping.")

        self._validate_thinker_boundary()
        # The accepted Transformers build exposes Qwen's vocabulary head as a
        # position-wise Linear, so applying it after slicing the sequence is
        # exactly equivalent to slicing full logits and avoids a large prefill
        # allocation.  Refuse the optimization if that API boundary changes.
        self.supports_last_position_only = isinstance(self.thinker.lm_head, nn.Linear)

    @property
    def frames_per_chunk(self) -> int:
        return self.timeline.frames_per_chunk

    def _validate_thinker_boundary(self) -> None:
        required = ("audio_tower", "model", "lm_head", "get_audio_features")
        missing = [name for name in required if not hasattr(self.thinker, name)]
        if missing:
            raise TypeError(f"Thinker is missing required Qwen components: {missing}.")
        if not hasattr(self.thinker, "get_input_embeddings"):
            raise TypeError("Thinker must expose its original input embedding table.")
        if not hasattr(self.thinker.audio_tower, "_get_feat_extract_output_lengths"):
            raise TypeError("Qwen audio tower has no output-length helper.")

        thinker_config = getattr(self.thinker, "config", None)
        text_config = getattr(thinker_config, "text_config", None)
        model_vocab_size = getattr(text_config, "vocab_size", None)
        if (
            model_vocab_size is not None
            and model_vocab_size != self.control_tokens.thinker_vocab_size
        ):
            raise ValueError(
                "Root and loaded Thinker vocabulary sizes differ: "
                f"{self.control_tokens.thinker_vocab_size} != {model_vocab_size}."
            )

    @staticmethod
    def _check_binary_mask(name: str, mask: torch.Tensor) -> None:
        if mask.dtype == torch.bool:
            return
        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError(f"{name} must contain only zero/one mask values.")

    def _validate_timeline_inputs(
        self,
        *,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor,
        control_ids: torch.Tensor,
        control_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        past_key_values: object | None,
    ) -> tuple[int, int, torch.Tensor]:
        if text_ids.ndim != 2:
            raise ValueError("text_ids must have shape [batch, timeline].")
        batch_size, timeline_length = text_ids.shape
        if timeline_length == 0:
            raise ValueError("Timeline inputs must contain at least one position.")
        expected = (batch_size, timeline_length)
        for name, tensor in (
            ("text_mask", text_mask),
            ("control_ids", control_ids),
            ("control_mask", control_mask),
        ):
            if tuple(tensor.shape) != expected:
                raise ValueError(f"{name} must have shape {expected}, got {tuple(tensor.shape)}.")
        if labels is not None and tuple(labels.shape) != expected:
            raise ValueError(
                f"labels must have shape {expected}, got {tuple(labels.shape)}."
            )
        if attention_mask.ndim != 2 or attention_mask.shape[0] != batch_size:
            raise ValueError("attention_mask must have shape [batch, sequence].")
        if past_key_values is None and attention_mask.shape[1] != timeline_length:
            raise ValueError(
                "attention_mask length must equal the timeline length without a cache."
            )
        if attention_mask.shape[1] < timeline_length:
            raise ValueError("attention_mask is shorter than the current timeline input.")
        if position_ids is not None and tuple(position_ids.shape) != expected:
            raise ValueError(
                f"position_ids must have shape {expected}, got {tuple(position_ids.shape)}."
            )

        for name, mask in (
            ("text_mask", text_mask),
            ("control_mask", control_mask),
            ("attention_mask", attention_mask),
        ):
            self._check_binary_mask(name, mask)
        current_attention = attention_mask[:, -timeline_length:].bool()
        if torch.any(text_mask.bool() & ~current_attention):
            raise ValueError("text_mask enables a position disabled by attention_mask.")
        if torch.any(control_mask.bool() & ~current_attention):
            raise ValueError("control_mask enables a position disabled by attention_mask.")
        return batch_size, timeline_length, current_attention

    def _restore_audio(
        self,
        *,
        input_features: torch.Tensor,
        feature_attention_mask: torch.Tensor,
        preconv_feature_lengths: torch.Tensor,
        current_attention: torch.Tensor,
        timeline_length: int,
        hidden_width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        batch_size = current_attention.shape[0]
        if input_features.ndim < 2 or input_features.shape[0] != batch_size:
            raise ValueError("input_features batch dimension does not match the timeline.")
        if feature_attention_mask.ndim != 2 or feature_attention_mask.shape[0] != batch_size:
            raise ValueError(
                "feature_attention_mask must have shape [batch, preconv_time]."
            )
        self._check_binary_mask("feature_attention_mask", feature_attention_mask)
        if preconv_feature_lengths.ndim != 1 or preconv_feature_lengths.shape[0] != batch_size:
            raise ValueError("preconv_feature_lengths must have shape [batch].")

        mask_lengths = feature_attention_mask.to(dtype=torch.long).sum(dim=-1)
        supplied_lengths = preconv_feature_lengths.to(
            device=mask_lengths.device, dtype=torch.long
        )
        if not torch.equal(mask_lengths, supplied_lengths):
            raise ValueError(
                "preconv_feature_lengths do not match feature_attention_mask sums."
            )

        length_result = self.thinker.audio_tower._get_feat_extract_output_lengths(
            mask_lengths
        )
        if not isinstance(length_result, (tuple, list)) or len(length_result) != 2:
            raise ValueError("Qwen audio length helper returned an unexpected value.")
        output_lengths = torch.as_tensor(
            length_result[1], device=mask_lengths.device, dtype=torch.long
        )
        if output_lengths.ndim != 1 or output_lengths.shape[0] != batch_size:
            raise ValueError("Qwen audio output lengths do not match the batch.")
        if torch.any(output_lengths < 0):
            raise ValueError("Qwen audio output lengths must be non-negative.")

        timeline_lengths = current_attention.to(dtype=torch.long).sum(dim=-1)
        if not torch.equal(output_lengths, timeline_lengths.to(output_lengths.device)):
            raise ValueError(
                "Qwen audio output lengths cannot be aligned to the enabled timeline "
                f"positions: audio={output_lengths.tolist()}, "
                f"timeline={timeline_lengths.tolist()}."
            )
        expected_padding = (
            torch.arange(timeline_length, device=current_attention.device)[None, :]
            < timeline_lengths[:, None]
        )
        if not torch.equal(current_attention, expected_padding):
            raise ValueError("Timeline attention must be contiguous with trailing padding.")

        audio_output = self.thinker.get_audio_features(
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            return_dict=True,
        )
        flattened = getattr(audio_output, "last_hidden_state", None)
        if flattened is None or flattened.ndim != 2:
            raise ValueError("Qwen audio tower must return flattened [positions, hidden].")
        if flattened.shape[0] != int(output_lengths.sum().item()):
            raise ValueError(
                "Flattened Qwen audio output length does not equal the model-derived "
                "per-sample lengths."
            )
        if flattened.shape[1] != hidden_width:
            raise ValueError(
                f"Audio hidden width {flattened.shape[1]} does not match Thinker "
                f"hidden width {hidden_width}."
            )

        restored = torch.split(flattened, output_lengths.tolist(), dim=0)
        aligned = pad_sequence(restored, batch_first=True)
        if aligned.shape[1] < timeline_length:
            aligned = F.pad(aligned, (0, 0, 0, timeline_length - aligned.shape[1]))
        if tuple(aligned.shape[:2]) != (batch_size, timeline_length):
            raise ValueError("Restored Qwen audio features have an unexplained timeline length.")
        return aligned.to(device=device, dtype=dtype)

    def _target_groups(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        valid = values != IGNORE_LABEL
        idle = valid & (values == self.control_tokens.idle)
        start = valid & (values == self.control_tokens.start)
        stop = valid & (values == self.control_tokens.stop)
        return {
            "text": valid & ~(idle | start | stop),
            "idle": idle,
            "start": start,
            "stop": stop,
        }

    def _weighted_loss(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        torch.Tensor,
    ]:
        token_losses = F.cross_entropy(
            logits.transpose(1, 2),
            labels,
            reduction="none",
            ignore_index=IGNORE_LABEL,
        )
        target_groups = self._target_groups(labels)
        applied_weights = torch.zeros_like(token_losses)
        group_losses: dict[str, torch.Tensor] = {}
        target_counts: dict[str, torch.Tensor] = {}
        for name, mask in target_groups.items():
            count = mask.sum()
            target_counts[name] = count
            group_total = (token_losses * mask).sum()
            group_losses[name] = group_total / count.clamp_min(1)
            applied_weights = applied_weights + mask * getattr(self.loss_weights, name)

        weight_sum = applied_weights.sum()
        loss = (token_losses * applied_weights).sum() / weight_sum.clamp_min(1)

        predictions = logits.argmax(dim=-1)
        prediction_groups = self._target_groups(predictions)
        valid_targets = labels != IGNORE_LABEL
        prediction_counts = {
            name: (mask & valid_targets).sum()
            for name, mask in prediction_groups.items()
        }
        return loss, group_losses, target_counts, prediction_counts, weight_sum

    def forward(
        self,
        *,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor,
        control_ids: torch.Tensor,
        control_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        input_features: torch.Tensor | None = None,
        feature_attention_mask: torch.Tensor | None = None,
        preconv_feature_lengths: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: object | None = None,
        use_cache: bool | None = None,
        last_position_only: bool = False,
    ) -> DuplexThinkerOutput:
        _, timeline_length, current_attention = self._validate_timeline_inputs(
            text_ids=text_ids,
            text_mask=text_mask,
            control_ids=control_ids,
            control_mask=control_mask,
            attention_mask=attention_mask,
            labels=labels,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
        if last_position_only and labels is not None:
            raise ValueError("last_position_only is inference-only; labels must be omitted.")
        if last_position_only and not self.supports_last_position_only:
            raise RuntimeError(
                "The installed Thinker LM head does not safely support "
                "last-position-only logits."
            )

        embedding = self.thinker.get_input_embeddings()
        text_embeddings = embedding(text_ids)
        control_embeddings = embedding(control_ids)
        text_embeddings = text_embeddings * text_mask.unsqueeze(-1).to(
            device=text_embeddings.device, dtype=text_embeddings.dtype
        )
        control_embeddings = control_embeddings * control_mask.unsqueeze(-1).to(
            device=control_embeddings.device, dtype=control_embeddings.dtype
        )
        if text_embeddings.shape != control_embeddings.shape:
            raise ValueError("Text and control embeddings do not share one hidden space.")

        audio_arguments = (input_features, feature_attention_mask, preconv_feature_lengths)
        if any(value is not None for value in audio_arguments):
            if not all(value is not None for value in audio_arguments):
                raise ValueError(
                    "input_features, feature_attention_mask, and "
                    "preconv_feature_lengths must be supplied together."
                )
            audio_embeddings = self._restore_audio(
                input_features=input_features,
                feature_attention_mask=feature_attention_mask,
                preconv_feature_lengths=preconv_feature_lengths,
                current_attention=current_attention,
                timeline_length=timeline_length,
                hidden_width=text_embeddings.shape[-1],
                device=text_embeddings.device,
                dtype=text_embeddings.dtype,
            )
        else:
            audio_embeddings = torch.zeros_like(text_embeddings)

        fused_embeddings = text_embeddings + audio_embeddings + control_embeddings
        model_outputs = self.thinker.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=fused_embeddings,
            use_cache=use_cache,
            return_dict=True,
        )
        lexical_hidden_states = getattr(model_outputs, "last_hidden_state", None)
        if lexical_hidden_states is None:
            lexical_hidden_states = model_outputs[0]

        logits_input = lexical_hidden_states
        if last_position_only:
            logits_input = logits_input[:, -1:, :]
        logits = self.thinker.lm_head(logits_input)

        loss = None
        loss_weight_sum = None
        group_losses: dict[str, torch.Tensor] = {}
        target_counts: dict[str, torch.Tensor] = {}
        prediction_counts: dict[str, torch.Tensor] = {}
        if labels is not None:
            (
                loss,
                group_losses,
                target_counts,
                prediction_counts,
                loss_weight_sum,
            ) = self._weighted_loss(logits, labels)

        return DuplexThinkerOutput(
            loss=loss,
            logits=logits,
            past_key_values=getattr(model_outputs, "past_key_values", None),
            lexical_hidden_states=lexical_hidden_states,
            hidden_states=getattr(model_outputs, "hidden_states", None),
            attentions=getattr(model_outputs, "attentions", None),
            group_losses=group_losses,
            target_counts=target_counts,
            prediction_counts=prediction_counts,
            loss_weight_sum=loss_weight_sum,
        )
