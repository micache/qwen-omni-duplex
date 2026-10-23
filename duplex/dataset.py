"""Pad event timelines and extract fixed two-second audio chunks for training."""

from __future__ import annotations

import hashlib
import os
import random
from collections.abc import Mapping, Sequence
from typing import Any


class _WorkerRandomState:
    """Lazily own one independent Python RNG per process/DataLoader worker."""

    def __init__(self, seed: int) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"augmentation seed must be an integer, got {seed!r}.")
        self.seed = seed
        self._identity: tuple[int, int] | None = None
        self._rng: random.Random | None = None

    def current(self) -> random.Random:
        import torch

        worker = torch.utils.data.get_worker_info()
        worker_id = -1 if worker is None else int(worker.id)
        worker_seed = self.seed if worker is None else int(worker.seed) ^ self.seed
        identity = (os.getpid(), worker_id)
        if self._rng is None or self._identity != identity:
            self._identity = identity
            self._rng = random.Random(worker_seed)
        return self._rng


def _batch_value(batch: object, name: str) -> object | None:
    if isinstance(batch, Mapping):
        return batch.get(name)
    return getattr(batch, name, None)


class DuplexCollator:
    """Pad event timelines and extract their audio features.

    The supplied audio object may be a feature extractor directly or a Qwen
    processor exposing ``feature_extractor``.  This class never calls an audio
    tower and intentionally retains the extractor's pre-convolution mask.
    """

    def __init__(
        self,
        *,
        audio_processor: object,
        tokenizer: object,
        control_tokens: object,
        thinker_bos_token_id: int,
        frame_rate_hz: int = 25,
        augmentation_seed: int = 0,
        context_token_ids: Sequence[int] = (),
    ) -> None:
        from .timeline import TimelineSpec

        TimelineSpec(frame_rate_hz=frame_rate_hz)
        self.audio_processor = audio_processor
        self.tokenizer = tokenizer
        self.control_tokens = control_tokens
        self.thinker_bos_token_id = thinker_bos_token_id
        self.frame_rate_hz = frame_rate_hz
        self.context_token_ids = tuple(context_token_ids)
        self._random_state = _WorkerRandomState(augmentation_seed)

    def augmentation_rng(self) -> random.Random:
        """Return this process/worker's lazily seeded augmentation RNG."""

        return self._random_state.current()

    def _extract_audio(self, timelines: Sequence[object]) -> dict[str, Any]:
        import torch
        from .streaming import prepare_partial_audio_for_extractor

        extractor = getattr(
            self.audio_processor, "feature_extractor", self.audio_processor
        )
        if not callable(extractor):
            raise TypeError("audio_processor or its feature_extractor must be callable.")
        sample_rates = {timeline.audio_sample_rate_hz for timeline in timelines}
        if len(sample_rates) != 1:
            raise ValueError(
                f"A batch must use one audio sample rate, got {sorted(sample_rates)}."
            )
        waveforms = []
        valid_preconv_lengths = []
        chunk_counts = []
        for timeline in timelines:
            chunk_samples = round(2.0 * timeline.audio_sample_rate_hz)
            count = (len(timeline.user_waveform) + chunk_samples - 1) // chunk_samples
            if count < 1:
                raise ValueError(f"{timeline.sample_id}: empty audio.")
            chunk_counts.append(count)
            for index in range(count):
                block = timeline.user_waveform[index * chunk_samples:(index + 1) * chunk_samples]
                prepared, valid_preconv = prepare_partial_audio_for_extractor(block, extractor)
                waveforms.append(prepared)
                valid_preconv_lengths.append(valid_preconv)
        processed = extractor(
            waveforms,
            sampling_rate=sample_rates.pop(),
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_features = _batch_value(processed, "input_features")
        feature_mask = _batch_value(processed, "feature_attention_mask")
        if feature_mask is None:
            feature_mask = _batch_value(processed, "attention_mask")
        if input_features is None:
            raise ValueError("Audio feature extractor returned no input_features.")
        if feature_mask is None:
            raise ValueError(
                "Audio feature extractor must return feature_attention_mask or "
                "attention_mask; its time axis cannot be inferred safely."
            )
        input_features = torch.as_tensor(input_features)
        feature_mask = torch.as_tensor(feature_mask)
        for index, valid_preconv in enumerate(valid_preconv_lengths):
            if len(waveforms[index]) < chunk_samples:
                feature_mask[index].zero_()
                feature_mask[index, :min(valid_preconv, feature_mask.shape[1])] = 1
        if input_features.shape[0] != len(waveforms):
            raise ValueError("input_features batch dimension does not match audio chunks.")
        if feature_mask.ndim != 2 or feature_mask.shape[0] != len(waveforms):
            raise ValueError(
                "feature_attention_mask must have shape [batch, preconv_time]."
            )
        # These lengths describe only the extractor mask.  They are not
        # flattened/restored audio-tower lengths; model.py owns that later API
        # probe and conversion.
        preconv_lengths = feature_mask.to(dtype=torch.long).sum(dim=-1)
        return {
            "input_features": input_features,
            "feature_attention_mask": feature_mask,
            "preconv_feature_lengths": preconv_lengths,
            "audio_chunk_counts": torch.tensor(chunk_counts, dtype=torch.long),
        }

    def __call__(self, items: Sequence[object]) -> dict[str, Any]:
        import torch

        from .timeline import WindowTimeline, encode_causal_timeline

        if not items:
            raise ValueError("Cannot collate an empty batch.")
        if not all(isinstance(item, WindowTimeline) for item in items):
            raise TypeError("DuplexCollator requires event timelines.")
        timelines = items

        timeline_length = max(len(timeline.targets.events) for timeline in timelines)
        causal = [
            encode_causal_timeline(
                timeline.targets,
                control_tokens=self.control_tokens,
                thinker_bos_token_id=self.thinker_bos_token_id,
                pad_to_length=timeline_length,
            )
            for timeline in timelines
        ]
        result: dict[str, Any] = self._extract_audio(timelines)
        result.update(
            {
                "audio_cache_keys": tuple(
                    hashlib.sha256(b"qwen-2s-mask-v2\0" + memoryview(timeline.user_waveform).tobytes()).hexdigest()
                    for timeline in timelines
                ),
                "text_ids": torch.tensor(
                    [row.text_ids for row in causal], dtype=torch.long
                ),
                "text_mask": torch.tensor(
                    [row.text_mask for row in causal], dtype=torch.bool
                ),
                "control_ids": torch.tensor(
                    [row.control_ids for row in causal], dtype=torch.long
                ),
                "control_mask": torch.tensor(
                    [row.control_mask for row in causal], dtype=torch.bool
                ),
                "labels": torch.tensor(
                    [row.labels for row in causal], dtype=torch.long
                ),
                "attention_mask": torch.tensor(
                    [row.attention_mask for row in causal], dtype=torch.bool
                ),
                "position_ids": torch.arange(timeline_length, dtype=torch.long)[None, :].expand(len(causal), -1).clone(),
                "event_types": tuple(tuple(row.event_types) for row in causal),
                "bootstrap_mask": torch.tensor([row.bootstrap_mask for row in causal], dtype=torch.bool),
                "context_ids": torch.tensor([self.context_token_ids] * len(causal), dtype=torch.long),
                "frame_times": tuple(
                    None if row.frame_times is None else tuple(row.frame_times)
                    for row in causal
                ),
                "sample_metadata": tuple(
                    {
                        "sample_id": timeline.sample_id,
                        "conversation_id": timeline.conversation_id,
                        "valid_sequence_length": timeline.valid_sequence_length,
                        "window_start_seconds": timeline.window_start_seconds,
                        "window_end_seconds": timeline.window_end_seconds,
                        "interruption": timeline.interruption,
                    }
                    for timeline in timelines
                ),
            }
        )
        return result
