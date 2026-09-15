"""Explicit fixed-chunk streaming for the Qwen2.5-Omni Thinker."""

from __future__ import annotations

import math
import re
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

from .model import QwenDuplexThinker
from .timeline import (
    DEFAULT_SYSTEM_PROMPT,
    EventKind,
    ResponseState,
    frame_event_inputs,
    prompt_token_ids,
)


AUDIO_SAMPLE_RATE_HZ = 16_000
CHUNK_SECONDS = 2.0
FRAME_RATE_HZ = 25
SAMPLES_PER_CHUNK = int(AUDIO_SAMPLE_RATE_HZ * CHUNK_SECONDS)


@dataclass(frozen=True)
class AudioChunk:
    """One padded two-second input block and its real-sample mask."""

    index: int
    waveform: np.ndarray
    valid_mask: np.ndarray
    start_time_s: float
    end_time_s: float
    available_time_s: float
    silent_tail: bool = False

    @property
    def valid_samples(self) -> int:
        return int(self.valid_mask.sum())


@dataclass(frozen=True)
class EventTraceRecord:
    sample_id: str
    chunk_index: int
    frame_index: int
    audio_time_s: float
    available_time_s: float
    compute_ms: float
    event_id: int
    event_type: str
    decoded_delta: str
    state_before: str
    state_after: str
    grammar_mask_changed_raw_argmax: bool
    raw_argmax_id: int
    chunk_end_time_s: float
    silent_tail: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TokenTiming:
    trace_index: int
    token_id: int
    audio_time_s: float
    available_time_s: float


@dataclass(frozen=True)
class LexicalWordSpan:
    text: str
    character_start: int
    character_end: int
    token_timings: tuple[TokenTiming, ...]


@dataclass
class StreamingResult:
    sample_id: str
    text: str
    trace: list[EventTraceRecord]
    word_spans: list[LexicalWordSpan]
    lexical_hidden_states: list[torch.Tensor]
    timed_out: bool
    stop_reason: str
    input_duration_s: float
    processed_silent_chunks: int


@dataclass
class DuplexGenerationResult:
    text: str
    segments: list[dict[str, Any]]
    events: list[EventTraceRecord]
    first_word_time_s: float | None
    last_word_time_s: float | None

    @classmethod
    def from_streaming(cls, result: StreamingResult) -> "DuplexGenerationResult":
        segments: list[dict[str, Any]] = []
        active: dict[str, Any] | None = None
        completed_times: list[float] = []
        for row in result.trace:
            if row.event_type == "START":
                active = {"start_time_s": row.available_time_s, "text": "", "tokens": [], "lexical_times": []}
            elif row.event_type == "TEXT" and active is not None:
                active["tokens"].append(row.event_id)
                active["text"] += row.decoded_delta
                active["lexical_times"].append(row.available_time_s)
            elif row.event_type == "STOP" and active is not None:
                active["stop_time_s"] = row.available_time_s
                completed_times.extend(active.pop("lexical_times"))
                segments.append(active)
                active = None
        return cls("".join(segment["text"] for segment in segments), segments, result.trace,
                   min(completed_times) if completed_times else None,
                   max(completed_times) if completed_times else None)


def split_fixed_audio_chunks(
    waveform: np.ndarray | torch.Tensor | Sequence[float],
    *,
    sample_rate_hz: int = AUDIO_SAMPLE_RATE_HZ,
) -> list[AudioChunk]:
    """Split mono 16 kHz audio, padding only its final two-second chunk."""

    if sample_rate_hz != AUDIO_SAMPLE_RATE_HZ:
        raise ValueError(
            "Streaming input must be mono 16 kHz audio, "
            f"got {sample_rate_hz} Hz."
        )
    values = (
        waveform.detach().cpu().numpy()
        if isinstance(waveform, torch.Tensor)
        else np.asarray(waveform)
    )
    if values.ndim != 1:
        raise ValueError(f"Streaming input must be mono, got shape {values.shape}.")
    if values.size == 0:
        raise ValueError("Streaming input must contain at least one audio sample.")
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError("Streaming waveform must contain numeric samples.")
    values = values.astype(np.float32, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("Streaming waveform contains non-finite samples.")

    chunks: list[AudioChunk] = []
    for index, offset in enumerate(range(0, values.size, SAMPLES_PER_CHUNK)):
        valid = min(SAMPLES_PER_CHUNK, values.size - offset)
        padded = np.zeros(SAMPLES_PER_CHUNK, dtype=np.float32)
        padded[:valid] = values[offset : offset + valid]
        valid_mask = np.zeros(SAMPLES_PER_CHUNK, dtype=np.bool_)
        valid_mask[:valid] = True
        start_time = offset / sample_rate_hz
        end_time = (offset + valid) / sample_rate_hz
        chunks.append(
            AudioChunk(
                index=index,
                waveform=padded,
                valid_mask=valid_mask,
                start_time_s=start_time,
                end_time_s=end_time,
                available_time_s=end_time,
            )
        )
    return chunks


def prepare_partial_audio_for_extractor(values: np.ndarray, extractor) -> tuple[np.ndarray, int]:
    """Pad only enough for Whisper's reflect STFT; retain real preconv length."""
    hop = int(getattr(extractor, "hop_length", 160))
    minimum = max(int(getattr(extractor, "n_fft", 400)) + 1, 3 * hop)
    # Whisper drops the final STFT frame: valid preconv frames are L // hop.
    valid_length = max(3, len(values) // hop)
    if len(values) < minimum:
        values = np.pad(values, (0, minimum - len(values)))
    return values, valid_length


def _silent_chunk(index: int, start_time_s: float) -> AudioChunk:
    end_time = start_time_s + CHUNK_SECONDS
    return AudioChunk(
        index=index,
        waveform=np.zeros(SAMPLES_PER_CHUNK, dtype=np.float32),
        valid_mask=np.ones(SAMPLES_PER_CHUNK, dtype=np.bool_),
        start_time_s=start_time_s,
        end_time_s=end_time,
        available_time_s=end_time,
        silent_tail=True,
    )


def event_kind(token_id: int, model: QwenDuplexThinker) -> EventKind:
    controls = model.control_tokens
    if token_id == controls.idle:
        return EventKind.IDLE
    if token_id == controls.start:
        return EventKind.START
    if token_id == controls.stop:
        return EventKind.STOP
    return EventKind.TEXT


def grammar_mask_logits(
    logits: torch.Tensor,
    *,
    state: ResponseState,
    model: QwenDuplexThinker,
) -> tuple[torch.Tensor, int, bool]:
    """Apply the event grammar and retain the raw argmax diagnosis."""

    if logits.ndim != 1:
        raise ValueError("Event logits must be a one-dimensional vocabulary vector.")
    raw_argmax = int(logits.argmax().item())
    controls = model.control_tokens
    masked = torch.full_like(logits, -torch.inf)
    if state is ResponseState.INACTIVE:
        masked[controls.idle] = logits[controls.idle]
        masked[controls.start] = logits[controls.start]
    else:
        masked.copy_(logits)
        masked[controls.start] = -torch.inf
    selected_argmax = int(masked.argmax().item())
    return masked, raw_argmax, selected_argmax != raw_argmax


def _decode(tokenizer: object, token_ids: Sequence[int]) -> str:
    return tokenizer.decode(
        list(token_ids),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def _assign_exact_decoded_deltas(
    records: list[EventTraceRecord], tokenizer: object
) -> tuple[list[EventTraceRecord], str]:
    lexical_positions = [
        index for index, record in enumerate(records) if record.event_type == EventKind.TEXT.value
    ]
    token_ids = [records[index].event_id for index in lexical_positions]
    final_text = _decode(tokenizer, token_ids)
    emitted = 0
    updated = list(records)
    for count, record_index in enumerate(lexical_positions, start=1):
        prefix = _decode(tokenizer, token_ids[:count])
        stable = 0
        limit = min(len(prefix), len(final_text))
        while stable < limit and prefix[stable] == final_text[stable]:
            stable += 1
        stable = max(stable, emitted)
        delta = final_text[emitted:stable]
        emitted = stable
        updated[record_index] = replace(updated[record_index], decoded_delta=delta)
    if emitted != len(final_text):
        if not lexical_positions:
            raise RuntimeError("Tokenizer produced text without lexical events.")
        last = lexical_positions[-1]
        updated[last] = replace(
            updated[last], decoded_delta=updated[last].decoded_delta + final_text[emitted:]
        )
    visible = "".join(record.decoded_delta for record in updated)
    if visible != final_text:
        raise RuntimeError("Incremental decoding did not reconstruct the exact final text.")
    return updated, final_text


def group_lexical_events_into_words(
    records: Sequence[EventTraceRecord], tokenizer: object
) -> list[LexicalWordSpan]:
    """Group decoded lexical events while preserving every contributing token time."""

    lexical = [
        (index, record)
        for index, record in enumerate(records)
        if record.event_type == EventKind.TEXT.value
    ]
    if not lexical:
        return []
    token_ids = [record.event_id for _, record in lexical]
    final_text = _decode(tokenizer, token_ids)

    boundaries: list[int | None] = [0]
    for count in range(1, len(token_ids)):
        prefix = _decode(tokenizer, token_ids[:count])
        boundaries.append(len(prefix) if final_text.startswith(prefix) else None)
    boundaries.append(len(final_text))

    token_ranges: list[tuple[int, int]] = []
    for token_index in range(len(token_ids)):
        left = max(
            boundary
            for boundary in boundaries[: token_index + 1]
            if boundary is not None
        )
        right = next(
            boundary
            for boundary in boundaries[token_index + 1 :]
            if boundary is not None
        )
        token_ranges.append((left, right))

    words: list[LexicalWordSpan] = []
    for match in re.finditer(r"\S+", final_text, flags=re.UNICODE):
        timings = tuple(
            TokenTiming(
                trace_index=record_index,
                token_id=record.event_id,
                audio_time_s=record.audio_time_s,
                available_time_s=record.available_time_s,
            )
            for (record_index, record), (left, right) in zip(lexical, token_ranges)
            if right > match.start() and left < match.end()
        )
        words.append(
            LexicalWordSpan(
                text=match.group(0),
                character_start=match.start(),
                character_end=match.end(),
                token_timings=timings,
            )
        )
    return words


class QwenDuplexStreamer:
    """Run the trained event model explicitly without Hugging Face ``generate``."""

    def __init__(
        self,
        model: QwenDuplexThinker,
        processor: object,
        tokenizer: object | None = None,
        *,
        max_silent_chunks: int = 4,
        sample: bool = False,
        temperature: float = 1.0,
        top_k: int | None = None,
        feature_count_tolerance: int = 1,
        max_new_tokens: int = 256,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if model.timeline.chunk_seconds != CHUNK_SECONDS:
            raise ValueError("Streaming requires the validated fixed two-second timeline.")
        if model.timeline.frame_rate_hz != FRAME_RATE_HZ:
            raise ValueError("Streaming requires the validated 25 Hz timeline.")
        if max_silent_chunks < 0:
            raise ValueError("max_silent_chunks must be non-negative.")
        if temperature <= 0 or not math.isfinite(temperature):
            raise ValueError("temperature must be finite and positive.")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive when supplied.")
        if feature_count_tolerance < 0:
            raise ValueError("feature_count_tolerance must be non-negative.")
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer or getattr(processor, "tokenizer", None)
        if self.tokenizer is None:
            raise TypeError("A tokenizer is required for exact incremental decoding.")
        self.max_silent_chunks = max_silent_chunks
        self.sample = sample
        self.temperature = temperature
        self.top_k = top_k
        self.feature_count_tolerance = feature_count_tolerance
        self.max_new_tokens = max_new_tokens
        self.clock = clock

    @property
    def device(self) -> torch.device:
        return self.model.base_thinker.get_input_embeddings().weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.model.base_thinker.get_input_embeddings().weight.dtype

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _prefill(self, context_token_ids: Sequence[int]) -> tuple[object, int]:
        token_ids = list(context_token_ids)
        if not token_ids:
            return None, 0
        tokens = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        embedding = self.model.base_thinker.get_input_embeddings()(tokens)
        length = tokens.shape[1]
        output = self.model.base_thinker.model(
            inputs_embeds=embedding,
            attention_mask=torch.ones((1, length), dtype=torch.bool, device=self.device),
            position_ids=torch.arange(length, device=self.device).unsqueeze(0),
            past_key_values=None,
            use_cache=True,
            return_dict=True,
        )
        cache = output.past_key_values
        if cache is None:
            raise RuntimeError("Thinker prefill returned no KV cache.")
        self._validate_cache_length(cache, length)
        return cache, length

    @staticmethod
    def _validate_cache_length(cache: object, expected: int) -> None:
        get_length = getattr(cache, "get_seq_length", None)
        if callable(get_length):
            actual = int(get_length())
            if actual != expected:
                raise RuntimeError(
                    f"Thinker KV cache length is {actual}, expected {expected}."
                )

    def _audio_features(self, chunk: AudioChunk) -> torch.Tensor:
        extractor = getattr(self.processor, "feature_extractor", self.processor)
        if not callable(extractor):
            raise TypeError("Processor or processor.feature_extractor must be callable.")
        valid_waveform = chunk.waveform[chunk.valid_mask]
        extractor_waveform, valid_preconv = prepare_partial_audio_for_extractor(valid_waveform, extractor)
        processed = extractor(
            [extractor_waveform],
            sampling_rate=AUDIO_SAMPLE_RATE_HZ,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        get_value = lambda name: (
            processed.get(name) if isinstance(processed, dict) else getattr(processed, name, None)
        )
        input_features = get_value("input_features")
        feature_mask = get_value("feature_attention_mask")
        if feature_mask is None:
            feature_mask = get_value("attention_mask")
        if input_features is None or feature_mask is None:
            raise ValueError("Qwen feature extraction returned no features or valid mask.")
        input_features = torch.as_tensor(input_features, device=self.device)
        feature_mask = torch.as_tensor(feature_mask, device=self.device)
        if chunk.valid_samples < SAMPLES_PER_CHUNK:
            feature_mask.zero_()
            feature_mask[:, :min(valid_preconv, feature_mask.shape[1])] = 1
        preconv_lengths = feature_mask.to(dtype=torch.long).sum(dim=-1)
        length_result = self.model.base_thinker.audio_tower._get_feat_extract_output_lengths(
            preconv_lengths
        )
        if not isinstance(length_result, (tuple, list)) or len(length_result) != 2:
            raise ValueError("Qwen audio length helper returned an unexpected value.")
        valid_features = int(torch.as_tensor(length_result[1]).item())
        expected = max(
            1,
            round(chunk.valid_samples * FRAME_RATE_HZ / AUDIO_SAMPLE_RATE_HZ),
        )
        if abs(valid_features - expected) > self.feature_count_tolerance:
            raise ValueError(
                "Qwen audio feature count does not match the valid chunk duration: "
                f"got {valid_features}, expected approximately {expected}."
            )
        if (
            chunk.valid_samples == SAMPLES_PER_CHUNK
            and abs(valid_features - 50) > self.feature_count_tolerance
        ):
            raise ValueError(
                f"A full two-second chunk produced {valid_features} valid features, expected approximately 50."
            )
        current_attention = torch.ones(
            (1, valid_features), dtype=torch.bool, device=self.device
        )
        return self.model._restore_audio(
            input_features=input_features,
            feature_attention_mask=feature_mask,
            preconv_feature_lengths=preconv_lengths,
            current_attention=current_attention,
            timeline_length=valid_features,
            hidden_width=self.model.base_thinker.get_input_embeddings().weight.shape[1],
            device=self.device,
            dtype=self.dtype,
        )

    def _select_event(self, logits: torch.Tensor, state: ResponseState) -> tuple[int, int, bool]:
        masked, raw_argmax, changed = grammar_mask_logits(
            logits, state=state, model=self.model
        )
        if not self.sample:
            return int(masked.argmax().item()), raw_argmax, changed
        scores = masked / self.temperature
        if self.top_k is not None and self.top_k < scores.numel():
            threshold = torch.topk(scores, self.top_k).values[-1]
            scores = scores.masked_fill(scores < threshold, -torch.inf)
        probabilities = torch.softmax(scores, dim=-1)
        return int(torch.multinomial(probabilities, 1).item()), raw_argmax, changed

    def _step(
        self,
        *,
        audio_feature: torch.Tensor,
        previous_event_id: int | None,
        cache: object,
        sequence_length: int,
        state: ResponseState,
    ) -> tuple[int, int, bool, torch.Tensor, object, float]:
        embedding = self.model.base_thinker.get_input_embeddings()
        text_id, text_mask, control_id, control_mask = frame_event_inputs(
            previous_event_id, bos_token_id=self.model.base_thinker.config.bos_token_id,
            control_ids=(self.model.control_tokens.idle, self.model.control_tokens.start,
                         self.model.control_tokens.stop))
        text_embedding = embedding(torch.tensor([[text_id]], dtype=torch.long, device=self.device)) * text_mask
        control_embedding = embedding(torch.tensor([[control_id]], dtype=torch.long, device=self.device)) * control_mask
        fused = audio_feature + text_embedding + control_embedding
        self._sync()
        started = self.clock()
        output = self.model.base_thinker.model(
            inputs_embeds=fused,
            attention_mask=torch.ones(
                (1, sequence_length + 1), dtype=torch.bool, device=self.device
            ),
            position_ids=torch.tensor([[sequence_length]], device=self.device),
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        hidden = output.last_hidden_state[:, -1:, :]
        logits = self.model.base_thinker.lm_head(hidden)[0, 0]
        self._sync()
        compute_ms = (self.clock() - started) * 1000.0
        next_cache = output.past_key_values
        if next_cache is None:
            raise RuntimeError("Thinker streaming step returned no KV cache.")
        self._validate_cache_length(next_cache, sequence_length + 1)
        selected, raw_argmax, changed = self._select_event(logits, state)
        return selected, raw_argmax, changed, hidden.detach().cpu(), next_cache, compute_ms

    def run(
        self,
        waveform: np.ndarray | torch.Tensor | Sequence[float],
        *,
        sample_id: str,
        context_token_ids: Sequence[int],
        sample_rate_hz: int = AUDIO_SAMPLE_RATE_HZ,
    ) -> StreamingResult:
        chunks = split_fixed_audio_chunks(waveform, sample_rate_hz=sample_rate_hz)
        input_duration = sum(chunk.valid_samples for chunk in chunks) / sample_rate_hz
        state = ResponseState.INACTIVE
        previous_event_id: int | None = None
        records: list[EventTraceRecord] = []
        lexical_hidden_states: list[torch.Tensor] = []
        seen_stop = False
        silent_count = 0

        self.model.eval()
        gradient_disable = getattr(self.model, "gradient_checkpointing_disable", None)
        if callable(gradient_disable):
            gradient_disable()
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
            cache, sequence_length = self._prefill(context_token_ids)
            pending_chunks = list(chunks)
            while pending_chunks:
                chunk = pending_chunks.pop(0)
                audio_features = self._audio_features(chunk)
                feature_count = audio_features.shape[1]
                for frame_index in range(feature_count):
                    before = state
                    (
                        selected,
                        raw_argmax,
                        changed,
                        hidden,
                        cache,
                        compute_ms,
                    ) = self._step(
                        audio_feature=audio_features[:, frame_index : frame_index + 1],
                        previous_event_id=previous_event_id,
                        cache=cache,
                        sequence_length=sequence_length,
                        state=state,
                    )
                    sequence_length += 1
                    kind = event_kind(selected, self.model)
                    if kind is EventKind.START:
                        state = ResponseState.ACTIVE
                    elif kind is EventKind.STOP:
                        state = ResponseState.INACTIVE
                        seen_stop = True
                    if kind is EventKind.TEXT:
                        if len(lexical_hidden_states) >= self.max_new_tokens:
                            pending_chunks.clear()
                            break
                        lexical_hidden_states.append(hidden)
                    audio_time = chunk.start_time_s + min(
                        chunk.end_time_s - chunk.start_time_s,
                        frame_index / FRAME_RATE_HZ,
                    )
                    if chunk.available_time_s + 1e-9 < chunk.end_time_s:
                        raise RuntimeError("A chunk became available before its end time.")
                    records.append(
                        EventTraceRecord(
                            sample_id=sample_id,
                            chunk_index=chunk.index,
                            frame_index=frame_index,
                            audio_time_s=audio_time,
                            available_time_s=chunk.available_time_s,
                            compute_ms=compute_ms,
                            event_id=selected,
                            event_type=kind.value,
                            decoded_delta="",
                            state_before=before.value,
                            state_after=state.value,
                            grammar_mask_changed_raw_argmax=changed,
                            raw_argmax_id=raw_argmax,
                            chunk_end_time_s=chunk.end_time_s,
                            silent_tail=chunk.silent_tail,
                        )
                    )
                    previous_event_id = selected
                    if chunk.silent_tail and kind is EventKind.STOP:
                        pending_chunks.clear()
                        break

                if pending_chunks:
                    continue
                if not chunk.silent_tail and chunk.index + 1 < len(chunks):
                    continue
                if chunk.silent_tail and seen_stop and state is ResponseState.INACTIVE:
                    break
                if not chunk.silent_tail and seen_stop and state is ResponseState.INACTIVE:
                    break
                if silent_count >= self.max_silent_chunks:
                    break
                tail_start = input_duration + silent_count * CHUNK_SECONDS
                pending_chunks.append(_silent_chunk(len(chunks) + silent_count, tail_start))
                silent_count += 1

        records, final_text = _assign_exact_decoded_deltas(records, self.tokenizer)
        word_spans = group_lexical_events_into_words(records, self.tokenizer)
        stopped = seen_stop and state is ResponseState.INACTIVE
        return StreamingResult(
            sample_id=sample_id,
            text=final_text,
            trace=records,
            word_spans=word_spans,
            lexical_hidden_states=lexical_hidden_states,
            timed_out=not stopped,
            stop_reason="STOP" if stopped else "max_silent_chunks",
            input_duration_s=input_duration,
            processed_silent_chunks=silent_count,
        )


def write_trace_jsonl(path: str | Path, records: Sequence[EventTraceRecord]) -> None:
    import json

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record.as_dict(), ensure_ascii=False) + "\n")
