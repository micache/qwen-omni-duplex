"""Small, validated reader for the DailyTalkContiguous dataset view.

The public Kyutai sidecars contain ``[word, [start, end], speaker]`` entries.
Their annotation script writes ``SPEAKER_MAIN`` for channel 0 only. Channel 0
is the assistant/model (left) channel and channel 1 is the user (right)
channel. Additional speaker labels are accepted only through an explicit
mapping, because the public data does not establish names for them.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


DATASET_VIEW = "DailyTalkContiguous"
DATASET_ID = "kyutai/DailyTalkContiguous"
INTERRUPTION_AUGMENTATION = "synthetic"
TARGET_SAMPLE_RATE_HZ = 16_000
WINDOW_SECONDS = 2.0
MAIN_SPEAKER_LABEL = "SPEAKER_MAIN"


class Speaker(str, Enum):
    ASSISTANT = "assistant"
    USER = "user"


EXPECTED_CHANNEL_ROLES = (Speaker.ASSISTANT, Speaker.USER)


class Split(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class WindowKind(str, Enum):
    RANDOM = "random"
    USER_TO_ASSISTANT = "user_to_assistant"
    ASSISTANT_TO_USER = "assistant_to_user"


@dataclass(frozen=True)
class ManifestEntry:
    conversation_id: str
    relative_audio_path: Path
    duration_seconds: float


@dataclass(frozen=True)
class WordSpan:
    text: str
    start_seconds: float
    end_seconds: float
    speaker: Speaker
    source_index: int | None = None


@dataclass(frozen=True)
class ConversationRecord:
    """One conversation with only the user waveform intended as model input."""

    conversation_id: str
    duration_seconds: float
    sample_rate_hz: int
    source_sample_rate_hz: int
    user_waveform: np.ndarray
    assistant_reference_path: Path
    assistant_waveform: np.ndarray | None
    user_words: tuple[WordSpan, ...]
    assistant_words: tuple[WordSpan, ...]
    channel_roles: tuple[Speaker, Speaker] = EXPECTED_CHANNEL_ROLES


@dataclass(frozen=True)
class TurnBoundary:
    time_seconds: float
    from_speaker: Speaker
    to_speaker: Speaker


@dataclass(frozen=True)
class WindowMetadata:
    conversation_id: str
    split: Split
    start_seconds: float
    end_seconds: float
    kind: WindowKind
    boundary_seconds: float | None = None

    def __post_init__(self) -> None:
        if not math.isclose(
            self.end_seconds - self.start_seconds,
            WINDOW_SECONDS,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("Window metadata must describe exactly 2 seconds.")


@dataclass(frozen=True)
class SpanMetadata:
    """One outer eight-second supervision boundary, with four audio chunks."""

    conversation_id: str
    split: Split
    start_seconds: float
    end_seconds: float

    def __post_init__(self) -> None:
        if not math.isclose(self.end_seconds - self.start_seconds, 8.0, abs_tol=1e-9):
            raise ValueError("Continuous span metadata must describe exactly 8 seconds.")


@dataclass(frozen=True)
class ConversationMetadata:
    conversation_id: str
    split: Split
    start_seconds: float
    end_seconds: float

    def __post_init__(self) -> None:
        if self.start_seconds != 0 or self.end_seconds <= 0:
            raise ValueError("Conversation metadata must cover the complete recording.")


def _positive_finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number, got {value!r}.")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{field} must be positive and finite, got {value!r}.")
    return converted


def _safe_relative_audio_path(raw_path: object, *, source: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{source}: path must be a non-empty string.")
    path = Path(raw_path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{source}: unsafe non-relative audio path {raw_path!r}.")
    if path.suffix.lower() != ".wav":
        raise ValueError(f"{source}: expected a .wav path, got {raw_path!r}.")
    return path


def read_manifest(manifest_path: str | Path) -> tuple[ManifestEntry, ...]:
    """Read the public JSONL manifest without opening any audio files."""

    manifest_path = Path(manifest_path)
    entries: list[ManifestEntry] = []
    seen_ids: set[str] = set()
    with manifest_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            source = f"{manifest_path}:{line_number}"
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{source}: invalid JSON: {error.msg}.") from error
            if not isinstance(row, Mapping):
                raise ValueError(f"{source}: manifest row must be a JSON object.")
            path = _safe_relative_audio_path(row.get("path"), source=source)
            duration = _positive_finite_number(
                row.get("duration"), field=f"{source}: duration"
            )
            conversation_id = path.with_suffix("").as_posix()
            if conversation_id in seen_ids:
                raise ValueError(f"{source}: duplicate conversation ID {conversation_id!r}.")
            seen_ids.add(conversation_id)
            entries.append(ManifestEntry(conversation_id, path, duration))
    if not entries:
        raise ValueError(f"{manifest_path}: manifest contains no conversations.")
    return tuple(entries)


def resolve_dataset_path(dataset_root: str | Path, relative_path: Path) -> Path:
    """Resolve a manifest path while preventing traversal and symlink escape."""

    root = Path(dataset_root).resolve()
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"Audio path {relative_path!s} resolves outside dataset root {root!s}."
        ) from error
    return candidate


def parse_sidecar_word_spans(
    payload: object,
    *,
    duration_seconds: float,
    source: str = "<sidecar>",
    speaker_label_map: Mapping[str, Speaker | str] | None = None,
) -> dict[Speaker, tuple[WordSpan, ...]]:
    """Parse the isolated Kyutai sidecar schema into speaker-normalized spans.

    By default only the verified ``SPEAKER_MAIN`` label is accepted and mapped
    to the assistant/left channel. A caller may provide a verified mapping for
    additional labels; unknown labels are rejected.
    """

    if not isinstance(payload, Mapping) or set(payload) != {"alignments"}:
        raise ValueError(f"{source}: expected one 'alignments' field.")
    alignments = payload["alignments"]
    if not isinstance(alignments, list):
        raise ValueError(f"{source}: alignments must be a list.")

    raw_map = (
        {MAIN_SPEAKER_LABEL: Speaker.ASSISTANT}
        if speaker_label_map is None
        else speaker_label_map
    )
    if not isinstance(raw_map, Mapping):
        raise ValueError("speaker_label_map must be a mapping.")
    normalized_map: dict[str, Speaker] = {}
    for label, speaker in raw_map.items():
        if not isinstance(label, str) or not label:
            raise ValueError("Speaker labels must be non-empty strings.")
        try:
            normalized_map[label] = Speaker(speaker)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid normalized speaker for {label!r}: {speaker!r}."
            ) from error
    if normalized_map.get(MAIN_SPEAKER_LABEL) is not Speaker.ASSISTANT:
        raise ValueError("SPEAKER_MAIN must map to the assistant/left channel.")

    duration_seconds = _positive_finite_number(
        duration_seconds, field=f"{source}: duration"
    )
    by_speaker: dict[Speaker, list[WordSpan]] = {
        Speaker.USER: [],
        Speaker.ASSISTANT: [],
    }
    previous_start = -math.inf
    for index, alignment in enumerate(alignments):
        item_source = f"{source}: alignment {index}"
        if not isinstance(alignment, list) or len(alignment) != 3:
            raise ValueError(
                f"{item_source}: expected [word, [start, end], speaker]."
            )
        text, timestamps, label = alignment
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{item_source}: word must be a non-empty string.")
        if not isinstance(timestamps, list) or len(timestamps) != 2:
            raise ValueError(f"{item_source}: timestamps must be [start, end].")
        start, end = timestamps
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in timestamps
        ):
            raise ValueError(f"{item_source}: timestamps must be numbers.")
        start = float(start)
        end = float(end)
        if not math.isfinite(start) or not math.isfinite(end):
            raise ValueError(f"{item_source}: timestamps must be finite.")
        if start < 0.0 or end <= start:
            raise ValueError(f"{item_source}: require 0 <= start < end.")
        if start < previous_start:
            raise ValueError(
                f"{item_source}: timestamps are not ordered by start time."
            )
        if end > duration_seconds + 1e-3:
            raise ValueError(
                f"{item_source}: end {end:.6f} exceeds duration "
                f"{duration_seconds:.6f}."
            )
        if not isinstance(label, str) or label not in normalized_map:
            raise ValueError(
                f"{item_source}: unverified speaker label {label!r}; provide an "
                "explicit speaker_label_map after confirming its channel meaning."
            )
        speaker = normalized_map[label]
        by_speaker[speaker].append(
            WordSpan(text, start, end, speaker, source_index=index)
        )
        previous_start = start

    return {speaker: tuple(spans) for speaker, spans in by_speaker.items()}


def resample_waveform_to_16khz(
    waveform: np.ndarray, source_sample_rate_hz: int
) -> np.ndarray:
    """Resample one mono waveform to 16 kHz; no other helper resamples audio."""

    import numpy as np
    from scipy.signal import resample_poly

    if (
        isinstance(source_sample_rate_hz, bool)
        or not isinstance(source_sample_rate_hz, int)
        or source_sample_rate_hz <= 0
    ):
        raise ValueError("Source sample rate must be a positive integer.")
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim != 1:
        raise ValueError(f"Expected a mono waveform, got shape {waveform.shape}.")
    if source_sample_rate_hz == TARGET_SAMPLE_RATE_HZ:
        return waveform.copy()
    divisor = math.gcd(source_sample_rate_hz, TARGET_SAMPLE_RATE_HZ)
    result = resample_poly(
        waveform,
        TARGET_SAMPLE_RATE_HZ // divisor,
        source_sample_rate_hz // divisor,
    )
    return np.asarray(result, dtype=np.float32)


def _validate_channel_roles(channel_roles: Sequence[Speaker | str]) -> None:
    try:
        normalized = tuple(Speaker(role) for role in channel_roles)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid channel roles {channel_roles!r}.") from error
    if normalized != EXPECTED_CHANNEL_ROLES:
        raise ValueError(
            "DailyTalkContiguous channels must be left=assistant/model and right=user; "
            f"got {[role.value for role in normalized]!r}."
        )


def load_conversation(
    entry: ManifestEntry,
    *,
    dataset_root: str | Path,
    expected_source_sample_rate_hz: int | None = None,
    retain_assistant_waveform: bool = False,
    channel_roles: Sequence[Speaker | str] = EXPECTED_CHANNEL_ROLES,
    speaker_label_map: Mapping[str, Speaker | str] | None = None,
) -> ConversationRecord:
    """Validate and load one conversation, resampling returned audio to 16 kHz."""

    import soundfile as sf

    _validate_channel_roles(channel_roles)
    audio_path = resolve_dataset_path(dataset_root, entry.relative_audio_path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Missing audio file: {audio_path}")
    sidecar_path = resolve_dataset_path(
        dataset_root, entry.relative_audio_path.with_suffix(".json")
    )
    if not sidecar_path.is_file():
        raise FileNotFoundError(f"Missing JSON sidecar: {sidecar_path}")

    waveform, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    if waveform.shape[1] != 2:
        raise ValueError(
            f"{audio_path}: expected exactly 2 channels, got {waveform.shape[1]}."
        )
    if (
        isinstance(sample_rate, bool)
        or not isinstance(sample_rate, int)
        or sample_rate <= 0
    ):
        raise ValueError(f"{audio_path}: invalid sample rate {sample_rate!r}.")
    if (
        expected_source_sample_rate_hz is not None
        and sample_rate != expected_source_sample_rate_hz
    ):
        raise ValueError(
            f"{audio_path}: sample rate {sample_rate} Hz does not match expected "
            f"{expected_source_sample_rate_hz} Hz."
        )
    audio_duration = waveform.shape[0] / sample_rate
    duration_tolerance = max(1.0 / sample_rate, 1e-4)
    if not math.isclose(
        audio_duration,
        entry.duration_seconds,
        rel_tol=0.0,
        abs_tol=duration_tolerance,
    ):
        raise ValueError(
            f"{audio_path}: manifest duration {entry.duration_seconds:.6f} s "
            f"does not match WAV duration {audio_duration:.6f} s."
        )

    try:
        with sidecar_path.open(encoding="utf-8") as handle:
            sidecar = json.load(handle)
    except json.JSONDecodeError as error:
        raise ValueError(f"{sidecar_path}: invalid JSON: {error.msg}.") from error
    words = parse_sidecar_word_spans(
        sidecar,
        duration_seconds=audio_duration,
        source=str(sidecar_path),
        speaker_label_map=speaker_label_map,
    )

    assistant_waveform = None
    if retain_assistant_waveform:
        assistant_waveform = resample_waveform_to_16khz(waveform[:, 0], sample_rate)
    user_waveform = resample_waveform_to_16khz(waveform[:, 1], sample_rate)
    return ConversationRecord(
        conversation_id=entry.conversation_id,
        duration_seconds=audio_duration,
        sample_rate_hz=TARGET_SAMPLE_RATE_HZ,
        source_sample_rate_hz=sample_rate,
        user_waveform=user_waveform,
        assistant_reference_path=audio_path,
        assistant_waveform=assistant_waveform,
        user_words=words[Speaker.USER],
        assistant_words=words[Speaker.ASSISTANT],
    )


def assign_split(
    conversation_id: str,
    *,
    ratios: tuple[float, float, float] = (0.9, 0.05, 0.05),
    salt: str = "DailyTalkContiguous-v1",
) -> Split:
    """Assign a conversation before any crop/window creation using SHA-256."""

    if not conversation_id:
        raise ValueError("conversation_id must not be empty.")
    if (
        len(ratios) != 3
        or any(ratio < 0.0 for ratio in ratios)
        or not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ValueError("Split ratios must be three non-negative values summing to 1.")
    digest = hashlib.sha256(f"{salt}\0{conversation_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest, "big") / (1 << (8 * len(digest)))
    if value < ratios[0]:
        return Split.TRAIN
    if value < ratios[0] + ratios[1]:
        return Split.VALIDATION
    return Split.TEST


def partition_by_split(
    entries: Iterable[ManifestEntry],
) -> dict[Split, tuple[ManifestEntry, ...]]:
    """Partition whole conversations and reject duplicate IDs (split leakage)."""

    partitions: dict[Split, list[ManifestEntry]] = {split: [] for split in Split}
    seen: set[str] = set()
    for entry in entries:
        if entry.conversation_id in seen:
            raise ValueError(
                f"Conversation {entry.conversation_id!r} occurs more than once; "
                "refusing a potentially leaky split."
            )
        seen.add(entry.conversation_id)
        partitions[assign_split(entry.conversation_id)].append(entry)
    return {split: tuple(split_entries) for split, split_entries in partitions.items()}


def find_turn_boundaries(record: ConversationRecord) -> tuple[TurnBoundary, ...]:
    """Find speaker changes in the available word annotations."""

    words = sorted(
        (*record.user_words, *record.assistant_words),
        key=lambda span: (span.start_seconds, span.end_seconds),
    )
    boundaries: list[TurnBoundary] = []
    for previous, current in zip(words, words[1:]):
        if previous.speaker is current.speaker:
            continue
        boundaries.append(
            TurnBoundary(current.start_seconds, previous.speaker, current.speaker)
        )
    return tuple(boundaries)


def _window_around(time_seconds: float, duration_seconds: float) -> tuple[float, float]:
    if duration_seconds < WINDOW_SECONDS:
        raise ValueError(
            f"Conversation duration {duration_seconds:.6f} s is shorter than "
            f"the fixed {WINDOW_SECONDS:.0f} s window."
        )
    start = min(
        max(time_seconds - WINDOW_SECONDS / 2.0, 0.0),
        duration_seconds - WINDOW_SECONDS,
    )
    return start, start + WINDOW_SECONDS


def sample_window_metadata(
    record: ConversationRecord,
    *,
    random_count: int = 1,
    seed: int = 0,
    include_boundaries: bool = True,
) -> tuple[WindowMetadata, ...]:
    """Describe random and boundary-centered fixed windows without making tensors."""

    if (
        isinstance(random_count, bool)
        or not isinstance(random_count, int)
        or random_count < 0
    ):
        raise ValueError("random_count must be a non-negative integer.")
    if record.duration_seconds < WINDOW_SECONDS:
        raise ValueError(
            f"Conversation {record.conversation_id!r} is shorter than 2 seconds."
        )

    split = assign_split(record.conversation_id)
    rng = random.Random(seed)
    max_start = record.duration_seconds - WINDOW_SECONDS
    windows: list[WindowMetadata] = []
    for _ in range(random_count):
        start = rng.uniform(0.0, max_start) if max_start else 0.0
        windows.append(
            WindowMetadata(
                record.conversation_id,
                split,
                start,
                start + WINDOW_SECONDS,
                WindowKind.RANDOM,
            )
        )

    if include_boundaries:
        for boundary in find_turn_boundaries(record):
            start, end = _window_around(boundary.time_seconds, record.duration_seconds)
            kind = (
                WindowKind.USER_TO_ASSISTANT
                if boundary.from_speaker is Speaker.USER
                else WindowKind.ASSISTANT_TO_USER
            )
            windows.append(
                WindowMetadata(
                    record.conversation_id,
                    split,
                    start,
                    end,
                    kind,
                    boundary.time_seconds,
                )
            )
    return tuple(windows)


@dataclass(frozen=True)
class TimelineSample:
    """A conversation and one explicit fixed-window view of it."""

    record: ConversationRecord
    window: WindowMetadata


# A descriptive alias for callers that think of dataset items as windowed
# records rather than timeline samples.
WindowedRecord = TimelineSample


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
    """Build, optionally augment, and pad Session 05 training examples.

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
        interruption_config: object | None = None,
        interruption_probability: float = 0.0,
        min_assistant_frames: int = 1,
        augmentation_seed: int = 0,
        context_token_ids: Sequence[int] = (),
    ) -> None:
        from .timeline import InterruptionConfig, TimelineSpec

        TimelineSpec(frame_rate_hz=frame_rate_hz)
        if interruption_config is not None and (
            interruption_probability != 0.0 or min_assistant_frames != 1
        ):
            raise ValueError(
                "Pass interruption_config or interruption probability options, "
                "not both."
            )
        self.audio_processor = audio_processor
        self.tokenizer = tokenizer
        self.control_tokens = control_tokens
        self.thinker_bos_token_id = thinker_bos_token_id
        self.frame_rate_hz = frame_rate_hz
        self.context_token_ids = tuple(context_token_ids)
        self.interruption_config = (
            InterruptionConfig(
                probability=interruption_probability,
                min_assistant_frames=min_assistant_frames,
            )
            if interruption_config is None
            else interruption_config
        )
        if not isinstance(self.interruption_config, InterruptionConfig):
            raise TypeError("interruption_config must be an InterruptionConfig.")
        self._random_state = _WorkerRandomState(augmentation_seed)

    def augmentation_rng(self) -> random.Random:
        """Return this process/worker's lazily seeded augmentation RNG."""

        return self._random_state.current()

    def _timeline(self, item: object) -> object:
        from .timeline import WindowTimeline, build_window_timeline

        if isinstance(item, WindowTimeline):
            return item
        if isinstance(item, TimelineSample):
            record, window = item.record, item.window
        elif (
            isinstance(item, Sequence)
            and not isinstance(item, (str, bytes))
            and len(item) == 2
        ):
            record, window = item
        elif isinstance(item, Mapping):
            try:
                record, window = item["record"], item["window"]
            except KeyError as error:
                raise ValueError(
                    "Mapping samples require 'record' and 'window' keys."
                ) from error
        else:
            raise TypeError(
                "Collator samples must be TimelineSample, WindowTimeline, "
                "(record, window), or a matching mapping."
            )
        if not isinstance(record, ConversationRecord) or not isinstance(
            window, (WindowMetadata, SpanMetadata, ConversationMetadata)
        ):
            raise TypeError("Sample record/window values have unexpected types.")
        return build_window_timeline(
            record,
            window,
            tokenizer=self.tokenizer,
            control_tokens=self.control_tokens,
            thinker_bos_token_id=self.thinker_bos_token_id,
            frame_rate_hz=self.frame_rate_hz,
        )

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
            duration = timeline.window_end_seconds - timeline.window_start_seconds
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

        from .timeline import (
            augment_synthetic_interruption,
            encode_causal_timeline,
        )

        if not items:
            raise ValueError("Cannot collate an empty batch.")
        rng = self.augmentation_rng()
        timelines = []
        for item in items:
            timeline = self._timeline(item)
            timeline = augment_synthetic_interruption(
                timeline,
                config=self.interruption_config,
                rng=rng,
                control_tokens=self.control_tokens,
                thinker_bos_token_id=self.thinker_bos_token_id,
            )
            timelines.append(timeline)

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


TrainingCollator = DuplexCollator
