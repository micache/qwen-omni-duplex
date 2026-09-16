"""Turn-packed spoken-instruction data and on-the-fly augmentation.

Each item is one complete conversation.  User turns contribute their real
waveform and IDLE targets.  Assistant turns contribute equally long silence;
their response tokens are packed contiguously at the start of that silent
block, bracketed by START and STOP, with IDLE filling the remaining frames.
"""

from __future__ import annotations

import hashlib
import io
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .conversations import exact_audio_frame_count
from .dataset import TARGET_SAMPLE_RATE_HZ, resample_waveform_to_16khz
from .timeline import (
    ControlTokenIds,
    EventKind,
    FrameAlignment,
    InterruptionMetadata,
    ResponseState,
    TargetEvent,
    TargetEventSequence,
    WindowTimeline,
    _decode,
    _encode_utterance,
    encode_causal_timeline,
)


DATASET_VIEW = "TASTE-IF-SFT-48K"
DATASET_ID = "Jaylin0418/TASTE-IF-SFT-48K"
DATASET_REVISION = "main"
NOISE_DATASET_ID = "corypaik/musan"
NOISE_CONFIG = "noise"


@dataclass(frozen=True)
class PackedConversation:
    conversation_id: str
    user_waveform: np.ndarray
    assistant_samples: int
    assistant_text: str
    sample_rate_hz: int = TARGET_SAMPLE_RATE_HZ

    def __post_init__(self) -> None:
        waveform = np.asarray(self.user_waveform, dtype=np.float32)
        if waveform.ndim != 1 or not len(waveform):
            raise ValueError("user_waveform must be a non-empty mono waveform.")
        if self.sample_rate_hz != TARGET_SAMPLE_RATE_HZ:
            raise ValueError("Packed conversations must be resampled to 16 kHz.")
        if self.assistant_samples <= 0:
            raise ValueError("assistant_samples must be positive.")
        if not isinstance(self.assistant_text, str) or not self.assistant_text.strip():
            raise ValueError("assistant_text must be non-empty.")
        object.__setattr__(self, "user_waveform", waveform)

    @property
    def input_waveform(self) -> np.ndarray:
        return np.concatenate(
            (self.user_waveform, np.zeros(self.assistant_samples, dtype=np.float32))
        )


def _audio_bytes(value: object, *, field: str) -> bytes:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an Audio mapping.")
    payload = value.get("bytes")
    if isinstance(payload, memoryview):
        payload = payload.tobytes()
    if not isinstance(payload, bytes):
        path = value.get("path")
        if isinstance(path, str) and path:
            return Path(path).read_bytes()
        raise ValueError(f"{field} contains neither bytes nor a readable path.")
    return payload


def decode_audio(value: object, *, field: str) -> np.ndarray:
    """Decode an HF Audio(decode=False) value and normalize it to mono 16 kHz."""

    import soundfile as sf

    waveform, sample_rate = sf.read(
        io.BytesIO(_audio_bytes(value, field=field)), dtype="float32", always_2d=True
    )
    if waveform.shape[1] == 1:
        mono = waveform[:, 0]
    else:
        mono = waveform.mean(axis=1, dtype=np.float32)
    return resample_waveform_to_16khz(mono, int(sample_rate))


def conversation_from_row(row: Mapping[str, object]) -> PackedConversation:
    """Validate one default-config TASTE row without using timestamp alignment."""

    conversation_id = row.get("idx")
    messages = row.get("message")
    if not isinstance(conversation_id, str) or not conversation_id:
        raise ValueError("TASTE row has no non-empty idx.")
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        raise ValueError(f"{conversation_id}: message must be a two-turn sequence.")
    if len(messages) != 2 or not all(isinstance(item, Mapping) for item in messages):
        raise ValueError(f"{conversation_id}: expected exactly user and assistant turns.")
    user, assistant = messages
    if user.get("role") != "user" or assistant.get("role") != "assistant":
        raise ValueError(f"{conversation_id}: expected user then assistant roles.")
    assistant_text = assistant.get("text")
    if not isinstance(assistant_text, str) or not assistant_text.strip():
        raise ValueError(f"{conversation_id}: assistant text is empty.")
    user_waveform = decode_audio(row.get("instruction_audio"), field="instruction_audio")
    assistant_waveform = decode_audio(row.get("response_audio"), field="response_audio")
    return PackedConversation(
        conversation_id=conversation_id,
        user_waveform=user_waveform,
        assistant_samples=len(assistant_waveform),
        assistant_text=assistant_text,
    )


class TasteConversationDataset(Dataset):
    """Lazy local-Parquet view; every index is one complete conversation."""

    def __init__(self, dataset: object) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> PackedConversation:
        return conversation_from_row(self.dataset[index])  # type: ignore[index]


def load_local_taste(
    root: str | Path,
    *,
    split: str,
    max_samples: int | None = None,
) -> TasteConversationDataset:
    """Open the downloaded default config without contacting the Hub."""

    from datasets import load_dataset

    root = Path(root)
    pattern = "shuffled_dev.parquet" if split == "dev" else "shuffled_train_part_*.parquet"
    files = sorted((root / "data").glob(pattern))
    if not files:
        raise FileNotFoundError(f"No TASTE {split} Parquet files under {root / 'data'}.")
    dataset = load_dataset(
        "parquet",
        data_files={split: [str(path) for path in files]},
        split=split,
        cache_dir=str(root / ".datasets_cache"),
    )
    if max_samples is not None:
        if max_samples < 1:
            raise ValueError("max_samples must be positive when supplied.")
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return TasteConversationDataset(dataset)


def build_turn_packed_timeline(
    conversation: PackedConversation,
    *,
    tokenizer: object,
    control_tokens: ControlTokenIds,
    thinker_bos_token_id: int,
    valid_frame_count: int,
) -> WindowTimeline:
    """Pack response tokens at the assistant boundary; timestamps are unused."""

    if valid_frame_count < 3:
        raise ValueError(f"{conversation.conversation_id}: fewer than three audio frames.")
    token_ids, _ = _encode_utterance(tokenizer, conversation.assistant_text)
    decoded = _decode(tokenizer, token_ids)
    if decoded != conversation.assistant_text:
        raise ValueError(
            f"{conversation.conversation_id}: tokenizer round trip changed assistant text; "
            f"target={conversation.assistant_text!r}, decoded={decoded!r}."
        )

    total_samples = len(conversation.user_waveform) + conversation.assistant_samples
    user_frames = round(len(conversation.user_waveform) * 25 / conversation.sample_rate_hz)
    user_frames = min(max(user_frames, 1), valid_frame_count - 2)
    assistant_frames = valid_frame_count - user_frames
    required = len(token_ids) + 2
    if required > assistant_frames:
        raise ValueError(
            f"{conversation.conversation_id}: {len(token_ids)} response tokens plus "
            f"START/STOP need {required} frames, but the assistant waveform supplies "
            f"only {assistant_frames}."
        )

    events = [TargetEvent(EventKind.IDLE) for _ in range(valid_frame_count)]
    events[user_frames] = TargetEvent(EventKind.START)
    for offset, token_id in enumerate(token_ids, start=1):
        events[user_frames + offset] = TargetEvent(EventKind.TEXT, token_id)
    stop_frame = user_frames + len(token_ids) + 1
    events[stop_frame] = TargetEvent(EventKind.STOP)
    frame_times = [frame / 25 for frame in range(valid_frame_count)]
    targets = TargetEventSequence(
        events, sample_id=conversation.conversation_id, frame_times=frame_times
    )
    causal = encode_causal_timeline(
        targets,
        control_tokens=control_tokens,
        thinker_bos_token_id=thinker_bos_token_id,
    )

    frames: list[FrameAlignment] = []
    active = False
    for index, event in enumerate(events):
        if event.kind is EventKind.START:
            active = True
        elif event.kind is EventKind.STOP:
            active = False
        frames.append(
            FrameAlignment(
                frame_index=index,
                audio_time_seconds=frame_times[index],
                user_active=index < user_frames,
                target_event=event,
                decoded_token=(
                    _decode(tokenizer, [event.text_id])
                    if event.kind is EventKind.TEXT and event.text_id is not None
                    else ""
                ),
                state=ResponseState.ACTIVE if active else ResponseState.INACTIVE,
                source_turn_id=(
                    f"{conversation.conversation_id}:assistant:0"
                    if index >= user_frames
                    else f"{conversation.conversation_id}:user:0"
                ),
            )
        )
    duration = total_samples / conversation.sample_rate_hz
    return WindowTimeline(
        sample_id=conversation.conversation_id,
        window_start_seconds=0.0,
        window_end_seconds=duration,
        frame_rate_hz=25,
        user_waveform=conversation.input_waveform,
        utterances=(),
        frames=tuple(frames),
        targets=targets,
        causal=causal,
        audio_sample_rate_hz=conversation.sample_rate_hz,
    )


@dataclass(frozen=True)
class NoiseAugmentationConfig:
    probability: float = 0.0
    min_snr_db: float = 5.0
    max_snr_db: float = 20.0

    def __post_init__(self) -> None:
        values = (self.probability, self.min_snr_db, self.max_snr_db)
        if any(isinstance(value, bool) or not math.isfinite(float(value)) for value in values):
            raise ValueError("Noise settings must be finite numbers.")
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("noise probability must be in [0, 1].")
        if self.min_snr_db > self.max_snr_db:
            raise ValueError("min_snr_db cannot exceed max_snr_db.")


class MusanNoiseDataset:
    """Lazy local view of only MUSAN's noise config (930 clips, about 696 MB)."""

    def __init__(self, root: str | Path) -> None:
        from datasets import Audio, load_dataset

        files = sorted((Path(root) / "noise").glob("train-*.parquet"))
        if not files:
            raise FileNotFoundError(f"No MUSAN noise Parquet files under {Path(root) / 'noise'}.")
        dataset = load_dataset(
            "parquet",
            data_files={"train": [str(path) for path in files]},
            split="train",
            cache_dir=str(Path(root) / ".datasets_cache"),
        )
        self.dataset = dataset.cast_column("audio", Audio(decode=False))

    def __len__(self) -> int:
        return len(self.dataset)

    def waveform(self, index: int) -> np.ndarray:
        row = self.dataset[index]
        return decode_audio(row["audio"], field="MUSAN audio")


def mix_noise(
    waveform: np.ndarray,
    noise: np.ndarray,
    *,
    snr_db: float,
    rng: random.Random,
) -> np.ndarray:
    """Tile/crop one noise clip and add it at a measured full-input SNR."""

    signal = np.asarray(waveform, dtype=np.float32)
    source = np.asarray(noise, dtype=np.float32)
    if signal.ndim != 1 or source.ndim != 1 or not len(source):
        raise ValueError("Signal and noise must be non-empty mono waveforms.")
    repeats = (len(signal) + len(source) - 1) // len(source) + 1
    tiled = np.tile(source, repeats)
    maximum_start = len(tiled) - len(signal)
    start = rng.randrange(maximum_start + 1) if maximum_start else 0
    selected = tiled[start : start + len(signal)].astype(np.float32, copy=True)
    selected -= selected.mean(dtype=np.float64)
    signal_power = float(np.mean(np.square(signal, dtype=np.float64)))
    noise_power = float(np.mean(np.square(selected, dtype=np.float64)))
    if signal_power <= 0.0 or noise_power <= 0.0:
        return signal.copy()
    scale = math.sqrt(signal_power / (noise_power * 10.0 ** (snr_db / 10.0)))
    return (signal + selected * scale).astype(np.float32)


def augment_packed_interruption(
    timeline: WindowTimeline,
    donor_user_waveform: np.ndarray,
    *,
    probability: float,
    min_assistant_frames: int,
    rng: random.Random,
    control_tokens: ControlTokenIds,
    thinker_bos_token_id: int,
) -> WindowTimeline:
    """Overlay donor user speech during the packed response and stop immediately."""

    if probability == 0.0 or (probability < 1.0 and rng.random() >= probability):
        return timeline
    starts = [i for i, event in enumerate(timeline.targets.events) if event.kind is EventKind.START]
    stops = [i for i, event in enumerate(timeline.targets.events) if event.kind is EventKind.STOP]
    if len(starts) != 1 or len(stops) != 1:
        return timeline
    start_frame, stop_frame = starts[0], stops[0]
    first_cut = start_frame + max(1, min_assistant_frames)
    if first_cut >= stop_frame:
        return timeline
    cut_frame = rng.randrange(first_cut, stop_frame)
    cut_sample = round(cut_frame * timeline.audio_sample_rate_hz / timeline.frame_rate_hz)
    waveform = np.asarray(timeline.user_waveform, dtype=np.float32).copy()
    donor = np.asarray(donor_user_waveform, dtype=np.float32)
    available = min(len(donor), len(waveform) - cut_sample)
    if available <= 0:
        return timeline
    waveform[cut_sample : cut_sample + available] += donor[:available]

    events = list(timeline.targets.events)
    events[cut_frame] = TargetEvent(EventKind.STOP)
    for index in range(cut_frame + 1, len(events)):
        events[index] = TargetEvent(EventKind.IDLE)
    targets = TargetEventSequence(
        events, sample_id=timeline.sample_id, frame_times=timeline.targets.frame_times
    )
    causal = encode_causal_timeline(
        targets,
        control_tokens=control_tokens,
        thinker_bos_token_id=thinker_bos_token_id,
    )
    shifted_frames = tuple(
        replace(
            frame,
            target_event=events[frame.frame_index],
            user_active=(frame.user_active or frame.frame_index >= cut_frame),
            decoded_token=(
                frame.decoded_token
                if events[frame.frame_index].kind is EventKind.TEXT
                else ""
            ),
            state=(
                ResponseState.ACTIVE
                if start_frame <= frame.frame_index < cut_frame
                else ResponseState.INACTIVE
            ),
        )
        for frame in timeline.frames
    )
    return replace(
        timeline,
        user_waveform=waveform,
        targets=targets,
        causal=causal,
        frames=shifted_frames,
        interruption=InterruptionMetadata(
            source_turn_id=f"{timeline.sample_id}:assistant:0",
            cut_frame=cut_frame,
            original_stop_frame=stop_frame,
            original_user_frame=cut_frame,
            shifted_frames=0,
        ),
    )


class TurnPackedCollator:
    """Build packed conversations and apply interruption/noise per batch call."""

    def __init__(
        self,
        *,
        audio_processor: object,
        audio_tower: object,
        tokenizer: object,
        control_tokens: ControlTokenIds,
        thinker_bos_token_id: int,
        interruption_probability: float,
        min_assistant_frames: int,
        noise_dataset: MusanNoiseDataset | None,
        noise_config: NoiseAugmentationConfig,
        augmentation_seed: int,
        context_token_ids: Sequence[int] = (),
    ) -> None:
        from .dataset import DuplexCollator

        self.base = DuplexCollator(
            audio_processor=audio_processor,
            tokenizer=tokenizer,
            control_tokens=control_tokens,
            thinker_bos_token_id=thinker_bos_token_id,
            frame_rate_hz=25,
            interruption_probability=0.0,
            augmentation_seed=augmentation_seed,
            context_token_ids=context_token_ids,
        )
        self.audio_processor = audio_processor
        self.audio_tower = audio_tower
        self.tokenizer = tokenizer
        self.control_tokens = control_tokens
        self.thinker_bos_token_id = thinker_bos_token_id
        self.interruption_probability = float(interruption_probability)
        self.min_assistant_frames = int(min_assistant_frames)
        if not 0.0 <= self.interruption_probability <= 1.0:
            raise ValueError("interruption_probability must be in [0, 1].")
        if self.min_assistant_frames < 1:
            raise ValueError("min_assistant_frames must be positive.")
        self.noise_dataset = noise_dataset
        self.noise_config = noise_config

    def _timeline(self, item: object) -> WindowTimeline:
        if isinstance(item, WindowTimeline):
            return item
        if not isinstance(item, PackedConversation):
            raise TypeError("TurnPackedCollator requires PackedConversation items.")
        frames = exact_audio_frame_count(
            item.input_waveform, self.audio_processor, self.audio_tower
        )
        return build_turn_packed_timeline(
            item,
            tokenizer=self.tokenizer,
            control_tokens=self.control_tokens,
            thinker_bos_token_id=self.thinker_bos_token_id,
            valid_frame_count=frames,
        )

    def __call__(self, items: Sequence[object]) -> dict[str, Any]:
        if not items:
            raise ValueError("Cannot collate an empty batch.")
        rng = self.base.augmentation_rng()
        conversations = [item for item in items if isinstance(item, PackedConversation)]
        timelines = [self._timeline(item) for item in items]
        for index, timeline in enumerate(timelines):
            if conversations:
                donor = conversations[(index + 1) % len(conversations)].user_waveform
                timeline = augment_packed_interruption(
                    timeline,
                    donor,
                    probability=self.interruption_probability,
                    min_assistant_frames=self.min_assistant_frames,
                    rng=rng,
                    control_tokens=self.control_tokens,
                    thinker_bos_token_id=self.thinker_bos_token_id,
                )
            if (
                self.noise_dataset is not None
                and self.noise_config.probability > 0.0
                and rng.random() < self.noise_config.probability
            ):
                noise_index = rng.randrange(len(self.noise_dataset))
                snr = rng.uniform(
                    self.noise_config.min_snr_db, self.noise_config.max_snr_db
                )
                timeline = replace(
                    timeline,
                    user_waveform=mix_noise(
                        timeline.user_waveform,
                        self.noise_dataset.waveform(noise_index),
                        snr_db=snr,
                        rng=rng,
                    ),
                )
            timelines[index] = timeline

        # The established collator owns extractor masks, padding, and the exact
        # model-input schema.  Its interruption probability is fixed at zero.
        batch = self.base(timelines)
        batch["audio_cache_keys"] = tuple(
            hashlib.sha256(
                b"qwen-turn-packed-v1\0" + memoryview(timeline.user_waveform).tobytes()
            ).hexdigest()
            for timeline in timelines
        )
        return batch
