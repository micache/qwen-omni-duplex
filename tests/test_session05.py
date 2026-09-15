from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from duplex.dataset import (
    ConversationRecord,
    DuplexCollator,
    Speaker,
    Split,
    TimelineSample,
    WindowKind,
    WindowMetadata,
    WordSpan,
)
from duplex.timeline import (
    IGNORE_LABEL,
    ControlTokenIds,
    EventKind,
    InterruptionConfig,
    TargetEventSequence,
    augment_synthetic_interruption,
    build_window_timeline,
    encode_causal_timeline,
    format_timeline_table,
    validate_target_sequence,
)


class CharacterTokenizer:
    def __init__(self) -> None:
        self.piece_to_id: dict[str, int] = {}
        self.id_to_piece: dict[int, str] = {}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        result = []
        for character in text:
            if character not in self.piece_to_id:
                token_id = 20 + len(self.piece_to_id)
                self.piece_to_id[character] = token_id
                self.id_to_piece[token_id] = character
            result.append(self.piece_to_id[character])
        return result

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
    ) -> dict[str, object]:
        assert return_offsets_mapping
        return {
            "input_ids": self.encode(text, add_special_tokens=add_special_tokens),
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }

    def decode(self, token_ids: list[int], **_: object) -> str:
        return "".join(self.id_to_piece[token_id] for token_id in token_ids)


def controls() -> ControlTokenIds:
    return ControlTokenIds(10, 11, 12, thinker_vocab_size=1_000)


def sample() -> tuple[ConversationRecord, WindowMetadata, CharacterTokenizer]:
    waveform = np.zeros(200, dtype=np.float32)
    waveform[140:160] = 1.0
    waveform[180:190] = 2.0
    record = ConversationRecord(
        conversation_id="interruption-test",
        duration_seconds=2.0,
        sample_rate_hz=100,
        source_sample_rate_hz=100,
        user_waveform=waveform,
        assistant_reference_path=Path("assistant.wav"),
        assistant_waveform=None,
        user_words=(
            WordSpan("interrupt", 1.40, 1.60, Speaker.USER, source_index=1),
        ),
        assistant_words=(
            WordSpan("abcdefghij", 0.20, 1.00, Speaker.ASSISTANT, source_index=0),
            WordSpan("ok", 1.75, 1.95, Speaker.ASSISTANT, source_index=2),
        ),
    )
    window = WindowMetadata(
        record.conversation_id,
        Split.TRAIN,
        0.0,
        2.0,
        WindowKind.ASSISTANT_TO_USER,
        boundary_seconds=1.4,
    )
    return record, window, CharacterTokenizer()


def timeline():
    record, window, tokenizer = sample()
    return build_window_timeline(
        record,
        window,
        tokenizer=tokenizer,
        control_tokens=controls(),
        thinker_bos_token_id=1,
        frame_rate_hz=25,
    )


def augment(source, seed: int = 17, minimum: int = 3):
    return augment_synthetic_interruption(
        source,
        config=InterruptionConfig(
            probability=1.0, min_assistant_frames=minimum
        ),
        rng=random.Random(seed),
        control_tokens=controls(),
        thinker_bos_token_id=1,
    )


def test_augmentation_is_deterministic_for_python_and_torch_seeds() -> None:
    source = timeline()
    first = augment(source, seed=91)
    second = augment(source, seed=91)
    assert first.interruption == second.interruption
    assert first.targets.events == second.targets.events
    np.testing.assert_array_equal(first.user_waveform, second.user_waveform)

    torch_first = augment_synthetic_interruption(
        source,
        config=InterruptionConfig(probability=1.0, min_assistant_frames=3),
        rng=torch.Generator().manual_seed(44),
        control_tokens=controls(),
        thinker_bos_token_id=1,
    )
    torch_second = augment_synthetic_interruption(
        source,
        config=InterruptionConfig(probability=1.0, min_assistant_frames=3),
        rng=torch.Generator().manual_seed(44),
        control_tokens=controls(),
        thinker_bos_token_id=1,
    )
    assert torch_first.interruption == torch_second.interruption


def test_probability_zero_is_identity_and_one_applies() -> None:
    source = timeline()
    rng = random.Random(1)
    rng_state = rng.getstate()
    untouched = augment_synthetic_interruption(
        source,
        config=InterruptionConfig(probability=0.0, min_assistant_frames=3),
        rng=rng,
        control_tokens=controls(),
        thinker_bos_token_id=1,
    )
    assert untouched is source
    assert rng.getstate() == rng_state
    assert augment(source) is not source


def test_no_eligible_boundary_and_short_turn_are_identity_noops() -> None:
    source = timeline()
    assert augment(source, minimum=30) is source

    no_user = replace(source, user_words=())
    no_user = replace(
        no_user,
        frames=tuple(replace(frame, user_active=False) for frame in no_user.frames),
    )
    assert augment(no_user) is no_user


def test_minimum_cut_shifted_timing_and_full_aligned_truncation() -> None:
    source = timeline()
    result = augment(source, seed=12, minimum=4)
    assert result.interruption is not None
    interruption = result.interruption
    selected = source.utterances[0]
    original_start = next(
        frame.frame_index
        for frame in source.frames
        if frame.source_turn_id == selected.source_turn_id
        and frame.target_event.kind is EventKind.START
    )
    assert interruption.cut_frame >= original_start + 1 + 4
    assert result.targets.events[interruption.cut_frame].kind is EventKind.STOP
    assert result.frames[interruption.cut_frame].user_active
    assert result.user_words[0].start_seconds == pytest.approx(
        result.frames[interruption.cut_frame].audio_time_seconds
    )

    retained = result.utterances[0]
    retained_events = [
        frame.target_event.text_id
        for frame in result.frames[: interruption.cut_frame]
        if frame.source_turn_id == selected.source_turn_id
        and frame.target_event.kind is EventKind.TEXT
    ]
    assert list(retained.token_ids) == retained_events
    assert retained.target_text == selected.target_text[: len(retained.token_ids)]
    assert all(
        frame.source_turn_id != selected.source_turn_id
        for frame in result.frames[interruption.cut_frame + 1 :]
    )
    assert result.utterances[1].source_start_seconds == pytest.approx(
        source.utterances[1].source_start_seconds
        - interruption.shifted_frames / source.frame_rate_hz
    )

    cut_sample = interruption.cut_frame * 4
    assert result.user_waveform[cut_sample] == 1.0
    assert np.all(result.user_waveform[-cut_sample:] == 0.0)
    validate_target_sequence(result.targets, thinker_vocab_size=1_000)
    streams = (
        result.frames,
        result.targets.events,
        result.causal.text_ids,
        result.causal.text_mask,
        result.causal.control_ids,
        result.causal.control_mask,
        result.causal.labels,
        result.causal.attention_mask,
    )
    assert {len(stream) for stream in streams} == {50}

    rendered = format_timeline_table(result)
    stop_row = rendered.splitlines()[interruption.cut_frame + 2]
    assert "yes" in stop_row and "STOP" in stop_row


class FakeFeatureExtractor:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, waveforms, **kwargs):
        self.calls += 1
        assert kwargs == {
            "sampling_rate": 100,
            "padding": True,
            "return_attention_mask": True,
            "return_tensors": "pt",
        }
        batch = len(waveforms)
        features = torch.arange(batch * 4 * 7).reshape(batch, 4, 7)
        mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])[:batch]
        return {"input_features": features, "attention_mask": mask}


class FakeProcessor:
    def __init__(self) -> None:
        self.feature_extractor = FakeFeatureExtractor()

    @property
    def audio_tower(self):
        raise AssertionError("the collator must not inspect or run the audio tower")


def make_collator(processor: object, tokenizer: object) -> DuplexCollator:
    return DuplexCollator(
        audio_processor=processor,
        tokenizer=tokenizer,
        control_tokens=controls(),
        thinker_bos_token_id=1,
        interruption_probability=0.0,
        augmentation_seed=123,
    )


def test_collator_builds_windowed_record_and_does_not_run_audio_tower() -> None:
    record, window, tokenizer = sample()
    processor = FakeProcessor()
    batch = make_collator(processor, tokenizer)([TimelineSample(record, window)])

    assert processor.feature_extractor.calls == 1
    assert batch["input_features"].shape == (1, 4, 7)
    assert batch["feature_attention_mask"].shape == (1, 5)
    assert batch["preconv_feature_lengths"].tolist() == [5]
    assert batch["labels"].shape == (1, 50)
    assert batch["sample_metadata"][0]["sample_id"].startswith("session-05@")


def test_collator_padding_ignores_labels_and_preserves_valid_values() -> None:
    source = timeline()
    short_length = 30
    short_targets = TargetEventSequence(
        source.targets.events[:short_length],
        sample_id="short",
        frame_times=source.targets.frame_times[:short_length],
    )
    short_causal = encode_causal_timeline(
        short_targets,
        control_tokens=controls(),
        thinker_bos_token_id=1,
    )
    short = replace(
        source,
        sample_id="short",
        user_waveform=source.user_waveform[:120],
        frames=source.frames[:short_length],
        targets=short_targets,
        causal=short_causal,
    )
    batch = make_collator(FakeProcessor(), CharacterTokenizer())([short, source])

    assert batch["labels"].shape == (2, 50)
    assert batch["labels"][0, :short_length].tolist() == short_causal.labels
    assert batch["text_ids"][0, :short_length].tolist() == short_causal.text_ids
    assert batch["control_ids"][0, :short_length].tolist() == short_causal.control_ids
    assert torch.all(batch["labels"][0, short_length:] == IGNORE_LABEL)
    assert not torch.any(batch["attention_mask"][0, short_length:])
    assert not torch.any(batch["text_mask"][0, short_length:])
    assert not torch.any(batch["control_mask"][0, short_length:])
    assert batch["labels"][1].tolist() == source.causal.labels
    # Feature and mask time axes deliberately differ; lengths come only from
    # the returned pre-convolution mask.
    assert batch["input_features"].shape[-1] == 7
    assert batch["preconv_feature_lengths"].tolist() == [5, 3]
    assert "audio_feature_lengths" not in batch


def test_collator_rng_is_lazily_seeded_per_worker(monkeypatch) -> None:
    collator = make_collator(FakeProcessor(), CharacterTokenizer())
    monkeypatch.setattr(
        torch.utils.data,
        "get_worker_info",
        lambda: SimpleNamespace(id=0, seed=500),
    )
    worker_zero = collator.augmentation_rng()
    first_zero = worker_zero.random()
    assert collator.augmentation_rng() is worker_zero

    monkeypatch.setattr(
        torch.utils.data,
        "get_worker_info",
        lambda: SimpleNamespace(id=1, seed=501),
    )
    worker_one = collator.augmentation_rng()
    assert worker_one is not worker_zero
    assert worker_one.random() != first_zero
