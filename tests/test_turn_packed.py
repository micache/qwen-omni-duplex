from __future__ import annotations

import io
import random

import numpy as np
import soundfile as sf

from duplex.timeline import ControlTokenIds, EventKind
from duplex.turn_packed import (
    PackedConversation,
    augment_packed_interruption,
    build_turn_packed_timeline,
    conversation_from_row,
    mix_noise,
)
from test_timeline_alignment import CharacterTokenizer


def controls() -> ControlTokenIds:
    return ControlTokenIds(idle=6, start=7, stop=8, thinker_vocab_size=1000)


def wav_bytes(values: np.ndarray, sample_rate: int = 16_000) -> bytes:
    target = io.BytesIO()
    sf.write(target, values, sample_rate, format="WAV", subtype="FLOAT")
    return target.getvalue()


def test_taste_row_becomes_one_conversation_with_assistant_length_only():
    user = np.linspace(-0.1, 0.1, 16_000, dtype=np.float32)
    assistant = np.ones(8_000, dtype=np.float32) * 0.25
    value = conversation_from_row(
        {
            "idx": "sample-1",
            "instruction_audio": {"bytes": wav_bytes(user), "path": None},
            "response_audio": {"bytes": wav_bytes(assistant), "path": None},
            "message": [
                {"role": "user", "text": "question"},
                {"role": "assistant", "text": "Hi"},
            ],
        }
    )
    assert value.conversation_id == "sample-1"
    assert len(value.user_waveform) == 16_000
    assert value.assistant_samples == 8_000
    np.testing.assert_array_equal(value.input_waveform[:16_000], value.user_waveform)
    np.testing.assert_array_equal(value.input_waveform[16_000:], 0.0)


def test_turn_packing_is_contiguous_and_does_not_align_text_to_audio():
    value = PackedConversation(
        "sample-1", np.ones(16_000, dtype=np.float32), 16_000, "Hi"
    )
    timeline = build_turn_packed_timeline(
        value,
        tokenizer=CharacterTokenizer(),
        control_tokens=controls(),
        thinker_bos_token_id=11,
        valid_frame_count=50,
    )
    kinds = [event.kind for event in timeline.targets.events]
    assert kinds[:25] == [EventKind.IDLE] * 25
    assert kinds[25:29] == [
        EventKind.START,
        EventKind.TEXT,
        EventKind.TEXT,
        EventKind.STOP,
    ]
    assert kinds[29:] == [EventKind.IDLE] * 21
    assert [event.text_id for event in timeline.targets.events[26:28]] == [20, 21]
    assert timeline.causal.labels[:25] == [controls().idle] * 25
    assert timeline.causal.labels[25] == controls().start
    assert timeline.causal.labels[28] == controls().stop


def test_interruption_is_on_the_fly_audio_overlay_and_early_stop():
    value = PackedConversation(
        "sample-1", np.ones(16_000, dtype=np.float32), 32_000, "abcdefghij"
    )
    timeline = build_turn_packed_timeline(
        value,
        tokenizer=CharacterTokenizer(),
        control_tokens=controls(),
        thinker_bos_token_id=11,
        valid_frame_count=75,
    )
    interrupted = augment_packed_interruption(
        timeline,
        np.ones(8_000, dtype=np.float32),
        probability=1.0,
        min_assistant_frames=2,
        rng=random.Random(3),
        control_tokens=controls(),
        thinker_bos_token_id=11,
    )
    assert interrupted.interruption is not None
    cut = interrupted.interruption.cut_frame
    assert interrupted.targets.events[cut].kind is EventKind.STOP
    assert all(
        event.kind is EventKind.IDLE
        for event in interrupted.targets.events[cut + 1 :]
    )
    cut_sample = round(cut * 16_000 / 25)
    assert np.any(interrupted.user_waveform[cut_sample:] != 0.0)


def test_noise_mix_respects_requested_snr():
    signal = np.sin(np.arange(16_000, dtype=np.float32) * 0.01)
    noise = np.cos(np.arange(4_000, dtype=np.float32) * 0.031)
    mixed = mix_noise(signal, noise, snr_db=10.0, rng=random.Random(4))
    added = mixed - signal
    measured = 10.0 * np.log10(np.mean(signal.astype(np.float64) ** 2) / np.mean(added.astype(np.float64) ** 2))
    assert abs(measured - 10.0) < 0.01
