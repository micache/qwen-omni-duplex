from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from duplex.model import QwenDuplexThinker
from duplex.streaming import (
    AUDIO_SAMPLE_RATE_HZ,
    EventTraceRecord,
    QwenDuplexStreamer,
    group_lexical_events_into_words,
    split_fixed_audio_chunks,
)


VOCAB = 10
HIDDEN = 4
IDLE = 7
START = 8
STOP = 9


class Cache:
    def __init__(self, length):
        self.length = length

    def get_seq_length(self):
        return self.length


class ScriptedCausalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.step = 0

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        current = kwargs["inputs_embeds"].shape[1]
        previous = kwargs["past_key_values"]
        length = current if previous is None else previous.length + current
        hidden = torch.zeros_like(kwargs["inputs_embeds"])
        if previous is not None:
            hidden[..., 0] = self.step
            self.step += 1
        return SimpleNamespace(last_hidden_state=hidden, past_key_values=Cache(length))


class ScriptedHead(nn.Linear):
    def __init__(self, events):
        super().__init__(HIDDEN, VOCAB, bias=False)
        self.events = events

    def forward(self, hidden):
        rows = []
        for code in hidden[..., 0].reshape(-1).tolist():
            item = self.events[int(code)]
            if isinstance(item, tuple):
                selected, raw = item
            else:
                selected = raw = item
            logits = torch.full((VOCAB,), -20.0, device=hidden.device)
            logits[selected] = 10.0
            logits[raw] = 20.0
            rows.append(logits)
        return torch.stack(rows).reshape(*hidden.shape[:-1], VOCAB)


class MockAudioTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def _get_feat_extract_output_lengths(self, lengths):
        return lengths, lengths

    def forward(self, input_features, *, feature_attention_mask):
        self.calls += 1
        length = int(feature_attention_mask.sum())
        return SimpleNamespace(
            last_hidden_state=torch.zeros(
                (length, HIDDEN), device=input_features.device, dtype=input_features.dtype
            )
        )


class MockThinker(nn.Module):
    def __init__(self, events):
        super().__init__()
        self.config = SimpleNamespace(
            bos_token_id=6,
            text_config=SimpleNamespace(vocab_size=VOCAB),
        )
        self.embedding = nn.Embedding(VOCAB, HIDDEN)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.arange(VOCAB * HIDDEN).reshape(VOCAB, HIDDEN))
        self.audio_tower = MockAudioTower()
        self.model = ScriptedCausalModel()
        self.lm_head = ScriptedHead(events)

    def get_input_embeddings(self):
        return self.embedding

    def get_audio_features(self, *, input_features, feature_attention_mask, return_dict):
        assert return_dict
        return self.audio_tower(
            input_features, feature_attention_mask=feature_attention_mask
        )

    def gradient_checkpointing_disable(self):
        pass


class MockFeatureExtractor:
    def __init__(self):
        self.input_lengths = []

    def __call__(self, waveforms, **kwargs):
        assert kwargs["sampling_rate"] == AUDIO_SAMPLE_RATE_HZ
        self.input_lengths.append(len(waveforms[0]))
        frames = max(1, round(len(waveforms[0]) * 25 / AUDIO_SAMPLE_RATE_HZ))
        return {
            "input_features": torch.zeros((1, 1, frames)),
            "feature_attention_mask": torch.ones((1, frames), dtype=torch.bool),
        }


class MockTokenizer:
    pieces = {1: "Hel", 2: "lo", 3: " ", 4: "�", 5: "世界"}

    def decode(self, token_ids, **kwargs):
        assert kwargs == {
            "skip_special_tokens": False,
            "clean_up_tokenization_spaces": False,
        }
        return "".join(self.pieces[token] for token in token_ids).replace("�世界", "世界")


def qwen_config():
    return SimpleNamespace(
        thinker_config=SimpleNamespace(
            seconds_per_chunk=2,
            position_id_per_seconds=25,
            text_config=SimpleNamespace(vocab_size=VOCAB),
        ),
        talker_config=SimpleNamespace(
            tts_text_pad_token_id=IDLE,
            tts_text_start_token_id=START,
            tts_text_end_token_id=STOP,
        ),
    )


def streamer(events, *, max_silent_chunks=0):
    thinker = MockThinker(events)
    model = QwenDuplexThinker(thinker, qwen_config=qwen_config())
    extractor = MockFeatureExtractor()
    value = QwenDuplexStreamer(
        model,
        extractor,
        MockTokenizer(),
        max_silent_chunks=max_silent_chunks,
    )
    return value, thinker, extractor


def samples_for_frames(count):
    return np.zeros(round(count * AUDIO_SAMPLE_RATE_HZ / 25), dtype=np.float32)


def test_fixed_chunks_pad_only_final_and_reject_wrong_audio_contract():
    chunks = split_fixed_audio_chunks(np.arange(36_000, dtype=np.float32))

    assert [chunk.valid_samples for chunk in chunks] == [32_000, 4_000]
    assert chunks[0].valid_mask.all()
    assert chunks[1].valid_mask[:4_000].all()
    assert not chunks[1].valid_mask[4_000:].any()
    assert np.all(chunks[1].waveform[4_000:] == 0)
    assert chunks[1].end_time_s == chunks[1].available_time_s == 2.25
    with pytest.raises(ValueError, match="mono"):
        split_fixed_audio_chunks(np.zeros((2, 100)))
    with pytest.raises(ValueError, match="16 kHz"):
        split_fixed_audio_chunks(np.zeros(100), sample_rate_hz=8_000)


def test_scripted_wait_start_text_idle_stop_and_previous_event_inputs():
    value, thinker, _ = streamer([(IDLE, 1), START, 1, IDLE, STOP])
    result = value.run(samples_for_frames(5), sample_id="flow", context_token_ids=[2, 3])

    assert [record.event_type for record in result.trace] == [
        "IDLE",
        "START",
        "TEXT",
        "IDLE",
        "STOP",
    ]
    assert result.text == "Hel"
    assert result.stop_reason == "STOP"
    assert result.trace[0].grammar_mask_changed_raw_argmax
    assert result.trace[0].raw_argmax_id == 1
    assert len(result.lexical_hidden_states) == 1
    assert thinker.audio_tower.calls == 1
    assert len(thinker.model.calls) == 6  # one prefill, five cached feature steps
    step_fusions = [call["inputs_embeds"] for call in thinker.model.calls[1:]]
    assert torch.equal(step_fusions[0], thinker.get_input_embeddings()(torch.tensor([[6]])))
    for index, previous in enumerate([IDLE, START, 1, IDLE], start=1):
        torch.testing.assert_close(
            step_fusions[index], thinker.embedding(torch.tensor([[previous]]))
        )


def test_active_interruption_masks_start_and_logs_changed_raw_argmax():
    value, _, _ = streamer([START, 1, (STOP, START)])
    result = value.run(samples_for_frames(3), sample_id="interrupt", context_token_ids=[2])

    assert [record.event_type for record in result.trace] == ["START", "TEXT", "STOP"]
    assert result.trace[-1].raw_argmax_id == START
    assert result.trace[-1].grammar_mask_changed_raw_argmax
    assert result.trace[-1].state_before == "active"
    assert result.trace[-1].state_after == "inactive"


def test_silent_tail_runs_until_stop_and_audio_tower_runs_once_per_chunk():
    events = [START, 1, STOP] + [IDLE] * 49
    value, thinker, extractor = streamer(events, max_silent_chunks=2)
    result = value.run(samples_for_frames(2), sample_id="tail", context_token_ids=[2])

    assert [record.event_type for record in result.trace] == ["START", "TEXT", "STOP"]
    assert result.trace[-1].silent_tail
    assert result.processed_silent_chunks == 1
    assert thinker.audio_tower.calls == 2
    assert extractor.input_lengths == [1_280, 32_000]


def test_cache_growth_and_chunk_boundary_causality():
    value, thinker, _ = streamer([IDLE] * 55)
    result = value.run(samples_for_frames(55), sample_id="causal", context_token_ids=[2, 3])

    assert result.timed_out
    assert thinker.audio_tower.calls == 2
    assert [call["past_key_values"].length for call in thinker.model.calls[1:]] == list(
        range(2, 57)
    )
    assert {record.available_time_s for record in result.trace[:50]} == {2.0}
    assert {record.available_time_s for record in result.trace[50:]} == {2.2}
    assert all(
        record.available_time_s >= record.chunk_end_time_s for record in result.trace
    )


def test_unicode_subwords_decode_exactly_and_words_keep_token_times():
    value, _, _ = streamer([START, 1, 2, 3, 4, 5, STOP])
    result = value.run(samples_for_frames(7), sample_id="unicode", context_token_ids=[2])

    assert result.text == "Hello 世界"
    assert "".join(record.decoded_delta for record in result.trace) == result.text
    assert [record.decoded_delta for record in result.trace[1:6]] == [
        "Hel",
        "lo",
        " ",
        "",
        "世界",
    ]
    assert [word.text for word in result.word_spans] == ["Hello", "世界"]
    assert [timing.token_id for timing in result.word_spans[0].token_timings] == [1, 2]
    assert [timing.token_id for timing in result.word_spans[1].token_timings] == [4, 5]
    assert [timing.audio_time_s for timing in result.word_spans[1].token_timings] == [
        result.trace[4].audio_time_s,
        result.trace[5].audio_time_s,
    ]


def test_word_grouping_uses_token_events_instead_of_uniform_final_timestamps():
    records = [
        EventTraceRecord(
            sample_id="words",
            chunk_index=0,
            frame_index=index,
            audio_time_s=index / 10,
            available_time_s=2.0,
            compute_ms=1.0,
            event_id=token,
            event_type="TEXT",
            decoded_delta="",
            state_before="active",
            state_after="active",
            grammar_mask_changed_raw_argmax=False,
            raw_argmax_id=token,
            chunk_end_time_s=2.0,
            silent_tail=False,
        )
        for index, token in enumerate([1, 2, 3, 4, 5])
    ]
    words = group_lexical_events_into_words(records, MockTokenizer())

    assert [timing.audio_time_s for timing in words[0].token_timings] == [0.0, 0.1]
    assert [timing.audio_time_s for timing in words[1].token_timings] == [0.3, 0.4]


def test_timeout_without_stop_after_configured_silent_tail():
    value, thinker, _ = streamer([IDLE] * 51, max_silent_chunks=1)
    result = value.run(samples_for_frames(1), sample_id="timeout", context_token_ids=[2])

    assert result.timed_out
    assert result.stop_reason == "max_silent_chunks"
    assert result.processed_silent_chunks == 1
    assert len(result.trace) == 51
    assert thinker.audio_tower.calls == 2
