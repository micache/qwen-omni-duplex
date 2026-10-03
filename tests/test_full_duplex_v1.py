from types import SimpleNamespace

import pytest

from benchmarks.full_duplex_v1 import event_segments, score_output, timed_output


CONTROLS = SimpleNamespace(idle=1, start=2, stop=3)


class Tokenizer:
    all_special_ids = [99]

    def decode(self, ids, **kwargs):
        return "".join({1: "", 2: "", 3: "", 10: " Hello", 11: " wor", 12: "ld.",
            13: " Next", 99: ""}[i] for i in ids)


def test_word_times_include_every_subtoken_and_separate_answers():
    segments = event_segments([1, 2, 10, 11, 12, 99, 3, 2, 13, 3], Tokenizer(), CONTROLS)
    assert [s["text"] for s in segments] == [" Hello world.", " Next"]
    assert segments[0]["words"][1]["timestamp"] == pytest.approx([3 / 25, 5 / 25])
    assert segments[1]["start_frame"] == 7
    assert segments[1]["stop_frame"] == 9
    assert len(segments[1]["words"]) == 1


def test_two_second_buffer_is_causal_at_chunk_boundaries():
    events = [1] * 48 + [2, 10, 11, 12, 3]
    segments = event_segments(events, Tokenizer(), CONTROLS)
    causal = timed_output(segments, 10)
    nominal = timed_output(segments, 10, delay_s=0)
    for word, original in zip(causal["chunks"], nominal["chunks"]):
        assert word["timestamp"][0] - original["timestamp"][0] == pytest.approx(2)
        assert word["timestamp"][1] - word["timestamp"][0] == pytest.approx(
            original["timestamp"][1] - original["timestamp"][0])
        # The first token of each word cannot be emitted before its chunk arrives.
        frame = round(original["timestamp"][0] * 25)
        assert word["timestamp"][0] >= (frame // 50 + 1) * 2


def test_empty_control_answers_do_not_count_and_open_answers_are_kept():
    segments = event_segments([2, 1, 3, 2, 10], Tokenizer(), CONTROLS)
    assert segments[0]["interval"] is None
    assert segments[1]["stop_frame"] is None
    output = timed_output(segments, 8)
    assert len(output["chunks"]) == len(output["response_intervals"]) == 1


def test_recording_horizon_and_interruption_crop_keep_original_clock():
    segments = [{"words": [{"text": "a", "timestamp": [0, 0.4]},
        {"text": "b", "timestamp": [1, 1.4]}, {"text": "c", "timestamp": [4, 5]}]}]
    output = timed_output(segments, 3.2, crop_start=2.2)
    assert output["chunks"] == [{"text": "a", "timestamp": [2.2, 2.4]},
        {"text": "b", "timestamp": [3.0, 3.2]}]
    assert output["response_intervals"] == [[2.2, 3.2]]


def output_for_words(count, duration):
    return {"chunks": [{"text": "word", "timestamp": [i * duration / count, (i + 1) * duration / count]}
        for i in range(count)], "response_intervals": [[0, duration]] if count else [], "duration_s": 10}


@pytest.mark.parametrize("count,duration,takeover", [(0, 0, False), (3, .9, False),
    (4, .9, True), (1, 1, True)])
def test_pause_and_turn_taking_use_released_three_word_threshold(count, duration, takeover):
    row = {"task": "smooth_turn_taking", "annotation": [{"timestamp": [0.5, 1]}]}
    score = score_output(row, output_for_words(count, duration))
    assert score["takeover"] is takeover
    assert score["latency_s"] == (0 if takeover else None)


def test_backchannel_silence_has_jsd_one_and_frequency_zero():
    score = score_output({"task": "backchannel"}, output_for_words(0, 0), [1, 1])
    assert score["takeover"] is False
    assert score["jsd"] == 1
    assert score["frequency_per_second"] == 0


def test_backchannel_matches_released_last_region_and_long_region_rules():
    output = {"chunks": [{"text": "a", "timestamp": [0, 1]},
        {"text": "b", "timestamp": [2, 2.3]}],
        "response_intervals": [[0, 1], [2, 2.3]], "duration_s": 10}
    score = score_output({"task": "backchannel"}, output, [1, 1])
    assert score["takeover"] is False  # Literal upstream overwrite for short regions.
    assert score["any_region_takeover"] is True
    assert score["frequency_per_second"] == .2
    output["response_intervals"] = [[0, 3.01], [4, 4.3]]
    score = score_output({"task": "backchannel"}, output, [1, 1])
    assert score["takeover"] is True
    assert score["counted_regions"] == 0
    assert score["jsd"] == 1


def test_interruption_latency_is_based_on_interrupt_end_after_crop():
    row = {"task": "user_interruption", "annotation": [{"timestamp": [7, 10]}]}
    output = {"chunks": [{"text": "a", "timestamp": [11, 12]}],
        "response_intervals": [[11, 12]], "duration_s": 20}
    assert score_output(row, output)["latency_s"] == 1


@pytest.mark.parametrize("events", [[3], [10], [2, 2]])
def test_malformed_grammar_is_rejected(events):
    with pytest.raises(ValueError):
        event_segments(events, Tokenizer(), CONTROLS)


def test_fixed_recording_inference_keeps_second_response_after_stop(monkeypatch):
    from contextlib import nullcontext
    import numpy as np
    import torch
    from benchmarks.voicebench_batch import BatchedDuplex

    monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: nullcontext())
    plan = {0: 2, 1: 10, 2: 3, 55: 2, 56: 10, 57: 3}

    class Decoder:
        frame = -1

        def __call__(self, **kwargs):
            self.frame += 1
            return SimpleNamespace(last_hidden_state=torch.tensor([[[float(self.frame), 0]]]),
                past_key_values=None)

    def head(hidden):
        logits = torch.full((1, 16), -100.0)
        logits[0, plan.get(int(hidden[0, 0]), 1)] = 100
        return logits

    thinker = SimpleNamespace(config=SimpleNamespace(bos_token_id=0), model=Decoder(), lm_head=head,
        get_input_embeddings=lambda: torch.nn.Embedding(16, 2))
    engine = BatchedDuplex.__new__(BatchedDuplex)
    engine.model = SimpleNamespace(control_tokens=CONTROLS, base_thinker=thinker,
        scale_audio_for_response=lambda audio, active: audio)
    engine.processor = SimpleNamespace(tokenizer=Tokenizer())
    engine.device, engine.dtype = torch.device("cpu"), torch.float32
    engine.silence = torch.zeros(50, 2)
    engine.encode_chunks = lambda chunks: [torch.zeros(50, 2) for _ in chunks]
    engine.max_new_tokens, engine.max_silent_chunks, engine.min_silent_chunks = 100000, 0, 0
    result = engine.generate([np.zeros(64000, dtype=np.float32)], use_graph=False)[0]
    assert len(result["event_ids"]) == result["input_frames"] == 100
    assert result["event_ids"].count(CONTROLS.start) == 2
    assert result["event_ids"].count(CONTROLS.stop) == 2
    assert result["event_ids"][55:58] == [2, 10, 3]
