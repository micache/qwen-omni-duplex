import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from benchmarks import full_duplex_bench as bench
from duplex.streaming import EventTraceRecord


class Tokenizer:
    pieces = {1: "I", 2: " was", 3: " explaining", 4: " New", 5: " answer", 6: " now"}

    def decode(self, ids, **kwargs):
        return "".join(self.pieces[i] for i in ids)


def event(index, kind, when, token=0, *, logical=None):
    return EventTraceRecord(
        sample_id="17", chunk_index=0, frame_index=index,
        audio_time_s=when if logical is None else logical,
        available_time_s=when, compute_ms=0.0, event_id=token,
        event_type=kind, decoded_delta="", state_before="inactive" if kind == "START" else "active",
        state_after="inactive" if kind == "STOP" else "active",
        grammar_mask_changed_raw_argmax=False, raw_argmax_id=token,
        chunk_end_time_s=when, silent_tail=False)


def generated(events):
    return bench.GeneratedTimeline(Tokenizer(), tuple(events), 4.0, "STOP")


def sample(task):
    return bench.BenchmarkSample("17", task, Path("/unused/17"), Path("/unused/17/input.wav"),
                                 4.0, {"timestamps": [1.0, 2.0], "context_text": "Prior question",
                                       "current_turn_text": "Overlap request"})


PRE_AND_NEXT = [event(0, "START", 0.2), event(1, "TEXT", 0.4, 1),
                event(2, "TEXT", 0.8, 2), event(3, "TEXT", 1.3, 3),
                event(4, "STOP", 1.5), event(5, "START", 2.1),
                event(6, "TEXT", 2.2, 4), event(7, "TEXT", 2.3, 5),
                event(8, "STOP", 2.5)]


@pytest.mark.parametrize("task", bench.V15_TASKS)
def test_each_scenario_exact_paper_formulas_and_post_overlap_slice(task):
    record = bench.evaluate_v15_timeline(sample(task), generated(PRE_AND_NEXT),
                                         model_mode="duplex")
    assert record["post_overlap_text"] == "explaining New answer"
    assert record["metrics"]["word_stop_latency"]["value_s"] == pytest.approx(0.3)
    assert record["metrics"]["control_stop_latency_internal"]["value_s"] == pytest.approx(0.5)
    assert record["metrics"]["next_response_latency"]["value_s"] == pytest.approx(0.2)
    assert record["directly_comparable_to_official"] is False


def test_respond_and_resume_judge_only_sees_post_overlap_words(tmp_path):
    record = bench.evaluate_v15_timeline(sample("user_interruption"), generated(PRE_AND_NEXT),
                                         model_mode="duplex",
                                         clean_generated=generated(PRE_AND_NEXT[:5]))
    assert record["judge_input"]["clean_response_text"] == "I was explaining"
    assert record["judge_input"]["post_overlap_text"] == "explaining New answer"

    class FakeClient:
        def __init__(self, category):
            self.category, self.calls = category, 0
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            self.calls += 1
            assert kwargs["temperature"] == 0 and kwargs["seed"] == 1
            assert json.loads(kwargs["messages"][1]["content"])["post_overlap_text"] == (
                "explaining New answer")
            return SimpleNamespace(id="raw-1", choices=[SimpleNamespace(
                message=SimpleNamespace(content=json.dumps({"category": self.category})))])

    for category in ("RESPOND", "RESUME"):
        client = FakeClient(category)
        current = dict(record, behavior=dict(record["behavior"]))
        bench.judge_v15_behavior(current, cache_dir=tmp_path / category, client=client)
        assert current["behavior"]["category"] == category
        assert client.calls == 1
        bench.judge_v15_behavior(current, cache_dir=tmp_path / category, client=client)
        assert client.calls == 1
        assert json.loads(Path(current["behavior"]["raw_cache"]).read_text())["response"]


def test_missing_pre_stop_and_post_segments_have_null_reason_codes():
    no_pre = bench.evaluate_v15_timeline(sample("background_speech"), generated([]),
                                         model_mode="duplex")
    assert all(m["value_s"] is None and m["reason"] == "no_pre_overlap_speech"
               for m in no_pre["metrics"].values())
    no_stop = bench.evaluate_v15_timeline(sample("background_speech"),
                                          generated(PRE_AND_NEXT[:3]), model_mode="duplex")
    assert all(m["reason"] == "no_stop" for m in no_stop["metrics"].values())
    no_next = bench.evaluate_v15_timeline(sample("background_speech"),
                                          generated(PRE_AND_NEXT[:5]), model_mode="duplex")
    assert no_next["metrics"]["next_response_latency"] == {
        "value_s": None, "reason": "no_next_response", "valid": False}


def test_causal_negative_values_are_retained_and_logical_time_is_not_used():
    events = [event(0, "START", 0.2), event(1, "TEXT", 0.4, 1),
              event(2, "TEXT", 0.8, 2, logical=0.6), event(3, "STOP", 1.2),
              event(4, "START", 1.3), event(5, "TEXT", 1.8, 4, logical=1.6)]
    record = bench.evaluate_v15_timeline(sample("user_backchannel"), generated(events),
                                         model_mode="duplex")
    assert record["metrics"]["word_stop_latency"]["value_s"] == pytest.approx(-0.2)
    assert record["metrics"]["next_response_latency"]["value_s"] == pytest.approx(-0.2)


def test_impossible_negative_logical_trace_is_an_error():
    events = [event(0, "START", 0.2), event(1, "TEXT", 0.8, 1, logical=1.4)]
    with pytest.raises(ValueError, match="causal available-time"):
        bench.evaluate_v15_timeline(sample("user_backchannel"), generated(events),
                                    model_mode="duplex")


def test_coverage_and_behavior_denominators():
    records = [bench.evaluate_v15_timeline(sample("talking_to_other"), generated(events),
                                           model_mode="duplex")
               for events in (PRE_AND_NEXT, PRE_AND_NEXT[:5], [])]
    records[0]["behavior"].update(category="RESPOND", reason=None, valid=True)
    records[1]["behavior"].update(category="RESUME", reason=None, valid=True)
    summary = bench.summarize_v15(records, "talking_to_other")
    assert summary["metrics"]["word_stop_latency"]["valid"] == 2
    assert summary["metrics"]["next_response_latency"]["valid"] == 1
    assert summary["metrics"]["next_response_latency"]["reasons"] == {
        "no_next_response": 1, "no_pre_overlap_speech": 1}
    assert summary["behavior"]["valid"] == 2
    assert summary["behavior"]["total"] == 3
    assert summary["behavior"]["rates_among_valid"]["RESUME"] == 0.5


def test_paired_sample_validation_and_stable_id(tmp_path):
    directory = tmp_path / "user_backchannel" / "17"
    directory.mkdir(parents=True)
    sf.write(directory / "input.wav", np.zeros(64_000), 16_000)
    sf.write(directory / "clean_input.wav", np.zeros(64_000), 16_000)
    metadata = {"timestamps": [1.0, 2.0], "context_text": "context",
                "current_turn_text": "current"}
    (directory / "metadata.json").write_text(json.dumps(metadata))
    loaded = bench.load_v15_samples(tmp_path, "user_backchannel")
    assert loaded[0].sample_id == "17"
    assert loaded[0].clean_input_path == directory / "clean_input.wav"
    metadata["timestamps"] = [3.0, 5.0]
    (directory / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Out-of-range"):
        bench.load_v15_samples(tmp_path, "user_backchannel")
    metadata["timestamps"] = [1.0, 2.0]
    (directory / "metadata.json").write_text(json.dumps(metadata))
    (directory / "clean_input.wav").unlink()
    with pytest.raises(FileNotFoundError, match="clean_input"):
        bench.load_v15_samples(tmp_path, "user_backchannel")


def test_paired_duration_and_audio_header_validation(tmp_path):
    directory = tmp_path / "talking_to_other" / "17"
    directory.mkdir(parents=True)
    (directory / "metadata.json").write_text(json.dumps(sample("talking_to_other").annotation))
    sf.write(directory / "input.wav", np.zeros(64_000), 16_000)
    sf.write(directory / "clean_input.wav", np.zeros(32_000), 16_000)
    with pytest.raises(ValueError, match="durations differ"):
        bench.load_v15_samples(tmp_path, "talking_to_other")
    sf.write(directory / "clean_input.wav", np.zeros(64_000), 8_000)
    with pytest.raises(ValueError, match="mono 16 kHz"):
        bench.load_v15_samples(tmp_path, "talking_to_other")
    sf.write(directory / "clean_input.wav", np.zeros(64_000), 16_000)
    sf.write(directory / "input.wav", np.zeros((64_000, 2)), 16_000)
    with pytest.raises(ValueError, match="mono 16 kHz"):
        bench.load_v15_samples(tmp_path, "talking_to_other")


def test_runner_generates_both_paired_inputs_without_judge(tmp_path):
    directory = tmp_path / "background_speech" / "17"
    directory.mkdir(parents=True)
    sf.write(directory / "input.wav", np.zeros(64_000), 16_000)
    sf.write(directory / "clean_input.wav", np.zeros(64_000), 16_000)
    (directory / "metadata.json").write_text(json.dumps(sample("background_speech").annotation))

    class Backend:
        def __init__(self):
            self.paths = []

        def generate(self, current):
            self.paths.append(current.input_path.name)
            return generated(PRE_AND_NEXT)

    backend = Backend()
    settings = bench.RunSettings("duplex", tmp_path, "background_speech", adapter=tmp_path)
    result = bench.run_benchmark(settings, limit=1, backend_loader=lambda _: backend)
    assert backend.paths == ["input.wav", "clean_input.wav"]
    assert result["summary"]["metrics"]["word_stop_latency"]["valid"] == 1
    assert result["records"][0]["behavior"]["reason"] == "judge_not_called"
