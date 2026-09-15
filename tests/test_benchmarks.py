import json
from pathlib import Path

import pytest

from benchmarks import full_duplex_bench as fdb
from benchmarks import voicebench
from duplex.streaming import EventTraceRecord
from duplex.metrics import FULL_DUPLEX_BENCH_VARIANT, OFFICIAL_SPEECH_OUTPUT_SCORE


class FakeBackend:
    def __init__(self):
        self.calls = []

    def generate_audio(self, audio, *, example_id):
        self.calls.append(("audio", example_id))
        return "heard response"

    def generate_text(self, prompt, *, example_id):
        self.calls.append(("text", example_id))
        return f"text response: {prompt}"


DATA = [
    {"prompt": "First prompt", "reference": "first", "key": 10,
     "metadata": {"group": "a"},
     "audio": {"array": [0.0, 0.1], "sampling_rate": 16_000}},
    {"prompt": "Second prompt", "reference": "second", "key": 11,
     "metadata": {"group": "b"},
     "audio": {"array": [0.2, 0.3], "sampling_rate": 16_000}},
]


def _adapter(tmp_path: Path) -> Path:
    path = tmp_path / "adapter"
    path.mkdir(exist_ok=True)
    (path / "adapter_model.safetensors").write_bytes(b"fake weights")
    (path / "adapter_config.json").write_text('{"r": 1}\n', encoding="utf-8")
    return path


def _settings(tmp_path: Path, *, mode="base", modality="audio", **changes):
    values = dict(model_mode=mode, data="sd-qa", split="usa", modality=modality,
                  seed=23, system_prompt="Shared prompt", max_new_tokens=19,
                  max_silent_chunks=2,
                  adapter=_adapter(tmp_path) if mode == "duplex" else None)
    values.update(changes)
    return voicebench.RunSettings(**values)


def _rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _run(tmp_path, settings, output, *, limit=2, backend=None, start_index=0):
    backend = backend or FakeBackend()
    return voicebench.run_benchmark(
        settings, output=output, limit=limit, start_index=start_index,
        dataset_loader=lambda current: DATA,
        backend_loader=lambda current: backend)


def test_full_duplex_bench_is_labeled_as_adaptation():
    assert FULL_DUPLEX_BENCH_VARIANT == "text-timeline adaptation"
    assert OFFICIAL_SPEECH_OUTPUT_SCORE is False


def test_jsonl_schema_preserves_non_audio_fields_and_manifest(tmp_path):
    output = tmp_path / "base.jsonl"
    result = _run(tmp_path, _settings(tmp_path), output, limit=1)
    row = _rows(output)[0]
    assert {key: row[key] for key in DATA[0] if key != "audio"} == {
        key: value for key, value in DATA[0].items() if key != "audio"}
    assert "audio" not in row
    assert row["response"] == "heard response"
    assert row["voicebench_id"].startswith("sd-qa:usa:0:")
    manifest = row["run_manifest"]
    assert manifest["model"]["base_revision"] == voicebench.BASE_REVISION
    assert manifest["model"]["adapter"] is None
    assert manifest["upstream"]["revision"] == voicebench.VOICEBENCH_REVISION
    assert manifest["modality"] == "audio"
    assert manifest["seed"] == 23
    assert manifest["system_prompt"] == "Shared prompt"
    assert manifest["decoding"]["max_new_tokens"] == 19
    assert manifest["timing_mode"] == "official_half_duplex"
    assert result["written"] == 1


def test_stable_ids_match_modes_modalities_and_audio_changes(tmp_path):
    copied = dict(DATA[1], audio={"array": [9.0], "sampling_rate": 16_000})
    assert voicebench.stable_example_id(
        DATA[1], data="sd-qa", split="usa", index=1
    ) == voicebench.stable_example_id(
        copied, data="sd-qa", split="usa", index=1)
    ids = []
    for mode, modality in (("base", "audio"), ("duplex", "text")):
        output = tmp_path / f"{mode}-{modality}.jsonl"
        _run(tmp_path, _settings(tmp_path, mode=mode, modality=modality), output)
        ids.append([row["voicebench_id"] for row in _rows(output)])
    assert ids[0] == ids[1]


def test_resume_without_duplicate_output_or_model_reload(tmp_path):
    output = tmp_path / "resume.jsonl"
    settings = _settings(tmp_path)
    first = _run(tmp_path, settings, output)
    second = voicebench.run_benchmark(
        settings, output=output, limit=2,
        dataset_loader=lambda current: DATA,
        backend_loader=lambda current: pytest.fail("model loaded"))
    assert first["written"] == 2
    assert second["written"] == 0 and second["resumed"] is True
    assert len(_rows(output)) == 2
    assert len({row["voicebench_id"] for row in _rows(output)}) == 2


def test_start_index_uses_absolute_stable_id(tmp_path):
    output = tmp_path / "slice.jsonl"
    _run(tmp_path, _settings(tmp_path), output, limit=1, start_index=1)
    assert _rows(output)[0]["voicebench_id"] == voicebench.stable_example_id(
        DATA[1], data="sd-qa", split="usa", index=1)


def _paired_outputs(tmp_path):
    outputs = []
    for mode in ("base", "duplex"):
        output = tmp_path / f"{mode}.jsonl"
        _run(tmp_path, _settings(tmp_path, mode=mode), output)
        outputs.append(output)
    return outputs


def test_summarizer_pairs_settings_and_prepares_commands(tmp_path):
    base, duplex = _paired_outputs(tmp_path)
    summary = voicebench.summarize_pair(base, duplex, upstream_dir=tmp_path / "VoiceBench")
    assert summary["pair_count"] == 2
    assert summary["settings"]["seed"] == 23
    assert summary["settings"]["decoding"]["max_new_tokens"] == 19
    assert "api_judge.py" in summary["commands"]["base"][0]
    assert "--evaluator qa" in summary["commands"]["duplex"][-1]
    assert "no external judge was called" in summary["warning"]


def test_summarizer_rejects_mismatched_manifests(tmp_path):
    base, duplex = _paired_outputs(tmp_path)
    rows = _rows(duplex)
    for row in rows:
        row["run_manifest"]["system_prompt"] = "different"
    duplex.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="system_prompt"):
        voicebench.summarize_pair(base, duplex, upstream_dir=tmp_path)


def test_summarizer_rejects_missing_and_duplicate_pairs(tmp_path):
    base, duplex = _paired_outputs(tmp_path)
    rows = _rows(duplex)
    duplex.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing paired"):
        voicebench.summarize_pair(base, duplex, upstream_dir=tmp_path)
    duplex.write_text(json.dumps(rows[0]) + "\n" + json.dumps(rows[0]) + "\n",
                      encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        voicebench.summarize_pair(base, duplex, upstream_dir=tmp_path)


def test_base_and_duplex_backend_routing(monkeypatch, tmp_path):
    calls = []

    class Base:
        def __init__(self, settings):
            calls.append("base")

    class Duplex:
        def __init__(self, settings):
            calls.append("duplex")

    monkeypatch.setattr(voicebench, "BaseQwenBackend", Base)
    monkeypatch.setattr(voicebench, "DuplexQwenBackend", Duplex)
    assert isinstance(voicebench.load_backend(_settings(tmp_path)), Base)
    assert isinstance(voicebench.load_backend(_settings(tmp_path, mode="duplex")), Duplex)
    assert calls == ["base", "duplex"]


def test_run_routes_selected_modality(tmp_path):
    for modality in ("audio", "text"):
        backend = FakeBackend()
        _run(tmp_path, _settings(tmp_path, modality=modality),
             tmp_path / f"{modality}.jsonl", limit=1, backend=backend)
        assert backend.calls[0][0] == modality


class PieceTokenizer:
    pieces = {1: "Okay", 2: " yes", 3: " I", 4: " understand", 5: " now", 6: "!"}

    def decode(self, token_ids, **kwargs):
        return "".join(self.pieces[token_id] for token_id in token_ids)


def _event(index, kind, token_id=0, *, logical=None, available=None):
    logical = index * 0.1 if logical is None else logical
    available = logical if available is None else available
    return EventTraceRecord(
        sample_id="sample", chunk_index=0, frame_index=index,
        audio_time_s=logical, available_time_s=available, compute_ms=0.0,
        event_id=token_id, event_type=kind, decoded_delta="",
        state_before="inactive" if kind == "START" else "active",
        state_after="inactive" if kind == "STOP" else "active",
        grammar_mask_changed_raw_argmax=False, raw_argmax_id=token_id,
        chunk_end_time_s=available, silent_tail=False)


def _sample(task="pause_handling", *, duration=4.0, annotation=None, tmp_path=None):
    directory = tmp_path or Path("/external/full-duplex-bench/1")
    return fdb.BenchmarkSample(
        sample_id="1", task=task, directory=directory,
        input_path=directory / "input.wav", input_duration_s=duration,
        annotation=annotation)


def _timeline(events):
    return fdb.GeneratedTimeline(PieceTokenizer(), tuple(events), 4.0, "STOP")


def _evaluate(events, task="pause_handling", *, duration=4.0, annotation=None, gt=None):
    return fdb.evaluate_timeline(
        _sample(task, duration=duration, annotation=annotation), _timeline(events),
        model_mode="duplex", ground_truth_distribution=gt)


def test_full_duplex_v1_pin_and_every_schema_label():
    assert fdb.UPSTREAM_REVISION == "3e799c45a045256f47d5f1c9cda90157e2d2ec9e"
    record = _evaluate([])
    summary = fdb.summarize([record], "pause_handling")
    assert record["label"] == summary["label"] == (
        "Full-Duplex-Bench v1.0 text-timeline adaptation")
    assert record["official_speech_output_score"] is False


def test_silence_and_missing_response_are_not_takeover_and_latency_is_null():
    record = _evaluate([], "smooth_turn_taking", annotation={"timestamp": [3.0, 3.2]})
    assert record["decoded_text_exact"] == ""
    assert record["timeline"]["first_word"] is None
    assert record["metrics"]["takeover"] is False
    assert record["metrics"]["first_word_latency_s"] is None
    aggregate = fdb.summarize([record], "smooth_turn_taking")
    assert aggregate["first_word_latency_s"] == {
        "value": None, "coverage": {"available": 0, "total": 1, "rate": 0.0}}


def test_short_two_word_subsecond_response_is_backchannel():
    events = [_event(0, "START", available=1.0), _event(1, "TEXT", 1, available=1.1),
              _event(2, "TEXT", 2, available=1.2), _event(3, "STOP", available=1.8)]
    record = _evaluate(events, "backchannel", gt=[0, 0, 0, 0, 0, 1] + [0] * 15)
    segment = record["timeline"]["response_segments"][0]
    assert segment["classification"] == "backchannel"
    assert segment["word_count"] == 2 and segment["duration_s"] == pytest.approx(0.8)
    assert record["metrics"]["takeover"] is False
    assert record["metrics"]["backchannel_frequency_per_second"] == 0.25


@pytest.mark.parametrize(
    "events",
    [
        [_event(0, "START", available=1.0), _event(1, "TEXT", 1, available=1.1),
         _event(2, "TEXT", 2, available=1.2), _event(3, "TEXT", 3, available=1.3),
         _event(4, "STOP", available=1.8)],
        [_event(0, "START", available=1.0), _event(1, "TEXT", 1, available=1.2),
         _event(2, "STOP", available=2.0)],
    ],
)
def test_three_words_or_one_second_is_takeover(events):
    record = _evaluate(events)
    assert record["metrics"]["takeover"] is True
    assert record["timeline"]["response_segments"][0]["classification"] == "takeover"


def test_multiple_segments_and_exact_decoded_text_are_preserved():
    events = [
        _event(0, "START", available=0.5), _event(1, "TEXT", 1, available=0.6),
        _event(2, "STOP", available=0.8), _event(3, "START", available=2.0),
        _event(4, "TEXT", 3, available=2.1), _event(5, "TEXT", 4, available=2.2),
        _event(6, "TEXT", 5, available=2.3), _event(7, "TEXT", 6, available=2.4),
        _event(8, "STOP", available=3.2),
    ]
    record = _evaluate(events)
    assert record["decoded_text_exact"] == "Okay I understand now!"
    assert [row["classification"] for row in record["timeline"]["response_segments"]] == [
        "backchannel", "takeover"]
    assert record["timeline"]["response_segments"][1]["exact_text"] == " I understand now!"
    assert len(record["timeline"]["start_times"]) == len(record["timeline"]["stop_times"]) == 2


def test_word_timing_uses_last_contributing_token_and_keeps_first_last_separate():
    events = [_event(0, "START", available=2.0),
              _event(1, "TEXT", 1, logical=1.7, available=2.1),
              _event(2, "TEXT", 2, logical=1.8, available=2.4),
              _event(3, "TEXT", 3, logical=1.9, available=2.5),
              _event(4, "STOP", logical=2.0, available=3.2)]
    record = _evaluate(events)
    words = record["timeline"]["word_chunks"]
    assert [word["text"] for word in words] == ["Okay", "yes", "I"]
    assert words[1]["available_time_s"] == 2.4
    assert record["timeline"]["first_word"] == words[0]
    assert record["timeline"]["last_word"] == words[-1]
    assert record["timeline"]["start_times"][0]["available_time_s"] == 2.0
    assert record["timeline"]["stop_times"][0]["available_time_s"] == 3.2


def test_negative_logical_latency_retains_nonnegative_causal_latency():
    events = [_event(0, "START", logical=2.7, available=4.0),
              _event(1, "TEXT", 1, logical=2.8, available=4.0),
              _event(2, "TEXT", 2, logical=2.9, available=4.0),
              _event(3, "TEXT", 3, logical=3.0, available=4.0),
              _event(4, "STOP", logical=3.1, available=4.0)]
    record = _evaluate(events, "smooth_turn_taking", annotation={"timestamp": [3.0, 3.1]})
    assert record["metrics"]["logical_first_word_latency_s"] == pytest.approx(-0.2)
    assert record["metrics"]["first_word_latency_s"] == pytest.approx(1.0)
    assert all(word["available_time_s"] >= 0 for word in record["timeline"]["word_chunks"])


def test_200ms_distribution_uses_inclusive_response_interval_bins_and_jsd():
    config = fdb.BenchmarkConfig()
    distribution = fdb.response_interval_distribution([[0.2, 0.4]], 0.8, config)
    assert len(distribution) == 5
    assert distribution[1] == pytest.approx(distribution[2])
    assert distribution[1] > distribution[0]
    assert fdb.jensen_shannon_distance(distribution, distribution) == pytest.approx(0.0)


def test_interruption_latency_and_judge_ready_record_do_not_call_judge():
    events = [_event(0, "START", logical=2.9, available=3.2),
              _event(1, "TEXT", 3, logical=3.0, available=3.2),
              _event(2, "TEXT", 4, logical=3.1, available=3.3),
              _event(3, "TEXT", 5, logical=3.2, available=3.4),
              _event(4, "STOP", logical=3.3, available=4.5)]
    annotation = {"timestamp": [2.0, 3.0], "context": "Earlier context",
                  "interrupt": "Please change topics"}
    record = _evaluate(events, "user_interruption", annotation=annotation)
    assert record["metrics"]["takeover"] is True
    assert record["metrics"]["first_word_latency_s"] == pytest.approx(0.2)
    judge = record["judge_ready_relevance"]
    assert judge["assistant_response"] == "I understand now"
    assert judge["rating_0_to_5"] is None and judge["status"] == "not_called"
    aggregate = fdb.summarize([record], "user_interruption")
    assert aggregate["relevance_rating_0_to_5"]["coverage"]["available"] == 0


def test_interruption_ignores_pre_interruption_words_for_latency_and_relevance():
    events = [_event(0, "START", available=2.0),
              _event(1, "TEXT", 1, logical=2.1, available=2.1),
              _event(2, "TEXT", 2, logical=2.2, available=2.2),
              _event(3, "TEXT", 3, logical=3.0, available=3.4),
              _event(4, "STOP", available=4.2)]
    annotation = {"timestamp": [2.5, 3.0], "context": "Context", "interrupt": "New request"}
    record = _evaluate(events, "user_interruption", annotation=annotation)
    assert record["metrics"]["takeover"] is True
    assert record["metrics"]["first_word_latency_s"] == pytest.approx(0.4)
    assert record["judge_ready_relevance"]["assistant_response"] == "I"


def test_interruption_pre_only_takeover_does_not_count_as_response():
    events = [_event(0, "START", available=1.0),
              _event(1, "TEXT", 1, available=1.1),
              _event(2, "TEXT", 2, available=1.2),
              _event(3, "TEXT", 3, available=1.3),
              _event(4, "STOP", available=1.8)]
    annotation = {"timestamp": [2.5, 3.0], "context": "Context", "interrupt": "New request"}
    record = _evaluate(events, "user_interruption", annotation=annotation)
    assert record["metrics"]["takeover"] is False
    assert record["metrics"]["first_word_latency_s"] is None
    assert record["judge_ready_relevance"]["assistant_response"] == ""


def test_full_duplex_runner_routes_base_and_duplex_modes(tmp_path):
    class FakeTimelineBackend:
        def generate(self, sample):
            return _timeline([])

    sample = _sample(tmp_path=tmp_path)
    for mode in ("base", "duplex"):
        adapter = tmp_path if mode == "duplex" else None
        settings = fdb.RunSettings(mode, tmp_path, "pause_handling", adapter=adapter)
        seen = []
        result = fdb.run_benchmark(
            settings, limit=1, sample_loader=lambda data, task: [sample],
            backend_loader=lambda current: seen.append(current.model_mode) or FakeTimelineBackend())
        assert seen == [mode]
        assert result["model_mode"] == mode
        assert result["records"][0]["label"] == fdb.ADAPTATION_NAME
