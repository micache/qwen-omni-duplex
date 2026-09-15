import json
from pathlib import Path

import pytest

from benchmarks import voicebench
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
