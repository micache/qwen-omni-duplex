"""Compare saved text proxy metrics with the unmodified pinned v1 scorers.

Only the audio loader/VAD and external judge are stubbed. VAD returns native
control intervals, not synthesized audio. Dummy ratings test judge plumbing
only and are never reported as measured relevance scores.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace, ModuleType

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.full_duplex_v1 import SUBSETS, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/full-duplex-bench-v1-response-gain003"))
    parser.add_argument("--upstream", type=Path, default=Path("outputs/full-duplex-bench-upstream"))
    args = parser.parse_args()
    summary = json.loads((args.output / "summary.json").read_text())
    control_ids = yaml.safe_load((Path(summary["manifest"]["adapter"]["path"]) / "duplex_config.yaml").read_text())["control_ids"]
    coverage = {"samples": 0, "frames": 0, "closed_turns": 0, "open_turns": 0}
    for path in sorted(args.output.glob("*/*/result.json")):
        result = json.loads(path.read_text())
        events = result["generation"]["event_ids"]
        assert len(events) == ((result["samples"] + 31999) // 32000) * 50
        boundaries, active = [], None
        for frame, event in enumerate(events):
            if event == control_ids["START"]:
                assert active is None, path
                active = frame
            elif event == control_ids["STOP"]:
                assert active is not None, path
                boundaries.append((active, frame))
                active = None
            elif event != control_ids["IDLE"]:
                assert active is not None, path
        if active is not None:
            boundaries.append((active, None))
        assert boundaries == [(s["start_frame"], s["stop_frame"]) for s in result["segments"]]
        for filename, delay in (("output.json", 0), ("causal.json", 2)):
            output = json.loads(path.with_name(filename).read_text())
            previous_end = output["crop_start_s"]
            for word in output["chunks"]:
                start, end = word["timestamp"]
                assert np.isfinite([start, end]).all() and previous_end <= start + 1e-12 and start < end <= result["duration_s"], path
                previous_end = end
            assert output["text"] == " ".join(w["text"] for w in output["chunks"])
            for turn in output["turns"]:
                start, stop = turn["start_frame"], turn["stop_frame"]
                assert (start, stop) in boundaries
                assert turn["timestamp"] == [max(output["crop_start_s"], start / 25 + delay),
                    min(result["duration_s"], (stop if stop is not None else len(events)) / 25 + delay)]
                assert turn["right_censored"] == (stop is None or stop / 25 + delay > result["duration_s"])
        coverage["samples"] += 1
        coverage["frames"] += len(events)
        coverage["closed_turns"] += sum(stop is not None for _, stop in boundaries)
        coverage["open_turns"] += sum(stop is None for _, stop in boundaries)
    assert coverage["samples"] == summary["sample_count"]
    if not summary["manifest"]["smoke"]:
        assert summary["complete"] and coverage["samples"] == 727
    evaluation = (args.upstream / "v1_v1.5/evaluation").resolve()
    current = {}
    vad = ModuleType("silero_vad")
    vad.load_silero_vad = lambda: None
    vad.get_speech_timestamps = lambda *a, **kw: [dict(start=s, end=e) for s, e in current["output"]["response_intervals"]]
    audio = ModuleType("torchaudio")
    def load(path):
        current["output"] = json.loads(Path(path).with_name("output.json").read_text())
        # No waveform exists. Pass the exact original-clock duration through
        # the scorer's shape/sr interface, without quantizing it to 16 kHz.
        return SimpleNamespace(shape=(1, current["output"]["duration_s"] * 16000)), 16000
    audio.load = load
    sys.modules["silero_vad"], sys.modules["torchaudio"] = vad, audio
    judge = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw:
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=
            "Analysis: plumbing fixture only.\nI would rate the AI's response as 3"))]))))
    report = {"sample_count": summary["sample_count"], "coverage": coverage, "clocks": {},
              "vad": "native START/STOP intervals and exact original-clock duration substituted; no waveform generated",
              "judge": "dummy response for scorer parity only; relevance remains unmeasured"}
    previous_cwd = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="full-duplex-parity-") as temporary:
        temporary = Path(temporary)
        try:
            os.chdir(evaluation)
            for clock, filename in (("nominal", "output.json"), ("causal", "causal.json")):
                checks = {}
                for subset, (task, _, annotation) in SUBSETS.items():
                    source = (previous_cwd / args.output / subset)
                    root = temporary / clock / subset
                    root.mkdir(parents=True)
                    for directory in sorted(source.iterdir()):
                        target = root / directory.name
                        target.mkdir()
                        shutil.copyfile(directory / filename, target / "output.json")
                        if annotation:
                            shutil.copyfile(directory / annotation, target / annotation)
                        if task == "backchannel":
                            (target / "output.wav").touch()  # loader stub; never interpreted as audio
                    name = "eval_" + ("backchannel" if task == "backchannel" else task)
                    spec = importlib.util.spec_from_file_location(name, evaluation / f"{name}.py")
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    function = getattr(module, name)
                    captured = {}
                    def trace(frame, event, arg):
                        if frame.f_code is function.__code__ and event in {"return", "exception"}:
                            captured.update(frame.f_locals)
                        return trace
                    try:
                        sys.settrace(trace)
                        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                            try:
                                function(str(root), judge) if task == "user_interruption" else function(str(root))
                            except ZeroDivisionError:
                                # Released scorer divides by zero when no turn qualifies.
                                assert captured.get("take_turn_list") and not any(captured["take_turn_list"])
                    finally:
                        sys.settrace(None)
                    expected = summary[clock][subset]
                    actual = {"tor": float(np.mean(captured["TOR_list" if task == "backchannel" else "take_turn_list"]))}
                    if task == "backchannel":
                        actual.update(frequency_per_second=float(np.mean(captured["freq_list"])),
                                      jsd=float(np.mean(captured["jsd_list"])))
                    if task in {"smooth_turn_taking", "user_interruption"}:
                        latencies = captured["latency_list"]
                        actual.update(latency_s=float(np.mean(latencies)) if latencies else None,
                                      latency_samples=len(latencies))
                    count = len(captured["TOR_list" if task == "backchannel" else "take_turn_list"])
                    assert count == expected["samples"], (subset, count, expected)
                    for key, value in actual.items():
                        assert value is None and expected[key] is None or value is not None and np.isclose(
                            value, expected[key], rtol=0, atol=1e-12), (clock, subset, key, value, expected[key])
                    checks[subset] = {"samples": count, "matched": actual}
                report["clocks"][clock] = checks
        finally:
            os.chdir(previous_cwd)
    report["passed"] = True
    write_json(args.output / "upstream-parity.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
