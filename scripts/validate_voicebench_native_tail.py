"""Check native tail-cache positions, padding removal, EOS and token parity."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import torch
from run_voicebench_suite import load_engine, read_audio
from score_voicebench_suite import read


def main():
    torch.set_num_threads(4)
    root = Path("outputs/voicebench-response-gain003")
    source = read(Path("data/VoiceBench-eval/ifeval/test/manifest.jsonl"))
    obq = read(Path("data/VoiceBench-eval/openbookqa/test/manifest.jsonl"))[0]
    common = read(Path("data/VoiceBench-eval/commoneval/test/manifest.jsonl"))[0]
    cases = [[source[264]], [common], [source[264], obq]]
    engine = load_engine("base", SimpleNamespace())
    engine.max_new_tokens = 320
    report = []
    for rows in cases:
        waves = [read_audio(r) for r in rows]
        engine.use_tail_graph = False
        started = time.monotonic()
        reference = engine.generate(waves)
        reference_s = time.monotonic() - started
        engine.use_tail_graph = True
        started = time.monotonic()
        actual = engine.generate(waves)
        parity = [a["token_ids"] == b["token_ids"] for a, b in zip(reference, actual)]
        result = {"ids": [r["voicebench_id"] for r in rows], "token_parity": parity,
                  "reference_seconds": reference_s, "graph_seconds": time.monotonic() - started,
                  "token_counts": [len(r["token_ids"]) for r in actual],
                  "finish_reasons": [r["finish_reason"] for r in actual],
                  "matching_prefix_tokens": [next((i for i, (x, y) in enumerate(zip(a["token_ids"], b["token_ids"])) if x != y), min(len(a["token_ids"]), len(b["token_ids"]))) for a, b in zip(reference, actual)],
                  "reference_responses": [r["response"] for r in reference],
                  "actual_responses": [r["response"] for r in actual]}
        report.append(result)
        print(json.dumps(result), flush=True)
    (root / "native-tail-graph-validation.json").write_text(json.dumps(report, indent=2))
    if not all(all(r["token_parity"]) for r in report):
        raise RuntimeError("Tail decoder differs from native generation on validation cases")


if __name__ == "__main__":
    main()
