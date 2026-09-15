"""Fresh-process public generate() usability gate for an epoch adapter."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from duplex.training import _load_adapter_weights, build_training_model, load_training_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--cases", nargs="+", required=True)
    args = parser.parse_args()
    config = load_training_config(args.config)
    model, _, _ = build_training_model(config)
    adapter = Path(args.adapter)
    _load_adapter_weights(model, adapter)
    model.eval()
    results = []
    for path in args.cases:
        result = model.generate(path)
        trace_path = adapter / "traces" / f"{Path(path).stem}.jsonl"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text("".join(json.dumps(row.as_dict()) + "\n" for row in result.events))
        counts = Counter(row.event_type for row in result.events)
        results.append({"audio": path, "text": result.text,
                        "trace": str(trace_path),
                        "segments": [{"text": segment["text"], "tokens": len(segment["tokens"])}
                                     for segment in result.segments],
                        "events": dict(counts),
                        "first_word_time_s": result.first_word_time_s,
                        "last_word_time_s": result.last_word_time_s})
    summary = json.loads((Path(config["training"]["output_dir"]) / "data_summary.json").read_text())
    gold = summary["validation_gold"]
    gold_total = max(1, sum(gold.values()))
    limit = max(0.05, 5 * (gold["start"] + gold["stop"]) / gold_total)
    eligible = True
    failures = []
    for row in results:
        counts = row["events"]
        total = max(1, sum(counts.values()))
        if not row["text"].strip() or not row["segments"]:
            eligible = False
            failures.append(f"{row['audio']}: empty decoded text")
        if any(segment["tokens"] == 0 for segment in row["segments"]):
            eligible = False
            failures.append(f"{row['audio']}: empty START/STOP segment")
        if counts.get("START", 0) > 3 or counts.get("STOP", 0) > 3:
            eligible = False
            failures.append(f"{row['audio']}: repeated START/STOP cycling")
        if (counts.get("START", 0) + counts.get("STOP", 0)) / total > limit:
            eligible = False
            failures.append(f"{row['audio']}: excessive control-event fraction")
        if counts.get("IDLE", 0) / total > 0.995:
            eligible = False
            failures.append(f"{row['audio']}: near-all-IDLE distribution")
    report = {"eligible": eligible, "failures": failures, "generations": results,
              "validation_gold": gold, "max_start_stop_fraction": limit}
    (adapter / "generation_gate.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"GENERATION_GATE={'PASS' if eligible else 'FAIL'} adapter={adapter}", flush=True)


if __name__ == "__main__":
    main()
