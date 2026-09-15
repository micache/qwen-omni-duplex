"""Read-only forensic summary of the frozen Session 15 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from dataclasses import replace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from duplex.dataset import load_conversation, read_manifest
from duplex.timeline import _assistant_turns, align_assistant_utterance
from duplex.training import load_training_config
from scripts.run_session11_diagnostic import _infer_user_words


EVIDENCE = Path("outputs/session15-main-111b")


def segments(rows):
    result = []
    active = None
    lexical = 0
    for index, row in enumerate(rows):
        kind = row["event_type"]
        if kind == "START":
            if active is not None:
                result.append({"start": active, "end": index, "lexical_events": lexical, "closed": False})
            active, lexical = index, 0
        elif kind == "TEXT" and active is not None:
            lexical += 1
        elif kind == "STOP" and active is not None:
            result.append({"start": active, "end": index, "length": index - active + 1,
                           "lexical_events": lexical, "closed": True})
            active, lexical = None, 0
    if active is not None:
        result.append({"start": active, "end": len(rows), "length": len(rows) - active,
                       "lexical_events": lexical, "closed": False})
    return result


def crop_loss(config, manifest, tokenizer):
    entries = {entry.conversation_id: entry for entry in read_manifest(
        Path(config["data"]["root"]) / config["data"]["manifest"])}
    counts = Counter()
    examples = []
    for item in manifest["items"]:
        record = load_conversation(entries[item["conversation_id"]],
                                   dataset_root=config["data"]["root"],
                                   speaker_label_map=config["data"]["speaker_label_map"])
        record = replace(record, user_words=_infer_user_words(record, config["diagnostic"]))
        outer_start, outer_end = item["span_start_seconds"], item["span_end_seconds"]
        boundaries = item["chunk_starts_seconds"] + [outer_end]
        for turn in _assistant_turns(record):
            start = min(word.span.start_seconds for word in turn.words)
            end = max(word.span.end_seconds for word in turn.words)
            if end <= outer_start or start >= outer_end:
                continue
            complete = start >= outer_start and end <= outer_end
            if not complete:
                counts["outer_boundary_discarded_turns"] += 1
                continue
            aligned = align_assistant_utterance(
                tokenizer, tuple((word.source_word_index, word.span) for word in turn.words),
                sample_id=item["conversation_id"], source_turn_id=turn.source_turn_id)
            lexical = len(aligned.token_ids)
            counts["outer_retained_turns"] += 1
            counts["outer_retained_lexical_tokens"] += lexical
            internal = any(start < boundary < end for boundary in boundaries[1:-1])
            key = "internal_discarded" if internal else "formerly_retained"
            counts[f"{key}_turns"] += 1
            counts[f"{key}_lexical_tokens"] += lexical
            if not internal:
                chunk_start = max(boundary for boundary in boundaries[:-1] if boundary <= start)
                preceding = any(word.end_seconds <= start and word.end_seconds > chunk_start
                                for word in record.user_words)
                counts["formerly_retained_with_preceding_user_activity"] += int(preceding)
            if internal and len(examples) < 20:
                examples.append({"conversation_id": item["conversation_id"],
                                 "turn_id": turn.source_turn_id, "start": start, "end": end,
                                 "lexical_tokens": lexical})
    counts["average_lexical_tokens_per_formerly_retained_turn"] = (
        counts["formerly_retained_lexical_tokens"] / counts["formerly_retained_turns"]
        if counts["formerly_retained_turns"] else 0)
    counts["average_lexical_tokens_per_outer_retained_turn"] = (
        counts["outer_retained_lexical_tokens"] / counts["outer_retained_turns"]
        if counts["outer_retained_turns"] else 0)
    return {"counts": dict(counts), "cross_boundary_examples": examples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/session15-repair-diagnosis"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    config = load_training_config("configs/session15_main.yaml")
    manifest_path = EVIDENCE / "subset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    checkpoints = []
    for path in sorted((EVIDENCE / "validation").glob("step-*.json"),
                       key=lambda p: int(p.stem.split("-")[1])):
        report = json.loads(path.read_text())
        step = int(path.stem.split("-")[1])
        traces = []
        for trace_path in sorted((EVIDENCE / "traces" / f"step-{step}").glob("*.jsonl")):
            rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
            real = [row for row in rows if not row["silent_tail"]]
            s = segments(real)
            traces.append({"path": str(trace_path), "counts": dict(Counter(row["event_type"] for row in real)),
                           "raw_lexical_argmax_count": sum(row["raw_argmax_id"] not in
                           (151643, 151644, 151645) for row in real),
                           "segments": s, "empty_closed_segments": sum(segment["closed"] and
                           segment["lexical_events"] == 0 for segment in s),
                           "idle_fraction": sum(row["event_type"] == "IDLE" for row in real) / len(real)})
        teacher = report["teacher_forced"]
        free = report["free_streaming"]
        checkpoints.append({"step": step,
                            "target_counts": teacher["events"]["label_counts"],
                            "teacher_predicted_counts": teacher["events"]["prediction_counts"],
                            "free_target_counts": free["events"]["label_counts"],
                            "free_predicted_counts": free["events"]["prediction_counts"],
                            "teacher_lexical_argmax_count": teacher["events"]["prediction_counts"]["TEXT"],
                            "free_lexical_event_count": free["events"]["prediction_counts"]["TEXT"],
                            "interruption_stop_recall": free["interruption_STOP"],
                            "traces": traces})
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"], revision=config["model"]["revision"],
                                              local_files_only=True)
    report = {"schema_version": 1, "source_revision": "5e8ba81cdacd087c52c55d25b48fac246b23a90d",
              "evidence_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in (manifest_path, EVIDENCE / "selection.json",
                                               Path("outputs/session15b-train.log"),
                                               Path("outputs/session15b-fresh-verify.log"))},
              "checkpoints": checkpoints,
              "crop_loss": crop_loss(config, manifest, tokenizer),
              "saved_fresh_verify": json.loads((EVIDENCE / "fresh_verify" / "report.json").read_text())}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"checkpoints": len(checkpoints), "crop_loss": report["crop_loss"]["counts"]}, indent=2))


if __name__ == "__main__":
    main()
