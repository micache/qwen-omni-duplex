"""Prepare, run, and score all four Full-Duplex-Bench v1 tasks locally."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.full_duplex_v1 import (CHUNK_DELAY_S, REVISION, SUBSETS, digest, event_segments,
    download_data, prepare_data, read_audio, score_output, summarize, timed_output, write_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/Full-Duplex-Bench-v1"))
    parser.add_argument("--output", type=Path, default=Path("outputs/full-duplex-bench-v1-response-gain003"))
    parser.add_argument("--adapter", type=Path, default=Path("outputs/instructs2s-response-gain003-one-epoch/final"))
    parser.add_argument("--upstream", type=Path, default=Path("outputs/full-duplex-bench-upstream"))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--download", action="store_true", help="Explicitly download the five v1 data archives")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="One input per subset in a separate output folder")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--frame-budget", type=int, default=120000)
    args = parser.parse_args()
    if args.batch_size < 1 or args.frame_budget < 1:
        parser.error("batch size and frame budget must be positive")
    if args.download:
        download_data(args.data)
    data_manifest = args.data / "prepared/manifest.json"
    rows = prepare_data(args.data) if args.prepare_only or not data_manifest.exists() else json.loads(data_manifest.read_text())["records"]
    if args.prepare_only:
        print(json.dumps({subset: sum(r["subset"] == subset for r in rows) for subset in SUBSETS}), flush=True)
        return
    if len({r["id"] for r in rows}) != 727 or any(
        sum(r["subset"] == subset for r in rows) != expected for subset, (_, expected, _) in SUBSETS.items()):
        raise ValueError("The data manifest must cover all 727 unique v1 samples")
    from benchmarks.voicebench import _adapter_identity
    manifest = {"schema": "full_duplex_bench_v1_text_timing_v1", "upstream_revision": REVISION,
        "adapter": _adapter_identity(args.adapter), "data_sha256": digest(rows),
        "training_config_sha256": hashlib.sha256((args.adapter / "training_config.yaml").read_bytes()).hexdigest(),
        "empty_context": True, "sample": False, "seed": 17, "frame_rate_hz": 25,
        "chunk_seconds": 2, "causal_playout_delay_s": CHUNK_DELAY_S,
        "horizon": "original recording duration; no additional silence", "speech_output": False,
        "special_tokens_skipped": True, "judge": "not called", "smoke": args.smoke}
    if args.smoke:
        rows = [next(r for r in rows if r["subset"] == subset) for subset in SUBSETS]
        args.output = args.output / "smoke"
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Existing run has different weights, data or protocol")
    write_json(manifest_path, manifest)
    pending = [row for row in rows if not (args.output / row["id"] / "result.json").exists()]
    reference = json.loads((args.upstream / "v1_v1.5/evaluation/icc_gt_distribution.json").read_text())
    if pending and not args.score_only:
        import torch
        from benchmarks.voicebench_batch import BatchedDuplex
        from duplex.training import build_training_model, _load_adapter_weights, load_training_config, seed_everything
        torch.set_num_threads(4)
        seed_everything(17)
        config = load_training_config(args.adapter / "training_config.yaml")
        config["model"]["local_files_only"] = True
        model, processor, _ = build_training_model(config)
        _load_adapter_weights(model, args.adapter)
        # Every input frame, including frames after earlier STOPs, is processed.
        engine = BatchedDuplex(model, processor, max_new_tokens=100000,
            max_silent_chunks=0, min_silent_chunks=0)
        if args.smoke:
            precision = engine.audit_graph([read_audio(row) for row in pending[:2]])
            write_json(args.output / "graph_precision.json", precision)
            print(json.dumps({"graph_precision": precision}), flush=True)
        pending.sort(key=lambda row: (row["samples"], row["id"]))
        started, completed = time.monotonic(), 0
        while pending:
            batch = []
            while pending and len(batch) < args.batch_size:
                longest_frames = ((pending[0]["samples"] + 31999) // 32000) * 50
                if batch and (len(batch) + 1) * longest_frames > args.frame_budget:
                    break
                batch.append(pending.pop(0))
            generated = engine.generate([read_audio(row) for row in batch])
            for row, output in zip(batch, generated, strict=True):
                if len(output["event_ids"]) != output["input_frames"]:
                    raise ValueError(f"Input stream truncated: {row['id']}")
                segments = event_segments(output["event_ids"], processor.tokenizer, model.control_tokens)
                crop = row["annotation"][0]["timestamp"][1] if row["task"] == "user_interruption" else 0.0
                clocks = {clock: timed_output(segments, row["duration_s"], delay_s=delay, crop_start=crop)
                    for clock, delay in (("causal", CHUNK_DELAY_S), ("nominal", 0.0))}
                gt = reference.get(row["id"].split("/")[-1]) if row["task"] == "backchannel" else None
                result = {**row, "generation": output, "segments": segments,
                    "scores": {clock: score_output(row, value, gt) for clock, value in clocks.items()},
                    "manifest_sha256": digest(manifest)}
                directory = args.output / row["id"]
                write_json(directory / "output.json", clocks["causal"])
                write_json(directory / "nominal_output.json", clocks["nominal"])
                if row["annotation"] is not None:
                    write_json(directory / SUBSETS[row["subset"]][2], row["annotation"])
                write_json(directory / "result.json", result)
            completed += len(batch)
            print(json.dumps({"completed_this_run": completed, "remaining": len(pending),
                "elapsed_s": round(time.monotonic() - started, 1),
                "batch": len(batch), "gpu_peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}), flush=True)
    records = []
    for row in rows:
        path = args.output / row["id"] / "result.json"
        if not path.exists():
            raise ValueError(f"Missing result: {row['id']}")
        result = json.loads(path.read_text())
        if result["manifest_sha256"] != digest(manifest) or result["source_sha256"] != row["source_sha256"]:
            raise ValueError(f"Result identity mismatch: {row['id']}")
        # Recompute every score from persisted output, including silence cases.
        for clock, filename in (("causal", "output.json"), ("nominal", "nominal_output.json")):
            output = json.loads((path.parent / filename).read_text())
            expected = score_output(row, output, reference.get(row["id"].split("/")[-1]))
            if result["scores"][clock] != expected:
                raise ValueError(f"Saved score mismatch: {row['id']}, {clock}")
        records.append(result)
    summary = {"manifest": manifest, "complete": len(records) == 727 and not args.smoke,
        "sample_count": len(records), "causal": summarize(records), "nominal": summarize(records, "nominal")}
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
