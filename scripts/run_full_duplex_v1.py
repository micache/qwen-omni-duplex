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
    download_data, partition_records, prepare_data, read_audio, score_output, summarize, timed_output, write_json)


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
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--initialize-only", action="store_true")
    parser.add_argument("--eager", action="store_true", help="Disable CUDA graphs for reference checks")
    args = parser.parse_args()
    if args.batch_size < 1 or args.frame_budget < 1:
        parser.error("batch size and frame budget must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("shard index must be within num-shards")
    if args.score_only and args.num_shards != 1:
        parser.error("Aggregate scoring must use num-shards=1")
    if (args.upstream / ".git").exists():
        import subprocess
        revision = subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip()
        if revision != REVISION:
            raise ValueError("Upstream checkout does not match the pinned v1 revision")
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
    manifest = {"schema": "full_duplex_bench_v1_control_timing_v2", "upstream_revision": REVISION,
        "adapter": _adapter_identity(args.adapter), "data_sha256": digest(rows),
        "training_config_sha256": hashlib.sha256((args.adapter / "training_config.yaml").read_bytes()).hexdigest(),
        "empty_context": True, "sample": False, "seed": 17, "frame_rate_hz": 25,
        "chunk_seconds": 2, "causal_playout_delay_s": CHUNK_DELAY_S,
        "horizon": "original recording duration; no additional silence", "speech_output": False,
        "special_tokens_skipped": True, "judge": "not called", "smoke": args.smoke,
        "turn_boundaries": "START/25 and STOP/25", "primary_clock": "nominal",
        "inference": {"batch_size": args.batch_size, "frame_budget": args.frame_budget,
                      "cuda_graph": not args.eager}}
    if args.smoke:
        rows = [next(r for r in rows if r["subset"] == subset) for subset in SUBSETS]
        args.output = args.output / "smoke"
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Existing run has different weights, data or protocol")
    if not manifest_path.exists():
        write_json(manifest_path, manifest)
    if args.initialize_only:
        return
    if not args.score_only:
        rows = partition_records(rows, args.num_shards)[args.shard_index]
    pending = [row for row in rows if not (args.output / row["id"] / "result.json").exists()]
    tokenizer, controls = None, None
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
        tokenizer, controls = processor.tokenizer, model.control_tokens
        # Every input frame, including frames after earlier STOPs, is processed.
        engine = BatchedDuplex(model, processor, max_new_tokens=100000,
            max_silent_chunks=0, min_silent_chunks=0)
        if args.smoke and not args.eager:
            precision = engine.audit_graph([read_audio(row) for row in pending[:2]])
            write_json(args.output / f"graph_precision-{args.shard_index}.json", precision)
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
            generated = engine.generate([read_audio(row) for row in batch], use_graph=not args.eager)
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
                write_json(directory / "output.json", clocks["nominal"])
                write_json(directory / "causal.json", clocks["causal"])
                if row["annotation"] is not None:
                    write_json(directory / SUBSETS[row["subset"]][2], row["annotation"])
                write_json(directory / "result.json", result)
            completed += len(batch)
            print(json.dumps({"completed_this_run": completed, "remaining": len(pending),
                "elapsed_s": round(time.monotonic() - started, 1),
                "batch": len(batch), "gpu_peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}), flush=True)
    records = []
    if tokenizer is None:
        from transformers import AutoTokenizer
        from types import SimpleNamespace
        import yaml
        tokenizer = AutoTokenizer.from_pretrained(args.adapter, local_files_only=True)
        control_ids = yaml.safe_load((args.adapter / "duplex_config.yaml").read_text())["control_ids"]
        controls = SimpleNamespace(idle=control_ids["IDLE"], start=control_ids["START"], stop=control_ids["STOP"])
    for row in rows:
        path = args.output / row["id"] / "result.json"
        if not path.exists():
            raise ValueError(f"Missing result: {row['id']}")
        result = json.loads(path.read_text())
        if result["manifest_sha256"] != digest(manifest) or result["source_sha256"] != row["source_sha256"]:
            raise ValueError(f"Result identity mismatch: {row['id']}")
        # Recompute every score from persisted output, including silence cases.
        segments = event_segments(result["generation"]["event_ids"], tokenizer, controls)
        if segments != result["segments"]:
            raise ValueError(f"Saved segment mismatch: {row['id']}")
        crop = row["annotation"][0]["timestamp"][1] if row["task"] == "user_interruption" else 0.0
        for clock, filename, delay in (("nominal", "output.json", 0.0),
                                       ("causal", "causal.json", CHUNK_DELAY_S)):
            output = json.loads((path.parent / filename).read_text())
            if output != timed_output(segments, row["duration_s"], delay_s=delay, crop_start=crop):
                raise ValueError(f"Saved timestamp mismatch: {row['id']}, {clock}")
            expected = score_output(row, output, reference.get(row["id"].split("/")[-1]))
            if result["scores"][clock] != expected:
                raise ValueError(f"Saved score mismatch: {row['id']}, {clock}")
        records.append(result)
    summary = {"manifest": manifest, "complete": len(records) == 727 and not args.smoke,
        "sample_count": len(records), "causal": summarize(records, "causal"), "nominal": summarize(records, "nominal")}
    write_json(args.output / ("summary.json" if args.num_shards == 1 else f"summary-shard-{args.shard_index}.json"), summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
