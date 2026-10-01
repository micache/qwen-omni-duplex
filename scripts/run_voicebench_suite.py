"""Run every standard scored VoiceBench subset, with resumable paired outputs."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import soundfile as sf
import torch

from benchmarks.voicebench import BaseQwenBackend, RunSettings, _adapter_identity
from benchmarks.voicebench_batch import BatchedDuplex, BatchedNative
from duplex.training import _load_adapter_weights, build_training_model, load_training_config, seed_everything
from prepare_voicebench import CONFIGS, OFFLINE_CONFIGS, REVISION


def read_audio(record):
    wave, rate = sf.read(record["audio_path"], dtype="float32")
    wave *= record["audio_scale"]
    if rate != 16000 or len(wave) != record["samples"] or hashlib.sha256(wave.tobytes()).hexdigest() != record["pcm_sha256"]:
        raise ValueError("Prepared waveform identity mismatch")
    return wave


def load_engine(mode, args):
    if mode == "duplex":
        config = load_training_config(args.adapter / "training_config.yaml")
        model, processor, _ = build_training_model(config)
        _load_adapter_weights(model, args.adapter)
        return BatchedDuplex(model, processor)
    settings = RunSettings(model_mode="base", data="suite", split="all", modality="audio", max_new_tokens=2048)
    backend = BaseQwenBackend(settings)
    backend.model.visual = None
    return BatchedNative(backend.model, backend.processor)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/VoiceBench-eval"))
    parser.add_argument("--output", type=Path, default=Path("outputs/voicebench-response-gain003"))
    parser.add_argument("--adapter", type=Path, default=Path("outputs/instructs2s-response-gain003-one-epoch/final"))
    parser.add_argument("--mode", choices=("base", "duplex", "both"), default="both")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--frame-budget", type=int, default=100000)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--offline-only", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("shard-index must be in [0, num-shards)")
    configs = OFFLINE_CONFIGS if args.offline_only else CONFIGS
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.cpu_threads)
    seed_everything(17)
    started = time.monotonic()
    modes = ("duplex", "base") if args.mode == "both" else (args.mode,)
    for mode in modes:
        print(f"loading {mode}", flush=True)
        engine = load_engine(mode, args)
        if args.preflight:
            rows = [json.loads(s) for s in (args.data / "commoneval/test/manifest.jsonl").read_text().split("\n") if s.strip()][:4]
            waves = [read_audio(r) for r in rows]
            if mode == "duplex":
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    serial = engine.streamer.run(waves[0], sample_id="parity", context_token_ids=[])
                expected = [r.event_id for r in serial.trace]
                single = engine.generate(waves[:1])[0]
                if single["event_ids"] != expected:
                    raise RuntimeError("Batched duplex B=1 does not reproduce the reference streamer")
                serial_response = engine.processor.tokenizer.decode([r.event_id for r in serial.trace if r.event_type == "TEXT"],
                    skip_special_tokens=True, clean_up_tokenization_spaces=False)
                if single["response"] != serial_response:
                    raise RuntimeError("Decoded special-token removal differs")
                repeat = engine.generate(waves[:1])[0]
                if repeat["event_ids"] != single["event_ids"]:
                    raise RuntimeError("Reused graph/cache did not reproduce identical input")
                compacted = engine.generate([waves[0], waves[3], waves[3], waves[3]])
                if any(r["finish_reason"] != "STOP" for r in compacted):
                    raise RuntimeError("Compacted batch did not complete known short cases")
                batch = engine.generate(waves, audit_frames=(92,))
                other_singles = [single] + [engine.generate([w], audit_frames=(92,))[0] for w in waves[1:]]
                parity = {"reference_event_parity": True, "graph_reuse_parity": True, "reference_events": len(expected),
                    "compacted_long_row_parity": compacted[0]["event_ids"] == single["event_ids"],
                    "batch_event_parity": [a["event_ids"] == b["event_ids"] for a, b in zip(batch, other_singles)],
                    "batch_text_parity": [a["response"] == b["response"] for a, b in zip(batch, other_singles)],
                    "batch": batch, "single": other_singles}
                parity["forced_prefix_precision"] = engine.audit_graph(waves)
            else:
                # Verify that projecting only the last position preserves native generation.
                engine.max_new_tokens = 64
                with_hook = engine.generate(waves[:1])[0]
                engine.head_hook.remove()
                from benchmarks.voicebench import _messages
                text = engine.processor.apply_chat_template(_messages("You are a helpful assistant.",
                    [{"type": "audio", "audio": waves[0]}]), tokenize=False, add_generation_prompt=True)
                original_inputs = engine.processor(text=text, audio=[waves[0]], padding=True,
                    return_tensors="pt", use_audio_in_video=False).to(engine.model.device).to(engine.model.dtype)
                with torch.inference_mode():
                    original_ids = engine.model.generate(**original_inputs, do_sample=False,
                        use_cache=True, max_new_tokens=64)[:, original_inputs.input_ids.shape[1]:][0].cpu().tolist()
                if with_hook["token_ids"] != original_ids:
                    raise RuntimeError("Optimized native B=1 differs from the original audio generation path")
                from benchmarks.voicebench_batch import final_position_head
                engine.head_hook = engine.model.lm_head.register_forward_pre_hook(final_position_head)
                batch = engine.generate(waves)
                singles = [with_hook] + [engine.generate([w])[0] for w in waves[1:]]
                parity = {"original_native_audio_parity": True, "last_projection_parity": True,
                    "batch_token_parity": [a["token_ids"] == b["token_ids"] for a, b in zip(batch, singles)],
                    "batch_text_parity": [a["response"] == b["response"] for a, b in zip(batch, singles)],
                    "batch": batch, "single": singles}
            (args.output / f"preflight-{mode}.json").write_text(json.dumps(parity, ensure_ascii=False, indent=2))
            print(json.dumps({k: v for k, v in parity.items() if k not in ("batch", "single")}), flush=True)
        else:
            manifest = {"model_mode": mode, "base_id": "Qwen/Qwen2.5-Omni-3B",
                "base_revision": "f75b40e3da2003cdd6e1829b1f420ca70797c34e",
                "adapter": _adapter_identity(args.adapter) if mode == "duplex" else None,
                "dataset_id": "hlt-lab/voicebench", "dataset_revision": REVISION,
                "voicebench_revision": "b56154172f2a57a43d29005de7d0471d748d70d1",
                "max_new_tokens": 2048, "do_sample": False, "seed": 17,
                "max_silent_chunks": 42 if mode == "duplex" else None,
                "context": [] if mode == "duplex" else "native audio chat; You are a helpful assistant.",
                "text_dtype": "bfloat16", "audio_dtype": "float32" if mode == "duplex" else "bfloat16",
                "skip_special_tokens": True, "clean_up_tokenization_spaces": False,
                "batch_size_max": args.batch_size, "kv_frame_budget": args.frame_budget,
                "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "lora_merged": False,
                "decoder_execution": "packed_static_cache_cuda_graph_varlen_flash" if mode == "duplex" else "native_generate_final_position_head"}
            model_folder = args.output / mode
            model_folder.mkdir(exist_ok=True)
            manifest_path = model_folder / "manifest.json"
            if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
                raise ValueError("Cannot resume a run with changed model/protocol settings")
            # Separate model processes may reach this file together. Publish
            # complete JSON atomically so another worker cannot read a truncation.
            temporary = model_folder / f"manifest-{args.shard_index}.tmp"
            temporary.write_text(json.dumps(manifest, indent=2))
            temporary.replace(manifest_path)
            for config in configs:
                while not (args.data / config).exists() or not list((args.data / config).glob("*/complete.json")):
                    print(f"waiting for {config} preparation", flush=True)
                    time.sleep(10)
                # Metadata determines every split, including all 11 accents/12 subjects.
                metadata = json.loads((args.output / "parquet-metadata.json").read_text())["parquet_files"]
                splits = sorted({f["split"] for f in metadata if f["config"] == config})
                for split in splits:
                    folder = args.data / config / split
                    while not (folder / "complete.json").exists():
                        time.sleep(10)
                    rows = [json.loads(s) for s in (folder / "manifest.jsonl").read_text().split("\n") if s.strip()]
                    if args.num_shards > 1:
                        rows = [r for r in rows if r["index"] % args.num_shards == args.shard_index]
                    suffix = f"--shard-{args.shard_index}-of-{args.num_shards}" if args.num_shards > 1 else ""
                    output = model_folder / f"{config}--{split}{suffix}.jsonl"
                    old = [json.loads(s) for s in output.read_text().split("\n") if s.strip()] if output.exists() else []
                    seen = {r["voicebench_id"] for r in old}
                    if len(old) != len(seen):
                        raise ValueError("Duplicate completed example IDs")
                    pending = sorted((r for r in rows if r["voicebench_id"] not in seen), key=lambda r: r["samples"])
                    written = len(seen)
                    current_batch_size = args.batch_size
                    with output.open("a") as handle:
                        while pending:
                            size = min(current_batch_size, len(pending))
                            while size > 1 and size * (math.ceil(pending[size - 1]["samples"] / 32000) * 50 + 2100) > args.frame_budget:
                                size -= 1
                            batch_rows = pending[:size]
                            waves = [read_audio(r) for r in batch_rows]
                            tick = time.monotonic()
                            try:
                                results = engine.generate(waves)
                            except torch.cuda.OutOfMemoryError:
                                if size == 1:
                                    raise
                                gc.collect()
                                torch.cuda.empty_cache()
                                current_batch_size = max(1, size // 2)
                                print(f"OOM: retrying with batch size {current_batch_size}", flush=True)
                                continue
                            traces = model_folder / "tokens" / config / split
                            traces.mkdir(parents=True, exist_ok=True)
                            for row, result in zip(batch_rows, results):
                                ids = result.pop("token_ids")
                                events = result.pop("event_ids", [])
                                raw = result.pop("raw_event_ids", [])
                                path = traces / f"{row['index']:06d}.npz"
                                np.savez_compressed(path, token_ids=np.array(ids, dtype=np.int32),
                                    event_ids=np.array(events, dtype=np.int32), raw_event_ids=np.array(raw, dtype=np.int32))
                                record = {**row["source"], **result, "voicebench_id": row["voicebench_id"],
                                    "pcm_sha256": row["pcm_sha256"], "duration_s": row["duration_s"],
                                    "token_count": len(ids), "tokens_path": str(path.resolve())}
                                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                            handle.flush()
                            written += len(results)
                            del pending[:size]
                            print(f"{mode} {config}/{split} {written}/{len(rows)} batch={size} seconds={time.monotonic() - tick:.1f}", flush=True)
                            status = f"status-{mode}-{args.shard_index}.json" if args.num_shards > 1 else "status.json"
                            (args.output / status).write_text(json.dumps({"model": mode, "config": config,
                                "split": split, "written": written, "total": len(rows), "elapsed_s": time.monotonic() - started}))
                    if written != len(rows):
                        raise RuntimeError("Incomplete benchmark split")
            marker = f"complete-shard-{args.shard_index}-of-{args.num_shards}.json" if args.num_shards > 1 else "complete.json"
            (model_folder / marker).write_text(json.dumps({"elapsed_s": time.monotonic() - started,
                "configs": configs, "num_shards": args.num_shards}))
        del engine
        gc.collect()
        torch.cuda.empty_cache()
    print("inference complete", flush=True)


if __name__ == "__main__":
    main()
