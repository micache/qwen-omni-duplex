"""Prepare the complete scored VoiceBench suite without duplicate Arrow caches."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import time
import urllib.request

import numpy as np
import pyarrow.parquet as pq
from scipy.signal import resample_poly
import soundfile as sf

CONFIGS = ("commoneval", "alpacaeval_full", "wildvoice", "sd-qa", "ifeval",
           "advbench", "openbookqa", "mmsu", "bbh")
OFFLINE_CONFIGS = ("ifeval", "advbench", "openbookqa", "mmsu", "bbh")
REVISION = "de69a22f41561676635bef0b31681df4b866ec07"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=Path("outputs/voicebench-response-gain003/parquet-metadata.json"))
    parser.add_argument("--output", type=Path, default=Path("data/VoiceBench-eval"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    files = json.loads(args.metadata.read_text())["parquet_files"]
    for config in CONFIGS:
        splits = sorted({f["split"] for f in files if f["config"] == config})
        for split in splits:
            folder = args.output / config / split
            folder.mkdir(parents=True, exist_ok=True)
            if (folder / "complete.json").exists():
                continue
            manifest = folder / "manifest.jsonl"
            # Completed source shards are individually reusable after a restart.
            existing = [json.loads(s) for s in manifest.read_text().split("\n") if s.strip()] if manifest.exists() else []
            completed = set(json.loads((folder / "shards.json").read_text())) if (folder / "shards.json").exists() else set()
            kept = [r for r in existing if r["source_shard"] in completed]
            manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept))
            index = len(kept)
            for source in sorted((f for f in files if f["config"] == config and f["split"] == split), key=lambda f: f["filename"]):
                name = source["filename"]
                if name in completed:
                    continue
                url = source["url"].replace("refs%2Fconvert%2Fparquet", REVISION)
                temporary = args.output / "download.parquet.partial"
                print(f"download {config}/{split}/{name}", flush=True)
                for attempt in range(8):
                    try:
                        urllib.request.urlretrieve(url, temporary)
                        if temporary.stat().st_size != source["size"]:
                            raise IOError("Parquet byte count mismatch")
                        break
                    except Exception:
                        if attempt == 7:
                            raise
                        time.sleep(min(2 ** attempt, 30))
                with manifest.open("a") as handle:
                    for batch in pq.ParquetFile(temporary).iter_batches(batch_size=4):
                        for row in batch.to_pylist():
                            audio = row.pop("audio")
                            if audio.get("bytes") is None:
                                raise ValueError("Expected embedded source audio bytes")
                            waveform, rate = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
                            if waveform.ndim == 2:
                                waveform = waveform.mean(axis=1)
                            if rate != 16000:
                                divisor = math.gcd(rate, 16000)
                                waveform = resample_poly(waveform, 16000 // divisor, rate // divisor).astype(np.float32)
                            if not len(waveform) or not np.isfinite(waveform).all():
                                raise ValueError("Invalid source waveform")
                            scale = max(1.0, float(np.abs(waveform).max()) * 1.000001)
                            path = folder / f"{index:06d}.flac"
                            sf.write(path, waveform / scale, 16000, subtype="PCM_24")
                            restored, restored_rate = sf.read(path, dtype="float32")
                            restored *= scale
                            error = float(np.abs(restored - waveform).max())
                            if restored_rate != 16000 or error > 2e-7 * scale:
                                raise ValueError("Lossless-container quantization audit failed")
                            record = {"voicebench_id": f"{config}:{split}:{index}", "index": index,
                                      "source_shard": name, "source": row,
                                      "audio_path": str(path.resolve()), "audio_scale": scale,
                                      "samples": len(waveform), "duration_s": len(waveform) / 16000,
                                      "pcm_sha256": hashlib.sha256(restored.tobytes()).hexdigest(),
                                      "max_encoding_error": error}
                            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                            index += 1
                completed.add(name)
                (folder / "shards.json").write_text(json.dumps(sorted(completed)))
                temporary.unlink()
                print(f"prepared {config}/{split}: {index} samples", flush=True)
            (folder / "complete.json").write_text(json.dumps({"rows": index, "revision": REVISION}))
    print("all scored subsets prepared", flush=True)


if __name__ == "__main__":
    main()
