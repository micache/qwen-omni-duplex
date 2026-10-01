"""Compare one, two, and four independent native model copies on equal work."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int)
    parser.add_argument("--copies", type=int)
    parser.add_argument("--folder", type=Path)
    args = parser.parse_args()
    if args.worker is not None:
        import torch
        from run_voicebench_suite import load_engine, read_audio, seed_everything
        torch.set_num_threads(max(1, min(4, 12 // args.copies)))
        seed_everything(17)
        rows = json.loads((args.folder / "inputs.json").read_text())
        size = len(rows) // args.copies
        selected = rows[args.worker * size:(args.worker + 1) * size]
        engine = load_engine("base", args)
        engine.max_new_tokens = 64
        engine.generate([read_audio(selected[0])])
        torch.cuda.empty_cache()
        (args.folder / f"ready-{args.worker}").touch()
        while not (args.folder / "start").exists():
            time.sleep(.1)
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        results = engine.generate([read_audio(r) for r in selected])
        torch.cuda.synchronize()
        (args.folder / f"result-{args.worker}.json").write_text(json.dumps({
            "seconds": time.monotonic() - started, "samples": len(results),
            "tokens": sum(len(r["token_ids"]) for r in results),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated()}))
        return
    output = ROOT / "outputs/voicebench-response-gain003/concurrency"
    output.mkdir(exist_ok=True)
    rows = []
    for config in ("advbench", "bbh"):
        source = ROOT / f"data/VoiceBench-eval/{config}/test/manifest.jsonl"
        all_rows = [json.loads(s) for s in source.read_text().split("\n") if s.strip()]
        rows.extend(sorted(all_rows, key=lambda r: r["samples"])[:32])
    rows.sort(key=lambda r: r["samples"])
    trials = []
    for copies in (1, 2, 4):
        folder = output / f"copies-{copies}"
        folder.mkdir(exist_ok=True)
        for old in list(folder.glob("ready-*")) + list(folder.glob("result-*.json")) + [folder / "start"]:
            old.unlink(missing_ok=True)
        (folder / "inputs.json").write_text(json.dumps(rows))
        processes, logs = [], []
        for rank in range(copies):
            log = (folder / f"worker-{rank}.log").open("w")
            logs.append(log)
            processes.append(subprocess.Popen([str(ROOT / ".venv/bin/python"), "-u", __file__,
                "--worker", str(rank), "--copies", str(copies), "--folder", str(folder)], cwd=ROOT,
                stdout=log, stderr=subprocess.STDOUT))
        load_started = time.monotonic()
        while not all((folder / f"ready-{rank}").exists() for rank in range(copies)):
            if any(p.poll() is not None for p in processes) or time.monotonic() - load_started > 900:
                break
            time.sleep(.25)
        ready = all((folder / f"ready-{rank}").exists() for rank in range(copies))
        started = time.monotonic()
        if ready:
            (folder / "start").touch()
            for process in processes:
                try:
                    process.wait(timeout=max(1, 900 - (time.monotonic() - started)))
                except subprocess.TimeoutExpired:
                    process.terminate()
        for process in processes:
            if process.poll() is None:
                process.terminate()
            process.wait()
        for log in logs:
            log.close()
        result_files = list(folder.glob("result-*.json"))
        valid = ready and all(p.returncode == 0 for p in processes) and len(result_files) == copies
        trial = {"copies": copies, "success": valid, "seconds": time.monotonic() - started,
                 "workers": [json.loads(p.read_text()) for p in result_files]}
        trials.append(trial)
        print(json.dumps(trial), flush=True)
    valid = [t for t in trials if t["success"]]
    if not valid:
        raise RuntimeError("All native concurrency trials failed; inspect worker logs")
    best = min(valid, key=lambda t: t["seconds"])
    single = next((t for t in valid if t["copies"] == 1), None)
    selected = best["copies"] if single is None or best["seconds"] < .9 * single["seconds"] else 1
    (output / "results.json").write_text(json.dumps({"trials": trials, "selected_copies": selected,
        "samples": len(rows), "max_new_tokens": 64,
        "note": "Diagnostic timing only; production uses the full 2048-token cap."}, indent=2))


if __name__ == "__main__":
    main()
