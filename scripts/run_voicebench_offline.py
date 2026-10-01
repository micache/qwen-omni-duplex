"""Finish offline paired inference and scoring, surviving terminal disconnects."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = str(ROOT / ".venv/bin/python")
OUTPUT = ROOT / "outputs/voicebench-response-gain003-silence"


def main():
    profile = OUTPUT / "concurrency/results.json"
    while not profile.exists():
        time.sleep(5)
    copies = json.loads(profile.read_text())["selected_copies"]
    print(f"Using {copies} native model copies after timing comparison", flush=True)
    completed_duplex = OUTPUT / "duplex/complete.json"
    required = {"ifeval", "advbench", "openbookqa", "mmsu", "bbh"}
    if not completed_duplex.exists() or not required.issubset(
            json.loads(completed_duplex.read_text()).get("configs", [])):
        with (OUTPUT / "offline-duplex.log").open("a") as log:
            subprocess.run([PYTHON, "-u", "scripts/run_voicebench_suite.py", "--offline-only",
                "--mode", "duplex", "--batch-size", "128", "--frame-budget", "300000",
                "--output", str(OUTPUT)],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    completed_base = OUTPUT / "base/complete.json"
    if completed_base.exists() and required.issubset(json.loads(completed_base.read_text()).get("configs", [])):
        with (OUTPUT / "offline-scoring.log").open("w") as log:
            subprocess.run([PYTHON, "-u", "scripts/score_voicebench_suite.py", "--offline-only",
                "--output", str(OUTPUT)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        print("Offline inference, artifact validation, and official scoring complete", flush=True)
        return
    processes, logs = [], []
    for rank in range(copies):
        log = (OUTPUT / f"offline-base-{rank}.log").open("a")
        logs.append(log)
        processes.append(subprocess.Popen([PYTHON, "-u", "scripts/run_voicebench_suite.py",
            "--offline-only", "--mode", "base", "--batch-size", str(128 // copies),
            "--frame-budget", str(240000 // copies), "--cpu-threads", str(min(4, 12 // copies)),
            "--num-shards", str(copies), "--shard-index", str(rank), "--output", str(OUTPUT)],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT))
    codes = [p.wait() for p in processes]
    for log in logs:
        log.close()
    if any(codes):
        raise RuntimeError(f"Native inference workers failed: {codes}; inspect per-worker logs")
    if copies > 1:
        for rank in range(copies):
            marker = json.loads((OUTPUT / "base" / f"complete-shard-{rank}-of-{copies}.json").read_text())
            if marker["num_shards"] != copies:
                raise ValueError("Incorrect shard completion metadata")
        (OUTPUT / "base/complete.json").write_text(json.dumps({"num_shards": copies,
            "configs": ["ifeval", "advbench", "openbookqa", "mmsu", "bbh"]}))
    with (OUTPUT / "offline-scoring.log").open("w") as log:
        subprocess.run([PYTHON, "-u", "scripts/score_voicebench_suite.py", "--offline-only",
            "--output", str(OUTPUT)],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    print("Offline inference, artifact validation, and official scoring complete", flush=True)


if __name__ == "__main__":
    main()
