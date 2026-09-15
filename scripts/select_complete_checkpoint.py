"""Evaluate all epoch adapters after training exits and select a usable one."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from duplex.training import load_training_config


def main():
    config_path = "configs/full_conversation_main.yaml"
    config = load_training_config(config_path)
    output = Path(config["training"]["output_dir"])
    summary = json.loads((output / "data_summary.json").read_text())
    cases = summary["frozen_cases"]
    candidates = []
    for checkpoint in sorted(output.glob("checkpoint-*"), key=lambda path: int(path.name.split("-")[-1])):
        command = [str(Path(".venv/bin/python").resolve()),
                   str(Path("scripts/verify_complete_checkpoint.py").resolve()),
                   "--config", config_path, "--adapter", str(checkpoint), "--cases", *cases]
        completed = subprocess.run(command, text=True, capture_output=True)
        gate_path = checkpoint / "generation_gate.json"
        gate = json.loads(gate_path.read_text()) if completed.returncode == 0 and gate_path.exists() else {"eligible": False, "error": completed.stderr}
        step = int(checkpoint.name.split("-")[-1])
        state = json.loads((checkpoint / "trainer_state.json").read_text())
        validation = next((row for row in reversed(state["log_history"])
                           if row.get("step") == step and "eval_total_loss" in row), {})
        candidates.append({"checkpoint": str(checkpoint), "gate": gate, "validation": validation})
    eligible = [row for row in candidates if row["gate"].get("eligible")]
    selected = min(eligible, key=lambda row: (row["validation"].get("eval_total_loss", float("inf")),
                                               row["validation"].get("eval_lexical_loss", float("inf")))) if eligible else None
    report = {"selected_checkpoint": selected["checkpoint"] if selected else None,
              "candidates": candidates, "command": ".venv/bin/python scripts/select_complete_checkpoint.py"}
    (output / "selection.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"MAIN_CHECKPOINT_READY={'PASS' if selected else 'FAIL'} selected={report['selected_checkpoint']}", flush=True)


if __name__ == "__main__":
    main()
