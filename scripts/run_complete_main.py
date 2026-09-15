"""One three-epoch complete-conversation BF16 LoRA run and usable selection."""

from __future__ import annotations

import argparse
import json
import gc
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import soundfile as sf
import torch

from duplex.contract import prompt_token_ids
from duplex.conversations import preflight_complete_conversations
from duplex.dataset import DuplexCollator, Split
from duplex.model import NextEventLossWeights
from duplex.training import (build_training_model, load_training_config,
                             make_trainer, seed_everything)


def dump(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def scan_gold(dataset):
    counts = Counter()
    invalid = []
    for index in range(len(dataset)):
        try:
            timeline = dataset[index]
            for event in timeline.targets.events:
                counts["text" if event.kind.value == "TEXT" else event.kind.value.lower()] += 1
        except Exception as error:
            invalid.append({"id": dataset.entries[index].conversation_id, "error": str(error)})
    if invalid:
        raise ValueError(f"Complete-conversation target construction failed: {invalid}")
    return {name: counts[name] for name in ("text", "idle", "start", "stop")}


def make_frozen_cases(validation, output: Path):
    cases = []
    for index in range(min(3, len(validation))):
        timeline = validation[index]
        first_user_end = next((word.end_seconds for word in timeline.user_words), 2.0)
        question = timeline.user_waveform[:round(first_user_end * 16_000)]
        path = output / "frozen-cases" / f"{timeline.sample_id.split('@')[0].replace('/', '-')}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, np.concatenate((question, np.zeros(32_000, dtype=np.float32))), 16_000)
        cases.append(path)
    return cases


def eligible_case(result):
    counts = result["events"]
    total = max(1, sum(counts.values()))
    return (bool(result["text"].strip()) and result["segments"]
            and all(segment["tokens"] > 0 for segment in result["segments"])
            and counts.get("START", 0) <= 3
            and counts.get("STOP", 0) <= 3
            and counts.get("IDLE", 0) / total < 0.995)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/full_conversation_main.yaml")
    args = parser.parse_args()
    config = load_training_config(args.config)
    overfit = json.loads(Path("outputs/full-conversation-overfit-111/report.json").read_text())
    if overfit.get("gate") != "PASS":
        raise RuntimeError("Mandatory complete-conversation overfit gate has not passed.")
    seed_everything(config["training"]["seed"])
    model, processor, targets = build_training_model(config)
    datasets, preflight = preflight_complete_conversations(config, processor, model)
    output = Path(config["training"]["output_dir"])
    dump(output / "preflight.json", preflight)
    train, validation = datasets[Split.TRAIN], datasets[Split.VALIDATION]
    train_counts = scan_gold(train)
    validation_counts = scan_gold(validation)
    if not train_counts["start"] or not validation_counts["start"]:
        raise RuntimeError("No START labels in complete-conversation splits.")
    model.loss_weights = NextEventLossWeights.capped_inverse_sqrt(
        train_counts, max_ratio=config.get("loss_weight_max_ratio", 5.0))
    model.audio_cache_dir = output / "audio-cache"
    prompt_ids = prompt_token_ids(processor.tokenizer)
    collator = DuplexCollator(audio_processor=processor, tokenizer=processor.tokenizer,
        control_tokens=model.control_tokens, thinker_bos_token_id=model.base_thinker.config.bos_token_id,
        context_token_ids=prompt_ids, interruption_probability=config["data"]["interruption_probability"],
        min_assistant_frames=config["data"]["min_assistant_frames"],
        augmentation_seed=config["training"]["interruption_seed"])
    cases = make_frozen_cases(validation, output)
    dump(output / "data_summary.json", {"preflight": preflight,
        "train_gold": train_counts, "validation_gold": validation_counts,
        "loss_weights": model.loss_weights.as_dict(), "frozen_cases": [str(path) for path in cases],
        "seed": config["training"]["seed"], "command": ".venv/bin/python scripts/run_complete_main.py --config configs/full_conversation_main.yaml"})
    print(f"STRUCTURAL_GATE=PASS train={len(train)} validation={len(validation)} frames={preflight['audio_frames']['train']} excluded={len(preflight['excluded'])}", flush=True)
    trainer, audit = make_trainer(config, model, processor, targets, train, validation, collator)
    trainer.train()
    checkpoints = sorted(output.glob("checkpoint-*"), key=lambda path: int(path.name.split("-")[-1]))
    history = list(trainer.state.log_history)
    del trainer, audit, model, processor, collator, datasets, train, validation
    gc.collect()
    torch.cuda.empty_cache()
    assessments = []
    for checkpoint in checkpoints:
        command = [str(Path(".venv/bin/python").resolve()),
                   str(Path("scripts/verify_complete_checkpoint.py").resolve()),
                   "--config", args.config, "--adapter", str(checkpoint),
                   "--cases", *[str(path) for path in cases]]
        result = subprocess.run(command, text=True, capture_output=True)
        report_path = checkpoint / "generation_gate.json"
        gate = json.loads(report_path.read_text()) if result.returncode == 0 else {"eligible": False, "error": result.stderr}
        assessments.append({"checkpoint": str(checkpoint), "gate": gate})
    eligible = [row for row in assessments if row["gate"].get("eligible")]
    for row in assessments:
        step = int(Path(row["checkpoint"]).name.split("-")[-1])
        eval_metrics = next((record for record in reversed(history)
                             if record.get("step") == step and "eval_total_loss" in record), {})
        row["validation"] = eval_metrics
    if eligible:
        selected = min(eligible, key=lambda row: (
            row["validation"].get("eval_total_loss", float("inf")),
            row["validation"].get("eval_text_loss", float("inf"))))
        selected_path = selected["checkpoint"]
    else:
        selected_path = None
    dump(output / "selection.json", {"selected_checkpoint": selected_path,
                                     "candidates": assessments,
                                     "epoch_metrics": [row for row in history if "eval_total_loss" in row]})
    print(f"MAIN_CHECKPOINT_READY={'PASS' if selected_path else 'FAIL'} selected={selected_path}", flush=True)


if __name__ == "__main__":
    main()
