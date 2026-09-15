"""Run one frozen Session 15 BF16-LoRA experiment and preserve validation evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from pathlib import Path

import torch
import yaml
import importlib.metadata
from transformers import TrainerCallback

from duplex.dataset import DuplexCollator, Split
from duplex.training import _ListDataset, assert_optimizer_scope, build_training_model, load_training_config, make_trainer, seed_everything
from scripts.run_session11_diagnostic import _free_streaming_validation, _load_spans, _teacher_forced_validation, prepare_subset


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class MainValidation(TrainerCallback):
    def __init__(self, config, model, processor, validation_values, validation_spans, collator):
        self.config = config
        self.model = model
        self.processor = processor
        self.values = validation_values
        self.spans = validation_spans
        self.collator = collator
        self.output = Path(config["training"]["output_dir"])
        self.best_score = None
        self.best_step = None
        self.history = []
        self.idle_consecutive = 0

    def on_save(self, args, state, control, **kwargs):
        step = state.global_step
        teacher = _teacher_forced_validation(self.model, self.values, self.collator)
        original = self.config["diagnostic"]["trace_directory"]
        self.config["diagnostic"]["trace_directory"] = str(self.output / "traces" / f"step-{step}")
        try:
            free, summaries = _free_streaming_validation(self.model, self.processor, self.spans, self.config)
        finally:
            self.config["diagnostic"]["trace_directory"] = original
        counts = teacher["events"]["label_counts"]
        if counts["START"] == 0 or counts["STOP"] == 0:
            raise RuntimeError("Zero START/STOP validation labels")
        text = teacher["text_token_loss"]["value"]
        weighted = teacher["weighted_loss"]["value"]
        if text is None or weighted is None or not all(math.isfinite(v) for v in (text, weighted)):
            raise FloatingPointError("Non-finite validation loss")
        predictions = free["events"]["prediction_counts"]
        total = sum(predictions.values())
        idle = predictions["IDLE"] / total
        self.idle_consecutive = self.idle_consecutive + 1 if idle >= 0.995 else 0
        if self.idle_consecutive >= 2:
            raise RuntimeError("Persistent all-IDLE validation predictions")
        control = teacher["events"]["control"]
        f1 = (control["START"]["f1"]["value"] or 0) + (control["STOP"]["f1"]["value"] or 0)
        stop_recall = free["interruption_STOP"]["recall"]["value"] or 0
        # Trade lexical loss against both control F1 and post-onset STOP recall.
        score = (text + 2.0 * (1.0 - f1 / 2.0) + 2.0 * (1.0 - stop_recall) + 0.1 * weighted,
                 text, -f1, -stop_recall)
        result = {"step": step, "teacher_forced": {k: v for k, v in teacher.items() if k not in ("labels", "predictions")},
                  "free_streaming": free, "traces": summaries, "selection_score": list(score)}
        dump(self.output / "validation" / f"step-{step}.json", result)
        self.history.append(result)
        if self.best_score is None or score < self.best_score:
            source = self.output / f"checkpoint-{step}"
            destination = self.output / "selected"
            temporary = self.output / "selected.tmp"
            if temporary.exists():
                shutil.rmtree(temporary)
            temporary.mkdir()
            for name in ("adapter_model.safetensors", "adapter_config.json", "duplex_config.yaml", "training_config.yaml", "dependency_versions.json"):
                shutil.copy2(source / name, temporary / name)
            if destination.exists():
                shutil.rmtree(destination)
            temporary.rename(destination)
            self.best_score, self.best_step = score, step
        dump(self.output / "selection.json", {"selected_step": self.best_step, "score": self.best_score,
                                                  "evaluated_steps": [item["step"] for item in self.history]})
        print(f"SESSION15 step={step} text_loss={text:.4f} weighted={weighted:.4f} idle={idle:.3f} selected={self.best_step}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    config = load_training_config(args.config)
    output = Path(config["training"]["output_dir"])
    source = Path(config["data"]["root"]) / config["data"]["manifest"]
    if digest(source) != config["data"]["source_manifest_sha256"]:
        raise RuntimeError("Public source manifest hash changed")
    if args.prepare_only:
        if output.exists():
            raise FileExistsError(f"Distinct run directory already exists: {output}")
        output.mkdir(parents=True)
        shutil.copy2(args.config, output / "resolved_config.yaml")
        packages = ("torch", "transformers", "accelerate", "peft", "bitsandbytes", "datasets", "huggingface_hub", "PyYAML", "soundfile", "scipy", "numpy")
        versions = {name: importlib.metadata.version(name) for name in packages}
        dump(output / "freeze.json", {"run_id": output.name, "config_sha256": digest(args.config),
                                       "source_manifest_sha256": digest(source), "base_revision": config["model"]["revision"],
                                       "dependency_versions": versions})
        prepare_subset(config)
        manifest = json.loads(Path(config["diagnostic"]["subset_manifest"]).read_text())
        ids = [item["conversation_id"] for item in manifest["items"]]
        if len(ids) != len(set(ids)):
            raise RuntimeError("Conversation leakage: duplicate IDs")
        freeze = json.loads((output / "freeze.json").read_text())
        freeze["subset_manifest_sha256"] = digest(Path(config["diagnostic"]["subset_manifest"]))
        dump(output / "freeze.json", freeze)
        return
    if digest(output / "resolved_config.yaml") != digest(args.config):
        raise RuntimeError("Frozen config differs from launch config")
    freeze = json.loads((output / "freeze.json").read_text())
    if digest(Path(config["diagnostic"]["subset_manifest"])) != freeze["subset_manifest_sha256"]:
        raise RuntimeError("Frozen subset manifest changed")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 preflight failed")
    seed_everything(config["training"]["seed"])
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model, processor, targets = build_training_model(config)
    spans, manifest = _load_spans(config, tokenizer=processor.tokenizer, controls=model.control_tokens,
                                  bos_token_id=model.base_thinker.config.bos_token_id)
    train_spans = [span for span in spans if span.split is Split.TRAIN]
    validation_spans = [span for span in spans if span.split is Split.VALIDATION]
    if set(s.conversation_id for s in train_spans) & set(s.conversation_id for s in validation_spans):
        raise RuntimeError("Conversation leakage between train and validation")
    train_values = [timeline for span in train_spans for timeline in (*span.normal, span.interrupted)]
    validation_values = [timeline for span in validation_spans for timeline in (*span.normal, span.interrupted)]
    counts = manifest["complete_target_histogram"]
    if not counts["START"] or not counts["STOP"]:
        raise RuntimeError("Zero START/STOP labels")
    collator = DuplexCollator(audio_processor=processor, tokenizer=processor.tokenizer,
                             control_tokens=model.control_tokens, thinker_bos_token_id=model.base_thinker.config.bos_token_id,
                             frame_rate_hz=25, interruption_probability=0.0,
                             augmentation_seed=config["training"]["interruption_seed"])
    trainer, audit = make_trainer(config, model, processor, targets, _ListDataset(train_values),
                                  _ListDataset(validation_values), collator)
    callback = MainValidation(config, model, processor, validation_values, validation_spans, collator)
    trainer.add_callback(callback)
    checkpoints = sorted(output.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    resume = None
    if checkpoints:
        candidate = checkpoints[-1]
        metadata = candidate / "training_config.yaml"
        if not metadata.exists() or yaml.safe_load(metadata.read_text()) != config:
            raise RuntimeError(f"Checkpoint config validation failed: {candidate}")
        resume = str(candidate)
    trainer.train(resume_from_checkpoint=resume)
    assert_optimizer_scope(trainer)
    if not audit.nonzero_gradient_names:
        raise RuntimeError("No nonzero LoRA gradients")
    dump(output / "runtime.json", {"optimizer_steps": trainer.state.global_step,
                                   "runtime_seconds": time.perf_counter() - started,
                                   "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(),
                                   "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved(),
                                   "gpu": torch.cuda.get_device_name(0), "selected_step": callback.best_step})
    print(f"MAIN_RUN_DONE selected={callback.best_step}", flush=True)


if __name__ == "__main__":
    main()
