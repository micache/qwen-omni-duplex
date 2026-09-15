"""Fixed 80/20 continuous-span, 300-step loss-calibration candidate."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from duplex.dataset import DuplexCollator, Split
from duplex.recovery import load_continuous_spans
from duplex.training import (_ListDataset, _load_adapter_weights, build_training_model,
                             load_training_config, make_trainer, seed_everything)
from generate import _context_token_ids
from scripts.run_session15_repair_overfit import assess, dump


def pilot_gates(report):
    teacher = report["teacher_forced"]["events"]
    predicted = teacher["prediction_counts"]
    labels = teacher["label_counts"]
    cases = report["cases"]
    failures = []
    if predicted["TEXT"] == 0 or not any(case["lexical_events"] for case in cases):
        failures.append("lexical events absent in teacher or free execution")
    if predicted["IDLE"] / sum(predicted.values()) >= 0.98:
        failures.append("teacher predictions collapsed to IDLE")
    if any(sum(segment["lexical_events"] == 0 for segment in case["segments"]) >= 2 for case in cases):
        failures.append("repeated empty control cycles")
    for kind in ("START", "STOP"):
        metric = teacher["control"][kind]
        precision = metric["precision"]["value"] or 0
        recall = metric["recall"]["value"] or 0
        if recall <= 0 or precision <= 0.15:
            failures.append(f"{kind} teacher precision/recall inadequate: {precision:.3f}/{recall:.3f}")
    responding = [case for case in cases if case["mode"] == "normal"]
    interrupted = [case for case in cases if case["mode"] == "interrupted"]
    if not any(case["text"] for case in responding) or not any(case["text"] for case in interrupted):
        failures.append("completed or interrupted responding probes all have empty text")
    if not any(case["after_onset_stop"] for case in interrupted):
        failures.append("interruption STOP recall is zero")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--fresh-verify", action="store_true")
    args = parser.parse_args()
    config = load_training_config(args.config)
    output = Path(config["training"]["output_dir"])
    if output.exists() and not args.fresh_verify:
        raise FileExistsError(output)
    seed_everything(config["training"]["seed"])
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    model, processor, targets = build_training_model(config)
    if args.fresh_verify:
        _load_adapter_weights(model, output / "final")
    spans, manifest = load_continuous_spans(config, processor.tokenizer,
                                            model.control_tokens,
                                            model.base_thinker.config.bos_token_id)
    train = [span for span in spans if span.split is Split.TRAIN][:80]
    validation = [span for span in spans if span.split is Split.VALIDATION][:20]
    if len(train) != 80 or len(validation) != 20:
        raise ValueError("Frozen 80/20 pilot quota changed")
    context = _context_token_ids(processor.tokenizer, config["generation"]["system"], "")
    collator = DuplexCollator(audio_processor=processor, tokenizer=processor.tokenizer,
                             control_tokens=model.control_tokens,
                             thinker_bos_token_id=model.base_thinker.config.bos_token_id,
                             context_token_ids=context, interruption_probability=0)
    if not args.fresh_verify:
        values = [timeline for span in train for timeline in (span.normal, span.interrupted)]
        trainer, audit = make_trainer(config, model, processor, targets,
                                      _ListDataset(values), None, collator)
        trainer.train()
        trainer.save_model(output / "final")
        if not audit.nonzero_gradient_names:
            raise RuntimeError("No nonzero LoRA gradient")
    report = assess(model, processor, validation[:10], collator, context,
                    output / ("fresh" if args.fresh_verify else "trained"),
                    teacher_selected=validation)
    failures = pilot_gates(report)
    report.update({"pilot_gate": "PASS" if not failures else "FAIL",
                   "pilot_failures": failures,
                   "train_conversation_ids": [span.conversation_id for span in train],
                   "validation_conversation_ids": [span.conversation_id for span in validation],
                   "train_sequence_count": len(train) * 2,
                   "validation_sequence_count": len(validation) * 2,
                   "optimizer_steps": None if args.fresh_verify else trainer.state.global_step,
                   "runtime_seconds": time.perf_counter() - started,
                   "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(),
                   "dataset_revision": manifest["dataset_revision"],
                   "model_revision": config["model"]["revision"]})
    dump(output / ("fresh_report.json" if args.fresh_verify else "report.json"), report)
    print(f"PILOT_GATE={report['pilot_gate']}", flush=True)
    if failures:
        print("; ".join(failures), flush=True)


if __name__ == "__main__":
    main()
