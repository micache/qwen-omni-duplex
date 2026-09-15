"""Bounded continuous-span overfit gate; no main run is launched here."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from duplex.dataset import DuplexCollator
from duplex.recovery import load_continuous_spans
from duplex.streaming import QwenDuplexStreamer, write_trace_jsonl
from duplex.training import (_ListDataset, _load_adapter_weights, build_training_model,
                             load_training_config, make_trainer, seed_everything)
from generate import _context_token_ids
from scripts.run_session11_diagnostic import _teacher_forced_validation


PUBLIC_IDS = ("data_stereo/38", "data_stereo/64", "data_stereo/84", "data_stereo/93")


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def assess(model, processor, selected, collator, context, output, teacher_selected=None):
    model.eval()
    model.gradient_checkpointing_disable()
    values = [timeline for span in (teacher_selected or selected)
              for timeline in (span.normal, span.interrupted)]
    teacher = _teacher_forced_validation(model, values, collator)
    streamer = QwenDuplexStreamer(model, processor, max_silent_chunks=1)
    cases = []
    for span in selected:
        for mode, timeline in (("normal", span.normal), ("interrupted", span.interrupted)):
            name = f"{span.conversation_id.replace('/', '-')}-{mode}"
            result = streamer.run(timeline.user_waveform, sample_id=name,
                                  context_token_ids=context, sample_rate_hz=timeline.audio_sample_rate_hz)
            trace = output / "traces" / f"{name}.jsonl"
            write_trace_jsonl(trace, result.trace)
            onset = timeline.interruption.cut_frame if timeline.interruption else None
            after_onset_stop = onset is not None and any(
                index >= onset and row.event_type == "STOP" for index, row in enumerate(result.trace))
            lexical = sum(row.event_type == "TEXT" for row in result.trace)
            segments = []
            active = None
            text_count = 0
            for index, row in enumerate(result.trace):
                if row.event_type == "START":
                    active, text_count = index, 0
                elif row.event_type == "TEXT" and active is not None:
                    text_count += 1
                elif row.event_type == "STOP" and active is not None:
                    segments.append({"start": active, "stop": index, "lexical_events": text_count})
                    active = None
            cases.append({"sample_id": name, "mode": mode, "text": result.text,
                          "event_counts": dict(Counter(row.event_type for row in result.trace)),
                          "segments": segments, "lexical_events": lexical,
                          "interruption_onset": onset, "after_onset_stop": after_onset_stop,
                          "trace": str(trace)})
    counts = teacher["events"]["prediction_counts"]
    label_counts = teacher["events"]["label_counts"]
    kind_accuracy = sum(teacher["events"]["control"][kind]["true_positive_count"]
                        for kind in ("IDLE", "START", "STOP")) / sum(label_counts.values())
    complete = any(any(segment["lexical_events"] > 0 for segment in case["segments"])
                   for case in cases)
    empty_cycles = any(sum(segment["lexical_events"] == 0 for segment in case["segments"]) >= 2
                       for case in cases)
    failures = []
    if kind_accuracy < 0.6:
        failures.append(f"teacher event-kind accuracy {kind_accuracy:.3f} < 0.6")
    if counts["TEXT"] == 0:
        failures.append("teacher lexical argmax count is zero")
    if not any(case["text"] for case in cases):
        failures.append("all greedy responses are empty")
    if not complete:
        failures.append("no START-lexical-STOP response")
    if not any(case["after_onset_stop"] for case in cases if case["mode"] == "interrupted"):
        failures.append("no causal interruption STOP")
    if empty_cycles:
        failures.append("repeated empty START/STOP cycles")
    return {"gate": "PASS" if not failures else "FAIL", "failures": failures,
            "teacher_kind_accuracy": kind_accuracy,
            "teacher_forced": {k: v for k, v in teacher.items() if k not in ("labels", "predictions")},
            "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/session15_repair_overfit.yaml")
    parser.add_argument("--fresh-verify", action="store_true")
    args = parser.parse_args()
    config = load_training_config(args.config)
    output = Path(config["training"]["output_dir"])
    if output.exists() and not args.fresh_verify:
        raise FileExistsError(output)
    seed_everything(config["training"]["seed"])
    started = time.perf_counter()
    model, processor, targets = build_training_model(config)
    if args.fresh_verify:
        _load_adapter_weights(model, output / "final")
    spans, manifest = load_continuous_spans(config, processor.tokenizer, model.control_tokens,
                                            model.base_thinker.config.bos_token_id)
    selected = [span for span in spans if span.conversation_id in PUBLIC_IDS]
    if len(selected) != len(PUBLIC_IDS):
        raise ValueError("Frozen public overfit slice changed.")
    if not any(any(utterance.source_start_seconds < boundary < utterance.source_end_seconds
                   for boundary in (span.normal.window_start_seconds + 2,
                                    span.normal.window_start_seconds + 4,
                                    span.normal.window_start_seconds + 6)
                   for utterance in span.normal.utterances) for span in selected):
        raise ValueError("Overfit slice lacks an internal cross-boundary response.")
    context = _context_token_ids(processor.tokenizer, config["generation"]["system"], "")
    collator = DuplexCollator(audio_processor=processor, tokenizer=processor.tokenizer,
                             control_tokens=model.control_tokens,
                             thinker_bos_token_id=model.base_thinker.config.bos_token_id,
                             context_token_ids=context, interruption_probability=0)
    if not args.fresh_verify:
        torch.cuda.reset_peak_memory_stats()
        values = [timeline for span in selected for timeline in (span.normal, span.interrupted)]
        trainer, audit = make_trainer(config, model, processor, targets, _ListDataset(values), None, collator)
        trainer.train()
        trainer.save_model(output / "final")
        if not audit.nonzero_gradient_names:
            raise RuntimeError("No LoRA gradients were observed.")
    report = assess(model, processor, selected, collator, context,
                    output / ("fresh" if args.fresh_verify else "trained"))
    report.update({"public_ids": PUBLIC_IDS, "context_ids": context,
                   "optimizer_steps": None if args.fresh_verify else trainer.state.global_step,
                   "runtime_seconds": time.perf_counter() - started,
                   "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(),
                   "dataset_revision": manifest["dataset_revision"],
                   "model_revision": config["model"]["revision"]})
    dump(output / ("fresh_report.json" if args.fresh_verify else "report.json"), report)
    print(f"OVERFIT_GATE={report['gate']}", flush=True)
    if report["failures"]:
        print("; ".join(report["failures"]), flush=True)


if __name__ == "__main__":
    main()
