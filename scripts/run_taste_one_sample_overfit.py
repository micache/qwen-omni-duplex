"""Memorize one TASTE row with the main Thinker/LoRA recipe and inspect decoding."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from duplex.streaming import QwenDuplexStreamer, write_trace_jsonl
from duplex.training import (
    _ListDataset,
    _cached_free_report,
    _teacher_forced_report,
    assert_optimizer_scope,
    build_datasets_and_collator,
    build_training_model,
    load_training_config,
    make_trainer,
    seed_everything,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/taste_one_sample_overfit.yaml"))
    args = parser.parse_args()
    config = load_training_config(args.config)
    main_config = load_training_config("configs/turn_packed_main.yaml")
    for section in ("model", "timeline", "task", "loss_weights", "lora"):
        if config[section] != main_config[section]:
            raise ValueError(f"{section} differs from the full TASTE training recipe.")
    for key in ("method", "learning_rate", "lr_scheduler_type", "weight_decay",
                "max_grad_norm", "bf16", "gradient_checkpointing", "optimizer"):
        if config["training"][key] != main_config["training"][key]:
            raise ValueError(f"training.{key} differs from the full TASTE run.")
    sample_id = config["data"].get("overfit_sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("The one-sample diagnostic needs data.overfit_sample_id.")
    if config["data"].get("interruption_probability") != 0 or config["data"]["noise"]["probability"] != 0:
        raise ValueError("The memorization diagnostic requires deterministic audio and labels.")
    output = Path(config["training"]["output_dir"])
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing diagnostic: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("The BF16 LoRA run requires a CUDA GPU.")

    seed_everything(config["training"]["seed"])
    model, processor, targets = build_training_model(config)
    dataset, eval_dataset, collator = build_datasets_and_collator(config, processor, model)
    if len(dataset) != 1 or eval_dataset is not None:
        raise AssertionError("Expected exactly one training row and no evaluation split.")
    sample = dataset[0]
    if sample.conversation_id != sample_id:
        raise AssertionError(f"Selected wrong row: {sample.conversation_id}")
    timeline = collator._timeline(sample)
    fixed = _ListDataset([timeline])
    baseline = _teacher_forced_report(model, fixed, collator, processor.tokenizer)

    trainer, audit = make_trainer(config, model, processor, targets, dataset, None, collator)
    trainer.train()
    if not audit.nonzero_gradient_names:
        raise RuntimeError("No LoRA adapter received a nonzero gradient.")
    assert_optimizer_scope(trainer)
    trainer.save_model(output / "final")

    teacher = _teacher_forced_report(model, fixed, collator, processor.tokenizer)
    cached = _cached_free_report(model, fixed, collator, processor.tokenizer)
    target = list(timeline.causal.labels)
    teacher_ids = teacher["examples"][0]["prediction_ids"]
    cached_ids = cached["examples"][0]["prediction_ids"]
    streamer = QwenDuplexStreamer(model, processor, max_silent_chunks=2)
    aligned_stream = streamer.run(
        timeline.user_waveform,
        sample_id=f"{sample.conversation_id}-training-audio",
        context_token_ids=(),
        sample_rate_hz=sample.sample_rate_hz,
    )
    write_trace_jsonl(output / "training_audio_stream.jsonl", aligned_stream.trace)
    stream = streamer.run(
        sample.user_waveform,
        sample_id=sample.conversation_id,
        context_token_ids=(),
        sample_rate_hz=sample.sample_rate_hz,
    )
    write_trace_jsonl(output / "user_only_stream.jsonl", stream.trace)
    stream_ids = [row.event_id for row in stream.trace]
    report = {
        "sample_id": sample.conversation_id,
        "source_split": config["data"]["overfit_split"],
        "reference_text": sample.assistant_text,
        "user_audio_seconds": len(sample.user_waveform) / sample.sample_rate_hz,
        "training_audio_seconds": len(sample.input_waveform) / sample.sample_rate_hz,
        "training_frames": len(target),
        "optimizer_steps": trainer.state.global_step,
        "baseline": baseline,
        "final_teacher_forced": teacher,
        "final_cached_free": cached,
        "teacher_exact_all_frames": teacher_ids == target,
        "cached_free_exact_all_frames": cached_ids == target,
        "training_audio_stream": {
            "text": aligned_stream.text,
            "event_ids": [row.event_id for row in aligned_stream.trace],
            "timed_out": aligned_stream.timed_out,
            "stop_reason": aligned_stream.stop_reason,
            "trace": "training_audio_stream.jsonl",
            "exact_training_frames": [row.event_id for row in aligned_stream.trace[:len(target)]] == target,
        },
        "user_only_stream": {
            "text": stream.text,
            "event_ids": stream_ids,
            "event_types": [row.event_type for row in stream.trace],
            "timed_out": stream.timed_out,
            "stop_reason": stream.stop_reason,
            "trace": "user_only_stream.jsonl",
            "exact_reference_text": stream.text == sample.assistant_text,
        },
        "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    report["training_overfit_gate"] = (
        "PASS" if report["teacher_exact_all_frames"] and report["cached_free_exact_all_frames"]
        else "FAIL"
    )
    report["aligned_stream_gate"] = (
        "PASS" if report["training_audio_stream"]["exact_training_frames"] else "FAIL"
    )
    report["streaming_gate"] = (
        "PASS" if stream.text == sample.assistant_text and not stream.timed_out else "FAIL"
    )
    with (output / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"TRAINING_OVERFIT_GATE={report['training_overfit_gate']}")
    print(f"ALIGNED_STREAM_GATE={report['aligned_stream_gate']}")
    print(f"STREAMING_GATE={report['streaming_gate']}")
    print(f"baseline_loss={baseline['total_loss']:.6f} final_loss={teacher['total_loss']:.6f}")
    print(f"reference={sample.assistant_text!r} generated={stream.text!r}")
    print(f"report={output / 'report.json'}")


if __name__ == "__main__":
    main()
