"""Prepare and run the bounded Session 11 real-data diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from duplex.dataset import (
    ConversationRecord,
    DuplexCollator,
    Speaker,
    Split,
    WindowKind,
    WindowMetadata,
    WordSpan,
    assign_split,
    load_conversation,
    read_manifest,
)
from duplex.metrics import (
    boundary_error_metrics,
    event_classification_metrics,
    free_streaming_rates,
    interruption_stop_metrics,
    invalid_state_transition_rate,
    teacher_forced_text_metrics,
)
from duplex.streaming import QwenDuplexStreamer, write_trace_jsonl
from duplex.timeline import (
    ControlTokenIds,
    EventKind,
    InterruptionConfig,
    WindowTimeline,
    augment_synthetic_interruption,
    build_window_timeline,
)
from duplex.training import (
    _ListDataset,
    _cuda_batch,
    assert_optimizer_scope,
    build_training_model,
    load_training_config,
    make_trainer,
    seed_everything,
)


@dataclass
class SpanData:
    conversation_id: str
    split: Split
    span_start_seconds: float
    record: ConversationRecord
    normal: tuple[WindowTimeline, ...]
    interrupted_chunk_index: int
    interrupted: WindowTimeline


def _json_dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def _stable_seed(seed: int, value: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _infer_user_words(
    record: ConversationRecord, diagnostic: Mapping[str, Any]
) -> tuple[WordSpan, ...]:
    frame_samples = round(
        float(diagnostic["vad_frame_seconds"]) * record.sample_rate_hz
    )
    waveform = np.asarray(record.user_waveform, dtype=np.float32)
    padding = (-len(waveform)) % frame_samples
    framed = np.pad(waveform, (0, padding)).reshape(-1, frame_samples)
    rms = np.sqrt(np.mean(np.square(framed, dtype=np.float64), axis=1))
    threshold = max(
        float(diagnostic["vad_floor_rms"]),
        float(np.percentile(rms, 95)) * float(diagnostic["vad_peak_fraction"]),
    )
    active = rms >= threshold

    # Channel leakage must not invent a user turn over an annotated assistant word.
    frame_seconds = frame_samples / record.sample_rate_hz
    for word in record.assistant_words:
        first = max(0, math.floor(word.start_seconds / frame_seconds))
        last = min(len(active), math.ceil(word.end_seconds / frame_seconds))
        active[first:last] = False

    merge_gap = int(diagnostic["vad_merge_gap_frames"])
    active_indices = np.flatnonzero(active)
    if active_indices.size:
        for left, right in zip(active_indices, active_indices[1:]):
            if 1 < right - left <= merge_gap + 1:
                active[left + 1 : right] = True

    minimum = int(diagnostic["vad_min_active_frames"])
    words = []
    start: int | None = None
    for frame, enabled in enumerate(np.append(active, False)):
        if enabled and start is None:
            start = frame
        elif not enabled and start is not None:
            if frame - start >= minimum:
                words.append(
                    WordSpan(
                        "<synthetic-user-activity>",
                        start * frame_seconds,
                        min(frame * frame_seconds, record.duration_seconds),
                        Speaker.USER,
                    )
                )
            start = None
    return tuple(words)


def _timeline_histogram(timelines: Sequence[WindowTimeline]) -> dict[str, int]:
    counts = Counter(
        event.kind.value for timeline in timelines for event in timeline.targets.events
    )
    return {
        kind.value: counts[kind.value]
        for kind in (
            EventKind.IDLE,
            EventKind.START,
            EventKind.TEXT,
            EventKind.STOP,
            EventKind.PADDING,
        )
    }


def _build_chunks(
    record: ConversationRecord,
    *,
    split: Split,
    span_start: float,
    chunks_per_span: int,
    tokenizer: object,
    controls: ControlTokenIds,
    bos_token_id: int,
) -> tuple[WindowTimeline, ...]:
    timelines = []
    for chunk_index in range(chunks_per_span):
        start = span_start + 2.0 * chunk_index
        window = WindowMetadata(
            conversation_id=record.conversation_id,
            split=split,
            start_seconds=start,
            end_seconds=start + 2.0,
            kind=WindowKind.RANDOM,
        )
        timelines.append(
            build_window_timeline(
                record,
                window,
                tokenizer=tokenizer,
                control_tokens=controls,
                thinker_bos_token_id=bos_token_id,
                frame_rate_hz=25,
            )
        )
    return tuple(timelines)


def _interrupted_candidates(
    timelines: Sequence[WindowTimeline],
    *,
    seed: int,
    min_assistant_frames: int,
    controls: ControlTokenIds,
    bos_token_id: int,
) -> list[tuple[int, WindowTimeline]]:
    candidates = []
    for chunk_index, timeline in enumerate(timelines):
        augmented = augment_synthetic_interruption(
            timeline,
            config=InterruptionConfig(
                probability=1.0,
                min_assistant_frames=min_assistant_frames,
            ),
            rng=random.Random(_stable_seed(seed, timeline.sample_id)),
            control_tokens=controls,
            thinker_bos_token_id=bos_token_id,
        )
        if augmented.interruption is not None:
            candidates.append((chunk_index, augmented))
    return candidates


def _best_span(
    record: ConversationRecord,
    *,
    split: Split,
    config: Mapping[str, Any],
    tokenizer: object,
    controls: ControlTokenIds,
    bos_token_id: int,
) -> SpanData | None:
    diagnostic = config["diagnostic"]
    span_seconds = int(diagnostic["span_seconds"])
    chunks_per_span = int(diagnostic["chunks_per_span"])
    if record.duration_seconds + 1e-9 < span_seconds:
        return None
    maximum_start = record.duration_seconds - span_seconds
    starts = [float(value) for value in range(0, math.floor(maximum_start) + 1, 2)]
    if not starts:
        starts = [0.0]
    best: tuple[tuple[int, int, int, float], SpanData] | None = None
    for span_start in starts:
        try:
            normal = _build_chunks(
                record,
                split=split,
                span_start=span_start,
                chunks_per_span=chunks_per_span,
                tokenizer=tokenizer,
                controls=controls,
                bos_token_id=bos_token_id,
            )
        except (ValueError, RuntimeError):
            continue
        interruptions = _interrupted_candidates(
            normal,
            seed=int(config["training"]["interruption_seed"]),
            min_assistant_frames=int(config["data"]["min_assistant_frames"]),
            controls=controls,
            bos_token_id=bos_token_id,
        )
        if not interruptions:
            continue
        # Prefer spans with more usable turns and lexical supervision. Ties select
        # the earlier public-data span to keep the manifest easy to inspect.
        nonidle = sum(
            event.kind is not EventKind.IDLE
            for timeline in normal
            for event in timeline.targets.events
        )
        text = sum(
            event.kind is EventKind.TEXT
            for timeline in normal
            for event in timeline.targets.events
        )
        chunk_index, interrupted = max(
            interruptions,
            key=lambda item: sum(
                event.kind is EventKind.TEXT for event in item[1].targets.events
            ),
        )
        score = (len(interruptions), nonidle, text, -span_start)
        value = SpanData(
            conversation_id=record.conversation_id,
            split=split,
            span_start_seconds=span_start,
            record=record,
            normal=normal,
            interrupted_chunk_index=chunk_index,
            interrupted=interrupted,
        )
        if best is None or score > best[0]:
            best = (score, value)
    return None if best is None else best[1]


def _manifest_item(span: SpanData) -> dict[str, Any]:
    metadata = span.interrupted.interruption
    assert metadata is not None
    return {
        "conversation_id": span.conversation_id,
        "split": span.split.value,
        "span_start_seconds": span.span_start_seconds,
        "span_end_seconds": span.span_start_seconds + 2.0 * len(span.normal),
        "chunk_starts_seconds": [
            timeline.window_start_seconds for timeline in span.normal
        ],
        "interrupted_chunk_index": span.interrupted_chunk_index,
        "interruption": {
            "source_turn_id": metadata.source_turn_id,
            "cut_frame": metadata.cut_frame,
            "original_stop_frame": metadata.original_stop_frame,
            "original_user_frame": metadata.original_user_frame,
            "shifted_frames": metadata.shifted_frames,
        },
    }


def _processor_and_controls(config: Mapping[str, Any]) -> tuple[object, ControlTokenIds, int]:
    from transformers import Qwen2_5OmniConfig, Qwen2_5OmniProcessor

    common = {
        "revision": config["model"]["revision"],
        "local_files_only": bool(config["model"]["local_files_only"]),
    }
    root_config = Qwen2_5OmniConfig.from_pretrained(config["model"]["name"], **common)
    processor = Qwen2_5OmniProcessor.from_pretrained(config["model"]["name"], **common)
    thinker = root_config.thinker_config
    controls = ControlTokenIds(
        idle=thinker.pad_token_id,
        start=thinker.bos_token_id,
        stop=thinker.eos_token_id,
        thinker_vocab_size=thinker.text_config.vocab_size,
    )
    return processor, controls, int(thinker.bos_token_id)


def prepare_subset(config: Mapping[str, Any]) -> dict[str, Any]:
    """Select the fixed subset and compute its complete target histogram."""

    processor, controls, bos_token_id = _processor_and_controls(config)
    data = config["data"]
    diagnostic = config["diagnostic"]
    root = Path(data["root"])
    entries = read_manifest(root / data["manifest"])
    quotas = {
        Split.TRAIN: int(diagnostic["train_conversations"]),
        Split.VALIDATION: int(diagnostic["validation_conversations"]),
    }
    split_salt = f"DailyTalkContiguous-session11-{config['training']['split_seed']}"
    spans: list[SpanData] = []
    skipped: Counter[str] = Counter()
    for entry in entries:
        split = assign_split(entry.conversation_id, salt=split_salt)
        if split not in quotas or sum(span.split is split for span in spans) >= quotas[split]:
            continue
        try:
            record = load_conversation(
                entry,
                dataset_root=root,
                speaker_label_map=data["speaker_label_map"],
            )
            record = replace(
                record,
                user_words=_infer_user_words(record, diagnostic),
            )
            span = _best_span(
                record,
                split=split,
                config=config,
                tokenizer=processor.tokenizer,
                controls=controls,
                bos_token_id=bos_token_id,
            )
        except (ValueError, RuntimeError, OSError) as error:
            skipped[type(error).__name__] += 1
            continue
        if span is None:
            skipped["no_eligible_span"] += 1
            continue
        spans.append(span)
        current = sum(value.split is split for value in spans)
        print(f"selected {split.value} conversation {current}/{quotas[split]}: {entry.conversation_id}")
        if all(sum(span.split is key for span in spans) >= quota for key, quota in quotas.items()):
            break
    actual = Counter(span.split.value for span in spans)
    for split, quota in quotas.items():
        if actual[split.value] != quota:
            raise RuntimeError(
                f"Only selected {actual[split.value]} of {quota} required {split.value} conversations."
            )

    complete = [timeline for span in spans for timeline in span.normal]
    complete.extend(span.interrupted for span in spans)
    histogram = _timeline_histogram(complete)
    manifest = {
        "schema_version": 1,
        "dataset": data["dataset"],
        "dataset_revision": data["dataset_revision"],
        "split_salt": split_salt,
        "selection": {
            "subset_conversations": len(spans),
            "train_conversations": actual[Split.TRAIN.value],
            "validation_conversations": actual[Split.VALIDATION.value],
            "span_seconds": diagnostic["span_seconds"],
            "chunks_per_span": diagnostic["chunks_per_span"],
            "fixed_chunk_seconds": 2.0,
            "vad": {
                key: diagnostic[key]
                for key in (
                    "vad_frame_seconds",
                    "vad_floor_rms",
                    "vad_peak_fraction",
                    "vad_merge_gap_frames",
                    "vad_min_active_frames",
                )
            },
        },
        "complete_target_histogram": histogram,
        "skipped_candidates": dict(sorted(skipped.items())),
        "items": [_manifest_item(span) for span in spans],
    }
    destination = Path(diagnostic["subset_manifest"])
    _json_dump(destination, manifest)
    payload = destination.read_bytes()
    print("complete target histogram:", json.dumps(histogram, sort_keys=True))
    print("subset manifest SHA-256:", hashlib.sha256(payload).hexdigest())
    return manifest


def _load_spans(
    config: Mapping[str, Any],
    *,
    tokenizer: object,
    controls: ControlTokenIds,
    bos_token_id: int,
) -> tuple[list[SpanData], dict[str, Any]]:
    manifest_path = Path(config["diagnostic"]["subset_manifest"])
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    entries = {
        entry.conversation_id: entry
        for entry in read_manifest(Path(config["data"]["root"]) / config["data"]["manifest"])
    }
    spans = []
    for item in manifest["items"]:
        conversation_id = item["conversation_id"]
        record = load_conversation(
            entries[conversation_id],
            dataset_root=config["data"]["root"],
            speaker_label_map=config["data"]["speaker_label_map"],
        )
        record = replace(
            record,
            user_words=_infer_user_words(record, config["diagnostic"]),
        )
        split = Split(item["split"])
        normal = _build_chunks(
            record,
            split=split,
            span_start=float(item["span_start_seconds"]),
            chunks_per_span=int(config["diagnostic"]["chunks_per_span"]),
            tokenizer=tokenizer,
            controls=controls,
            bos_token_id=bos_token_id,
        )
        chunk_index = int(item["interrupted_chunk_index"])
        interrupted = dict(
            _interrupted_candidates(
                normal,
                seed=int(config["training"]["interruption_seed"]),
                min_assistant_frames=int(config["data"]["min_assistant_frames"]),
                controls=controls,
                bos_token_id=bos_token_id,
            )
        )[chunk_index]
        if _manifest_item(
            SpanData(
                conversation_id,
                split,
                float(item["span_start_seconds"]),
                record,
                normal,
                chunk_index,
                interrupted,
            )
        ) != item:
            raise AssertionError(f"Rebuilt subset item changed: {conversation_id}.")
        spans.append(
            SpanData(
                conversation_id,
                split,
                float(item["span_start_seconds"]),
                record,
                normal,
                chunk_index,
                interrupted,
            )
        )
    complete = [timeline for span in spans for timeline in span.normal]
    complete.extend(span.interrupted for span in spans)
    if _timeline_histogram(complete) != manifest["complete_target_histogram"]:
        raise AssertionError("Rebuilt complete target histogram changed.")
    return spans, manifest


def _teacher_forced_validation(
    model: object,
    timelines: Sequence[WindowTimeline],
    collator: DuplexCollator,
) -> dict[str, Any]:
    labels = []
    predictions = []
    weighted_loss_sum = 0.0
    weight_sum = 0.0
    text_loss_sum = 0.0
    text_count = 0
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for timeline in timelines:
            batch = _cuda_batch(collator([timeline]), model)
            outputs = model(**batch)
            current_weight = float(outputs.loss_weight_sum)
            weighted_loss_sum += float(outputs.loss) * current_weight
            weight_sum += current_weight
            text_metrics = teacher_forced_text_metrics(
                outputs.logits,
                batch["labels"],
                control_tokens=model.control_tokens,
            )
            current_text = int(text_metrics["loss"]["denominator"])
            if current_text:
                text_loss_sum += float(text_metrics["loss"]["value"]) * current_text
                text_count += current_text
            labels.append(batch["labels"][0].detach().cpu().tolist())
            predictions.append(outputs.logits.argmax(dim=-1)[0].detach().cpu().tolist())
    total_loss = weighted_loss_sum / weight_sum if weight_sum else None
    text_loss = text_loss_sum / text_count if text_count else None
    return {
        "weighted_loss": {"value": total_loss, "denominator": weight_sum},
        "text_token_loss": {"value": text_loss, "denominator": text_count},
        "text_token_perplexity": {
            "value": None if text_loss is None else math.exp(text_loss),
            "denominator": text_count,
        },
        "events": event_classification_metrics(
            labels, predictions, control_tokens=model.control_tokens
        ),
        "invalid_transition_rate_before_masking": invalid_state_transition_rate(
            predictions, labels=labels, control_tokens=model.control_tokens
        ),
        "boundaries": boundary_error_metrics(
            labels, predictions, control_tokens=model.control_tokens
        ),
        "labels": labels,
        "predictions": predictions,
    }


def _span_waveform(span: SpanData, *, interrupted: bool) -> np.ndarray:
    start = round(span.span_start_seconds * span.record.sample_rate_hz)
    length = len(span.normal) * 2 * span.record.sample_rate_hz
    waveform = np.asarray(span.record.user_waveform[start : start + length]).copy()
    if interrupted:
        offset = span.interrupted_chunk_index * 2 * span.record.sample_rate_hz
        waveform[offset : offset + 2 * span.record.sample_rate_hz] = np.asarray(
            span.interrupted.user_waveform
        )
    return waveform


def _span_labels(span: SpanData, *, interrupted: bool) -> list[int]:
    chunks = list(span.normal)
    if interrupted:
        chunks[span.interrupted_chunk_index] = span.interrupted
    return [label for timeline in chunks for label in timeline.causal.labels]


def _decoded_target(labels: Sequence[int], model: object, tokenizer: object) -> str:
    controls = {
        model.control_tokens.idle,
        model.control_tokens.start,
        model.control_tokens.stop,
    }
    return tokenizer.decode(
        [value for value in labels if value not in controls],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def _free_streaming_validation(
    model: object,
    processor: object,
    spans: Sequence[SpanData],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    diagnostic = config["diagnostic"]
    normal_count = int(diagnostic["normal_traces"])
    interruption_count = int(diagnostic["interruption_traces"])
    selected = [(span, False) for span in spans[:normal_count]]
    selected.extend((span, True) for span in spans[:interruption_count])
    if len(selected) < int(diagnostic["validation_traces"]):
        raise RuntimeError("The fixed validation trace request exceeds available spans.")
    streamer = QwenDuplexStreamer(model, processor, max_silent_chunks=1)
    traces = []
    labels = []
    predictions = []
    raw_predictions = []
    onsets = []
    summaries = []
    trace_directory = Path(diagnostic["trace_directory"])
    for index, (span, interrupted) in enumerate(selected):
        mode = "interruption" if interrupted else "normal"
        sample_id = f"{span.conversation_id}@{span.span_start_seconds:.0f}s:{mode}"
        result = streamer.run(
            _span_waveform(span, interrupted=interrupted),
            sample_id=sample_id,
            context_token_ids=[],
            sample_rate_hz=span.record.sample_rate_hz,
        )
        trace_path = trace_directory / f"{index:02d}-{mode}.jsonl"
        write_trace_jsonl(trace_path, result.trace)
        target = _span_labels(span, interrupted=interrupted)
        selected_ids = [record.event_id for record in result.trace[: len(target)]]
        raw_ids = [record.raw_argmax_id for record in result.trace]
        if len(selected_ids) != len(target):
            raise RuntimeError(
                f"{sample_id}: streaming produced {len(selected_ids)} real-audio events, expected {len(target)}."
            )
        onset = None
        if interrupted:
            assert span.interrupted.interruption is not None
            onset = (
                span.interrupted_chunk_index * 50
                + span.interrupted.interruption.cut_frame
            )
        traces.append(result)
        labels.append(target)
        predictions.append(selected_ids)
        raw_predictions.append(raw_ids)
        onsets.append(onset)
        counts = Counter(record.event_type for record in result.trace)
        summary = {
            "sample_id": sample_id,
            "mode": mode,
            "target_text": _decoded_target(target, model, processor.tokenizer),
            "predicted_text": result.text,
            "event_counts": dict(sorted(counts.items())),
            "timed_out": result.timed_out,
            "stop_reason": result.stop_reason,
            "interruption_onset_frame": onset,
            "trace": str(trace_path),
        }
        summaries.append(summary)
        print(
            f"decoded validation trace {index + 1}/{len(selected)} {sample_id}: "
            f"{dict(sorted(counts.items()))}"
        )
    return (
        {
            "events": event_classification_metrics(
                labels, predictions, control_tokens=model.control_tokens
            ),
            "invalid_transition_rate_before_masking": invalid_state_transition_rate(
                raw_predictions, control_tokens=model.control_tokens
            ),
            "invalid_transition_rate_after_masking": invalid_state_transition_rate(
                predictions, control_tokens=model.control_tokens
            ),
            "boundaries": boundary_error_metrics(
                labels, predictions, control_tokens=model.control_tokens
            ),
            "interruption_STOP": interruption_stop_metrics(
                predictions, onsets, control_tokens=model.control_tokens
            ),
            **free_streaming_rates(traces),
        },
        summaries,
    )


def run_diagnostic(config: Mapping[str, Any]) -> dict[str, Any]:
    seed_everything(int(config["training"]["seed"]))
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model, processor, targets = build_training_model(config)
    spans, subset_manifest = _load_spans(
        config,
        tokenizer=processor.tokenizer,
        controls=model.control_tokens,
        bos_token_id=int(model.base_thinker.config.bos_token_id),
    )
    train_spans = [span for span in spans if span.split is Split.TRAIN]
    validation_spans = [span for span in spans if span.split is Split.VALIDATION]
    train_values = [timeline for span in train_spans for timeline in span.normal]
    train_values.extend(span.interrupted for span in train_spans)
    validation_values = [
        timeline for span in validation_spans for timeline in span.normal
    ]
    validation_values.extend(span.interrupted for span in validation_spans)
    collator = DuplexCollator(
        audio_processor=processor,
        tokenizer=processor.tokenizer,
        control_tokens=model.control_tokens,
        thinker_bos_token_id=model.base_thinker.config.bos_token_id,
        frame_rate_hz=25,
        interruption_probability=0.0,
        augmentation_seed=int(config["training"]["interruption_seed"]),
    )
    trainer, audit = make_trainer(
        config,
        model,
        processor,
        targets,
        _ListDataset(train_values),
        None,
        collator,
    )
    trainer.train()
    if not audit.nonzero_gradient_names:
        raise RuntimeError("No LoRA parameter received a nonzero gradient.")
    assert_optimizer_scope(trainer)
    trainer.save_model(Path(config["training"]["output_dir"]) / "final")
    teacher = _teacher_forced_validation(model, validation_values, collator)
    free, trace_summaries = _free_streaming_validation(
        model, processor, validation_spans, config
    )
    runtime = time.perf_counter() - started
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    teacher_counts = teacher["events"]["prediction_counts"]
    free_counts = free["events"]["prediction_counts"]
    failures = []
    for name, metric in (
        ("validation weighted loss", teacher["weighted_loss"]),
        ("validation text loss", teacher["text_token_loss"]),
    ):
        if metric["value"] is None or not math.isfinite(metric["value"]):
            failures.append(f"{name} is not finite")
    if teacher_counts["START"] + free_counts["START"] == 0:
        failures.append("validation produced no START predictions")
    if teacher_counts["STOP"] + free_counts["STOP"] == 0:
        failures.append("validation produced no STOP predictions")
    free_total = sum(free_counts.values())
    if free_total and free_counts["IDLE"] / free_total >= 0.98:
        failures.append("free streaming collapsed to near-total IDLE")
    if free["invalid_transition_rate_after_masking"]["invalid_count"] != 0:
        failures.append("grammar-masked streaming retained invalid transitions")
    if free["interruption_STOP"]["recalled_count"] == 0:
        failures.append("no synthetic interruption produced a correct STOP")
    if len(trace_summaries) < 20:
        failures.append("fewer than 20 validation traces were decoded")
    report = {
        "schema_version": 1,
        "small_data_gate": "PASS" if not failures else "FAIL",
        "failures": failures,
        "config": config,
        "subset_manifest": {
            "path": config["diagnostic"]["subset_manifest"],
            "sha256": hashlib.sha256(
                Path(config["diagnostic"]["subset_manifest"]).read_bytes()
            ).hexdigest(),
            "selection": subset_manifest["selection"],
            "complete_target_histogram": subset_manifest["complete_target_histogram"],
            "conversation_ids": [item["conversation_id"] for item in subset_manifest["items"]],
        },
        "optimizer_steps": trainer.state.global_step,
        "runtime_seconds": runtime,
        "peak_vram_allocated_bytes": peak_allocated,
        "peak_vram_reserved_bytes": peak_reserved,
        "teacher_forced": {
            key: value
            for key, value in teacher.items()
            if key not in {"labels", "predictions"}
        },
        "free_streaming": free,
        "decoded_validation_traces": trace_summaries,
        "trainable_lora_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "nonzero_lora_gradient_parameter_count": len(audit.nonzero_gradient_names),
    }
    _json_dump(Path(config["diagnostic"]["report"]), report)
    print(f"SMALL_DATA_GATE={report['small_data_gate']}")
    if failures:
        print("gate failures:", "; ".join(failures))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    config = load_training_config(args.config)
    if "diagnostic" not in config:
        raise SystemExit("Session 11 requires a diagnostic config section.")
    if args.prepare_only:
        prepare_subset(config)
        return
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("Session 11 requires a CUDA GPU with BF16 support.")
    run_diagnostic(config)


if __name__ == "__main__":
    main()
