"""Full-Duplex-Bench v1.0 text-timeline adaptation.

This evaluates text/control timelines only. It never creates speech, runs VAD,
or claims an official speech-output benchmark score.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from duplex.streaming import (
    EventTraceRecord,
    StreamingResult,
    _assign_exact_decoded_deltas,
    group_lexical_events_into_words,
)

ADAPTATION_NAME = "Full-Duplex-Bench v1.0 text-timeline adaptation"
IS_OFFICIAL_SPEECH_OUTPUT_SCORE = False
UPSTREAM_REPOSITORY = "https://github.com/DanielLin94144/Full-Duplex-Bench"
UPSTREAM_REVISION = "3e799c45a045256f47d5f1c9cda90157e2d2ec9e"
MODEL_ID = "Qwen/Qwen2.5-Omni-3B"
BASE_REVISION = "f75b40e3da2003cdd6e1829b1f420ca70797c34e"

TASK_DIRECTORIES = {
    "pause_handling": ("candor_pause_handling", "synthetic_pause_handling"),
    "backchannel": ("icc_backchannel",),
    "smooth_turn_taking": ("candor_turn_taking",),
    "user_interruption": ("synthetic_user_interruption",),
}


@dataclass(frozen=True)
class BenchmarkConfig:
    """Named thresholds from Full-Duplex-Bench v1.0.

    The paper defines backchannels as under one second and at most two words;
    the released evaluator also has a three-second long-speech takeover guard.
    ICC timing distributions use 200 ms bins. See arXiv:2503.04721 Sec. III
    and the pinned v1.0 evaluation scripts.
    """

    silence_max_words: int = 0
    backchannel_max_words: int = 2
    backchannel_max_duration_s: float = 1.0
    takeover_long_response_s: float = 3.0
    jsd_bin_width_s: float = 0.2
    distribution_epsilon: float = 1e-10

    def validate(self) -> None:
        required = (
            self.silence_max_words == 0,
            self.backchannel_max_words == 2,
            self.backchannel_max_duration_s == 1.0,
            self.takeover_long_response_s == 3.0,
            self.jsd_bin_width_s == 0.2,
            self.distribution_epsilon > 0,
        )
        if not all(required):
            raise ValueError("Full-Duplex-Bench v1.0 thresholds may not be changed.")


@dataclass(frozen=True)
class BenchmarkSample:
    sample_id: str
    task: str
    directory: Path
    input_path: Path
    input_duration_s: float
    annotation: Mapping[str, Any] | None


@dataclass(frozen=True)
class GeneratedTimeline:
    tokenizer: object
    trace: tuple[EventTraceRecord, ...]
    input_duration_s: float
    stop_reason: str


class TimelineBackend(Protocol):
    def generate(self, sample: BenchmarkSample) -> GeneratedTimeline: ...


@dataclass(frozen=True)
class RunSettings:
    model_mode: str
    data_dir: Path
    task: str
    adapter: Path | None = None
    max_new_tokens: int = 256
    max_silent_chunks: int = 4
    allow_download: bool = False

    def validate(self) -> None:
        if self.model_mode not in {"base", "duplex"}:
            raise ValueError("model_mode must be base or duplex.")
        if self.task not in TASK_DIRECTORIES:
            raise ValueError(f"Unsupported v1.0 task {self.task!r}.")
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"Benchmark data directory not found: {self.data_dir}")
        if self.model_mode == "duplex" and self.adapter is None:
            raise ValueError("Duplex mode requires --adapter or --checkpoint.")
        if self.model_mode == "base" and self.adapter is not None:
            raise ValueError("Base mode must use the untouched checkpoint.")
        if self.max_new_tokens <= 0 or self.max_silent_chunks < 0:
            raise ValueError("Generation limits are invalid.")


def _wav_duration(path: Path) -> float:
    import soundfile as sf

    info = sf.info(path)
    if info.frames <= 0 or info.samplerate <= 0:
        raise ValueError(f"Invalid or empty input WAV: {path}")
    return info.frames / info.samplerate


def _annotation(sample_dir: Path, task: str) -> Mapping[str, Any] | None:
    name = {"pause_handling": "pause.json", "smooth_turn_taking": "turn_taking.json",
            "user_interruption": "interrupt.json"}.get(task)
    if name is None:
        return None
    path = sample_dir / name
    if not path.is_file():
        raise FileNotFoundError(f"Required v1.0 annotation not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value or not isinstance(value[0], dict):
        raise ValueError(f"Invalid v1.0 annotation in {path}.")
    return value[0]


def load_samples(data_dir: Path, task: str) -> list[BenchmarkSample]:
    """Load metadata from a caller-owned v1.0 directory, never into git."""

    root = data_dir.resolve()
    if task not in TASK_DIRECTORIES:
        raise ValueError(f"Unsupported v1.0 task {task!r}.")
    if not root.is_dir():
        raise FileNotFoundError(f"Benchmark data directory not found: {root}")
    if (root / "input.wav").is_file():
        sample_dirs = [root]
    else:
        subset_roots = [root / name for name in TASK_DIRECTORIES[task] if (root / name).is_dir()]
        subset_roots = subset_roots or [root]
        sample_dirs = sorted(directory for subset in subset_roots for directory in subset.iterdir()
                             if directory.is_dir() and (directory / "input.wav").is_file())
    samples = []
    for directory in sample_dirs:
        relative = directory.relative_to(root) if directory != root else Path(directory.name)
        samples.append(BenchmarkSample(
            sample_id=relative.as_posix(), task=task, directory=directory,
            input_path=directory / "input.wav", input_duration_s=_wav_duration(directory / "input.wav"),
            annotation=_annotation(directory, task)))
    return samples


def _time(record: EventTraceRecord | None) -> dict[str, float] | None:
    return None if record is None else {
        "logical_time_s": float(record.audio_time_s),
        "available_time_s": float(record.available_time_s),
    }


def _word_chunks(records: Sequence[EventTraceRecord], tokenizer: object) -> tuple[str, list[dict[str, Any]]]:
    spans = group_lexical_events_into_words(records, tokenizer)
    chunks = []
    for span in spans:
        if not span.token_timings:
            raise RuntimeError(f"Word {span.text!r} has no contributing lexical token.")
        chunks.append({
            "text": span.text, "character_start": span.character_start,
            "character_end": span.character_end,
            "logical_start_s": min(t.audio_time_s for t in span.token_timings),
            "logical_end_s": max(t.audio_time_s for t in span.token_timings),
            # The word is causally available only after its final contributing token.
            "available_time_s": max(t.available_time_s for t in span.token_timings),
            "contributing_trace_indices": [t.trace_index for t in span.token_timings],
            "contributing_token_ids": [t.token_id for t in span.token_timings],
        })
    exact = "".join(record.decoded_delta for record in records)
    return exact, chunks


def _segment_record(start: tuple[int, EventTraceRecord], stop: tuple[int, EventTraceRecord] | None,
                    records: Sequence[EventTraceRecord], words: Sequence[Mapping[str, Any]],
                    config: BenchmarkConfig) -> dict[str, Any]:
    start_index, start_record = start
    stop_index = stop[0] if stop is not None else len(records)
    selected = [dict(word) for word in words if word["contributing_trace_indices"]
                and start_index < min(word["contributing_trace_indices"]) < stop_index]
    stop_record = stop[1] if stop is not None else None
    end_available = (stop_record.available_time_s if stop_record is not None else
                     selected[-1]["available_time_s"] if selected else start_record.available_time_s)
    duration = max(0.0, float(end_available) - float(start_record.available_time_s))
    word_count = len(selected)
    silence = word_count <= config.silence_max_words
    backchannel = (not silence and word_count <= config.backchannel_max_words
                   and duration < config.backchannel_max_duration_s)
    lexical_indices = [i for i in range(start_index + 1, stop_index) if records[i].event_type == "TEXT"]
    return {
        "start": _time(start_record), "stop": _time(stop_record),
        "first_word": selected[0] if selected else None,
        "last_word": selected[-1] if selected else None,
        "word_chunks": selected,
        "exact_text": "".join(records[i].decoded_delta for i in lexical_indices),
        "word_count": word_count,
        "response_interval_s": [float(start_record.available_time_s), float(end_available)],
        "duration_s": duration,
        "classification": "silence" if silence else "backchannel" if backchannel else "takeover",
    }


def _segments(records: Sequence[EventTraceRecord], words: Sequence[Mapping[str, Any]],
              config: BenchmarkConfig) -> list[dict[str, Any]]:
    result, active = [], None
    for index, record in enumerate(records):
        if record.event_type == "START":
            if active is not None:
                raise ValueError("Trace contains START while a response is active.")
            active = (index, record)
        elif record.event_type == "STOP":
            if active is None:
                raise ValueError("Trace contains STOP while no response is active.")
            result.append(_segment_record(active, (index, record), records, words, config))
            active = None
    if active is not None:
        result.append(_segment_record(active, None, records, words, config))
    return result


def _annotation_values(sample: BenchmarkSample) -> tuple[float | None, str | None, str | None]:
    annotation = sample.annotation
    if sample.task == "smooth_turn_taking":
        assert annotation is not None
        timestamp = annotation.get("timestamp")
        if not isinstance(timestamp, list) or not timestamp:
            raise ValueError(f"Invalid turn_taking timestamp for {sample.sample_id}.")
        return float(timestamp[0]), None, None
    if sample.task == "user_interruption":
        assert annotation is not None
        timestamp = annotation.get("timestamp")
        if not isinstance(timestamp, list) or len(timestamp) != 2:
            raise ValueError(f"Invalid interruption timestamp for {sample.sample_id}.")
        return float(timestamp[1]), str(annotation.get("context", "")), str(annotation.get("interrupt", ""))
    return None, None, None


def _interpolate(values: Sequence[float], size: int) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    if source.ndim != 1 or source.size == 0 or not np.isfinite(source).all() or (source < 0).any():
        raise ValueError("Ground-truth distribution must be finite and non-negative.")
    if source.size == 1:
        return np.full(size, source[0], dtype=np.float64)
    return np.interp(np.linspace(0.0, 1.0, size), np.linspace(0.0, 1.0, source.size), source)


def response_interval_distribution(intervals: Sequence[Sequence[float]], duration_s: float,
                                   config: BenchmarkConfig = BenchmarkConfig()) -> list[float] | None:
    """Build the released inclusive 200 ms histogram from text intervals."""

    config.validate()
    if not intervals:
        return None
    size = int(duration_s / config.jsd_bin_width_s) + 1
    counts = np.zeros(size, dtype=np.float64)
    for interval in intervals:
        if len(interval) != 2:
            raise ValueError("Response intervals require start and end.")
        start, end = map(float, interval)
        if start < 0 or end < start:
            raise ValueError(f"Invalid response interval {interval!r}.")
        first, last = int(start / config.jsd_bin_width_s), int(end / config.jsd_bin_width_s)
        for index in range(first, min(last, size - 1) + 1):
            counts[index] += 1
    counts += config.distribution_epsilon
    return (counts / counts.sum()).tolist()


def jensen_shannon_distance(predicted: Sequence[float], expected: Sequence[float]) -> float:
    """Match scipy.spatial.distance.jensenshannon used by upstream."""

    left = np.asarray(predicted, dtype=np.float64)
    right = _interpolate(expected, left.size)
    left, right = left / left.sum(), right / right.sum()
    middle = (left + right) / 2
    left_terms = np.zeros_like(left)
    right_terms = np.zeros_like(right)
    left_nonzero, right_nonzero = left > 0, right > 0
    left_terms[left_nonzero] = left[left_nonzero] * np.log(left[left_nonzero] / middle[left_nonzero])
    right_terms[right_nonzero] = right[right_nonzero] * np.log(right[right_nonzero] / middle[right_nonzero])
    return float(math.sqrt(max(0.0, 0.5 * (left_terms.sum() + right_terms.sum()))))


def evaluate_timeline(sample: BenchmarkSample, generated: GeneratedTimeline, *, model_mode: str,
                      config: BenchmarkConfig = BenchmarkConfig(),
                      ground_truth_distribution: Sequence[float] | None = None) -> dict[str, Any]:
    """Convert one lexical event trace to the labeled v1.0 schema."""

    config.validate()
    if model_mode not in {"base", "duplex"}:
        raise ValueError("model_mode must be base or duplex.")
    records = list(generated.trace)
    if records:
        records, exact_text = _assign_exact_decoded_deltas(records, generated.tokenizer)
        exact_text, words = _word_chunks(records, generated.tokenizer)
    else:
        exact_text, words = "", []
    segments = _segments(records, words, config)
    event_end, context, interruption = _annotation_values(sample)
    takeovers = [row for row in segments if row["classification"] == "takeover"]
    if sample.task == "user_interruption":
        takeovers = [
            row for row in takeovers
            if any(float(word["available_time_s"]) >= event_end for word in row["word_chunks"])
        ]
    backchannels = [row for row in segments if row["classification"] == "backchannel"]
    takeover = bool(takeovers)
    first_takeover = takeovers[0] if takeovers else None
    first_word = first_takeover["first_word"] if first_takeover else None
    if sample.task == "user_interruption" and first_takeover is not None:
        first_word = next(
            word for word in first_takeover["word_chunks"]
            if float(word["available_time_s"]) >= event_end
        )
    logical_latency = causal_latency = None
    if event_end is not None and first_word is not None:
        logical_latency = float(first_word["logical_start_s"]) - event_end
        causal_latency = max(0.0, float(first_word["available_time_s"]) - event_end)

    metrics: dict[str, Any] = {"takeover": takeover}
    judge_record = None
    if sample.task == "backchannel":
        intervals = [row["response_interval_s"] for row in backchannels]
        horizon = max([sample.input_duration_s] + [float(interval[1]) for interval in intervals])
        distribution = response_interval_distribution(intervals, horizon, config)
        jsd = (jensen_shannon_distance(distribution, ground_truth_distribution)
               if not takeover and distribution is not None and ground_truth_distribution is not None else None)
        metrics.update(backchannel_count=len(backchannels),
                       backchannel_frequency_per_second=(len(backchannels) / sample.input_duration_s
                                                         if not takeover else None),
                       timing_distribution_200ms=distribution if not takeover else None,
                       timing_jsd=jsd)
    elif sample.task in {"smooth_turn_taking", "user_interruption"}:
        metrics.update(annotated_user_event_end_s=event_end, first_word_latency_s=causal_latency,
                       logical_first_word_latency_s=logical_latency)
    if sample.task == "user_interruption":
        response = ""
        if first_takeover is not None:
            eligible = [word for word in first_takeover["word_chunks"]
                        if float(word["available_time_s"]) >= event_end]
            segment_end = first_takeover["last_word"]["character_end"]
            response = exact_text[eligible[0]["character_start"]:segment_end]
        judge_record = {
            "label": ADAPTATION_NAME, "sample_id": sample.sample_id,
            "contextual_user_turn": context, "user_interrupting_turn": interruption,
            "assistant_response": response,
            "rating_0_to_5": None, "status": "not_called",
            "instruction": "Rate response relevance/adaptation to the interrupting turn from 0 to 5.",
        }
    return {
        "label": ADAPTATION_NAME, "schema": "full_duplex_bench_v1_text_timeline_result_v1",
        "official_speech_output_score": False, "sample_id": sample.sample_id,
        "task": sample.task, "model_mode": model_mode,
        "input_duration_s": sample.input_duration_s, "stop_reason": generated.stop_reason,
        "decoded_text_exact": exact_text,
        "timeline": {
            "events": [record.as_dict() for record in records], "word_chunks": words,
            "first_word": words[0] if words else None, "last_word": words[-1] if words else None,
            "start_times": [_time(row) for row in records if row.event_type == "START"],
            "stop_times": [_time(row) for row in records if row.event_type == "STOP"],
            "response_segments": segments,
        },
        "metrics": metrics, "judge_ready_relevance": judge_record,
        "upstream": {"repository": UPSTREAM_REPOSITORY, "revision": UPSTREAM_REVISION},
        "thresholds": asdict(config),
    }


def _mean_with_coverage(records: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    values = [row["metrics"].get(key) for row in records]
    available = [float(value) for value in values if value is not None]
    total = len(values)
    return {"value": sum(available) / len(available) if available else None,
            "coverage": {"available": len(available), "total": total,
                         "rate": len(available) / total if total else None}}


def summarize(records: Sequence[Mapping[str, Any]], task: str) -> dict[str, Any]:
    if any(row.get("label") != ADAPTATION_NAME or row.get("task") != task for row in records):
        raise ValueError("Cannot summarize mixed or unlabeled v1.0 records.")
    result = {"label": ADAPTATION_NAME, "schema": "full_duplex_bench_v1_text_timeline_summary_v1",
              "task": task, "sample_count": len(records),
              "takeover_rate": _mean_with_coverage(records, "takeover")}
    if task == "backchannel":
        result.update(backchannel_frequency_per_second=_mean_with_coverage(
            records, "backchannel_frequency_per_second"), timing_jsd=_mean_with_coverage(records, "timing_jsd"))
    elif task in {"smooth_turn_taking", "user_interruption"}:
        result["first_word_latency_s"] = _mean_with_coverage(records, "first_word_latency_s")
    if task == "user_interruption":
        result["relevance_rating_0_to_5"] = {
            "value": None, "coverage": {"available": 0, "total": len(records),
                                        "rate": 0.0 if records else None}, "status": "not_called"}
    return result


def _simple_record(sample_id: str, index: int, kind: str, token_id: int, when: float) -> EventTraceRecord:
    return EventTraceRecord(
        sample_id=sample_id, chunk_index=0, frame_index=index, audio_time_s=when,
        available_time_s=when, compute_ms=0.0, event_id=token_id, event_type=kind,
        decoded_delta="", state_before="inactive" if kind == "START" else "active",
        state_after="inactive" if kind == "STOP" else "active",
        grammar_mask_changed_raw_argmax=False, raw_argmax_id=token_id,
        chunk_end_time_s=when, silent_tail=False)


class BaseTimelineBackend:
    """Untouched Thinker streamed only after the complete input is available."""

    def __init__(self, settings: RunSettings) -> None:
        import torch
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
        common = {"revision": BASE_REVISION, "local_files_only": not settings.allow_download}
        self.torch, self.settings = torch, settings
        self.processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID, **common)
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            MODEL_ID, **common, torch_dtype=torch.bfloat16, device_map={"": "cuda:0"},
            attn_implementation="sdpa", low_cpu_mem_usage=True)
        self.model.disable_talker()
        self.model.eval()

    def generate(self, sample: BenchmarkSample) -> GeneratedTimeline:
        import soundfile as sf
        values, rate = sf.read(sample.input_path, dtype="float32", always_2d=False)
        if values.ndim != 1 or rate != 16_000:
            raise ValueError("Full-Duplex-Bench input must be mono 16 kHz audio.")
        messages = [{"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                    {"role": "user", "content": [{"type": "audio", "audio": values}]}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=prompt, audio=[values], return_tensors="pt", padding=True,
                                use_audio_in_video=False).to(self.model.device).to(self.model.dtype)
        prompt_length = inputs.input_ids.shape[1]

        class TimingStreamer:
            def __init__(inner) -> None:
                inner.first, inner.times = True, []

            def put(inner, value: object) -> None:
                if inner.first:
                    inner.first = False
                    return
                inner.times.extend([time.perf_counter()] * int(value.numel()))

            def end(inner) -> None:
                return None

        streamer, started = TimingStreamer(), time.perf_counter()
        with self.torch.inference_mode():
            output = self.model.thinker.generate(**inputs, max_new_tokens=self.settings.max_new_tokens,
                                                 do_sample=False, streamer=streamer)[0, prompt_length:].tolist()
        finished = time.perf_counter()
        eos = self.model.thinker.generation_config.eos_token_id
        eos_ids = {int(eos)} if isinstance(eos, int) else {int(value) for value in eos or []}
        lexical, stop_available, saw_eos = [], sample.input_duration_s + finished - started, False
        for index, token_id in enumerate(output):
            available = sample.input_duration_s + streamer.times[index] - started
            if token_id in eos_ids:
                stop_available = available
                saw_eos = True
                break
            lexical.append((int(token_id), available))
        records: list[EventTraceRecord] = []
        if lexical:
            records.append(_simple_record(sample.sample_id, 0, "START", 0, sample.input_duration_s))
            for index, (token_id, available) in enumerate(lexical, 1):
                records.append(_simple_record(sample.sample_id, index, "TEXT", token_id, available))
            if saw_eos:
                records.append(_simple_record(sample.sample_id, len(records), "STOP", 0, stop_available))
        reason = "EOS" if saw_eos else "max_new_tokens" if output else "no_response"
        return GeneratedTimeline(self.processor.tokenizer, tuple(records), sample.input_duration_s, reason)


class DuplexTimelineBackend:
    """Adapter-active fixed-two-second streamer returning its native trace."""

    def __init__(self, settings: RunSettings) -> None:
        from duplex.training import _load_adapter_weights, build_training_model, load_training_config
        assert settings.adapter is not None
        config = load_training_config(settings.adapter / "training_config.yaml")
        config["model"]["local_files_only"] = not settings.allow_download
        self.model, self.processor, _ = build_training_model(config)
        _load_adapter_weights(self.model, settings.adapter)
        self.model.eval()
        self.settings = settings

    def generate(self, sample: BenchmarkSample) -> GeneratedTimeline:
        import soundfile as sf
        from duplex.streaming import QwenDuplexStreamer
        values, rate = sf.read(sample.input_path, dtype="float32", always_2d=False)
        if values.ndim != 1 or rate != 16_000:
            raise ValueError("Full-Duplex-Bench input must be mono 16 kHz audio.")
        messages = [{"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                    {"role": "user", "content": [{"type": "text", "text": ""}]}]
        ids = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                 return_tensors="pt")
        result: StreamingResult = QwenDuplexStreamer(
            self.model, self.processor, max_silent_chunks=self.settings.max_silent_chunks,
            sample=False).run(values, sample_id=sample.sample_id,
                              context_token_ids=ids[0].tolist(), sample_rate_hz=rate)
        return GeneratedTimeline(self.processor.tokenizer, tuple(result.trace),
                                 result.input_duration_s, result.stop_reason)


def load_backend(settings: RunSettings) -> TimelineBackend:
    settings.validate()
    return BaseTimelineBackend(settings) if settings.model_mode == "base" else DuplexTimelineBackend(settings)


def _load_ground_truth(path: Path | None) -> Mapping[str, Sequence[float]]:
    if path is None:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Backchannel ground truth must be a JSON object keyed by sample ID.")
    return value


def run_benchmark(settings: RunSettings, *, limit: int | None = None, start_index: int = 0,
                  ground_truth_path: Path | None = None,
                  sample_loader: Callable[[Path, str], list[BenchmarkSample]] = load_samples,
                  backend_loader: Callable[[RunSettings], TimelineBackend] = load_backend) -> dict[str, Any]:
    settings.validate()
    if start_index < 0 or (limit is not None and limit < 0):
        raise ValueError("start_index and limit must be non-negative.")
    samples = sample_loader(settings.data_dir, settings.task)
    selected = samples[start_index:] if limit is None else samples[start_index:start_index + limit]
    backend = backend_loader(settings) if selected else None
    ground_truth, records = _load_ground_truth(ground_truth_path), []
    for sample in selected:
        assert backend is not None
        gt = ground_truth.get(sample.sample_id) or ground_truth.get(sample.directory.name)
        records.append(evaluate_timeline(sample, backend.generate(sample), model_mode=settings.model_mode,
                                         ground_truth_distribution=gt))
    return {
        "label": ADAPTATION_NAME, "schema": "full_duplex_bench_v1_text_timeline_run_v1",
        "model_mode": settings.model_mode, "data_directory": str(settings.data_dir.resolve()),
        "upstream": {"repository": UPSTREAM_REPOSITORY, "revision": UPSTREAM_REVISION},
        "records": records, "summary": summarize(records, settings.task)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-mode", required=True, choices=["base", "duplex"])
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--task", required=True, choices=sorted(TASK_DIRECTORIES))
    adapter = parser.add_mutually_exclusive_group()
    adapter.add_argument("--adapter", type=Path)
    adapter.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ground-truth-distribution", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-silent-chunks", type=int, default=4)
    parser.add_argument("--allow-download", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    settings = RunSettings(model_mode=args.model_mode, data_dir=args.data_dir, task=args.task,
                           adapter=args.adapter or args.checkpoint, max_new_tokens=args.max_new_tokens,
                           max_silent_chunks=args.max_silent_chunks, allow_download=args.allow_download)
    result = run_benchmark(settings, limit=args.limit, start_index=args.start_index,
                           ground_truth_path=args.ground_truth_distribution)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
