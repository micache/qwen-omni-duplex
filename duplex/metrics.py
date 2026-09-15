"""Internal text-timeline validation metrics.

The metrics in this module are diagnostic adaptations for the repository's
event timeline. They are not official speech-output benchmark scores.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any

from .timeline import EventKind, ResponseState


FULL_DUPLEX_BENCH_VARIANT = "text-timeline adaptation"
OFFICIAL_SPEECH_OUTPUT_SCORE = False
IGNORE_LABEL = -100
EVENT_NAMES = (
    EventKind.TEXT.value,
    EventKind.IDLE.value,
    EventKind.START.value,
    EventKind.STOP.value,
)


@dataclass(frozen=True)
class _Turn:
    start: int
    stop: int | None


def _metric(value: float | None, denominator: int) -> dict[str, float | int | None]:
    return {"value": value, "denominator": denominator}


def _mean(values: Sequence[float]) -> dict[str, float | int | None]:
    return _metric(sum(values) / len(values) if values else None, len(values))


def _as_rows(values: Any, *, name: str) -> list[list[int]]:
    if hasattr(values, "detach"):
        values = values.detach().cpu().tolist()
    elif hasattr(values, "tolist") and not isinstance(values, (list, tuple)):
        values = values.tolist()
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a one- or two-dimensional sequence.")
    values = list(values)
    if not values:
        return []
    if isinstance(values[0], Sequence) and not isinstance(values[0], (str, bytes)):
        rows = [list(row) for row in values]
    else:
        rows = [values]
    for row in rows:
        if any(isinstance(value, bool) or not isinstance(value, Integral) for value in row):
            raise TypeError(f"{name} must contain integer token IDs.")
    return [[int(value) for value in row] for row in rows]


def _control_value(control_tokens: object, name: str) -> int:
    if isinstance(control_tokens, Mapping):
        value = control_tokens.get(name, control_tokens.get(name.upper()))
    else:
        value = getattr(control_tokens, name, None)
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"control_tokens.{name} must be an integer token ID.")
    return int(value)


def _control_ids(control_tokens: object) -> dict[str, int]:
    controls = {
        EventKind.IDLE.value: _control_value(control_tokens, "idle"),
        EventKind.START.value: _control_value(control_tokens, "start"),
        EventKind.STOP.value: _control_value(control_tokens, "stop"),
    }
    if len(set(controls.values())) != len(controls):
        raise ValueError("IDLE, START, and STOP token IDs must be distinct.")
    return controls


def _kind(token_id: int, controls: Mapping[str, int]) -> str:
    for name, control_id in controls.items():
        if token_id == control_id:
            return name
    return EventKind.TEXT.value


def _paired_rows(
    labels: Any,
    predictions: Any,
    *,
    ignore_index: int,
) -> list[tuple[list[int], list[int]]]:
    label_rows = _as_rows(labels, name="labels")
    prediction_rows = _as_rows(predictions, name="predictions")
    if len(label_rows) != len(prediction_rows):
        raise ValueError("labels and predictions must have the same batch size.")
    result = []
    for index, (label_row, prediction_row) in enumerate(
        zip(label_rows, prediction_rows)
    ):
        if len(label_row) != len(prediction_row):
            raise ValueError(
                f"labels and predictions row {index} have different lengths."
            )
        keep = [frame for frame, label in enumerate(label_row) if label != ignore_index]
        result.append(
            ([label_row[frame] for frame in keep], [prediction_row[frame] for frame in keep])
        )
    return result


def event_classification_metrics(
    labels: Any,
    predictions: Any,
    *,
    control_tokens: object,
    ignore_index: int = IGNORE_LABEL,
) -> dict[str, Any]:
    """Count event classes and compute control precision/recall/F1.

    Predictions at ignored label positions are excluded from every count.
    """

    controls = _control_ids(control_tokens)
    label_counts = {name: 0 for name in EVENT_NAMES}
    prediction_counts = {name: 0 for name in EVENT_NAMES}
    true_positives = {name: 0 for name in controls}
    for label_row, prediction_row in _paired_rows(
        labels, predictions, ignore_index=ignore_index
    ):
        for label, prediction in zip(label_row, prediction_row):
            label_kind = _kind(label, controls)
            prediction_kind = _kind(prediction, controls)
            label_counts[label_kind] += 1
            prediction_counts[prediction_kind] += 1
            if label_kind == prediction_kind and label_kind in true_positives:
                true_positives[label_kind] += 1

    control_metrics: dict[str, Any] = {}
    for name in (EventKind.IDLE.value, EventKind.START.value, EventKind.STOP.value):
        tp = true_positives[name]
        predicted = prediction_counts[name]
        labeled = label_counts[name]
        f1_denominator = 2 * tp + (predicted - tp) + (labeled - tp)
        control_metrics[name] = {
            "true_positive_count": tp,
            "precision": _metric(tp / predicted if predicted else None, predicted),
            "recall": _metric(tp / labeled if labeled else None, labeled),
            "f1": _metric(
                2 * tp / f1_denominator if f1_denominator else None,
                f1_denominator,
            ),
        }
    return {
        "label_counts": label_counts,
        "prediction_counts": prediction_counts,
        "control": control_metrics,
    }


def invalid_state_transition_rate(
    predictions: Any,
    *,
    control_tokens: object,
    labels: Any | None = None,
    ignore_index: int = IGNORE_LABEL,
) -> dict[str, float | int | None]:
    """Measure raw predictions rejected by the inactive/active grammar.

    State changes only after legal START or STOP events. When labels are
    supplied, their ignored positions mask prediction padding.
    """

    controls = _control_ids(control_tokens)
    prediction_rows = _as_rows(predictions, name="predictions")
    if labels is None:
        paired = [([0] * len(row), row) for row in prediction_rows]
    else:
        paired = _paired_rows(labels, predictions, ignore_index=ignore_index)

    invalid = 0
    total = 0
    for _, prediction_row in paired:
        state = ResponseState.INACTIVE
        for prediction in prediction_row:
            total += 1
            kind = _kind(prediction, controls)
            legal = (
                kind in {EventKind.IDLE.value, EventKind.START.value}
                if state is ResponseState.INACTIVE
                else kind != EventKind.START.value
            )
            if not legal:
                invalid += 1
                continue
            if kind == EventKind.START.value:
                state = ResponseState.ACTIVE
            elif kind == EventKind.STOP.value:
                state = ResponseState.INACTIVE
    return {
        "value": invalid / total if total else None,
        "denominator": total,
        "invalid_count": invalid,
    }


def _turns(token_ids: Sequence[int], controls: Mapping[str, int]) -> list[_Turn]:
    state = ResponseState.INACTIVE
    turns: list[_Turn] = []
    current_start: int | None = None
    for frame, token_id in enumerate(token_ids):
        kind = _kind(token_id, controls)
        if state is ResponseState.INACTIVE and kind == EventKind.START.value:
            current_start = frame
            state = ResponseState.ACTIVE
        elif state is ResponseState.ACTIVE and kind == EventKind.STOP.value:
            assert current_start is not None
            turns.append(_Turn(current_start, frame))
            current_start = None
            state = ResponseState.INACTIVE
    if current_start is not None:
        turns.append(_Turn(current_start, None))
    return turns


def boundary_error_metrics(
    labels: Any,
    predictions: Any,
    *,
    control_tokens: object,
    frame_rate_hz: int = 25,
    ignore_index: int = IGNORE_LABEL,
) -> dict[str, Any]:
    """Compute absolute START/STOP error for ordinally matched response turns."""

    if (
        isinstance(frame_rate_hz, bool)
        or not isinstance(frame_rate_hz, int)
        or frame_rate_hz <= 0
    ):
        raise ValueError("frame_rate_hz must be a positive integer.")
    controls = _control_ids(control_tokens)
    start_errors: list[float] = []
    stop_errors: list[float] = []
    target_turn_count = 0
    predicted_turn_count = 0
    matched_turn_count = 0
    for label_row, prediction_row in _paired_rows(
        labels, predictions, ignore_index=ignore_index
    ):
        target_turns = [
            turn for turn in _turns(label_row, controls) if turn.stop is not None
        ]
        predicted_turns = _turns(prediction_row, controls)
        target_turn_count += len(target_turns)
        predicted_turn_count += len(predicted_turns)
        for target, prediction in zip(target_turns, predicted_turns):
            matched_turn_count += 1
            start_errors.append(float(abs(prediction.start - target.start)))
            if prediction.stop is not None:
                assert target.stop is not None
                stop_errors.append(float(abs(prediction.stop - target.stop)))

    start_frames = _mean(start_errors)
    stop_frames = _mean(stop_errors)
    return {
        "target_response_turn_count": target_turn_count,
        "predicted_response_turn_count": predicted_turn_count,
        "matched_response_turn_count": matched_turn_count,
        "START": {
            "absolute_error_frames": start_frames,
            "absolute_error_seconds": _metric(
                None
                if start_frames["value"] is None
                else start_frames["value"] / frame_rate_hz,
                int(start_frames["denominator"]),
            ),
        },
        "STOP": {
            "absolute_error_frames": stop_frames,
            "absolute_error_seconds": _metric(
                None
                if stop_frames["value"] is None
                else stop_frames["value"] / frame_rate_hz,
                int(stop_frames["denominator"]),
            ),
        },
    }


def interruption_stop_metrics(
    predictions: Any,
    interruption_onsets: Sequence[int | None],
    *,
    control_tokens: object,
    labels: Any | None = None,
    frame_rate_hz: int = 25,
    ignore_index: int = IGNORE_LABEL,
) -> dict[str, Any]:
    """Compute synthetic-interruption STOP recall and nonnegative latency."""

    if (
        isinstance(frame_rate_hz, bool)
        or not isinstance(frame_rate_hz, int)
        or frame_rate_hz <= 0
    ):
        raise ValueError("frame_rate_hz must be a positive integer.")
    controls = _control_ids(control_tokens)
    prediction_rows = _as_rows(predictions, name="predictions")
    if labels is not None:
        paired = _paired_rows(labels, predictions, ignore_index=ignore_index)
        prediction_rows = [prediction_row for _, prediction_row in paired]
    if len(prediction_rows) != len(interruption_onsets):
        raise ValueError("interruption_onsets must have one entry per prediction row.")

    interruption_count = 0
    recalled = 0
    latencies: list[float] = []
    for row, onset in zip(prediction_rows, interruption_onsets):
        if onset is None:
            continue
        if (
            isinstance(onset, bool)
            or not isinstance(onset, int)
            or not 0 <= onset < len(row)
        ):
            raise ValueError("Each interruption onset must be null or a valid frame index.")
        interruption_count += 1
        stop = next(
            (
                frame
                for frame in range(onset, len(row))
                if row[frame] == controls[EventKind.STOP.value]
            ),
            None,
        )
        if stop is not None:
            recalled += 1
            latencies.append(float(stop - onset))
    latency_frames = _mean(latencies)
    return {
        "recall": _metric(
            recalled / interruption_count if interruption_count else None,
            interruption_count,
        ),
        "recalled_count": recalled,
        "latency_frames": latency_frames,
        "latency_seconds": _metric(
            None
            if latency_frames["value"] is None
            else latency_frames["value"] / frame_rate_hz,
            int(latency_frames["denominator"]),
        ),
    }


def teacher_forced_text_metrics(
    logits: Any,
    labels: Any,
    *,
    control_tokens: object,
    ignore_index: int = IGNORE_LABEL,
) -> dict[str, Any]:
    """Compute teacher-forced lexical-token cross-entropy and perplexity."""

    import torch
    import torch.nn.functional as F

    logits = torch.as_tensor(logits)
    label_tensor = torch.as_tensor(labels, device=logits.device)
    if (
        logits.ndim != 3
        or label_tensor.ndim != 2
        or logits.shape[:2] != label_tensor.shape
    ):
        raise ValueError(
            "logits must be [batch, frames, vocab] and labels [batch, frames]."
        )
    controls = set(_control_ids(control_tokens).values())
    lexical = label_tensor != ignore_index
    for control_id in controls:
        lexical &= label_tensor != control_id
    count = int(lexical.sum().item())
    if not count:
        undefined = _metric(None, 0)
        return {"loss": undefined, "perplexity": dict(undefined)}
    selected_logits = logits[lexical].float()
    selected_labels = label_tensor[lexical].long()
    loss = float(
        F.cross_entropy(selected_logits, selected_labels, reduction="mean").item()
    )
    perplexity = (
        math.exp(loss)
        if loss <= math.log(float.fromhex("0x1.fffffffffffffp+1023"))
        else math.inf
    )
    return {
        "loss": _metric(loss, count),
        "perplexity": _metric(perplexity, count),
    }


def free_streaming_rates(traces: Sequence[object]) -> dict[str, Any]:
    """Compute no-response and no-STOP rates over free-streaming traces."""

    no_response = 0
    no_stop = 0
    for trace in traces:
        records = getattr(trace, "trace", trace)
        kinds = []
        for record in records:
            if isinstance(record, Mapping):
                value = record.get("event_type")
            else:
                value = getattr(record, "event_type", record)
            value = value.value if isinstance(value, EventKind) else value
            if value not in EVENT_NAMES:
                raise ValueError(f"Unknown streaming event type {value!r}.")
            kinds.append(value)
        no_response += EventKind.START.value not in kinds
        no_stop += EventKind.STOP.value not in kinds
    denominator = len(traces)
    return {
        "no_response_rate": _metric(
            no_response / denominator if denominator else None, denominator
        ),
        "no_response_count": no_response,
        "no_STOP_rate": _metric(
            no_stop / denominator if denominator else None, denominator
        ),
        "no_STOP_count": no_stop,
    }
