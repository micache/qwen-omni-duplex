import math
from types import SimpleNamespace

import pytest
import torch

from duplex.metrics import (
    boundary_error_metrics,
    event_classification_metrics,
    free_streaming_rates,
    interruption_stop_metrics,
    invalid_state_transition_rate,
    teacher_forced_text_metrics,
)


CONTROLS = SimpleNamespace(idle=10, start=11, stop=12)


def test_event_counts_control_scores_and_padding() -> None:
    labels = [[10, 11, 20, 12, -100], [10, 11, 21, 12, -100]]
    predictions = [[10, 11, 20, 12, 11], [20, 10, 21, 10, 12]]

    metrics = event_classification_metrics(
        labels, predictions, control_tokens=CONTROLS
    )

    assert metrics["label_counts"] == {"TEXT": 2, "IDLE": 2, "START": 2, "STOP": 2}
    assert metrics["prediction_counts"] == {
        "TEXT": 3,
        "IDLE": 3,
        "START": 1,
        "STOP": 1,
    }
    assert metrics["control"]["START"]["precision"] == {
        "value": 1.0,
        "denominator": 1,
    }
    assert metrics["control"]["START"]["recall"] == {
        "value": 0.5,
        "denominator": 2,
    }
    assert metrics["control"]["STOP"]["f1"] == {
        "value": pytest.approx(2 / 3),
        "denominator": 3,
    }


def test_undefined_control_scores_are_null_with_denominators() -> None:
    metrics = event_classification_metrics(
        [[20, -100]], [[21, 12]], control_tokens=CONTROLS
    )

    assert metrics["prediction_counts"]["STOP"] == 0
    assert metrics["control"]["STOP"]["precision"] == {
        "value": None,
        "denominator": 0,
    }
    assert metrics["control"]["STOP"]["recall"] == {
        "value": None,
        "denominator": 0,
    }
    assert metrics["control"]["STOP"]["f1"] == {
        "value": None,
        "denominator": 0,
    }


def test_invalid_transition_rate_uses_raw_state_and_ignores_padding() -> None:
    predictions = [[20, 12, 11, 11, 20, 12, 10, 12, 11]]
    labels = [[10, 10, 11, 20, 20, 12, 10, 10, -100]]

    metric = invalid_state_transition_rate(
        predictions, labels=labels, control_tokens=CONTROLS
    )

    assert metric == {"value": 0.5, "denominator": 8, "invalid_count": 4}


def test_boundary_errors_pair_controls_within_ordinal_response_turns() -> None:
    labels = [[10, 11, 20, 20, 12, 10, 11, 21, 12, -100]]
    predictions = [[11, 20, 20, 12, 10, 10, 10, 11, 21, 21]]

    metrics = boundary_error_metrics(
        labels, predictions, control_tokens=CONTROLS, frame_rate_hz=25
    )

    assert metrics["target_response_turn_count"] == 2
    assert metrics["predicted_response_turn_count"] == 2
    assert metrics["matched_response_turn_count"] == 2
    assert metrics["START"]["absolute_error_frames"] == {
        "value": 1.0,
        "denominator": 2,
    }
    assert metrics["START"]["absolute_error_seconds"] == {
        "value": 0.04,
        "denominator": 2,
    }
    assert metrics["STOP"]["absolute_error_frames"] == {
        "value": 1.0,
        "denominator": 1,
    }
    assert metrics["STOP"]["absolute_error_seconds"] == {
        "value": 0.04,
        "denominator": 1,
    }


def test_boundary_errors_are_null_when_no_turn_is_matched() -> None:
    metrics = boundary_error_metrics(
        [[10, 10]], [[10, 10]], control_tokens=CONTROLS
    )

    assert metrics["START"]["absolute_error_frames"] == {
        "value": None,
        "denominator": 0,
    }
    assert metrics["STOP"]["absolute_error_seconds"] == {
        "value": None,
        "denominator": 0,
    }


def test_interruption_stop_recall_and_latency() -> None:
    predictions = [
        [11, 20, 10, 12, 10],
        [11, 20, 10, 10, 10],
        [10, 10, 10, 10, 10],
    ]
    metrics = interruption_stop_metrics(
        predictions,
        [2, 1, None],
        control_tokens=CONTROLS,
        frame_rate_hz=25,
    )

    assert metrics["recall"] == {"value": 0.5, "denominator": 2}
    assert metrics["recalled_count"] == 1
    assert metrics["latency_frames"] == {"value": 1.0, "denominator": 1}
    assert metrics["latency_seconds"] == {"value": 0.04, "denominator": 1}


def test_interruption_metrics_are_undefined_without_interruptions() -> None:
    metrics = interruption_stop_metrics(
        [[10, 10]], [None], control_tokens=CONTROLS
    )

    assert metrics["recall"] == {"value": None, "denominator": 0}
    assert metrics["latency_frames"] == {"value": None, "denominator": 0}


def test_teacher_forced_text_loss_excludes_controls_and_padding() -> None:
    logits = torch.zeros((1, 5, 4), dtype=torch.float32)
    labels = torch.tensor([[0, 1, 3, 2, -100]])
    controls = SimpleNamespace(idle=0, start=3, stop=10)
    logits[0, 1] = torch.tensor([0.0, 2.0, 0.0, 0.0])
    logits[0, 3] = torch.tensor([0.0, 0.0, 2.0, 0.0])

    metrics = teacher_forced_text_metrics(
        logits, labels, control_tokens=controls
    )
    expected = float(
        torch.nn.functional.cross_entropy(
            torch.tensor(
                [[0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0]]
            ),
            torch.tensor([1, 2]),
        )
    )

    assert metrics["loss"] == {
        "value": pytest.approx(expected),
        "denominator": 2,
    }
    assert metrics["perplexity"] == {
        "value": pytest.approx(math.exp(expected)),
        "denominator": 2,
    }


def test_teacher_forced_text_metrics_are_null_without_text() -> None:
    metrics = teacher_forced_text_metrics(
        torch.zeros((1, 3, 13)),
        torch.tensor([[10, 11, 12]]),
        control_tokens=CONTROLS,
    )

    assert metrics == {
        "loss": {"value": None, "denominator": 0},
        "perplexity": {"value": None, "denominator": 0},
    }


def test_free_streaming_no_response_and_no_stop_rates() -> None:
    traces = [
        ["IDLE", "IDLE"],
        ["START", "TEXT", "STOP"],
        [SimpleNamespace(event_type="START"), SimpleNamespace(event_type="TEXT")],
    ]

    metrics = free_streaming_rates(traces)

    assert metrics["no_response_rate"] == {
        "value": pytest.approx(1 / 3),
        "denominator": 3,
    }
    assert metrics["no_response_count"] == 1
    assert metrics["no_STOP_rate"] == {
        "value": pytest.approx(2 / 3),
        "denominator": 3,
    }
    assert metrics["no_STOP_count"] == 2


def test_free_streaming_empty_input_uses_null_and_zero_denominator() -> None:
    metrics = free_streaming_rates([])

    assert metrics["no_response_rate"] == {"value": None, "denominator": 0}
    assert metrics["no_STOP_rate"] == {"value": None, "denominator": 0}
