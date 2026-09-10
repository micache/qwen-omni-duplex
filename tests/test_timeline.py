from types import SimpleNamespace

import pytest

from duplex.timeline import (
    IGNORE_LABEL,
    ControlTokenIds,
    EventKind,
    TargetEvent,
    TargetEventSequence,
    TimelineSpec,
    encode_causal_timeline,
)


def qwen_config(
    idle: int = 10,
    start: int = 11,
    stop: int = 12,
    vocab_size: int = 100,
) -> dict:
    return {
        "talker_config": {
            "tts_text_pad_token_id": idle,
            "tts_text_start_token_id": start,
            "tts_text_end_token_id": stop,
        },
        "thinker_config": {"text_config": {"vocab_size": vocab_size}},
    }


def controls() -> ControlTokenIds:
    return ControlTokenIds.from_qwen_config(qwen_config())


def event(kind: EventKind, text_id: int | None = None) -> TargetEvent:
    return TargetEvent(kind, text_id)


def encode(events: list[TargetEvent], **kwargs):
    return encode_causal_timeline(
        TargetEventSequence(events, sample_id="sample-7"),
        control_tokens=controls(),
        thinker_bos_token_id=1,
        **kwargs,
    )


def test_fixed_timeline_contract() -> None:
    spec = TimelineSpec()
    assert spec.frames_per_chunk == 50
    assert [kind.value for kind in EventKind] == [
        "IDLE",
        "START",
        "TEXT",
        "STOP",
        "PADDING",
    ]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"chunk_seconds": 1.0}, "2 s chunks"),
        ({"frame_rate_hz": 50}, "25 Hz"),
    ],
)
def test_out_of_scope_timeline_is_rejected(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        TimelineSpec(**kwargs)


def test_control_ids_are_read_from_dict_config() -> None:
    token_ids = ControlTokenIds.from_qwen_config(qwen_config())
    assert token_ids == ControlTokenIds(10, 11, 12, thinker_vocab_size=100)


def test_control_ids_are_read_from_object_config() -> None:
    config = SimpleNamespace(
        talker_config=SimpleNamespace(
            tts_text_pad_token_id=20,
            tts_text_start_token_id=21,
            tts_text_end_token_id=22,
        ),
        thinker_config=SimpleNamespace(
            text_config=SimpleNamespace(vocab_size=200)
        ),
    )
    assert ControlTokenIds.from_qwen_config(config) == ControlTokenIds(
        20, 21, 22, 200
    )


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (qwen_config(idle=-1), "non-negative"),
        (qwen_config(start=10), "must be unique"),
        (qwen_config(stop=100), "outside the Thinker text vocabulary"),
        (qwen_config(vocab_size=0), "positive integer"),
    ],
)
def test_invalid_control_ids_are_rejected(config: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ControlTokenIds.from_qwen_config(config)


def test_causal_shift_uses_only_the_previous_event() -> None:
    timeline = encode(
        [
            event(EventKind.START),
            event(EventKind.TEXT, 40),
            event(EventKind.TEXT, 41),
            event(EventKind.STOP),
        ]
    )

    assert timeline.labels == [11, 40, 41, 12]
    assert timeline.text_ids == [1, 0, 40, 41]
    assert timeline.text_mask == [True, False, True, True]
    assert timeline.control_ids == [0, 11, 0, 0]
    assert timeline.control_mask == [False, True, False, False]
    assert timeline.labels[1] != timeline.control_ids[1]
    assert timeline.labels[2] != timeline.text_ids[2]


def test_valid_state_transitions_and_idle_inside_response() -> None:
    timeline = encode(
        [
            event(EventKind.IDLE),
            event(EventKind.START),
            event(EventKind.TEXT, 40),
            event(EventKind.IDLE),
            event(EventKind.TEXT, 41),
            event(EventKind.STOP),
            event(EventKind.IDLE),
        ]
    )
    assert timeline.event_types[3] is EventKind.IDLE
    assert timeline.labels == [10, 11, 40, 10, 41, 12, 10]


@pytest.mark.parametrize(
    ("events", "frame", "message"),
    [
        ([event(EventKind.TEXT, 40)], 0, "TEXT is illegal before START"),
        (
            [event(EventKind.START), event(EventKind.START)],
            1,
            "START is illegal while active",
        ),
        ([event(EventKind.STOP)], 0, "STOP is illegal while inactive"),
    ],
)
def test_illegal_state_transitions_include_sample_and_frame(
    events: list[TargetEvent], frame: int, message: str
) -> None:
    with pytest.raises(ValueError) as error:
        encode(events)
    rendered = str(error.value)
    assert "sample 'sample-7'" in rendered
    assert f"frame {frame}" in rendered
    assert message in rendered


def test_padding_uses_ignore_label_and_false_masks() -> None:
    timeline = encode(
        [event(EventKind.START), event(EventKind.TEXT, 40)], pad_to_length=5
    )

    assert timeline.labels == [11, 40, IGNORE_LABEL, IGNORE_LABEL, IGNORE_LABEL]
    assert timeline.event_types[-3:] == [EventKind.PADDING] * 3
    assert timeline.attention_mask == [True, True, False, False, False]
    assert timeline.text_mask[-3:] == [False, False, False]
    assert timeline.control_mask[-3:] == [False, False, False]


def test_all_streams_have_identical_length_and_optional_times_align() -> None:
    sequence = TargetEventSequence(
        [event(EventKind.START), event(EventKind.TEXT, 40), event(EventKind.STOP)],
        sample_id="timed",
        frame_times=[0.0, 0.04, 0.08],
    )
    timeline = encode_causal_timeline(
        sequence,
        control_tokens=controls(),
        thinker_bos_token_id=1,
        pad_to_length=5,
    )

    streams = (
        timeline.text_ids,
        timeline.text_mask,
        timeline.control_ids,
        timeline.control_mask,
        timeline.labels,
        timeline.event_types,
        timeline.attention_mask,
        timeline.frame_times,
    )
    assert {len(stream) for stream in streams if stream is not None} == {5}
    assert timeline.frame_times == [0.0, 0.04, 0.08, None, None]
