"""Control vocabulary and causal event-timeline primitives."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any


IGNORE_LABEL = -100
_MASKED_INPUT_ID = 0


class EventKind(str, Enum):
    IDLE = "IDLE"
    START = "START"
    TEXT = "TEXT"
    STOP = "STOP"
    PADDING = "PADDING"


@dataclass(frozen=True)
class TimelineSpec:
    chunk_seconds: float = 2.0
    frame_rate_hz: int = 25

    def __post_init__(self) -> None:
        if self.chunk_seconds != 2.0:
            raise ValueError("Only fixed 2 s chunks are in scope.")
        if self.frame_rate_hz != 25:
            raise ValueError("Only 25 Hz is valid after configuration validation.")

    @property
    def frames_per_chunk(self) -> int:
        return int(self.chunk_seconds * self.frame_rate_hz)


def _config_value(container: object, name: str, path: str) -> Any:
    if isinstance(container, Mapping):
        if name not in container:
            raise ValueError(f"Qwen config is missing {path}.")
        return container[name]
    if not hasattr(container, name):
        raise ValueError(f"Qwen config is missing {path}.")
    return getattr(container, name)


def _integer_config_value(container: object, name: str, path: str) -> int:
    value = _config_value(container, name, path)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer, got {value!r}.")
    return value


@dataclass(frozen=True)
class ControlTokenIds:
    """Control IDs borrowed from Qwen's Talker config and checked for Thinker use."""

    idle: int
    start: int
    stop: int
    thinker_vocab_size: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.thinker_vocab_size, bool)
            or not isinstance(self.thinker_vocab_size, int)
            or self.thinker_vocab_size <= 0
        ):
            raise ValueError(
                "thinker_config.text_config.vocab_size must be a positive integer."
            )

        named_ids = {"IDLE": self.idle, "START": self.start, "STOP": self.stop}
        for kind, token_id in named_ids.items():
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise ValueError(f"{kind} token ID must be an integer, got {token_id!r}.")
            if token_id < 0:
                raise ValueError(f"{kind} token ID must be non-negative, got {token_id}.")
            if token_id >= self.thinker_vocab_size:
                raise ValueError(
                    f"{kind} token ID {token_id} is outside the Thinker text "
                    f"vocabulary of size {self.thinker_vocab_size}."
                )

        if len(set(named_ids.values())) != len(named_ids):
            raise ValueError(f"IDLE, START, and STOP token IDs must be unique: {named_ids}.")

    @classmethod
    def from_qwen_config(cls, config: object) -> "ControlTokenIds":
        talker = _config_value(config, "talker_config", "talker_config")
        thinker = _config_value(config, "thinker_config", "thinker_config")
        text = _config_value(thinker, "text_config", "thinker_config.text_config")

        return cls(
            idle=_integer_config_value(
                talker,
                "tts_text_pad_token_id",
                "talker_config.tts_text_pad_token_id",
            ),
            start=_integer_config_value(
                talker,
                "tts_text_start_token_id",
                "talker_config.tts_text_start_token_id",
            ),
            stop=_integer_config_value(
                talker,
                "tts_text_end_token_id",
                "talker_config.tts_text_end_token_id",
            ),
            thinker_vocab_size=_integer_config_value(
                text,
                "vocab_size",
                "thinker_config.text_config.vocab_size",
            ),
        )

    def for_event(self, kind: EventKind) -> int:
        if kind is EventKind.IDLE:
            return self.idle
        if kind is EventKind.START:
            return self.start
        if kind is EventKind.STOP:
            return self.stop
        raise ValueError(f"{kind.value} is not a control event.")


@dataclass(frozen=True)
class TargetEvent:
    """One target event before causal shifting."""

    kind: EventKind
    text_id: int | None = None


@dataclass(frozen=True)
class TargetEventSequence:
    """Targets and optional timing metadata, separate from model inputs."""

    events: Sequence[TargetEvent]
    sample_id: str = "<unknown>"
    frame_times: Sequence[float] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "events", tuple(self.events))
        if self.frame_times is not None:
            object.__setattr__(self, "frame_times", tuple(self.frame_times))
            if len(self.frame_times) != len(self.events):
                raise ValueError(
                    f"sample {self.sample_id!r}: frame_times has length "
                    f"{len(self.frame_times)}, expected {len(self.events)}."
                )


@dataclass(frozen=True)
class CausalTimeline:
    text_ids: list[int]
    text_mask: list[bool]
    control_ids: list[int]
    control_mask: list[bool]
    labels: list[int]
    event_types: list[EventKind]
    attention_mask: list[bool]
    frame_times: list[float | None] | None


def _event_error(sequence: TargetEventSequence, frame: int, message: str) -> ValueError:
    return ValueError(f"sample {sequence.sample_id!r}, frame {frame}: {message}")


def validate_target_sequence(
    sequence: TargetEventSequence,
    *,
    thinker_vocab_size: int,
) -> None:
    """Validate token membership, trailing padding, and inactive/active grammar."""

    active = False
    padding_started = False

    for frame, event in enumerate(sequence.events):
        if not isinstance(event.kind, EventKind):
            raise _event_error(sequence, frame, f"unknown event kind {event.kind!r}.")

        if event.kind is EventKind.PADDING:
            if event.text_id is not None:
                raise _event_error(sequence, frame, "PADDING cannot carry a text ID.")
            padding_started = True
            continue

        if padding_started:
            raise _event_error(sequence, frame, "non-padding event follows padding.")

        if event.kind is EventKind.TEXT:
            if not active:
                raise _event_error(sequence, frame, "TEXT is illegal before START.")
            if isinstance(event.text_id, bool) or not isinstance(event.text_id, int):
                raise _event_error(sequence, frame, "TEXT requires an integer text ID.")
            if not 0 <= event.text_id < thinker_vocab_size:
                raise _event_error(
                    sequence,
                    frame,
                    f"text ID {event.text_id} is outside the Thinker text vocabulary "
                    f"of size {thinker_vocab_size}.",
                )
            continue

        if event.text_id is not None:
            raise _event_error(
                sequence, frame, f"{event.kind.value} cannot carry a text ID."
            )

        if event.kind is EventKind.START:
            if active:
                raise _event_error(sequence, frame, "START is illegal while active.")
            active = True
        elif event.kind is EventKind.STOP:
            if not active:
                raise _event_error(sequence, frame, "STOP is illegal while inactive.")
            active = False
        # IDLE is legal in both states and does not change state.


def encode_causal_timeline(
    sequence: TargetEventSequence,
    *,
    control_tokens: ControlTokenIds,
    thinker_bos_token_id: int,
    pad_to_length: int | None = None,
) -> CausalTimeline:
    """Validate targets and create one-step-shifted causal model inputs."""

    vocab_size = control_tokens.thinker_vocab_size
    if (
        isinstance(thinker_bos_token_id, bool)
        or not isinstance(thinker_bos_token_id, int)
        or not 0 <= thinker_bos_token_id < vocab_size
    ):
        raise ValueError(
            f"Thinker BOS token ID must be in [0, {vocab_size}), "
            f"got {thinker_bos_token_id!r}."
        )

    validate_target_sequence(sequence, thinker_vocab_size=vocab_size)

    target_length = len(sequence.events)
    if pad_to_length is not None and (
        isinstance(pad_to_length, bool) or not isinstance(pad_to_length, int)
    ):
        raise ValueError(
            f"sample {sequence.sample_id!r}: pad_to_length must be an integer, "
            f"got {pad_to_length!r}."
        )
    output_length = target_length if pad_to_length is None else pad_to_length
    if output_length < target_length:
        raise ValueError(
            f"sample {sequence.sample_id!r}: pad_to_length {output_length} is shorter "
            f"than target length {target_length}."
        )

    events = list(sequence.events) + [
        TargetEvent(EventKind.PADDING) for _ in range(output_length - target_length)
    ]
    times: list[float | None] | None = None
    if sequence.frame_times is not None:
        times = list(sequence.frame_times) + [None] * (output_length - target_length)

    text_ids: list[int] = []
    text_mask: list[bool] = []
    control_ids: list[int] = []
    control_mask: list[bool] = []
    labels: list[int] = []
    event_types: list[EventKind] = []
    attention_mask: list[bool] = []

    for frame, target in enumerate(events):
        event_types.append(target.kind)
        is_padding = target.kind is EventKind.PADDING
        attention_mask.append(not is_padding)

        if is_padding:
            text_ids.append(_MASKED_INPUT_ID)
            text_mask.append(False)
            control_ids.append(_MASKED_INPUT_ID)
            control_mask.append(False)
            labels.append(IGNORE_LABEL)
            continue

        if target.kind is EventKind.TEXT:
            assert target.text_id is not None
            labels.append(target.text_id)
        else:
            labels.append(control_tokens.for_event(target.kind))

        if frame == 0:
            text_ids.append(thinker_bos_token_id)
            text_mask.append(True)
            control_ids.append(_MASKED_INPUT_ID)
            control_mask.append(False)
            continue

        previous = events[frame - 1]
        if previous.kind is EventKind.TEXT:
            assert previous.text_id is not None
            text_ids.append(previous.text_id)
            text_mask.append(True)
            control_ids.append(_MASKED_INPUT_ID)
            control_mask.append(False)
        else:
            text_ids.append(_MASKED_INPUT_ID)
            text_mask.append(False)
            control_ids.append(control_tokens.for_event(previous.kind))
            control_mask.append(True)

    return CausalTimeline(
        text_ids=text_ids,
        text_mask=text_mask,
        control_ids=control_ids,
        control_mask=control_mask,
        labels=labels,
        event_types=event_types,
        attention_mask=attention_mask,
        frame_times=times,
    )
