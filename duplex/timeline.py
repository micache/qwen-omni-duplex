"""Control vocabulary, word/token alignment, and causal event timelines."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

from .dataset import ConversationMetadata, ConversationRecord, Speaker, WindowMetadata, WordSpan
from .contract import frame_event_inputs


IGNORE_LABEL = -100
_MASKED_INPUT_ID = 0


class EventKind(str, Enum):
    IDLE = "IDLE"
    START = "START"
    TEXT = "TEXT"
    STOP = "STOP"
    PADDING = "PADDING"


class ResponseState(str, Enum):
    INACTIVE = "inactive"
    ACTIVE = "active"


class InvalidTimelineError(ValueError):
    """A crop that cannot represent all required events without corruption."""


@dataclass(frozen=True)
class TimelineSpec:
    chunk_seconds: float = 2.0
    frame_rate_hz: int = 25

    def __post_init__(self) -> None:
        if self.chunk_seconds != 2.0:
            raise ValueError("Only fixed 2 s chunks are in scope.")
        if (
            isinstance(self.frame_rate_hz, bool)
            or not isinstance(self.frame_rate_hz, int)
            or self.frame_rate_hz != 25
        ):
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
    bootstrap_mask: list[bool] | None = None


@dataclass(frozen=True)
class TextNormalization:
    """An explicit edit made while reconstructing utterance text from words."""

    source_word_index: int
    description: str


@dataclass(frozen=True)
class AlignedToken:
    token_id: int
    decoded_token: str
    character_start: int
    character_end: int
    source_word_index: int
    source_turn_id: str
    source_start_seconds: float
    source_end_seconds: float


@dataclass(frozen=True)
class UtteranceAlignment:
    source_turn_id: str
    source_word_indices: tuple[int, ...]
    source_start_seconds: float
    source_end_seconds: float
    target_text: str
    token_ids: tuple[int, ...]
    tokens: tuple[AlignedToken, ...]
    alignment_method: str
    normalizations: tuple[TextNormalization, ...]


@dataclass(frozen=True)
class FrameAlignment:
    frame_index: int
    audio_time_seconds: float
    user_active: bool
    target_event: TargetEvent
    decoded_token: str
    state: ResponseState
    source_word_index: int | None = None
    source_turn_id: str | None = None


@dataclass(frozen=True)
class InterruptionConfig:
    """Public-data synthetic interruption controls, measured at 25 Hz."""

    probability: float = 0.0
    min_assistant_frames: int = 5

    def __post_init__(self) -> None:
        if (
            isinstance(self.probability, bool)
            or not isinstance(self.probability, (int, float))
            or not math.isfinite(float(self.probability))
            or not 0.0 <= float(self.probability) <= 1.0
        ):
            raise ValueError("interruption probability must be finite and in [0, 1].")
        if (
            isinstance(self.min_assistant_frames, bool)
            or not isinstance(self.min_assistant_frames, int)
            or self.min_assistant_frames < 1
        ):
            raise ValueError("min_assistant_frames must be a positive integer.")


@dataclass(frozen=True)
class InterruptionMetadata:
    source_turn_id: str
    cut_frame: int
    original_stop_frame: int
    original_user_frame: int
    shifted_frames: int


@dataclass(frozen=True)
class WindowTimeline:
    """One fixed-rate crop, its inspection metadata, and shifted inputs."""

    sample_id: str
    window_start_seconds: float
    window_end_seconds: float
    frame_rate_hz: int
    user_waveform: Any
    utterances: tuple[UtteranceAlignment, ...]
    frames: tuple[FrameAlignment, ...]
    targets: TargetEventSequence
    causal: CausalTimeline
    user_words: tuple[WordSpan, ...] = ()
    interruption: InterruptionMetadata | None = None
    audio_sample_rate_hz: int = 16_000

    @property
    def conversation_id(self) -> str:
        return self.sample_id.split("@", 1)[0]

    @property
    def valid_sequence_length(self) -> int:
        return len(self.targets.events)


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
    bootstrap_mask: list[bool] = []

    for frame, target in enumerate(events):
        event_types.append(target.kind)
        is_padding = target.kind is EventKind.PADDING
        attention_mask.append(not is_padding)
        bootstrap_mask.append(frame == 0 and not is_padding)

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

        previous_id = None if frame == 0 else (
            events[frame - 1].text_id if events[frame - 1].kind is EventKind.TEXT
            else control_tokens.for_event(events[frame - 1].kind)
        )
        text_id, has_text, control_id, has_control = frame_event_inputs(
            previous_id, bos_token_id=thinker_bos_token_id,
            control_ids=(control_tokens.idle, control_tokens.start, control_tokens.stop))
        text_ids.append(text_id)
        text_mask.append(has_text)
        control_ids.append(control_id)
        control_mask.append(has_control)

    return CausalTimeline(
        text_ids=text_ids,
        text_mask=text_mask,
        control_ids=control_ids,
        control_mask=control_mask,
        labels=labels,
        event_types=event_types,
        attention_mask=attention_mask,
        frame_times=times,
        bootstrap_mask=bootstrap_mask,
    )


@dataclass(frozen=True)
class _IndexedWord:
    source_word_index: int
    span: WordSpan


@dataclass(frozen=True)
class _AssistantTurn:
    source_turn_id: str
    words: tuple[_IndexedWord, ...]


_CLOSING_PUNCTUATION = frozenset(
    ",.;:!?%)]}\u00bb\u201d\u2019\u2026，。；：！？、）】》」』"
)
_OPENING_PUNCTUATION = frozenset("([{\u00ab\u201c\u2018（【《「『")


def _assistant_turns(record: ConversationRecord) -> tuple[_AssistantTurn, ...]:
    """Return maximal annotated speaker runs with stable assistant turn IDs."""

    ordered: list[tuple[WordSpan, int | None]] = [
        (
            word,
            index if word.source_index is None else word.source_index,
        )
        for index, word in enumerate(record.assistant_words)
    ]
    ordered.extend((word, None) for word in record.user_words)
    ordered.sort(
        key=lambda item: (
            item[0].start_seconds,
            item[0].end_seconds,
            item[0].speaker.value,
        )
    )

    runs: list[list[_IndexedWord]] = []
    current: list[_IndexedWord] = []
    previous_speaker: Speaker | None = None
    for word, assistant_index in ordered:
        if word.speaker is not previous_speaker:
            if current:
                runs.append(current)
                current = []
            previous_speaker = word.speaker
        if word.speaker is Speaker.ASSISTANT:
            assert assistant_index is not None
            current.append(_IndexedWord(assistant_index, word))
    if current:
        runs.append(current)

    return tuple(
        _AssistantTurn(
            source_turn_id=f"{record.conversation_id}:assistant:{turn_index}",
            words=tuple(words),
        )
        for turn_index, words in enumerate(runs)
    )


def _utterance_text(
    words: Sequence[_IndexedWord],
) -> tuple[str, tuple[tuple[int, int, int], ...], tuple[TextNormalization, ...]]:
    """Rebuild readable text and retain every explicit spacing decision."""

    pieces: list[str] = []
    ranges: list[tuple[int, int, int]] = []
    normalizations: list[TextNormalization] = []
    length = 0
    previous_text = ""
    for position, indexed in enumerate(words):
        text = indexed.span.text
        if not text:
            raise ValueError(
                f"source word {indexed.source_word_index} has empty text."
            )
        separator = ""
        if position and not text[0].isspace() and not previous_text[-1].isspace():
            if text[0] not in _CLOSING_PUNCTUATION and previous_text[-1] not in (
                _OPENING_PUNCTUATION
            ):
                separator = " "
                normalizations.append(
                    TextNormalization(
                        indexed.source_word_index,
                        "inserted one space between adjacent word annotations",
                    )
                )
        pieces.append(separator)
        length += len(separator)
        start = length
        pieces.append(text)
        length += len(text)
        ranges.append((start, length, indexed.source_word_index))
        previous_text = text
    return "".join(pieces), tuple(ranges), tuple(normalizations)


def _decode(tokenizer: object, token_ids: Sequence[int]) -> str:
    try:
        return tokenizer.decode(  # type: ignore[attr-defined]
            list(token_ids),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode(list(token_ids))  # type: ignore[attr-defined]


def _tokenizer_output_value(output: object, name: str) -> object | None:
    if isinstance(output, Mapping):
        return output.get(name)
    return getattr(output, name, None)


def _flatten_single_sequence(value: object, *, field: str) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()  # type: ignore[union-attr]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"tokenizer {field} must be a sequence, got {value!r}.")
    result = list(value)
    if len(result) == 1 and isinstance(result[0], Sequence) and not isinstance(
        result[0], (str, bytes)
    ):
        result = list(result[0])
    return result


def _flatten_offsets(value: object) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()  # type: ignore[union-attr]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(
            f"tokenizer offset_mapping must be a sequence, got {value!r}."
        )
    result = list(value)
    if (
        len(result) == 1
        and isinstance(result[0], Sequence)
        and not isinstance(result[0], (str, bytes))
        and result[0]
        and isinstance(result[0][0], Sequence)
        and not isinstance(result[0][0], (str, bytes))
    ):
        result = list(result[0])
    return result


def _encode_utterance(
    tokenizer: object,
    text: str,
) -> tuple[list[int], list[tuple[int, int]] | None]:
    output: object | None = None
    if callable(tokenizer):
        try:
            output = tokenizer(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
        except (NotImplementedError, TypeError):
            output = None

    if output is not None:
        raw_ids = _tokenizer_output_value(output, "input_ids")
        if raw_ids is None:
            raise ValueError("tokenizer output has no input_ids.")
        token_ids = _flatten_single_sequence(raw_ids, field="input_ids")
        raw_offsets = _tokenizer_output_value(output, "offset_mapping")
        offsets = None
        if raw_offsets is not None:
            unpacked = _flatten_offsets(raw_offsets)
            offsets = []
            for offset in unpacked:
                if hasattr(offset, "tolist"):
                    offset = offset.tolist()
                if (
                    not isinstance(offset, Sequence)
                    or isinstance(offset, (str, bytes))
                    or len(offset) != 2
                ):
                    raise ValueError(f"invalid tokenizer offset {offset!r}.")
                offsets.append((int(offset[0]), int(offset[1])))
        return _validate_token_ids(token_ids), offsets

    try:
        raw_ids = tokenizer.encode(  # type: ignore[attr-defined]
            text, add_special_tokens=False
        )
    except TypeError:
        raw_ids = tokenizer.encode(text)  # type: ignore[attr-defined]
    return _validate_token_ids(
        _flatten_single_sequence(raw_ids, field="input_ids")
    ), None


def _validate_token_ids(token_ids: Sequence[object]) -> list[int]:
    validated: list[int] = []
    for position, token_id in enumerate(token_ids):
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise ValueError(
                f"tokenizer ID at position {position} must be an integer, "
                f"got {token_id!r}."
            )
        if token_id < 0:
            raise ValueError(
                f"tokenizer ID at position {position} is negative: {token_id}."
            )
        validated.append(token_id)
    return validated


def _word_for_character_span(
    character_start: int,
    character_end: int,
    word_ranges: Sequence[tuple[int, int, int]],
    *,
    previous_word_index: int | None,
) -> int:
    overlapping = [
        source_index
        for start, end, source_index in word_ranges
        if start < character_end and character_start < end
    ]
    if overlapping:
        chosen = overlapping[0]
    else:
        following = [
            source_index
            for start, _, source_index in word_ranges
            if start >= character_end
        ]
        chosen = following[0] if following else word_ranges[-1][2]
    if previous_word_index is not None and chosen < previous_word_index:
        raise ValueError(
            f"token-to-word mapping moved backward from source word "
            f"{previous_word_index} to {chosen}."
        )
    return chosen


def _incremental_offsets(
    tokenizer: object,
    token_ids: Sequence[int],
    text: str,
) -> list[tuple[int, int]]:
    """Infer spans from exact decoded prefixes, grouping incomplete byte tokens."""

    offsets: list[tuple[int, int] | None] = [None] * len(token_ids)
    committed = 0
    pending_start = 0
    for end in range(1, len(token_ids) + 1):
        prefix = _decode(tokenizer, token_ids[:end])
        if not text.startswith(prefix) or len(prefix) < committed:
            continue
        if len(prefix) == committed and end != len(token_ids):
            continue
        for token_position in range(pending_start, end):
            offsets[token_position] = (committed, len(prefix))
        committed = len(prefix)
        pending_start = end
    if committed != len(text) or pending_start != len(token_ids):
        raise ValueError(
            "incremental decoded prefixes could not be aligned exactly to the "
            f"target text; committed {committed} of {len(text)} characters."
        )
    return [offset for offset in offsets if offset is not None]


def align_assistant_utterance(
    tokenizer: object,
    words: Sequence[tuple[int, WordSpan]],
    *,
    sample_id: str,
    source_turn_id: str,
) -> UtteranceAlignment:
    """Tokenize a complete utterance once and align every token to a source word."""

    indexed_words = tuple(_IndexedWord(index, word) for index, word in words)
    if not indexed_words:
        raise ValueError(
            f"sample {sample_id!r}, turn {source_turn_id!r}: utterance has no words."
        )
    seen_word_indices: set[int] = set()
    previous_start = -math.inf
    previous_source_word_index = -1
    for indexed in indexed_words:
        word = indexed.span
        if (
            isinstance(indexed.source_word_index, bool)
            or not isinstance(indexed.source_word_index, int)
            or indexed.source_word_index < 0
        ):
            raise ValueError(
                f"sample {sample_id!r}, turn {source_turn_id!r}, word "
                f"{indexed.source_word_index!r}: source word index must be a "
                "non-negative integer."
            )
        if indexed.source_word_index in seen_word_indices:
            raise ValueError(
                f"sample {sample_id!r}, turn {source_turn_id!r}, word "
                f"{indexed.source_word_index}: duplicate source word index."
            )
        if indexed.source_word_index < previous_source_word_index:
            raise ValueError(
                f"sample {sample_id!r}, turn {source_turn_id!r}, word "
                f"{indexed.source_word_index}: source word indices are not "
                "monotonic."
            )
        if word.speaker is not Speaker.ASSISTANT:
            raise ValueError(
                f"sample {sample_id!r}, turn {source_turn_id!r}, word "
                f"{indexed.source_word_index}: expected assistant word, got "
                f"{word.speaker.value}."
            )
        if word.start_seconds < previous_start:
            raise ValueError(
                f"sample {sample_id!r}, turn {source_turn_id!r}, word "
                f"{indexed.source_word_index}: source word times are not monotonic."
            )
        seen_word_indices.add(indexed.source_word_index)
        previous_start = word.start_seconds
        previous_source_word_index = indexed.source_word_index
    text, word_ranges, normalizations = _utterance_text(indexed_words)
    try:
        token_ids, offsets = _encode_utterance(tokenizer, text)
        decoded = _decode(tokenizer, token_ids)
    except Exception as error:
        if isinstance(error, ValueError) and str(error).startswith("sample "):
            raise
        raise ValueError(
            f"sample {sample_id!r}, turn {source_turn_id!r}, words "
            f"{[word.source_word_index for word in indexed_words]}: tokenizer "
            f"alignment failed: {error}"
        ) from error
    if not token_ids:
        raise ValueError(
            f"sample {sample_id!r}, turn {source_turn_id!r}, words "
            f"{[word.source_word_index for word in indexed_words]}: non-empty "
            "target text produced no BPE tokens."
        )
    if decoded != text:
        raise ValueError(
            f"sample {sample_id!r}, turn {source_turn_id!r}, words "
            f"{[word.source_word_index for word in indexed_words]}: exact tokenizer "
            f"round trip failed; target={text!r}, decoded={decoded!r}."
        )

    method = "tokenizer_offsets"
    if offsets is None:
        method = "incremental_decode_round_trip"
        try:
            offsets = _incremental_offsets(tokenizer, token_ids, text)
        except ValueError as error:
            raise ValueError(
                f"sample {sample_id!r}, turn {source_turn_id!r}, words "
                f"{[word.source_word_index for word in indexed_words]}: {error}"
            ) from error
    if len(offsets) != len(token_ids):
        raise ValueError(
            f"sample {sample_id!r}, turn {source_turn_id!r}, words "
            f"{[word.source_word_index for word in indexed_words]}: tokenizer "
            f"returned {len(offsets)} offsets for {len(token_ids)} tokens."
        )

    words_by_index = {word.source_word_index: word.span for word in indexed_words}
    aligned: list[AlignedToken] = []
    previous_word_index: int | None = None
    previous_character_start = -1
    covered_until = 0
    for position, (token_id, offset) in enumerate(zip(token_ids, offsets)):
        start, end = offset
        if (
            not 0 <= start < end <= len(text)
            or start < previous_character_start
            or start > covered_until
        ):
            raise ValueError(
                f"sample {sample_id!r}, turn {source_turn_id!r}, token {position}, "
                f"words {[word.source_word_index for word in indexed_words]}: "
                f"invalid, gapped, or non-monotonic character offset "
                f"{(start, end)!r}; text is covered through {covered_until}."
            )
        source_word_index = _word_for_character_span(
            start,
            end,
            word_ranges,
            previous_word_index=previous_word_index,
        )
        word = words_by_index[source_word_index]
        aligned.append(
            AlignedToken(
                token_id=token_id,
                decoded_token=_decode(tokenizer, [token_id]),
                character_start=start,
                character_end=end,
                source_word_index=source_word_index,
                source_turn_id=source_turn_id,
                source_start_seconds=word.start_seconds,
                source_end_seconds=word.end_seconds,
            )
        )
        previous_character_start = start
        covered_until = max(covered_until, end)
        previous_word_index = source_word_index

    if covered_until != len(text):
        raise ValueError(
            f"sample {sample_id!r}, turn {source_turn_id!r}, words "
            f"{[word.source_word_index for word in indexed_words]}: tokenizer "
            f"offsets cover {covered_until} of {len(text)} target characters."
        )

    return UtteranceAlignment(
        source_turn_id=source_turn_id,
        source_word_indices=tuple(
            word.source_word_index for word in indexed_words
        ),
        source_start_seconds=indexed_words[0].span.start_seconds,
        source_end_seconds=indexed_words[-1].span.end_seconds,
        target_text=text,
        token_ids=tuple(token_ids),
        tokens=tuple(aligned),
        alignment_method=method,
        normalizations=normalizations,
    )


def _frame_for_time(
    time_seconds: float,
    *,
    window_start_seconds: float,
    frame_rate_hz: int,
) -> int:
    return math.floor((time_seconds - window_start_seconds) * frame_rate_hz + 1e-9)


def _allocate_token_frames(
    alignment: UtteranceAlignment,
    *,
    sample_id: str,
    window_start_seconds: float,
    frame_rate_hz: int,
    frame_count: int,
    after_frame: int,
    allow_time_extension: bool = False,
) -> list[int]:
    tokens = alignment.tokens
    turn_start = alignment.source_start_seconds
    turn_end = alignment.source_end_seconds
    first_assistant_frame = max(
        0,
        _frame_for_time(
            turn_start,
            window_start_seconds=window_start_seconds,
            frame_rate_hz=frame_rate_hz,
        ),
    )
    last_assistant_frame = min(
        frame_count - 1,
        math.ceil(
            (turn_end - window_start_seconds) * frame_rate_hz - 1e-9
        )
        - 1,
    )
    lower = max(first_assistant_frame, 1, after_frame + 2)
    upper = min(last_assistant_frame, frame_count - 2)
    if allow_time_extension and upper - lower + 1 < len(tokens):
        # A 25-Hz event stream cannot place several tokens inside a very short
        # annotated word. Keep every token and move its STOP by the minimum
        # necessary frames on the complete conversation timeline.
        upper = min(frame_count - 2, lower + len(tokens) - 1)
    if upper - lower + 1 < len(tokens):
        raise InvalidTimelineError(
            f"sample {sample_id!r}, turn {alignment.source_turn_id!r}, words "
            f"{list(alignment.source_word_indices)}: invalid "
            f"timeline: {len(tokens)} lexical tokens need {len(tokens) + 2} "
            f"event frames, but assistant time {turn_start:.6f}..{turn_end:.6f}s "
            f"offers only {max(0, upper - lower + 1)} lexical frames after "
            "reserving adjacent START/STOP and prior events."
        )

    tokens_per_word: dict[int, int] = {}
    token_ordinal: dict[int, int] = {}
    for token in tokens:
        tokens_per_word[token.source_word_index] = (
            tokens_per_word.get(token.source_word_index, 0) + 1
        )
    desired: list[int] = []
    for token in tokens:
        ordinal = token_ordinal.get(token.source_word_index, 0)
        token_ordinal[token.source_word_index] = ordinal + 1
        fraction = (ordinal + 0.5) / tokens_per_word[token.source_word_index]
        token_time = token.source_start_seconds + fraction * (
            token.source_end_seconds - token.source_start_seconds
        )
        desired.append(
            _frame_for_time(
                token_time,
                window_start_seconds=window_start_seconds,
                frame_rate_hz=frame_rate_hz,
            )
        )

    allocated: list[int] = []
    for position, preferred in enumerate(desired):
        earliest = lower + position if not allocated else allocated[-1] + 1
        latest = upper - (len(tokens) - position - 1)
        allocated.append(min(max(preferred, earliest), latest))
    return allocated


def build_window_timeline(
    record: ConversationRecord,
    window: WindowMetadata,
    *,
    tokenizer: object,
    control_tokens: ControlTokenIds,
    thinker_bos_token_id: int,
    frame_rate_hz: int,
    valid_frame_count: int | None = None,
) -> WindowTimeline:
    """Convert one complete crop into aligned frame targets.

    Eight-second spans are supervised as one causal sequence; the audio is
    still divided into four two-second chunks by the collator.
    """

    if window.conversation_id != record.conversation_id:
        raise ValueError(
            f"Window conversation {window.conversation_id!r} does not match record "
            f"{record.conversation_id!r}."
        )
    if (
        window.start_seconds < 0.0
        or window.end_seconds > record.duration_seconds + 1e-9
    ):
        raise ValueError(
            f"Window {window.start_seconds:.6f}..{window.end_seconds:.6f}s is "
            f"outside conversation {record.conversation_id!r} duration "
            f"{record.duration_seconds:.6f}s."
        )
    spec = TimelineSpec(frame_rate_hz=frame_rate_hz)
    duration = window.end_seconds - window.start_seconds
    if not isinstance(window, ConversationMetadata) and not math.isclose(duration, 2.0, abs_tol=1e-9) and not math.isclose(duration, 8.0, abs_tol=1e-9):
        raise ValueError(f"sample {record.conversation_id!r}: crop must be 2 or 8 seconds.")
    frame_count = round(duration * frame_rate_hz) if valid_frame_count is None else valid_frame_count
    if frame_count < 3:
        raise InvalidTimelineError(f"sample {record.conversation_id!r}: only {frame_count} valid audio frames.")
    sample_id = (
        f"{record.conversation_id}@{window.start_seconds:.6f}.."
        f"{window.end_seconds:.6f}"
    )

    start_sample = round(window.start_seconds * record.sample_rate_hz)
    end_sample = len(record.user_waveform) if isinstance(window, ConversationMetadata) else start_sample + round(duration * record.sample_rate_hz)
    user_waveform = record.user_waveform[start_sample:end_sample]
    if len(user_waveform) != end_sample - start_sample:
        raise ValueError(
            f"sample {sample_id!r}: user/right-channel crop has {len(user_waveform)} "
            f"samples, expected {end_sample - start_sample}."
        )

    events = [TargetEvent(EventKind.IDLE) for _ in range(frame_count)]
    frame_tokens: dict[int, AlignedToken] = {}
    frame_turns: dict[int, str] = {}
    frame_words: dict[int, int] = {}
    utterances: list[UtteranceAlignment] = []
    last_event_frame = -1
    for turn in _assistant_turns(record):
        overlaps = any(
            word.span.end_seconds > window.start_seconds
            and word.span.start_seconds < window.end_seconds
            for word in turn.words
        )
        complete = all(
            word.span.start_seconds >= window.start_seconds
            and word.span.end_seconds <= window.end_seconds
            for word in turn.words
        )
        if not overlaps:
            continue
        if not complete:
            if isinstance(window, ConversationMetadata):
                raise InvalidTimelineError(f"sample {sample_id!r}, turn {turn.source_turn_id!r}: annotation extends outside complete audio.")
            continue
        alignment = align_assistant_utterance(
            tokenizer,
            tuple((word.source_word_index, word.span) for word in turn.words),
            sample_id=sample_id,
            source_turn_id=turn.source_turn_id,
        )
        token_frames = _allocate_token_frames(
            alignment,
            sample_id=sample_id,
            window_start_seconds=window.start_seconds,
            frame_rate_hz=frame_rate_hz,
            frame_count=frame_count,
            after_frame=last_event_frame,
            allow_time_extension=isinstance(window, ConversationMetadata),
        )
        start_frame = token_frames[0] - 1
        stop_frame = token_frames[-1] + 1
        proposed = [start_frame, *token_frames, stop_frame]
        if len(set(proposed)) != len(proposed) or any(
            events[frame].kind is not EventKind.IDLE for frame in proposed
        ):
            raise InvalidTimelineError(
                f"sample {sample_id!r}, turn {turn.source_turn_id!r}: "
                f"deterministic allocation collided at frames {proposed!r}."
            )
        events[start_frame] = TargetEvent(EventKind.START)
        frame_turns[start_frame] = turn.source_turn_id
        frame_words[start_frame] = alignment.source_word_indices[0]
        for frame, token in zip(token_frames, alignment.tokens):
            events[frame] = TargetEvent(EventKind.TEXT, token.token_id)
            frame_tokens[frame] = token
            frame_turns[frame] = turn.source_turn_id
            frame_words[frame] = token.source_word_index
        events[stop_frame] = TargetEvent(EventKind.STOP)
        frame_turns[stop_frame] = turn.source_turn_id
        frame_words[stop_frame] = alignment.source_word_indices[-1]
        utterances.append(alignment)
        last_event_frame = stop_frame

    frame_times = [
        window.start_seconds + frame / frame_rate_hz for frame in range(frame_count)
    ]
    targets = TargetEventSequence(events, sample_id=sample_id, frame_times=frame_times)
    causal = encode_causal_timeline(
        targets,
        control_tokens=control_tokens,
        thinker_bos_token_id=thinker_bos_token_id,
    )

    frames: list[FrameAlignment] = []
    active = False
    frame_seconds = 1.0 / frame_rate_hz
    for frame_index, (time_seconds, target) in enumerate(zip(frame_times, events)):
        if target.kind is EventKind.START:
            active = True
        elif target.kind is EventKind.STOP:
            active = False
        token = frame_tokens.get(frame_index)
        user_active = any(
            word.end_seconds > time_seconds
            and word.start_seconds < time_seconds + frame_seconds
            for word in record.user_words
        )
        frames.append(
            FrameAlignment(
                frame_index=frame_index,
                audio_time_seconds=time_seconds,
                user_active=user_active,
                target_event=target,
                decoded_token="" if token is None else token.decoded_token,
                state=ResponseState.ACTIVE if active else ResponseState.INACTIVE,
                source_word_index=frame_words.get(frame_index),
                source_turn_id=frame_turns.get(frame_index),
            )
        )

    return WindowTimeline(
        sample_id=sample_id,
        window_start_seconds=window.start_seconds,
        window_end_seconds=window.end_seconds,
        frame_rate_hz=frame_rate_hz,
        user_waveform=user_waveform,
        utterances=tuple(utterances),
        frames=tuple(frames),
        targets=targets,
        causal=causal,
        user_words=tuple(
            word
            for word in record.user_words
            if word.end_seconds > window.start_seconds
            and word.start_seconds < window.end_seconds
        ),
        audio_sample_rate_hz=record.sample_rate_hz,
    )


@dataclass(frozen=True)
class _InterruptionCandidate:
    utterance_index: int
    start_frame: int
    stop_frame: int
    user_frame: int
    user_start_seconds: float
    legal_cuts: tuple[int, ...]


def _random_unit(rng: object) -> float:
    if isinstance(rng, random.Random):
        return rng.random()
    try:
        import torch
    except ImportError as error:  # pragma: no cover - torch is a requirement
        raise TypeError("rng must be random.Random or torch.Generator.") from error
    if isinstance(rng, torch.Generator):
        return float(torch.rand((), generator=rng).item())
    raise TypeError("rng must be random.Random or torch.Generator.")


def _random_index(rng: object, size: int) -> int:
    if size <= 0:
        raise ValueError("cannot choose from an empty sequence.")
    if isinstance(rng, random.Random):
        return rng.randrange(size)
    try:
        import torch
    except ImportError as error:  # pragma: no cover - torch is a requirement
        raise TypeError("rng must be random.Random or torch.Generator.") from error
    if isinstance(rng, torch.Generator):
        return int(torch.randint(size, (), generator=rng).item())
    raise TypeError("rng must be random.Random or torch.Generator.")


def _interruption_candidates(
    timeline: WindowTimeline,
    *,
    min_assistant_frames: int,
) -> tuple[_InterruptionCandidate, ...]:
    candidates: list[_InterruptionCandidate] = []
    for utterance_index, utterance in enumerate(timeline.utterances):
        turn_frames = [
            frame.frame_index
            for frame in timeline.frames
            if frame.source_turn_id == utterance.source_turn_id
        ]
        start_frames = [
            frame
            for frame in turn_frames
            if timeline.targets.events[frame].kind is EventKind.START
        ]
        stop_frames = [
            frame
            for frame in turn_frames
            if timeline.targets.events[frame].kind is EventKind.STOP
        ]
        if len(start_frames) != 1 or len(stop_frames) != 1:
            continue
        start_frame = start_frames[0]
        stop_frame = stop_frames[0]

        following_words = [
            word
            for word in timeline.user_words
            if word.start_seconds >= utterance.source_end_seconds - 1e-9
        ]
        if following_words:
            user_start_seconds = following_words[0].start_seconds
            if any(
                later.source_start_seconds < user_start_seconds - 1e-9
                for later in timeline.utterances[utterance_index + 1 :]
            ):
                continue
            user_frame = max(
                0,
                _frame_for_time(
                    user_start_seconds,
                    window_start_seconds=timeline.window_start_seconds,
                    frame_rate_hz=timeline.frame_rate_hz,
                ),
            )
        else:
            active_frames = [
                frame.frame_index
                for frame in timeline.frames[stop_frame + 1 :]
                if frame.user_active
            ]
            if not active_frames:
                continue
            user_frame = active_frames[0]
            user_start_seconds = (
                timeline.window_start_seconds
                + user_frame / timeline.frame_rate_hz
            )

        if user_frame <= stop_frame or user_frame >= len(timeline.frames):
            continue
        # The cut replaces the first removed response frame with STOP.  At
        # least ``min_assistant_frames`` complete active frames remain between
        # START and the synthetic STOP, and at least one suffix frame is cut.
        legal_cuts = tuple(
            range(start_frame + 1 + min_assistant_frames, stop_frame)
        )
        if not legal_cuts:
            continue
        candidates.append(
            _InterruptionCandidate(
                utterance_index,
                start_frame,
                stop_frame,
                user_frame,
                user_start_seconds,
                legal_cuts,
            )
        )
    return tuple(candidates)


def _shift_waveform_suffix(
    waveform: Any,
    *,
    source_sample: int,
    destination_sample: int,
) -> Any:
    """Move a waveform suffix earlier and zero the vacated tail."""

    length = len(waveform)
    source_sample = min(max(source_sample, 0), length)
    destination_sample = min(max(destination_sample, 0), source_sample)
    if hasattr(waveform, "new_zeros"):
        shifted = waveform.new_zeros(waveform.shape)
        shifted[:destination_sample] = waveform[:destination_sample]
        shifted[destination_sample : destination_sample + length - source_sample] = (
            waveform[source_sample:]
        )
        return shifted
    if hasattr(waveform, "copy") and hasattr(waveform, "shape"):
        shifted = waveform.copy()
        shifted[...] = 0
        shifted[:destination_sample] = waveform[:destination_sample]
        shifted[destination_sample : destination_sample + length - source_sample] = (
            waveform[source_sample:]
        )
        return shifted
    shifted = [0 for _ in range(length)]
    shifted[:destination_sample] = list(waveform[:destination_sample])
    shifted[destination_sample : destination_sample + length - source_sample] = list(
        waveform[source_sample:]
    )
    return tuple(shifted) if isinstance(waveform, tuple) else shifted


def _shift_alignment(
    alignment: UtteranceAlignment,
    *,
    seconds: float,
) -> UtteranceAlignment:
    return replace(
        alignment,
        source_start_seconds=alignment.source_start_seconds - seconds,
        source_end_seconds=alignment.source_end_seconds - seconds,
        tokens=tuple(
            replace(
                token,
                source_start_seconds=token.source_start_seconds - seconds,
                source_end_seconds=token.source_end_seconds - seconds,
            )
            for token in alignment.tokens
        ),
    )


def augment_synthetic_interruption(
    timeline: WindowTimeline,
    *,
    config: InterruptionConfig,
    rng: object,
    control_tokens: ControlTokenIds,
    thinker_bos_token_id: int,
) -> WindowTimeline:
    """Splice a later user turn into an eligible assistant response.

    Randomness is wholly caller-owned.  This keeps dataset-worker seeding out
    of timeline logic and makes probability zero an identity-preserving no-op.
    """

    if config.probability == 0.0:
        return timeline
    if config.probability < 1.0 and _random_unit(rng) >= config.probability:
        return timeline
    candidates = _interruption_candidates(
        timeline, min_assistant_frames=config.min_assistant_frames
    )
    if not candidates:
        return timeline
    candidate = candidates[_random_index(rng, len(candidates))]
    cut_frame = candidate.legal_cuts[
        _random_index(rng, len(candidate.legal_cuts))
    ]
    shift_frames = candidate.user_frame - cut_frame
    frame_count = len(timeline.frames)
    cut_time = timeline.window_start_seconds + cut_frame / timeline.frame_rate_hz
    shift_seconds = candidate.user_start_seconds - cut_time

    selected = timeline.utterances[candidate.utterance_index]
    retained_frame_tokens = [
        frame
        for frame in timeline.frames[:cut_frame]
        if frame.source_turn_id == selected.source_turn_id
        and frame.target_event.kind is EventKind.TEXT
    ]
    retained_count = len(retained_frame_tokens)
    retained_tokens = tuple(selected.tokens[:retained_count])
    retained_word_indices = tuple(
        dict.fromkeys(token.source_word_index for token in retained_tokens)
    )
    retained_end = retained_tokens[-1].character_end
    truncated_tokens = tuple(
        replace(
            token,
            source_end_seconds=min(token.source_end_seconds, cut_time),
        )
        for token in retained_tokens
    )
    truncated = replace(
        selected,
        source_word_indices=retained_word_indices,
        source_end_seconds=cut_time,
        target_text=selected.target_text[:retained_end],
        token_ids=tuple(token.token_id for token in retained_tokens),
        tokens=truncated_tokens,
        normalizations=tuple(
            normalization
            for normalization in selected.normalizations
            if normalization.source_word_index in retained_word_indices
        ),
    )

    utterances: list[UtteranceAlignment] = []
    for index, utterance in enumerate(timeline.utterances):
        if index < candidate.utterance_index:
            utterances.append(utterance)
        elif index == candidate.utterance_index:
            utterances.append(truncated)
        elif utterance.source_start_seconds >= candidate.user_start_seconds - 1e-9:
            utterances.append(_shift_alignment(utterance, seconds=shift_seconds))
        else:
            utterances.append(utterance)

    shifted_user_words = tuple(
        replace(
            word,
            start_seconds=cut_time
            + (word.start_seconds - candidate.user_start_seconds),
            end_seconds=cut_time
            + (word.end_seconds - candidate.user_start_seconds),
        )
        if word.start_seconds >= candidate.user_start_seconds - 1e-9
        else word
        for word in timeline.user_words
    )
    fallback_user_activity = [False] * frame_count
    for destination in range(cut_frame, frame_count):
        source = candidate.user_frame + destination - cut_frame
        if source < frame_count:
            fallback_user_activity[destination] = timeline.frames[source].user_active
    for frame in timeline.frames[:cut_frame]:
        fallback_user_activity[frame.frame_index] = frame.user_active

    new_events: list[TargetEvent] = []
    source_frames: list[FrameAlignment | None] = []
    for destination in range(frame_count):
        if destination < cut_frame:
            source = timeline.frames[destination]
            event = source.target_event
        elif destination == cut_frame:
            source = timeline.frames[candidate.user_frame]
            event = TargetEvent(EventKind.STOP)
        else:
            source_index = candidate.user_frame + destination - cut_frame
            source = timeline.frames[source_index] if source_index < frame_count else None
            event = (
                TargetEvent(EventKind.IDLE)
                if source is None
                else source.target_event
            )
        new_events.append(event)
        source_frames.append(source)

    frame_times = [
        timeline.window_start_seconds + frame / timeline.frame_rate_hz
        for frame in range(frame_count)
    ]
    targets = TargetEventSequence(
        new_events,
        sample_id=timeline.sample_id,
        frame_times=frame_times,
    )
    causal = encode_causal_timeline(
        targets,
        control_tokens=control_tokens,
        thinker_bos_token_id=thinker_bos_token_id,
    )

    frames: list[FrameAlignment] = []
    active = False
    frame_seconds = 1.0 / timeline.frame_rate_hz
    for frame_index, (time_seconds, target, source) in enumerate(
        zip(frame_times, new_events, source_frames)
    ):
        if target.kind is EventKind.START:
            active = True
        elif target.kind is EventKind.STOP:
            active = False
        if shifted_user_words:
            user_active = any(
                word.end_seconds > time_seconds
                and word.start_seconds < time_seconds + frame_seconds
                for word in shifted_user_words
            )
        else:
            user_active = fallback_user_activity[frame_index]
        if frame_index == cut_frame:
            decoded_token = ""
            source_word_index = retained_word_indices[-1]
            source_turn_id = selected.source_turn_id
        elif source is None or target.kind is EventKind.IDLE:
            decoded_token = ""
            source_word_index = None
            source_turn_id = None
        else:
            decoded_token = source.decoded_token
            source_word_index = source.source_word_index
            source_turn_id = source.source_turn_id
        frames.append(
            FrameAlignment(
                frame_index=frame_index,
                audio_time_seconds=time_seconds,
                user_active=user_active,
                target_event=target,
                decoded_token=decoded_token,
                state=ResponseState.ACTIVE if active else ResponseState.INACTIVE,
                source_word_index=source_word_index,
                source_turn_id=source_turn_id,
            )
        )

    source_sample = round(
        (candidate.user_start_seconds - timeline.window_start_seconds)
        * timeline.audio_sample_rate_hz
    )
    destination_sample = round(
        (cut_time - timeline.window_start_seconds) * timeline.audio_sample_rate_hz
    )
    waveform = _shift_waveform_suffix(
        timeline.user_waveform,
        source_sample=source_sample,
        destination_sample=destination_sample,
    )
    interruption = InterruptionMetadata(
        source_turn_id=selected.source_turn_id,
        cut_frame=cut_frame,
        original_stop_frame=candidate.stop_frame,
        original_user_frame=candidate.user_frame,
        shifted_frames=shift_frames,
    )
    return replace(
        timeline,
        user_waveform=waveform,
        utterances=tuple(utterances),
        frames=tuple(frames),
        targets=targets,
        causal=causal,
        user_words=shifted_user_words,
        interruption=interruption,
    )


def format_timeline_table(timeline: WindowTimeline) -> str:
    """Render the requested short-sample inspection table."""

    header = "frame  time(s)  user  target  decoded token  state"
    rule = "-----  -------  ----  ------  -------------  --------"
    rows = [header, rule]
    for frame in timeline.frames:
        decoded = repr(frame.decoded_token) if frame.decoded_token else ""
        rows.append(
            f"{frame.frame_index:5d}  {frame.audio_time_seconds:7.3f}  "
            f"{'yes' if frame.user_active else 'no ':>4}  "
            f"{frame.target_event.kind.value:6}  {decoded:13}  "
            f"{frame.state.value}"
        )
    return "\n".join(rows)
