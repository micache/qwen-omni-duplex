from __future__ import annotations

from pathlib import Path

import pytest

from duplex.dataset import (
    ConversationRecord,
    Speaker,
    Split,
    WindowKind,
    WindowMetadata,
    WordSpan,
)
from duplex.timeline import (
    ControlTokenIds,
    EventKind,
    InvalidTimelineError,
    ResponseState,
    align_assistant_utterance,
    build_window_timeline,
    format_timeline_table,
)


class CharacterTokenizer:
    """Small reversible tokenizer with optional offset support."""

    def __init__(self, *, offsets: bool = True) -> None:
        self.offsets = offsets
        self.piece_to_id: dict[str, int] = {}
        self.id_to_piece: dict[int, str] = {}

    def _pieces(self, text: str) -> list[tuple[str, tuple[int, int]]]:
        pieces: list[tuple[str, tuple[int, int]]] = []
        index = 0
        while index < len(text):
            end = index + 1
            if text[index] == " " and end < len(text):
                end += 1
            pieces.append((text[index:end], (index, end)))
            index = end
        return pieces

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        result = []
        for piece, _ in self._pieces(text):
            if piece not in self.piece_to_id:
                token_id = len(self.piece_to_id) + 20
                self.piece_to_id[piece] = token_id
                self.id_to_piece[token_id] = piece
            result.append(self.piece_to_id[piece])
        return result

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
    ) -> dict:
        if not self.offsets:
            raise NotImplementedError
        ids = self.encode(text, add_special_tokens=add_special_tokens)
        return {
            "input_ids": ids,
            "offset_mapping": [offset for _, offset in self._pieces(text)],
        }

    def decode(self, token_ids: list[int], **_: object) -> str:
        return "".join(self.id_to_piece[token_id] for token_id in token_ids)


class LossyTokenizer(CharacterTokenizer):
    def decode(self, token_ids: list[int], **_: object) -> str:
        return super().decode(token_ids).replace("é", "e")


def assistant(text: str, start: float, end: float) -> WordSpan:
    return WordSpan(text, start, end, Speaker.ASSISTANT)


def user(text: str, start: float, end: float) -> WordSpan:
    return WordSpan(text, start, end, Speaker.USER)


def record(
    *,
    assistant_words: tuple[WordSpan, ...] = (),
    user_words: tuple[WordSpan, ...] = (),
    duration: float = 4.0,
) -> ConversationRecord:
    return ConversationRecord(
        conversation_id="conversation-04",
        duration_seconds=duration,
        sample_rate_hz=100,
        source_sample_rate_hz=100,
        user_waveform=[float(index) for index in range(round(duration * 100))],
        assistant_reference_path=Path("reference.wav"),
        assistant_waveform=None,
        user_words=user_words,
        assistant_words=assistant_words,
    )


def window(start: float = 0.0) -> WindowMetadata:
    return WindowMetadata(
        "conversation-04",
        Split.TRAIN,
        start,
        start + 2.0,
        WindowKind.RANDOM,
    )


def controls() -> ControlTokenIds:
    return ControlTokenIds(10, 11, 12, thinker_vocab_size=1_000)


def build(
    source: ConversationRecord,
    *,
    crop: WindowMetadata | None = None,
    tokenizer: object | None = None,
):
    return build_window_timeline(
        source,
        window() if crop is None else crop,
        tokenizer=CharacterTokenizer() if tokenizer is None else tokenizer,
        control_tokens=controls(),
        thinker_bos_token_id=1,
        frame_rate_hz=25,
    )


def test_context_tokenization_preserves_multibpe_punctuation_and_unicode() -> None:
    tokenizer = CharacterTokenizer()
    words = (
        (0, assistant("unbelievable", 0.20, 0.70)),
        (1, assistant(",", 0.70, 0.76)),
        (2, assistant("café", 0.80, 1.20)),
        (3, assistant("世界", 1.22, 1.55)),
        (4, assistant("!", 1.55, 1.60)),
    )

    aligned = align_assistant_utterance(
        tokenizer,
        words,
        sample_id="unicode-sample",
        source_turn_id="turn-2",
    )

    assert aligned.target_text == "unbelievable, café 世界!"
    assert tokenizer.decode(list(aligned.token_ids)) == aligned.target_text
    assert len(aligned.tokens) == len(aligned.token_ids)
    assert len(aligned.tokens) > len(words)
    assert any(token.decoded_token.startswith(" ") for token in aligned.tokens)
    assert [token.source_word_index for token in aligned.tokens] == sorted(
        token.source_word_index for token in aligned.tokens
    )
    assert {token.source_word_index for token in aligned.tokens} == {0, 1, 2, 3, 4}
    assert aligned.alignment_method == "tokenizer_offsets"
    assert aligned.normalizations


def test_explicit_leading_space_is_preserved_exactly() -> None:
    tokenizer = CharacterTokenizer()
    aligned = align_assistant_utterance(
        tokenizer,
        ((0, assistant(" leading", 0.2, 0.8)),),
        sample_id="leading-space",
        source_turn_id="assistant-0",
    )

    assert aligned.target_text == " leading"
    assert tokenizer.decode(list(aligned.token_ids)) == " leading"


def test_incremental_round_trip_alignment_is_used_without_offsets() -> None:
    tokenizer = CharacterTokenizer(offsets=False)
    aligned = align_assistant_utterance(
        tokenizer,
        (
            (4, assistant("Hello", 0.2, 0.6)),
            (5, assistant("world", 0.7, 1.1)),
        ),
        sample_id="fallback",
        source_turn_id="assistant-1",
    )

    assert aligned.alignment_method == "incremental_decode_round_trip"
    assert tokenizer.decode(list(aligned.token_ids)) == "Hello world"
    assert len(aligned.tokens) == len(aligned.token_ids)
    assert [token.source_word_index for token in aligned.tokens] == sorted(
        token.source_word_index for token in aligned.tokens
    )


def test_single_token_offset_shape_is_not_treated_as_a_batch() -> None:
    tokenizer = CharacterTokenizer()
    aligned = align_assistant_utterance(
        tokenizer,
        ((3, assistant("x", 0.2, 0.8)),),
        sample_id="one-token",
        source_turn_id="assistant-0",
    )

    assert aligned.token_ids == (tokenizer.piece_to_id["x"],)
    assert aligned.tokens[0].source_word_index == 3


def test_round_trip_failure_names_sample_turn_and_words() -> None:
    with pytest.raises(ValueError) as error:
        align_assistant_utterance(
            LossyTokenizer(),
            ((7, assistant("café", 0.2, 0.8)),),
            sample_id="bad-unicode",
            source_turn_id="turn-9",
        )
    message = str(error.value)
    assert "bad-unicode" in message
    assert "turn-9" in message
    assert "[7]" in message
    assert "round trip failed" in message


def test_tokens_expand_beyond_colliding_word_frames_without_overwrite() -> None:
    timeline = build(
        record(
            assistant_words=(
                assistant("abc", 0.20, 0.24),
                assistant("def", 1.00, 1.04),
            )
        )
    )
    lexical_frames = [
        frame.frame_index
        for frame in timeline.frames
        if frame.target_event.kind is EventKind.TEXT
    ]

    assert len(lexical_frames) == len(timeline.utterances[0].token_ids)
    assert lexical_frames == sorted(set(lexical_frames))
    # Each source word initially intersects only one frame (5 and 25), so the
    # extra BPE pieces have been spread across valid time in the same turn.
    assert lexical_frames[:3] == [5, 6, 7]
    assert lexical_frames[-3:] == [23, 24, 25]
    assert timeline.frames[lexical_frames[0] - 1].target_event.kind is EventKind.START
    assert timeline.frames[lexical_frames[-1] + 1].target_event.kind is EventKind.STOP


def test_impossibly_short_assistant_span_is_invalid_with_details() -> None:
    with pytest.raises(InvalidTimelineError) as error:
        build(record(assistant_words=(assistant("crowded", 0.20, 0.28),)))
    message = str(error.value)
    assert "conversation-04" in message
    assert "assistant:0" in message
    assert "lexical tokens" in message
    assert "adjacent START/STOP" in message


def test_crop_edges_omit_partial_turns_but_keep_complete_later_turn() -> None:
    source = record(
        assistant_words=(
            assistant("cut", 0.80, 1.10),
            assistant("edge", 1.15, 1.35),
            assistant("whole", 1.80, 2.20),
            assistant("ending", 2.50, 3.10),
        ),
        user_words=(user("next", 1.50, 1.65), user("then", 2.30, 2.40)),
    )
    timeline = build(source, crop=window(1.0))

    assert [utterance.target_text for utterance in timeline.utterances] == ["whole"]
    kinds = [frame.target_event.kind for frame in timeline.frames]
    assert kinds.count(EventKind.START) == 1
    assert kinds.count(EventKind.STOP) == 1
    assert kinds.index(EventKind.START) < kinds.index(EventKind.STOP)
    assert timeline.frames[0].target_event.kind is EventKind.IDLE
    assert timeline.frames[-1].target_event.kind is EventKind.IDLE


def test_empty_assistant_span_is_idle_and_causal_without_target_leakage() -> None:
    timeline = build(record(user_words=(user("speaking", 0.4, 0.8),)))

    assert not timeline.utterances
    assert all(frame.target_event.kind is EventKind.IDLE for frame in timeline.frames)
    assert timeline.causal.labels == [10] * 50
    assert timeline.causal.text_ids[0] == 1
    assert timeline.causal.control_mask[0] is False
    assert timeline.causal.control_ids[1:] == [10] * 49
    assert timeline.causal.control_mask[1:] == [True] * 49
    assert [frame.user_active for frame in timeline.frames].count(True) == 10


def test_frame_metadata_monotonic_no_collisions_and_shifted_inputs() -> None:
    timeline = build(
        record(
            assistant_words=(assistant("hello", 0.20, 0.80),),
            user_words=(user("yes", 1.00, 1.30),),
        )
    )
    occupied = [
        frame
        for frame in timeline.frames
        if frame.target_event.kind is not EventKind.IDLE
    ]

    assert [frame.frame_index for frame in occupied] == sorted(
        {frame.frame_index for frame in occupied}
    )
    lexical = [frame for frame in occupied if frame.target_event.kind is EventKind.TEXT]
    assert [frame.source_word_index for frame in lexical] == [0] * len(lexical)
    assert all(frame.source_turn_id for frame in occupied)
    assert timeline.frames[occupied[0].frame_index].state is ResponseState.ACTIVE
    assert timeline.frames[occupied[-1].frame_index].state is ResponseState.INACTIVE
    for frame in range(1, 50):
        previous = timeline.targets.events[frame - 1]
        if previous.kind is EventKind.TEXT:
            assert timeline.causal.text_ids[frame] == previous.text_id
            assert timeline.causal.text_mask[frame]
            assert not timeline.causal.control_mask[frame]
        else:
            assert timeline.causal.control_ids[frame] == controls().for_event(
                previous.kind
            )
            assert timeline.causal.control_mask[frame]
            assert not timeline.causal.text_mask[frame]


def test_user_waveform_crop_and_inspection_table() -> None:
    timeline = build(
        record(
            assistant_words=(assistant("ok", 1.20, 1.60),),
            user_words=(user("question", 1.00, 1.10),),
        ),
        crop=window(1.0),
    )

    assert timeline.user_waveform[0] == 100.0
    assert len(timeline.user_waveform) == 200
    rendered = format_timeline_table(timeline)
    assert "frame  time(s)  user  target  decoded token  state" in rendered
    assert "START" in rendered
    assert "TEXT" in rendered
    assert "STOP" in rendered
    assert "yes" in rendered


def test_real_cached_tokenizer_round_trip_if_available() -> None:
    transformers = pytest.importorskip("transformers")
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    tokenizer_files = sorted(
        cache_root.glob("models--*/snapshots/*/tokenizer_config.json")
    )
    for tokenizer_file in tokenizer_files:
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                tokenizer_file.parent,
                local_files_only=True,
            )
            aligned = align_assistant_utterance(
                tokenizer,
                (
                    (0, assistant("Hello", 0.2, 0.6)),
                    (1, assistant(",", 0.6, 0.7)),
                    (2, assistant("café", 0.8, 1.2)),
                    (3, assistant("世界", 1.3, 1.7)),
                    (4, assistant("!", 1.7, 1.8)),
                ),
                sample_id="real-local-tokenizer",
                source_turn_id="turn-0",
            )
        except (OSError, ValueError):
            continue
        assert tokenizer.decode(
            list(aligned.token_ids),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ) == aligned.target_text
        assert len(aligned.tokens) == len(aligned.token_ids)
        return
    pytest.skip("no usable tokenizer is installed in the local Hugging Face cache")
