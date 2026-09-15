from __future__ import annotations

import numpy as np
import torch

from duplex.contract import DEFAULT_SYSTEM_PROMPT, frame_event_inputs, prompt_token_ids
from duplex.dataset import ConversationMetadata, Split
from duplex.dataset import ManifestEntry
from duplex.conversations import CompleteConversationDataset, ConversationLength
from duplex.conversations import exact_audio_frame_count
from duplex.model import NextEventLossWeights
from duplex.streaming import split_fixed_audio_chunks, DuplexGenerationResult, StreamingResult, QwenDuplexStreamer
from duplex.timeline import ResponseState
from duplex.timeline import EventKind, build_window_timeline
from test_timeline_alignment import CharacterTokenizer, assistant, user, record, controls
from test_model import wrapper
from types import SimpleNamespace
import soundfile as sf


def complete(words, users=(), duration=5.3, frames=None):
    source = record(assistant_words=words, user_words=users, duration=duration)
    return build_window_timeline(source,
        ConversationMetadata(source.conversation_id, Split.TRAIN, 0.0, duration),
        tokenizer=CharacterTokenizer(), control_tokens=controls(),
        thinker_bos_token_id=11, frame_rate_hz=25, valid_frame_count=frames)


def test_all_response_lengths_and_internal_boundaries_are_kept():
    for start, end, text in ((0.5, 1.5, "Hi"), (1.8, 2.4, "Hello"),
                             (1.5, 4.3, "Long answer")):
        timeline = complete((assistant(text, start, end),))
        kinds = [event.kind for event in timeline.targets.events]
        assert kinds.count(EventKind.START) == kinds.count(EventKind.STOP) == 1
        assert kinds.count(EventKind.TEXT) == len(timeline.utterances[0].token_ids)
        assert sum(timeline.causal.bootstrap_mask) == 1
        assert not any(timeline.causal.bootstrap_mask[1:])


def test_very_short_annotated_word_retains_all_tokens():
    timeline = complete((assistant("Hi", 1.0, 1.02),))
    kinds = [event.kind for event in timeline.targets.events]
    assert kinds.count(EventKind.START) == kinds.count(EventKind.STOP) == 1
    assert kinds.count(EventKind.TEXT) == len(timeline.utterances[0].token_ids)


def test_multiple_turns_are_one_complete_timeline():
    value = complete((assistant("Hi", 1.1, 1.7), assistant("Bye", 3.0, 3.8)),
                     (user("Q1", 0.2, 0.6), user("Q2", 2.1, 2.6)))
    assert len(value.targets.events) == round(5.3 * 25)
    kinds = [event.kind for event in value.targets.events]
    assert kinds.count(EventKind.START) == kinds.count(EventKind.STOP) == 2
    assert kinds.count(EventKind.TEXT) == 5


def test_dataset_index_returns_one_complete_conversation(monkeypatch):
    import duplex.conversations as conversations
    import scripts.run_session11_diagnostic as diagnostic
    source = record(assistant_words=(assistant("Hi", 1.0, 1.5), assistant("Bye", 3.0, 3.7)),
                    user_words=(user("Q", 0.2, 0.7), user("Q", 2.1, 2.5)), duration=5.3)
    monkeypatch.setattr(conversations, "load_conversation", lambda *args, **kwargs: source)
    monkeypatch.setattr(diagnostic, "_infer_user_words", lambda record, options: record.user_words)
    model = SimpleNamespace(control_tokens=controls(),
        base_thinker=SimpleNamespace(config=SimpleNamespace(bos_token_id=11)))
    processor = SimpleNamespace(tokenizer=CharacterTokenizer())
    entry = ManifestEntry(source.conversation_id, __import__("pathlib").Path("fixture.wav"), 5.3)
    dataset = CompleteConversationDataset([entry], root=".", split=Split.TRAIN,
        config={"data": {}, "diagnostic": {}}, processor=processor, model=model,
        preflight_lengths={source.conversation_id: ConversationLength(source.conversation_id, Split.TRAIN, 5.3, 132, 135)})
    assert len(dataset) == 1
    item = dataset[0]
    assert item.conversation_id == source.conversation_id
    assert item.valid_sequence_length == 132
    assert len(item.targets.events) == 132
    assert len(item.utterances) == 2


def test_final_partial_block_and_exact_frame_count():
    chunks = split_fixed_audio_chunks(np.zeros(16_000 * 4 + 8_000, dtype=np.float32))
    assert [chunk.valid_samples for chunk in chunks] == [32_000, 32_000, 8_000]
    assert chunks[-1].end_time_s == chunks[-1].available_time_s == 4.5
    value = complete((assistant("Hi", 1.0, 1.5),), duration=5.3, frames=129)
    assert len(value.targets.events) == 129


def test_very_short_partial_block_uses_valid_encoder_mask():
    class Extractor:
        n_fft = 400
        hop_length = 160
        def __call__(self, waveforms, **kwargs):
            assert len(waveforms[0]) >= 401
            return {"feature_attention_mask": torch.ones((1, 3), dtype=torch.long)}
    class AudioTower:
        def _get_feat_extract_output_lengths(self, lengths):
            first = (lengths - 1) // 2 + 1
            return first, (first - 2) // 2 + 1
    assert exact_audio_frame_count(np.zeros(2, dtype=np.float32), Extractor(), AudioTower()) == 1


def test_shared_prompt_first_frame_and_capped_weights():
    class PromptTokenizer:
        def encode(self, text, add_special_tokens=True):
            return [1, 2, 3]
    tokenizer = PromptTokenizer()
    assert prompt_token_ids(tokenizer, DEFAULT_SYSTEM_PROMPT)
    assert frame_event_inputs(None, bos_token_id=11, control_ids=(6, 7, 8)) == (11, True, 0, False)
    value = complete((assistant("Hi", 1.0, 1.5),))
    assert (value.causal.text_ids[0], value.causal.text_mask[0],
            value.causal.control_ids[0], value.causal.control_mask[0]) == (11, True, 0, False)
    weights = NextEventLossWeights.capped_inverse_sqrt(
        {"text": 100, "idle": 10000, "start": 2, "stop": 2})
    assert max(weights.as_dict().values()) / min(weights.as_dict().values()) <= 5.00001


def test_full_and_cached_teacher_forced_logits_match_with_prompt_positions():
    class CacheModel(torch.nn.Module):
        def forward(self, *, inputs_embeds, attention_mask, position_ids,
                    past_key_values=None, use_cache=None, return_dict=True):
            offset, length = (torch.zeros_like(inputs_embeds[:, :1]), 0) if past_key_values is None else past_key_values
            assert attention_mask.shape[1] == length + inputs_embeds.shape[1]
            assert torch.equal(position_ids, torch.arange(length, length + inputs_embeds.shape[1])[None, :])
            accumulated = inputs_embeds.cumsum(dim=1) + offset
            return SimpleNamespace(last_hidden_state=accumulated,
                                   past_key_values=(accumulated[:, -1:], length + inputs_embeds.shape[1]))
    model = wrapper()
    model.thinker.model = CacheModel()
    model.eval()
    values = [None, 6, 7, 2, 8]
    inputs = [frame_event_inputs(value, bos_token_id=1, control_ids=(6, 7, 8)) for value in values]
    text_ids = torch.tensor([[value[0] for value in inputs]])
    text_mask = torch.tensor([[value[1] for value in inputs]])
    control_ids = torch.tensor([[value[2] for value in inputs]])
    control_mask = torch.tensor([[value[3] for value in inputs]])
    prompt = torch.tensor([[5, 4]])
    full = model(text_ids=text_ids, text_mask=text_mask, control_ids=control_ids,
                 control_mask=control_mask, attention_mask=torch.ones((1, 5), dtype=torch.bool),
                 context_ids=prompt, position_ids=torch.arange(5)[None, :]).logits
    model.thinker.config.bos_token_id = 1
    streamer = QwenDuplexStreamer(model, SimpleNamespace(tokenizer=SimpleNamespace(decode=lambda *args, **kwargs: "")))
    cache, prefix_length = streamer._prefill(prompt[0].tolist())
    assert prefix_length == 2
    cached = []
    for frame in range(5):
        _, _, _, hidden, cache, _ = streamer._step(
            audio_feature=torch.zeros((1, 1, 3)), previous_event_id=values[frame],
            cache=cache, sequence_length=prefix_length + frame,
            state=ResponseState.INACTIVE)
        cached.append(model.base_thinker.lm_head(hidden))
    torch.testing.assert_close(torch.cat(cached, dim=1), full, rtol=0, atol=1e-5)


def test_generate_audio_path_returns_public_structure(tmp_path, monkeypatch):
    import duplex.streaming as streaming
    path = tmp_path / "question-silence.wav"
    sf.write(path, np.zeros(16_000 * 3, dtype=np.float32), 16_000)
    model = wrapper()
    model.gradient_checkpointing_disable = lambda: None
    class Tokenizer:
        def encode(self, text, add_special_tokens=True):
            return [1]
        def decode(self, values, **kwargs):
            return "Hello" if values else ""
    model.processor = SimpleNamespace(tokenizer=Tokenizer())
    seen_blocks = []
    seen_positions = []
    def fake_audio(self, chunk):
        seen_blocks.append((chunk.index, chunk.valid_samples, chunk.available_time_s))
        frames = 50 if chunk.valid_samples == 32_000 else 25
        return torch.zeros((1, frames, 3))
    def fake_step(self, *, audio_feature, previous_event_id, cache, sequence_length, state):
        seen_positions.append(sequence_length)
        selected = {52: 7, 53: 2, 54: 8}.get(sequence_length, 6)
        return selected, selected, False, torch.zeros((1, 1, 3)), sequence_length + 1, 0.0
    monkeypatch.setattr(streaming.QwenDuplexStreamer, "_audio_features", fake_audio)
    monkeypatch.setattr(streaming.QwenDuplexStreamer, "_step", fake_step)
    result = model.generate(path)
    assert isinstance(result, DuplexGenerationResult)
    assert result.text == "Hello" and len(result.events) == 75 and len(result.segments) == 1
    assert result.first_word_time_s == result.last_word_time_s == 3.0
    assert seen_blocks == [(0, 32_000, 2.0), (1, 16_000, 3.0)]
    assert seen_positions == list(range(1, 76))


def test_padded_positions_have_zero_loss_weight():
    model = wrapper()
    logits = torch.randn(1, 4, 9)
    labels = torch.tensor([[6, 7, -100, -100]])
    baseline = model._weighted_loss(logits[:, :2], labels[:, :2])
    padded = model._weighted_loss(logits, labels)
    torch.testing.assert_close(baseline[0], padded[0])
    assert padded[-1].item() == baseline[-1].item()


def test_unchanged_frozen_audio_uses_simple_embedding_cache(tmp_path):
    model = wrapper()
    model.audio_cache_dir = tmp_path
    model.thinker.audio_tower.flattened = torch.ones((4, 3))
    inputs = {"text_ids": torch.tensor([[1, 2, 3, 4]]),
              "text_mask": torch.ones((1, 4), dtype=torch.bool),
              "control_ids": torch.zeros((1, 4), dtype=torch.long),
              "control_mask": torch.zeros((1, 4), dtype=torch.bool),
              "attention_mask": torch.ones((1, 4), dtype=torch.bool),
              "input_features": torch.zeros((1, 4, 3)),
              "feature_attention_mask": torch.ones((1, 4), dtype=torch.bool),
              "preconv_feature_lengths": torch.tensor([4]),
              "audio_cache_keys": ("unchanged",)}
    model(**inputs)
    model(**inputs)
    assert model.thinker.audio_tower.calls == 1
    assert (tmp_path / "unchanged.pt").exists()
