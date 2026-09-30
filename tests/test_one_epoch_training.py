import pytest
import torch
import torch.nn.functional as F
import yaml

from duplex.batching import FrameBudgetBatchSampler
from duplex.contract import prompt_token_ids
from duplex.model import NextEventLossWeights, QwenDuplexThinker
from duplex.timeline import ControlTokenIds
from duplex.training import _load_adapter_weights, load_training_config
from generate import _context_token_ids
import numpy as np
from duplex.streaming import pad_audio_to_chunk_boundary, split_fixed_audio_chunks
from duplex.streaming import QwenDuplexStreamer
from types import SimpleNamespace
from duplex.turn_packed import PackedConversation, build_turn_packed_timeline


def test_empty_inference_context_does_not_add_untrained_chat_tokens():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{"role": "system", "content": "Explicit prefix"}]
            return [151644, 123, 151645]

    tokenizer = Tokenizer()
    assert prompt_token_ids(tokenizer, "") == []
    assert _context_token_ids(tokenizer, "", "") == []
    assert _context_token_ids(tokenizer, "Explicit prefix", "") == [151644, 123, 151645]


def test_full_chunk_padding_preserves_speech_boundary_and_answer_targets():
    waveform = np.linspace(-0.1, 0.1, 35000, dtype=np.float32)
    padded = pad_audio_to_chunk_boundary(waveform)
    assert len(padded) == 64000
    np.testing.assert_array_equal(padded[:len(waveform)], waveform)
    assert not padded[len(waveform):].any()
    chunks = split_fixed_audio_chunks(padded)
    assert [chunk.valid_samples for chunk in chunks] == [32000, 32000]
    class Tokenizer:
        def encode(self, text, **kwargs):
            return [4, 5, 6]
        def decode(self, ids, **kwargs):
            return 'answer' if len(ids) == 3 else 'part'
    conversation = PackedConversation('padded', padded, 32000, 'answer',
                                      user_valid_samples=len(waveform))
    timeline = build_turn_packed_timeline(conversation, tokenizer=Tokenizer(),
        control_tokens=ControlTokenIds(idle=1, start=2, stop=3, thinker_vocab_size=11),
        thinker_bos_token_id=2, valid_frame_count=150)
    start = round(len(waveform) * 25 / 16000)
    assert timeline.targets.events[start].kind.value == 'START'
    assert [event.text_id for event in timeline.targets.events[start+1:start+4]] == [4,5,6]
    assert timeline.targets.events[start+4].kind.value == 'STOP'
    np.testing.assert_array_equal(timeline.user_waveform[:64000], padded)


def test_prompt_is_rejected_for_empty_context_adapter():
    streamer = QwenDuplexStreamer.__new__(QwenDuplexStreamer)
    streamer.model = SimpleNamespace(requires_empty_context=True)
    with pytest.raises(ValueError, match='context must be empty'):
        streamer.run(np.zeros(32000, dtype=np.float32), sample_id='prompt', context_token_ids=[123])


def test_audio_gain_scales_shared_encoder_output_once_including_cache(tmp_path):
    class Tower(torch.nn.Module):
        def _get_feat_extract_output_lengths(self, lengths):
            return lengths // 2, lengths // 4
    class Thinker(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.audio_tower = Tower()
            self.calls = 0
        def get_audio_features(self, **kwargs):
            self.calls += 1
            return SimpleNamespace(last_hidden_state=torch.ones(50, 3))
    model = QwenDuplexThinker.__new__(QwenDuplexThinker)
    torch.nn.Module.__init__(model)
    model.thinker = Thinker()
    arguments = dict(input_features=torch.zeros(1, 128, 200),
        feature_attention_mask=torch.ones(1, 200, dtype=torch.bool),
        preconv_feature_lengths=torch.tensor([200]), audio_chunk_counts=torch.tensor([1]),
        current_attention=torch.ones(1, 50, dtype=torch.bool), timeline_length=50,
        hidden_width=3, device=torch.device('cpu'), dtype=torch.float32,
        audio_cache_keys=('sample',))
    model.audio_gain = 1.0
    torch.testing.assert_close(model._restore_audio(**arguments), torch.ones(1, 50, 3))
    model.audio_gain = 0.03
    model.audio_cache_dir = tmp_path
    expected = torch.full((1, 50, 3), 0.03)
    torch.testing.assert_close(model._restore_audio(**arguments), expected)
    calls = model.thinker.calls
    torch.testing.assert_close(model._restore_audio(**arguments), expected)
    assert model.thinker.calls == calls


def test_response_gain_uses_past_controls_and_matches_scalar_stream_state():
    model = QwenDuplexThinker.__new__(QwenDuplexThinker)
    torch.nn.Module.__init__(model)
    model.control_tokens = ControlTokenIds(idle=1, start=2, stop=3, thinker_vocab_size=11)
    model.response_audio_gain = 0.03
    controls = torch.tensor([[2,1,2,1,3,1]])
    mask = torch.tensor([[False,True,True,True,True,True]])
    active = model.response_active_from_inputs(controls, mask)
    assert active.tolist() == [[False,False,True,True,False,False]]
    audio = torch.randn(1,6,3).bfloat16()
    batched = model.scale_audio_for_response(audio, active)
    streamed = torch.cat([model.scale_audio_for_response(audio[:,i:i+1], bool(active[0,i]))
                          for i in range(6)], dim=1)
    torch.testing.assert_close(batched, streamed, rtol=0, atol=0)


@pytest.mark.parametrize('field,value', [('fusion_gain',0.03), ('response_fusion_gain',0.03)])
def test_adapter_reload_rejects_a_different_audio_gain_before_loading_weights(tmp_path, field, value):
    saved = {'dataset_view':'InstructS2SFirstTurn', 'pad_to_full_chunks':True,
             'encoder_precision':'float32', 'fusion_gain':1.0, 'response_fusion_gain':1.0}
    saved[field] = value
    (tmp_path / 'duplex_config.yaml').write_text(yaml.safe_dump({'audio_input':saved}))
    model = SimpleNamespace(pad_audio_to_full_chunks=True, audio_encoder_fp32=True,
                            requires_empty_context=True, audio_gain=1.0, response_audio_gain=1.0)
    with pytest.raises(ValueError, match='audio contract differs'):
        _load_adapter_weights(model, tmp_path)


def test_frame_budget_keeps_all_samples_and_bounds_padding():
    lengths = [27, 40, 135, 566, 777, 99, 80] * 100
    sampler = FrameBudgetBatchSampler(lengths, max_frames=3360,
                                      max_batch_size=128, seed=212)
    for epoch in (0, 1):
        sampler.set_epoch(epoch)
        indices = [index for batch in sampler for index in batch]
        assert sorted(indices) == list(range(len(lengths)))
        assert len(sampler) == len(list(sampler))
        assert all(len(batch) <= 128 and len(batch) * max(lengths[i] for i in batch) <= 3360
                   for batch in sampler)
    other = FrameBudgetBatchSampler(lengths, max_frames=3360,
                                    max_batch_size=128, seed=212)
    sampler.set_epoch(0)
    assert list(sampler) == list(other)


@pytest.mark.parametrize("epochs,valid", [(1, True), (3, True), (2, False), (0, False)])
def test_explicit_epoch_budget(tmp_path, epochs, valid):
    config = yaml.safe_load(open("configs/turn_packed_main.yaml"))
    config["training"]["num_train_epochs"] = epochs
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    if valid:
        assert load_training_config(path)["training"]["num_train_epochs"] == epochs
    else:
        with pytest.raises(ValueError, match="one or three epochs"):
            load_training_config(path)


def test_flat_cross_entropy_preserves_weighted_loss_and_gradients():
    model = QwenDuplexThinker.__new__(QwenDuplexThinker)
    torch.nn.Module.__init__(model)
    model.control_tokens = ControlTokenIds(idle=1, start=2, stop=3, thinker_vocab_size=11)
    model.loss_weights = NextEventLossWeights(text=1, idle=0.1, start=4, stop=4)
    logits = torch.randn(2, 7, 11, generator=torch.Generator().manual_seed(3), requires_grad=True)
    reference_logits = logits.detach().clone().requires_grad_(True)
    labels = torch.tensor([[1, 2, 4, 5, 3, 1, 1], [1, 2, 6, 3, -100, -100, -100]])
    actual = model._weighted_loss(logits, labels)[0]
    losses = F.cross_entropy(reference_logits.transpose(1, 2), labels,
                            reduction="none", ignore_index=-100)
    weights = torch.zeros_like(losses)
    for name, mask in model._target_groups(labels).items():
        weights += mask * getattr(model.loss_weights, name)
    reference = (losses * weights).sum() / weights.sum()
    torch.testing.assert_close(actual, reference)
    actual.backward()
    reference.backward()
    torch.testing.assert_close(logits.grad, reference_logits.grad)
