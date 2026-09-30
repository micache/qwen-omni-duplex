import pytest
import torch
import torch.nn.functional as F
import yaml

from duplex.batching import FrameBudgetBatchSampler
from duplex.contract import prompt_token_ids
from duplex.model import NextEventLossWeights, QwenDuplexThinker
from duplex.timeline import ControlTokenIds
from duplex.training import load_training_config
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
