import pytest
import torch
import torch.nn.functional as F
import yaml

from duplex.batching import FrameBudgetBatchSampler
from duplex.model import NextEventLossWeights, QwenDuplexThinker
from duplex.timeline import ControlTokenIds
from duplex.training import load_training_config


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
