"""Global batches and weighted gradients must survive uneven GPU partitions."""

import pytest
import torch

from duplex.distributed import split_global_batch
from duplex.model import NextEventLossWeights, QwenDuplexThinker
from duplex.timeline import ControlTokenIds


@pytest.mark.parametrize("size", [1, 2, 3, 5, 11, 26])
def test_split_preserves_examples_and_ragged_audio(size):
    counts = torch.arange(1, size + 1)
    batch = {
        "audio_chunk_counts": counts,
        "input_features": torch.arange(int(counts.sum()))[:, None, None],
        "feature_attention_mask": torch.ones(int(counts.sum()), 3),
        "preconv_feature_lengths": torch.full((int(counts.sum()),), 3),
        "labels": torch.arange(size)[:, None],
        "audio_cache_keys": tuple(str(i) for i in range(size)),
    }
    parts = [split_global_batch(batch, rank, 4) for rank in range(4)]
    real = [part for part in parts if not part["_empty_rank"]]
    assert torch.equal(torch.cat([part["labels"] for part in real]), batch["labels"])
    assert torch.equal(torch.cat([part["input_features"] for part in real]), batch["input_features"])
    for part in parts:
        assert len(part["input_features"]) == int(part["audio_chunk_counts"].sum())
        if part["_empty_rank"]:
            assert (part["labels"] == -100).all()


@pytest.mark.parametrize("size", [1, 3, 11])
def test_weighted_ddp_gradient_matches_original_global_batch(size):
    model = QwenDuplexThinker.__new__(QwenDuplexThinker)
    torch.nn.Module.__init__(model)
    model.control_tokens = ControlTokenIds(idle=1, start=2, stop=3, thinker_vocab_size=11)
    model.loss_weights = NextEventLossWeights(text=1, idle=0.1, start=4, stop=4)
    logits = torch.randn(size, 7, 11, generator=torch.Generator().manual_seed(9), requires_grad=True)
    labels = torch.tensor([[1, 2, 4, 5, 3, 1, -100]]).repeat(size, 1)
    if size > 1:
        labels[-1, 3:] = -100
    reference, *_, global_weight = model._weighted_loss(logits, labels)
    reference.backward()
    partitioned = logits.detach().clone().requires_grad_(True)
    losses = []
    for rank in range(4):
        start, end = size * rank // 4, size * (rank + 1) // 4
        if start == end:
            local_logits = partitioned[:1]
            local_labels = torch.full_like(labels[:1], -100)
        else:
            local_logits, local_labels = partitioned[start:end], labels[start:end]
        local_loss, *_, local_weight = model._weighted_loss(local_logits, local_labels)
        losses.append(local_loss * 4 * local_weight / global_weight)
    # DDP sums rank gradients and divides by world size.
    (sum(losses) / 4).backward()
    torch.testing.assert_close(partitioned.grad, logits.grad)
