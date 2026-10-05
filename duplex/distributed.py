"""Split an existing global frame-budget batch without changing its examples."""

import torch


class SplitGlobalBatchCollator:
    """Augment the full batch first, retaining cross-example interruption donors."""

    def __init__(self, collator, rank, world_size):
        self.collator = collator
        self.rank = rank
        self.world_size = world_size

    def __call__(self, items):
        return split_global_batch(self.collator(items), self.rank, self.world_size)


def split_global_batch(batch, rank, world_size):
    counts = batch["audio_chunk_counts"]
    size = len(counts)
    start = size * rank // world_size
    end = size * (rank + 1) // world_size
    empty = start == end
    # A zero-target duplicate keeps all DDP ranks in forward/backward when the
    # original batch has fewer examples than GPUs. It contributes no events.
    if empty:
        start, end = 0, 1
    chunk_start = int(counts[:start].sum())
    chunk_end = int(counts[:end].sum())
    chunk_keys = {"input_features", "feature_attention_mask", "preconv_feature_lengths"}
    result = {}
    for key, value in batch.items():
        section = slice(chunk_start, chunk_end) if key in chunk_keys else slice(start, end)
        if isinstance(value, (torch.Tensor, tuple, list)):
            result[key] = value[section]
        else:
            result[key] = value
    if empty:
        result["labels"] = torch.full_like(result["labels"], -100)
    result["_empty_rank"] = empty
    return result


def normalize_distributed_loss(loss, weight_sum):
    """DDP averages gradients: scale local sums by world size / global weights."""
    totals = torch.stack((loss.detach() * weight_sum.detach(), weight_sum.detach()))
    torch.distributed.all_reduce(totals)
    if totals[1].item() <= 0:
        raise RuntimeError("Empty global weighted batch.")
    scaled_loss = loss * (torch.distributed.get_world_size() * weight_sum / totals[1])
    return scaled_loss, totals[0] / totals[1]
