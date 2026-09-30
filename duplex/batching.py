"""Length-grouped batches with an explicit padded-frame memory budget."""

import random
from collections.abc import Sequence

from torch.utils.data import Sampler


class FrameBudgetBatchSampler(Sampler[list[int]]):
    """Visit every conversation once, varying batch size rather than chunk size."""

    def __init__(self, lengths: Sequence[int], *, max_frames: int,
                 max_batch_size: int, seed: int) -> None:
        self.lengths = tuple(lengths)
        if not self.lengths or any(length < 1 for length in self.lengths):
            raise ValueError("Batch lengths must be positive and nonempty.")
        if max_batch_size < 1 or max_frames < max(self.lengths):
            raise ValueError("Frame budget must fit the longest complete conversation.")
        self.max_frames = max_frames
        self.batch_size = None  # Accelerate must retain the variable batch sizes.
        self.max_batch_size = max_batch_size
        self.drop_last = False
        self.seed = seed
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        rng = random.Random(self.seed + epoch)
        indices = list(range(len(self.lengths)))
        rng.shuffle(indices)
        batches = []
        # Sorting shuffled pools keeps padding low without imposing a curriculum.
        pool_size = self.max_batch_size * 50
        for offset in range(0, len(indices), pool_size):
            pool = sorted(indices[offset:offset + pool_size],
                          key=lambda index: self.lengths[index], reverse=True)
            batch = []
            longest = 0
            for index in pool:
                proposed_longest = max(longest, self.lengths[index])
                if batch and (len(batch) == self.max_batch_size
                              or proposed_longest * (len(batch) + 1) > self.max_frames):
                    batches.append(batch)
                    batch = []
                    longest = 0
                batch.append(index)
                longest = max(longest, self.lengths[index])
            if batch:
                batches.append(batch)
        rng.shuffle(batches)
        # Exercise the longest padded batch first, before investing in an epoch.
        largest = max(range(len(batches)), key=lambda i:
                      max(self.lengths[j] for j in batches[i]) * len(batches[i]))
        batches[0], batches[largest] = batches[largest], batches[0]
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)
