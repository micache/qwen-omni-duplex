# Longer-response InstructS2S experiment

The user explicitly requested replacing TASTE with longer, more informative
responses, preferably the InstructS2S data used successfully with their previous
larger model. This overrides the older dataset-only restriction. The backbone
remains Qwen2.5-Omni-3B Thinker, fixed 2-second chunks and validated 25 Hz,
additive fusion, weighted next-event loss, synthetic interruption, and text only.

## Storage and selection

The original [ICTNLP dataset](https://huggingface.co/datasets/ICTNLP/InstructS2S-200K)
contains roughly 345 GB of archives/metadata and cannot fit this 96 GB disk.
The [yuekai Parquet conversion](https://huggingface.co/datasets/yuekai/InstructS2S-200K)
contains roughly 127 GB across 423 shards. Its conversion source was inspected,
not executed; the first two prompt/answer pairs were checked against the original.
Its audio is the current question, while later-round text includes conversation
history. Consequently this experiment uses only first turns and no text history.

`scripts/prepare_instructs2s.py` downloads pinned-revision shards in seeded order,
extracts real question audio into mono 16 kHz PCM16 FLAC, and deletes each source
Parquet after processing. It retains complete 16–512-token answers, question
clips no longer than 30 seconds, and unique normalized prompts. A seed-212 split
holds out 1,000 examples and trains on 44,000. This is an explicitly filtered
subset, not a full epoch over the original 200,000 multi-turn conversations.
Speech tokens and assistant speech are neither retained nor trained.

Each answer receives a synthetic zero-audio canvas rounded up to full two-second
blocks, sufficient for all tokens plus START/STOP at 25 Hz. These durations are
not original speech timings. Question audio is padded to a two-second boundary,
while START remains at the actual question-end frame. The same padding occurs
inside streaming inference; its caller supplies only the unpadded question.

The local provenance JSON records source revisions, shard hashes, downloaded
bytes, filters, split seed, lengths, and full response-token statistics. Only
regenerable TASTE/MUSAN Arrow caches were deleted to recover space; raw datasets,
models, and prior experiment checkpoints were preserved.

## Audio and prompt contract

The new adapter always uses empty text context. CLI and model generation reject
a nonempty prompt for this recipe. They share the complete-chunk padding helper,
feature masks, 16 kHz input, and empty silent response chunks with training.
Inference has a fixed 16-chunk silence allowance and 768-token ceiling independent
of any reference answer length. These are upper bounds, not gold response timing.

Preflight found batch-size-dependent BF16 audio-encoder numerics even with
identical mel features: roughly 5–9% relative RMS difference on three inspected
clips. FP32 evaluation of the same frozen weights agreed much more closely.
The new recipe therefore keeps the frozen audio encoder in FP32 and evaluation
mode, disables audio autocast, and applies the configured TF32 policy consistently
when constructing either a training or an inference model. Text-decoder training
still uses BF16. TF32 is disabled for this run after its FP32 encoder comparison
exceeded the preflight tolerance. No encoder parameters are updated.

Exploratory parity reports are preserved under `outputs/instructs2s-one-epoch/`.
The first precision experiment temporarily recast encoder buffers and therefore
its cached-logit comparison is not a valid deployment-parity result. Subsequent
checks preserve the final precision policy. BF16 text full-sequence versus cached
decoding still has rounding differences; report argmax agreement and logit error
alongside audio parity rather than claiming bitwise decoder equivalence.

## Run and results

Preparation completed in 1,322 seconds, processing 100 source shards / 30.13 GB
of source bytes. The prepared view occupies 6.54 GB. Training has 2,218,152
response tokens (median 47, p90 77, maximum 276); dev has 50,728. There is no ID
or normalized-prompt overlap, and every selected FLAC exists. Inference/clean
training mel features agree exactly. With the final FP32/no-TF32 policy, the
three initial clips had encoder relative RMS differences below 0.0002 after the
BF16 fusion cast and FP32 maximum differences below 0.000109.

Raw forward/backward/Adam-memory benchmarks passed budgets 3,360, 3,680, and
4,000; 4,320 ran out of memory. The 4,000 budget had a raw benchmark peak of
37.64 GiB allocated / 38.75 GiB reserved. The actual Trainer completed one update
but encountered allocator fragmentation on the next batch. That attempt's log
and one-update metrics were archived; its weights were discarded. Restarting
from the same fresh base with `PYTORCH_ALLOC_CONF=expandable_segments:True` passed
the changing batch shapes. The selected sampler plans 3,750 updates, batch sizes
1–26 (median 11), 0.128% padding, and exactly one visit to each of 44,000 examples.
No completed training result is claimed yet. The run starts from the fresh cached base, uses the
same rank-16 projections, constant 2e-4 learning rate, loss weights, and 0.2
noise/interruption probabilities as the TASTE epoch. Thus the experiment changes
both data and the verified audio preparation/precision contract; improvement
cannot be attributed exclusively to response length.

Reproduction:

```bash
.venv/bin/python scripts/prepare_instructs2s.py
OMP_NUM_THREADS=4 .venv/bin/python scripts/evaluate_instructs2s.py --preflight --output outputs/instructs2s-one-epoch/preflight-new
OMP_NUM_THREADS=4 .venv/bin/python scripts/benchmark_instructs2s.py
PYTORCH_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 .venv/bin/python -u train.py --config configs/instructs2s_one_epoch_a100.yaml > outputs/instructs2s-one-epoch/train.log 2>&1
OMP_NUM_THREADS=4 .venv/bin/python scripts/evaluate_instructs2s.py --adapter outputs/instructs2s-one-epoch/final --output outputs/instructs2s-one-epoch/final-evaluation
```

Native-text probes use Qwen's native chat format only to measure preservation of
base capabilities; the actual duplex inference uses no prompt or reference text.
Held-out generations are judged for meaning and completion, not just exact
matching: an alternate valid long answer need not reproduce its reference.
