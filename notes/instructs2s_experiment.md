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
The full epoch completed successfully: 3,750 updates in 4,786.43 seconds
(79.8 minutes including epoch validation). Actual logged targets total
14,550,100 frames, 2,003,872 lexical tokens after interruption augmentation,
44,000 START and 44,000 STOP events. Peak allocated/reserved memory was
37.643/38.959 GiB. The epoch mean weighted training loss is 2.65063; the final
25-update window is 2.29963 weighted, 4.02080 text, 0.03501 IDLE, 0.11229 START,
0.57092 STOP. Augmented 1,000-example dev losses are 2.26187 weighted, 4.02930
text, 0.03736 IDLE, 0.07233 START, and 0.25319 STOP.

Fresh-reload, empty-context user-audio-only inference checked the first 12 train
and first 12 dev questions, plus six fixed spoken fact/arithmetic questions
verified absent from both manifests. The actual outputs and frame traces are in
`outputs/instructs2s-one-epoch/final-evaluation/`, including `samples.md`.
Most answers are fragmented or repetitive, including training examples. This
is not an all-IDLE failure: the Spain probe correctly answers Madrid; the France
probe answers Paris twice; the rectangle probe repeats "cent" without computing
50; the triangle probe is empty and times out. Dev example 5 correctly mentions
mock trials and moot court competitions; dev example 7 names relevant freelance
industries but contains malformed words. These are isolated successes, not a
passing generalization result.

Clean evaluation of all 1,000 dev examples gives weighted loss 2.35839,
text 3.99616, IDLE 0.02391, START 0.06047, STOP 0.17920 and lexical next-token
accuracy 28.01%. Those are teacher-forced metrics, not free-generation accuracy.
With identical gold history, cached/full decoding agrees on 49/49 lexical argmax
predictions for train example 0 (maximum BF16 logit difference 0.3125).
The separately tested native audio baseline produces substantially more fluent
answers. The final adapter's native text probes answer none of 12 correctly;
this measures interference when using native formatting, not proof that frozen
base weights permanently lost knowledge.

The run started from the fresh cached base, uses the
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

## Matched 200-update audio-gain diagnostics

The encoded audio frame norm is roughly 32–35 versus 0.86–0.88 for an answer
token embedding, including large nonzero representations of synthetic silence.
This motivates testing input scale before changing loss weights. The following
three runs start from the same fresh base, use identical first 200 batches,
learning rate, augmentation probabilities and objective, and change only audio
gain. Each is 200/3,750 updates (5.33% of an epoch), not a completed epoch.

| Audio gain | Last-window text CE | IDLE CE | START CE | STOP CE |
| --- | ---: | ---: | ---: | ---: |
| 1.0 everywhere (control) | 5.92312 | 0.23415 | 1.18181 | 2.12767 |
| 0.03 everywhere | 1.94094 | 0.05938 | 0.58039 | 1.19921 |
| 1.0 while listening, 0.03 while responding | 1.13768 | 0.04263 | 0.30784 | 1.00279 |

All three final windows contain 12,921 text labels. The fresh control's logged
target counts agree with the original full run at every shared logging step;
its step-200 text loss differs by 0.00303. GPU training is not claimed bitwise
deterministic. The pilot runner saves the actual step override with each adapter.
The completed original epoch remains gain 1.0: neither pilot gain is applied
retroactively to those weights.
The 200 planned batches contain 2,324 unique examples. The three probed train
split rows are absent from those batches: their pilot outputs are not training
memorization demonstrations. Full-epoch coverage, in contrast, includes them.

Global gain 0.03 produces fluent but unrelated paragraphs about weddings, cars,
and churches on every inspected audio question. It does not solve grounding.
The matched gain-1 control correctly answers zero of six spoken fact probes:
its responses are repetitive or empty despite preserved native text ability.
The separate unadapted native audio baseline answers all six questions correctly,
including full DNA expansion and triangle area 15. It uses Qwen's native audio
format only as a capability reference; it is not the duplex generation path.
Side-by-side full-epoch, three pilot and native outputs are saved in
`outputs/instructs2s-one-epoch/gain-comparison.md` and `gain-comparison.json`.
Response-only gain retains full question input before START and improves actual
question-dependent generation: Paris, Madrid, six times two, and the rectangle's
50 square centimeters are correct on the fixed six spoken probes. DNA stops at
"Deoxyribonucleic"; the triangle answer ends before the division by two or result.
Its train example 2 gives coherent, relevant environmental-advocacy advice;
dev example 0 starts twice and stops mid-sentence; dev example 1 makes incorrect
claims about philosophers; dev example 2 gives relevant wine advice but calls
Marlin "Marilyn". Better fluency and grounding do not establish reliable answers.

The response gain follows only previous input/generated START and STOP events,
not target labels, transcripts, question length, or reference answers. Masked
bootstrap BOS does not count as START. Training and streaming share the scaling
helper, including FP32 multiplication followed by the BF16 fusion cast. Reload
rejects an incompatible audio contract; inference still requires empty context.
Full/cached gold-history argmax agreement is 49/49 for train 0, 70/71 for dev 0,
and 7/7 for the France probe, with maximum BF16 logit differences 0.3125, 0.375,
and 0.1875 respectively. Encoder/mel parity checks also pass. This is close
numerical agreement, not exact equivalence of all cached decoder logits.
The ordinary `generate.py` CLI was also tested in a fresh process on the rectangle
recording with this pilot's config/adapter and no prompt arguments. Its text and
entire generated event-ID sequence match the evaluator; result and trace are in
`outputs/instructs2s-response-gain003-pilot/cli-rectangle-result.json` and
`cli-rectangle-trace.jsonl`. All 13 local pytest checks pass, including causal
phase scaling, cached gain application, padding/boundary preservation, prompt
rejection, and rejecting a mismatched adapter audio contract before weight load.

All three early adapters answer the 12 simple native-text probes correctly by
meaning. The global and response-only gains each score 10/12 with the script's
strict prefix rule because some answers use words or explanatory prefixes
instead of beginning with the numeric reference.
The control scores 12/12 by prefix. The gain-1 completed epoch scores 0/12.
Thus extended training does impair this adapter's native-format responses;
shortening alone preserves those probes but does not establish a good duplex
model. Retention after a full epoch with response-only gain is not yet tested.

Pilot reproduction (run each GPU job sequentially):

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 .venv/bin/python scripts/run_taste_diagnostic_pilot.py --config configs/instructs2s_one_epoch_a100.yaml --output outputs/instructs2s-gain1-pilot --learning-rate 0.0002 --steps 200
PYTORCH_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 .venv/bin/python scripts/run_taste_diagnostic_pilot.py --config configs/instructs2s_gain003_pilot.yaml --output outputs/instructs2s-gain003-pilot --learning-rate 0.0002 --steps 200
PYTORCH_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 .venv/bin/python scripts/run_taste_diagnostic_pilot.py --config configs/instructs2s_response_gain003_pilot.yaml --output outputs/instructs2s-response-gain003-pilot --learning-rate 0.0002 --steps 200
OMP_NUM_THREADS=4 .venv/bin/python scripts/evaluate_instructs2s.py --config configs/instructs2s_response_gain003_pilot.yaml --adapter outputs/instructs2s-response-gain003-pilot/final --samples-per-split 3 --short-audio-probes --check-parity --output outputs/instructs2s-response-gain003-pilot/evaluation
```

## Next experiment suggested by the evidence

Keep this richer first-turn dataset and the verified no-prompt audio contract.
The response-only gain is a candidate for further training, with checkpoint
selection based on held-out audio meaning, completion, START/STOP behavior, and
native retention together. Loss alone would have selected the ungrounded global
gain. Use checkpoints at short intervals before committing to another complete
epoch; preserve the baseline and evaluate a larger locked held-out set.

If retention declines, test a smaller learning rate and less adaptation of the
language decoder, then a small audio interface trained with most language layers
frozen. That is a proposed experiment, not implemented or validated here.
[Freeze-Omni](https://arxiv.org/abs/2411.00774) provides precedent for preserving
language weights while learning speech interfaces, although its architecture and
multi-stage recipe differ. Research on
[implicit task inference after fine-tuning](https://arxiv.org/abs/2309.10105)
also cautions against equating a failed native-format probe with irreversible
loss of base knowledge. Neither paper proves our specific gain setting or
training schedule will retain capabilities. Any follow-up remains Thinker-only,
text-output, additive fusion, fixed two-second chunks and the same event objective.
