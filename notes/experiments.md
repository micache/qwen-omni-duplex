# Experiments

## 2026-09-30 — native retention, early stopping and fusion diagnosis

Starting code revision `8c319a5`, same cached Qwen2.5-Omni-3B Thinker, TASTE
and A100 as the completed epoch. Compared native chat, teacher-forced duplex,
streamed responses, adapter strengths and same-input cached/full execution.
Two deterministic 200-update pilots tested 2e-4 versus 5e-5; the higher-rate
pilot exactly reproduced original training loss logs through step 200.

On the same first six dev rows / six native questions:

| run | native correct | duplex text NLL | stream exact |
| --- | ---: | ---: | ---: |
| 2e-4 / 50 updates | 6/6 | 6.743 | 0/6 |
| 2e-4 / 200 updates | 6/6 | 4.210 | 0/6 |
| 5e-5 / 50 updates | 6/6 | 6.756 | 0/6 |
| 5e-5 / 200 updates | 6/6 | 4.959 | 0/6 |
| original / 1,500 updates | 0/6 | 1.890 | 1/6 |
| original / 1,919 updates | 0/6 | 1.654 | 1/6 |

The full-strength final adapter suppresses native chat; quarter strength
restores most native answers but worsens duplex text. The full-forward and
cached decoder agree at 34/34 checked lexical positions. Correct event timing
and one forced correct initial token do not repair lexical errors. In a 31-row
token decomposition, separators are 95.6% correct versus 65.4% for tokens
containing alphanumeric characters. Data is short (median seven response
tokens), but the native base model can interpret the kitchen-tools recording.

All three original dev clips become all-IDLE timeouts when the old default
system prefix is inserted; training uses empty context. Fixed the CLI and
model generation defaults and empty-string tokenization. The real fixed CLI
produces the color reference and STOP; seven local tests pass. No fusion or
weight changes were made to the original adapter based on exploratory norm
measurements. See [the full diagnosis](one_epoch_diagnosis.md) for exact
commands, source papers, artifacts and the untested 200–1,500 interval.

## 2026-09-30 — full one-epoch TASTE run on A100

The user explicitly chose the current TASTE recipe over the stale DailyTalk-only
instruction and authorized dependency/model/data downloads. This fresh clone
started at repository commit `38c83eb`. Installed the pinned requirements into
Python 3.11.16; the uv invocation required `--index-strategy unsafe-best-match`
because the PyTorch index also exposes an older `requests` wheel. Dependencies
passed `uv pip check` (94 packages). Hardware: one NVIDIA A100-SXM4-40GB,
driver 580.173.02. Model: direct BF16 `Qwen/Qwen2.5-Omni-3B` Thinker at
`f75b40e3da2003cdd6e1829b1f420ca70797c34e`, with the frozen vision tower
removed, frozen audio/embedding/head weights and the existing rank-16 decoder
LoRA recipe. Native PAD/BOS/EOS controls, additive fusion, fixed 2-second
chunks, 25 Hz and weights text/IDLE/START/STOP = 1/0.1/4/4 were retained.
TASTE snapshot: `028e90e49bccfd0ca27f85e1dc4137decc661c83`; MUSAN snapshot:
`76f9882cfa4475efe11508ac9aa32722f84ca5b7`.

Configuration: `configs/turn_packed_one_epoch_a100.yaml`. A shuffled,
length-grouped padded-frame budget of 3,360 with sample cap 128 yielded actual
batches of 4–48 (median 23), 1,919 updates, no accumulation and no dropped
conversation. Used fused AdamW, four persistent loader workers, TF32 enabled,
constant 2e-4 and gradient checkpointing disabled. Full gradient audits ran
at the first/last update and every 100 updates; loss checks remained per batch.
Noise and synthetic interruption each retained probability 0.2.

Microbenchmarks included actual forward/backward/optimizer updates. At 135
frames, the original layout without checkpointing reached 25.34 samples/s at
batch 24. Flattening cross-entropy to `[events, vocabulary]` improved this to
38.48 samples/s with the same loss/gradient formula. Batch 32 ran out of memory.
Checkpointing fit median-length batch 96 but only reached 14.04 samples/s.
At 566 frames the optimized non-checkpointed path fit batch 6 but not batch 8.
The frame budget therefore increases batches on short examples while retaining
complete long conversations. These are microbenchmarks, not an epoch speedup
comparison. Raw results are in the ignored run directory.

Reproduction commands, after public assets are prepared:

```bash
.venv/bin/python scripts/prepare_taste_lengths.py --output outputs/one-epoch-a100/lengths.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 .venv/bin/python -u train.py --config configs/turn_packed_one_epoch_a100.yaml > outputs/one-epoch-a100/train.log 2>&1
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false .venv/bin/python -u scripts/evaluate_taste_samples.py --config configs/turn_packed_one_epoch_a100.yaml > outputs/one-epoch-a100/inference.log 2>&1
```

The epoch plus full dev evaluation took 1,369.072 seconds; dev evaluation alone
took 90.428 seconds. Training mean loss was 0.628777; the first batch was
9.828 and the final 19-update window averaged 0.334857. Dev weighted/text
losses were 0.333011/1.071998 over all 4,000 dev rows, using the recipe's
augmented collator. Peak allocated/reserved memory was
34,151,796,736/40,259,026,944 bytes. Summing the saved training label counts
verified exactly 44,000 START, 44,000 STOP, 322,164 text and 5,851,586 IDLE:
all 44,000 conversations and 6,261,750 frames were processed once. Final
adapter and reconstruction metadata are under `outputs/one-epoch-a100/final/`;
the downloadable archive is `adapter-one-epoch.tar.gz` in that run folder.

Fresh-process reload and fixed inference took 228.423 seconds. Selection was
the first three train rows and first three dev rows, fixed before evaluation.
Used greedy decoding, empty context matching training, no noise, four allowed
silent chunks and a 128-token cap. Three input views were compared: training
audio (user plus reference-duration silence), raw user-only audio, and user
audio padded only to the next 2-second boundary, with no reference length.

| split / sample | reference | raw user-only generation |
| --- | --- | --- |
| train / 047388 | Red and green. | Red and green. |
| train / 001431 | I got a promotion at work! | I got a promotion at work! |
| train / 018014 | Two, three, four. | Two, three, four. |
| dev / 010890 | Baseball, boxing, football, soccer, and hockey. | Sug, basketball, baseball, baseball, and baseball. |
| dev / 010838 | Bears, wolves, eagles, and sharks. | Tions, bear, bear, and bear. |
| dev / 012247 | A whisk, a blender, and a spatula. | Blrying, knife, and whisk. |

All three train responses were exact and all three dev responses had malformed
or repetitive content, across all three input views. These dev failures are
more than alternative valid answers; chunk padding did not repair them.
All six raw user-only runs produced one START/text/STOP segment without timeout.
None of the six training-audio traces exactly matched every target frame.
In six synthetic interruption variants, four were still responding at the
injected onset and emitted STOP within 0–1 logical frames (0–0.04 s); two had
already stopped and do not establish interruption success. Logical-frame
delays are not wall-clock or live latency measurements: complete 2-second
chunks supply all their features together. Interrupted text did not generally
match the precise target prefix. This small sample is a diagnostic, not a
generalization benchmark or reliable full-duplex acceptance gate.

The new local correctness suite passed six tests covering epoch validation,
complete sampler coverage/memory bounds/determinism and weighted-loss/gradient
parity. Compilation and whitespace checks passed. `train.log`, `metrics.jsonl`,
`inference.log`, `inference-report.json`, `coverage.json`, data revisions,
run manifest, sample WAVs and event traces are saved locally under the ignored
run directory. SSH access to the authorized remote currently fails with
`Permission denied (publickey)`; no remote history was changed.

## 2026-09-16 — turn-packed data-path correctness check

This was a data-path check, not a training run. The selected public
TASTE-IF-SFT-48K default config and MUSAN noise-only config were downloaded
locally. The former reconstructed to about 12 GB and contains 44,000 train and
4,000 development conversations; the latter is about 696 MB in two Parquet
shards. A decoded development sample retained its real 41,796-sample user
waveform, replaced the 38,639-sample response waveform with exact zeros, and
retained the response text. Synthetic tests checked contiguous response-token
packing, all-IDLE user targets, early STOP plus user-audio overlay for
interruption, and measured 10 dB noise mixing. No optimizer step, model load,
or benchmark was run.

The subsequent full training-label scan covered all 44,000 train rows. Before
augmentation it found 5,825,957 IDLE and 347,793 text events plus 44,000 each
of START and STOP. Modeling the configured 20% interruption probability gives
expected counts of 5,851,670.5 IDLE and 322,079.5 text per epoch. Exact
IDLE/text aggregate balancing would use IDLE weight 0.05504 at text weight 1,
but that would remove the intended conservative silence bias. The selected
full-run weights are IDLE 0.1, text 1.0, START 4.0, STOP 4.0, producing expected
weighted fractions 46.47%, 25.58%, 13.98%, and 13.98%. The minimum assistant
capacity margin across the split was one frame, so no sample overflowed.

Session 01 created structure and executable scope checks only; later dated
entries record the first local probes and experiments.

Future entries should record the date, configuration file, code revision,
hardware, data snapshot, exact command, result, and interpretation. Adapted
Full-Duplex-Bench results must be labeled “text-timeline adaptation,” never as
official speech-output scores.

## 2026-09-15 — Session 14 v1.5 evaluator correctness

Handcrafted token/control traces and temporary paired WAVs exercised the four
v1.5 overlap scenarios. This was evaluator correctness validation, with no
model weights, external dataset, semantic API call, or benchmark score. The
focused command was `.venv/bin/python -m pytest -q
tests/test_full_duplex_v15.py tests/test_benchmarks.py`; 35 tests passed. Paper
timing equations were adapted to causal lexical availability and kept distinct
from internal predicted STOP timing. No full-data experiment was run.

## 2026-09-11 — Session 05 synthetic inspection

This was a local correctness inspection, not a model experiment. A two-second,
100 Hz synthetic waveform and character-token timeline were augmented with
probability one and seed 12. The selected assistant turn was cut at timeline
frame 14; its natural `STOP` had been frame 25 and the following user onset had
been frame 35. The rendered rows around the splice showed no user activity at
frame 13 and simultaneous user activity plus `STOP` at frame 14. No hardware,
model weights, external data snapshot, or benchmark configuration was used.

```text
frame  time(s)  user  target  decoded token  state
-----  -------  ----  ------  -------------  --------
   10    0.400   no   TEXT    'c'            active
   11    0.440   no   IDLE                   active
   12    0.480   no   TEXT    'd'            active
   13    0.520   no   IDLE                   active
   14    0.560   yes  STOP                   inactive
   15    0.600   yes  IDLE                   inactive
   16    0.640   yes  IDLE                   inactive
```

## 2026-09-11 — Session 06 Qwen Thinker compatibility probe

This was an inference-only API and memory probe, not training or a benchmark.
On an RTX 2060 SUPER with torch 2.10.0+cu126, checkpoint commit
`f75b40e3da2003cdd6e1829b1f420ca70797c34e` passed direct Thinker loading,
batch-two 2-second audio encoding and `[50,2048]` restoration, vision pruning,
batch-two Thinker prefill, and one cached decode step. Peak allocated CUDA
memory was 9,446,580,736 bytes for load, 9,470,790,144 for audio encoding, and
8,154,948,608 after vision pruning for prefill plus cached decode. The raw
result and exact command are recorded in `notes/session06_probe.json` and
`notes/compatibility.md`.

## 2026-09-14 — Session 08 synthetic LoRA smoke

This was a three-optimizer-step synthetic correctness smoke, not the main
DailyTalk experiment. It used `configs/debug.yaml` at rank 16 in BF16 on one
NVIDIA GeForce RTX 3090 (24,576 MiB), with the pinned base revision and package
set. The command was:

```bash
.venv/bin/python train.py --config configs/debug.yaml --smoke-test
```

Two initial steps saved `checkpoint-2`; a fresh direct Thinker load plus the
saved adapter reproduced the fixed-batch logits exactly (maximum absolute
difference 0.0 at `rtol=atol=1e-3`). Trainer state and optimizer state then
resumed from step 2 and completed exactly one further update at learning rate
2e-4, ending at step 3. All 504 LoRA A/B tensors (29,933,568 parameters across
252 discovered text-decoder projections) received nonzero gradients during the
run. No audio-tower parameter had a gradient, and optimizer groups/state were
limited to trainable LoRA parameters. Peak CUDA allocated/reserved memory was
9,433,796,608/10,104,078,336 bytes. `outputs/session08-smoke/smoke_report.json`
contains the machine-readable local result and is intentionally gitignored.

## 2026-09-14 — Session 09 DailyTalk tiny-slice overfit

Status: **OVERFIT_GATE=PASS**. This session stopped at the tiny-slice gate; no
larger-data job was launched. The implementation is commit
`e281f6b870b33c40e690739bcfd16017636daa02`, based on `ff13f77`, and used the
pinned Qwen revision `f75b40e3da2003cdd6e1829b1f420ca70797c34e` and public
DailyTalkContiguous snapshot `33e1b501f725a6f4ed4ded95e16cd7f66b9d4bdc`.

The fixed debug slice contains only these public IDs and window metadata; the
downloaded public WAVs remain outside the repository under `/workspace/data`
and are not copied into configuration, notes, or Git:

| public conversation ID | fixed window (s) | sidecar word indices | synthetic interruption source |
| --- | ---: | ---: | --- |
| `data_stereo/1292` | 0.0–2.0 | 0 | channel-1 activity 1.68–2.0 s |
| `data_stereo/2196` | 0.0–2.0 | 0 | none |
| `data_stereo/382` | 1.3–3.3 | 0, 1 | none |
| `data_stereo/1942` | 29.0–31.0 | 47, 48 | none |

The channel-1 onset was selected deterministically from 40 ms frame energy.
With interruption seed 20, the `data_stereo/1292` user suffix moved from frame
42 to frame 31 (11 frames earlier). The target at frame 31 is `STOP`, and the
shifted user audio/activity is observable at that frame. The other decoded
targets are `Hello`, `I am.`, and `I see.`. The exact preflight histogram was:

```text
IDLE=184  START=4  TEXT=8  STOP=4  PADDING=0
```

`outputs/session09-overfit-native/preflight.json` and
`decoded_timeline.txt` contain the machine-readable histogram and complete
decoded interrupted timeline. They were printed before optimization as well.

### Initial failure and ordered diagnosis

The initial command was:

```bash
.venv/bin/python train.py --config configs/debug.yaml
```

It used the then-current config SHA-256
`ba80ed5b954ef4072f531114e78dc2ce70f215f3ebb37d828c48ff014886366a`,
150 optimizer steps, and the Talker-derived control IDs 151859/151860/151861.
It failed the gate. Total teacher-forced loss fell 17.7066 → 7.0572 and text
loss fell 13.1406 → 0.000435, but IDLE/START/STOP losses plateaued at
7.7807/7.7813/7.7813. Both teacher-forced and cached/free predictions were
100% lexical-token IDs; no example emitted START or STOP.

The requested diagnosis order was followed. The histogram was complete; the
one-event causal shift and decoded/token round trip passed; weighted
normalization and the 1.0/0.25/4.0/4.0 weights were applied; all 504 LoRA
tensors were trainable. At that trainable/output-boundary check, inspection of
the frozen Thinker LM head showed that rows 151859 (`IDLE`) and 151861 (`STOP`)
were exactly identical (maximum difference 0), while the START row differed by
only `9.1552734375e-05`. Decoder-only LoRA therefore could not distinguish
IDLE from STOP. Audio lengths/masks had passed, and the learning rate was not
changed.

The evidence-backed fix maps IDLE/START/STOP to the already-existing native
Thinker PAD/BOS/EOS rows 151643/151644/151645. It adds no token or head and
does not change the 252 decoder-projection LoRA targets or 29,933,568 trainable
parameters. Failed artifacts remain locally in `outputs/session09-overfit/`.

### Passing run

The exact passing command was unchanged:

```bash
.venv/bin/python train.py --config configs/debug.yaml
```

The final `configs/debug.yaml` SHA-256 is
`3015072d83f54634e92f04e76f5ce826a92e6ee38e69c7a059e263ba526ac7d3`.
It ran BF16 LoRA for exactly 150 optimizer steps at learning rate 2e-4,
batch size 1, no accumulation, and no evaluation split. Hardware was one
NVIDIA GeForce RTX 3090, 24,576 MiB, driver 610.57.04, with BF16 support.
Training runtime was 93.3674 s. Peak CUDA allocated/reserved memory was exactly
9,414,631,936/10,101,981,184 bytes (8.768/9.408 GiB).

Compact numeric results:

| metric | initial teacher-forced | final teacher-forced | first 10 train steps mean | last 10 train steps mean |
| --- | ---: | ---: | ---: | ---: |
| total weighted loss | 10.382182 | 0.000839 | 6.776841 | 0.001734 |
| text loss | 12.902344 | 0.003390 | 13.641667 | 0.000586 |
| IDLE loss | 6.535846 | 0.000008 | 3.023210 | 0.002247 |
| START loss | 15.828125 | 0.001441 | 12.318750 | 0.000645 |
| STOP loss | 14.734375 | 0.001351 | 9.007813 | 0.001555 |

Final teacher-forced and cached/free event fractions were identical to the
labels: IDLE 0.92, text 0.04, START 0.02, and STOP 0.02. Both modes matched all
200 target event kinds. Every example produced its exact START, lexical token
sequence, and STOP at the target frames. In particular, the interrupted
`data_stereo/1292` cached/free sequence emitted START at frame 25, lexical
`Hi` at frame 26, and STOP at shifted user-onset frame 31.

Fresh base-model plus final-adapter reload preserved every teacher-forced and
cached/free prediction. The maximum absolute fixed-batch logit difference was
0.0 at `rtol=atol=1e-3`. Alignment, causal shift/leakage, timeline masks,
Qwen-derived audio lengths, finite loss/gradient, frozen-tower, optimizer
scope, and adapter-only checkpoint assertions all passed. The full local suite
after the fix passed with 67 tests and five expected opt-in/cache-dependent
skips. Machine-readable metrics and the gate result are in the gitignored
`outputs/session09-overfit-native/` directory.

## 2026-09-15 — Session 11 internal metrics and small-data diagnostic

Status: **SMALL_DATA_GATE=PASS**. This was one weighted diagnostic on a fixed
public subset, not the main run and not a paper ablation. The prerequisite
`OVERFIT_GATE=PASS` was confirmed from Session 09, and the Session 10 real
streaming smoke was confirmed from its 56-record adapter trace and legal STOP.
No unweighted training run was needed because the complete target histogram
already demonstrated IDLE dominance.

The Session 11 implementation and fixed compact manifest are commit
`7bcb5d7d2e3ae8b117d56704d49acb5b376ddac9`, based on Session 10 commit
`69836b224bc1d5805e8c19299cb3a574ed80bb60`.

### Data, selection, and weights

The public DailyTalkContiguous revision was
`33e1b501f725a6f4ed4ded95e16cd7f66b9d4bdc`. The fixed subset contains 100
conversation IDs: 80 train and 20 validation under split salt
`DailyTalkContiguous-session11-112`. Each selected 8-second span is represented
as four contiguous fixed 2-second chunks; chunking therefore remains exactly
2 seconds at 25 Hz. Each conversation contributes its four normal chunks and
one deterministic interrupted duplicate, giving 400 train and 100 validation
examples. The exact IDs, spans, interruption chunk indices, cut frames, and
source user frames are in `notes/session11_subset.yaml`. The full local
manifest is `outputs/session11-diagnostic/subset_manifest.json`, SHA-256
`b9e5dae51d84a1587739bbfe657a70e3966e33f8de5e9c798ceab8ba1ac0b16c`.
The compact checked-in manifest SHA-256 is
`e28c0ae0c43cd35bcc838b093b477c832136a1fd7cfd9108ab7de3dcbd4c35b3`.

User activity for the synthetic splice was derived only from the public right
channel with deterministic 40 ms RMS frames, floor 0.005, 0.10 times the 95th
percentile, gaps of at most two frames merged, and a minimum two active frames.
Frames overlapping annotated assistant words were removed before runs were
formed. This is synthetic interruption metadata, not a claim that the public
sidecar supplies user transcripts.

The complete histogram was computed and printed before selecting weights:

| split | frames | IDLE | TEXT | START | STOP | PADDING |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 20,000 | 18,821 | 807 | 186 | 186 | 0 |
| validation | 5,000 | 4,729 | 181 | 45 | 45 | 0 |
| all | 25,000 | 23,550 | 988 | 231 | 231 | 0 |

IDLE was 94.2% of all targets. The provisional public weights were chosen as
`text=1.0`, `idle=0.05`, `start=4.0`, and `stop=4.0`; their full-subset weighted
target masses are 988, 1,177.5, 924, and 924. These are public-data-derived
diagnostic weights and are not Huawei values. The passing evidence promoted
these weights and the 8-second/four-chunk selection span to
`configs/train_lora.yaml`; the QLoRA fallback was deliberately unchanged.

### Configuration and execution

The run used `configs/session11_diagnostic.yaml` at SHA-256
`cf9e1a38d0b03104c3c94513226ec17243653cc88c0fffe01306d9375e082a8c`,
the pinned Qwen revision `f75b40e3da2003cdd6e1829b1f420ca70797c34e`,
native Thinker PAD/BOS/EOS control rows 151643/151644/151645, BF16 LoRA rank
16, constant learning rate 2e-4, batch size one, no accumulation, and exactly
250 optimizer steps. Seeds were process/trainer 111, split/data 112, recorded
window seed 113, and interruption 114. Span selection was deterministic by
usable-turn score and did not consume the recorded window RNG.

Hardware was one NVIDIA GeForce RTX 3090 with 24,576 MiB and driver 595.84.
The software path was torch 2.10.0+cu126, Transformers 5.17.0, Accelerate
1.15.0, PEFT 0.20.0, bitsandbytes 0.50.2, and SDPA. All 504 LoRA tensors
(29,933,568 parameters) received nonzero gradients. Trainer runtime was
152.9 seconds; total model-load, training, held-out evaluation, and trace
runtime was 475.864 seconds. Peak allocated/reserved CUDA memory was exactly
9,414,631,936/10,104,078,336 bytes (8.768/9.408 GiB). The mean of the first
ten logged step losses was 6.7540 and the mean of the final nine step losses
was 2.4462; every logged loss was finite.

Commands:

```bash
PYTHONPATH=. .venv/bin/python scripts/run_session11_diagnostic.py --config configs/session11_diagnostic.yaml --prepare-only
PYTHONPATH=. .venv/bin/python scripts/run_session11_diagnostic.py --config configs/session11_diagnostic.yaml
```

### Held-out internal metrics

Teacher-forced weighted validation loss was 3.029406 with applied-weight
denominator 777.45. Lexical-token loss was 8.797134 and perplexity 6,615.258
over 181 text tokens. Counts were:

| event | labels | predictions | precision | recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| IDLE | 4,729 | 4,220 | 0.9687 | 0.8645 | 0.9136 |
| START | 45 | 478 | 0.0544 | 0.5778 | 0.0994 |
| STOP | 45 | 302 | 0.0795 | 0.5333 | 0.1383 |
| text | 181 | 0 | n/a | n/a | n/a |

The teacher-forced raw invalid-transition rate before grammar masking was
719/5,000 = 0.1438. Twenty-seven of 45 target response turns received an
ordinally matched predicted turn. Mean START absolute error was 5.556 frames
(0.222 s, denominator 27); mean STOP error was 3.923 frames (0.157 s,
denominator 26).

Twenty free-streaming traces covered ten normal and ten synthetically
interrupted 8-second spans. Over the 4,000 real-audio target frames, selected
predictions were IDLE=3,565, text=419, START=11, and STOP=5, so IDLE was 89.1%
rather than a near-total collapse. The raw pre-mask invalid-transition rate
was 1,135/4,800 = 0.23646 across real audio and silent tails; the selected
post-mask rate was 0/4,000. No-response was 10/20 = 0.50 and no-STOP was
16/20 = 0.80. Synthetic-interruption STOP recall was 2/10 = 0.20, with mean
latency 71 frames (2.84 s, denominator 2). Free boundary errors remained weak:
START 33.3 frames (1.332 s, denominator 10) and STOP 34.25 frames (1.37 s,
denominator 4).

Representative public-safe trace observations:

- Normal `data_stereo/691@0s` emitted START at frame 5, two lexical events at
  frames 13–14, and STOP at frame 24; all transitions were legal.
- Interrupted `data_stereo/654@0s` had synthetic onset frame 18. It emitted a
  second active response at frame 10 and STOP at frame 42, a post-onset latency
  of 24 frames (0.96 s).
- Interrupted `data_stereo/442@0s` had onset frame 17 and emitted STOP at frame
  135, a delayed but correct post-onset STOP.
- Normal `data_stereo/10@8s` emitted 250 IDLE events including its silent tail,
  documenting a no-response failure rather than hiding it.

The gate passed because validation losses were finite, START and STOP counts
were nonzero, aggregate free predictions did not collapse to near-total IDLE,
the selected stream grammar was legal, at least 20 traces were inspected, and
two synthetic interruptions received post-onset STOP events. Semantic quality
was intentionally not an acceptance condition: teacher-forced lexical argmax
count was zero, generated text was poor, raw grammar violations were frequent,
and no-response/no-STOP rates were high. These are explicit limitations for a
later main run, not reasons to reinterpret this sanity check as a quality
result. The corrected local report is
`outputs/session11-diagnostic/report.json`, SHA-256
`9cc0d62365da18b9ea049ecc395adca8a9e199492871aec755f66ba8e17fe8a9`.

## 2026-09-15 — Session 15 controlled main BF16-LoRA run

Status: **MAIN_CHECKPOINT_READY=FAIL**. Prerequisites were the recorded
`OVERFIT_GATE=PASS` and `SMALL_DATA_GATE=PASS`. This was one controlled main
run, with no sweep and no benchmark judging. The local RTX 3090 had 24,576 MiB
VRAM, driver 595.84, and was idle at preflight. The workspace had 47 GB free
before the run; the local public DailyTalk snapshot occupied 13 GB. Git was
clean at the start, on `master` at `671558d35d1e59fc38683158a580f50e6af76a32`.
No model or dataset was downloaded in this session.

An initial preparation under `outputs/session15-main-111/` requested 300 train
and 50 validation conversations, but the deterministic split had only 23
eligible validation conversations. That directory contains only its frozen
config and freeze metadata. No model load or training occurred there. A new
run ID, `session15-main-111b`, used the same train quota and 20 validation
conversations under the same split seed; the failed directory was not reused
or overwritten.

### Frozen inputs and exact commands

The resolved config is `outputs/session15-main-111b/resolved_config.yaml`,
identical to `configs/session15_main.yaml`, SHA-256
`5ee5f025cf329ed62ac41ad856bb4f97d373e0c324b7097c23c8d6e36e350897`.
The freeze record is `outputs/session15-main-111b/freeze.json`. Base Thinker
revision was `f75b40e3da2003cdd6e1829b1f420ca70797c34e`; public data
revision was `33e1b501f725a6f4ed4ded95e16cd7f66b9d4bdc`. The source
JSONL manifest SHA-256 was
`e00c5d5aec839a46181b4a680242f43733b5e73d3b7e2ef023140bf0f6b9a49f`.
The selected 320-conversation manifest SHA-256 was
`0c121500c9e43c4fd99c86647ec9e992127b0abf0d5e67ebd80c3ab7a71afe39`.
All IDs were unique and train/validation conversation sets were disjoint under
split salt `DailyTalkContiguous-session11-115`. Four normal fixed chunks and
one deterministic interrupted chunk per conversation yielded 1,500 train and
100 validation timeline examples. The complete 80,000-frame histogram was
IDLE=75,389, text=3,123, START=744, STOP=744, padding=0. Each selected span
was 8 seconds, four contiguous 2-second chunks at 25 Hz. The one interrupted
duplicate in five examples implements a fixed 0.2 interruption fraction; cuts
used interruption seed 114 and minimum four assistant frames. Window seed 113
was recorded; span selection was deterministic by usable-turn score. Process
seed 111 and split seed 115 were fixed.

Weights were text=1, IDLE=0.05, START=4, STOP=4. Native Thinker control rows
were PAD/BOS/EOS 151643/151644/151645. BF16 LoRA used rank 16, alpha 32,
dropout 0.05, and the seven decoder projections
`q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` (252 exact targets).
Optimizer was `adamw_torch`, batch size 1, accumulation 1, constant 2e-4,
zero warmup/weight decay, max gradient norm 1, non-reentrant gradient
checkpointing, and 800 steps. Evaluation and rolling checkpoints were at every
100 steps; `save_total_limit=2`. Generation was greedy with the frozen
"You are a concise spoken-dialogue assistant." system prompt, no sampling,
temperature 1, no top-k, and one silent-tail chunk. Internal training
validation used empty context, as in Session 11; the separate fresh-process
checks used the frozen generation prompt. This context difference is a known
validation limitation.

The dependency freeze recorded torch 2.10.0+cu126, Transformers 5.17.0,
Accelerate 1.15.0, PEFT 0.20.0, bitsandbytes 0.50.2, datasets 5.0.1,
huggingface_hub 1.31.0, PyYAML 6.0.3, SoundFile 0.14.0, SciPy 1.17.1, and
NumPy 2.4.6. Training runner SHA-256 was
`9d69dc20e6545f0da21d6d5400f4290d290b5c2e75836a80d3dce795181b8c92`.

```bash
PYTHONPATH=. .venv/bin/python scripts/run_session15_main.py --config configs/session15_main.yaml --prepare-only > outputs/session15b-prepare.log 2>&1
PYTHONPATH=. .venv/bin/python scripts/run_session15_main.py --config configs/session15_main.yaml > outputs/session15b-train.log 2>&1
PYTHONPATH=. .venv/bin/python scripts/verify_session15_adapter.py > outputs/session15b-fresh-verify.log 2>&1
```

### Checkpoint selection and limitations

The full load/training/validation runtime was 1,211.132 seconds (20 min 11 s),
800 optimizer steps, about 0.66 steps/s including fixed evaluations and trace
decoding. Peak allocated/reserved CUDA memory was
9,414,631,936/10,158,604,288 bytes (8.77/9.46 GiB). Learning rate stayed at
2e-4; logged total and group losses were finite. Step 200's six free traces
were all IDLE, but step 300 recovered to 75.8% IDLE, so the persistent-IDLE
abort rule did not trigger. There were no NaN, zero aggregate START/STOP labels,
data leakage, or alignment assertions.

Eight validation reports and fixed normal/interrupted traces are under
`outputs/session15-main-111b/validation/` and `traces/`. The selection score
combined teacher-forced lexical loss with START/STOP F1, interruption STOP
recall, and weighted loss. Step 600 beat step 800 and all earlier candidates:
its lexical loss was 6.462247 over 181 text targets, weighted loss was
2.190343, and teacher-forced START/STOP F1 were 0.146154/0.115502 over 43
labels each. Their recalls were both 38/43=0.8837, but precision was weak
(38/477 START, 38/615 STOP). Teacher-forced START and STOP absolute boundary
errors were each 4.795 frames (0.192 s; 39 matched turns); raw invalid-state
argmax rate was 985/5,000=0.197. In six selected free traces, real-audio
predictions were IDLE=1,174, START=12, STOP=12, text=2 over 1,200 frames;
no-response and no-STOP were both 2/6. Synthetic interruption STOP recall was
1/3 with 16-frame (0.64 s) latency in the recalled case. Free START/STOP
boundary errors were 33/48.33 frames; raw grammar violation rate was
224/1,300, while selected transitions were legal. Step 800 emitted 99.5% IDLE
and had no post-onset STOP among the three interruptions. The selected PEFT
safetensors digest is
`6c9d507c23f4bb251bac11d4bf16bc9a1f9f0546e6bb9ba115d8c97f7b09c01e`.
`outputs/session15-main-111b/selected/` contains only the adapter and project
metadata, with no redistributed Qwen base weights.

## 2026-09-16 — TASTE full-run batch-size adjustment

The first TASTE full-data launch used the checked-in weighted objective and
batch size one. It was stopped after 580 unsaved steps to use the observed
headroom on an RTX 3090: peak reserved memory was 10.53 GB of 24 GB. The
second launch used batch size two and stopped after 230 unsaved steps: its
peak reserved memory was 13.61 GB. Batch size four then completed 30 steps
with an 11.82 GB peak reservation, and batch size six completed 20 steps with
a 13.72 GB peak reservation. The current full run uses batch size eight and
`outputs/turn-packed-main-batch8`, preserving the initial metrics as
diagnostic evidence. The loss weights remain text=1.0, IDLE=0.1, START=4.0,
STOP=4.0; raw per-group cross-entropies are diagnostics and are intentionally
not used as a reason to change the precomputed global weighting.

Fresh-process loading of that adapter passed. The three cases and full traces
are in `outputs/session15-main-111b/fresh_verify/report.json`. With the frozen
system prompt, the wait case emitted 11 START and 11 STOP events, including
START at frame 11 before its first user activity at 5.44 seconds; it did not
wait reliably. The completed-turn case emitted 25 START and 25 STOP events,
beginning with START at frame 0 before the first observed user activity at
2.76 seconds; it did not start at a clean turn boundary. The interrupted case
had synthetic onset at frame 171 and emitted 28 START and 28 STOP events;
START/STOP cycles at frames 177–180 and 186–187 included post-onset STOP but
no sustained response to interrupt. All three emitted zero lexical events and
zero text. These failures outweigh the improvement in teacher-forced loss;
step 600 is the selected experiment adapter, but **MAIN_CHECKPOINT_READY=FAIL**.

## 2026-09-27 — TASTE `read_aloud_012247` one-sample overfit recipe

This diagnostic replaces the obsolete DailyTalk complete-conversation overfit
entry point. The current epoch-one TASTE adapter is **not** used as initialization:
the script starts the pinned Qwen2.5-Omni-3B Thinker with the same BF16 rank-16
LoRA targets and weighted next-event loss as `configs/turn_packed_main.yaml`.
Only the single dev row with `idx=read_aloud_012247` is put in the training
dataset. Batch size is one, the budget is 300 optimizer steps, and stochastic
noise/interruption are disabled so memorization is measurable on fixed inputs.

Run `.venv/bin/python scripts/run_taste_one_sample_overfit.py` on a 24 GB GPU.
The output report distinguishes exact teacher-forced and cached free predictions
on the training timeline, streaming on the training waveform, and streaming on
the user-only waveform. These are separate gates because chunk context can
change when the assistant-silence block is omitted. No overfit success or GPU
result is claimed yet; this machine was used only for local config, row-selection,
and code checks.

The rented RTX 3090 run later completed all 300 steps. Its report records
`training_overfit_gate=PASS`, `aligned_stream_gate=PASS`, and
`streaming_gate=FAIL`: the 4.919125-second training waveform yielded the exact
123-frame event sequence and response text, while the 2.531-second user-only
audio yielded no response. The static demo under `demo/` replays the passing
training-audio stream with the public input audio and labels the failed
user-only case in the README. The replay is a presentation of the saved trace,
not another model evaluation.

That replay was subsequently removed. A scan of the 4,000 TASTE dev rows and
44,000 training rows found only a handful of 15–20-second conversations; most
are slow recitations. `read_aloud_038934` (training split) is a more natural
15.45-second candidate: 4.41 seconds of instruction and 11.04 seconds of
reference response. The current targets pack response tokens contiguously, so
even this longer example cannot show `IDLE` between answer tokens without a new
alignment experiment. The available Vast machine also changed from an RTX 3090
to an RTX 2080 Ti, which lacks native BF16 support. No longer-sample training
result is claimed.

For the unchanged-model follow-up, the selected row is `read_aloud_038934`
from `shuffled_train_part_0008.parquet`. A separate config selects this one row
and allows up to 500 optimizer steps; the overfit step validator was extended
to 1,000 to permit longer deterministic examples. The actual gate still
requires exact teacher-forced, cached-free, and training-audio streaming events.

The 500-step run completed on the available RTX 2080 Ti. Its final logged
training loss was 3.065e-06 (text 1.505e-06; `IDLE` 1.382e-07; `START`
2.445e-05; `STOP` 1.422e-05). The report records
`training_overfit_gate=PASS`, `aligned_stream_gate=PASS`, and
`streaming_gate=FAIL`. All 386 training-waveform events matched, including 370
`IDLE`, one `START` at frame 110, 14 text tokens, and one `STOP` at frame 125.
The training-waveform text exactly matched the reference countryside sentence;
the user-only waveform returned an empty string after the silent-chunk limit.
The terminal recording under `demo/` replays the verified event sequence at
25 Hz with the TASTE reference speech audible for comparison, not as model
input or synthesized output. It is not a latency measurement.

For a second, visibly longer console example, the selected training row is
`read_aloud_016395` in `shuffled_train_part_0008.parquet`. Its 2.28-second
instruction asks for a quick count from one to thirty; the 276-character
reference text is spoken over 16.24 seconds (18.51 seconds total). The same
Thinker/LoRA/timeline/loss recipe ran for 500 updates on the one-row dataset
under `configs/taste_one_sample_overfit_016395.yaml`. The final teacher-forced
loss was 5.599e-06 (text 3.450e-06; `IDLE` 3.552e-07; `START` 5.794e-05;
`STOP` 4.172e-05). Teacher-forced, cached-free, and training-waveform
streaming predictions matched all 463 frames: 392 `IDLE`, one `START`, 69 text
tokens, and one `STOP`. Thus `training_overfit_gate=PASS` and
`aligned_stream_gate=PASS`. User-only streaming returned empty text, so
`streaming_gate=FAIL`. The saved 25 Hz trace and video under `demo/` replay the
training-waveform path, with the dataset reference speech audible only for
comparison. They do not demonstrate user-only response generation or live
inference speed.


## 2026-09-30 — InstructS2S first-turn comparison

The experiment selects 44,000 training and 1,000 held-out first turns from
InstructS2S, with complete 16–512-token answers and real question audio only.
Shards are processed and discarded to stay within the local disk budget.
It retains the prior rank-16 LoRA, constant 2e-4 learning rate and weighted loss,
but also corrects padding and encoder precision in the audio contract. It is
therefore a data-plus-input-contract comparison, not an isolated response-length
ablation. See [the detailed note](instructs2s_experiment.md) for provenance,
preflight measurements, reproduction commands, and actual results.

The prepared subset occupies 6.54 GB rather than retaining the 345 GB source
archive. Training response length is median 47 tokens versus 7 in TASTE. Full
training completed 3,750 updates / one visit per training example in 4,786.43
seconds, using a 4,000-frame budget with expandable allocator segments. The
original fragmented-allocator failed start is archived separately. Final-window
text loss is 4.02080; augmented dev text loss is 4.02930. Clean 1,000-example dev
text loss is 3.99616 and teacher-forced lexical accuracy is 28.01%. Actual raw
question inference gives some correct facts but mostly malformed/repetitive
answers. This is a failed generalization run despite successful epoch coverage.

Matched 200-update fresh-base gain pilots give text losses 5.92312 (gain 1),
1.94094 (gain 0.03 everywhere), and 1.13768 (gain 1 while listening / 0.03 while
responding). Global attenuation is fluent but ungrounded. Response-only gain
answers Paris, Madrid, six times two, and the rectangle's 50 square centimeters
correctly, but truncates DNA and triangle answers. Longer held-out responses
still contain errors. Training and inference derive response phase from past
START/STOP events through one shared scaling helper; no generation prompt or
reference timing is supplied. These are short diagnostics, not replacement
full-epoch results. Their actual outputs live in each pilot's `evaluation/`.
All three early adapters preserve 12 simple native-text answers by meaning;
the full original adapter answers none of them correctly. Further training with
response-only gain needs retention and completion gates before acceptance.

## 2026-10-01 — complete timelines and continuous conversation

Added a concrete inference-only diagnostic for post-STOP padding and a genuine
three-turn `instruct_en_20` recording, excluded by conversation ID from train/dev.
The rectangle's full 250-frame waveform gives the exact prior 224-event prefix
plus 26 predicted IDLE frames. Continuous DNA/RNA/ATP audio with four-second
silent gaps gives 500 events: 454 IDLE, three START, 40 TEXT, three STOP, with
zero grammar-mask changes. Outputs are respectively truncated, incorrect and
correct. Isolated RNA also fails, so conversation history is not the sole cause.
Only user recordings are provided, without textual history or reference answers.
The checkpoint remains the 200-step response-gain pilot, trained on first turns.
See [the detailed diagnostic](instructs2s_multiturn.md) for strings, all frame
counts, limitations, artifact locations and commands. Default generation and
training were not changed; runtime checks verify all supplied frames are consumed.

## 2026-10-01 — full-epoch response-gain continuation launched

Continued `instructs2s-response-gain003-pilot/checkpoint-200` toward 3,750 total
updates with the exact response-only gain, data, loss, LoRA, learning rate and
augmentation setup. Optimizer state at the first resumed checkpoint verifies
step 250 rather than a restart. The sampler covers 44,000 unique rows once.
TensorBoard and JSONL include the pilot's earlier logging windows; a public HTTPS
tunnel serves only this run's dashboard. Setuptools 80.9.0 restores the dependency
needed by TensorBoard 2.20. Thirteen local checks pass. This run is ongoing, and
post-epoch evaluation is gated on epoch 1.0 / global step 3,750. See
[the full-epoch note](response_gain_full_epoch.md) for settings and commands.

After the server/session interruption, resumed again from the complete
checkpoint 1,200, with all 504 Adam parameter states verified at that update.
The logging history ends at that checkpoint and was preserved. Restarted
dashboard/tunnel and final evaluation watcher; no recipe or epoch-target change.

## 2026-10-01 — response-gain full epoch results

The continuation completed successfully at update 3,750 / epoch 1.0, with all
44,000 conversations represented once in the sampler and logging totals.
Both training and gated evaluation exit zero. Clean 1,000-row dev text loss
is 0.939978, lexical token accuracy 72.71%, weighted loss 0.567940. Last training
window text/IDLE/START/STOP losses are 0.935871 / 0.019539 / 0.102034 / 0.317126.
The complete-epoch target-weighted training loss is 0.618899; the resumed
Trainer's printed 0.3884 is not the complete mean.

All 30 fixed audio cases respond without timeouts; five of six excluded spoken
facts are correct. Three cases have multiple response bursts and four start
before the last question chunk is available. Long-form factual and premature
STOP errors remain. Full supplied recordings preserve post-STOP IDLE:
47 frames for the rectangle and 87 after the final multi-turn ATP response.
DNA/ATP are correct; RNA is wrong continuously and in isolation.

Basic native checks retain correct meaning on 12/12 questions (strict prefix
8/12), but native controls show a real triangle-calculation regression:
base 15, full adapter 25.5, in both transcript and audio modes. Base native
audio also mishears the RNA clip, while both models know the text expansion.
A targeted audit finds an incorrect geometric training reference, without a
matching 25.5 triangle label. These results establish a large improvement over
gain-1 training, not complete semantic correctness or unchanged intelligence.
See [the full experiment note](response_gain_full_epoch.md).

The paired 200-step native controls also answer the triangle correctly as 15
in both modalities. The final 25.5 answer is a continuation-time regression,
despite improved aggregate duplex accuracy. Native base/pilot/full text know
RNA; all three native audio variants mishear this isolated clip. Preserved
basic-check accuracy alone therefore understates skill-specific regression.

## 2026-10-01 — VoiceBench full-suite comparison

Evaluate the completed 3,750-step gain-0.03 checkpoint against original native
Qwen2.5-Omni-3B on all 13,313 standard scored VoiceBench audio examples. Keep
training's empty context / 2-second FP32 encoder duplex representation; use
benchmark native audio chat for baseline. Greedy 2,048-token cap, up to 42
duplex silent chunks, decoded answers without special tokens, saved original
IDs and waveform identities. GPT-4o uses the pinned three-vote open/QA rubric;
other metrics use official scorers with parser-failure strict-score audits.

Optimization trials preserved as diagnostics: dynamic cache, masked static
graph, then packed KV / variable-length FlashAttention with unmerged weights.
Final B=1 event/native parity and cache reuse checks pass; identical-history
graph/eager selected-event agreement 599/600, mean logit difference 0.022636.
Record BF16 batching variation. Local tests: 18 passed. Unicode record reading
and survivor-prefix KV compaction added after 3,198 completed duplex answers;
150-event surviving-row parity passes. Full results pending.
Protocol: [VoiceBench experiment](voicebench_full_epoch.md).
