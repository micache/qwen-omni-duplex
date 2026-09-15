# Experiments

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
