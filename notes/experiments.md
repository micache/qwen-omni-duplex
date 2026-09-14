# Experiments

Session 01 created structure and executable scope checks only; later dated
entries record the first local probes and experiments.

Future entries should record the date, configuration file, code revision,
hardware, data snapshot, exact command, result, and interpretation. Adapted
Full-Duplex-Bench results must be labeled “text-timeline adaptation,” never as
official speech-output scores.

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
