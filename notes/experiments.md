# Experiments

No experiments have been run. Session 01 creates structure and executable scope
checks only.

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
