# Qwen Omni Duplex

A small research repository for adapting the
[Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B) **Thinker** to
text-only full-duplex dialogue. The model listens to user audio in fixed
two-second windows and predicts one next event every 40 ms: a text token,
`IDLE`, `START`, or `STOP`.

The project uses the public
[DailyTalkContiguous](https://huggingface.co/datasets/kyutai/DailyTalkContiguous)
dataset and creates synthetic interruptions for overlap training. It is based
on the Thinker/audio tower from Qwen2.5-Omni, but it does not train the Talker,
generate speech, or add a separate controller.

This is a compact student research reproduction. The aim is to keep the data
alignment, causal timeline, fusion rule, and loss easy to inspect before
running a larger training experiment.

## Method

Each example is a two-second chunk with 50 positions at 25 Hz. The input at a
position contains the previous causal text or control event, while the label is
the current event to predict.

```text
user audio       -> Qwen audio tower -------------------+
previous text    -> Qwen shared token embedding -> mask +--> Qwen Thinker --> next event
previous control -> Qwen shared token embedding -> mask +
```

The three streams are aligned in the Thinker hidden space and added directly:

```text
fused = audio + masked_text + masked_control
```

There is no learned projection, fusion gate, event classifier, or new tokenizer
token. The original Thinker text model and language-model head predict both
lexical tokens and the three control events.

Training uses direct, unshifted weighted cross-entropy because the timeline
builder has already shifted the causal inputs:

```text
loss = sum(weight[target] * cross_entropy(logits, target))
       --------------------------------------------------
                    sum(weight[target])
```

Text, `IDLE`, `START`, and `STOP` have independently configurable weights.
Padding is ignored.

## Current status

The repository is implemented through the small-scale training/checkpoint stage:

- strict two-second/25 Hz causal event timelines;
- whole-utterance text tokenization and word-to-frame alignment;
- contiguous DailyTalk loading, validation, splitting, and collation;
- deterministic synthetic user interruptions;
- Qwen audio-tower feature extraction and flattened batch restoration;
- additive audio/text/control fusion in the shared 2,048-wide hidden space;
- weighted next-event loss with per-group losses and counts;
- BF16 LoRA training with explicit NF4/BF16 QLoRA fallback configuration;
- adapter-only checkpoints, deterministic seeding, metric logging, and resume;
- cache, position-ID, and last-position decoding support;
- CPU unit tests and opt-in real-checkpoint GPU integration tests.

The generation loop is not implemented. No main dataset experiment or benchmark
has been run; Session 08 only ran a three-step synthetic smoke test. The YAML
files in `configs/` are runnable local recipes after the dataset path and any
separately verified speaker-label mapping are supplied.

## Setup

The repository uses Python 3.11 and a local `.venv`. From the repository root:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

The pinned environment uses PyTorch 2.10.0 with CUDA 12.6 and Transformers
5.17.0. FlashAttention is optional and was not used for the accepted
compatibility path; SDPA is the tested attention implementation.

## Dataset preparation

The expected dataset layout contains `dailytalk.jsonl` and `data_stereo/` under
one directory. To validate an existing local copy and create a deterministic
conversation-level split index:

```bash
.venv/bin/python scripts/prepare_dailytalk.py \
  --dataset-root /path/to/DailyTalkContiguous \
  --output data/dailytalk_splits.jsonl
```

The dataset is not downloaded automatically. To explicitly download the public
Hugging Face snapshot first, add `--download`:

```bash
.venv/bin/python scripts/prepare_dailytalk.py \
  --dataset-root data/DailyTalkContiguous \
  --output data/dailytalk_splits.jsonl \
  --download
```

Stereo channel 0 is treated as assistant/reference audio and channel 1 as the
user/model input. Only the user waveform is passed to the Thinker. The public
sidecar format does not establish every user word label, so cross-speaker
boundary construction requires an explicitly verified speaker mapping.

Inspect one local conversation and its proposed two-second windows with:

```bash
.venv/bin/python scripts/inspect_sample.py \
  /path/to/DailyTalkContiguous/dailytalk.jsonl \
  --index 0
```

## Qwen compatibility probe

The repository pins checkpoint revision
`f75b40e3da2003cdd6e1829b1f420ca70797c34e`. The direct Thinker loading path,
real two-second audio encoding, batch restoration, text prefill, and cached
one-step decoding were checked with:

```bash
.venv/bin/python scripts/probe_qwen.py \
  --revision f75b40e3da2003cdd6e1829b1f420ca70797c34e
```

This probe loads the real checkpoint and requires CUDA. It passed on an RTX
2060 SUPER using WSL managed-memory paging, but a GPU with at least 24 GB VRAM
is preferred for practical experiments. Exact shapes, package versions, and
memory measurements are recorded in
[`notes/compatibility.md`](notes/compatibility.md) and
[`notes/session06_probe.json`](notes/session06_probe.json).

## Tests

Run the normal CPU suite with:

```bash
.venv/bin/python -m pytest -q
```

Real-checkpoint GPU tests are opt-in so normal pytest never downloads or loads
the model. They require the pinned checkpoint to already exist in the local
Hugging Face cache:

```bash
RUN_QWEN_GPU_TESTS=1 \
  .venv/bin/python -m pytest -q tests/test_model_gpu.py -s
```

The GPU suite checks text-only parity with the base Thinker, one real
two-second audio forward, a batch-of-two forward, and no-gradient peak VRAM.

## Training

The primary 24 GB path is BF16 LoRA. Edit only the local dataset path and any
verified user speaker label in `configs/train_lora.yaml`, then run:

```bash
.venv/bin/python train.py --config configs/train_lora.yaml
```

The explicit 16 GB fallback is `configs/train_qlora_16gb.yaml`, which selects
4-bit NF4 loading with BF16 compute and labels outputs as QLoRA. Training never
switches to it automatically after an out-of-memory error. Both paths save only
PEFT adapters, processor/tokenizer files, reconstruction metadata, and Trainer
resume state; full Qwen base weights are never written.

## Repository structure

```text
duplex/dataset.py       DailyTalk reader, windows, augmentation, and collator
duplex/timeline.py      event grammar, token alignment, and causal shifting
duplex/model.py         additive full-duplex Thinker wrapper and weighted loss
duplex/training.py      LoRA/QLoRA loading, Trainer, logging, and checkpoints
scripts/                data inspection, preparation, and compatibility probe
tests/                  CPU unit tests and opt-in GPU integration tests
notes/                  design decisions, compatibility evidence, and progress
configs/                BF16 LoRA, explicit QLoRA fallback, and smoke recipes
```

## References

- [Qwen2.5-Omni Technical Report](https://arxiv.org/abs/2503.20215)
- [Official Qwen2.5-Omni repository](https://github.com/QwenLM/Qwen2.5-Omni)
- [Qwen/Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B)
- [kyutai/DailyTalkContiguous](https://huggingface.co/datasets/kyutai/DailyTalkContiguous)
- [Full-Duplex-Bench](https://full-duplex-bench.github.io/)

Any future Full-Duplex-Bench result from this repository must be labeled a
**text-timeline adaptation**, not an official speech-output score.

## License and scope

The repository is released under the Apache 2.0 license. It is an independent
research project and is not affiliated with or endorsed by Qwen, Alibaba,
Kyutai, DailyTalk, or Full-Duplex-Bench. Upstream models and datasets remain
subject to their own licenses and terms.

See [`notes/design.md`](notes/design.md) for the detailed timeline decisions and
[`notes/progress.md`](notes/progress.md) for the session-by-session record.
