# qwen-omni-duplex

> Work in progress: this repository currently contains only the Session 01
> scaffold and scope contract.

This is a personal, clean-room research reproduction exploring whether the
**Thinker** in `Qwen/Qwen2.5-Omni-3B` can be adapted to predict a text-only
full-duplex dialogue timeline. The goal is deliberately narrow: reproduce the
core modeling ideas in a small, inspectable experiment, not create a production
package or reproduce the complete Qwen-Omni speech stack.

## Ideas to reproduce

1. Represent a conversation as fixed two-second chunks.
2. Fuse audio, text, and control information additively.
3. Predict `IDLE`, `START`, and `STOP` alongside text as next events.
4. Train with a weighted next-event objective on contiguous DailyTalk examples
   augmented with synthetic interruptions.

## Current status

Session 01 is complete: repository layout, configuration sketches, scope
validation, placeholder entry points, and local contract tests exist. Model
loading, dataset downloading/preparation, training, generation, and benchmark
execution are not implemented. No model or dataset is bundled or downloaded.

## Planned commands

These commands describe the intended interface. Except for the local tests,
they are placeholders until later sessions implement them.

```bash
python -m pip install -r requirements.txt
python scripts/probe_qwen.py --model Qwen/Qwen2.5-Omni-3B
python scripts/prepare_dailytalk.py --output data/dailytalk_contiguous
python train.py --config configs/debug.yaml
python train.py --config configs/train_lora.yaml
python generate.py --config configs/debug.yaml
python benchmarks/voicebench.py --help
python benchmarks/full_duplex_bench.py --help
pytest -q
```

Full-Duplex-Bench results in this project will be a **text-timeline
adaptation**, not official speech-output scores, and will always be labeled as
such.

## Clean-room notice

This independent research repository is written from public information and
original experimentation. It is not affiliated with or endorsed by Qwen,
Alibaba, DailyTalk, VoiceBench, or Full-Duplex-Bench. Do not add proprietary,
leaked, or Huawei-confidential material. Model and dataset use remains subject
to each upstream license and terms.

See [AGENTS.md](AGENTS.md) for the non-negotiable scope and
[notes/design.md](notes/design.md) for the initial design boundary.
