# Repository instructions

This is a compact personal research reproduction, not a production package.
Keep changes small, explicit, and tied to the session plan. Do not download
models or datasets unless a later session explicitly asks for it. Never commit
or push on the user's behalf.

## Non-negotiable scope

- Backbone: **`Qwen/Qwen2.5-Omni-3B` Thinker only**.
- Timeline: **fixed 2 s chunks**.
- Frame rate: **25 Hz only after config validation**.
- Fusion: **additive audio/text/control fusion**.
- Control events: **`IDLE` / `START` / `STOP`**.
- Objective: **weighted next-event loss**.
- Dataset view: **DailyTalkContiguous**.
- Interruption data: **synthetic interruption**.
- Output: **text output only**.

## Explicit exclusions

Do not add Talker training, audio generation, codec work, Qwen3, MoE/DeepSpeed
patches, other backbones, adaptive chunking, plugin abstractions, production
services, or Huawei-confidential material.

## Repository conventions

- Use `requirements.txt` and plain YAML files.
- Do not add a `pyproject.toml`, Docker setup, web server, CI workflow, Hydra,
  or a generic registry.
- Use pytest only for local correctness checks.
- Keep unimplemented modules as honest, minimal placeholders; do not invent
  APIs before an experiment requires them.
- Document material experiment changes in `notes/progress.md` and
  `notes/experiments.md`.
