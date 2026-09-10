# Compatibility

## Known contract

- Intended model: `Qwen/Qwen2.5-Omni-3B`.
- Intended component: Thinker only.
- Output: text only.
- QLoRA is represented by a 16 GB-oriented configuration sketch, not yet
  validated against any particular GPU or software version.

## Not yet verified

The exact upstream `transformers` class, processor inputs, audio feature shape,
Thinker module names, LoRA target modules, quantization compatibility, and
memory use remain unverified. `scripts/probe_qwen.py` is a placeholder for the
next compatibility session. Session 01 performs no network access and downloads
neither model weights nor datasets.
