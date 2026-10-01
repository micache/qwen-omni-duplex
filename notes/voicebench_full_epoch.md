# VoiceBench: original 3B versus completed duplex epoch

Status: full paired inference and GPT-4o judging running; final scores pending.

## Fixed protocol

- Adapter: `outputs/instructs2s-response-gain003-one-epoch/final`, completed
  3,750-step / 44,000-conversation epoch, response audio gain 0.03.
- Baseline: untouched `Qwen/Qwen2.5-Omni-3B` Thinker, revision
  `f75b40e3da2003cdd6e1829b1f420ca70797c34e`. Our own 3B baseline, not the
  published VoiceBench 7B score. No Talker is constructed.
- VoiceBench repository: `b56154172f2a57a43d29005de7d0471d748d70d1`.
  HF dataset: `hlt-lab/voicebench`, converted Parquet revision
  `de69a22f41561676635bef0b31681df4b866ec07`. Metadata is saved with the run.
- Standard scored suite: commoneval 200, alpacaeval_full 636, wildvoice 1,000,
  SD-QA 6,083 (all 11 accents, 553 each), IFEval 345, AdvBench 520,
  OpenBookQA 455, MMSU 3,074 (all 12 subjects), BBH 1,000. Total 13,313 paired
  audio examples. The overlapping 199-item alpacaeval, optional speaker
  variants, and unscored MTBench are outside this standard suite.
- Both receive the same SHA-256-checked 16 kHz waveform, with no transcription,
  reference, or answer hint. Mono/resampling use soundfile/scipy. PCM-24 FLAC
  encoding error is checked below `2e-7 * audio_scale`; scale is restored
  before inference, without amplitude normalization of actual model input.
- Duplex: exact trained empty context, padded 2-second chunks, 25 Hz, FP32
  audio encoder, BF16 decoder, additive fusion, causal response gain. Clean
  evaluation without training noise or synthetic interruptions.
- Native baseline: VoiceBench audio chat, `You are a helpful assistant.`
  system, original BF16 audio/text weights and native positional encoding.
- Greedy decoding, 2,048-token cap for both; up to 42 extra duplex silent
  chunks (84 seconds). No forced STOP; empty/capped/timed-out answers retained.
- Actual generated IDs are decoded with `skip_special_tokens=True` and
  `clean_up_tokenization_spaces=False`. Original event/raw-argmax IDs and
  native IDs are retained in compressed NPZ sidecars.

## Judge and scoring

GPT-4o access works (resolved `gpt-4o-2024-08-06`). Credential confined to an
owner-only temporary runtime file; excluded from code, logs, artifacts and git.
Use pinned `api_judge.py` rubrics/system with temperature 0.5, top_p 0.95, n=3,
maximum 1,024 judge tokens, zero penalties. Only replace upstream's GPT-4o-mini
with requested GPT-4o. Open subsets average three 1–5 ratings; SD-QA uses
majority Yes/No accuracy. Optional PEDANT/PANDA is outside this GPT metric.
Every vote, returned model, usage and request ID is recorded.

Other metrics use unchanged pinned official scorers. MCQ/BBH randomly guess
when extraction fails: seed 17, additionally report strict accuracy (failed
extraction wrong) and failure rates. AdvBench is keyword refusal rate and
counts empty answers as refusals. IFEval averages strict/loose prompt/instruction
accuracy. MMSU aggregate is weighted over all examples; preserve subject and
accent splits. This compares native audio processing with duplex streaming,
so differences include the interface change.

## Speed and validation

Sequential Parquet-to-FLAC preparation deletes only temporary downloads,
avoiding a second 9 GB Arrow cache. Duration-sorted inference batches up to
128 within a 300,000-frame KV budget, halves on OOM, and flushes each batch.
Finished rows compact at chunk boundaries once half the batch ends; each
survivor retains its KV prefix, previous event, position and control state.
This began after 3,198 completed duplex answers (see execution-update metadata).
The waveform, checkpoint, decoding limits and input protocol remain fixed.
Inference and 16-worker judging run in tmux and resume finished examples.

Native preprocessing extracts each clip before padding and projects only the
last position. B=1 reproduces the original unoptimized native audio path for
64 generated IDs. Duplex keeps unmerged LoRA and original projections/RoPE,
using packed static KV and PyTorch 2.10 variable-length FlashAttention in CUDA
graphs. An extreme-future-slot CUDA test exactly matches prefix-only SDPA for
four lengths. Autocast weight caching is disabled during graph capture/replay.

B=1 graph inference reproduces all 150 reference events on a fixed case;
reused graph/cache also reproduces it. With identical histories on 150 frames
and four examples, graph/eager selected events agree 599/600; mean absolute
logit difference 0.022636, worst vocabulary logit difference 1.625 (low-scoring
token). BF16 batching/kernel near-ties can change subsequent text; do not claim
bitwise equality across batch sizes. Earlier dynamic-cache and masked-graph
trials are retained under `diagnostic-*` and excluded from final scoring.
Live compaction also preserves all 150 events for the surviving longer case;
a cache-selection test verifies prefix/order preservation. JSONL readers keep
Unicode separators inside strings; the live judge consumes only physically
newline-terminated records. Local tests: 18 passed.

## Reproduce and artifacts

Use `.venv/bin/python`. Clone the pinned upstream into
`outputs/voicebench-upstream`; preserve HF metadata in the run directory.

```bash
.venv/bin/python scripts/prepare_voicebench.py
.venv/bin/python scripts/run_voicebench_suite.py --preflight --mode duplex
.venv/bin/python scripts/run_voicebench_suite.py --preflight --mode base
.venv/bin/python scripts/run_voicebench_suite.py --batch-size 128 --frame-budget 300000
.venv/bin/python scripts/judge_voicebench.py --workers 16
.venv/bin/python scripts/score_voicebench_suite.py
.venv/bin/python -m pytest -q
```

Artifacts: `outputs/voicebench-response-gain003/`: `logs/`, `base/`, `duplex/`,
`judged/`, model `manifest.json`, compressed `tokens/`. Source audio/manifests
in `data/VoiceBench-eval/`. Final audited table: `comparison.md`; full metrics:
`scores.json`. Verify all sample counts, source fields, waveform identities,
decoded tokens and judge responses before interpreting the table.
