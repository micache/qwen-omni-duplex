# Benchmarks

## VoiceBench Session 12 adaptation

`voicebench.py` produces text responses in the JSONL shape used by official
VoiceBench: it preserves every dataset field except `audio`, adds `response`,
and adds `voicebench_id` plus a compact `run_manifest`. This is a local
text-output adaptation, not an official score.

The implementation was inspected against the official
[MatthewCYM/VoiceBench](https://github.com/MatthewCYM/VoiceBench) `main` branch
at revision `6992cf4fc51d0426c52c4805b5002e0aae49118a`, verified on 2026-09-15.
That revision is pinned in every manifest. It is not vendored here.

Base audio mode loads the exact pinned `Qwen/Qwen2.5-Omni-3B` revision, calls
`disable_talker()`, and uses the model's official half-duplex audio-to-text
generation. Duplex audio mode loads that same revision with a Session 08+
adapter and uses the repository's fixed two-second streamer, including silent
tail chunks until STOP or the declared limit. Text mode uses ordinary Qwen
chat/text generation; in duplex mode the adapter remains active. Pair runs must
use the same prompt, seed, and `--max-new-tokens` value.

Example bounded runs:

~~~bash
.venv/bin/python benchmarks/voicebench.py run \
  --model-mode base --data sd-qa --split usa --modality audio \
  --limit 2 --start-index 0 --seed 17 --max-new-tokens 256 \
  --output outputs/voicebench/base-sdqa-usa-audio.jsonl

.venv/bin/python benchmarks/voicebench.py run \
  --model-mode duplex --adapter outputs/session11-diagnostic/final \
  --data sd-qa --split usa --modality audio \
  --limit 2 --start-index 0 --seed 17 --max-new-tokens 256 \
  --output outputs/voicebench/duplex-sdqa-usa-audio.jsonl
~~~

`--checkpoint` is an alias for `--adapter`; both name an adapter-only Trainer or
final checkpoint directory. Loading is cache-only unless `--allow-download` is
explicit. Re-running the same range resumes by stable ID and never duplicates
records; a changed manifest is rejected.

Validate a complete pair and prepare, but do not execute, the pinned upstream
judge/evaluator commands:

~~~bash
.venv/bin/python benchmarks/voicebench.py summarize \
  --base outputs/voicebench/base-sdqa-usa-audio.jsonl \
  --duplex outputs/voicebench/duplex-sdqa-usa-audio.jsonl \
  --upstream-dir /path/to/VoiceBench \
  --output outputs/voicebench/sdqa-usa-audio-pair.json
~~~

The summarizer rejects missing IDs, duplicate IDs, mixed manifests, and paired
setting mismatches. GPT-judged subsets require the emitted `api_judge.py`
command before `evaluate.py`; pytest never executes either command. Smoke
outputs must only be checked for nonempty compatible JSONL and must not be
interpreted as scores.

## Full-Duplex-Bench Session 13 v1.0 adaptation

`full_duplex_bench.py` implements only the four v1.0 tasks as a
**Full-Duplex-Bench v1.0 text-timeline adaptation**. It was inspected against
the public v1/v1.5 READMEs, v1.0 paper, and v1.0 evaluators at upstream revision
`3e799c45a045256f47d5f1c9cda90157e2d2ec9e`. No upstream data or code is
vendored. Supply a local v1.0 data directory explicitly:

~~~bash
.venv/bin/python benchmarks/full_duplex_bench.py \
  --model-mode duplex \
  --adapter outputs/session11-diagnostic/final \
  --data-dir /path/to/data-full-duplex-bench/v1_0 \
  --task smooth_turn_taking \
  --limit 1 \
  --output outputs/full-duplex-bench/smooth-duplex.json
~~~

For backchannel JSD, also pass the caller-owned upstream
`icc_gt_distribution.json` with `--ground-truth-distribution`. Duplex mode
consumes the native event trace. Base mode uses manual streamed generation only
after the entire input is available and retains measured token availability
times. Results preserve exact decoded text and separately expose lexical word,
START, STOP, logical, and causal-availability timing. The runner does not use
TTS/VAD, synthesize speech, or call the interruption relevance judge. Missing
conditional metrics are `null` with coverage. v1.5 is not implemented.

This remains a text-timeline adaptation, not an official speech-output score.
