# Full-Duplex-Bench v1: full-epoch checkpoint

## Status

Dataset preparation and the v1 scoring bridge are complete. Model inference
has not started: the A100 40 GB has about 38.4 GB reserved by a VLLM engine
belonging to `qvr_phase4`. We are awaiting the user's answer about releasing
that GPU. A detached local runner polls for at least 30,000 MiB free, then runs
the five-sample smoke/graph-precision check, all 727 inputs, and score validation.
It does not stop the VLLM workload. Its PID and status are in
`outputs/full-duplex-bench-v1-response-gain003/supervisor.pid` and `supervisor.log`.
No checkpoint benchmark scores or comparison claims are available yet.

## Sources and data

Use only v1 from [the benchmark release](https://github.com/DanielLin94144/Full-Duplex-Bench/tree/3e799c45a045256f47d5f1c9cda90157e2d2ec9e/v1_v1.5),
revision `3e799c45a045256f47d5f1c9cda90157e2d2ec9e`.
The authors' [data instructions](https://github.com/DanielLin94144/Full-Duplex-Bench/blob/3e799c45a045256f47d5f1c9cda90157e2d2ec9e/v1_v1.5/dataset/README.md)
link the public Google Drive release. Downloaded the five v1 archives only.

| Test / subset | Samples | Original sample rate |
| --- | ---: | ---: |
| Pause / Candor | 216 | 16 kHz |
| Pause / synthetic | 137 | 24 kHz |
| Backchannel / ICC | 55 | 48 kHz |
| Smooth turn-taking / Candor | 119 | 16 kHz |
| User interruption / synthetic | 200 | 24 kHz |

All 727 inputs are mono, finite and nonempty. Every annotation fits within its
recording. The 24/48 kHz inputs are polyphase-resampled to 16 kHz, with durations
preserved within one 16 kHz sample; timestamps remain on the original clock.
No normalization, noise or text prompt is added. The prepared manifest stores
source-file and resampled-PCM SHA256s plus the original annotations.

Local data: `data/Full-Duplex-Bench-v1/` (2.1 GB including archives and prepared
inputs). Preparation log: `outputs/full-duplex-bench-prepare.log`.
Checkpoint: `outputs/instructs2s-response-gain003-one-epoch/final/`.
Planned results: `outputs/full-duplex-bench-v1-response-gain003/`.

## Timing and scoring

The native model produces one event every 40 ms and sees complete 2-second
audio chunks. A frame-position timestamp alone can precede audio that influenced
the prediction. Primary scoring therefore uses a fixed 2-second playout buffer:
event frame `f` is placed at `f / 25 + 2`. This is never earlier than the end of
the chunk used by that event and preserves the 25 Hz text-emission durations.
The nominal `f / 25` clock is also retained, separately. Neither clock measures
GPU runtime or establishes real-time performance.

All input chunks are processed with the existing batched decoder. Configure
zero additional silent chunks and a lexical cap above the maximum possible
recording frame count. Earlier STOPs do not terminate the recording. Partial
input chunks are padded exactly as in training. Emitted words after the original
recording end are excluded from scoring; raw events and unfinished answers remain
saved. For interruption, crop word intervals at the interruption **end**, as
the released ASR script crops output audio there. Do not reset that clock to zero.

Decode words independently within each START/STOP response, skipping special
tokens but preserving raw event IDs. Multi-token words span the first through
last contributing event, with one frame of width for the final token.
Control-only answers contribute no words or response intervals. Nonempty answer
envelopes replace speech VAD regions; they are a text-only proxy, not measured
speech durations. Published audio-model scores are not directly comparable.

The v1 formulas match the pinned evaluators, including their edge cases:

- Pause and turn-taking: takeover if the full transcript span is at least one
  second or contains more than three words. They do not use the backchannel
  two-word threshold. Latency is clamped at zero and averaged only for takeovers.
- Backchannel: the duration/word thresholds are applied to each proxy region;
  no response gives frequency zero and JSD one. Use the released inclusive
  200 ms bins, epsilon, linear reference interpolation and SciPy JSD.
- The released backchannel scorer overwrites TOR on successive short regions,
  includes those regions in frequency/JSD even if they are takeovers, and stops
  scanning at a region longer than three seconds. Preserve this behavior for
  reference parity; additionally save `any_region_takeover_rate` for inspection.
- Skip the interruption relevance judge. There are no external API calls.

The earlier Session 13 adapter is not used: its loader adds an incompatible chat
prefix, uses different takeover aggregation, and excludes some backchannel
cases from conditional metrics. The final checkpoint requires empty context.

## Validation so far

15 focused tests cover subtoken word timing, special-token exclusion, independent
responses, open answers, control-only silence, crop/horizon boundaries, causal
chunk boundaries, released scoring thresholds and multi-response inference.
The multi-response test executes the batched engine's eager loop with a small
deterministic CPU decoder and confirms all 100 frames are processed after the
first STOP. Together with the existing inference checks, 20 tests pass;
the full repository suite passes all 50 tests (14 existing Torch deprecation
warnings).

Also executed all four actual pinned upstream evaluators against 48 synthetic
fixtures (12 per test). The adapter matches TOR and conditional latency exactly,
and backchannel frequency/JSD within the released four-decimal output precision.
Only synthetic VAD regions and an in-process synthetic judge response were
substituted for that code-parity check; no speech or model outputs were fabricated,
and no judge score will be reported. Fixture report:
`outputs/full-duplex-bench-v1-fixture-validation.json`.

## Comparison source after inference

Use the benchmark maintainers' [pinned v1 result table](https://github.com/DanielLin94144/Full-Duplex-Bench/blob/3e799c45a045256f47d5f1c9cda90157e2d2ec9e/v1_v1.5/README.md)
for PersonaPlex, Moshi and Freeze-Omni. Do not mix it with NVIDIA's separately
reported PersonaPlex protocol or newer third-party reproductions. Published
models produce speech, and PersonaPlex uses task-specific text/voice conditioning.
Our checkpoint has empty context and text-only output. The eventual README table
must identify this distinction and leave our judge column unmeasured.
