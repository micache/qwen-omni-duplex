# Full-Duplex-Bench v1.0 — completed 2026-10-05

Evaluated the newly reproduced one-epoch checkpoint on all 727 recordings and
all four v1.0 tasks. The previous machine's queued evaluation was not used.
Source and comparison table:
[pinned benchmark release](https://github.com/DanielLin94144/Full-Duplex-Bench/blob/3e799c45a045256f47d5f1c9cda90157e2d2ec9e/v1_v1.5/README.md),
revision `3e799c45a045256f47d5f1c9cda90157e2d2ec9e` (also upstream HEAD when checked).

## Checkpoint and data

Checkpoint: `outputs/instructs2s-response-gain003-one-epoch/final/`.
Adapter identity SHA256:
`0cc4d4f16adeb754ee795c0a75e5c298b8a586bffe71375cae65cf418e433a68`.
Training-config SHA256:
`7274c15d5ccb9f0c6175280a14e6e5e757abe26e8f9de7af10cc02eea50be684`.
The model remains Qwen2.5-Omni-3B Thinker only, with empty context, additive
fusion, fixed two-second chunks and 25 Hz events. Greedy decoding, seed 17.

Downloaded only the five v1 archives linked by the authors' data instructions.
All inputs are finite, mono and nonempty; annotations fit the original
recordings. Polyphase resampling from 24/48 kHz to 16 kHz preserves duration
within one 16 kHz sample. No noise, normalization or text prompt is added.
The manifest retains original duration/annotations, source-file SHA256 and
prepared-PCM SHA256. Total original input duration: 12,813.4653 seconds.
Data: `data/Full-Duplex-Bench-v1/`; preparation log:
`outputs/full-duplex-bench-prepare.log`.

## Timing bridge

Primary `output.json` uses native START/STOP control boundaries: start at
`start_frame / 25`, end at `stop_frame / 25`. Decode words independently within
each turn, skipping special tokens and retaining subtoken frame times. Anchor
the first word's start and last word's end to the control boundaries so the
released transcript-span formulas see the requested turn times. Interior words
retain their lexical times. Control-only turns have no scored text interval.

Process every padded input frame, including frames after earlier STOPs. No
extra silence or early termination is added. Clip scoring to the original
recording duration. Open turns remain explicitly right-censored; never invent
a STOP. For interruption, crop output at the annotated interruption **end**,
preserving the original clock, as the released ASR preparation does.

The model sees a complete two-second chunk before predicting its events.
Native frame timestamps therefore credit within-chunk lookahead. A separate
`causal.json` places each event at `frame / 25 + 2` and applies the same horizon
and crop. This is a conservative playout schedule, not GPU execution latency.
The alternative file deliberately does not end in `output.json`: upstream
pause/turn scorers would otherwise count it as a second prediction.

Nonempty control intervals replace speech VAD; decoded word timestamps replace
ASR. These are **text timing proxies**, not official speech measurements.
Published comparator conditioning and audio output differ. No speech is
synthesized. Differences between clocks include clipping and therefore change
takeover/frequency as well as latency.

## Results and exact denominators

| Subset | N | Native takeovers | Native TOR | Buffered takeovers | Buffered TOR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Synthetic pause | 137 | 37 | 0.270073 | 19 | 0.138686 |
| Candor pause | 216 | 113 | 0.523148 | 71 | 0.328704 |
| ICC backchannel | 55 | 52 | 0.945455 | 17 | 0.309091 |
| Candor smooth turn-taking | 119 | 115 | 0.966387 | 114 | 0.957983 |
| Synthetic user interruption | 200 | 198 | 0.990000 | 200 | 1.000000 |

| Metric | Native | Buffered |
| --- | ---: | ---: |
| Backchannel frequency / second | 0.046434798108135736 | 0.01661375264058375 |
| Backchannel JSD | 0.8256058553189928 | 0.9134311367571518 |
| Smooth latency / second | 0.059478260869571725 (115 responses) | 1.73719298245616 (114 responses) |
| Interruption latency / second | 0.06697643097643087 (198 responses) | 1.9113199999999997 (200 responses) |
| Interruption relevance judge | Unmeasured | Unmeasured |

Latency is zero-clamped and conditional on takeover. Pause, smooth and
interruption takeover uses the full transcript envelope of at least one second
or more than three words. Backchannel preserves the released short-region TOR
overwrite, inclusion of takeover regions in frequency/JSD, and break at >3 s.
Additional any-region takeover rate is 0.945455 native / 0.327273 buffered.
JSD uses inclusive 200 ms bins, epsilon, linear reference interpolation and
SciPy Jensen-Shannon distance, exactly as released. Frequency and JSD are means
over all 55 recordings, including silence. They use original-clock duration.

No OpenAI API key is configured, so no external judge was called and no rating
was inferred. The upstream table labels its column GPT-4o, while the pinned
interruption script actually requests `gpt-4-turbo`; the comparison retains the
published column label and leaves ours unmeasured.

## Four-GPU execution and verification

Four independent A100-SXM4-40GB workers, one model replica per GPU; shards
contain 182/181/182/182 unique recordings, balanced by padded frame count.
Duration-sorted batches up to 64, frame budget 120,000, four CPU threads per
worker. Reused the existing packed KV cache, builtin FlashAttention and CUDA
graphs. No new dependencies or training changes. Bulk GPU utilization was
96–97%; peak allocated memory 11.52/13.37/13.02/12.56 GiB. Worker inference
times including output persistence were 97.0/74.5/71.7/68.5 seconds. From run
manifest creation through final aggregation: 119.84 seconds (not a streaming
real-time factor or a speedup comparison).

Before full inference, ran one sample from every subset on all four GPUs.
Each worker compared 150 optimized decoding steps with native eager attention:
grammar argmax agreement 0.993333/1.0/1.0/1.0; mean absolute logit differences
0.018771/0.017322/0.020614/0.020052. One frame's argmax differed under BF16;
the optimized path is numerically close, not bitwise identical. These checks
validate representative prefixes, not every frame's optimized/eager equality.

After inference, the runner reloaded and reconstructed all segments and both
output clocks from persisted token IDs, checked checkpoint/data identities,
and recomputed every score. A separate audit checked all **337,750 frames**,
**800 closed turns** and **207 open turns**, raw control grammar, timestamp
ordering/horizons/crops, exact START/STOP conversion and complete coverage.
All 727 samples are present; no truncation or excluded silent recording.

The separate audit executed all four unmodified pinned upstream scorers on the
five smoke samples, then all 727 predictions, for both clocks. Only the VAD/
audio interfaces and external judge were substituted: VAD returns native turn
intervals, the duration-only shape interface passes exact original duration
(without rounding to a fabricated 16 kHz waveform), and a dummy judge response
checks scorer plumbing. Dummy ratings are never included in measured results.
Every TOR, conditional latency, frequency and JSD matches within 1e-12.
Upstream divides by zero for zero qualifying turns; the bridge correctly
records null latency instead. The repository suite passes 62 tests with 14
existing Torch deprecation warnings.

Local artifacts: `outputs/full-duplex-bench-v1-response-gain003/` contains
`manifest.json`, `summary.json`, `upstream-parity.json`, `gpu-*.log`,
`aggregate.log`, per-recording raw events/segments/two clocks/scores, and
`smoke/` with graph precision and scorer-parity reports. README comparators
were checked against the pinned/current upstream table; no comparator reruns.

```bash
# Requires the pinned upstream clone, prepared data and final local checkpoint.
bash scripts/run_full_duplex_v1_4gpu.sh --smoke
.venv/bin/python scripts/audit_full_duplex_v1.py --output outputs/full-duplex-bench-v1-response-gain003/smoke
bash scripts/run_full_duplex_v1_4gpu.sh
.venv/bin/python scripts/audit_full_duplex_v1.py
.venv/bin/python -m pytest -q
```

Existing matching results are resumed and revalidated, not silently overwritten.
Native backchannel behavior remains poor; the very short native response
latencies must be interpreted with the chunk-lookahead limit above.
