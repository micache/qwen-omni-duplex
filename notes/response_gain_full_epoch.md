# Response-only audio-gain full epoch

The user explicitly requested completing the current 200-update pilot to a full
epoch and evaluating only after completion. The continuation starts from
`outputs/instructs2s-response-gain003-pilot/checkpoint-200`, including LoRA,
optimizer, scheduler and saved Trainer/RNG state. The new recipe is
`configs/instructs2s_response_gain003_one_epoch_a100.yaml` and its output directory
is `outputs/instructs2s-response-gain003-one-epoch/`.

The target is **3,750 total updates / one epoch**, not 3,750 additional updates.
The sampler plans 44,000 unique examples exactly once: 2,324 examples in the
first 200 batches, then 41,676 in the remaining 3,550. A saved step-250 optimizer
has all 504 parameter states at step 250, verifying optimizer continuation.
The resumed Trainer state also has maximum step 3,750. Its checkpoint interval
is inherited as 50 from the pilot state despite the recipe requesting 500;
retention is capped at 12 checkpoints. This affects disk writes, not the objective.

The numerical/data setup is unchanged: full audio gain while listening, gain
0.03 after past START until past STOP; rank-16 LoRA on the same seven projection
types; constant 2e-4 learning rate; frozen FP32 audio encoder, BF16 text decoder,
TF32 off; fixed two-second chunks / validated 25 Hz; empty context; weighted
next-event loss; 20% MUSAN noise at 5–20 dB across the full input timeline;
20% synthetic interruption. Normal Trainer resume restores model-side RNG;
per-worker augmentation draws are not claimed bitwise identical to an
uninterrupted process.

The server/session interruption stopped the first continuation after checkpoint
1,200 (epoch 0.32). All 504 saved Adam parameter states are at update 1,200;
the maximum remains 3,750. Recovered from that checkpoint with the same
optimizer, scheduler and Trainer/RNG state. Metrics end at exactly 1,200, so
no logged windows needed discarding. Previous logs and dashboard metadata are
archived under `recovery/20261001T085200Z/`; new training output appends to
`train.log`. A TensorBoard purge marker at step 1,201 retains earlier curves
and prevents stale future points. The dashboard, public tunnel and gated
evaluation watcher were restarted; the temporary URL changed. Recovery details
are saved in `resume-provenance.json`. The target remains 3,750 total updates.

## Monitoring and commands

JSONL logging and TensorBoard are enabled. The first 200 pilot logging windows
are seeded into the continuation JSONL and scalar event files, preserving the
full curve. `TENSORBOARD_LOGGING_DIR` is the supported environment variable in
the installed Transformers 5.17 TensorBoard callback. Scalar tags include
`train/train_text_loss`, IDLE/START/STOP losses, epoch, learning rate and peak GPU
memory. TensorBoard reloads every five seconds.

TensorBoard 2.20 imports `pkg_resources`, which
[Setuptools removed in version 82](https://setuptools.pypa.io/en/stable/deprecated/pkg_resources.html).
Pinned `setuptools==80.9.0` in requirements after the dashboard's first startup
failed with that missing import. The restarted dashboard serves the scalar API.

The dashboard listens on localhost port 6006, exposed through an official
[Cloudflare Quick Tunnel](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/).
The temporary public HTTPS URL and tmux session names are saved in
`outputs/instructs2s-response-gain003-one-epoch/monitoring.json`. Public HTTPS
scalar/plugin requests were verified. Only this run's TensorBoard directory is
served. The tunnel/dashboard run in detached tmux sessions and remain available
during and after training while the server sessions are alive.

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 TENSORBOARD_LOGGING_DIR=outputs/instructs2s-response-gain003-one-epoch/tensorboard .venv/bin/python -u train.py --config configs/instructs2s_response_gain003_one_epoch_a100.yaml
.venv/bin/python -m tensorboard.main --logdir outputs/instructs2s-response-gain003-one-epoch/tensorboard --host 127.0.0.1 --port 6006 --reload_interval 5 --load_fast false
/home/gpuuser/.local/bin/cloudflared tunnel --url http://127.0.0.1:6006 --protocol http2 --no-autoupdate
```

## Completion and evaluation

Training and the gated evaluation both finished with exit code zero. The final
checkpoint records **step 3,750 / epoch 1.0**, and all 504 Adam parameter states
are at step 3,750. Accumulated training windows cover 14,550,100 frames,
2,007,129 lexical labels after interruption augmentation, 12,454,971 IDLE labels,
and 44,000 START / 44,000 STOP labels, with no duplicate logging steps. The
saved adapter is `outputs/instructs2s-response-gain003-one-epoch/final/`.

The final evaluation ran the first 12 train / first 12 dev questions, six fixed
excluded spoken facts, clean teacher-forced evaluation of all 1,000 dev rows,
cached/full parity, the same continuous three-turn recording, and ordinary
no-prompt CLI generation. All 30 questions produced nonempty responses without
timeouts. This is not a claim that all responses are correct.

| Metric | Response-gain full epoch | Original gain-1 full epoch |
| --- | ---: | ---: |
| Clean dev text loss | 0.939978 | 3.996157 |
| Clean dev lexical token accuracy | 72.71% | 28.01% |
| Clean dev weighted loss | 0.567940 | 2.358392 |

Last 25-update training losses are text 0.935871, IDLE 0.019539, START 0.102034,
STOP 0.317126. Clean dev control losses are IDLE 0.014796, START 0.030704,
STOP 0.247552. The training weighted mean across all recorded windows is
0.618899, calculated with each window's weighted target count. The Trainer's
printed `train_loss=0.3884` uses the resumed process's numerator with the global
step denominator; it is not the complete epoch mean. The last continuation
reports 3,318 seconds including its epoch-end evaluation, rather than the
entire epoch's active duration or the interruption downtime.

The native text diagnostic retains the correct meaning on all 12 basic
questions. Strict expected-prefix matching is 8/12: several numerical answers
are spelled out or contain introductory words. This demonstrates retained
basic knowledge, not unchanged general intelligence or instruction following.

Five of the six fixed excluded spoken facts are now exactly correct: Paris,
DNA's full expansion, 6 times 2, rectangle area 50, and Madrid. The pilot had
four complete correct facts and truncated DNA. Triangle area is still wrong:
an early response asks for measurements already present later in the clip;
a second response says twenty-five point five rather than fifteen.

Several longer outputs are relevant and coherent. Remaining failures include
incorrect hexadecimal conversion, misassigned philosophical positions relative
to the dataset reference, Marlin rendered as Marilyn, a repeated movie title,
and a feminist-poets response ending at `by challenging traditional`.
`control-timing.json` records three multi-response cases and four STARTs before
the final question chunk was available. The early test uses chunk availability,
not frame time, to avoid mistaking ordinary two-second lookahead for premature
speech. Three grammar-mask overrides occur in the hexadecimal sample; the
other 29 samples have none. Event traces retain both emitted and raw argmax IDs.

Train/inference mel tensors match exactly on all nine checked examples. Frozen
FP32 audio embeddings differ by at most 0.000433 in batch versus individual
chunk encoding. Cached/full gold-history argmax agreement is 49/49 on the
first train example, 69/71 on the first dev example, and 7/7 on the France
probe, with maximum logit differences 0.3125 / 0.25 / 0.25 from BF16 rounding.
The ordinary CLI rectangle generation has identical text and event IDs to the
evaluator. It ends at STOP after 203 events; a full supplied 250-frame waveform
adds 47 predicted IDLE events after that same STOP.

The 500-frame continuous DNA/RNA/ATP recording retains one cache and emits
three START/STOP pairs, 33 TEXT events and 461 IDLE events. DNA and ATP are
correct; RNA repeats the DNA answer. With a fresh cache, RNA instead says
`Zarrene stands` and ATP remains correct. Therefore history alone does not
explain the RNA failure. The continuous recording ends with 87 predicted IDLE
frames after the last STOP. Multi-turn audio processing works, but multi-turn
semantic correctness is not established by this first-turn-trained adapter.

Artifacts in the run directory include `run-summary.json`,
`completed-epoch-verification.json`, `fixed-question-comparison.md`,
`final-evaluation/report.json`, `final-evaluation/samples.md`, all 30
`final-evaluation/raw/*.txt` strings, input FLACs and event traces,
`multiturn-evaluation/` and `multiturn-isolated-controls/`. Raw strings preserve
native IDLE `<|endoftext|>`, START `<|im_start|>` and STOP `<|im_end|>` tokens.

## Paired native failure controls and data audit

After all main evaluation jobs completed, compared base and full adapter on
the hexadecimal, philosophers, triangle and isolated RNA failures, using both
native question transcript and native question audio. These are separate native
chat diagnostics, not the empty-context duplex inference protocol. Results and
unabridged generated native token IDs/strings are in
`native-failure-controls.json`.

The base answers the triangle correctly as 15 in both modalities. The final
adapter answers 25.5 in both native modalities, matching its wrong duplex answer.
This is a specific regression in the adapted decoder, not just audio perception
or premature START. Retaining the 12 elementary native checks is insufficient
to claim unchanged mathematical intelligence. The base already makes errors on
the hexadecimal task: its transcript answer maps hexadecimal 21 to 101001,
whereas 0x21 is 100001. The dataset reference itself treats 21 as decimal and
contains malformed wording for 3A; these labels are not a trustworthy benchmark
for exact correctness.

Both base and final adapter correctly expand RNA from a native text question.
The base hears the recorded question as `Zar Rene`; the adapter responds with
a fabricated expansion for `ZARRENE`. The same clip fails with a fresh duplex
cache as well. This points to an audio-perception issue for this recording in
the 3B model, not loss of the underlying RNA fact. It does not establish that
the recording is objectively unintelligible or that every audio path is equal.

The targeted training-label audit finds six prompts containing both `triangle`
and `area` (one concerns the Coral Triangle region), and one answer containing
`twenty-five point five`, which concerns Maputo birth rates rather than geometry.
One geometry reference demonstrably miscalculates sides 5, 10, 8: it gives a
semiperimeter of 17 and area 20.98, versus 11.5 and approximately 19.8100.
`triangle-label-audit.json` preserves the selected rows and calculation. This
establishes noisy mathematical supervision but does not directly explain the
specific learned 25.5 answer.

Repeated the paired controls with the unchanged 200-step pilot, saved in
`pilot-native-failure-controls.json`. The pilot answers the triangle correctly
as 15 in both native transcript and native audio modes; its duplex answer had
been truncated. Thus the native arithmetic regression develops somewhere
between update 200 and the completed epoch, despite overall duplex quality
improving. Hexadecimal conversion is already unreliable at step 200. Native
pilot audio also hears `Zar Rene`, and native pilot transcript expands RNA
correctly. These controls separate a specific training-duration regression
from pre-existing numerical and audio-perception weaknesses; they do not
establish the mechanism of every failure.

## Follow-up interpretation

Keep the response-only gain fix: the matched pilots and completed epoch support
it, and the full run improves five factual probes over four in the pilot.
Longer training does not cause wholesale collapse here, but it can regress
individual skills; twelve simple native checks would have missed the triangle
regression. Next experiments should curate mathematically verified targets and
retain explicit calculation steps, test a lower learning rate with decay in
place of constant 2e-4, and select checkpoints using both native and audio-only
regression tests while still completing the requested epoch. Separately,
recalibrate START supervision on ongoing speech: four premature starts and
three fragmented responses remain. These are proposals, not additional runs
already performed, and numerical label noise is not proved to cause 25.5.

All 13 existing local checks pass in the monitoring environment. The resume
sampler coverage and optimizer-step checks are saved in `sampler-coverage.json`.
