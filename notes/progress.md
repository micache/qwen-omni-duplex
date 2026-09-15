# Progress

## Session 01 — scaffold and scope contract

Status: scaffold complete; pytest invocation is blocked by the uninstalled local
`pytest` dependency.

Exists now:

- Plain README, repository instructions, Apache-2.0 license, requirements, and
  three YAML configuration sketches.
- Minimal package boundaries for model, dataset, timeline, streaming, and
  metrics work.
- Placeholder CLIs for training, generation, preparation, inspection, probing,
  and two benchmark adaptations.
- Contract tests for the fixed timeline, model scope, and benchmark labeling.
- Design, compatibility, progress, and experiment notes.

Validation commands run:

```bash
git status --short --branch
python -m compileall -q duplex benchmarks scripts train.py generate.py tests
pytest -q
python3 --version
python3 -m compileall -q duplex benchmarks scripts train.py generate.py tests
python3 -m pytest -q
python3 -c 'from duplex import ControlEvent, TimelineSpec; from duplex import model; from duplex.metrics import OFFICIAL_SPEECH_OUTPUT_SCORE; assert TimelineSpec().frames_per_chunk == 50; assert [e.value for e in ControlEvent] == ["IDLE", "START", "STOP"]; assert model.MODEL_ID == "Qwen/Qwen2.5-Omni-3B"; assert OFFICIAL_SPEECH_OUTPUT_SCORE is False'
```

The workspace has an empty/nonfunctional `.git` directory, so Git reported
`fatal: not a git repository` and no status was available. The `python` and
`pytest` executables are absent. Python 3.12.3 is available as `python3`:
compilation and the direct import/contract smoke check passed, while
`python3 -m pytest -q` reported `No module named pytest`. No dependency, model,
or dataset was downloaded.

Next session: probe the installed/upstream Qwen2.5-Omni interfaces and record
the Thinker input/output and module boundaries before designing model code.

## Session 02 — control vocabulary and causal timeline

Status: implementation complete; pytest remains unavailable in the current
environment.

Implemented:

- Explicit `IDLE`, `START`, `TEXT`, `STOP`, and `PADDING` event kinds.
- `ControlTokenIds` extraction from mapping-style and attribute-style Qwen
  configs, with uniqueness, non-negativity, and Thinker-vocabulary checks.
- Separate target-event and causal-input representations.
- Inactive/active grammar with sample/frame diagnostics and non-terminating
  `IDLE` inside active responses.
- One-event causal shift with Thinker BOS at the first real position.
- Aligned text/control streams, labels, target event types, attention, optional
  frame times, and `-100` padding labels with all padded input masks disabled.
- Focused pytest coverage for config validation, leakage prevention, valid and
  invalid transitions, padding, timing, and stream length equality.

Commands run:

```bash
PYTHONPYCACHEPREFIX=/tmp/qwen-omni-duplex-pyc python3 -m compileall -q duplex tests
PYTHONDONTWRITEBYTECODE=1 python3 -c 'from duplex import EventKind, TimelineSpec; from duplex.timeline import ControlTokenIds, TargetEvent, TargetEventSequence, encode_causal_timeline; config={"talker_config":{"tts_text_pad_token_id":10,"tts_text_start_token_id":11,"tts_text_end_token_id":12},"thinker_config":{"text_config":{"vocab_size":100}}}; ids=ControlTokenIds.from_qwen_config(config); targets=TargetEventSequence([TargetEvent(EventKind.START),TargetEvent(EventKind.TEXT,40),TargetEvent(EventKind.IDLE),TargetEvent(EventKind.STOP)],sample_id="import-check"); encoded=encode_causal_timeline(targets,control_tokens=ids,thinker_bos_token_id=1,pad_to_length=6); assert encoded.labels == [11,40,10,12,-100,-100]; assert encoded.text_ids == [1,0,40,0,0,0]; assert encoded.control_ids == [0,11,0,10,0,0]; assert not any(encoded.text_mask[-2:]); assert not any(encoded.control_mask[-2:])'
python3 -m pytest -q
```

Compilation and the recorded import/encoding command passed. A dependency-free
inline Python contract exercise also passed for config objects, all required
state errors, causal shifting, active-response `IDLE`, padding, timing, and
equal stream lengths. The pytest command reported `No module named pytest`;
`python3 -m pip` is also unavailable, so no dependency was installed. No model
or dataset was downloaded.

Next session: run the suite in an environment with the declared requirements,
then probe the real Qwen2.5-Omni config and Thinker boundaries without loading
model weights unless that session explicitly authorizes it.

## Session 03 — DailyTalkContiguous reader and window metadata

Status: implementation and compile checks complete; pytest is blocked by the
same dependency/network limitation as earlier sessions.

Implemented:

- A strict JSONL manifest reader with relative-path, duplicate-conversation,
  positive-duration, resolved-path, WAV/sidecar existence, stereo-channel,
  source-rate, and header-duration validation.
- The documented channel contract: left/channel 0 is assistant/model and
  right/channel 1 is user. The user channel is the only waveform intended for
  the future Thinker; assistant audio is a reference path by default and is
  retained as a waveform only through an explicit validation option.
- One clearly named 16 kHz resampling helper. No token targets, frame targets,
  augmentation, or model tensors were added.
- Isolated sidecar parsing for the verified
  `[word, [start, end], speaker_label]` schema, including finite, bounded,
  ordered timestamps and explicit speaker-label interpretation.
- Stable SHA-256 assignment of whole conversation IDs to 90/5/5
  train/validation/test splits before window creation, with duplicate-ID
  rejection to prevent conversation leakage.
- Metadata-only fixed two-second random windows and windows centered on
  available user-to-assistant or assistant-to-user word boundaries.
- An opt-in preparation CLI that validates local data and writes a compact
  split index. Hugging Face snapshot download is lazy and runs only with the
  explicit `--download` flag. The inspection CLI prints channel, word,
  boundary, and proposed-window summaries for one local sample.
- Temporary-directory synthetic WAV/JSON/JSONL tests for valid loading,
  channel order/count, sample rate and duration, unsafe paths, missing
  sidecars, timestamp order, public and explicitly mapped speaker labels,
  deterministic/leak-free splits, and boundary-centered windows.

Schema finding and limitation:

The public dataset page and a public `data_stereo/100.json` sample confirm the
alignment tuple schema. Kyutai's public `moshi-finetune/annotate.py` reads only
channel 0 and emits only `SPEAKER_MAIN`; it does not establish a label for
right-channel/user words. Therefore the default parser maps
`SPEAKER_MAIN -> assistant`, returns an empty user-word tuple, and rejects
unknown labels. A second speaker can be parsed only when the caller supplies a
separately verified explicit mapping. Consequently real public samples can
propose random windows, but cross-speaker boundary windows require additional
verified user annotations; no label or timestamp fields were invented.

Validation commands run:

```bash
PYTHONPYCACHEPREFIX=/tmp/full-duplex-pyc python3 -m compileall -q duplex benchmarks scripts train.py generate.py tests
git diff --check
PYTHONDONTWRITEBYTECODE=1 python3 scripts/prepare_dailytalk.py --help
PYTHONDONTWRITEBYTECODE=1 python3 scripts/inspect_sample.py --help
PYTHONDONTWRITEBYTECODE=1 python3 -c '<metadata parser/split/window contract exercise>'
PYTHONPATH='<cached NumPy/SciPy>' python3.11 -c '<synthetic WAV reader/resampling contract exercise with a SoundFile-compatible read shim>'
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q
UV_CACHE_DIR=/tmp/full-duplex-uv-cache uv run --no-project --with pytest --with numpy --with soundfile --with scipy pytest -q
```

Compilation, whitespace checks, both CLI help paths, and the dependency-free
metadata contract exercise passed. A synthetic stereo WAV reader/resampling
exercise also passed using cached NumPy/SciPy and a small SoundFile-compatible
read shim; this did not replace the unavailable real dependency. The direct
pytest command reported `No module named pytest`, and the isolated `uv` attempt
made no dataset or model request but could not resolve `pypi.org`; the base
Python environment has none of pytest, NumPy, SoundFile, or SciPy installed. No
model or dataset was downloaded.

Stop boundary: Session 03 ends here. Token/frame targets, synthetic
interruption augmentation, and model code remain unimplemented.

## Session 04 — word/BPE alignment and frame targets

Status: implementation and compile checks complete; the declared `pytest`
dependency is still unavailable in the base Python environment.

Implemented:

- Full-utterance, context-preserving assistant tokenization with special tokens
  disabled, exact cleanup-free decode round trips, tokenizer-offset alignment,
  and a deterministic decoded-prefix fallback for tokenizers without offsets.
- Explicit reconstruction records for inserted inter-word spaces; punctuation,
  explicit whitespace, and Unicode are preserved. Alignment failures report the
  sample, source turn, and source word indices instead of dropping or guessing
  tokens.
- Preservation of original sidecar row indices on `WordSpan`, stable derived
  assistant turn IDs, token character/word/time provenance, and per-frame source
  word and turn inspection fields.
- Fixed 50-frame construction from the cropped right/user waveform, with user
  activity, absolute frame-start time, one target event per frame, and response
  state recorded for inspection.
- Deterministic strictly monotonic lexical scheduling. Colliding BPE pieces can
  expand across free frames inside the annotated assistant turn; insufficient
  spans raise `InvalidTimelineError` with capacity details. `START` and `STOP`
  are immediately adjacent to the first and last lexical events, respectively.
- A crop policy that omits an assistant run cut by either crop boundary, so no
  partial turn creates an illegal target state. Crops without a complete
  assistant run remain all `IDLE`.
- Causal input construction through the existing Session 02
  `encode_causal_timeline` function, with no duplicate shift implementation.
- A `--timeline` inspection mode in `scripts/inspect_sample.py` using an
  explicitly supplied local tokenizer and control IDs. It prints frame, time,
  user activity, target event, decoded token, and state without network access.
- Tokenizer-stub tests for multi-BPE words, punctuation, inserted and explicit
  leading spaces, Unicode, offset and no-offset paths, single-token offset
  shapes, exact round trips and diagnostic failures, frame expansion, invalid
  capacity, crop boundaries, empty assistant spans, monotonic/no-collision
  allocation, waveform cropping, inspection output, and causal leakage. A
  conditional test exercises a real cached Transformers tokenizer when one is
  locally available and otherwise skips without downloading.

Validation commands run:

```bash
PYTHONPYCACHEPREFIX=/tmp/full-duplex-session04-pyc python3 -m compileall -q duplex benchmarks scripts train.py generate.py tests
git diff --check
PYTHONDONTWRITEBYTECODE=1 python3 scripts/inspect_sample.py --help
PYTHONDONTWRITEBYTECODE=1 python3 -c '<Session 04 tokenizer-stub test harness>'
PYTHONDONTWRITEBYTECODE=1 <cached Python 3.11> -c '<cached LlamaTokenizerFast exact Unicode alignment check>'
python3 -m pytest -q
```

Compilation, whitespace validation, CLI help, and all 11 Session 04 stub test
functions passed. The cached real `LlamaTokenizerFast` check tokenized and
exactly decoded `Hello, café 世界!`, retaining and aligning all eight produced
tokens. The direct pytest command reported `No module named pytest`; no package,
model, or dataset was downloaded. No interruption augmentation or Qwen model
code was added.

Stop boundary: Session 04 ends here.

## Local Python environment

The repository now standardizes all Python commands on `.venv/bin/python`
using Python 3.11. `.python-version` records the interpreter series,
`.gitignore` excludes the environment, and `AGENTS.md` requires future sessions
to use the local interpreter without falling back to or mixing system Python.

The environment was created successfully with CPython 3.11.16. Installing
`requirements.txt` was attempted with:

```bash
UV_CACHE_DIR=/tmp/full-duplex-uv-cache uv pip install --python .venv/bin/python -r requirements.txt
```

Dependency resolution could not reach `pypi.org` because DNS/network access is
unavailable. The read-only shared uv cache also lacks `peft`, `soundfile`, and
`pytest`, so it cannot supply a complete offline installation. The venv is
therefore present but intentionally recorded as incomplete; no checks should
silently use another interpreter. Retry the command above when package access
is available, then run `.venv/bin/python -m pytest -q`.

## Session 05 — synthetic interruption and training collator

Status: implementation and local validation complete.

Implemented:

- Caller-seeded synthetic interruption using either `random.Random` or
  `torch.Generator`, with identity-preserving probability-zero and ineligible
  paths.
- Direct assistant-to-user boundary selection, minimum retained active-frame
  enforcement, deterministic legal cuts, assistant suffix truncation across
  targets, causal masks, frame provenance, tokens, and utterance alignments.
- Frame-aligned shifting of the following user waveform, activity, and word
  timestamps to the synthetic cut, plus shifting of later aligned suffix data.
  The synthetic `STOP` is the first frame where the moved user turn is
  observable, and fixed window/array lengths and state grammar are preserved.
- A worker-seeded `DuplexCollator` that accepts explicit windowed records or
  completed timelines, optionally augments them, recreates causal streams,
  pads labels with `-100`, and retains sample metadata.
- Audio preprocessing through only the caller-supplied feature extractor. The
  batch retains `input_features`, the raw `feature_attention_mask`, and
  explicitly named pre-convolution lengths; no audio tower runs and no
  flattened/restored audio length is guessed.
- Tests for Python and Torch seed determinism, probabilities zero/one,
  ineligible and too-short turns, minimum cuts, shifted timing/waveforms,
  aligned truncation, state/length invariants, worker-safe randomness, batch
  padding, ignored labels, preserved valid values, and the processor boundary.

Validation commands run:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q duplex benchmarks scripts train.py generate.py tests
git diff --check
.venv/bin/python -c '<render augmented synthetic timeline table>'
```

The full suite passed with 50 tests. Compilation and whitespace checks passed.
The visual table showed no user activity before the cut and showed `user=yes`,
`target=STOP`, and `state=inactive` on the cut frame. No model or dataset was
loaded, no audio tower or trainer was added, and no network access occurred.

Stop boundary: Session 05 ends here. Audio-length restoration and model
integration remain deferred to `model.py` after the real API probe.

## Session 06 — Qwen Thinker compatibility probe

Status: stopped at the required CUDA preflight; compatibility acceptance did
not run.

Implemented:

- Replaced the probe placeholder with an inference-only, stage-checkpointed
  JSON probe for `Qwen/Qwen2.5-Omni-3B`.
- Added exact environment and installed-source recording, immutable checkpoint
  resolution, direct-Thinker and audio-disabled-parent loading attempts with
  raw missing/unexpected/mismatched keys, and explicit Talker/token-to-wave and
  vision-tower retention checks.
- Added deterministic one- and two-sample 2-second/16 kHz processor checks,
  flattened audio-feature restoration from the model's own output-length
  helper, strict `[50, 2048]` assertions, batch-two Thinker prefill, and a
  cached one-step call with 2-D position IDs.
- Added Thinker-vocabulary/control-ID validation, BF16 and FlashAttention 2
  reporting, and peak allocated/reserved VRAM recording for load, audio encode,
  forward, pruning, and cleanup.

Command run:

```bash
.venv/bin/python scripts/probe_qwen.py --model Qwen/Qwen2.5-Omni-3B --revision main --device cuda:0 --output notes/session06_probe.json
```

The command exited 1 before Hub access. One invocation identified an NVIDIA
GeForce RTX 2060 SUPER with 8,192 MiB, driver 576.88, and compute capability
7.5, after which torch rejected driver API version 12090 as too old for the
installed CUDA 13.0 build. Later invocations reported that the operating system
blocked NVML and exposed no CUDA device; that final raw failure is retained in
`notes/session06_probe.json`. `torch.cuda.is_available()` was false throughout;
no model or config was downloaded, no VRAM stage could be measured, and no
training occurred.

Per the acceptance stop rule, the checkpoint revision and dependency versions
were not pinned, and unexecuted shape/loading/cache assertions are not reported
as results. Session 06 must be rerun from the command above in a compatible CUDA
environment, preferably with at least 24 GB VRAM.

Stop boundary: Session 06 ends here with the hardware/driver mismatch recorded.

### Session 06 follow-up — compatible PyTorch wheel blocked by DNS

The official `torch==2.10.0+cu128` CPython 3.11 Linux wheel was selected because
it satisfies the installed Transformers 5.17 `torch>=2.5` constraint and is
compatible with the observed driver 576.88. A dry-run install using the command
below failed before changing the environment because `download.pytorch.org`
could not be resolved after three retries:

```bash
UV_CACHE_DIR='<dedicated temporary directory>' uv pip install --python .venv/bin/python --dry-run 'torch==2.10.0' --index-url https://download.pytorch.org/whl/cu128
```

No cached CUDA 12.x torch wheel or alternate local installation was available.
The existing torch 2.14.0+cu130 was preserved. The dedicated failed-install
cache was deleted, then `uv cache clean` removed the disposable prior session
cache (26,267 entries across two cleanup passes; about 21.3 MiB allocated).
`uv pip check` still reports all 83 installed packages compatible. Session 06
remains blocked before checkpoint download and was not rerun past CUDA
preflight.

### Session 06 completion — CUDA 12.6 probe accepted

Status: complete. No training was run.

The environment was moved to the official stable torch 2.10.0+cu126 and
torchvision 0.25.0+cu126 pair. The repository-local environment sees the RTX
2060 SUPER and passed both a BF16 kernel check and `uv pip check` (86 packages).
Pillow 12.3.0 was added because the official Omni processor requires an image
backend even for this raw-audio probe. All dedicated uv installation caches
were removed after use.

The final inference-only command was:

```bash
.venv/bin/python scripts/probe_qwen.py --revision f75b40e3da2003cdd6e1829b1f420ca70797c34e
```

It passed every acceptance item. Direct loading of
`Qwen2_5OmniThinkerForConditionalGeneration` from the root checkpoint had no
critical missing, mismatched, or unexpected keys, so constructing the full
model and calling `disable_talker()` is unnecessary. Talker and token-to-wave
were never retained; deleting the unused vision tower preserved the audio and
text modules and was followed by a successful Thinker prefill and cached step.

One and two 2-second/16 kHz clips produced processor feature tensors
`[1,128,30000]` and `[2,128,30000]` with corresponding masks `[1,30000]` and
`[2,30000]`. The model-derived length chain is 200 valid processor positions to
100 post-convolution positions to 50 LLM vectors. Batch-two audio output was
flattened `[100,2048]` and restored with lengths `[50,50]` to two `[50,2048]`
tensors. Two-dimensional position IDs `[2,50]` and cached `[2,1]` were accepted;
the cache advanced from 50 to 51. The 151,936-entry Thinker vocabulary contains
all three `tts_text_pad/start/end` IDs.

Peak allocated/reserved CUDA bytes were 9,446,580,736/9,527,361,536 for load,
9,470,790,144/9,554,624,512 for one batch-two audio call, and
8,154,948,608/8,193,572,864 for the pruned Thinker prefill plus cached step.
Cleanup reduced live allocated/reserved bytes to 9,671,168/29,360,128. The 8 GB
GPU completed through WSL paging, but 24 GB remains preferred.

`requirements.txt` now pins the accepted dependencies and CUDA wheel index,
and all three existing configs pin the accepted checkpoint commit. Full raw
evidence is in `notes/session06_probe.json`; the concise interpretation is in
`notes/compatibility.md`.

Stop boundary: Session 06 ends here. No model wrapper, LoRA/QLoRA integration,
training, generation, dataset download, or benchmark work was started.

## Session 07 — Qwen full-duplex Thinker wrapper

Status: implementation and CPU validation complete; the requested GPU smoke is
blocked at CUDA visibility preflight in the current execution context. No
training or long-running job was started.

Implemented:

- A Qwen-specific wrapper around the direct `Qwen2.5-Omni-3B` Thinker selected
  in Session 06. Checkpoint-derived two-second/25 Hz constants and the three
  Talker-config text control IDs are validated at construction.
- Audio extraction through the existing Thinker `get_audio_features` and audio
  tower length helper, including mask/pre-convolution checks, flattened
  batch-output splitting by model-derived per-sample lengths, trailing padding,
  hidden-width validation, and explicit failure for unexplained lengths.
- Exact additive fusion of independently zero-masked text and control embeddings
  from the original shared token table with aligned audio embeddings. The fused
  sequence uses the original Thinker text model and LM head; no projection,
  gate, classifier, controller, or new token was introduced.
- Optional labels and inference output, full lexical final hidden states,
  `past_key_values`, `use_cache`, and two-dimensional `position_ids` forwarding.
  Inference can request last-position-only logits when the loaded Qwen head is
  the installed position-wise `Linear`; training labels deliberately reject
  that reduced-logit path.
- Direct, unshifted cross-entropy over the labels already causally shifted by
  `timeline.py`. Per-position losses are weighted by text/idle/start/stop group,
  ignored padding contributes no weight, and the result is normalized by the
  sum of applied weights. Outputs include per-group mean losses, target counts,
  prediction counts over supervised positions, and the applied weight sum.
- Tiny CPU mocks and focused tests covering exact fusion/masking, restoration of
  unequal flattened batch segments, manual weighted-loss equality, all-padding
  behavior, group accounting, cache forwarding, inference-only logits, config
  validation, and unexplained-length failures.
- Four opt-in GPU integration tests for text-only base-Thinker parity, one real
  two-second audio forward, batch two, and no-grad peak CUDA memory. Normal
  pytest skips these tests, and their explicit path reads only the pinned local
  checkpoint snapshot.

Validation commands run:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q duplex tests
git diff --check
RUN_QWEN_GPU_TESTS=1 .venv/bin/python -m pytest -q tests/test_model_gpu.py -s
```

The CPU suite passed with 61 tests and four expected opt-in GPU skips.
Compilation and whitespace checks passed. The explicit GPU command exited at
fixture preflight before loading the checkpoint because torch reported
`cuda_available=False` and zero devices; CUDA initialization reported no
accessible NVIDIA driver, while `nvidia-smi`/NVML reported that GPU access was
blocked by the operating system. This is an execution-environment blocker, not
a model-forward failure, and no GPU integration or peak-memory result is
recorded for Session 07.

Stop boundary: Session 07 ends here. No trainer, Talker, generation path, model
or dataset download, or long job was added or run.

## Session 08 — BF16 LoRA training and adapter checkpoints

Status: implementation and bounded synthetic GPU smoke complete. The main
DailyTalk experiment was not launched.

Implemented:

- A single-YAML `train.py` path using a small Transformers `Trainer` subclass
  and the Session 07 model's already-aligned weighted loss, with no second token
  shift or label smoothing.
- Independent explicit seeds for Python, NumPy, torch/CUDA, conversation split
  salt, fixed-window sampling, Trainer sampling, workers, and synthetic
  interruption augmentation.
- The primary BF16 LoRA recipe for a 24 GB GPU and a separate, explicitly
  selected 4-bit NF4 QLoRA recipe with BF16 compute for a 16 GB fallback. OOM
  fallback is disabled, and output labels must match `LoRA` or `QLoRA`.
- Full-base freezing before PEFT injection, explicit freezing/pruning of the
  unused vision tower, and freezing of the audio tower plus any Talker,
  token-to-wave/waveform decoder if present. Embeddings and the LM head remain
  frozen.
- Runtime discovery from the loaded pinned model of 36 decoder layers at
  `model.layers.0` through `.35`. LoRA targets are the 252 exact existing paths
  under `self_attn.{q,k,v,o}_proj` and `mlp.{gate,up,down}_proj`; same-named
  audio/vision modules are excluded. Rank starts at 16.
- Gradient accumulation, BF16 autocast, non-reentrant gradient checkpointing,
  evaluation/save cadence, retention, and resume through Trainer. Non-finite
  loss/gradients/logs, empty weighted batches, and trainable parameters outside
  the exact LoRA allowlist fail closed.
- JSONL logging of total and per-event losses, label counts, predicted event
  fractions, learning rate, lexical-token and frame counts, and peak allocated
  and reserved VRAM. TensorBoard receives the same merged metrics when selected.
- Adapter-only checkpoints containing PEFT safetensors/config, the Qwen
  processor/tokenizer, exact base ID/revision, dependency versions, control-ID
  mapping, two-second/25 Hz timeline, task/loss/LoRA metadata, and the complete
  training YAML. Trainer checkpoints additionally retain optimizer, scheduler,
  RNG, and Trainer state; full base weights are rejected.
- Focused CPU tests for exact target discovery, tower exclusion, the trainable
  allowlist, checked-in configuration contracts, and explicit QLoRA selection.

Validation commands run:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q duplex train.py tests
git diff --check
.venv/bin/python train.py --help
.venv/bin/python train.py --config configs/debug.yaml --smoke-test
```

The canonical smoke used BF16 rank-16 LoRA and synthetic two-second audio with
deterministic interruption. It ran two optimizer steps, loaded `checkpoint-2`
into a fresh pinned Thinker, reproduced logits with maximum absolute difference
0.0 (`rtol=atol=1e-3`), then resumed for exactly one further nonzero-learning-
rate update and ended at step 3. All 504 adapter tensors, representing
29,933,568 trainable parameters on 252 text-decoder projections, received a
nonzero gradient during the run. Audio-tower gradient count was zero, and the
optimizer parameter groups and state contained only trainable LoRA parameters.
Peak allocated/reserved CUDA memory was 9,433,796,608/10,104,078,336 bytes on
an RTX 3090 with 24,576 MiB. Checkpoint artifact inspection found no full base
weight file and confirmed all required metadata and processor/tokenizer files.

The normal suite passes with 66 tests and five expected opt-in/cache-dependent
skips. TensorBoard 2.20.0 was added to the local environment and requirements;
no model or dataset was downloaded. The smoke artifacts remain local under the
gitignored `outputs/session08-smoke/` directory.

Stop boundary: Session 08 ends here. No main DailyTalk training, QLoRA smoke,
generation, benchmark, Talker, audio generation, or full-base publication was
run or added.

## Session 09 — tiny DailyTalk overfit gate

OVERFIT_GATE=PASS

Four explicit public DailyTalkContiguous windows were trained for exactly 150
BF16-LoRA optimizer steps. The initial Talker-derived control IDs failed
because frozen Thinker output rows for IDLE and STOP are identical. Switching
to the existing native Thinker PAD/BOS/EOS rows fixed the unlearnable control
mapping without adding a head or tokens. Total and text/START/STOP losses all
fell, teacher-forced and cached/free predictions exactly reproduced every
window's event kinds and lexical order, the interrupted example stopped at
shifted user onset, and a fresh save/reload preserved logits and behavior.
All alignment, causal-shift, mask, audio-length, gradient, optimizer-scope, and
adapter-only assertions passed. No larger-data run was started.

Exact commands, revisions, hashes, window metadata, hardware/VRAM, the failed
run, diagnosis, fix, and compact numeric results are recorded in
`notes/experiments.md`. Local machine-readable outputs are under
`outputs/session09-overfit-native/` and remain gitignored.

Stop boundary: Session 09 ends here.

## Session 10 — explicit full-duplex streaming generation

Status: implementation, scripted validation, and real-checkpoint smoke complete.
The prerequisite remains **OVERFIT_GATE=PASS**. On this new machine the exact
Session 09 four-window run was reproduced for 150 optimizer steps and again
printed `OVERFIT_GATE=PASS`; no larger-data training was started.

Implemented:

- An explicit Thinker loop that prefills system/text context once, encodes each
  fixed two-second mono 16 kHz chunk through the Qwen audio tower exactly once,
  restores and validates approximately 50 features for a full chunk, and feeds
  those features one at a time through the original text decoder with a growing
  KV cache. Hugging Face `generate()` is not used or customized.
- Final-chunk zero padding with an explicit valid-sample mask. Only valid audio
  reaches feature extraction, while every feature from one chunk becomes
  available together at that chunk's end. Trace validation requires
  `available_time_s >= chunk_end_time_s`.
- Additive audio/text/control inputs in which the current step receives only
  the prior predicted event after the initial context boundary. Session 02
  inactive/active grammar masking is applied before greedy or optional sampled
  selection, with both the raw argmax and whether masking changed it retained.
- Hidden IDLE/START/STOP events, exact cleanup-disabled decoding of the full
  lexical-token sequence, stable per-event decoded deltas for incomplete
  Unicode/BPE pieces, word spans derived from lexical token events, and the
  individual contributing token times rather than timestamps spread uniformly
  over the final string.
- Silent two-second tail chunks until a legal STOP or the configured limit,
  retained lexical hidden states, and no Talker invocation. Every event JSONL
  record contains sample/chunk/frame identity, audio and availability times,
  per-step compute time, selected and raw IDs, event type, decoded delta,
  state transition, grammar-mask diagnosis, chunk end, and silent-tail status.
- A generation CLI for strict mono 16 kHz WAV input, pinned base/adapted model
  loading, system/text context, greedy or sampled selection, JSON summary, and
  JSONL trace output.

Scripted tests cover wait/START/text/IDLE/STOP, illegal raw argmaxes in both
inactive and active states, interruption while active, final padding, silent
tail termination, one audio-tower call per chunk, exact KV-cache growth,
chunk-boundary causality, Unicode/subword reconstruction, token-based word
grouping, and timeout without STOP. The full suite passed with 75 tests and five
expected opt-in/cache-dependent skips.

The pinned base checkpoint and DailyTalk snapshot were absent and were
downloaded at revisions `f75b40e3da2003cdd6e1829b1f420ca70797c34e` and
`33e1b501f725a6f4ed4ded95e16cd7f66b9d4bdc`, respectively. Full-dataset
validation remains strict and reported the upstream invalid zero/negative span
at `data_stereo/4.json` alignment 56; the four configured Session 09 samples
loaded and reproduced the gate without changing that validator.

The final real smoke used the freshly reproduced adapter, a two-second silent
mono 16 kHz input, greedy decoding, the default system context, and one allowed
silent chunk on an NVIDIA GeForce RTX 3090 (24,576 MiB, driver 595.84). It made
one audio-tower call per processed chunk, emitted 56 event records, changed 16
illegal raw argmaxes through grammar masking, retained 27 lexical hidden
states, and reached STOP at silent-tail frame 5 (`available_time_s=4.0`). All
56 availability times were at or after their source chunk end. This bounded
smoke establishes execution and causality only; the tiny four-window adapter's
visible text under a new system-prefill/silence context is not a quality result.
The trace and summary are saved locally under the gitignored
`outputs/session10-smoke/adapter-trace.jsonl` and `adapter-summary.json`.

Commands run:

```bash
uv python install 3.11.16
uv venv --python 3.11.16 --clear .venv
UV_CACHE_DIR=/tmp/qwen-duplex-uv-cache uv pip install --python .venv/bin/python --index-strategy unsafe-best-match -r requirements.txt
.venv/bin/python scripts/prepare_dailytalk.py --dataset-root /workspace/data/DailyTalkContiguous-session09 --output outputs/session10-smoke/dailytalk-index.jsonl --download --revision 33e1b501f725a6f4ed4ded95e16cd7f66b9d4bdc
.venv/bin/python train.py --config configs/debug.yaml
.venv/bin/python generate.py --config configs/debug.yaml --audio outputs/session10-smoke/silence-2s.wav --trace outputs/session10-smoke/adapter-trace.jsonl --sample-id session10-real-adapter-smoke --max-silent-chunks 1
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q duplex benchmarks scripts train.py generate.py tests
git diff --check
```

Stop boundary: Session 10 ends here. No Talker, audio generation, Hugging Face
generation customization, benchmark, larger-data training, or production path
was added.

## Session 11 — internal validation metrics and small-data diagnostic

SMALL_DATA_GATE=PASS

Implemented padding-aware internal metrics for text/IDLE/START/STOP label and
prediction counts; control precision, recall, and F1; raw pre-mask grammar
violations; within-response START/STOP boundary error in frames and seconds;
synthetic-interruption STOP recall and latency; teacher-forced lexical loss and
perplexity; and free-streaming no-response/no-STOP rates. Undefined metrics use
`null` with an explicit zero denominator. Eleven handcrafted metric tests were
added.

One weighted BF16-LoRA diagnostic ran for 250 steps on a fixed public
100-conversation subset (80 train, 20 validation). Each 8-second selection span
was four contiguous fixed 2-second/25 Hz chunks, preserving the chunk contract.
The complete 25,000-frame histogram was printed before weight selection and
showed 94.2% IDLE, so no unweighted run was needed. Public-data-derived weights
`text=1.0`, `idle=0.05`, `start=4.0`, `stop=4.0` balanced aggregate weighted
target mass and were used for the single diagnostic.

All validation losses were finite; held-out teacher and free paths emitted
START and STOP; free real-audio predictions were 89.1% IDLE; grammar-masked
streams had zero invalid transitions; and two of ten synthetic-interruption
traces emitted STOP after onset. Twenty validation traces were inspected across
normal and interrupted inputs. High raw grammar violations, no-response and
no-STOP rates, zero teacher-forced lexical argmaxes, and poor text remain
explicit small-data limitations. Exact configs, seeds, IDs/manifest, runtime,
VRAM, denominated metrics, and representative traces are recorded in
`notes/experiments.md` and `notes/session11_subset.yaml`.

The evidenced weights and 8-second/four-chunk selection span were promoted only
to `configs/train_lora.yaml`. No main run, QLoRA run, Talker, audio generation,
benchmark, or production work was performed.

Stop boundary: Session 11 ends here.
