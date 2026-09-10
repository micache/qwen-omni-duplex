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
