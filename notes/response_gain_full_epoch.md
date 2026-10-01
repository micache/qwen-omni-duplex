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

Training is currently running. No full-epoch quality result is claimed yet.
The queued evaluation checks a successful training exit, step 3,750 and epoch
1.0 before loading final weights. It then runs the first 12 train / first 12 dev
questions, six fixed excluded spoken facts, clean teacher-forced evaluation of
all 1,000 dev rows, cached/full parity, the same continuous three-turn recording,
and ordinary no-prompt CLI generation. It saves all answers, event traces and
unabridged special-token strings. Those completed outputs, rather than the
earlier pilot, will determine the result.

All 13 existing local checks pass in the monitoring environment. The resume
sampler coverage and optimizer-step checks are saved in `sampler-coverage.json`.
