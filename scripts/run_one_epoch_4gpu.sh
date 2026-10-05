#!/usr/bin/env bash
# Run from the repository root after preparing the pinned model and datasets.
set -euo pipefail

output=outputs/instructs2s-response-gain003-one-epoch
if [[ -e "$output/metrics.jsonl" || -e "$output/final" ]]; then
    echo "Fresh-run output already exists: $output; preserve it and resume explicitly." >&2
    exit 1
fi
mkdir -p "$output"
export NCCL_NET=Socket  # This server's inherited gIB plugin does not initialize.
export OMP_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TENSORBOARD_LOGGING_DIR="$output/tensorboard"
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 \
    train.py --config configs/instructs2s_response_gain003_one_epoch_4gpu.yaml \
    > "$output/train.log" 2>&1
