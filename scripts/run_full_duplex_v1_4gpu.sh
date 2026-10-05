#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
run_dir=outputs/full-duplex-bench-v1-response-gain003
for arg in "$@"; do
    if [[ "$arg" == --smoke ]]; then run_dir="$run_dir/smoke"; fi
done
mkdir -p "$run_dir"
.venv/bin/python scripts/run_full_duplex_v1.py "$@" --initialize-only
pids=()
for gpu in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4 .venv/bin/python -u scripts/run_full_duplex_v1.py \
        "$@" --num-shards 4 --shard-index "$gpu" > "$run_dir/gpu-$gpu.log" 2>&1 &
    pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if ((failed)); then echo "A worker failed; inspect $run_dir/gpu-*.log" >&2; exit 1; fi
.venv/bin/python scripts/run_full_duplex_v1.py "$@" --score-only > "$run_dir/aggregate.log" 2>&1
