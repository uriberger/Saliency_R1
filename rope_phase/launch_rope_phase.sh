#!/usr/bin/env bash
# E0 sharded across the GPUs of a node you already hold interactively.
#
# For one GPU through the batch queue use submit_rope_phase_job.sh instead; this
# one exists for runs large enough to want 8-way fan-out.  UNTESTED as of the
# recorded results, which were all produced single-GPU by submit_rope_phase_job.sh.
#
#   ROPE_PHASE_DATASET=/path/to/dataset bash launch_rope_phase.sh --gpus 8 \
#       --out-dir $PWD/outputs/e0_8way
#   python rope_phase_probe.py --stage report --out-dir $PWD/outputs/e0_8way
#
# Use a fresh --out-dir per run, and keep --gpus constant within one: the report
# refuses to merge shards scanned with different num_shards, because a 1-shard run
# writes shard00 covering every case and would then be double-counted.
#
# Resuming: re-run the identical command; a shard whose scan/shardNN.npz exists is
# skipped (--overwrite to redo).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

GPUS=8
OUT_DIR=""
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)    GPUS="$2";    shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        *)         EXTRA+=("$1"); shift  ;;
    esac
done

[[ -n "$OUT_DIR" ]] || { echo "--out-dir is required" >&2; exit 2; }
[[ -n "${ROPE_PHASE_DATASET:-}" || "${EXTRA[*]:-}" == *--dataset* ]] || {
    echo "set ROPE_PHASE_DATASET or pass --dataset" >&2; exit 2; }

CONDA_ENV=${CONDA_ENV:-saliency_r1_qwen3_vllm}   # transformers with qwen*_vl + peft
set +u
source "/home/uberger/scratch/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u

export HF_HOME=${HF_HOME:-/home/uberger/scratch/cache/hf_cache}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/scan"

echo "=========================================================================="
echo "Out dir : $OUT_DIR"
echo "Shards  : $GPUS"
echo "Env     : $CONDA_ENV"
echo "Extra   : ${EXTRA[*]:-(none)}"
echo "=========================================================================="

pids=()
for ((i = 0; i < GPUS; i++)); do
    CUDA_VISIBLE_DEVICES="$i" python rope_phase_probe.py \
        --stage scan --shard "$i" --num-shards "$GPUS" \
        --out-dir "$OUT_DIR" --device cuda:0 \
        "${EXTRA[@]+"${EXTRA[@]}"}" \
        >"$OUT_DIR/logs/scan_shard${i}.log" 2>&1 &
    pids+=($!)
    echo "[launch] shard $i -> GPU $i (pid ${pids[-1]})"
done

fail=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "[done] shard $i ok"
        tail -2 "$OUT_DIR/logs/scan_shard${i}.log" | sed 's/^/        /'
    else
        echo "[FAIL] shard $i -- see $OUT_DIR/logs/scan_shard${i}.log" >&2
        tail -20 "$OUT_DIR/logs/scan_shard${i}.log" >&2 || true
        fail=1
    fi
done

if [[ $fail -eq 0 ]]; then
    python rope_phase_probe.py --stage report --out-dir "$OUT_DIR"
else
    echo "WARNING: a shard failed; re-run to resume, then:" >&2
    echo "  python rope_phase_probe.py --stage report --out-dir $OUT_DIR" >&2
fi
exit "$fail"
