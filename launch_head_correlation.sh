#!/usr/bin/env bash
# Run head_correlation_probe.py sharded across an interactive node's GPUs.
#
# Reads the chains and per-step DINO unions an intervene_probe `prepare` already
# built, so nothing is regenerated. One shard per GPU, ~50 MB of output in total.
#
#   bash launch_head_correlation.sh --gpus 8 --out-dir DIR --cases-dir PROBE_DIR
#   python head_correlation_probe.py --stage report --out-dir DIR
#
# Resuming: re-run the identical command; a shard whose scan/shardNN.npz exists is
# skipped (--overwrite to redo).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

GPUS=8
OUT_DIR=""
CASES_DIR=""
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)       GPUS="$2";      shift 2 ;;
        --out-dir)    OUT_DIR="$2";   shift 2 ;;
        --cases-dir)  CASES_DIR="$2"; shift 2 ;;
        *)            EXTRA+=("$1");  shift   ;;
    esac
done

[[ -n "$OUT_DIR" ]] || { echo "--out-dir is required" >&2; exit 2; }
[[ -n "$CASES_DIR" ]] || CASES_DIR="$OUT_DIR"
if [[ ! -d "$CASES_DIR/cases" ]]; then
    echo "no $CASES_DIR/cases -- point --cases-dir at an intervene_probe out-dir" >&2
    exit 2
fi

CONDA_ENV=${CONDA_ENV:-saliency_r1_qwen3_vllm}
set +u
source "/home/uberger/scratch/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u

export HF_HOME=${HF_HOME:-/home/uberger/scratch/cache/hf_cache}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false

mkdir -p "$OUT_DIR/logs"
echo "=========================================================================="
echo "Out dir   : $OUT_DIR"
echo "Cases     : $CASES_DIR"
echo "Shards    : $GPUS"
echo "Extra     : ${EXTRA[*]:-(none)}"
echo "=========================================================================="

pids=()
for ((i = 0; i < GPUS; i++)); do
    CUDA_VISIBLE_DEVICES="$i" python head_correlation_probe.py \
        --stage scan --shard "$i" --num-shards "$GPUS" \
        --out-dir "$OUT_DIR" --cases-dir "$CASES_DIR" --device cuda:0 \
        "${EXTRA[@]+"${EXTRA[@]}"}" \
        >"$OUT_DIR/logs/scan_shard${i}.log" 2>&1 &
    pids+=($!)
    echo "[launch] shard $i -> GPU $i (pid ${pids[-1]})"
done

sleep 5
python head_correlation_probe.py --stage report --out-dir "$OUT_DIR" >/dev/null 2>&1 || true
python "$REPO/intervene_probe.py" --stage monitor --monitor-stage scan \
    --out-dir "$OUT_DIR" &
mon=$!

fail=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "[done] shard $i ok"
        grep -v "Loading weights" "$OUT_DIR/logs/scan_shard${i}.log" | tail -2 | sed 's/^/        /'
    else
        echo "[FAIL] shard $i -- see $OUT_DIR/logs/scan_shard${i}.log" >&2
        tail -20 "$OUT_DIR/logs/scan_shard${i}.log" >&2 || true
        fail=1
    fi
done
kill "$mon" 2>/dev/null || true
python "$REPO/intervene_probe.py" --stage monitor --once --monitor-stage scan \
    --out-dir "$OUT_DIR" || true

[[ $fail -eq 0 ]] || echo "WARNING: a shard failed; re-run to resume." >&2
echo "[next] python head_correlation_probe.py --stage report --out-dir $OUT_DIR"
echo "       it opens with the level by union-size decile; --max-union 0.5 restricts"
echo "       everything after that table to the steps whose union stays localised"
exit "$fail"
