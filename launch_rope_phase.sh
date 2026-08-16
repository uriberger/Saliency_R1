#!/usr/bin/env bash
# E0 -- the RoPE phase lock-in test, sharded across an interactive node's GPUs.
#
# Asks whether the positional overlay M-RoPE lays on the image marches across the
# patch grid as the generating token moves away from the image, at the rate the
# config predicts and nothing else does.  See rope_phase_probe.py's docstring.
#
#   bash launch_rope_phase.sh --gpus 8 --out-dir outputs/rope_phase/e0_qwen3
#   python rope_phase_probe.py --stage report --out-dir outputs/rope_phase/e0_qwen3
#
# The differential is the cheap part of the experiment: run it twice, once per
# model.  Qwen2.5-VL's W axis turns 0.0055 rad across a whole image, Qwen3-VL's
# turns 19 rad, so the column march should be absent on one and obvious on the
# other -- same code, same data, a prediction nothing about content can imitate.
#
#   bash launch_rope_phase.sh --gpus 8 --out-dir outputs/rope_phase/e0_qwen25 \
#        --base-model Qwen/Qwen2.5-VL-7B-Instruct
#
# Resuming: re-run the identical command; a shard whose scan/shardNN.npz exists is
# skipped (--overwrite to redo).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

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

CONDA_ENV=${CONDA_ENV:-saliency_r1_qwen3_vllm}
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
