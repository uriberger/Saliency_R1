#!/usr/bin/env bash
# Run one GPU stage of selfground_audit.py sharded across the node's GPUs, then merge.
#
#   bash launch_selfground_audit.sh --stage dino --out-dir <dir> [--gpus 8] -- <probe args>
#
# Both GPU stages are embarrassingly parallel over their unit (a distinct (image,
# sentence) for `dino`, a completion for `crosspass`), so a shard is self-contained on one
# GPU and a dead shard costs only its slice -- same split as launch_overlap_probe.sh.
# Anything after `--` is forwarded verbatim, which is where --probe lives.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

STAGE=""
GPUS=8
OUT_DIR=""
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage)   STAGE="$2";   shift 2 ;;
        --gpus)    GPUS="$2";    shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --)        shift; EXTRA+=("$@"); break ;;
        *)         EXTRA+=("$1"); shift ;;
    esac
done

[[ -n "$STAGE"   ]] || { echo "ERROR: --stage {dino,crosspass} is required" >&2; exit 2; }
[[ -n "$OUT_DIR" ]] || { echo "ERROR: --out-dir is required" >&2; exit 2; }

CONDA_ENV=${CONDA_ENV:-saliency_r1_qwen3_vllm}
set +u
source "/home/uberger/scratch/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u

export HF_HOME=${HF_HOME:-/home/uberger/scratch/cache/hf_cache}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export OVERLAP_STEPS_CKPT=${OVERLAP_STEPS_CKPT:-$REPO/checkpoint/steps_classifier/best}
export TOKENIZERS_PARALLELISM=false

mkdir -p "$OUT_DIR"
echo "=========================================================================="
echo "Stage    : $STAGE      shards/GPUs: $GPUS"
echo "Out dir  : $OUT_DIR"
echo "Extra    : ${EXTRA[*]:-(none)}"
echo "=========================================================================="

pids=()
for ((i = 0; i < GPUS; i++)); do
    CUDA_VISIBLE_DEVICES="$i" python selfground_audit.py \
        --stage "$STAGE" --out-dir "$OUT_DIR" \
        --shard "$i" --num-shards "$GPUS" --device cuda:0 \
        "${EXTRA[@]+"${EXTRA[@]}"}" \
        >"$OUT_DIR/${STAGE}_shard$i.log" 2>&1 &
    pids+=($!)
    echo "[launch] $STAGE shard $i -> GPU $i (pid ${pids[-1]})"
done

fail=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then echo "[done] shard $i ok"
    else echo "[FAIL] shard $i -- see $OUT_DIR/${STAGE}_shard$i.log" >&2; fail=1; fi
done

echo "[merge] $STAGE"
python selfground_audit.py --stage "$STAGE" --out-dir "$OUT_DIR" --merge
[[ $fail -eq 0 ]] || echo "WARNING: at least one shard failed; the merge covers the rest." >&2
exit 0
