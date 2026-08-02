#!/usr/bin/env bash
# Run overlap_probe.py sharded across the node's GPUs, then render the report.
#
# Every shard is self-contained on ONE GPU: the 8B VLM (~17 GB bf16) + Grounding-DINO
# (~1 GB) + the FLAN-T5 step classifier (~0.5 GB). No inter-process communication, so
# a dead shard costs only its slice.
#
#   NVIDIA_API_KEY=... bash launch_overlap_probe.sh [--n-samples 30] [--gpus 8] [--no-judge]
#
# Anything after the recognised flags is forwarded verbatim to overlap_probe.py.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

N_SAMPLES=30
GPUS=8
JUDGE="--judge"
OUT_DIR="$REPO/outputs/overlap_probe/$(date +%Y%m%d-%H%M%S)"
EXTRA=()

SMOKE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --n-samples) N_SAMPLES="$2"; shift 2 ;;
        --gpus)      GPUS="$2";      shift 2 ;;
        --out-dir)   OUT_DIR="$2";   shift 2 ;;
        --no-judge)  JUDGE="";       shift   ;;
        # One GPU, two samples, no judge: exercises model+adapter load, generation,
        # the layer-L hook, DINO and the step classifier before committing 8 GPUs.
        --smoke)     SMOKE=1;        shift   ;;
        *)           EXTRA+=("$1");  shift   ;;
    esac
done

if [[ $SMOKE -eq 1 ]]; then
    N_SAMPLES=2
    GPUS=1
    JUDGE=""
    OUT_DIR="$REPO/outputs/overlap_probe/smoke-$(date +%Y%m%d-%H%M%S)"
fi

CONDA_ENV=${CONDA_ENV:-saliency_r1_qwen3_vllm}
CONDA_ROOT=${CONDA_ROOT:-/lustre/fs12/portfolios/nvr/projects/nvr_israel_rlop/users/uberger/research/miniforge3}
set +u
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u

export HF_HOME=${HF_HOME:-/home/uberger/scratch/cache/hf_cache}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export OVERLAP_STEPS_CKPT=${OVERLAP_STEPS_CKPT:-$REPO/checkpoint/steps_classifier/best}
export TOKENIZERS_PARALLELISM=false

mkdir -p "$OUT_DIR"
echo "=========================================================================="
echo "Probe out dir : $OUT_DIR"
echo "Samples       : $N_SAMPLES   shards/GPUs: $GPUS   judge: ${JUDGE:-off}"
echo "Extra args    : ${EXTRA[*]:-(none)}"
echo "=========================================================================="

pids=()
for ((i = 0; i < GPUS; i++)); do
    CUDA_VISIBLE_DEVICES="$i" python overlap_probe.py \
        --shard "$i" --num-shards "$GPUS" \
        --n-samples "$N_SAMPLES" \
        --device cuda:0 \
        --out-dir "$OUT_DIR" \
        $JUDGE "${EXTRA[@]+"${EXTRA[@]}"}" \
        >"$OUT_DIR/shard$i.log" 2>&1 &
    pids+=($!)
    echo "[launch] shard $i -> GPU $i (pid ${pids[-1]}, log $OUT_DIR/shard$i.log)"
done

fail=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "[done] shard $i ok"
    else
        echo "[FAIL] shard $i -- see $OUT_DIR/shard$i.log" >&2
        fail=1
    fi
done

if [[ $SMOKE -eq 1 ]]; then
    echo "---------------- smoke shard log (tail) ----------------"
    tail -40 "$OUT_DIR/shard0.log"
    echo "--------------------------------------------------------"
fi

echo "[render] merging shards"
python overlap_probe.py --render --out-dir "$OUT_DIR"
echo "Report: $OUT_DIR/probe_report.html"
[[ $fail -eq 0 ]] || echo "WARNING: at least one shard failed; the report covers the shards that finished." >&2
exit 0
