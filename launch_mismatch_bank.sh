#!/usr/bin/env bash
# Build the donor bank for the mismatched-box control (--mismatch_bank), end to end.
#
# Assumes it is already ON a node with GPUs, like launch_overlap_probe.sh; the
# `_job.sh` half submits this to SLURM. Runs all four phases of
# build_mismatch_bank.py: plan (CPU), one generate+ground shard per visible GPU,
# merge, verify.
#
#   bash launch_mismatch_bank.sh [--out-dir <dir>] [--gpus 8] \
#       [--n-donors 256] [--n-generations 64] [--smoke]
#
# Anything else is forwarded verbatim to build_mismatch_bank.py (--dataset,
# --index-dataset, --model, --box-threshold, --gen-batch ...).
#
# Runtime reference: the cost is 256 x 64 = 16,384 completions of up to 1024 tokens
# from an 8B model, plus one Grounding-DINO pass over their observe steps (~50k at the
# cold start's median of 3 per chain). On 8 cards that is roughly 1-2 hours. The plan
# phase reads and hashes every image in the corpus once and takes ~2 minutes on CPU.
#
# --smoke does 4 donors x 4 generations on one GPU: it exercises the model load,
# generation, the FLAN-T5 segmentation, DINO, the merge and the verify in a few minutes,
# which is the whole pipeline at 1/1000 of the cost.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

GPUS=8
N_DONORS=256
N_GEN=64
OUT_DIR=""
SMOKE=0
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)           GPUS="$2";      shift 2 ;;
        --n-donors)       N_DONORS="$2";  shift 2 ;;
        --n-generations)  N_GEN="$2";     shift 2 ;;
        --out-dir)        OUT_DIR="$2";   shift 2 ;;
        --smoke)          SMOKE=1;        shift   ;;
        *)                EXTRA+=("$1");  shift   ;;
    esac
done

if [[ $SMOKE -eq 1 ]]; then
    GPUS=1
    N_DONORS=4
    N_GEN=4
    OUT_DIR=${OUT_DIR:-$REPO/outputs/mismatch_bank/smoke-$(date +%Y%m%d-%H%M%S)}
fi
OUT_DIR=${OUT_DIR:-$REPO/outputs/mismatch_bank/$(date +%Y%m%d-%H%M%S)}

CONDA_ENV=${CONDA_ENV:-saliency_r1_qwen3_vllm}
CONDA_ROOT=${CONDA_ROOT:-/home/uberger/scratch/miniconda3}
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
echo "Bank out dir  : $OUT_DIR"
echo "Donors        : $N_DONORS x $N_GEN chains   shards/GPUs: $GPUS"
echo "Extra args    : ${EXTRA[*]:-(none)}"
echo "=========================================================================="

# ── 1. plan: hash every image, choose the donor rows ────────────────────────
# CPU only, and deliberately not per shard: every shard reads the SAME donor list out
# of bank_plan.json, so which rows are donors cannot depend on how the work was split.
python build_mismatch_bank.py --plan \
    --out-dir "$OUT_DIR" --n-donors "$N_DONORS" --n-generations "$N_GEN" \
    "${EXTRA[@]+"${EXTRA[@]}"}"

# ── 2. generate + ground, one shard per GPU ─────────────────────────────────
# Self-contained on one card, like the probe's shards: the 8B VLM, Grounding-DINO and
# the FLAN-T5 classifier, no inter-process communication, so a dead shard costs only
# its slice of the donor rows and the merge says which ones are missing.
pids=()
for ((i = 0; i < GPUS; i++)); do
    CUDA_VISIBLE_DEVICES="$i" python build_mismatch_bank.py \
        --shard "$i" --num-shards "$GPUS" \
        --out-dir "$OUT_DIR" --n-generations "$N_GEN" \
        --device cuda:0 --steps-device cuda:0 \
        "${EXTRA[@]+"${EXTRA[@]}"}" \
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

# ── 3+4. merge and verify ───────────────────────────────────────────────────
echo "[merge] shards -> bank.json"
python build_mismatch_bank.py --merge --out-dir "$OUT_DIR" "${EXTRA[@]+"${EXTRA[@]}"}"
echo "[verify]"
python build_mismatch_bank.py --verify --out-dir "$OUT_DIR" "${EXTRA[@]+"${EXTRA[@]}"}"

echo
echo "Bank: $OUT_DIR/bank.json"
echo "Train against it with:  --mismatch_bank $OUT_DIR/bank.json"
[[ $fail -eq 0 ]] || echo "WARNING: at least one shard failed; the bank is missing its donor rows." >&2
exit 0
