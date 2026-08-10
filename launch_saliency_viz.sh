#!/usr/bin/env bash
# Draw the four saliency maps for N samples, sharded across an interactive node's GPUs.
#
#   bash launch_saliency_viz.sh --gpus 8 --out-dir outputs/saliency_viz/run1
#
# Defaults: the coldstart SFT model, 20 samples from cold_data/grpo_sets/val_natural,
# maps direct / rollout_mean / rollout_wnorm / grad. Anything unrecognised is forwarded
# verbatim to saliency_viz.py, so e.g. `--n-samples 40 --cmap inferno` just works.
#
# Three phases, in order:
#   selftest  CPU, seconds. Gates the pixel->token regrouping the grad map depends on.
#   scan      one shard per GPU, each self-contained (8B VLM + the FLAN-T5 step
#             classifier). Resumable: a sample whose maps.npz exists is skipped.
#   render    single CPU process. Re-run it alone to redraw with different
#             --norm/--cmap/--overlay-alpha without touching a GPU:
#               python saliency_viz.py --stage render --out-dir DIR --norm rank
#
# Open <out-dir>/index.html for everything on one page.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

GPUS=8
OUT_DIR=""
N_SAMPLES=20
SKIP_SELFTEST=0
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)           GPUS="$2";      shift 2 ;;
        --out-dir)        OUT_DIR="$2";   shift 2 ;;
        --n-samples)      N_SAMPLES="$2"; shift 2 ;;
        --skip-selftest)  SKIP_SELFTEST=1; shift ;;
        *)                EXTRA+=("$1");  shift   ;;
    esac
done

if [[ -z "$OUT_DIR" ]]; then
    OUT_DIR="$REPO/outputs/saliency_viz/$(date +%Y%m%d-%H%M%S)"
    echo "[launch] no --out-dir given; using $OUT_DIR"
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
echo "Samples   : $N_SAMPLES  across $GPUS GPU(s)"
echo "Extra     : ${EXTRA[*]:-(none)}"
echo "Resume    : re-run this exact command; finished samples are skipped"
echo "=========================================================================="

if [[ $SKIP_SELFTEST -eq 0 ]]; then
    echo "##### selftest (pixel -> token regrouping)"
    CUDA_VISIBLE_DEVICES="" python saliency_viz.py --stage selftest \
        "${EXTRA[@]+"${EXTRA[@]}"}"
fi

echo
echo "##### scan"
pids=()
for ((i = 0; i < GPUS; i++)); do
    CUDA_VISIBLE_DEVICES="$i" python saliency_viz.py \
        --stage scan --shard "$i" --num-shards "$GPUS" \
        --n-samples "$N_SAMPLES" --out-dir "$OUT_DIR" --device cuda:0 \
        "${EXTRA[@]+"${EXTRA[@]}"}" \
        >"$OUT_DIR/logs/scan_shard${i}.log" 2>&1 &
    pids+=($!)
    echo "[launch] shard $i -> GPU $i (pid ${pids[-1]}, log $OUT_DIR/logs/scan_shard${i}.log)"
done

fail=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "[done] shard $i ok"
        grep -a "^\[scan\]" "$OUT_DIR/logs/scan_shard${i}.log" | tail -3 | sed 's/^/        /'
    else
        echo "[FAIL] shard $i -- see $OUT_DIR/logs/scan_shard${i}.log" >&2
        tail -20 "$OUT_DIR/logs/scan_shard${i}.log" >&2 || true
        fail=1
    fi
done

echo
echo "##### render"
python saliency_viz.py --stage render --out-dir "$OUT_DIR" \
    "${EXTRA[@]+"${EXTRA[@]}"}" 2>&1 | tail -25

echo
echo "[next] open $OUT_DIR/index.html"
echo "       per-sample dirs are under $OUT_DIR/samples/"
if [[ $fail -ne 0 ]]; then
    echo "WARNING: at least one scan shard failed; re-run the same command to resume." >&2
fi
exit "$fail"
