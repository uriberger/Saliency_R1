#!/usr/bin/env bash
# Run deepeyes_coord_format_probe.py over a node's GPUs, then report.
#
#   bash launch_deepeyes_coord_format.sh --out-dir outputs/deepeyes_coord_format
#   bash launch_deepeyes_coord_format.sh --gpus 4 --out-dir DIR --arms A
#   bash launch_deepeyes_coord_format.sh --out-dir DIR --model mine=checkpoint/...
#
# THE GRID IS (model x arm), AND BOTH ARMS ARE THE MEASUREMENT.
#
# Arm A runs under Qwen3-VL's own processor caps (max_pixels 16.7 MP), which do not
# downscale a V* photo at all. That makes `abs_orig` and `abs_resized` the SAME reading
# and leaves only relative-vs-absolute on the table. Arm B forces max_pixels down to
# 2048*28*28, so the post-resize frame is visibly smaller than the original and the two
# absolute readings separate. Running only arm A answers half the question and looks like
# it answered all of it -- which is why `--arms` defaults to both.
#
# COST. One prefill of a ~3k-token V* photo plus up to --max-new-tokens of decode, at
# batch size 1. ~10 s a row on an A100, 191 rows a cell, four cells: about 25 min on 8
# GPUs, an hour on 2. The request is deliberately small -- the account's GPU cap is what
# this queues behind.
#
# Resuming: re-run the identical command. Each shard's results are append-only JSONL keyed
# by uid and are skipped on a second pass, so a killed job loses at most one row.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

GPUS=8
OUT_DIR=""
ARMS="A,B"
MODELS=()
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)     GPUS="$2";       shift 2 ;;
        --out-dir)  OUT_DIR="$2";    shift 2 ;;
        --arms)     ARMS="$2";       shift 2 ;;
        --model)    MODELS+=("$2");  shift 2 ;;
        *)          EXTRA+=("$1");   shift   ;;
    esac
done

[[ -n "$OUT_DIR" ]] || { echo "--out-dir is required" >&2; exit 2; }

# tag=path. The tag is what appears in every table, so it is given here rather than
# derived from a 60-character checkpoint directory name.
if [[ ${#MODELS[@]} -eq 0 ]]; then
    MODELS=(
        "coldstart=checkpoint/coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged"
        "instruct=Qwen/Qwen3-VL-8B-Instruct"
    )
fi

IFS=',' read -r -a ARM_LIST <<< "$ARMS"

CONDA_ENV=${CONDA_ENV:-saliency_r1_qwen3_vllm}
set +u
source "/home/uberger/scratch/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u

export HF_HOME=${HF_HOME:-/home/uberger/scratch/cache/hf_cache}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/raw"

NCELLS=$(( ${#MODELS[@]} * ${#ARM_LIST[@]} ))
SHARDS=$(( GPUS / NCELLS ))
[[ $SHARDS -lt 1 ]] && SHARDS=1

echo "=========================================================================="
echo "Out dir   : $OUT_DIR"
echo "Models    : ${MODELS[*]}"
echo "Arms      : ${ARM_LIST[*]}"
echo "Cells     : $NCELLS x $SHARDS shards on $GPUS GPUs"
echo "Env       : $CONDA_ENV"
echo "Extra     : ${EXTRA[*]:-(none)}"
echo "=========================================================================="

pids=()
labels=()
gpu=0
for spec in "${MODELS[@]}"; do
    TAG="${spec%%=*}"
    PATH_="${spec#*=}"
    for arm in "${ARM_LIST[@]}"; do
        for ((s = 0; s < SHARDS; s++)); do
            lbl="${TAG}_${arm}_${s}"
            CUDA_VISIBLE_DEVICES="$(( gpu % GPUS ))" python deepeyes_coord_format_probe.py \
                --stage generate --out-dir "$OUT_DIR" \
                --model "$PATH_" --model-tag "$TAG" --arm "$arm" \
                --shard "$s" --num-shards "$SHARDS" \
                ${EXTRA[@]+"${EXTRA[@]}"} \
                >"$OUT_DIR/logs/$lbl.log" 2>&1 &
            pids+=($!)
            labels+=("$lbl")
            echo "[launch] $lbl -> GPU $(( gpu % GPUS )) (pid ${pids[-1]})"
            gpu=$(( gpu + 1 ))
        done
    done
done

fail=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "[done] ${labels[$i]} ok"
        tail -1 "$OUT_DIR/logs/${labels[$i]}.log" | sed 's/^/        /'
    else
        echo "[FAIL] ${labels[$i]} -- see $OUT_DIR/logs/${labels[$i]}.log" >&2
        tail -20 "$OUT_DIR/logs/${labels[$i]}.log" >&2 || true
        fail=1
    fi
done

# The report runs either way: a partial grid still answers the coordinate question, and a
# table built from what finished beats no table at all when the allocation ran out.
python deepeyes_coord_format_probe.py --stage report --out-dir "$OUT_DIR" \
    | tee "$OUT_DIR/report.txt"

if [[ $fail -ne 0 ]]; then
    echo "WARNING: a shard failed; the report above is partial. Re-run to resume." >&2
    exit 1
fi
