#!/usr/bin/env bash
# Run laser_probe.py sharded across an interactive node's GPUs.
#
#   bash launch_laser.sh --stage selftest --gpus 1 --out-dir DIR --model M
#   bash launch_laser.sh --stage collect  --gpus 8 --out-dir DIR --model M
#   python laser_probe.py --stage report --out-dir DIR
#
# SELFTEST GATES collect AND IS NOT OPTIONAL. The check it exists for is the second one:
# `A[t, j]` -- the array every number in the report is a function of -- is recomputed from
# post-RoPE Q and K rather than read out of `output_attentions`, and the selftest is what
# says the two agree. It also pins C1: `a_bos` is the attention paid by the last PROMPT
# token, so a different completion cannot change it, and if it does then `prompt_len` is
# wrong and every sink set in the run was drawn from the wrong query.
#
# SHARDING IS BY PROMPT, never by rollout. The go/no-go turns on the spread WITHIN a
# group of 8 rollouts on one prompt; splitting a group across shards would leave every
# within-group statistic uncomputable from any single shard.
#
# COST. One prompt is 8 sampled rollouts through the fused kernel plus 8 teacher-forced
# forwards; the forwards keep SDPA and only the reward's own (rows, kv) slice is ever
# materialised. Budget ~6 s per prompt on an H100:
#     selftest  ~3 min on 1 GPU        collect  ~4 min on 8 GPUs (256 prompts)
#     report    instant
#
# Resuming: re-run the identical command. Results are append-only JSONL keyed by prompt,
# and the bulk arrays are flushed in parts, so a killed shard loses at most one part.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

GPUS=8
STAGE=collect
OUT_DIR=""
MODEL=""
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)     GPUS="$2";    shift 2 ;;
        --stage)    STAGE="$2";   shift 2 ;;
        --out-dir)  OUT_DIR="$2"; shift 2 ;;
        --model)    MODEL="$2";   shift 2 ;;
        *)          EXTRA+=("$1"); shift  ;;
    esac
done

[[ -n "$OUT_DIR" ]] || { echo "--out-dir is required" >&2; exit 2; }
if [[ "$STAGE" == "selftest" || "$STAGE" == "collect" ]]; then
    [[ -n "$MODEL" ]] || { echo "--model is required for stage $STAGE" >&2; exit 2; }
fi

CONDA_ENV=${CONDA_ENV:-saliency_r1_qwen3_vllm}
set +u
source "/home/uberger/scratch/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u

export HF_HOME=${HF_HOME:-/home/uberger/scratch/cache/hf_cache}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/progress"

echo "=========================================================================="
echo "Stage     : $STAGE"
echo "Out dir   : $OUT_DIR"
echo "Model     : ${MODEL:-(none)}"
echo "Shards    : $GPUS"
echo "Extra     : ${EXTRA[*]:-(none)}"
echo "=========================================================================="

COMMON=(--out-dir "$OUT_DIR")
[[ -n "$MODEL" ]] && COMMON+=(--model "$MODEL")

if [[ "$STAGE" == "selftest" ]]; then
    # Deliberately one GPU: a check that passed on some shards and not others is not a gate.
    CUDA_VISIBLE_DEVICES=0 python laser_probe.py --stage selftest \
        "${COMMON[@]}" "${EXTRA[@]+"${EXTRA[@]}"}" 2>&1 \
        | tee "$OUT_DIR/logs/selftest.log"
    exit "${PIPESTATUS[0]}"
fi

if [[ "$STAGE" == "report" ]]; then
    python laser_probe.py --stage report --out-dir "$OUT_DIR" \
        "${EXTRA[@]+"${EXTRA[@]}"}" | tee "$OUT_DIR/report.txt"
    exit 0
fi

if [[ ! -f "$OUT_DIR/logs/selftest.log" ]] || \
   ! grep -q "SELFTEST PASS" "$OUT_DIR/logs/selftest.log"; then
    echo "refusing to run: no passing selftest in $OUT_DIR/logs/selftest.log" >&2
    echo "  bash launch_laser.sh --stage selftest --gpus 1 --out-dir $OUT_DIR --model $MODEL" >&2
    exit 2
fi

# Drop the previous attempt's heartbeats: a shard writes its first only after the model
# loads, so on a resume the monitor reads the dead run's files, calls them stale and exits
# while the shards it was watching are fine. Resume state is in the results files.
rm -f "$OUT_DIR"/progress/*.json

pids=()
for ((i = 0; i < GPUS; i++)); do
    CUDA_VISIBLE_DEVICES="$i" python laser_probe.py \
        --stage "$STAGE" --shard "$i" --num-shards "$GPUS" \
        "${COMMON[@]}" "${EXTRA[@]+"${EXTRA[@]}"}" \
        >"$OUT_DIR/logs/${STAGE}_shard${i}.log" 2>&1 &
    pids+=($!)
    echo "[launch] shard $i -> GPU $i (pid ${pids[-1]})"
done

sleep 5
python laser_probe.py --stage monitor --out-dir "$OUT_DIR" &
mon=$!

fail=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "[done] shard $i ok"
        grep -v "Loading weights" "$OUT_DIR/logs/${STAGE}_shard${i}.log" | tail -2 | sed 's/^/        /'
    else
        echo "[FAIL] shard $i -- see $OUT_DIR/logs/${STAGE}_shard${i}.log" >&2
        tail -20 "$OUT_DIR/logs/${STAGE}_shard${i}.log" >&2 || true
        fail=1
    fi
done
kill "$mon" 2>/dev/null || true
wait "$mon" 2>/dev/null || true

if [[ $fail -eq 0 ]]; then
    python laser_probe.py --stage report --out-dir "$OUT_DIR" \
        | tee "$OUT_DIR/report.txt" || true
else
    echo "WARNING: a shard failed; re-run the identical command to resume." >&2
    exit 1
fi
