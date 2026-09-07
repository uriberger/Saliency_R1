#!/usr/bin/env bash
# Run sink_shift_probe.py sharded across an interactive node's GPUs.
#
#   bash launch_sink_shift.sh --stage selftest --gpus 1 --out-dir DIR --model M
#   bash launch_sink_shift.sh --stage survey   --gpus 1 --out-dir DIR --model M
#   bash launch_sink_shift.sh --stage run      --gpus 8 --out-dir DIR --model M
#   python sink_shift_probe.py --stage report --out-dir DIR
#
# SELFTEST GATES THE RUN and is not optional. It checks that alpha=0 reproduces the
# un-hooked greedy generation TOKEN FOR TOKEN, that uninstalling puts the model back, and
# that at alpha>0 the border's share of the picture's attention really falls to (1-alpha)
# of itself. The last one is what the flow-intervention probe learned the hard way: when
# the thing you actuate is not the thing you measure, a null with a flat manipulation
# check says nothing at all.
#
# RUN SURVEY BEFORE THE RUN. It costs about a minute and it prices the whole experiment:
# the edit can only move as much of an attention row as that row spends on the picture,
# which is 0.4-1.4% at the two heads the reward trained.
#
# Resuming: re-run the identical command. Results are append-only JSONL keyed by
# (split, arm, alpha, row_index); anything already written is skipped.
#
# COST. One row is one greedy generation at batch size 1 -- the edit needs batch 1, since
# it locates the picture per prompt. At ~10 s a row, 256 rows x 2 splits x (1 baseline +
# arms x alphas) over 8 GPUs is roughly 20 minutes per (arm, alpha) cell. Start with
# --rows-per-split 128 and two alphas before asking for the whole grid.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

GPUS=8
STAGE=run
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
if [[ "$STAGE" != "report" && "$STAGE" != "monitor" ]]; then
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

# A worktree does not get cold_data/ symlinked in, so the validation sets are addressed
# in the central tree unless the caller says otherwise. Passing --val-sets-dir in EXTRA
# overrides this, because the probe's own argument wins on a repeated flag.
VAL_SETS_DIR=${VAL_SETS_DIR:-}
if [[ -z "$VAL_SETS_DIR" ]]; then
    if [[ -d "$REPO/cold_data/grpo_sets/val_natural" ]]; then
        VAL_SETS_DIR="$REPO/cold_data/grpo_sets"
    else
        VAL_SETS_DIR="/lustre/fs1/portfolios/nvr/projects/nvr_israel_rlop/users/uberger/research/saliency_r1/cold_data/grpo_sets"
    fi
fi
[[ -d "$VAL_SETS_DIR/val_natural" ]] || {
    echo "no val_natural under $VAL_SETS_DIR -- build it with" >&2
    echo "  python build_grpo_sets.py --build-val --out-dir <dir>" >&2
    echo "or set VAL_SETS_DIR." >&2
    exit 2
}

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/progress"

echo "=========================================================================="
echo "Stage     : $STAGE"
echo "Out dir   : $OUT_DIR"
echo "Model     : ${MODEL:-(none)}"
echo "Val sets  : $VAL_SETS_DIR"
echo "Shards    : $GPUS"
echo "Extra     : ${EXTRA[*]:-(none)}"
echo "=========================================================================="

COMMON=(--out-dir "$OUT_DIR" --val-sets-dir "$VAL_SETS_DIR")
[[ -n "$MODEL" ]] && COMMON+=(--model "$MODEL")

if [[ "$STAGE" == "selftest" || "$STAGE" == "survey" ]]; then
    CUDA_VISIBLE_DEVICES=0 python sink_shift_probe.py --stage "$STAGE" \
        "${COMMON[@]}" "${EXTRA[@]+"${EXTRA[@]}"}" 2>&1 \
        | tee "$OUT_DIR/logs/$STAGE.log"
    exit "${PIPESTATUS[0]}"
fi

if [[ "$STAGE" == "report" ]]; then
    python sink_shift_probe.py --stage report --out-dir "$OUT_DIR" \
        "${EXTRA[@]+"${EXTRA[@]}"}" | tee "$OUT_DIR/report.txt"
    exit 0
fi

if [[ ! -f "$OUT_DIR/logs/selftest.log" ]] || \
   ! grep -q "SELFTEST PASS" "$OUT_DIR/logs/selftest.log"; then
    echo "refusing to run: no passing selftest in $OUT_DIR/logs/selftest.log" >&2
    echo "  bash launch_sink_shift.sh --stage selftest --gpus 1 --out-dir $OUT_DIR --model $MODEL" >&2
    exit 2
fi

# Drop the previous attempt's heartbeats: a shard writes its first only after the model
# loads, so on a resume the monitor reads the dead run's files, calls them stale and
# exits while the shards it was watching are fine. Resume state is in the results files.
rm -f "$OUT_DIR"/progress/*.json

pids=()
for ((i = 0; i < GPUS; i++)); do
    CUDA_VISIBLE_DEVICES="$i" python sink_shift_probe.py \
        --stage run --shard "$i" --num-shards "$GPUS" \
        "${COMMON[@]}" "${EXTRA[@]+"${EXTRA[@]}"}" \
        >"$OUT_DIR/logs/run_shard${i}.log" 2>&1 &
    pids+=($!)
    echo "[launch] shard $i -> GPU $i (pid ${pids[-1]})"
done

sleep 5
python sink_shift_probe.py --stage monitor --out-dir "$OUT_DIR" &
mon=$!

fail=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "[done] shard $i ok"
        grep -v "Loading weights" "$OUT_DIR/logs/run_shard${i}.log" | tail -2 | sed 's/^/        /'
    else
        echo "[FAIL] shard $i -- see $OUT_DIR/logs/run_shard${i}.log" >&2
        tail -20 "$OUT_DIR/logs/run_shard${i}.log" >&2 || true
        fail=1
    fi
done
kill "$mon" 2>/dev/null || true
wait "$mon" 2>/dev/null || true

if [[ $fail -eq 0 ]]; then
    python sink_shift_probe.py --stage report --out-dir "$OUT_DIR" \
        | tee "$OUT_DIR/report.txt" || true
else
    echo "WARNING: a shard failed; re-run the identical command to resume." >&2
    exit 1
fi
