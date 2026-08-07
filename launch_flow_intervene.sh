#!/usr/bin/env bash
# Run flow_intervene_probe.py sharded across an interactive node's GPUs.
#
# Reads the chains and per-step DINO unions an intervene_probe `prepare` already built,
# so the causal numbers land on exactly the cases flow_correlation_probe.py scored.
#
#   bash launch_flow_intervene.sh --stage selftest --gpus 1 --out-dir DIR --cases-dir C
#   bash launch_flow_intervene.sh --stage run --gpus 8 --out-dir DIR --cases-dir C
#   python flow_intervene_probe.py --stage report --out-dir DIR
#
# SELFTEST GATES THE RUN and is not optional: it checks that alpha=0 reproduces the
# un-hooked forward, that alpha=1 moves it, and that alpha=1 actually raises the union
# share. The third is the one this probe cannot do without -- unlike the direct
# intervention, the thing being actuated (attention) is not the thing being measured
# (traceable mass), so a null with a flat ushare says nothing about grounding.
#
# Resuming: re-run the identical command. Results are append-only JSONL keyed by
# (row_index, cutoff, kind, alpha); anything already written is skipped.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

GPUS=8
STAGE=run
OUT_DIR=""
CASES_DIR=""
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)       GPUS="$2";      shift 2 ;;
        --stage)      STAGE="$2";     shift 2 ;;
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

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/progress"

echo "=========================================================================="
echo "Stage     : $STAGE"
echo "Out dir   : $OUT_DIR"
echo "Cases     : $CASES_DIR"
echo "Shards    : $GPUS"
echo "Extra     : ${EXTRA[*]:-(none)}"
echo "=========================================================================="

if [[ "$STAGE" == "selftest" ]]; then
    CUDA_VISIBLE_DEVICES=0 python flow_intervene_probe.py \
        --stage selftest --out-dir "$OUT_DIR" --cases-dir "$CASES_DIR" \
        --device cuda:0 "${EXTRA[@]+"${EXTRA[@]}"}" 2>&1 \
        | tee "$OUT_DIR/logs/selftest.log"
    exit "${PIPESTATUS[0]}"
fi

if [[ "$STAGE" == "report" ]]; then
    python flow_intervene_probe.py --stage report --out-dir "$OUT_DIR" \
        "${EXTRA[@]+"${EXTRA[@]}"}" | tee "$OUT_DIR/report.txt"
    exit 0
fi

# Drop the previous attempt's heartbeats: a shard writes its first only after the model
# loads, so on a resume the monitor reads the dead run's files, calls them stale and
# exits while the shards it was watching are fine. Resume state is in results/, not here.
rm -f "$OUT_DIR"/progress/*.json

pids=()
for ((i = 0; i < GPUS; i++)); do
    CUDA_VISIBLE_DEVICES="$i" python flow_intervene_probe.py \
        --stage run --shard "$i" --num-shards "$GPUS" \
        --out-dir "$OUT_DIR" --cases-dir "$CASES_DIR" --device cuda:0 \
        "${EXTRA[@]+"${EXTRA[@]}"}" \
        >"$OUT_DIR/logs/run_shard${i}.log" 2>&1 &
    pids+=($!)
    echo "[launch] shard $i -> GPU $i (pid ${pids[-1]})"
done

sleep 5
python "$REPO/intervene_probe.py" --stage monitor --monitor-stage run \
    --out-dir "$OUT_DIR" &
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
    python flow_intervene_probe.py --stage report --out-dir "$OUT_DIR" \
        | tee "$OUT_DIR/report.txt" || true
else
    echo "WARNING: a shard failed; re-run the identical command to resume." >&2
    exit 1
fi
