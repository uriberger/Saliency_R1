#!/usr/bin/env bash
# Run flow_correlation_probe.py sharded across an interactive node's GPUs.
#
# Reads the chains and per-step DINO unions an intervene_probe `prepare` already built,
# so nothing is regenerated and the numbers are comparable with the direct-map scan
# (head_correlation_probe.py) run on the same cases.
#
# Each map variant gets its own subdirectory under --out-dir and they run one after
# another, all 8 GPUs each -- running them concurrently would put three 8B models on
# every GPU.
#
#   bash launch_flow_correlation.sh --gpus 8 --out-dir DIR --cases-dir PROBE_DIR \
#        --maps rollout_mean,rollout_wnorm,grad
#   python flow_correlation_probe.py --stage report --out-dir DIR/rollout_mean
#
# Resuming: re-run the identical command; a shard whose scan/shardNN.npz exists is
# skipped, and a variant that finished is skipped entirely (--overwrite to redo).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

GPUS=8
OUT_DIR=""
CASES_DIR=""
MAPS="rollout_mean,rollout_wnorm,grad"
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)       GPUS="$2";      shift 2 ;;
        --out-dir)    OUT_DIR="$2";   shift 2 ;;
        --cases-dir)  CASES_DIR="$2"; shift 2 ;;
        --maps)       MAPS="$2";      shift 2 ;;
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

IFS=',' read -r -a MAP_LIST <<< "$MAPS"

echo "=========================================================================="
echo "Out dir   : $OUT_DIR"
echo "Cases     : $CASES_DIR"
echo "Maps      : ${MAP_LIST[*]}"
echo "Shards    : $GPUS"
echo "Extra     : ${EXTRA[*]:-(none)}"
echo "=========================================================================="

overall=0
for MAP in "${MAP_LIST[@]}"; do
    MOUT="$OUT_DIR/$MAP"
    mkdir -p "$MOUT/logs" "$MOUT/progress"
    # Drop the previous attempt's heartbeats. A shard writes its first one only after
    # the model is loaded, ~a minute in, so on a resume the monitor spends that minute
    # reading files from the run that died -- sees them stale, declares "no shard has
    # reported in 10 min", and exits, while the shards it was watching are fine.
    # Resume state lives in scan/shardNN.npz, so nothing is lost by clearing these.
    rm -f "$MOUT"/progress/*.json
    echo
    echo "##### $MAP -> $MOUT"

    pids=()
    for ((i = 0; i < GPUS; i++)); do
        CUDA_VISIBLE_DEVICES="$i" python flow_correlation_probe.py \
            --stage scan --map "$MAP" --shard "$i" --num-shards "$GPUS" \
            --out-dir "$MOUT" --cases-dir "$CASES_DIR" --device cuda:0 \
            "${EXTRA[@]+"${EXTRA[@]}"}" \
            >"$MOUT/logs/scan_shard${i}.log" 2>&1 &
        pids+=($!)
        echo "[launch] $MAP shard $i -> GPU $i (pid ${pids[-1]})"
    done

    sleep 5
    python "$REPO/intervene_probe.py" --stage monitor --monitor-stage scan \
        --out-dir "$MOUT" &
    mon=$!

    fail=0
    for i in "${!pids[@]}"; do
        if wait "${pids[$i]}"; then
            echo "[done] $MAP shard $i ok"
            grep -v "Loading weights" "$MOUT/logs/scan_shard${i}.log" | tail -2 | sed 's/^/        /'
        else
            echo "[FAIL] $MAP shard $i -- see $MOUT/logs/scan_shard${i}.log" >&2
            tail -20 "$MOUT/logs/scan_shard${i}.log" >&2 || true
            fail=1
        fi
    done
    kill "$mon" 2>/dev/null || true
    wait "$mon" 2>/dev/null || true

    if [[ $fail -eq 0 ]]; then
        python flow_correlation_probe.py --stage report --out-dir "$MOUT" \
            | tee "$MOUT/report.txt" || true
    else
        echo "WARNING: a $MAP shard failed; re-run to resume." >&2
        overall=1
    fi
done

echo
echo "[next] reports are in $OUT_DIR/<map>/report.txt; re-render any of them with"
echo "       python flow_correlation_probe.py --stage report --all-columns --out-dir $OUT_DIR/<map>"
echo "       each opens with the level by union-size decile, uncapped; add"
echo "       --max-union 0.5 to restrict every number after that table"
exit "$overall"
