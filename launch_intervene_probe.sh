#!/usr/bin/env bash
# Run intervene_probe.py sharded across an interactive node's GPUs, with a live
# aggregated ETA, and resumable.
#
# Every shard is self-contained on ONE GPU: the 8B VLM (~17 GB bf16), plus (prepare
# only) Grounding-DINO (~1 GB) and the FLAN-T5 step classifier (~0.5 GB). No IPC, so
# a dead shard costs only its slice and re-running the same command picks it up.
#
#   # 1. build the cases once (generation + DINO); ~n-samples/GPUS per shard
#   bash launch_intervene_probe.sh --stage prepare --n-samples 1000 --out-dir DIR
#
#   # 2. gate: the alpha=0 rebuild must reproduce the un-hooked forward
#   bash launch_intervene_probe.sh --stage selftest --out-dir DIR
#
#   # 3a. Stage 0 -- whole-layer intervention, every layer. Bounds the search.
#   bash launch_intervene_probe.sh --stage run --out-dir DIR \
#        --layers 0-35 --head-mode layer
#
#   # 3b. Stage 1 -- every head, on the layers Stage 0 flagged
#   bash launch_intervene_probe.sh --stage run --out-dir DIR \
#        --layers 12,18,22,26 --head-mode each
#
#   # 3c. Stage 2 -- full battery + dose-response on the survivors
#   bash launch_intervene_probe.sh --stage run --out-dir DIR \
#        --layers 22 --head-mode 28,31 \
#        --conditions box,roll,shape,image,perm --alphas 0.25,0.5,0.75,1.0
#
#   # 4. aggregate
#   python intervene_probe.py --stage report --out-dir DIR
#
# Two nodes: run the same command on each with --num-nodes 2 and --node-index 0 / 1,
# pointing at the same --out-dir on shared storage. Shards are numbered globally, so
# the results and heartbeats interleave without collision.
#
# RESUMING: just re-run the identical command. `run` skips every
# (case, layer, head, variant) already in results/shard*.jsonl; `prepare` skips a
# shard whose cases file exists (--overwrite to redo).
#
# Anything after the recognised flags is forwarded verbatim to intervene_probe.py.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

STAGE=run
GPUS=8
NUM_NODES=1
NODE_INDEX=0
N_SAMPLES=1000
OUT_DIR=""
NO_MONITOR=0
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage)       STAGE="$2";      shift 2 ;;
        --gpus)        GPUS="$2";       shift 2 ;;
        --num-nodes)   NUM_NODES="$2";  shift 2 ;;
        --node-index)  NODE_INDEX="$2"; shift 2 ;;
        --n-samples)   N_SAMPLES="$2";  shift 2 ;;
        --out-dir)     OUT_DIR="$2";    shift 2 ;;
        --no-monitor)  NO_MONITOR=1;    shift   ;;
        *)             EXTRA+=("$1");   shift   ;;
    esac
done

if [[ -z "$OUT_DIR" ]]; then
    OUT_DIR="$REPO/outputs/intervene_probe/$(date +%Y%m%d-%H%M%S)"
    echo "[launch] no --out-dir given; using $OUT_DIR"
    echo "[launch] pass the SAME --out-dir to later stages, and to resume."
fi

CONDA_ENV=${CONDA_ENV:-saliency_r1_qwen3_vllm}
set +u
source "/home/uberger/scratch/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u

export HF_HOME=${HF_HOME:-/home/uberger/scratch/cache/hf_cache}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export OVERLAP_STEPS_CKPT=${OVERLAP_STEPS_CKPT:-$REPO/checkpoint/steps_classifier/best}
export TOKENIZERS_PARALLELISM=false

mkdir -p "$OUT_DIR/logs"
TOTAL_SHARDS=$((GPUS * NUM_NODES))

# report and monitor are single-process: no sharding, no GPU.
if [[ "$STAGE" == "report" || "$STAGE" == "monitor" ]]; then
    exec python intervene_probe.py --stage "$STAGE" --out-dir "$OUT_DIR" \
        "${EXTRA[@]+"${EXTRA[@]}"}"
fi

# selftest is a gate, not a grid: one GPU, in the foreground, fail loudly.
if [[ "$STAGE" == "selftest" ]]; then
    CUDA_VISIBLE_DEVICES=0 exec python intervene_probe.py \
        --stage selftest --out-dir "$OUT_DIR" --device cuda:0 \
        --n-samples "$N_SAMPLES" "${EXTRA[@]+"${EXTRA[@]}"}"
fi

echo "=========================================================================="
echo "Stage        : $STAGE"
echo "Out dir      : $OUT_DIR"
echo "Shards       : $TOTAL_SHARDS  (node $NODE_INDEX of $NUM_NODES, $GPUS GPUs here)"
echo "Samples      : $N_SAMPLES"
echo "Extra args   : ${EXTRA[*]:-(none)}"
echo "Resume       : re-run this exact command; finished work is skipped"
echo "=========================================================================="

pids=()
shards=()
for ((i = 0; i < GPUS; i++)); do
    SHARD=$((NODE_INDEX * GPUS + i))
    CUDA_VISIBLE_DEVICES="$i" python intervene_probe.py \
        --stage "$STAGE" \
        --shard "$SHARD" --num-shards "$TOTAL_SHARDS" \
        --n-samples "$N_SAMPLES" \
        --device cuda:0 \
        --out-dir "$OUT_DIR" \
        "${EXTRA[@]+"${EXTRA[@]}"}" \
        >"$OUT_DIR/logs/${STAGE}_shard${SHARD}.log" 2>&1 &
    pids+=($!)
    shards+=("$SHARD")
    echo "[launch] shard $SHARD -> GPU $i (pid ${pids[-1]}, log $OUT_DIR/logs/${STAGE}_shard${SHARD}.log)"
done

# Live aggregated progress in the foreground. It exits when every heartbeat reports
# complete, or when they all go stale (a crash), so it never hangs the shell.
mon_pid=""
if [[ $NO_MONITOR -eq 0 ]]; then
    sleep 5
    python intervene_probe.py --stage monitor --out-dir "$OUT_DIR" &
    mon_pid=$!
fi

fail=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "[done] shard ${shards[$i]} ok"
    else
        echo "[FAIL] shard ${shards[$i]} -- see $OUT_DIR/logs/${STAGE}_shard${shards[$i]}.log" >&2
        tail -20 "$OUT_DIR/logs/${STAGE}_shard${shards[$i]}.log" >&2 || true
        fail=1
    fi
done
[[ -n "$mon_pid" ]] && kill "$mon_pid" 2>/dev/null || true

if [[ $fail -ne 0 ]]; then
    echo "WARNING: at least one shard failed. Re-run the same command to resume; " >&2
    echo "         completed (case, layer, head, variant) units are skipped." >&2
fi
if [[ "$STAGE" == "run" ]]; then
    echo "[next] python intervene_probe.py --stage report --out-dir $OUT_DIR"
fi
exit "$fail"
