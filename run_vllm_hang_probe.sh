#!/bin/bash
# Bisect the validation hang: run each probe case against a FRESH vLLM server.
#
# One hang wedges the worker -- vllm_serve's loop has no try/except, so a failed
# generate never replies and every later request blocks on the same recv(). Reusing
# a server across cases would therefore report that every variable matters. Each
# case gets its own server, which costs a model load (~2-5 min) per case and is the
# only way the verdicts mean anything.
#
#   bash run_vllm_hang_probe.sh                 # all cases
#   bash run_vllm_hang_probe.sh --case n1_temp0 # just one
#
# Results are appended to outputs/logs/vllm_hang_probe.jsonl.
set -uo pipefail

REPO=${REPO:-/home/uberger/scratch/research/saliency_r1}
SCRIPT_DIR=$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")" && pwd)
CONDA_SH=/home/uberger/scratch/miniconda3/etc/profile.d/conda.sh
CONDA_ENV=saliency_r1_qwen3_vllm
PORT=${PORT:-8123}
TIMEOUT=${TIMEOUT:-300}
MODEL=${MODEL:-$REPO/checkpoint/coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged}
OUT=${OUT:-$REPO/outputs/logs/vllm_hang_probe.jsonl}
LOGDIR=${LOGDIR:-$REPO/outputs/logs/vllm_hang_probe}
ONLY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --case)  ONLY="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --port)  PORT="$2";  shift 2 ;;
        -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

mkdir -p "$LOGDIR" "$(dirname "$OUT")"
source "$CONDA_SH"; set +u; conda activate "$CONDA_ENV"; set -u
source "$REPO/setup_cuda_home.sh" >/dev/null 2>&1 || true
export HF_HOME=${HF_HOME:-/home/uberger/scratch/cache/hf_cache}
export HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1

CASES=$(python "$SCRIPT_DIR/probe_vllm_hang.py" --list | awk '{print $1}')
[[ -n "$ONLY" ]] && CASES="$ONLY"

SERVER_PID=""
cleanup() {
    [[ -n "$SERVER_PID" ]] && { pkill -TERM -P "$SERVER_PID" 2>/dev/null; kill -TERM "$SERVER_PID" 2>/dev/null; }
    pkill -TERM -u "$USER" -f "VLLM::EngineCore" 2>/dev/null
    sleep 2; pkill -KILL -u "$USER" -f "VLLM::EngineCore" 2>/dev/null
}
trap cleanup EXIT INT TERM

echo "=========================================================================="
echo "Model:   $MODEL"
echo "Cases:   $(echo $CASES | tr '\n' ' ')"
echo "Results: $OUT"
echo "=========================================================================="

for case_name in $CASES; do
    echo ""
    echo "--- $case_name : starting a fresh server on port $PORT ---"
    cd "$REPO/trl_repo"
    python -m trl.scripts.vllm_serve --model "$MODEL" --host 127.0.0.1 --port "$PORT" \
        --gpu_memory_utilization 0.90 --dtype bfloat16 --max_model_len 4096 \
        --enable_prefix_caching True > "$LOGDIR/$case_name.server.log" 2>&1 &
    SERVER_PID=$!

    waited=0
    until curl -sf "http://127.0.0.1:$PORT/health/" >/dev/null 2>&1; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "  server died before becoming healthy; see $LOGDIR/$case_name.server.log" >&2
            break
        fi
        sleep 5; waited=$((waited + 5))
        if (( waited > 1200 )); then echo "  server not healthy after 20 min" >&2; break; fi
    done

    if curl -sf "http://127.0.0.1:$PORT/health/" >/dev/null 2>&1; then
        python "$SCRIPT_DIR/probe_vllm_hang.py" --url "http://127.0.0.1:$PORT" \
            --case "$case_name" --model "$MODEL" --timeout "$TIMEOUT" --out "$OUT" \
            --val-dir "$REPO/cold_data/grpo_sets/val_natural"
        echo "  (server log tail)"; tail -3 "$LOGDIR/$case_name.server.log" | cut -c1-140
    fi

    cleanup; SERVER_PID=""; sleep 5
done

echo ""
echo "=========================================================================="
echo "Verdicts:"
[[ -f "$OUT" ]] && python -c "
import json,sys
for line in open('$OUT'):
    r = json.loads(line)
    print(f\"  {r['case']:24s} n_prompts={r['n_prompts']:<3} n={r['n']} temp={r['temperature']}  ->  {r['verdict']}  ({r['detail'][:60]})\")
"
echo "=========================================================================="
