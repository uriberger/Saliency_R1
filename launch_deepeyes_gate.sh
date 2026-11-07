#!/usr/bin/env bash
# Score ONE model on V* and HR-Bench with DeepEyes' own client-side zoom loop.
#
#   bash launch_deepeyes_gate.sh --tag deepeyes-7b --out-dir DIR \
#       --model ChenShawn/DeepEyes-7B [--bench vstar,hrbench] [--tp 1]
#
# This is the `_run.sh` half of the launch_laser_job.sh / launch_laser.sh split:
# it assumes an allocation already exists and does the work. launch_deepeyes_gate_job.sh
# is what submits it, and it is what loops over the three models.
#
# WHAT RUNS. vLLM serves the model; `eval/eval_vstar.py` and `eval/eval_hrbench.py` from
# their repo drive it over the OpenAI API with the crop loop entirely client-side; then
# `judge_result.py` / `judge_result_hrbench.py` grade the saved jsonl. The eval scripts
# are genuinely model-agnostic -- the zoom tool is advertised as PLAIN TEXT in the system
# message, not through the API's `tools=` field, so no chat template has to render a tool
# call and a Qwen3-VL checkpoint is driven by exactly the same code as a Qwen2.5-VL one.
#
# THREE THINGS THAT ARE NOT OPTIONAL.
#
#   * build_deepeyes_eval_data.py must have run first. It patches two missing imports
#     into their eval_hrbench.py (`pandas`, `copy` -- the released file raises NameError
#     before it reads a single row) and converts the cached HR-Bench parquet into the TSV
#     their loader opens. Both are CPU work and belong on cpu_short, not in here.
#   * NVIDIA_API_KEY must be set. The judge is ours, over the NVIDIA gateway, which is
#     what removes their 4 self-hosted Qwen2.5-72B GPUs from the job. With no key the
#     judge leg raises on the first non-trivial answer rather than scoring it wrong --
#     which is the behaviour we want, because a silently-ungraded benchmark reads as a
#     low number and a low number is what this gate is deciding on.
#   * --max-model-len must stay above ~33k. HR-Bench-8k images are resized by THEIR
#     script to at most 16384*28*28 px, i.e. 16,384 visual tokens, and the crop turn adds
#     a second image of up to the same size. A shorter context does not error, it makes
#     vLLM refuse the second turn, and a refused second turn is a benchmark where the
#     zoom tool silently never fires -- the exact failure this gate is measuring.
#
# WHAT THE FIRST LINES SHOULD SAY. The server banner prints the served model name and the
# derived max_model_len; then the first eval prints a tqdm over 191 (V*) or 800 (HR-Bench)
# rows. If V* reports fewer than 191 rows, the snapshot is incomplete and every number
# after it is drawn from a different benchmark than the paper's.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

TAG=""
OUT_DIR=""
MODEL=""
BENCH="vstar,hrbench"
TP=1
PORT=${PORT:-8${RANDOM:0:3}}
WORKERS=8
MAX_MODEL_LEN=49152
GPU_MEM=0.90
DATA_DIR=""
REF_ROOT=${DEEPEYES_REF_ROOT:-/home/uberger/scratch/deepeyes_ref}
SERVE=1
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag)            TAG="$2";            shift 2 ;;
        --out-dir)        OUT_DIR="$2";        shift 2 ;;
        --model)          MODEL="$2";          shift 2 ;;
        --bench)          BENCH="$2";          shift 2 ;;
        --tp)             TP="$2";             shift 2 ;;
        --port)           PORT="$2";           shift 2 ;;
        --workers)        WORKERS="$2";        shift 2 ;;
        --max-model-len)  MAX_MODEL_LEN="$2";  shift 2 ;;
        --gpu-mem)        GPU_MEM="$2";        shift 2 ;;
        --data-dir)       DATA_DIR="$2";       shift 2 ;;
        --ref-root)       REF_ROOT="$2";       shift 2 ;;
        --no-serve)       SERVE=0;             shift   ;;   # point at a server already up on --port
        --dry-run)        DRY_RUN=1;           shift   ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

[[ -n "$TAG"     ]] || { echo "ERROR: --tag is required (it names the result directory)." >&2; exit 2; }
[[ -n "$OUT_DIR" ]] || { echo "ERROR: --out-dir is required." >&2; exit 2; }
[[ -n "$MODEL"   ]] || { echo "ERROR: --model is required." >&2; exit 2; }
[[ "$OUT_DIR" = /* ]] || OUT_DIR="$REPO/$OUT_DIR"
DATA_DIR=${DATA_DIR:-$OUT_DIR/data}
[[ "$DATA_DIR" = /* ]] || DATA_DIR="$REPO/$DATA_DIR"

EVAL_SRC="$REF_ROOT/DeepEyes/eval"
[[ -d "$EVAL_SRC" ]] || { echo "ERROR: $EVAL_SRC not found. Run build_deepeyes_eval_data.py first." >&2; exit 2; }
grep -q "^import pandas as pd" "$EVAL_SRC/eval_hrbench.py" || {
    echo "ERROR: $EVAL_SRC/eval_hrbench.py is unpatched (no 'import pandas as pd')." >&2
    echo "       Run build_deepeyes_eval_data.py; the released file raises NameError." >&2; exit 2; }

MANIFEST="$DATA_DIR/manifest.json"
[[ -f "$MANIFEST" ]] || { echo "ERROR: $MANIFEST missing. Run build_deepeyes_eval_data.py --out-dir $DATA_DIR." >&2; exit 2; }
VSTAR_PATH=$(python3 -c "import json,sys; print(json.load(open('$MANIFEST'))['vstar']['root'])")

CONDA_ENV=${CONDA_ENV:-laser}
set +u
source "/home/uberger/scratch/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u

export HF_HOME=${HF_HOME:-/home/uberger/scratch/cache/hf_cache}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Our judge, not their self-hosted Qwen2.5-72B. See trl/rewards/openai_rewards.py for
# where these defaults come from; they are the same endpoint the trainer grades on, so a
# number here is comparable with a reward there.
JUDGE_URL=${OPENAI_BASE_URL:-https://inference-api.nvidia.com}
JUDGE_MODEL=${JUDGE_MODEL:-azure/openai/gpt-4o-mini}
JUDGE_KEY=${NVIDIA_API_KEY:-${OPENAI_API_KEY:-}}

SERVED_NAME="deepeyes_gate_model"
API_URL="http://127.0.0.1:${PORT}/v1"
RESULT_DIR="$OUT_DIR/results"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$RESULT_DIR" "$LOG_DIR"

echo "=========================================================================="
echo "Tag          : $TAG"
echo "Model        : $MODEL"
echo "Bench        : $BENCH"
echo "Serve        : $([[ $SERVE -eq 1 ]] && echo "vLLM tp=$TP port=$PORT len=$MAX_MODEL_LEN mem=$GPU_MEM" || echo "(external, port $PORT)")"
echo "Judge        : $JUDGE_MODEL @ $JUDGE_URL   key $([[ -n "$JUDGE_KEY" ]] && echo '(set)' || echo 'MISSING')"
echo "V* path      : $VSTAR_PATH"
echo "HR-Bench dir : $DATA_DIR"
echo "Results      : $RESULT_DIR/$TAG"
echo "=========================================================================="

if [[ -z "$JUDGE_KEY" ]]; then
    echo "ERROR: no NVIDIA_API_KEY / OPENAI_API_KEY. The judge leg cannot grade and the" >&2
    echo "       benchmark would read low rather than fail. Refusing to start." >&2
    exit 2
fi
[[ $DRY_RUN -eq 1 ]] && { echo "[dry-run] not starting anything."; exit 0; }

SERVER_PID=""
cleanup() { [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT

if [[ $SERVE -eq 1 ]]; then
    echo "[serve] starting vLLM on port $PORT ..."
    vllm serve "$MODEL" \
        --served-model-name "$SERVED_NAME" \
        --port "$PORT" \
        --tensor-parallel-size "$TP" \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_MEM" \
        --limit-mm-per-prompt '{"image": 12}' \
        --trust-remote-code \
        > "$LOG_DIR/vllm.$TAG.log" 2>&1 &
    SERVER_PID=$!

    # Model load plus engine init plus CUDA-graph capture. 20 min is generous for a 7B
    # and still short enough that a server that is never coming up does not eat the
    # whole allocation in silence.
    for i in $(seq 1 240); do
        if curl -sf "$API_URL/models" >/dev/null 2>&1; then
            echo "[serve] up after ~$((i * 5))s"; break
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "ERROR: vLLM died during startup; tail of $LOG_DIR/vllm.$TAG.log:" >&2
            tail -40 "$LOG_DIR/vllm.$TAG.log" >&2; exit 1
        fi
        sleep 5
    done
    curl -sf "$API_URL/models" >/dev/null || {
        echo "ERROR: vLLM did not answer /models within 20 min." >&2
        tail -40 "$LOG_DIR/vllm.$TAG.log" >&2; exit 1; }
fi

run_bench() {
    local name="$1" evalscript="$2" judgescript="$3" patharg="$4" pathval="$5"
    echo "--------------------------------------------------------------------------"
    echo "[$name] generating ..."
    python "$EVAL_SRC/$evalscript" \
        --model_name "$TAG" \
        --api_key EMPTY \
        --api_url "$API_URL" \
        --eval_model_name "$SERVED_NAME" \
        "$patharg" "$pathval" \
        --save_path "$RESULT_DIR" \
        --num_workers "$WORKERS" 2>&1 | tee "$LOG_DIR/$name.$TAG.gen.log"

    echo "[$name] judging ..."
    python "$EVAL_SRC/$judgescript" \
        --model_name "$TAG" \
        --api_key "$JUDGE_KEY" \
        --api_url "$JUDGE_URL" \
        --eval_model_name "$JUDGE_MODEL" \
        "$patharg" "$pathval" \
        --save_path "$RESULT_DIR" \
        --num_workers "$WORKERS" 2>&1 | tee "$LOG_DIR/$name.$TAG.judge.log"

    # judge_result*.py both write final_acc.json into <save_path>/<model_name>/ and
    # overwrite each other between benchmarks, so bank it under a per-benchmark name
    # before the next leg runs.
    if [[ -f "$RESULT_DIR/$TAG/final_acc.json" ]]; then
        cp "$RESULT_DIR/$TAG/final_acc.json" "$RESULT_DIR/$TAG/final_acc.$name.json"
        echo "[$name] banked -> $RESULT_DIR/$TAG/final_acc.$name.json"
    else
        echo "WARNING: [$name] produced no final_acc.json" >&2
    fi
}

IFS=',' read -r -a BENCHES <<< "$BENCH"
for b in "${BENCHES[@]}"; do
    case "$b" in
        vstar)   run_bench vstar   eval_vstar.py   judge_result.py         --vstar_bench_path "$VSTAR_PATH" ;;
        hrbench) run_bench hrbench eval_hrbench.py judge_result_hrbench.py --hrbench_path     "$DATA_DIR"   ;;
        *) echo "ERROR: --bench $b is not a benchmark (vstar|hrbench)." >&2; exit 2 ;;
    esac
done

echo "=========================================================================="
echo "Done: $TAG"
for f in "$RESULT_DIR/$TAG"/final_acc.*.json; do
    [[ -f "$f" ]] || continue
    echo "--- $(basename "$f")"
    python3 -c "
import json,sys
d=json.load(open('$f'))
for k,v in d.items():
    if isinstance(v,(int,float)): print(f'    {k:28s} {v:.2f}')
    elif isinstance(v,dict) and all(isinstance(x,(int,float)) for x in v.values()):
        print(f'    {k:28s} ' + ', '.join(f'{a}={b:.2f}' for a,b in v.items()))
"
done
echo "=========================================================================="
