#!/bin/bash
# All-in-one SINGLE-NODE runner for the attention-overlap GRPO training of Qwen3-VL-8B.
#
# Starts THREE co-located components inside ONE allocation, so the DINO reward server,
# the vLLM generation server, and the training processes are tied together (they live or
# die as one job -- solves the "DINO on a separate machine" scheduling problem):
#
#     GPU 0              -> Grounding-DINO reward server   (127.0.0.1:$DINO_PORT)
#     GPU 1              -> vLLM generation server         (127.0.0.1:$VLLM_PORT)
#     GPU 2 .. N-1       -> GRPO training, DeepSpeed ZeRO-3, (N-2) processes
#
# vLLM (server mode) replaces the slow HF generate() path -- this is the main speedup.
# The policy is LoRA; the trainer merges the adapter and pushes weights to the vLLM
# server over NCCL every step (grpo_trainer_qwen3.py:_move_model_to_vllm).
#
# Run it EITHER way:
#   * directly on an interactive N-GPU node:
#       WANDB_API_KEY=... NVIDIA_API_KEY=... bash run_grpo_overlap_colocated.sh --num-gpus 8
#   * or via submit_job (see launch_grpo_qwen3_overlap_job.sh), which just calls this.
#
# Any flag not parsed below is forwarded verbatim to grpo_vlm_qwen3.py via EXTRA_ARGS.
set -euo pipefail

REPO=/home/uberger/scratch/research/saliency_r1
CONDA_SH=/home/uberger/scratch/miniconda3/etc/profile.d/conda.sh
CONDA_ENV=${CONDA_ENV:-saliency_r1_qwen3_vllm}      # cloned env with vLLM 0.11 installed
HF_HOME=${HF_HOME:-/home/uberger/scratch/cache/hf_cache}

# ---------- training defaults ----------
MODEL="$REPO/checkpoint/coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged"
NUM_GPUS=8                       # total GPUs in THIS allocation (1 DINO + 1 vLLM + rest train)
OUTPUT_DIR=""
MAX_COMPLETION_LENGTH=1024
NUM_GENERATIONS=8
GRAD_ACCUM=8
PER_DEVICE_BATCH=1
LEARNING_RATE=1e-5
EXTRA_ARGS=""

# ---------- overlap-reward (swept) defaults ----------
W_OVERLAP=0.2
TOKEN_REDUCTION=mean
OVERLAP_HEADS="28,31"
OVERLAP_LAYER=22
BOX_THRESHOLD=0.10
MAX_BOX_AREA=0.5

# ---------- sidecar / vLLM defaults ----------
DINO_PORT=${DINO_PORT:-8100}
VLLM_PORT=${VLLM_PORT:-8000}
VLLM_GPU_MEM=${VLLM_GPU_MEM:-0.90}          # vLLM has its GPU to itself -> can be high
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-4096}   # >= max_prompt_length(2048)+max_completion(1024)
VLLM_ENFORCE_EAGER=${VLLM_ENFORCE_EAGER:-False}  # set True if CUDA-graph capture misbehaves
# FLAN-T5 observe-step classifier: run on the training GPU (was cpu -> a serial bottleneck)
OVERLAP_STEPS_DEVICE=${OVERLAP_STEPS_DEVICE:-cuda}

# ---------- parse args ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)                  MODEL="$2";                  shift 2 ;;
        --num-gpus)               NUM_GPUS="$2";               shift 2 ;;
        --output-dir)             OUTPUT_DIR="$2";             shift 2 ;;
        --max-completion-length)  MAX_COMPLETION_LENGTH="$2";  shift 2 ;;
        --num-generations)        NUM_GENERATIONS="$2";        shift 2 ;;
        --grad-accum)             GRAD_ACCUM="$2";             shift 2 ;;
        --per-device-batch)       PER_DEVICE_BATCH="$2";       shift 2 ;;
        --learning-rate)          LEARNING_RATE="$2";          shift 2 ;;
        --w-overlap)              W_OVERLAP="$2";              shift 2 ;;
        --token-reduction)        TOKEN_REDUCTION="$2";        shift 2 ;;
        --overlap-heads)          OVERLAP_HEADS="$2";          shift 2 ;;
        --overlap-layer)          OVERLAP_LAYER="$2";          shift 2 ;;
        --box-threshold)          BOX_THRESHOLD="$2";          shift 2 ;;
        --max-box-area)           MAX_BOX_AREA="$2";           shift 2 ;;
        --dino-port)              DINO_PORT="$2";              shift 2 ;;
        --vllm-port)              VLLM_PORT="$2";              shift 2 ;;
        --vllm-gpu-mem)           VLLM_GPU_MEM="$2";           shift 2 ;;
        --vllm-max-model-len)     VLLM_MAX_MODEL_LEN="$2";     shift 2 ;;
        --vllm-enforce-eager)     VLLM_ENFORCE_EAGER="$2";     shift 2 ;;
        *)                        EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

if (( NUM_GPUS < 3 )); then
    echo "ERROR: need >=3 GPUs (1 DINO + 1 vLLM + >=1 training); got NUM_GPUS=$NUM_GPUS" >&2
    exit 1
fi
DINO_GPU=0
VLLM_GPU=1
TRAIN_N=$(( NUM_GPUS - 2 ))
TRAIN_GPUS=$(seq -s, 2 $(( NUM_GPUS - 1 )))       # e.g. "2,3,4,5,6,7"

# reforward is mandatory: vLLM does not return attention, so saliency must re-forward.
REFORWARD_SALIENCY=True

# ---------- naming (every swept HP appears in model + wandb name) ----------
N_HEADS=$(echo "$OVERLAP_HEADS" | awk -F, '{print NF}')
SUFFIX="__wov${W_OVERLAP}_${N_HEADS}head_tr${TOKEN_REDUCTION}"
MODEL_SLUG=$(echo "$MODEL" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9_]/_/g')
RUN_NAME="grpo-${MODEL_SLUG}-overlap${SUFFIX}"
[[ -z "$OUTPUT_DIR" ]] && OUTPUT_DIR="$REPO/checkpoint/${RUN_NAME}"
mkdir -p "$OUTPUT_DIR"

# reward_funcs order (grpo_vlm_qwen3.py, ours branch): [format, overlap, accuracy, judge]
REWARD_WEIGHTS="1.0 ${W_OVERLAP} 1.0 1.0"

echo "=========================================================================="
echo "Model:            $MODEL"
echo "GPUs (total $NUM_GPUS):  DINO=cuda:$DINO_GPU  vLLM=cuda:$VLLM_GPU  train=cuda:[$TRAIN_GPUS] ($TRAIN_N procs)"
echo "Generation:       vLLM server mode  127.0.0.1:$VLLM_PORT  gpu_mem=$VLLM_GPU_MEM  max_len=$VLLM_MAX_MODEL_LEN"
echo "DINO reward:      127.0.0.1:$DINO_PORT  box_threshold=$BOX_THRESHOLD max_box_area=$MAX_BOX_AREA"
echo "Overlap reward:   layer=$OVERLAP_LAYER heads=[$OVERLAP_HEADS] token_reduction=$TOKEN_REDUCTION w_overlap=$W_OVERLAP"
echo "Batch:            per_device=$PER_DEVICE_BATCH num_generations=$NUM_GENERATIONS grad_accum=$GRAD_ACCUM  (gen_batch=$(( PER_DEVICE_BATCH * TRAIN_N * GRAD_ACCUM )))"
echo "T5 step clf:      $OVERLAP_STEPS_DEVICE"
echo "Run name:         $RUN_NAME"
echo "Output dir:       $OUTPUT_DIR"
echo "Judge key:        $([[ -n "${NVIDIA_API_KEY:-}${OPENAI_API_KEY:-}" ]] && echo '(set)' || echo '(MISSING - openai_reward will fail)')"
echo "WandB:            $([[ -n "${WANDB_API_KEY:-}" ]] && echo '(online)' || echo '(offline)')"
[[ -n "$EXTRA_ARGS" ]] && echo "Extra args:       $EXTRA_ARGS"
echo "=========================================================================="

# ---------- environment ----------
source "$CONDA_SH"
conda activate "$CONDA_ENV"

export CUDA_HOME=${CUDA_HOME:-/cm/shared/apps/cuda12.4/toolkit/12.4.1}
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
bash "$REPO/check_cuda_home.sh" || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME
export HF_HUB_OFFLINE=1
export HF_TOKEN=${HF_TOKEN:-}
export WANDB_API_KEY=${WANDB_API_KEY:-}
[ -z "${WANDB_API_KEY:-}" ] && export WANDB_MODE=offline
export WANDB_PROJECT=vlm_reasoning
export WANDB_ENTITY=nvr-israel
export WANDB_RUN_ID=${WANDB_RUN_ID:-$RUN_NAME}
export WANDB_NAME=${WANDB_NAME:-$RUN_NAME}
export WANDB_RESUME=${WANDB_RESUME:-allow}
export OVERLAP_STEPS_DEVICE
[ -n "${NVIDIA_API_KEY:-}" ] && export NVIDIA_API_KEY
[ -n "${OPENAI_API_KEY:-}" ] && export OPENAI_API_KEY
[ -n "${OPENAI_BASE_URL:-}" ] && export OPENAI_BASE_URL
[ -n "${JUDGE_MODEL:-}" ] && export JUDGE_MODEL

LOG_DIR="$OUTPUT_DIR/sidecar_logs"
mkdir -p "$LOG_DIR"

# ---------- cleanup: kill sidecars (and their worker children) on any exit ----------
DINO_PID=""
VLLM_PID=""
CLEANUP_PID=""
cleanup() {
    echo "[cleanup] shutting down sidecars ..."
    for pid in "$VLLM_PID" "$DINO_PID" "$CLEANUP_PID"; do
        [ -n "$pid" ] || continue
        pkill -TERM -P "$pid" 2>/dev/null || true
        kill -TERM "$pid" 2>/dev/null || true
    done
    # belt-and-suspenders: vLLM spawns detached workers
    pkill -TERM -u "$USER" -f "trl.scripts.vllm_serve --model $MODEL" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait_for_health() {   # $1=url  $2=name  $3=timeout_s  $4=pid
    local url="$1" name="$2" timeout_s="$3" pid="$4" waited=0
    echo "[health] waiting for $name at $url (timeout ${timeout_s}s) ..."
    until curl -sf "$url" >/dev/null 2>&1; do
        if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
            echo "[health] ERROR: $name process (pid $pid) died before becoming healthy." >&2
            echo "[health] --- last 40 lines of its log ---" >&2
            tail -40 "$LOG_DIR/${name}.log" >&2 2>/dev/null || true
            exit 1
        fi
        sleep 5; waited=$(( waited + 5 ))
        if (( waited >= timeout_s )); then
            echo "[health] ERROR: $name not healthy after ${timeout_s}s." >&2
            tail -40 "$LOG_DIR/${name}.log" >&2 2>/dev/null || true
            exit 1
        fi
    done
    echo "[health] $name is up (after ${waited}s)."
}

# ---------- 1. Grounding-DINO reward server on GPU 0 ----------
echo "[start] Grounding-DINO on cuda:$DINO_GPU -> 127.0.0.1:$DINO_PORT"
CUDA_VISIBLE_DEVICES=$DINO_GPU DINO_SERVER_BATCH=${DINO_SERVER_BATCH:-8} \
    python "$REPO/serve_grounding_dino.py" --host 127.0.0.1 --port "$DINO_PORT" \
    > "$LOG_DIR/dino.log" 2>&1 &
DINO_PID=$!

# ---------- 2. vLLM generation server on GPU 1 ----------
cd "$REPO/trl_repo"   # so `python -m trl.scripts.vllm_serve` resolves to the editable trl
echo "[start] vLLM server on cuda:$VLLM_GPU -> 127.0.0.1:$VLLM_PORT"
# Only pass --enforce_eager when enabling it (avoids bool-string parsing edge cases).
VLLM_EAGER_FLAG=""
case "$VLLM_ENFORCE_EAGER" in
    True|true|1) VLLM_EAGER_FLAG="--enforce_eager True" ;;
esac
CUDA_VISIBLE_DEVICES=$VLLM_GPU \
    python -m trl.scripts.vllm_serve \
        --model "$MODEL" \
        --host 127.0.0.1 --port "$VLLM_PORT" \
        --gpu_memory_utilization "$VLLM_GPU_MEM" \
        --dtype bfloat16 \
        --max_model_len "$VLLM_MAX_MODEL_LEN" \
        --enable_prefix_caching True \
        $VLLM_EAGER_FLAG \
    > "$LOG_DIR/vllm.log" 2>&1 &
VLLM_PID=$!

# DINO loads fast (~1-2 min); vLLM must load the 8B + capture CUDA graphs (~5-15 min).
wait_for_health "http://127.0.0.1:$DINO_PORT/health"  "dino" 600  "$DINO_PID"
wait_for_health "http://127.0.0.1:$VLLM_PORT/health/" "vllm" 1800 "$VLLM_PID"

# ---------- 3. checkpoint housekeeping (keep every-200; drop stale) ----------
_cleanup_checkpoints() {
    local output_dir="$1" prev_latest="" latest step
    while true; do
        sleep 30
        latest=$(ls -d "$output_dir"/checkpoint-* 2>/dev/null | sed 's|.*/checkpoint-||' | sort -n | tail -1)
        if [[ -n "$latest" && "$latest" != "$prev_latest" ]]; then
            prev_latest="$latest"
            ls -d "$output_dir"/checkpoint-* 2>/dev/null | sed 's|.*/checkpoint-||' | sort -n | while read -r step; do
                if (( step % 200 != 0 )) && [[ "$step" != "$latest" ]]; then
                    echo "[checkpoint cleanup] Removing $output_dir/checkpoint-$step"
                    rm -rf "$output_dir/checkpoint-$step"
                fi
            done
        fi
    done
}
_cleanup_checkpoints "$OUTPUT_DIR" &
CLEANUP_PID=$!

RESUME_FLAG=""
LATEST_CKPT=$(ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | sed 's|.*/checkpoint-||' | sort -n | tail -1)
[ -n "$LATEST_CKPT" ] && RESUME_FLAG="--resume_from_checkpoint $OUTPUT_DIR/checkpoint-$LATEST_CKPT"

MASTER_PORT=${MASTER_PORT:-$(shuf -i 29500-65000 -n 1)}

# ---------- 4. GRPO training on GPUs 2..N-1 ----------
echo "[start] training on cuda:[$TRAIN_GPUS] ($TRAIN_N procs)"
CUDA_VISIBLE_DEVICES=$TRAIN_GPUS accelerate launch \
    --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    --num_processes "$TRAIN_N" \
    --main_process_port "$MASTER_PORT" \
    examples/scripts/grpo_vlm_qwen3.py \
    --model_name_or_path "$MODEL" \
    --attn_implementation sdpa \
    --output_dir "$OUTPUT_DIR" \
    --learning_rate "$LEARNING_RATE" \
    --torch_dtype bfloat16 \
    --max_prompt_length 2048 \
    --max_completion_length "$MAX_COMPLETION_LENGTH" \
    --reforward_saliency "$REFORWARD_SALIENCY" \
    --reward_variant ours \
    --overlap_layer "$OVERLAP_LAYER" \
    --overlap_heads "$OVERLAP_HEADS" \
    --token_reduction "$TOKEN_REDUCTION" \
    --box_threshold "$BOX_THRESHOLD" \
    --max_box_area "$MAX_BOX_AREA" \
    --dino_api_base "http://127.0.0.1:$DINO_PORT" \
    --reward_weights $REWARD_WEIGHTS \
    --use_vllm \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port "$VLLM_PORT" \
    --use_peft \
    --lora_target_modules q_proj v_proj \
    --log_completions \
    --per_device_train_batch_size "$PER_DEVICE_BATCH" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --num_generations "$NUM_GENERATIONS" \
    --report_to wandb \
    --logging_steps 5 \
    --save_steps 200 \
    --num_train_epochs 3 \
    --temperature 1 \
    $RESUME_FLAG \
    $EXTRA_ARGS

echo "Finished $RUN_NAME"
