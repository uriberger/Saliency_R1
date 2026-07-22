#!/bin/bash
# GRPO training for Qwen3-VL-8B with attention-overlap reward (colocated).
# Co-locates DINO + vLLM sidecars on the same node as training:
#     GPU 0         Grounding-DINO reward server   (127.0.0.1:$DINO_PORT)
#     GPU 1         vLLM generation server         (127.0.0.1:$VLLM_PORT)
#     GPU 2..N-1    GRPO training, DeepSpeed ZeRO-3  (N-2 processes)
#
# vLLM replaces the slow HF generate() path -- this is the main speedup over the
# non-colocated launcher. The policy is LoRA; the trainer merges the adapter and
# pushes weights to the vLLM server over NCCL every step.
#
# Default: submits to the cluster via submit_job (SLURM).
# --direct: runs on the current node immediately (no SLURM), e.g. on an interactive GPU node.
#
# Usage:
#   WANDB_API_KEY=... NVIDIA_API_KEY=... bash launch_grpo_qwen3_overlap_colocated_job.sh [OPTIONS]
#   bash launch_grpo_qwen3_overlap_colocated_job.sh --direct --num-gpus 8
#
# Environment overrides:
#   PARTITION=batch_singlenode   DURATION=4 (hours)
#   SAVE_STEPS=10   CKPT_KEEP_EVERY=200
#   DINO_PORT=8100   VLLM_PORT=8000   VLLM_GPU_MEM=0.90   VLLM_MAX_MODEL_LEN=4096
#   VLLM_ENFORCE_EAGER=False
#   OVERLAP_STEPS_DEVICE=cpu   OVERLAP_STEPS_CKPT=<path>
#   NVIDIA_API_KEY / OPENAI_API_KEY / OPENAI_BASE_URL / JUDGE_MODEL
#   WANDB_API_KEY   (omit -> offline)   HF_TOKEN
set -euo pipefail

SCRIPT_PATH="$(realpath "$0")"
REPO=/home/uberger/scratch/research/saliency_r1
CONDA_SH=/home/uberger/scratch/miniconda3/etc/profile.d/conda.sh
# This colocated job runs both the vLLM server and the trainer in ONE env, so it
# MUST use the vllm-enabled env. Hardcoded (not overridable) to prevent picking up
# a stray CONDA_ENV from the shell, which silently breaks the vLLM sidecar.
CONDA_ENV=saliency_r1_qwen3_vllm
HF_HOME=${HF_HOME:-/home/uberger/scratch/cache/hf_cache}

# ---------- SLURM defaults ----------
ACCOUNT=nvr_israel_rlop
PARTITION=${PARTITION:-batch_singlenode}
DURATION=${DURATION:-4}

# ---------- training defaults ----------
MODEL="$REPO/checkpoint/coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged"
NUM_GPUS=8
OUTPUT_DIR=""
MAX_COMPLETION_LENGTH=1024
NUM_GENERATIONS=8
GRAD_ACCUM=8
PER_DEVICE_BATCH=1
LEARNING_RATE=1e-5
SAVE_STEPS=${SAVE_STEPS:-10}
CKPT_KEEP_EVERY=${CKPT_KEEP_EVERY:-200}
EXTRA_ARGS=""
DIRECT=false

# ---------- overlap-reward defaults ----------
W_OVERLAP=0.2
TOKEN_REDUCTION=mean
OVERLAP_HEADS="28,31"
OVERLAP_LAYER=22
BOX_THRESHOLD=0.10
MAX_BOX_AREA=0.5

# ---------- sidecar defaults ----------
DINO_PORT=${DINO_PORT:-8100}
VLLM_PORT=${VLLM_PORT:-8000}
VLLM_GPU_MEM=${VLLM_GPU_MEM:-0.90}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-4096}
VLLM_ENFORCE_EAGER=${VLLM_ENFORCE_EAGER:-False}
OVERLAP_STEPS_DEVICE=${OVERLAP_STEPS_DEVICE:-cpu}
OVERLAP_STEPS_CKPT=${OVERLAP_STEPS_CKPT:-$REPO/checkpoint/steps_classifier/best}

# ---------- parse args ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --direct)                 DIRECT=true;                  shift ;;
        --model)                  MODEL="$2";                   shift 2 ;;
        --num-gpus)               NUM_GPUS="$2";                shift 2 ;;
        --output-dir)             OUTPUT_DIR="$2";              shift 2 ;;
        --partition)              PARTITION="$2";               shift 2 ;;
        --duration)               DURATION="$2";                shift 2 ;;
        --nvidia-api-key)         NVIDIA_API_KEY="$2";          shift 2 ;;
        --openai-api-key)         OPENAI_API_KEY="$2";          shift 2 ;;
        --wandb-api-key)          WANDB_API_KEY="$2";           shift 2 ;;
        --hf-token)               HF_TOKEN="$2";                shift 2 ;;
        --max-completion-length)  MAX_COMPLETION_LENGTH="$2";   shift 2 ;;
        --num-generations)        NUM_GENERATIONS="$2";         shift 2 ;;
        --grad-accum)             GRAD_ACCUM="$2";              shift 2 ;;
        --per-device-batch)       PER_DEVICE_BATCH="$2";        shift 2 ;;
        --learning-rate)          LEARNING_RATE="$2";           shift 2 ;;
        --w-overlap)              W_OVERLAP="$2";               shift 2 ;;
        --token-reduction)        TOKEN_REDUCTION="$2";         shift 2 ;;
        --overlap-heads)          OVERLAP_HEADS="$2";           shift 2 ;;
        --overlap-layer)          OVERLAP_LAYER="$2";           shift 2 ;;
        --box-threshold)          BOX_THRESHOLD="$2";           shift 2 ;;
        --max-box-area)           MAX_BOX_AREA="$2";            shift 2 ;;
        --dino-port)              DINO_PORT="$2";               shift 2 ;;
        --vllm-port)              VLLM_PORT="$2";               shift 2 ;;
        --vllm-gpu-mem)           VLLM_GPU_MEM="$2";            shift 2 ;;
        --vllm-max-model-len)     VLLM_MAX_MODEL_LEN="$2";      shift 2 ;;
        --vllm-enforce-eager)     VLLM_ENFORCE_EAGER="$2";      shift 2 ;;
        *)                        EXTRA_ARGS="$EXTRA_ARGS $1";  shift ;;
    esac
done

if (( NUM_GPUS < 3 )); then
    echo "ERROR: need >=3 GPUs (1 DINO + 1 vLLM + >=1 training); got --num-gpus $NUM_GPUS" >&2
    exit 1
fi
if (( NUM_GPUS > 8 )); then
    echo "ERROR: single-node launcher (max 8 GPUs). localhost DINO/vLLM sidecars cannot serve a second node." >&2
    exit 1
fi

DINO_GPU=0
VLLM_GPU=1
TRAIN_N=$(( NUM_GPUS - 2 ))
TRAIN_GPUS=$(seq -s, 2 $(( NUM_GPUS - 1 )))

REFORWARD_SALIENCY=True

# ---------- naming: every swept HP appears in the model AND wandb name ----------
N_HEADS=$(echo "$OVERLAP_HEADS" | awk -F, '{print NF}')
SUFFIX="__wov${W_OVERLAP}_${N_HEADS}head_tr${TOKEN_REDUCTION}"
MODEL_SLUG=$(echo "$MODEL" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9_]/_/g')
RUN_NAME="grpo-${MODEL_SLUG}-overlap${SUFFIX}"
[[ -z "$OUTPUT_DIR" ]] && OUTPUT_DIR="$REPO/checkpoint/${RUN_NAME}"
mkdir -p "$OUTPUT_DIR"

REWARD_WEIGHTS="1.0 ${W_OVERLAP} 1.0 1.0"

echo "=========================================================================="
echo "Model:            $MODEL"
echo "GPUs (total $NUM_GPUS):  DINO=cuda:$DINO_GPU  vLLM=cuda:$VLLM_GPU  train=cuda:[$TRAIN_GPUS] ($TRAIN_N procs)"
echo "Generation:       vLLM server  127.0.0.1:$VLLM_PORT  gpu_mem=$VLLM_GPU_MEM  max_len=$VLLM_MAX_MODEL_LEN"
echo "DINO reward:      127.0.0.1:$DINO_PORT  box_threshold=$BOX_THRESHOLD max_box_area=$MAX_BOX_AREA"
echo "Overlap reward:   layer=$OVERLAP_LAYER heads=[$OVERLAP_HEADS] token_reduction=$TOKEN_REDUCTION w_overlap=$W_OVERLAP"
echo "Batch:            per_device=$PER_DEVICE_BATCH num_generations=$NUM_GENERATIONS grad_accum=$GRAD_ACCUM  (gen_batch=$(( PER_DEVICE_BATCH * TRAIN_N * GRAD_ACCUM )))"
echo "T5 step clf:      $OVERLAP_STEPS_DEVICE  ckpt=$OVERLAP_STEPS_CKPT"
echo "Run name:         $RUN_NAME"
echo "Output dir:       $OUTPUT_DIR"
echo "Mode:             $($DIRECT && echo 'direct (no SLURM)' || echo "SLURM ($PARTITION, ${DURATION}h)")"
echo "Judge key:        $([[ -n "${NVIDIA_API_KEY:-}${OPENAI_API_KEY:-}" ]] && echo '(set)' || echo '(MISSING - openai_reward will fail)')"
echo "WandB:            $([[ -n "${WANDB_API_KEY:-}" ]] && echo '(online)' || echo '(offline)')"
[[ -n "$EXTRA_ARGS" ]] && echo "Extra args:       $EXTRA_ARGS"
echo "=========================================================================="

# ---------- SLURM path ----------
if ! $DIRECT; then
    if ! command -v submit_job >/dev/null 2>&1; then
        for CI_ROOT in \
            /lustre/fs1/portfolios/adlr/projects/adlr_other_infra/release/cluster-interface \
            /lustre/fsw/portfolios/adlr/projects/adlr_other_infra/release/cluster-interface; do
            for CAND in "$CI_ROOT/latest" $(ls -1dt "$CI_ROOT"/*/ 2>/dev/null); do
                if [ -x "${CAND%/}/submit_job" ]; then export PATH="${CAND%/}:$PATH"; break 2; fi
            done
        done
    fi
    command -v submit_job >/dev/null 2>&1 || {
        echo "ERROR: submit_job not found under cluster-interface paths. Use --direct to run without SLURM." >&2
        exit 1
    }

    LOG_ROOT="$REPO/outputs/logs"
    mkdir -p "$LOG_ROOT"

    submit_job \
        --account "$ACCOUNT" \
        --partition "$PARTITION" \
        --name "$RUN_NAME" \
        --gpu "$NUM_GPUS" \
        --duration "$DURATION" \
        --autoresume_uninstrumented \
        --outfile "$LOG_ROOT/${RUN_NAME}.%j.out" \
        --logroot "$LOG_ROOT" \
        -c "bash -c '
            export CONDA_ENV=$CONDA_ENV;
            export HF_HOME=$HF_HOME;
            export WANDB_API_KEY=${WANDB_API_KEY:-};
            ${HF_TOKEN:+export HF_TOKEN=$HF_TOKEN;}
            ${NVIDIA_API_KEY:+export NVIDIA_API_KEY=$NVIDIA_API_KEY;}
            ${OPENAI_API_KEY:+export OPENAI_API_KEY=$OPENAI_API_KEY;}
            ${OPENAI_BASE_URL:+export OPENAI_BASE_URL=$OPENAI_BASE_URL;}
            ${JUDGE_MODEL:+export JUDGE_MODEL=$JUDGE_MODEL;}
            bash $SCRIPT_PATH --direct \
                --num-gpus $NUM_GPUS \
                --model $MODEL \
                --output-dir $OUTPUT_DIR \
                --max-completion-length $MAX_COMPLETION_LENGTH \
                --num-generations $NUM_GENERATIONS \
                --grad-accum $GRAD_ACCUM \
                --per-device-batch $PER_DEVICE_BATCH \
                --learning-rate $LEARNING_RATE \
                --w-overlap $W_OVERLAP \
                --token-reduction $TOKEN_REDUCTION \
                --overlap-heads $OVERLAP_HEADS \
                --overlap-layer $OVERLAP_LAYER \
                --box-threshold $BOX_THRESHOLD \
                --max-box-area $MAX_BOX_AREA \
                --dino-port $DINO_PORT \
                --vllm-port $VLLM_PORT \
                --vllm-gpu-mem $VLLM_GPU_MEM \
                --vllm-max-model-len $VLLM_MAX_MODEL_LEN \
                --vllm-enforce-eager $VLLM_ENFORCE_EAGER \
                $EXTRA_ARGS
        '"
    echo "Submitted $RUN_NAME"
    exit 0
fi

# ---------- direct path ----------
source "$CONDA_SH"
echo "Activating conda env $CONDA_ENV"
# conda activate and package activate.d hooks (e.g. cuda-nvcc's, which expands
# $NVCC_PREPEND_FLAGS with no default) assume nounset is OFF. Our `set -u` makes
# any such unguarded expansion a fatal "unbound variable". Disable nounset for
# the duration of activation only, then restore it.
set +u
conda activate "$CONDA_ENV"
set -u
# Activating across conda installs -- e.g. when this script is launched (bash)
# from a fish shell that already had a different env active -- can leave a stale
# env's bin/ ahead of ours on PATH, so `python` resolves to the WRONG interpreter
# even though CONDA_DEFAULT_ENV/CONDA_PREFIX are correct. Force this env's bin to
# the front and clear bash's command hash so the right python/torchrun are used.
[ -n "${CONDA_PREFIX:-}" ] || { echo "ERROR: 'conda activate $CONDA_ENV' failed (no CONDA_PREFIX)." >&2; exit 1; }
export PATH="$CONDA_PREFIX/bin:$PATH"
hash -r
if [ "$(command -v python)" != "$CONDA_PREFIX/bin/python" ]; then
    echo "ERROR: python resolves to '$(command -v python)', expected '$CONDA_PREFIX/bin/python' (env '$CONDA_ENV')." >&2
    exit 1
fi

source "$REPO/setup_cuda_home.sh"
if [ "$(command -v python)" != "$CONDA_PREFIX/bin/python" ]; then
    echo "ERROR: after CUDA_HOME setup, python resolves to '$(command -v python)', expected '$CONDA_PREFIX/bin/python' (env '$CONDA_ENV'). CUDA_HOME='$CUDA_HOME' shadowed it." >&2
    exit 1
fi
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
export OVERLAP_STEPS_CKPT
[ -d "$OVERLAP_STEPS_CKPT/encoder" ] || {
    echo "ERROR: steps-classifier ckpt not found at $OVERLAP_STEPS_CKPT (need encoder/ tokenizer/ head.pt). Set OVERLAP_STEPS_CKPT to a valid path." >&2
    exit 1
}
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
    # vLLM spawns a detached EngineCore worker that holds GPU memory and doesn't
    # match the serve cmdline -- kill it explicitly or it orphans GPU 1.
    pkill -TERM -u "$USER" -f "trl.scripts.vllm_serve --model $MODEL" 2>/dev/null || true
    pkill -TERM -u "$USER" -f "VLLM::EngineCore" 2>/dev/null || true
    sleep 2
    pkill -KILL -u "$USER" -f "VLLM::EngineCore" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait_for_health() {
    local url="$1" name="$2" timeout_s="$3" pid="$4" waited=0
    echo "[health] waiting for $name at $url (timeout ${timeout_s}s) ..."
    until curl -sf "$url" >/dev/null 2>&1; do
        if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
            echo "[health] ERROR: $name process (pid $pid) died before becoming healthy." >&2
            tail -40 "$LOG_DIR/${name}.log" >&2 2>/dev/null || true
            exit 1
        fi
        if grep -qE "Engine core initialization failed|EngineCore failed to start|Traceback \(most recent call last\)" "$LOG_DIR/${name}.log" 2>/dev/null; then
            echo "[health] ERROR: $name logged a fatal error (worker died); aborting." >&2
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
cd "$REPO/trl_repo"
echo "[start] vLLM server on cuda:$VLLM_GPU -> 127.0.0.1:$VLLM_PORT"
which python
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

# ---------- 3. checkpoint housekeeping ----------
_cleanup_checkpoints() {
    local output_dir="$1" prev_latest=""
    while true; do
        sleep 30
        local latest
        latest=$(ls -d "$output_dir"/checkpoint-* 2>/dev/null | sed 's|.*/checkpoint-||' | sort -n | tail -1 || true)
        if [[ -n "$latest" && "$latest" != "$prev_latest" ]]; then
            prev_latest="$latest"
            ls -d "$output_dir"/checkpoint-* 2>/dev/null | sed 's|.*/checkpoint-||' | sort -n | while read -r step; do
                if (( step % CKPT_KEEP_EVERY != 0 )) && [[ "$step" != "$latest" ]]; then
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
LATEST_CKPT=$(ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | sed 's|.*/checkpoint-||' | sort -n | tail -1 || true)
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
    --save_steps "$SAVE_STEPS" \
    --num_train_epochs 3 \
    --temperature 1 \
    $RESUME_FLAG \
    $EXTRA_ARGS

echo "Finished $RUN_NAME"
