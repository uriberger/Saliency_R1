#!/bin/bash
# Run Saliency-R1 GRPO training for Qwen3-VL-8B-Instruct directly on the current
# node (no submit_job/SLURM submission). Same arg parsing and env setup as
# launch_grpo_qwen3_job.sh; use this when you already have an interactive GPU
# allocation (salloc) and just want to launch in-place.
#
# Usage (from a GPU node with the allocation active):
#   WANDB_API_KEY=... bash launch_grpo_qwen3_job_direct.sh
#   bash launch_grpo_qwen3_job_direct.sh --model peterant330/Saliency-R1-CI-v2 --num-gpus 8 \
#       --nvidia-api-key nvapi-... --wandb-api-key ...
#
# Unlike launch_grpo_qwen3_job.sh this does NOT go through submit_job: no auto-resume
# wall-limit handling. It still resumes from the latest checkpoint in --output-dir
# if one exists, so rerunning after a crash picks up where it left off.
#
# Any flag not parsed below is forwarded verbatim to grpo_vlm_qwen3.py via EXTRA_ARGS.
set -e

REPO=/home/uberger/scratch/research/saliency_r1
CONDA_SH=/home/uberger/scratch/miniconda3/etc/profile.d/conda.sh
CONDA_ENV=saliency_r1_qwen3
HF_HOME=${HF_HOME:-/home/uberger/scratch/cache/hf_cache}

# ---------- training defaults ----------
MODEL="Qwen/Qwen3-VL-8B-Instruct"
NUM_GPUS=1
OUTPUT_DIR=""
EXTRA_ARGS=""

# ---------- parse args ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)          MODEL="$2";          shift 2 ;;
        --num-gpus)       NUM_GPUS="$2";       shift 2 ;;
        --output-dir)     OUTPUT_DIR="$2";     shift 2 ;;
        --nvidia-api-key) NVIDIA_API_KEY="$2"; shift 2 ;;
        --openai-api-key) OPENAI_API_KEY="$2"; shift 2 ;;
        --wandb-api-key)  WANDB_API_KEY="$2";  shift 2 ;;
        --hf-token)       HF_TOKEN="$2";       shift 2 ;;
        *)                EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

MODEL_SLUG=$(echo "$MODEL" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9_]/_/g')
[[ -z "$OUTPUT_DIR" ]] && OUTPUT_DIR="$REPO/checkpoint/grpo-${MODEL_SLUG}-saliency-r1-qwen3"
JOB_NAME="grpo_saliency_r1_qwen3_${MODEL_SLUG}"
mkdir -p "$OUTPUT_DIR"

echo "Model:      $MODEL"
echo "GPUs:       $NUM_GPUS"
echo "Output dir: $OUTPUT_DIR"
echo "Judge key:  $([[ -n "${NVIDIA_API_KEY:-}${OPENAI_API_KEY:-}" ]] && echo '(set)' || echo '(MISSING - openai_reward will fail)')  endpoint=${OPENAI_BASE_URL:-https://inference-api.nvidia.com} model=${JUDGE_MODEL:-azure/openai/gpt-4o-mini}"
echo "WandB:      $([[ -n "$WANDB_API_KEY" ]] && echo '(online)' || echo '(offline)')"
[[ -n "$EXTRA_ARGS" ]] && echo "Extra args: $EXTRA_ARGS"
echo ""

source "$CONDA_SH"
conda activate "$CONDA_ENV"

export CUDA_HOME=${CUDA_HOME:-/cm/shared/apps/cuda12.4/toolkit/12.4.1}
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export HF_HOME
export HF_HUB_OFFLINE=1
export HF_TOKEN=${HF_TOKEN:-}
export WANDB_API_KEY=${WANDB_API_KEY:-}
[ -z "$WANDB_API_KEY" ] && export WANDB_MODE=offline
export WANDB_PROJECT=vlm_reasoning
export WANDB_ENTITY=nvr-israel
export WANDB_RUN_ID=${WANDB_RUN_ID:-grpo-${MODEL_SLUG}-saliency-r1-qwen3}
export WANDB_RESUME=${WANDB_RESUME:-allow}
[ -n "${NVIDIA_API_KEY:-}" ] && export NVIDIA_API_KEY
[ -n "${OPENAI_API_KEY:-}" ] && export OPENAI_API_KEY
[ -n "${OPENAI_BASE_URL:-}" ] && export OPENAI_BASE_URL
[ -n "${JUDGE_MODEL:-}" ] && export JUDGE_MODEL
cd "$REPO/trl_repo"

RESUME_FLAG=""
LATEST_CKPT=$(ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | sed 's|.*/checkpoint-||' | sort -n | tail -1)
[ -n "$LATEST_CKPT" ] && RESUME_FLAG="--resume_from_checkpoint $OUTPUT_DIR/checkpoint-$LATEST_CKPT"

# Background loop: keep every-200-step checkpoints permanently; delete older
# non-200-step checkpoints as soon as a newer one is saved.
_cleanup_checkpoints() {
    local output_dir="$1"
    local prev_latest=""
    while true; do
        sleep 30
        local latest
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
trap "kill $CLEANUP_PID 2>/dev/null; wait $CLEANUP_PID 2>/dev/null" EXIT

MASTER_PORT=${MASTER_PORT:-$(shuf -i 29500-65000 -n 1)}

accelerate launch \
    --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    --num_processes "$NUM_GPUS" \
    --main_process_port "$MASTER_PORT" \
    examples/scripts/grpo_vlm_qwen3.py \
    --model_name_or_path "$MODEL" \
    --attn_implementation sdpa \
    --output_dir "$OUTPUT_DIR" \
    --learning_rate 1e-5 \
    --torch_dtype bfloat16 \
    --max_prompt_length 2048 \
    --max_completion_length 2048 \
    --use_peft \
    --lora_target_modules q_proj v_proj \
    --log_completions \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_generations 8 \
    --report_to wandb \
    --logging_steps 5 \
    --save_steps 10 \
    --num_train_epochs 3 \
    --temperature 1 \
    $RESUME_FLAG \
    $EXTRA_ARGS

echo "Finished $JOB_NAME"
