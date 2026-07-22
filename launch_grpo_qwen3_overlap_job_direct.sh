#!/bin/bash
# Run GRPO training for Qwen3-VL-8B with OUR attention-overlap reward (reward_variant=ours)
# directly on the current node (no submit_job/SLURM). Flag-selectable alternative to the
# Saliency-R1 reward: this launcher hard-selects reward_variant=ours and wires the swept
# hyperparameters (w_overlap, head-mode, token-reduction) into BOTH the checkpoint dir name
# and the wandb run name (naming convention required by grpo-reward-port-plan).
#
# Usage (from a GPU node with an 8-GPU allocation active):
#   WANDB_API_KEY=... NVIDIA_API_KEY=... bash launch_grpo_qwen3_overlap_job_direct.sh --num-gpus 8
#   bash launch_grpo_qwen3_overlap_job_direct.sh --num-gpus 8 --w-overlap 0.2 --token-reduction max
#
# Any flag not parsed below is forwarded verbatim to grpo_vlm_qwen3.py via EXTRA_ARGS.
set -e

REPO=/home/uberger/scratch/research/saliency_r1
CONDA_SH=/home/uberger/scratch/miniconda3/etc/profile.d/conda.sh
CONDA_ENV=saliency_r1_qwen3
HF_HOME=${HF_HOME:-/home/uberger/scratch/cache/hf_cache}

# ---------- training defaults ----------
# GRPO inits from the cold-started+merged Qwen3-VL-8B (same base as the saliency_r1 runs).
MODEL="$REPO/checkpoint/coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged"
NUM_GPUS=1
OUTPUT_DIR=""
MAX_COMPLETION_LENGTH=1024
EXTRA_ARGS=""

# ---------- overlap-reward (swept) defaults ----------
W_OVERLAP=0.2            # reward weight on the overlap term (weak teacher; honest |r|~0.22)
TOKEN_REDUCTION=mean     # mean | max   (trmean / trmax in the name)
OVERLAP_HEADS="28,31"    # fixed 2-head option at layer 22: (22,28)+(22,31)
OVERLAP_LAYER=22
BOX_THRESHOLD=0.10
MAX_BOX_AREA=0.5
DINO_API_BASE=""         # empty -> DINO runs locally on each training process's device

# ---------- parse args ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)                  MODEL="$2";                  shift 2 ;;
        --num-gpus)               NUM_GPUS="$2";               shift 2 ;;
        --output-dir)             OUTPUT_DIR="$2";             shift 2 ;;
        --nvidia-api-key)         NVIDIA_API_KEY="$2";         shift 2 ;;
        --openai-api-key)         OPENAI_API_KEY="$2";         shift 2 ;;
        --wandb-api-key)          WANDB_API_KEY="$2";          shift 2 ;;
        --hf-token)               HF_TOKEN="$2";               shift 2 ;;
        --max-completion-length)  MAX_COMPLETION_LENGTH="$2";  shift 2 ;;
        --w-overlap)              W_OVERLAP="$2";              shift 2 ;;
        --token-reduction)        TOKEN_REDUCTION="$2";        shift 2 ;;
        --overlap-heads)          OVERLAP_HEADS="$2";          shift 2 ;;
        --overlap-layer)          OVERLAP_LAYER="$2";          shift 2 ;;
        --box-threshold)          BOX_THRESHOLD="$2";          shift 2 ;;
        --max-box-area)           MAX_BOX_AREA="$2";           shift 2 ;;
        --dino-api-base)          DINO_API_BASE="$2";          shift 2 ;;
        *)                        EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

REFORWARD_SALIENCY=True

# ---------- naming: every swept HP appears in the model AND wandb name ----------
N_HEADS=$(echo "$OVERLAP_HEADS" | awk -F, '{print NF}')
HEADMODE="${N_HEADS}head"
TR_TAG="tr${TOKEN_REDUCTION}"
# strip leading "0." etc so wov0.2 reads cleanly
SUFFIX="__wov${W_OVERLAP}_${HEADMODE}_${TR_TAG}"

MODEL_SLUG=$(echo "$MODEL" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9_]/_/g')
RUN_NAME="grpo-${MODEL_SLUG}-overlap${SUFFIX}"
[[ -z "$OUTPUT_DIR" ]] && OUTPUT_DIR="$REPO/checkpoint/${RUN_NAME}"
JOB_NAME="$RUN_NAME"
mkdir -p "$OUTPUT_DIR"

# reward_funcs order (grpo_vlm_qwen3.py, ours branch): [format, overlap, accuracy, judge]
REWARD_WEIGHTS="1.0 ${W_OVERLAP} 1.0 1.0"

echo "Model:          $MODEL"
echo "GPUs:           $NUM_GPUS"
echo "Reward variant: ours (overlap)  layer=$OVERLAP_LAYER heads=[$OVERLAP_HEADS] token_reduction=$TOKEN_REDUCTION"
echo "w_overlap:      $W_OVERLAP   reward_weights=[$REWARD_WEIGHTS] (format, overlap, accuracy, judge)"
echo "DINO:           box_threshold=$BOX_THRESHOLD max_box_area=$MAX_BOX_AREA  $([[ -n "$DINO_API_BASE" ]] && echo "served=$DINO_API_BASE" || echo 'local-on-device')"
echo "Run name:       $RUN_NAME"
echo "Output dir:     $OUTPUT_DIR"
echo "Judge key:      $([[ -n "${NVIDIA_API_KEY:-}${OPENAI_API_KEY:-}" ]] && echo '(set)' || echo '(MISSING - openai_reward will fail)')"
echo "WandB:          $([[ -n "$WANDB_API_KEY" ]] && echo '(online)' || echo '(offline)')"
[[ -n "$EXTRA_ARGS" ]] && echo "Extra args:     $EXTRA_ARGS"
echo ""

source "$CONDA_SH"
conda activate "$CONDA_ENV"

export CUDA_HOME=${CUDA_HOME:-/cm/shared/apps/cuda12.4/toolkit/12.4.1}
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
bash "$REPO/check_cuda_home.sh" || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DS_BUILD_OPS=0

export HF_HOME
export HF_HUB_OFFLINE=1
export HF_TOKEN=${HF_TOKEN:-}
export WANDB_API_KEY=${WANDB_API_KEY:-}
[ -z "$WANDB_API_KEY" ] && export WANDB_MODE=offline
export WANDB_PROJECT=vlm_reasoning
export WANDB_ENTITY=nvr-israel
export WANDB_RUN_ID=${WANDB_RUN_ID:-$RUN_NAME}
export WANDB_NAME=${WANDB_NAME:-$RUN_NAME}
export WANDB_RESUME=${WANDB_RESUME:-allow}
# FLAN-T5 observe-step classifier stays on CPU by default (frees GPU for the policy).
export OVERLAP_STEPS_DEVICE=${OVERLAP_STEPS_DEVICE:-cpu}
# FLAN-T5 observe-step classifier checkpoint. Local (fs12) copy; the in-code default
# (overlap_steps.py _DEFAULT_CKPT) points at /lustre/fs1, which is NOT mounted on this
# cluster. Layout: best/{encoder/,tokenizer/,head.pt,cfg.json}. Override via env if moved.
export OVERLAP_STEPS_CKPT=${OVERLAP_STEPS_CKPT:-$REPO/checkpoint/steps_classifier/best}
[ -d "$OVERLAP_STEPS_CKPT/encoder" ] || { echo "ERROR: steps-classifier ckpt not found at $OVERLAP_STEPS_CKPT (need encoder/ tokenizer/ head.pt). Set OVERLAP_STEPS_CKPT to a valid path." >&2; exit 1; }
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

DINO_FLAG=""
[[ -n "$DINO_API_BASE" ]] && DINO_FLAG="--dino_api_base $DINO_API_BASE"

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
    --max_completion_length "$MAX_COMPLETION_LENGTH" \
    --reforward_saliency "$REFORWARD_SALIENCY" \
    --reward_variant ours \
    --overlap_layer "$OVERLAP_LAYER" \
    --overlap_heads "$OVERLAP_HEADS" \
    --token_reduction "$TOKEN_REDUCTION" \
    --box_threshold "$BOX_THRESHOLD" \
    --max_box_area "$MAX_BOX_AREA" \
    $DINO_FLAG \
    --reward_weights $REWARD_WEIGHTS \
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
