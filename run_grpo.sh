#!/usr/bin/env bash
# Launch Saliency-R1 GRPO training.
#
# Setup (done once, see SETUP_NOTES.md):
#   - conda env `saliency_r1` (python 3.10) with requirements_clean.txt installed
#   - transformers internals patched (patch_transformers.sh)
#   - patched TRL installed editable from ./trl_repo
#
# This node has NO GPU — run this on a GPU node (e.g. via the cluster scheduler).
#
# Reward functions in examples/scripts/grpo_vlm.py:
#   think_format_reward, think_saliency_reward, openai_reward
# openai_reward is an LLM-as-judge and needs a real key in
#   trl_repo/trl/rewards/openai_rewards.py (currently api_key="xxx").
# For a first "does it run" test without OpenAI, edit grpo_vlm.py to drop
#   openai_reward from reward_funcs (see SETUP_NOTES.md §Run options).

set -euo pipefail

export HF_HOME=${HF_HOME:-/home/uberger/scratch/cache/hf_cache}
export HF_TOKEN=${HF_TOKEN:-}       # set for gated model/dataset downloads
export WANDB_API_KEY=${WANDB_API_KEY:-}
[ -z "$WANDB_API_KEY" ] && export WANDB_MODE=offline

REPO=/home/uberger/scratch/research/saliency_r1
MODEL=${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}   # smoke-test default; paper uses peterant330/Saliency-R1-CI-v2 (7B)
OUTPUT_DIR=${OUTPUT_DIR:-$REPO/../checkpoint/grpo-saliency-r1-smoke}

source /home/uberger/scratch/miniconda3/etc/profile.d/conda.sh
conda activate saliency_r1
cd "$REPO/trl_repo"

accelerate launch \
    --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/grpo_vlm.py \
    --model_name_or_path "$MODEL" \
    --attn_implementation sdpa \
    --output_dir "$OUTPUT_DIR" \
    --learning_rate 1e-5 \
    --repetition_penalty 1.05 \
    --torch_dtype bfloat16 \
    --max_prompt_length 1024 \
    --max_completion_length 512 \
    --use_peft \
    --lora_target_modules q_proj v_proj \
    --log_completions \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --num_generations 8 \
    --report_to wandb \
    --save_steps 200 \
    --num_train_epochs 3 \
    --temperature 1
