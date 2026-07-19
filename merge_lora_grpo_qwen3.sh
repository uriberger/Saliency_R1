#!/bin/bash
# Merge the LoRA adapter from the GRPO run into the base (merged SFT) model,
# producing a standalone merged checkpoint ready for evaluation.
#
# Adapter: checkpoint/grpo-coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged-saliency-r1-qwen3
# Base:    checkpoint/coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged
# Output:  checkpoint/grpo-coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged-saliency-r1-qwen3_merged
#
# Usage (from any node with GPU or CPU — merge only needs CPU):
#   bash merge_lora_grpo_qwen3.sh
set -e

REPO=/home/uberger/scratch/research/saliency_r1
CONDA_SH=/home/uberger/scratch/miniconda3/etc/profile.d/conda.sh
CONDA_ENV=saliency_r1_qwen3

ADAPTER="$REPO/checkpoint/grpo-coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged-saliency-r1-qwen3"
BASE="$REPO/checkpoint/coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged"
OUTPUT="$REPO/checkpoint/grpo-coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged-saliency-r1-qwen3_merged"

echo "Adapter: $ADAPTER"
echo "Base:    $BASE"
echo "Output:  $OUTPUT"
echo ""

source "$CONDA_SH"
conda activate "$CONDA_ENV"

export CUDA_HOME=/cm/shared/apps/cuda12.4/toolkit/12.4.1
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

python "$REPO/merge_lora.py" \
    --adapter "$ADAPTER" \
    --base    "$BASE" \
    --output  "$OUTPUT"

echo "Merged model saved to $OUTPUT"
