#!/bin/bash
# Merge the LoRA adapter from a GRPO run into its base (merged SFT) model,
# producing a standalone merged checkpoint ready for evaluation.
#
# The base model is read automatically from the adapter's adapter_config.json
# (base_model_name_or_path), and the output defaults to "<adapter>_merged".
#
# Usage (from any node with GPU or CPU — merge only needs CPU):
#   # merge the default run:
#   bash merge_lora_grpo_qwen3.sh
#   # merge any adapter (e.g. a specific intermediate checkpoint):
#   bash merge_lora_grpo_qwen3.sh /path/to/adapter-dir
#   # merge with an explicit output dir:
#   bash merge_lora_grpo_qwen3.sh /path/to/adapter-dir /path/to/output-dir
set -euo pipefail

REPO=/home/uberger/scratch/research/saliency_r1
CONDA_SH=/home/uberger/scratch/miniconda3/etc/profile.d/conda.sh
CONDA_ENV=saliency_r1_qwen3

# Default adapter if none is passed on the command line.
DEFAULT_ADAPTER="$REPO/checkpoint/grpo-coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged-saliency-r1-qwen3"

ADAPTER="${1:-$DEFAULT_ADAPTER}"
ADAPTER="${ADAPTER%/}"   # strip trailing slash
OUTPUT="${2:-${ADAPTER}_merged}"

if [ ! -f "$ADAPTER/adapter_config.json" ]; then
    echo "ERROR: no adapter_config.json found in '$ADAPTER'" >&2
    echo "Pass a valid LoRA adapter directory as the first argument." >&2
    exit 1
fi

source "$CONDA_SH"
conda activate "$CONDA_ENV"

export CUDA_HOME=/cm/shared/apps/cuda12.4/toolkit/12.4.1
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# Read the base model path recorded in the adapter config.
BASE="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["base_model_name_or_path"])' "$ADAPTER/adapter_config.json")"

echo "Adapter: $ADAPTER"
echo "Base:    $BASE"
echo "Output:  $OUTPUT"
echo ""

python "$REPO/merge_lora.py" \
    --adapter "$ADAPTER" \
    --base    "$BASE" \
    --output  "$OUTPUT"

echo "Merged model saved to $OUTPUT"
