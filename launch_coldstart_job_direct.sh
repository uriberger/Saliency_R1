#!/bin/bash
# Run Saliency-R1 cold-start SFT (Qwen3-VL-8B-Instruct) directly on the current
# node (no submit_job/SLURM submission). Same arg parsing and env setup as
# launch_coldstart_job.sh; use this when you already have an interactive GPU
# allocation (salloc) and just want to launch in-place.
#
# Usage (from a GPU node with the allocation active):
#   bash launch_coldstart_job_direct.sh
#   bash launch_coldstart_job_direct.sh --num-gpus 4
#   WANDB_API_KEY=... bash launch_coldstart_job_direct.sh
#   bash launch_coldstart_job_direct.sh --config /path/other.yaml
#
# Unlike launch_coldstart_job.sh this does NOT go through submit_job: no
# auto-resume wall-limit handling. It still resumes from the latest checkpoint
# in --output-dir if one exists, so rerunning after a crash picks up where it
# left off.
#
# Any flag not parsed below is forwarded verbatim as a LLaMA-Factory (OmegaConf)
# override, e.g.:  bash launch_coldstart_job_direct.sh per_device_train_batch_size=4
set -e

REPO=/home/uberger/scratch/research/saliency_r1
LF_DIR=/home/uberger/scratch/research/LLaMA-Factory
CONDA_SH=/home/uberger/scratch/miniconda3/etc/profile.d/conda.sh
CONDA_ENV=sr1_coldstart
HF_HOME=${HF_HOME:-/home/uberger/scratch/cache/hf_cache}

# ---------- training defaults ----------
CONFIG="$REPO/train/cold_start/qwen3_vl_8b_instruct_sft/train.yaml"
NUM_GPUS=8
OUTPUT_DIR=""
EXTRA_ARGS=""

# ---------- parse args ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-gpus)    NUM_GPUS="$2";   shift 2 ;;
        --config)      CONFIG="$2";     shift 2 ;;
        --output-dir)  OUTPUT_DIR="$2"; shift 2 ;;
        --wandb-api-key) WANDB_API_KEY="$2"; shift 2 ;;
        --hf-token)    HF_TOKEN="$2";   shift 2 ;;
        *)             EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR=$(grep -E '^output_dir:' "$CONFIG" | head -1 | sed 's/^output_dir:[[:space:]]*//')
fi

JOB_NAME="coldstart_qwen3_vl_8b_instruct_sft"
mkdir -p "$OUTPUT_DIR"

echo "Config:     $CONFIG"
echo "GPUs:       $NUM_GPUS"
echo "Output dir: $OUTPUT_DIR"
echo "WandB:      $([[ -n "${WANDB_API_KEY:-}" ]] && echo '(online)' || echo '(offline)')"
[[ -n "$EXTRA_ARGS" ]] && echo "Overrides:  $EXTRA_ARGS"
echo ""

source "$CONDA_SH"
conda activate "$CONDA_ENV"

export CUDA_HOME=/cm/shared/apps/cuda12.4/toolkit/12.4.1
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
bash "$REPO/check_cuda_home.sh" || exit 1

export HF_HOME
export HF_HUB_OFFLINE=1
export HF_TOKEN=${HF_TOKEN:-}
export WANDB_API_KEY=${WANDB_API_KEY:-}
[ -z "$WANDB_API_KEY" ] && export WANDB_MODE=offline
export WANDB_PROJECT=vlm_reasoning
export WANDB_ENTITY=nvr-israel
export WANDB_RUN_ID=${WANDB_RUN_ID:-coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5}
export WANDB_RESUME=${WANDB_RESUME:-allow}

# DeepSpeed under LLaMA-Factory must be launched via torchrun.
export FORCE_TORCHRUN=1
export NPROC_PER_NODE=$NUM_GPUS

cd "$LF_DIR"

RESUME=""
LATEST=$(ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -1)
[ -n "$LATEST" ] && RESUME="resume_from_checkpoint=$LATEST"

llamafactory-cli train "$CONFIG" \
    output_dir="$OUTPUT_DIR" \
    $RESUME \
    $EXTRA_ARGS

echo "Finished $JOB_NAME"
