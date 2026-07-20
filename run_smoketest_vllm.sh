#!/bin/bash
# Fail-fast smoke test for the vLLM-colocated overlap GRPO pipeline.
#   Stage A: verify torch 2.8.0+cu128 actually runs a CUDA kernel on THIS node's driver
#            (cheap ~10s fail if the cu126->cu128 bump is incompatible with the driver).
#   Stage B: run run_grpo_overlap_colocated.sh for 3 steps -> exercises DINO + vLLM server
#            + weight-sync + Qwen3-VL image generation + saliency reforward + overlap reward.
# Runs on an 8-GPU node -- directly, or (usually) via submit_smoketest_vllm.sh.
set -euo pipefail

REPO=/home/uberger/scratch/research/saliency_r1
CONDA_SH=/home/uberger/scratch/miniconda3/etc/profile.d/conda.sh
CONDA_ENV=${CONDA_ENV:-saliency_r1_qwen3_vllm}
NUM_GPUS=${NUM_GPUS:-8}

source "$CONDA_SH"
conda activate "$CONDA_ENV"
export CUDA_HOME=${CUDA_HOME:-/cm/shared/apps/cuda12.4/toolkit/12.4.1}
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export WANDB_MODE=offline     # throwaway run -- do not pollute wandb

echo "==================== STAGE A: torch 2.8 / cu128 driver check ===================="
python - <<'PY'
import sys, torch
print("torch:", torch.__version__, "| cuda build:", torch.version.cuda)
if not torch.cuda.is_available():
    sys.exit("DRIVER_CHECK_FAILED: torch.cuda.is_available() is False (cu128 wheels vs node driver).")
print("device:", torch.cuda.get_device_name(0), "| visible GPUs:", torch.cuda.device_count())
# Force a real kernel launch -- this is what actually fails on a driver/runtime mismatch.
x = torch.randn(2048, 2048, device="cuda")
val = (x @ x).sum().item()
print("cuda matmul ok, checksum finite:", val == val)
print("STAGE_A_OK")
PY

echo ""
echo "==================== STAGE B: 3-step colocated pipeline ===================="
cd "$REPO"
exec bash run_grpo_overlap_colocated.sh \
    --num-gpus "$NUM_GPUS" \
    --output-dir "$REPO/checkpoint/_smoketest_vllm" \
    --max_steps 3 \
    "$@"
