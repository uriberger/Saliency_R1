#!/bin/bash
# Serve the batched Grounding-DINO endpoint for the attention-overlap GRPO reward.
# Run this on a GPU OUTSIDE the 8-GPU training allocation (spare card / MIG slice) so the
# ZeRO-3 policy keeps all 8 GPUs. Mirrors the serve_nvidia_model.sh pattern.
#
# Usage (on the DINO host):
#   bash serve_grounding_dino.sh                       # port 8100, GPU 0
#   bash serve_grounding_dino.sh --port 8100 --gpu 0
#
# Then launch training (on the training node) with:
#   bash launch_grpo_qwen3_overlap_job_direct.sh --num-gpus 8 \
#        --dino-api-base http://<this-host>:8100
set -e

REPO=/home/uberger/scratch/research/saliency_r1
CONDA_SH=/home/uberger/scratch/miniconda3/etc/profile.d/conda.sh
CONDA_ENV=saliency_r1_qwen3
HF_HOME=${HF_HOME:-/home/uberger/scratch/cache/hf_cache}

HOST=0.0.0.0
PORT=8100
GPU=0
SERVER_BATCH=${DINO_SERVER_BATCH:-32}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)          HOST="$2";          shift 2 ;;
        --port)          PORT="$2";          shift 2 ;;
        --gpu)           GPU="$2";           shift 2 ;;
        --server-batch)  SERVER_BATCH="$2";  shift 2 ;;
        --hf-token)      HF_TOKEN="$2";      shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

source "$CONDA_SH"
conda activate "$CONDA_ENV"

export CUDA_HOME=/cm/shared/apps/cuda12.4/toolkit/12.4.1
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HOME
export HF_HUB_OFFLINE=1
export HF_TOKEN=${HF_TOKEN:-}
export DINO_SERVER_BATCH="$SERVER_BATCH"

HOSTNAME_FQDN=$(hostname -f 2>/dev/null || hostname)
echo "Serving Grounding-DINO"
echo "  host/port:    $HOST:$PORT   (reachable at http://$HOSTNAME_FQDN:$PORT)"
echo "  GPU:          $GPU   server-batch=$SERVER_BATCH"
echo "  --dino-api-base http://$HOSTNAME_FQDN:$PORT"
echo ""

cd "$REPO"
exec python serve_grounding_dino.py --host "$HOST" --port "$PORT"
