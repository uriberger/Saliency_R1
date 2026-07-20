#!/bin/bash
# Submit the vLLM-colocated smoke test (run_smoketest_vllm.sh) as a short single-node job.
# Use this when interactive GPUs aren't available.
#
# Usage:
#   NVIDIA_API_KEY=... bash submit_smoketest_vllm.sh
#   DURATION=1 PARTITION=batch_short NVIDIA_API_KEY=... bash submit_smoketest_vllm.sh   # faster queue
#
# After it runs, inspect:
#   outputs/logs/grpo_smoketest_vllm.<jobid>.out         (main log: Stage A + health + training)
#   checkpoint/_smoketest_vllm/sidecar_logs/vllm.log     (vLLM server: Qwen3-VL load / errors)
#   checkpoint/_smoketest_vllm/sidecar_logs/dino.log     (Grounding-DINO server)
set -e

export PATH="/lustre/fs1/portfolios/adlr/projects/adlr_other_infra/release/cluster-interface/21.1_2026-04-15_21-25-57:$PATH"

ACCOUNT=nvr_israel_rlop
PARTITION=${PARTITION:-batch_singlenode}   # single-node (localhost sidecars); override for faster queue
DURATION=${DURATION:-1}
NUM_GPUS=${NUM_GPUS:-8}
REPO=/home/uberger/scratch/research/saliency_r1
CONDA_ENV=${CONDA_ENV:-saliency_r1_qwen3_vllm}
HF_HOME=${HF_HOME:-/home/uberger/scratch/cache/hf_cache}
JOB_NAME=grpo_smoketest_vllm
LOG_ROOT="$REPO/outputs/logs"
mkdir -p "$LOG_ROOT"

if (( NUM_GPUS < 3 )); then echo "ERROR: need >=3 GPUs; got NUM_GPUS=$NUM_GPUS" >&2; exit 1; fi

echo "Job:        $JOB_NAME  ($NUM_GPUS GPUs = 1 DINO + 1 vLLM + $(( NUM_GPUS - 2 )) train)"
echo "Partition:  $PARTITION   Duration: ${DURATION}h   Env: $CONDA_ENV"
echo "Judge key:  $([[ -n "${NVIDIA_API_KEY:-}${OPENAI_API_KEY:-}" ]] && echo '(set)' || echo '(MISSING - the judge reward will crash the run)')"
echo ""

submit_job \
    --account "$ACCOUNT" \
    --partition "$PARTITION" \
    --name "$JOB_NAME" \
    --gpu "$NUM_GPUS" \
    --duration "$DURATION" \
    --outfile "$LOG_ROOT/${JOB_NAME}.%j.out" \
    --logroot "$LOG_ROOT" \
    -c "bash -c '
        export CONDA_ENV=$CONDA_ENV;
        export HF_HOME=$HF_HOME;
        export NUM_GPUS=$NUM_GPUS;
        ${HF_TOKEN:+export HF_TOKEN=$HF_TOKEN;}
        ${NVIDIA_API_KEY:+export NVIDIA_API_KEY=$NVIDIA_API_KEY;}
        ${OPENAI_API_KEY:+export OPENAI_API_KEY=$OPENAI_API_KEY;}
        ${OPENAI_BASE_URL:+export OPENAI_BASE_URL=$OPENAI_BASE_URL;}
        ${JUDGE_MODEL:+export JUDGE_MODEL=$JUDGE_MODEL;}
        cd $REPO;
        bash run_smoketest_vllm.sh
    '"

echo ""
echo "Submitted $JOB_NAME. Watch the main log:"
echo "   tail -f $LOG_ROOT/${JOB_NAME}.<jobid>.out"
echo "and the vLLM sidecar log if generation fails:"
echo "   tail -f $REPO/checkpoint/_smoketest_vllm/sidecar_logs/vllm.log"
