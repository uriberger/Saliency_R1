#!/bin/bash
# Submit the attention-overlap GRPO training for Qwen3-VL-8B as a SINGLE-NODE cluster job.
#
# Unlike the interactive setup (8-GPU train allocation + a separate 1-GPU DINO allocation
# on another machine), this submits ONE job that co-locates everything on one node:
#     GPU 0        Grounding-DINO reward server   (127.0.0.1)
#     GPU 1        vLLM generation server         (127.0.0.1)
#     GPU 2..N-1   GRPO training (DeepSpeed ZeRO-3)
# so the reward server and the trainer are tied to the same allocation and start together
# whenever the job is scheduled. The actual orchestration lives in
# run_grpo_overlap_colocated.sh; this script only submits it.
#
# Usage:
#   WANDB_API_KEY=... NVIDIA_API_KEY=... bash launch_grpo_qwen3_overlap_job.sh              # 8 GPUs
#   ... bash launch_grpo_qwen3_overlap_job.sh --num-gpus 8 --w-overlap 0.2 --token-reduction max
#
# Environment overrides:
#   PARTITION=batch_singlenode   DURATION=4 (hours)   CONDA_ENV=saliency_r1_qwen3_vllm
#   NVIDIA_API_KEY / OPENAI_API_KEY / OPENAI_BASE_URL / JUDGE_MODEL   (LLM-as-judge)
#   WANDB_API_KEY   (omit -> offline)     HF_TOKEN
#
# Any flag not parsed below is forwarded verbatim to run_grpo_overlap_colocated.sh.
set -e

# ADLR cluster-interface tools (submit_job, etc.) on PATH.
# Locate the ADLR submit_job wrapper robustly. The cluster-interface tree is mounted under
# /lustre/fs1 on some hosts and /lustre/fsw on others, so try both; within each, prefer the
# 'latest' symlink, else the newest versioned build. `[ -x ]` follows symlinks, so a broken
# cross-filesystem symlink is skipped automatically.
if ! command -v submit_job >/dev/null 2>&1; then
    for CI_ROOT in \
        /lustre/fs1/portfolios/adlr/projects/adlr_other_infra/release/cluster-interface \
        /lustre/fsw/portfolios/adlr/projects/adlr_other_infra/release/cluster-interface; do
        for CAND in "$CI_ROOT/latest" $(ls -1dt "$CI_ROOT"/*/ 2>/dev/null); do
            if [ -x "${CAND%/}/submit_job" ]; then
                export PATH="${CAND%/}:$PATH"; break 2
            fi
        done
    done
fi
if ! command -v submit_job >/dev/null 2>&1; then
    echo "ERROR: submit_job not found under /lustre/fs1 or /lustre/fsw cluster-interface paths." >&2
    echo "  which submit_job    # is it perhaps provided by a module?" >&2
    exit 1
fi

# ---------- cluster / project constants ----------
ACCOUNT=nvr_israel_rlop
PARTITION=${PARTITION:-batch_singlenode}   # single-node so localhost sidecars are co-located
DURATION=${DURATION:-4}
REPO=/home/uberger/scratch/research/saliency_r1
CONDA_ENV=${CONDA_ENV:-saliency_r1_qwen3_vllm}
HF_HOME=${HF_HOME:-/home/uberger/scratch/cache/hf_cache}

# ---------- parse: capture --num-gpus (needed for --gpu), forward the rest ----------
NUM_GPUS=8
FWD_ARGS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-gpus) NUM_GPUS="$2"; FWD_ARGS="$FWD_ARGS --num-gpus $2"; shift 2 ;;
        *)          FWD_ARGS="$FWD_ARGS $1";                          shift ;;
    esac
done

if (( NUM_GPUS < 3 )); then
    echo "ERROR: need >=3 GPUs (1 DINO + 1 vLLM + >=1 training); got --num-gpus $NUM_GPUS" >&2
    exit 1
fi
if (( NUM_GPUS > 8 )); then
    echo "ERROR: this launcher is single-node (max 8 GPUs). localhost DINO/vLLM sidecars" >&2
    echo "       cannot serve a second node. For >8 GPUs a multi-node design is required." >&2
    exit 1
fi

JOB_NAME="grpo_overlap_qwen3_vllm_${NUM_GPUS}gpu"
LOG_ROOT="$REPO/outputs/logs"
mkdir -p "$LOG_ROOT"

echo "Job name:   $JOB_NAME"
echo "GPUs:       $NUM_GPUS (1 DINO + 1 vLLM + $(( NUM_GPUS - 2 )) train), single node"
echo "Partition:  $PARTITION  (duration ${DURATION}h, auto-resume)"
echo "Conda env:  $CONDA_ENV"
echo "Judge key:  $([[ -n "${NVIDIA_API_KEY:-}${OPENAI_API_KEY:-}" ]] && echo '(set)' || echo '(MISSING - openai_reward will fail)')"
echo "WandB:      $([[ -n "${WANDB_API_KEY:-}" ]] && echo '(online)' || echo '(offline)')"
echo "Forwarded:  $FWD_ARGS"
echo ""

# Secrets/config are baked into the job command (submit_job does not propagate the
# submitter's shell env). run_grpo_overlap_colocated.sh does the conda/CUDA/WANDB setup.
submit_job \
    --account "$ACCOUNT" \
    --partition "$PARTITION" \
    --name "$JOB_NAME" \
    --gpu "$NUM_GPUS" \
    --duration "$DURATION" \
    --autoresume_uninstrumented \
    --outfile "$LOG_ROOT/${JOB_NAME}.%j.out" \
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
        cd $REPO;
        bash run_grpo_overlap_colocated.sh $FWD_ARGS
    '"

echo "Submitted $JOB_NAME"
