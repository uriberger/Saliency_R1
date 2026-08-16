#!/bin/bash
# Submit the E0 RoPE phase lock-in pilot as a single-GPU batch job.
#
# The other probe launchers (launch_flow_correlation.sh and friends) assume an
# already-allocated interactive node and fan out over its GPUs with
# CUDA_VISIBLE_DEVICES.  This one goes through submit_job instead, because the
# pilot is one GPU for a few minutes and does not need an interactive shell.
#
#   bash submit_rope_phase_job.sh                       # 32 cases, Qwen3-VL-8B
#   N_SAMPLES=256 DURATION=3 bash submit_rope_phase_job.sh
#   OUT_DIR=.../e0_qwen25 bash submit_rope_phase_job.sh --base-model Qwen/Qwen2.5-VL-7B-Instruct
#
# Anything not parsed here is forwarded verbatim to rope_phase_probe.py.
#
# NOTE the scan writes its npz only when the shard finishes, so a job killed at
# the wall clock loses everything.  DURATION defaults to 2h against an estimated
# ~10 min for 32 cases; raise it before raising N_SAMPLES.
set -e

REPO="$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")" && pwd)"
source "$REPO/cluster_env.sh"
sr1_find_submit_job || { echo "ERROR: submit_job not found under the cluster-interface paths." >&2; exit 1; }

ACCOUNT=nvr_israel_rlop
PARTITION=${PARTITION:-$(sr1_pick_partition batch_singlenode batch_long batch)}
DURATION=${DURATION:-2}
CONDA_SH=/home/uberger/scratch/miniconda3/etc/profile.d/conda.sh
# The same env the other probes use; overlap_probe's import chain is known to
# resolve in it, which is what the scan stage needs.
CONDA_ENV=${CONDA_ENV:-saliency_r1_qwen3_vllm}
HF_HOME=${HF_HOME:-/home/uberger/scratch/cache/hf_cache}

N_SAMPLES=${N_SAMPLES:-32}
OUT_DIR=${OUT_DIR:-$REPO/outputs/rope_phase/e0_pilot}
JOB_NAME=${JOB_NAME:-rope_phase_e0}
LOG_ROOT="$REPO/outputs/logs"
EXTRA_ARGS="$*"

mkdir -p "$LOG_ROOT" "$OUT_DIR"

echo "Repo       : $REPO"
echo "Out dir    : $OUT_DIR"
echo "Cases      : $N_SAMPLES (1 shard, 1 GPU)"
echo "Partition  : $PARTITION (duration ${DURATION}h)"
echo "Env        : $CONDA_ENV"
[[ -n "$EXTRA_ARGS" ]] && echo "Extra args : $EXTRA_ARGS"
echo ""

submit_job \
    --account "$ACCOUNT" \
    --partition "$PARTITION" \
    --name "$JOB_NAME" \
    --gpu 1 \
    --duration "$DURATION" \
    --outfile "$LOG_ROOT/${JOB_NAME}.%j.out" \
    --logroot "$LOG_ROOT" \
    -c "bash -c '
        set -e;
        source $CONDA_SH;
        conda activate $CONDA_ENV;
        export HF_HOME=$HF_HOME;
        export HF_HUB_OFFLINE=1;
        export TOKENIZERS_PARALLELISM=false;
        cd $REPO;
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader;
        python -u rope_phase_probe.py --stage scan \
            --out-dir $OUT_DIR --shard 0 --num-shards 1 \
            --n-samples $N_SAMPLES --device cuda:0 $EXTRA_ARGS;
        python -u rope_phase_probe.py --stage report --out-dir $OUT_DIR;
    '"
