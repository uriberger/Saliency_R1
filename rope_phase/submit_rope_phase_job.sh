#!/usr/bin/env bash
# Submit an E0 RoPE phase lock-in run as a single-GPU batch job.
#
# Self-contained: nothing outside this directory is sourced.  The two site-specific
# bits -- finding the ADLR submit_job wrapper and picking a partition that exists --
# are resolved by asking the machine, so a different cluster needs no edit.  On a
# host with no submit_job at all, run rope_phase_probe.py directly instead.
#
#   ROPE_PHASE_DATASET=/path/to/dataset bash submit_rope_phase_job.sh
#   N_SAMPLES=256 DURATION=2 OUT_DIR=$PWD/outputs/run2 bash submit_rope_phase_job.sh
#   bash submit_rope_phase_job.sh --base-model Qwen/Qwen2.5-VL-7B-Instruct
#
# Anything not parsed here is forwarded verbatim to rope_phase_probe.py.
#
# Use a FRESH --out-dir per run.  The report refuses to merge shards scanned with
# different n_samples/num_shards, and an existing scan/shardNN.npz is skipped, so
# reusing a directory silently does nothing.
#
# NOTE the scan writes its npz only when the shard finishes, so a job killed at the
# wall clock loses everything.  DURATION defaults to 2h against ~5 s/case.
set -e

HERE="$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")" && pwd)"

find_submit_job() {
    command -v submit_job >/dev/null 2>&1 && return 0
    local root cand
    for root in \
        /lustre/fs1/portfolios/adlr/projects/adlr_other_infra/release/cluster-interface \
        /lustre/fsw/portfolios/adlr/projects/adlr_other_infra/release/cluster-interface; do
        for cand in "$root/latest" $(ls -1dt "$root"/*/ 2>/dev/null); do
            if [ -x "${cand%/}/submit_job" ]; then export PATH="${cand%/}:$PATH"; return 0; fi
        done
    done
    return 1
}

# First named partition that actually exists here, else the cluster default: a
# partition name that does not exist is not a clean failure, the job just never runs.
pick_partition() {
    local avail want default
    avail=$(sinfo -h -o '%P' 2>/dev/null | tr -d '*' | sort -u)
    [ -z "$avail" ] && { echo "${1:-batch}"; return 0; }
    for want in "$@"; do
        printf '%s\n' "$avail" | grep -qx -- "$want" && { echo "$want"; return 0; }
    done
    default=$(sinfo -h -o '%P' 2>/dev/null | grep -m1 '\*' | tr -d '*')
    echo "${default:-$(printf '%s\n' "$avail" | head -1)}"
}

find_submit_job || { echo "ERROR: submit_job not found; run rope_phase_probe.py directly." >&2; exit 1; }

ACCOUNT=${ACCOUNT:-nvr_israel_rlop}
PARTITION=${PARTITION:-$(pick_partition batch_singlenode batch_long batch)}
DURATION=${DURATION:-2}
CONDA_SH=${CONDA_SH:-/home/uberger/scratch/miniconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-saliency_r1_qwen3_vllm}   # needs transformers with qwen*_vl + peft
HF_HOME=${HF_HOME:-/home/uberger/scratch/cache/hf_cache}

N_SAMPLES=${N_SAMPLES:-32}
OUT_DIR=${OUT_DIR:-$HERE/outputs/e0}
JOB_NAME=${JOB_NAME:-rope_phase_e0}
LOG_ROOT=${LOG_ROOT:-$HERE/outputs/logs}
DATASET=${ROPE_PHASE_DATASET:-}
EXTRA_ARGS="$*"

[[ -n "$DATASET" || "$EXTRA_ARGS" == *--dataset* ]] || {
    echo "ERROR: set ROPE_PHASE_DATASET or pass --dataset" >&2; exit 2; }

mkdir -p "$LOG_ROOT" "$OUT_DIR"

echo "Project    : $HERE"
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
        ${DATASET:+export ROPE_PHASE_DATASET=$DATASET;}
        cd $HERE;
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader;
        python -u rope_phase_probe.py --stage scan \
            --out-dir $OUT_DIR --shard 0 --num-shards 1 \
            --n-samples $N_SAMPLES --device cuda:0 $EXTRA_ARGS;
        python -u rope_phase_probe.py --stage report --out-dir $OUT_DIR;
    '"
