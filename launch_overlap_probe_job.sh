#!/usr/bin/env bash
# Submit launch_overlap_probe.sh to SLURM. The launcher it wraps assumes it is already
# ON a node with GPUs -- it fans one shard per visible device and waits -- so every probe
# so far was started by hand from inside an allocation. This is the `_job.sh` half, the
# same split as launch_grpo_qwen3_overlap_job.sh / --direct.
#
#   bash launch_overlap_probe_job.sh --name <jobname> \
#       --trained-adapter NAME=/path/to/adapter[,NAME=/path...] \
#       [--gpus 8] [--duration 1] [--n-samples 30] [--out-dir <dir>] \
#       [-- <anything else, forwarded verbatim to launch_overlap_probe.sh>]
#
# Runtime reference: one model, 30 samples x 8 generations on 8 GPUs is ~10 minutes
# wall clock including the model load (outputs/overlap_probe/20260809-020113-newmodel).
# A 1-hour allocation is therefore generous and buys batch_short, which starts in
# about a minute where the 4-hour pools take hours -- see cluster_env.sh.
#
# Everything after `--` goes to the inner launcher untouched, which is where the
# probe's own flags live (--dataset, --split, --skip-base, --map, --overlap-metric...).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

NAME=""
ADAPTERS=""
GPUS=8
DURATION=1
N_SAMPLES=30
OUT_DIR=""
DRY_RUN=0
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)          DRY_RUN=1;       shift   ;;
        --name)             NAME="$2";       shift 2 ;;
        --trained-adapter)  ADAPTERS="$2";   shift 2 ;;
        --gpus)             GPUS="$2";       shift 2 ;;
        --duration)         DURATION="$2";   shift 2 ;;
        --n-samples)        N_SAMPLES="$2";  shift 2 ;;
        --out-dir)          OUT_DIR="$2";    shift 2 ;;
        --)                 shift; EXTRA+=("$@"); break ;;
        *)                  EXTRA+=("$1");   shift ;;
    esac
done

[[ -n "$NAME" ]] || { echo "ERROR: --name is required (it names the job and the log)." >&2; exit 2; }
OUT_DIR=${OUT_DIR:-$REPO/outputs/overlap_probe/$NAME}

# Partition list, filtered for what this cluster actually has and what a $DURATION-hour
# job is eligible for. Same helper every other launcher here uses.
# shellcheck source=/dev/null
source "$REPO/cluster_env.sh"
PARTITION=${PARTITION:-$(SR1_JOB_HOURS=$DURATION sr1_pick_partition)}
ACCOUNT=${ACCOUNT:-nvr_israel_rlop}
CONDA_ENV=${CONDA_ENV:-saliency_r1_qwen3_vllm}

if ! command -v submit_job >/dev/null 2>&1; then
    for CI_ROOT in \
        /lustre/fs1/portfolios/adlr/projects/adlr_other_infra/release/cluster-interface \
        /lustre/fsw/portfolios/adlr/projects/adlr_other_infra/release/cluster-interface; do
        for CAND in "$CI_ROOT/latest" $(ls -1dt "$CI_ROOT"/*/ 2>/dev/null); do
            if [ -x "${CAND%/}/submit_job" ]; then export PATH="${CAND%/}:$PATH"; break 2; fi
        done
    done
fi
command -v submit_job >/dev/null 2>&1 || [[ $DRY_RUN -eq 1 ]] || {
    echo "ERROR: submit_job not found under the cluster-interface paths." >&2; exit 1; }

LOG_ROOT="$REPO/outputs/logs"
mkdir -p "$LOG_ROOT" "$OUT_DIR"

# The inner command as a file rather than a quoted -c string: the probe flags are
# themselves comma- and equals-separated (NAME=PATH,NAME=PATH), and nesting those inside
# `bash -c '...'` is how a label silently loses its path.
RUNNER="$LOG_ROOT/$NAME.runner.sh"
{
    echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    echo "export CONDA_ENV=$CONDA_ENV"
    echo "export HF_HOME=${HF_HOME:-/home/uberger/scratch/cache/hf_cache}"
    echo "export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}"
    echo "export OVERLAP_STEPS_CKPT=${OVERLAP_STEPS_CKPT:-$REPO/checkpoint/steps_classifier/best}"
    printf 'bash %q/launch_overlap_probe.sh --n-samples %q --gpus %q --out-dir %q' \
        "$REPO" "$N_SAMPLES" "$GPUS" "$OUT_DIR"
    [[ -n "$ADAPTERS" ]] && printf ' --trained-adapter %q' "$ADAPTERS"
    for a in ${EXTRA[@]+"${EXTRA[@]}"}; do printf ' %q' "$a"; done
    echo
} > "$RUNNER"
chmod +x "$RUNNER"

echo "=========================================================================="
echo "Job       : $NAME   ($ACCOUNT, $PARTITION, ${DURATION}h, ${GPUS} GPU)"
echo "Out dir   : $OUT_DIR"
echo "Adapters  : ${ADAPTERS:-(base only)}"
echo "Extra     : ${EXTRA[*]:-(none)}"
echo "Runner    : $RUNNER"
echo "=========================================================================="
cat "$RUNNER"
echo "=========================================================================="

[[ $DRY_RUN -eq 1 ]] && { echo "[dry-run] not submitting."; exit 0; }

submit_job \
    --account "$ACCOUNT" \
    --partition "$PARTITION" \
    --name "$NAME" \
    --gpu "$GPUS" \
    --duration "$DURATION" \
    --outfile "$LOG_ROOT/$NAME.%j.out" \
    --logroot "$LOG_ROOT" \
    -c "bash $RUNNER"
