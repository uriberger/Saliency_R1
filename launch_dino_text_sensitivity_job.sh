#!/usr/bin/env bash
# Submit dino_text_sensitivity.py to SLURM. Same `_job.sh` split as
# launch_overlap_probe_job.sh, but far smaller: this probe loads Grounding-DINO and
# nothing else -- no VLM, no vLLM, no generation -- so ONE GPU is the whole request, and
# an hour is generous for the ~6k groundings a full probe_merged.json costs.
#
#   bash launch_dino_text_sensitivity_job.sh --name <jobname> \
#       --merged outputs/overlap_probe/<run>/probe_merged.json \
#       [--models base_coldstart] [--baseline outputs/step_box_similarity/mean_in/report.json] \
#       [--out-dir <dir>] [--duration 1] [--dry-run] \
#       [-- <anything else, forwarded verbatim to dino_text_sensitivity.py>]
#
# One GPU rather than eight is deliberate and not just thrift: the account's GPU cap is
# what this queues behind, so the smallest request that can run is the one that starts
# soonest. It will still queue while the trainers hold the cap -- poll, do not resubmit.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

NAME=""
MERGED=""
MODELS="base_coldstart"
BASELINE=""
OUT_DIR=""
DURATION=1
GPUS=1
DRY_RUN=0
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)   DRY_RUN=1;      shift   ;;
        --name)      NAME="$2";      shift 2 ;;
        --merged)    MERGED="$2";    shift 2 ;;
        --models)    MODELS="$2";    shift 2 ;;
        --baseline)  BASELINE="$2";  shift 2 ;;
        --out-dir)   OUT_DIR="$2";   shift 2 ;;
        --duration)  DURATION="$2";  shift 2 ;;
        --gpus)      GPUS="$2";      shift 2 ;;
        --)          shift; EXTRA+=("$@"); break ;;
        *)           EXTRA+=("$1");  shift ;;
    esac
done

[[ -n "$NAME"   ]] || { echo "ERROR: --name is required (it names the job and the log)." >&2; exit 2; }
[[ -n "$MERGED" ]] || { echo "ERROR: --merged is required." >&2; exit 2; }
[[ -f "$MERGED" ]] || { echo "ERROR: --merged $MERGED does not exist." >&2; exit 2; }
[[ -z "$BASELINE" || -f "$BASELINE" ]] || { echo "ERROR: --baseline $BASELINE does not exist." >&2; exit 2; }
OUT_DIR=${OUT_DIR:-$REPO/outputs/dino_text_sensitivity/$NAME}

# shellcheck source=/dev/null
source "$REPO/cluster_env.sh"
PARTITION=${PARTITION:-$(SR1_JOB_HOURS=$DURATION sr1_pick_partition)}
ACCOUNT=${ACCOUNT:-nvr_israel_rlop}
CONDA_ENV=${CONDA_ENV:-saliency_r1_qwen3_vllm}

# launch_overlap_probe.sh's CONDA_ROOT default names a path that does not exist on this
# cluster, and this probe has no inner launcher to inherit one from, so resolve it here
# against the roots that do exist and fail loudly rather than sourcing nothing.
CONDA_ROOT=${CONDA_ROOT:-}
if [[ -z "$CONDA_ROOT" ]]; then
    for CAND in /home/uberger/scratch/miniconda3 \
                /lustre/fs12/portfolios/nvr/projects/nvr_israel_rlop/users/uberger/research/miniforge3; do
        [[ -f "$CAND/etc/profile.d/conda.sh" ]] && { CONDA_ROOT="$CAND"; break; }
    done
fi
[[ -n "$CONDA_ROOT" ]] || { echo "ERROR: no conda root found." >&2; exit 1; }

sr1_find_submit_job || [[ $DRY_RUN -eq 1 ]] || {
    echo "ERROR: submit_job not found under the cluster-interface paths." >&2; exit 1; }

LOG_ROOT="$REPO/outputs/logs"
mkdir -p "$LOG_ROOT" "$OUT_DIR"

RUNNER="$LOG_ROOT/$NAME.runner.sh"
{
    echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    echo "set +u"
    printf 'source %q/etc/profile.d/conda.sh\n' "$CONDA_ROOT"
    printf 'conda activate %q\n' "$CONDA_ENV"
    echo "set -u"
    echo "export HF_HOME=${HF_HOME:-/home/uberger/scratch/cache/hf_cache}"
    echo "export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}"
    echo "export TOKENIZERS_PARALLELISM=false"
    printf 'cd %q\n' "$REPO"
    printf 'python dino_text_sensitivity.py %q --models %q --out-dir %q' \
        "$MERGED" "$MODELS" "$OUT_DIR"
    [[ -n "$BASELINE" ]] && printf ' --baseline %q' "$BASELINE"
    for a in ${EXTRA[@]+"${EXTRA[@]}"}; do printf ' %q' "$a"; done
    echo
} > "$RUNNER"
chmod +x "$RUNNER"

echo "=========================================================================="
echo "Job       : $NAME   ($ACCOUNT, $PARTITION, ${DURATION}h, ${GPUS} GPU)"
echo "Merged    : $MERGED"
echo "Models    : $MODELS"
echo "Baseline  : ${BASELINE:-(none)}"
echo "Out dir   : $OUT_DIR"
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
