#!/usr/bin/env bash
# Submit sink_observe_boxes.py to SLURM, on ONE GPU, over one or more scan directories.
#
#   bash launch_sink_observe_boxes_job.sh --name slxb-dino-c \
#       --dirs outputs/sink_location/xmodel/box_nemotron [--duration 2] \
#       [-- <forwarded to the python>]
#
# THE THIRD LEG, as a job. Grounding each observe step of an already-stored completion is
# a separate pass from the scan on purpose (see the module docstring), and it is small:
# no VLM is loaded, only the detector, so one GPU is the whole allocation and there is
# nothing to shard. 1,800 pictures at ~13 observe steps each run in well under an hour.
#
# `--dirs` takes a comma-separated list and they are grounded IN ORDER inside the one
# allocation, because the detector load dominates a single directory's cost and paying it
# once for four models beats queueing four jobs.
#
# The pass resumes: it skips any picture already in the directory's observe_boxes.jsonl,
# so a job that runs out of wall clock is re-submitted verbatim rather than restarted.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

NAME="sink-observe-boxes"
DIRS=""
DURATION=2
DRY_RUN=0
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)  DRY_RUN=1;     shift   ;;
        --name)     NAME="$2";     shift 2 ;;
        --dirs)     DIRS="$2";     shift 2 ;;
        --duration) DURATION="$2"; shift 2 ;;
        --)         shift; EXTRA+=("$@"); break ;;
        *)          EXTRA+=("$1"); shift ;;
    esac
done

[[ -n "$DIRS" ]] || { echo "ERROR: --dirs is required (comma-separated scan dirs)." >&2; exit 2; }
IFS=',' read -r -a DIR_LIST <<< "$DIRS"
for d in "${DIR_LIST[@]}"; do
    [[ -d "$d" ]] || { echo "ERROR: not a directory: $d" >&2; exit 2; }
    ls "$d"/scan_shard*.jsonl >/dev/null 2>&1 || {
        echo "ERROR: $d holds no scan results to ground." >&2; exit 2; }
done

# shellcheck source=/dev/null
source "$REPO/cluster_env.sh"
PARTITION=${PARTITION:-$(SR1_JOB_HOURS=$DURATION sr1_pick_partition)}
ACCOUNT=${ACCOUNT:-nvr_israel_rlop}
CONDA_ENV=${CONDA_ENV:-saliency_r1_qwen3_vllm}

sr1_find_submit_job || [[ $DRY_RUN -eq 1 ]] || {
    echo "ERROR: submit_job not found under the cluster-interface paths." >&2; exit 1; }

LOG_ROOT="$REPO/outputs/logs"
mkdir -p "$LOG_ROOT"

RUNNER="$LOG_ROOT/$NAME.runner.sh"
{
    echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    printf 'cd %q\n' "$REPO"
    echo 'source "$(conda info --base)/etc/profile.d/conda.sh"'
    echo "conda activate $CONDA_ENV"
    echo "export HF_HOME=${HF_HOME:-/home/uberger/scratch/cache/hf_cache}"
    echo "export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}"
    echo "export TOKENIZERS_PARALLELISM=false"
    for d in "${DIR_LIST[@]}"; do
        printf 'echo "################ %s"\n' "$d"
        printf 'python %q/sink_observe_boxes.py --scan-dir %q' "$REPO" "$d"
        for a in ${EXTRA[@]+"${EXTRA[@]}"}; do printf ' %q' "$a"; done
        printf '\n'
    done
} > "$RUNNER"
chmod +x "$RUNNER"

echo "=========================================================================="
echo "Job     : $NAME   ($ACCOUNT, $PARTITION, ${DURATION}h, 1 GPU)"
echo "Dirs    : ${DIR_LIST[*]}"
echo "Extra   : ${EXTRA[*]:-(none)}"
echo "=========================================================================="
cat "$RUNNER"
echo "=========================================================================="

[[ $DRY_RUN -eq 1 ]] && { echo "[dry-run] not submitting."; exit 0; }

submit_job \
    --account "$ACCOUNT" \
    --partition "$PARTITION" \
    --name "$NAME" \
    --gpu 1 \
    --duration "$DURATION" \
    --outfile "$LOG_ROOT/$NAME.%j.out" \
    --logroot "$LOG_ROOT" \
    -c "bash $RUNNER"
