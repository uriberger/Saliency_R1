#!/usr/bin/env bash
# Submit register_probe.py's `extract` stage to SLURM, on ONE GPU.
#
#   bash launch_register_probe_job.sh --out-dir outputs/register_probe/pope \
#       [--name register-probe] [--duration 1] [-- <forwarded to the python>]
#
# Only `extract` needs a GPU, and barely: it is one prefill per picture over 500 POPE
# pictures with no generation, no detector and no judge. One GPU rather than eight,
# because there is nothing to shard and the queue for one is shorter.
#
# `labels`, `probe` and `report` are CPU and run on the login node in under two minutes
# between them -- do not submit those.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

NAME=""
OUT_DIR=""
STAGE="extract"
DURATION=1
GPUS=""
DRY_RUN=0
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)   DRY_RUN=1;       shift   ;;
        --name)      NAME="$2";       shift 2 ;;
        --out-dir)   OUT_DIR="$2";    shift 2 ;;
        --stage)     STAGE="$2";      shift 2 ;;
        --duration)  DURATION="$2";   shift 2 ;;
        --gpus)      GPUS="$2";       shift 2 ;;
        --partition) PARTITION="$2";  shift 2 ;;
        --)          shift; EXTRA+=("$@"); break ;;
        *)           EXTRA+=("$1"); shift ;;
    esac
done

case "$STAGE" in
    extract) GPUS=${GPUS:-1} ;;
    # `probe` is pure CPU and takes about an hour: 16 classes x 11 arm-runs x 5 folds of a
    # regularisation search. That is too much for a contended login node, and this account
    # has 50 GPU-free nodes in cpu_short it can have immediately.
    probe)   GPUS=${GPUS:-0}; PARTITION=${PARTITION:-cpu_short} ;;
    *)       echo "ERROR: --stage must be extract or probe (labels and report are" \
                  "seconds on the login node)." >&2; exit 2 ;;
esac
NAME=${NAME:-register-probe-$STAGE}

[[ -n "$OUT_DIR" ]] || { echo "ERROR: --out-dir is required." >&2; exit 2; }
[[ -f "$OUT_DIR/labels.jsonl" ]] || {
    echo "ERROR: $OUT_DIR/labels.jsonl not found -- run the labels stage first:" >&2
    echo "  python register_probe.py labels --out-dir $OUT_DIR" >&2; exit 2; }

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
    printf 'python %q/register_probe.py %q --out-dir %q' "$REPO" "$STAGE" "$OUT_DIR"
    for a in ${EXTRA[@]+"${EXTRA[@]}"}; do printf ' %q' "$a"; done
    printf '\n'
} > "$RUNNER"
chmod +x "$RUNNER"

echo "=========================================================================="
echo "Job     : $NAME   ($ACCOUNT, $PARTITION, ${DURATION}h, stage=$STAGE, ${GPUS} GPU)"
echo "Out dir : $OUT_DIR"
echo "Extra   : ${EXTRA[*]:-(none)}"
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
