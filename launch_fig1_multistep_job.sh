#!/usr/bin/env bash
# Submit fig1_multistep.py to SLURM, on ONE GPU.
#
#   bash launch_fig1_multistep_job.sh --run-dir outputs/saliency_viz/fig1ms-valnat \
#       [--run-dir <another scan root>] [--name fig1-multistep] [--duration 1] \
#       [--out <json>] [-- <forwarded to the python>]
#
# Same shape and the same reason as launch_fig1_step_referent_job.sh: the script is
# CPU-capable, but Grounding-DINO over ~1,500 step sentences on a contended login node is
# tens of minutes against seconds on one GPU. One GPU, not eight -- a single batched
# detector pass is the whole job, there is nothing to shard.
#
# --run-dir is repeatable and is forwarded as repeated --run-dir, so several scans (one
# per validation set) are pooled into one search and one crossover rate.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

NAME="fig1-multistep"
RUN_DIRS=()
DURATION=1
OUT=""
DRY_RUN=0
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)  DRY_RUN=1;         shift   ;;
        --name)     NAME="$2";         shift 2 ;;
        --run-dir)  RUN_DIRS+=("$2");  shift 2 ;;
        --duration) DURATION="$2";     shift 2 ;;
        --out)      OUT="$2";          shift 2 ;;
        --)         shift; EXTRA+=("$@"); break ;;
        *)          EXTRA+=("$1");     shift ;;
    esac
done

[[ ${#RUN_DIRS[@]} -gt 0 ]] || {
    echo "ERROR: at least one --run-dir is required (a saliency_viz output root)." >&2; exit 2; }
declare -a ABS=()
for d in "${RUN_DIRS[@]}"; do
    [[ -d "$d" ]] || { echo "ERROR: --run-dir not found: $d" >&2; exit 2; }
    ABS+=("$(cd "$d" && pwd)")
done
OUT=${OUT:-$REPO/outputs/fig1-multistep/$NAME.json}

# shellcheck source=/dev/null
source "$REPO/cluster_env.sh"
PARTITION=${PARTITION:-$(SR1_JOB_HOURS=$DURATION sr1_pick_partition)}
ACCOUNT=${ACCOUNT:-nvr_israel_rlop}
CONDA_ENV=${CONDA_ENV:-saliency_r1_qwen3_vllm}

sr1_find_submit_job || [[ $DRY_RUN -eq 1 ]] || {
    echo "ERROR: submit_job not found under the cluster-interface paths." >&2; exit 1; }

LOG_ROOT="$REPO/outputs/logs"
mkdir -p "$LOG_ROOT" "$(dirname "$OUT")"

# A file rather than a quoted `bash -c` string: the forwarded flags carry commas
# (--maps a,b) that nested quoting eats.
RUNNER="$LOG_ROOT/$NAME.runner.sh"
{
    echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    printf 'cd %q\n' "$REPO"
    echo 'source "$(conda info --base)/etc/profile.d/conda.sh"'
    echo "conda activate $CONDA_ENV"
    echo "export HF_HOME=${HF_HOME:-/home/uberger/scratch/cache/hf_cache}"
    echo "export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}"
    printf 'python %q/fig1_multistep.py' "$REPO"
    for d in "${ABS[@]}"; do printf ' --run-dir %q' "$d"; done
    printf ' --out %q' "$OUT"
    for a in ${EXTRA[@]+"${EXTRA[@]}"}; do printf ' %q' "$a"; done
    printf '\n'
} > "$RUNNER"
chmod +x "$RUNNER"

echo "=========================================================================="
echo "Job      : $NAME   ($ACCOUNT, $PARTITION, ${DURATION}h, 1 GPU)"
echo "Run dirs : ${ABS[*]}"
echo "Out      : $OUT"
echo "Extra    : ${EXTRA[*]:-(none)}"
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
