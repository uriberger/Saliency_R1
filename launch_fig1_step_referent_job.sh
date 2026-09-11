#!/usr/bin/env bash
# Submit fig1_step_referent.py to SLURM, on ONE GPU.
#
#   bash launch_fig1_step_referent_job.sh --run-dir outputs/saliency_viz/sviz-3models \
#       [--name fig1-step-referent] [--duration 1] [-- <forwarded to the python>]
#
# The script itself is CPU-capable -- Grounding-DINO falls back to CPU when no GPU is
# visible -- and that is genuinely how it was first run. It is also why this launcher
# exists: 188 (image, sentence) pairs took over 45 minutes on a contended login node at
# ~2 cores and had not finished, against seconds of detector time on one GPU. This asks
# for one GPU rather than the eight the viz launchers take, because there is nothing to
# shard: a single batched DINO pass is the whole job.
#
# Everything after `--` is forwarded verbatim to fig1_step_referent.py (--model, --map,
# --iou-min, --box-threshold, --max-union-area, --top ...).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

NAME="fig1-step-referent"
RUN_DIR=""
DURATION=1
OUT=""
DRY_RUN=0
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)  DRY_RUN=1;     shift   ;;
        --name)     NAME="$2";     shift 2 ;;
        --run-dir)  RUN_DIR="$2";  shift 2 ;;
        --duration) DURATION="$2"; shift 2 ;;
        --out)      OUT="$2";      shift 2 ;;
        --)         shift; EXTRA+=("$@"); break ;;
        *)          EXTRA+=("$1"); shift ;;
    esac
done

[[ -n "$RUN_DIR" ]] || { echo "ERROR: --run-dir is required (a saliency_viz output root)." >&2; exit 2; }
[[ -d "$RUN_DIR" ]] || { echo "ERROR: --run-dir not found: $RUN_DIR" >&2; exit 2; }
RUN_DIR="$(cd "$RUN_DIR" && pwd)"
OUT=${OUT:-$REPO/outputs/fig1-step-referent/$NAME.json}

# shellcheck source=/dev/null
source "$REPO/cluster_env.sh"
PARTITION=${PARTITION:-$(SR1_JOB_HOURS=$DURATION sr1_pick_partition)}
ACCOUNT=${ACCOUNT:-nvr_israel_rlop}
CONDA_ENV=${CONDA_ENV:-saliency_r1_qwen3_vllm}

sr1_find_submit_job || [[ $DRY_RUN -eq 1 ]] || {
    echo "ERROR: submit_job not found under the cluster-interface paths." >&2; exit 1; }

LOG_ROOT="$REPO/outputs/logs"
mkdir -p "$LOG_ROOT" "$(dirname "$OUT")"

# A file rather than a quoted `bash -c` string, for the same reason as the viz launcher:
# the forwarded flags carry commas and equals signs that nested quoting eats.
RUNNER="$LOG_ROOT/$NAME.runner.sh"
{
    echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    printf 'cd %q\n' "$REPO"
    echo 'source "$(conda info --base)/etc/profile.d/conda.sh"'
    echo "conda activate $CONDA_ENV"
    echo "export HF_HOME=${HF_HOME:-/home/uberger/scratch/cache/hf_cache}"
    echo "export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}"
    printf 'python %q/fig1_step_referent.py --run-dir %q --out %q' "$REPO" "$RUN_DIR" "$OUT"
    for a in ${EXTRA[@]+"${EXTRA[@]}"}; do printf ' %q' "$a"; done
    printf '\n'
} > "$RUNNER"
chmod +x "$RUNNER"

echo "=========================================================================="
echo "Job     : $NAME   ($ACCOUNT, $PARTITION, ${DURATION}h, 1 GPU)"
echo "Run dir : $RUN_DIR"
echo "Out     : $OUT"
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
