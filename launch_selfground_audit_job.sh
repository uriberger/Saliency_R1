#!/usr/bin/env bash
# Submit the GPU stages of selfground_audit.py to SLURM -- the `_job.sh` half of
# launch_selfground_audit.sh, same split as launch_overlap_probe_job.sh.
#
#   bash launch_selfground_audit_job.sh --name sg-dino --stage dino \
#       --out-dir outputs/selfground/holdout [--gpus 8] [--duration 1] \
#       -- --probe outputs/overlap_probe/align-A --probe outputs/overlap_probe/align-C
#
# Runtime reference: `dino` is Grounding-DINO only (~1 GB) and runs at a few hundred
# calls a minute per GPU; `crosspass` loads the 8B policy once per map arm and does one
# teacher-forced forward per completion. Both fit comfortably in an hour on 8 GPUs, which
# buys batch_short.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

NAME=""
STAGE=""
GPUS=8
DURATION=1
OUT_DIR=""
DRY_RUN=0
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)  DRY_RUN=1;       shift   ;;
        --name)     NAME="$2";       shift 2 ;;
        --stage)    STAGE="$2";      shift 2 ;;
        --gpus)     GPUS="$2";       shift 2 ;;
        --duration) DURATION="$2";   shift 2 ;;
        --out-dir)  OUT_DIR="$2";    shift 2 ;;
        --)         shift; EXTRA+=("$@"); break ;;
        *)          EXTRA+=("$1");   shift ;;
    esac
done

[[ -n "$NAME"    ]] || { echo "ERROR: --name is required." >&2; exit 2; }
[[ -n "$STAGE"   ]] || { echo "ERROR: --stage is required." >&2; exit 2; }
[[ -n "$OUT_DIR" ]] || { echo "ERROR: --out-dir is required." >&2; exit 2; }
case "$OUT_DIR" in /*) ;; *) OUT_DIR="$REPO/$OUT_DIR" ;; esac

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

RUNNER="$LOG_ROOT/$NAME.runner.sh"
{
    echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    echo "export CONDA_ENV=$CONDA_ENV"
    echo "export HF_HOME=${HF_HOME:-/home/uberger/scratch/cache/hf_cache}"
    echo "export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}"
    echo "export OVERLAP_STEPS_CKPT=${OVERLAP_STEPS_CKPT:-$REPO/checkpoint/steps_classifier/best}"
    printf 'bash %q/launch_selfground_audit.sh --stage %q --gpus %q --out-dir %q --' \
        "$REPO" "$STAGE" "$GPUS" "$OUT_DIR"
    for a in ${EXTRA[@]+"${EXTRA[@]}"}; do printf ' %q' "$a"; done
    echo
} > "$RUNNER"
chmod +x "$RUNNER"

echo "=========================================================================="
echo "Job     : $NAME   ($ACCOUNT, $PARTITION, ${DURATION}h, ${GPUS} GPU)"
echo "Stage   : $STAGE"
echo "Out dir : $OUT_DIR"
echo "Runner  : $RUNNER"
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
