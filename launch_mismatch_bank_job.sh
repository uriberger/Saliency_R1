#!/usr/bin/env bash
# Submit launch_mismatch_bank.sh to SLURM. The launcher it wraps assumes it is already ON
# a node with GPUs -- it fans one shard per visible device and waits -- so this is the
# `_job.sh` half, the same split as launch_overlap_probe_job.sh / launch_overlap_probe.sh.
#
#   bash launch_mismatch_bank_job.sh --name mismatch-bank-8k \
#       [--gpus 8] [--duration 2] [--n-donors 256] [--n-generations 64] \
#       [--out-dir <dir>] [-- <anything else, forwarded to launch_mismatch_bank.sh>]
#
# Runtime reference: 256 donor rows x 64 generations is 16,384 completions of up to 1024
# tokens from the 8B cold start, plus one Grounding-DINO pass over their observe steps
# (~50k at the cold start's median of 3 per chain). On 8 cards that is roughly 1-2 hours,
# so --duration 2 is the honest default and it still buys the short pools -- see
# cluster_env.sh. The plan phase reads and hashes every image in the corpus once, on CPU,
# inside the same allocation (~2 minutes).
#
# Smoke first. --smoke is 4 donors x 4 generations on ONE GPU and exercises the model
# load, generation, the FLAN-T5 segmentation, DINO, the merge and the verify in a few
# minutes:
#
#   bash launch_mismatch_bank_job.sh --name mismatch-smoke --gpus 1 --duration 1 -- --smoke
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

NAME=""
GPUS=8
DURATION=2
N_DONORS=256
N_GEN=64
OUT_DIR=""
DRY_RUN=0
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)        DRY_RUN=1;      shift   ;;
        --name)           NAME="$2";      shift 2 ;;
        --gpus)           GPUS="$2";      shift 2 ;;
        --duration)       DURATION="$2";  shift 2 ;;
        --n-donors)       N_DONORS="$2";  shift 2 ;;
        --n-generations)  N_GEN="$2";     shift 2 ;;
        --out-dir)        OUT_DIR="$2";   shift 2 ;;
        --)               shift; EXTRA+=("$@"); break ;;
        *)                EXTRA+=("$1");  shift   ;;
    esac
done

[[ -n "$NAME" ]] || { echo "ERROR: --name is required (it names the job and the log)." >&2; exit 2; }
OUT_DIR=${OUT_DIR:-$REPO/outputs/mismatch_bank/$NAME}

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
    printf 'bash %q/launch_mismatch_bank.sh --gpus %q --n-donors %q --n-generations %q --out-dir %q' \
        "$REPO" "$GPUS" "$N_DONORS" "$N_GEN" "$OUT_DIR"
    for a in ${EXTRA[@]+"${EXTRA[@]}"}; do printf ' %q' "$a"; done
    echo
} > "$RUNNER"
chmod +x "$RUNNER"

echo "=========================================================================="
echo "Job       : $NAME   ($ACCOUNT, $PARTITION, ${DURATION}h, ${GPUS} GPU)"
echo "Out dir   : $OUT_DIR"
echo "Donors    : $N_DONORS x $N_GEN chains"
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
