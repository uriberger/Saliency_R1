#!/usr/bin/env bash
# Submit a sink-location stage to SLURM. Same `_job.sh` split as
# launch_dino_text_sensitivity_job.sh: this script only builds a runner and hands it to
# submit_job; launch_sink_location.sh is what actually shards the work over the node's
# GPUs once the allocation exists.
#
#   bash launch_sink_location_job.sh --name sl-selftest --stage selftest --gpus 1 \
#       --out-dir outputs/sink_location/coldstart --model checkpoint/...merged
#   bash launch_sink_location_job.sh --name sl-scan --stage scan --gpus 8 ...
#   bash launch_sink_location_job.sh --name sl-arms --stage arms --gpus 8 ...
#
# `corpus` does NOT belong here. It is CPU-only, it is the one stage that touches
# `datasets`, and this account has 50 GPU-free nodes in cpu_short that it can have
# immediately -- pass --partition cpu_short --gpus 0 if you want it queued anyway.
#
# COST. One picture is one prefill at batch size 1, so an hour is generous for the whole
# scan and the whole arms grid together. The request is small on purpose: the account's
# GPU cap is what this queues behind, and the smallest request that can run is the one
# that starts soonest. Poll, do not resubmit.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

NAME=""
STAGE="scan"
OUT_DIR=""
MODEL=""
DURATION=1
GPUS=8
PARTITION_OVERRIDE=""
DRY_RUN=0
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)   DRY_RUN=1;              shift   ;;
        --name)      NAME="$2";              shift 2 ;;
        --stage)     STAGE="$2";             shift 2 ;;
        --out-dir)   OUT_DIR="$2";           shift 2 ;;
        --model)     MODEL="$2";             shift 2 ;;
        --duration)  DURATION="$2";          shift 2 ;;
        --gpus)      GPUS="$2";              shift 2 ;;
        --partition) PARTITION_OVERRIDE="$2"; shift 2 ;;
        --)          shift; EXTRA+=("$@"); break ;;
        *)           EXTRA+=("$1");          shift ;;
    esac
done

[[ -n "$NAME"    ]] || { echo "ERROR: --name is required (it names the job and the log)." >&2; exit 2; }
[[ -n "$OUT_DIR" ]] || { echo "ERROR: --out-dir is required." >&2; exit 2; }
case "$STAGE" in
    corpus|selftest|scan|arms|report) ;;
    *) echo "ERROR: --stage $STAGE is not a stage." >&2; exit 2 ;;
esac
if [[ "$STAGE" != "corpus" && "$STAGE" != "report" ]]; then
    [[ -n "$MODEL" ]] || { echo "ERROR: --model is required for stage $STAGE." >&2; exit 2; }
fi
[[ "$OUT_DIR" = /* ]] || OUT_DIR="$REPO/$OUT_DIR"

# shellcheck source=/dev/null
source "$REPO/cluster_env.sh"
PARTITION=${PARTITION_OVERRIDE:-${PARTITION:-$(SR1_JOB_HOURS=$DURATION sr1_pick_partition)}}
ACCOUNT=${ACCOUNT:-nvr_israel_rlop}
CONDA_ENV=${CONDA_ENV:-saliency_r1_qwen3_vllm}

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
    printf 'cd %q\n' "$REPO"
    echo "export CONDA_ENV=${CONDA_ENV}"
    echo "export HF_HOME=${HF_HOME:-/home/uberger/scratch/cache/hf_cache}"
    echo "export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}"
    # The inner launcher activates conda itself and gates scan/arms on the selftest log,
    # so what runs inside the allocation is exactly what runs on an interactive node.
    printf 'bash launch_sink_location.sh --stage %q --gpus %q --out-dir %q' \
        "$STAGE" "$GPUS" "$OUT_DIR"
    [[ -n "$MODEL" ]] && printf ' --model %q' "$MODEL"
    for a in ${EXTRA[@]+"${EXTRA[@]}"}; do printf ' %q' "$a"; done
    echo
} > "$RUNNER"
chmod +x "$RUNNER"

echo "=========================================================================="
echo "Job       : $NAME   ($ACCOUNT, $PARTITION, ${DURATION}h, ${GPUS} GPU)"
echo "Stage     : $STAGE"
echo "Model     : ${MODEL:-(none)}"
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
