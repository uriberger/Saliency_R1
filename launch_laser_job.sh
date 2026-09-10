#!/usr/bin/env bash
# Submit the LASER go/no-go to SLURM. Same `_job.sh` split as
# launch_sink_location_job.sh: this script only builds a runner and hands it to
# submit_job; launch_laser.sh is what shards the work over the node's GPUs once the
# allocation exists.
#
#   bash launch_laser_job.sh --name laser-gonogo --stage selftest,collect,report \
#       --gpus 1 --out-dir outputs/laser/coldstart \
#       --model checkpoint/coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged
#
# --stage takes a comma-separated LIST, run in order inside one allocation, and
# `selftest` always runs on ONE GPU however many the job has: a check that passed on some
# shards and not others is not a gate. `set -e` in the runner is what makes a failing
# selftest stop the stages after it rather than print a line nobody reads.
#
# COST AND SIZE. 256 prompts x 8 rollouts is ~25 min on a single GPU, so the default
# request is 1 GPU for 2 hours. That is deliberate: the account's GPU cap is what this
# queues behind and the smallest request that can run is the one that starts soonest.
# --gpus 8 cuts the collect to ~4 minutes and will usually spend longer than that waiting.
# Poll, do not resubmit.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

NAME=""
STAGE="selftest,collect,report"
OUT_DIR=""
MODEL=""
DURATION=2
GPUS=1
PARTITION_OVERRIDE=""
DRY_RUN=0
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)   DRY_RUN=1;               shift   ;;
        --name)      NAME="$2";               shift 2 ;;
        --stage)     STAGE="$2";              shift 2 ;;
        --out-dir)   OUT_DIR="$2";            shift 2 ;;
        --model)     MODEL="$2";              shift 2 ;;
        --duration)  DURATION="$2";           shift 2 ;;
        --gpus)      GPUS="$2";               shift 2 ;;
        --partition) PARTITION_OVERRIDE="$2"; shift 2 ;;
        --)          shift; EXTRA+=("$@"); break ;;
        *)           EXTRA+=("$1");           shift ;;
    esac
done

[[ -n "$NAME"    ]] || { echo "ERROR: --name is required (it names the job and the log)." >&2; exit 2; }
[[ -n "$OUT_DIR" ]] || { echo "ERROR: --out-dir is required." >&2; exit 2; }
IFS=',' read -r -a STAGES <<< "$STAGE"
for s in "${STAGES[@]}"; do
    case "$s" in
        selftest|collect|report) ;;
        *) echo "ERROR: --stage $s is not a stage." >&2; exit 2 ;;
    esac
done
for s in "${STAGES[@]}"; do
    if [[ "$s" != "report" && -z "$MODEL" ]]; then
        echo "ERROR: --model is required for stage $s." >&2; exit 2
    fi
done
[[ "$OUT_DIR" = /* ]] || OUT_DIR="$REPO/$OUT_DIR"

# shellcheck source=/dev/null
source "$REPO/cluster_env.sh"
PARTITION=${PARTITION_OVERRIDE:-${PARTITION:-$(SR1_JOB_HOURS=$DURATION sr1_pick_partition)}}
ACCOUNT=${ACCOUNT:-nvr_israel_rlop}
CONDA_ENV=${CONDA_ENV:-saliency_r1_qwen3_vllm}

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
    for s in "${STAGES[@]}"; do
        printf 'bash launch_laser.sh --stage %q --gpus %q --out-dir %q' \
            "$s" "$([[ $s == selftest ]] && echo 1 || echo "$GPUS")" "$OUT_DIR"
        [[ -n "$MODEL" && "$s" != "report" ]] && printf ' --model %q' "$MODEL"
        for a in ${EXTRA[@]+"${EXTRA[@]}"}; do printf ' %q' "$a"; done
        echo
    done
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
