#!/usr/bin/env bash
# Submit the DeepEyes coordinate-frame / format probe to SLURM.
#
#   bash launch_deepeyes_coord_format_job.sh --name de-coordfmt \
#       --out-dir outputs/deepeyes_coord_format
#   bash launch_deepeyes_coord_format_job.sh --name de-coordfmt-greedy --gpus 4 \
#       --out-dir outputs/deepeyes_coord_format_greedy -- --temperature 0
#
# Same `_job.sh` (submit) / `_run.sh` (execute) split as launch_sink_location_job.sh:
# this script only builds a runner and hands it to submit_job. Everything about WHAT runs
# is in launch_deepeyes_coord_format.sh, so an interactive node runs the identical thing.
#
# WHAT THE FIRST LINES OF THE LOG SHOULD SAY, AND WHAT A MISMATCH MEANS.
#
#   `arm A  proc_kw={}  default size=SizeDict(... longest_edge=16777216 ...)`
#       The processor's own caps. If `longest_edge` is anything else, the checkpoint's
#       preprocessor config has moved and arm A is no longer "no downscale" -- which is
#       the assumption the three-readings table is read under.
#   `arm-B gate: (2250, 1500) -> arm A (1504, 2240) vs arm B (1024, 1536)  (1.458x linear)`
#       Arm B is the only thing that separates `abs_orig` from `abs_resized`, and the
#       override is a SILENT no-op if the processor ignores it -- passing `max_pixels`
#       without `min_pixels` does exactly that. The probe refuses to run if this ratio
#       falls below 1.2 rather than producing a tie that reads like a result.
#   `191 of 191 rows to do`
#       V*'s two scored splits, 115 + 76. A smaller number means a stale JSONL is being
#       resumed (fine) or a shard split (also fine, it says `of N`); a LARGER corpus means
#       VSTAR_BENCH_PATH is pointing somewhere else.
#
# The probe refuses to start if the embedded DeepEyes prompt is not character-identical to
# the one in the local clone. That check is the point of the measurement, not red tape --
# a tidied prompt measures a model DeepEyes never ran.
#
# COST. ~10 s a row at batch size 1; 191 rows x 2 models x 2 arms. 1 h is generous on 8
# GPUs and adequate on 4. Poll, do not resubmit.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

NAME=""
OUT_DIR=""
DURATION=1
GPUS=8
ARMS="A,B"
PARTITION_OVERRIDE=""
DRY_RUN=0
MODELS=()
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)   DRY_RUN=1;               shift   ;;
        --name)      NAME="$2";               shift 2 ;;
        --out-dir)   OUT_DIR="$2";            shift 2 ;;
        --duration)  DURATION="$2";           shift 2 ;;
        --gpus)      GPUS="$2";               shift 2 ;;
        --arms)      ARMS="$2";               shift 2 ;;
        --model)     MODELS+=("$2");          shift 2 ;;
        --partition) PARTITION_OVERRIDE="$2"; shift 2 ;;
        --)          shift; EXTRA+=("$@"); break ;;
        *)           EXTRA+=("$1");           shift   ;;
    esac
done

[[ -n "$NAME"    ]] || { echo "ERROR: --name is required (it names the job and the log)." >&2; exit 2; }
[[ -n "$OUT_DIR" ]] || { echo "ERROR: --out-dir is required." >&2; exit 2; }
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
    # A clone of Visual-Agent/DeepEyes, read-only. The probe degrades to a warning if it
    # is missing, but then the prompt is unverified -- so it is passed explicitly.
    echo "export DEEPEYES_REF=${DEEPEYES_REF:-/home/uberger/scratch/deepeyes_ref/DeepEyes}"
    printf 'bash launch_deepeyes_coord_format.sh --gpus %q --arms %q --out-dir %q' \
        "$GPUS" "$ARMS" "$OUT_DIR"
    for m in ${MODELS[@]+"${MODELS[@]}"}; do printf ' --model %q' "$m"; done
    for a in ${EXTRA[@]+"${EXTRA[@]}"}; do printf ' %q' "$a"; done
    echo
} > "$RUNNER"
chmod +x "$RUNNER"

echo "=========================================================================="
echo "Job       : $NAME   ($ACCOUNT, $PARTITION, ${DURATION}h, ${GPUS} GPU)"
echo "Out dir   : $OUT_DIR"
echo "Arms      : $ARMS"
echo "Models    : ${MODELS[*]:-(the two defaults: cold start and Instruct)}"
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
