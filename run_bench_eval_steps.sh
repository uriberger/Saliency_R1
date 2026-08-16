#!/bin/bash
# Score a named list of checkpoints from one GRPO run on the mini test suites,
# one after another, on the GPUs of the node this is run from.
#
#   bash run_bench_eval_steps.sh [--wandb-run-id ID] [--dry-run] [--force]
#   bash run_bench_eval_steps.sh --push-only --wandb-run-id ID   # ship, do not score
#
# Set RUN and STEPS below or in the environment. Steps are evaluated in the order
# they are written, not in numeric order: the point of a hand-written list is
# usually to get one checkpoint scored first and fill in the curve afterwards.
#
# This is the manual counterpart to watch_bench_evals.sh. That one submits a job
# per batch of pending checkpoints and drains them oldest-first; this one runs
# here, in the foreground, on the checkpoints you name. Everything else -- the
# merge, the three suites, the reduction to bench_eval/step-<N>.json -- is
# run_bench_eval.sh, unchanged, so a checkpoint scored this way is scored by
# exactly the same recipe as one scored by the dispatcher.
#
# Each step is evaluated through a SHADOW run directory holding a symlink to just
# that checkpoint. run_bench_eval.sh takes the oldest pending checkpoint in the
# directory it is given, and treats bench_eval/base_model.txt as a pending step 0,
# so pointing it at the real run directory would score the baseline instead of the
# checkpoint asked for. The shadow directory leaves it exactly one choice. It also
# keeps the merged model under a name of its own, so this cannot collide with a
# dispatcher job merging the same checkpoint at the same time.
#
# The finished step file is copied into the real run directory as the last act for
# each step. While the run is still training, that alone puts it in WandB: the
# trainer's callback polls that directory and logs whatever appears. Once training
# has exited nothing is watching, so pass --wandb-run-id and this script ships each
# step itself, through bench_eval.py --backfill, the moment that step is scored --
# not at the end, so a list cut short by the wall clock still leaves every step it
# did finish on the curve.
#
#   bash run_bench_eval_steps.sh --wandb-run-id grpo-coldstart-overlap__wov011_...
#
# Only the step just scored is pushed, never the whole directory: --backfill logs
# every step file it is shown, so pointing it at the run directory would re-log the
# hundreds of points already there.
#
# Note this takes all $NUM_GPUS GPUs for the whole list -- do not start it next to
# a training job on the same node.
set -uo pipefail

# Both may be overridden from the environment, so draining a long list across
# several allocations does not need an edit (and a commit) per allocation:
#
#   RUN=<run-name> STEPS="3990 900 2000" bash run_bench_eval_steps.sh
#
# STEPS is a space-separated string there, not an array -- the environment has no
# arrays -- and is split back into one here.
RUN=${RUN:-grpo-coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged-overlap__wov0.11_2head_trmean_saliency_r1_8k_auroc}
read -r -a STEPS <<< "${STEPS:-3990 900 2000 3000 1500 2500 3500 1000 1100 1200 1300 1400 1600 1700 1800 1900 2100 2200 2300 2400 2600 2700 2800 2900 3100 3200 3300 3400 3600 3700 3800 3900}"
NUM_GPUS=${NUM_GPUS:-8}

SCRIPT_DIR=$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")" && pwd)
REPO=${REPO:-/home/uberger/scratch/research/saliency_r1}
RUN_DIR="$REPO/checkpoint/$RUN"
SHADOW_ROOT="$REPO/outputs/bench_one"
CONDA_SH=/home/uberger/scratch/miniconda3/etc/profile.d/conda.sh
# Any env with wandb in it. lmms_eval is the one the eval itself runs in, so
# asking for it costs nothing that is not already paid for.
BACKFILL_ENV=${BACKFILL_ENV:-lmms_eval}

DRY_RUN=false
FORCE=false
PUSH_ONLY=false
WANDB_RUN_ID=${WANDB_RUN_ID:-}
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        # Re-score a step that already has a result. The step file is what marks a
        # checkpoint done, for this script and for the dispatcher both, so without
        # this a rerun of the list is a cheap no-op rather than a repeat of hours
        # of GPU time.
        --force)   FORCE=true;   shift ;;
        --wandb-run-id) WANDB_RUN_ID="$2"; shift 2 ;;
        # Push step files that are already on disk and evaluate nothing. For a push
        # that failed at the time (no network, a bad run id): re-running the list
        # normally would skip those steps as already scored and push nothing.
        --push-only) PUSH_ONLY=true; shift ;;
        -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

$PUSH_ONLY && [[ -z "$WANDB_RUN_ID" ]] && \
    { echo "error: --push-only needs --wandb-run-id" >&2; exit 2; }
[[ -d "$RUN_DIR" ]] || { echo "error: no such run dir: $RUN_DIR" >&2; exit 2; }
mkdir -p "$RUN_DIR/bench_eval" "$SHADOW_ROOT"

echo "=========================================================================="
echo "Run:      $RUN"
echo "Steps:    ${STEPS[*]}"
echo "GPUs:     $NUM_GPUS   (serial, one checkpoint at a time)"
echo "Results:  $RUN_DIR/bench_eval/step-<N>.json"
if [[ -n "$WANDB_RUN_ID" ]]; then
    echo "WandB:    each step pushed to run $WANDB_RUN_ID as it is scored"
else
    echo "WandB:    NOT pushed -- pass --wandb-run-id ID (only the live trainer logs these on its own)"
fi
$DRY_RUN && echo "Mode:     --dry-run, nothing will be evaluated"
echo "=========================================================================="

# Reported at the end rather than as they happen: a missing checkpoint scrolls off
# the top of an hours-long log, and a list that says it evaluated eight steps when
# it evaluated six is the failure that matters here.
declare -a DONE=() SKIPPED=() FAILED=() UNPUSHED=()

# Ship one scored step to WandB, through a staging directory holding a symlink to
# just that step file. bench_eval.py --backfill logs every step file under the
# directory it is given, and the run already carries every point the trainer
# logged live, so handing it the real bench_eval/ would re-log all of them.
#
# A push that fails is reported and does not stop the drain: the step file is on
# disk, which is the result that took the GPU hours, and it can be backfilled at
# any time afterwards.
push_to_wandb() {
    local step="$1"
    [[ -n "$WANDB_RUN_ID" ]] || return 0
    local staging="$SHADOW_ROOT/.wandb_push/step-$step"
    rm -rf "$staging"
    mkdir -p "$staging/bench_eval"
    ln -sfn "$RUN_DIR/bench_eval/step-$step.json" "$staging/bench_eval/step-$step.json"
    if ( source "$CONDA_SH"
         set +u; conda activate "$BACKFILL_ENV"; set -u
         python "$SCRIPT_DIR/bench_eval.py" --backfill \
             --run-dir "$staging" --wandb-run-id "$WANDB_RUN_ID" ); then
        echo "  step $step -> WandB run $WANDB_RUN_ID"
    else
        echo "  step $step: WandB push FAILED -- the result is on disk, backfill it later" >&2
        UNPUSHED+=("$step")
    fi
    rm -rf "$staging"
}

for STEP in "${STEPS[@]}"; do
    echo ""
    echo "Starting step $STEP"

    CKPT="$RUN_DIR/checkpoint-$STEP"
    RESULT="$RUN_DIR/bench_eval/step-$STEP.json"
    SHADOW="$SHADOW_ROOT/$RUN-cp$STEP"

    if $PUSH_ONLY; then
        if [[ -f "$RESULT" ]]; then
            $DRY_RUN && { echo "  would push $RESULT to $WANDB_RUN_ID"; DONE+=("$STEP (dry-run)"); continue; }
            push_to_wandb "$STEP"
            DONE+=("$STEP (pushed)")
        else
            echo "  no result at $RESULT -- nothing to push"
            SKIPPED+=("$STEP (not scored)")
        fi
        continue
    fi

    # Both files, not just the config: a checkpoint being written right now would
    # otherwise be merged from a partial adapter. Same test the dispatcher uses.
    if [[ ! -f "$CKPT/adapter_config.json" || ! -s "$CKPT/adapter_model.safetensors" ]]; then
        echo "  no complete checkpoint at $CKPT -- skipping"
        SKIPPED+=("$STEP (no checkpoint)")
        continue
    fi
    if [[ -f "$RESULT" ]] && ! $FORCE; then
        echo "  already scored ($RESULT) -- skipping; pass --force to redo it"
        SKIPPED+=("$STEP (already scored)")
        continue
    fi

    if $DRY_RUN; then
        echo "  would evaluate $CKPT on $NUM_GPUS GPUs -> $RESULT"
        DONE+=("$STEP (dry-run)")
        continue
    fi

    mkdir -p "$SHADOW"
    ln -sfn "$CKPT" "$SHADOW/checkpoint-$STEP"

    bash "$SCRIPT_DIR/run_bench_eval.sh" --run-dir "$SHADOW" --num-gpus "$NUM_GPUS" \
        --every 1 --min-minutes 0

    # run_bench_eval.sh reports a suite that failed and moves on, so its exit status
    # does not say whether this checkpoint was scored. The step file does.
    if [[ -f "$SHADOW/bench_eval/step-$STEP.json" ]]; then
        cp "$SHADOW/bench_eval/step-$STEP.json" "$RESULT"
        echo "  step $STEP done -> $RESULT"
        DONE+=("$STEP")
        push_to_wandb "$STEP"
    else
        echo "  step $STEP produced no result -- see the output above" >&2
        FAILED+=("$STEP")
    fi
done

echo ""
echo "=========================================================================="
echo "Evaluated: ${#DONE[@]}   ${DONE[*]:-}"
echo "Skipped:   ${#SKIPPED[@]}   ${SKIPPED[*]:-}"
echo "Failed:    ${#FAILED[@]}   ${FAILED[*]:-}"
if (( ${#UNPUSHED[@]} > 0 )); then
    echo "Unpushed:  ${#UNPUSHED[@]}   ${UNPUSHED[*]}   (scored, but WandB rejected them)"
    echo "           retry:  STEPS=\"${UNPUSHED[*]}\" bash $0 --push-only --wandb-run-id $WANDB_RUN_ID"
fi
echo "=========================================================================="
(( ${#FAILED[@]} + ${#UNPUSHED[@]} == 0 ))
