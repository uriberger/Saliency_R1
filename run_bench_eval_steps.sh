#!/bin/bash
# Score a named list of checkpoints from one GRPO run on the mini test suites,
# one after another, on the GPUs of the node this is run from.
#
#   bash run_bench_eval_steps.sh [--dry-run]
#
# Edit RUN and STEPS below. Steps are evaluated in the order they are written, not
# in numeric order: the point of a hand-written list is usually to get one
# checkpoint scored first and fill in the curve afterwards.
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
# each step, which is what puts it in WandB: the live trainer's callback polls that
# directory and logs whatever appears. If training has already exited, append them
# afterwards with
#
#   python bench_eval.py --backfill --run-dir checkpoint/<run> --wandb-run-id <id>
#
# Note this takes all $NUM_GPUS GPUs for the whole list -- do not start it next to
# a training job on the same node.
set -uo pipefail

RUN=grpo-coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged-overlap__wov0.033_2head_trmean_50k_set_a_mean_in_v2_k_proj_beta_004
STEPS=(3000 300 800 1300 1800 2300 2700 3300)
NUM_GPUS=8

SCRIPT_DIR=$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")" && pwd)
REPO=${REPO:-/home/uberger/scratch/research/saliency_r1}
RUN_DIR="$REPO/checkpoint/$RUN"
SHADOW_ROOT="$REPO/outputs/bench_one"

DRY_RUN=false
FORCE=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        # Re-score a step that already has a result. The step file is what marks a
        # checkpoint done, for this script and for the dispatcher both, so without
        # this a rerun of the list is a cheap no-op rather than a repeat of hours
        # of GPU time.
        --force)   FORCE=true;   shift ;;
        -h|--help) sed -n '2,33p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

[[ -d "$RUN_DIR" ]] || { echo "error: no such run dir: $RUN_DIR" >&2; exit 2; }
mkdir -p "$RUN_DIR/bench_eval" "$SHADOW_ROOT"

echo "=========================================================================="
echo "Run:      $RUN"
echo "Steps:    ${STEPS[*]}"
echo "GPUs:     $NUM_GPUS   (serial, one checkpoint at a time)"
echo "Results:  $RUN_DIR/bench_eval/step-<N>.json  -> WandB via the trainer"
$DRY_RUN && echo "Mode:     --dry-run, nothing will be evaluated"
echo "=========================================================================="

# Reported at the end rather than as they happen: a missing checkpoint scrolls off
# the top of an hours-long log, and a list that says it evaluated eight steps when
# it evaluated six is the failure that matters here.
declare -a DONE=() SKIPPED=() FAILED=()

for STEP in "${STEPS[@]}"; do
    echo ""
    echo "Starting step $STEP"

    CKPT="$RUN_DIR/checkpoint-$STEP"
    RESULT="$RUN_DIR/bench_eval/step-$STEP.json"
    SHADOW="$SHADOW_ROOT/$RUN-cp$STEP"

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
echo "=========================================================================="
(( ${#FAILED[@]} == 0 ))
