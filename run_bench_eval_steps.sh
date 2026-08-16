#!/bin/bash
# Score a named list of checkpoints from one GRPO run on the mini test suites,
# one after another, on the GPUs of the node this is run from.
#
#   bash run_bench_eval_steps.sh [--wandb-run-id ID] [--dry-run] [--force]
#   bash run_bench_eval_steps.sh --push-only    # ship what is scored, evaluate nothing
#
# RUN is the only thing that normally has to change. Everything else is read off
# the run directory: the steps still owed are the checkpoints with no step file,
# and the WandB run to log them to comes from the model card the trainer wrote
# there. Set STEPS (a space-separated list) to override the first, --wandb-run-id
# to override the second.
#
# A detected list is ordered furthest-first rather than numerically -- final
# checkpoint, then the middle of the largest remaining gap, and so on -- because
# an allocation nearly always ends part-way through. A hand-written STEPS is
# evaluated in exactly the order it is written.
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

# RUN is the only thing that has to change to point this at another run: with
# STEPS unset the list is read off the run directory, and with --wandb-run-id
# unset the run id is read out of the model card the trainer left there.
#
# All three can still be given, from the environment or the command line, and an
# explicit value always wins over a detected one:
#
#   RUN=<run-name> STEPS="3990 900 2000" bash run_bench_eval_steps.sh
#
# STEPS is a space-separated string there, not an array -- the environment has no
# arrays -- and is split back into one below, after the default is worked out.
RUN=${RUN:-grpo-coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged-overlap__wov0.11_2head_trmean_saliency_r1_8k_auroc}
STEPS_IN=${STEPS:-}
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
        -h|--help) sed -n '2,49p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

[[ -d "$RUN_DIR" ]] || { echo "error: no such run dir: $RUN_DIR" >&2; exit 2; }
mkdir -p "$RUN_DIR/bench_eval" "$SHADOW_ROOT"

# Only the standard library is needed below, so any interpreter will do -- and it
# has to be found before any conda env is activated, because this runs before the
# first suite does.
PY_BIN=$(command -v python || command -v python3) || \
    { echo "error: no python on PATH" >&2; exit 2; }

# ---------- which WandB run these results belong to ----------
# The trainer writes a model card into the run directory when it saves, and that
# card carries a badge linking to the WandB run -- entity, project and run id, in
# the one place that is guaranteed to be next to the checkpoints rather than in a
# launcher log whose name nobody remembers. It is also the only source that works
# offline.
#
# Not inferred from $RUN: the two are not the same string. The launchers let
# WANDB_RUN_ID be overridden and it usually is, so a run kept on disk as
# ..._qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged-overlap__wov0.11_... is
# ...-overlap__wov011_... in WandB. Guessing that transformation would sooner or
# later log one run's benchmarks onto another run's curve.
detect_wandb_run() {
    local card="$RUN_DIR/README.md"
    [[ -f "$card" ]] || return 1
    grep -o 'wandb\.ai/[A-Za-z0-9._-]*/[A-Za-z0-9._-]*/runs/[A-Za-z0-9._-]*' "$card" | head -1
}

WANDB_DETECTED=""
if [[ -z "$WANDB_RUN_ID" ]] && WANDB_DETECTED=$(detect_wandb_run) && [[ -n "$WANDB_DETECTED" ]]; then
    IFS=/ read -r _ WANDB_ENTITY_D WANDB_PROJECT_D _ WANDB_RUN_ID <<< "$WANDB_DETECTED"
    # bench_eval.py reads these from the environment, defaulting to
    # nvr-israel/vlm_reasoning. Take them from the card too: a run in another
    # project would otherwise be resumed as a NEW run of the default one.
    export WANDB_ENTITY=${WANDB_ENTITY:-$WANDB_ENTITY_D}
    export WANDB_PROJECT=${WANDB_PROJECT:-$WANDB_PROJECT_D}
fi

$PUSH_ONLY && [[ -z "$WANDB_RUN_ID" ]] && \
    { echo "error: --push-only needs a WandB run, and none was found in $RUN_DIR/README.md" >&2; exit 2; }

# ---------- which steps still need scoring, and in what order ----------
# Every checkpoint on disk that has no step file yet. The order is not numeric:
# an allocation nearly always ends part-way through a list this long, so each
# step is chosen to be the one furthest from anything already measured -- the
# final checkpoint first, then the middle of the largest remaining gap, and so
# on. Whenever it is interrupted, what exists is a curve over the whole run
# rather than a dense prefix and nothing after it.
default_steps() {
    "$PY_BIN" - "$RUN_DIR" <<'PY'
import os, re, sys

run = sys.argv[1]
bench = os.path.join(run, "bench_eval")

# Anchors are the steps already on the curve, step 0 (the baseline) included.
anchors = []
for name in os.listdir(bench):
    m = re.fullmatch(r"step-(\d+)\.json", name)
    if m:
        anchors.append(int(m.group(1)))

pending = []
for name in os.listdir(run):
    m = re.fullmatch(r"checkpoint-(\d+)", name)
    if not m:
        continue
    step = int(m.group(1))
    if step in anchors:
        continue
    # Both files, and a non-empty adapter: the same completeness test the loop
    # below applies, so a checkpoint still being written is not offered.
    d = os.path.join(run, name)
    weights = os.path.join(d, "adapter_model.safetensors")
    if not os.path.isfile(os.path.join(d, "adapter_config.json")):
        continue
    if not (os.path.isfile(weights) and os.path.getsize(weights) > 0):
        continue
    pending.append(step)

order = []
if pending and not anchors:
    # Nothing measured yet, so there is no gap to bisect. Start at the end: the
    # final checkpoint is the one the run is reported by.
    first = max(pending)
    pending.remove(first)
    order.append(first)
    anchors.append(first)

while pending:
    # Furthest from every anchor; ties go to the later step, which keeps the end
    # of the run ahead of its mirror image at the start.
    nxt = max(pending, key=lambda s: (min(abs(s - a) for a in anchors), s))
    pending.remove(nxt)
    order.append(nxt)
    anchors.append(nxt)

print(" ".join(str(s) for s in order))
PY
}

if [[ -n "$STEPS_IN" ]]; then
    STEPS_SOURCE="given"
else
    STEPS_IN=$(default_steps) || { echo "error: could not read the steps from $RUN_DIR" >&2; exit 2; }
    STEPS_SOURCE="detected"
fi
read -r -a STEPS <<< "$STEPS_IN"
(( ${#STEPS[@]} > 0 )) || { echo "nothing to do: every checkpoint in $RUN_DIR is already scored"; exit 0; }

echo "=========================================================================="
echo "Run:      $RUN"
echo "Steps:    ${#STEPS[@]} ($STEPS_SOURCE$([[ "$STEPS_SOURCE" == detected ]] && echo ', furthest-first'))   ${STEPS[*]}"
echo "GPUs:     $NUM_GPUS   (serial, one checkpoint at a time)"
echo "Results:  $RUN_DIR/bench_eval/step-<N>.json"
if [[ -n "$WANDB_RUN_ID" ]]; then
    echo "WandB:    ${WANDB_ENTITY:-nvr-israel}/${WANDB_PROJECT:-vlm_reasoning}/$WANDB_RUN_ID$([[ -n "$WANDB_DETECTED" ]] && echo '   (from README.md)')"
    echo "          each step pushed as it is scored"
else
    echo "WandB:    NOT pushed -- no run badge in $RUN_DIR/README.md; pass --wandb-run-id ID"
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
