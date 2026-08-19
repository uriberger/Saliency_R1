#!/bin/bash
# Score a named list of checkpoints from one GRPO run on the mini test suites,
# one after another, on the GPUs of the node this is run from.
#
#   bash run_bench_eval_steps.sh [--wandb-run-id ID] [--dry-run] [--force]
#   bash run_bench_eval_steps.sh --push-only    # ship what is scored, evaluate nothing
#
# It scores 300 documents per NATURAL benchmark and 100 per non-natural one --
# NOT the 100/100 the during-training dispatcher uses. Those results live under
# bench_eval/n300_100/ and log to WandB as bench_n300_100/*, so they are a
# separate curve from the 100-document history and cannot be plotted on the same
# line as it. `--natural-n 100` gives the old behaviour back; `--nonnatural-n`,
# `--task-n mmstar=200` and `--bank task` are also accepted.
#
# A checkpoint that already has a 100-document result therefore counts as UNSCORED
# here, which is the point: it is being measured again, more precisely, not
# skipped.
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
# merge, the suites, the reduction to a step file and the per-item harvest -- is
# run_bench_eval.sh, unchanged, so a checkpoint scored this way is scored by
# exactly the same recipe as one scored by the dispatcher at the same sample size.
#
# Each step is evaluated through a SHADOW run directory holding a symlink to just
# that checkpoint. run_bench_eval.sh takes the oldest pending checkpoint in the
# directory it is given, and treats bench_eval/base_model.txt as a pending step 0,
# so pointing it at the real run directory would score the baseline instead of the
# checkpoint asked for. The shadow directory leaves it exactly one choice. It also
# keeps the merged model under a name of its own, so this cannot collide with a
# dispatcher job merging the same checkpoint at the same time.
#
# The finished step file, and the per-item rows harvested beside it, are copied
# into the real run directory as the last act for each step. While the run is
# still training, that alone puts a 100-document result in WandB: the trainer's
# callback polls the flat bench_eval/ and logs whatever appears. It does NOT see
# the n300_100 subdirectory, and nothing is watching once training has exited, so
# in both of those cases pass --wandb-run-id: this script then ships each step
# itself, through bench_eval.py --backfill, the moment that step is scored -- not
# at the end, so a list cut short by the wall clock still leaves every step it did
# finish on the curve.
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

# Sample sizes. This script defaults to 300 documents per NATURAL benchmark, not
# to the 100 the during-training dispatcher uses, because the two exist for
# different purposes: the dispatcher tracks a run live on one GPU and has to keep
# up with training, while this is the deliberate re-scoring of named checkpoints
# on all 8, where the extra documents are what make a 0.03 difference resolvable
# (se on a paired difference falls from ~0.016 to ~0.009).
#
# The non-natural half stays at 100. It is the long-answer suite -- 4x the tokens
# per document -- and the effects being chased are on the natural benchmarks, so
# tripling it would spend most of the extra GPU time where it buys nothing.
#
# Results land under bench_eval/n300_100/ and log to WandB as bench_n300_100/*,
# so they cannot be drawn on the same curve as the 100-document history. To score
# a checkpoint at the old size, pass --natural-n 100.
NATURAL_N=${NATURAL_N:-300}
NONNATURAL_N=${NONNATURAL_N:-100}
declare -a TASK_N=()
# Suite banking, unlike the single-GPU path. At 8 GPUs the whole natural suite at
# 300 is ~16 min and the three suites together ~28, so there is no allocation to
# outrun and nothing to gain from paying four extra model loads.
BANK=${BANK:-suite}

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
        --natural-n)    NATURAL_N="$2";    shift 2 ;;
        --nonnatural-n) NONNATURAL_N="$2"; shift 2 ;;
        --task-n)       TASK_N+=("$2");    shift 2 ;;
        --bank)         BANK="$2";         shift 2 ;;
        # Push step files that are already on disk and evaluate nothing. For a push
        # that failed at the time (no network, a bad run id): re-running the list
        # normally would skip those steps as already scored and push nothing.
        --push-only) PUSH_ONLY=true; shift ;;
        -h|--help) sed -n '2,62p' "$0"; exit 0 ;;
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

# ---------- which sample profile ----------
# Resolved once, here, by the same code the eval job will use, so this script and
# the job cannot disagree about where a result belongs. SAMPLE_N_JSON is the
# per-task sizes; PROFILE_DIR is where results for them live (the flat bench_eval/
# for the 100/100 default, a subdirectory otherwise).
declare -a SIZE_ARGS=(--natural-n "$NATURAL_N" --nonnatural-n "$NONNATURAL_N")
for t in ${TASK_N[@]+"${TASK_N[@]}"}; do SIZE_ARGS+=(--task-n "$t"); done
SAMPLE_N_JSON=$("$PY_BIN" "$SCRIPT_DIR/eval_mini/make_mini_tasks.py" \
    "${SIZE_ARGS[@]}" --print-sample-n) || \
    { echo "error: could not resolve the sample sizes" >&2; exit 2; }

read -r PROFILE PROFILE_SUBDIR < <("$PY_BIN" - "$SCRIPT_DIR" "$SAMPLE_N_JSON" <<'PY'
import json, os, sys
sys.path.insert(0, os.path.join(sys.argv[1], "eval_mini"))
from benchmarks import profile_dir, profile_name
sizes = json.loads(sys.argv[2])
# The subdirectory relative to bench_eval/, or "." for the flat default layout.
print(profile_name(sizes), os.path.relpath(profile_dir("x", sizes), "x"))
PY
) || { echo "error: could not resolve the sample profile" >&2; exit 2; }

# Both the real run directory's and a shadow's, since results are produced in one
# and copied to the other.
profile_dir_of() {
    [[ "$PROFILE_SUBDIR" == "." ]] && { echo "$1/bench_eval"; return; }
    echo "$1/bench_eval/$PROFILE_SUBDIR"
}
PROFILE_DIR=$(profile_dir_of "$RUN_DIR")
mkdir -p "$PROFILE_DIR"

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
    "$PY_BIN" - "$RUN_DIR" "$PROFILE_DIR" <<'PY'
import os, re, sys

run, bench = sys.argv[1], sys.argv[2]

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
echo "Sample:   $PROFILE   (natural=$NATURAL_N, non-natural=$NONNATURAL_N per benchmark)"
echo "Banking:  per $BANK"
echo "Results:  $PROFILE_DIR/step-<N>.json"
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
    local staged; staged=$(profile_dir_of "$staging")
    mkdir -p "$staged"
    ln -sfn "$PROFILE_DIR/step-$step.json" "$staged/step-$step.json"
    if ( source "$CONDA_SH"
         set +u; conda activate "$BACKFILL_ENV"; set -u
         python "$SCRIPT_DIR/bench_eval.py" --backfill --sample-n "$SAMPLE_N_JSON" \
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
    RESULT="$PROFILE_DIR/step-$STEP.json"
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
        --every 1 --min-minutes 0 --bank "$BANK" "${SIZE_ARGS[@]}"

    # run_bench_eval.sh reports a suite that failed and moves on, so its exit status
    # does not say whether this checkpoint was scored. The step file does.
    SHADOW_PROFILE=$(profile_dir_of "$SHADOW")
    if [[ -f "$SHADOW_PROFILE/step-$STEP.json" ]]; then
        cp "$SHADOW_PROFILE/step-$STEP.json" "$RESULT"
        # The per-item rows the job harvested beside it. Without this they stay in
        # the shadow directory, which is keyed by <run>-cp<step> and is not where
        # anything looks for them -- bench_samples.py treats bench_one results as
        # copies of curve points and harvests them under the run itself.
        if [[ -f "$SHADOW_PROFILE/samples/step-$STEP.jsonl.gz" ]]; then
            mkdir -p "$PROFILE_DIR/samples"
            cp "$SHADOW_PROFILE/samples/step-$STEP.jsonl.gz" "$PROFILE_DIR/samples/"
        fi
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
