#!/bin/bash
# Score a named list of checkpoints from one GRPO run on the mini test suites,
# one after another, on the GPUs of the node this is run from.
#
#   bash run_bench_eval_steps.sh [--wandb-run-id ID] [--dry-run] [--force]
#   bash run_bench_eval_steps.sh --push-only    # ship what is scored, evaluate nothing
#
# To run it as a SLURM job rather than here, use launch_bench_eval_steps.sh: it
# submits this script with --job-minutes and chains the allocations until the list
# is done. See "WALL CLOCK" below for what --job-minutes changes.
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
# WALL CLOCK
#
# With --job-minutes N this script knows it is inside an allocation of N minutes
# and behaves as one link in a chain rather than as a single long run:
#
#   * it tells run_bench_eval.sh how much clock is left, so units are only started
#     when they can finish and a killed job loses at most the unit in flight. Left
#     out (the default, i.e. running this by hand in a shell you already hold), the
#     guard is disabled exactly as before -- --min-minutes 0.
#   * it stops entering new checkpoints once the remaining clock cannot hold the
#     cheapest unit plus the merge, instead of churning through the rest of the
#     list starting things it cannot finish.
#   * it EXITS 17 when the list still owes steps, and 0 only when the list is
#     done. That is the signal the chain runs on: submit_job's
#     --autoresume_ignore_failure requeues the job on a non-zero exit and stops on
#     a zero one, so the chain ends by itself the moment the work does.
#
# Nothing is held in memory between links. A step counts as done when its step
# file exists, a unit when its marker exists, and a merge when its .complete
# sentinel does -- all on /lustre, all written the moment they are earned. So a
# job killed at the wall clock costs the unit it was in the middle of and nothing
# else, and the next link resumes at that unit rather than at the checkpoint.
#
# The furthest-first order is recomputed each link from the same on-disk anchors,
# so it is stable: a partially evaluated step is still the first thing the next
# link picks up.
#
# A chain that banks nothing is stopped rather than repeated forever. See
# CHAIN_STATE below.
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
# Sized to the allocation, which on 8 GPUs means suite banking either way: the
# whole natural suite at 300 is ~16 min there and the three suites together ~28,
# so they fit an hour with the merge and there is nothing to gain from paying four
# extra model loads. `auto` is what makes that a derived fact rather than an
# assumption -- with no --job-minutes there is no window to size against and it
# resolves to exactly the `suite` this used to hardcode, and at --num-gpus 1 (where
# mmstar alone is 66 min) it correctly gives per-task units instead.
BANK=${BANK:-auto}

# Minutes of wall clock this allocation has, or 0 for "as long as it takes". See
# WALL CLOCK above: this is what turns the script from one long run into one link
# of a chain.
JOB_MINUTES=${JOB_MINUTES:-0}
START_EPOCH=$(date +%s)
# The same two figures run_bench_eval.sh reserves, repeated here because the
# decision "is there room for another checkpoint?" is taken before it is called.
# MERGE_MINUTES is what entering an unmerged checkpoint costs before any unit can
# be banked; SAFETY_MARGIN covers container startup and teardown.
MERGE_MINUTES=10
SAFETY_MARGIN=5
# How many links in a row may bank NOTHING before the chain is stopped. Without
# this, anything that fails the same way every time -- a broken env, a checkpoint
# that cannot be merged, an lmms-eval that dies on load -- is a job that exits
# "there is still work to do", is requeued, fails again, and holds 8 GPUs on a
# loop that nobody is watching. Progress means a step file or a banked unit that
# was not there when this link started; two links that produce neither is not bad
# luck.
MAX_NOPROGRESS=${MAX_NOPROGRESS:-2}
# Exit status meaning "the list still owes steps, run another link". Any non-zero
# value would do as far as submit_job is concerned; a distinctive one keeps it
# from being read as a crash in a log.
EXIT_MORE=17

SCRIPT_DIR=$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")" && pwd)
REPO=${REPO:-/home/uberger/scratch/research/saliency_r1}
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
        # RUN, STEPS and NUM_GPUS have always been settable from the environment.
        # They are flags as well because the launcher has to pass them through a
        # `submit_job -c "bash -c '...'"` string, where an env prefix is one more
        # layer of quoting to get wrong.
        --run)          RUN="$2";          shift 2 ;;
        --steps)        STEPS_IN="$2";     shift 2 ;;
        --num-gpus)     NUM_GPUS="$2";     shift 2 ;;
        --job-minutes)  JOB_MINUTES="$2";  shift 2 ;;
        --natural-n)    NATURAL_N="$2";    shift 2 ;;
        --nonnatural-n) NONNATURAL_N="$2"; shift 2 ;;
        --task-n)       TASK_N+=("$2");    shift 2 ;;
        --bank)         BANK="$2";         shift 2 ;;
        # Push step files that are already on disk and evaluate nothing. For a push
        # that failed at the time (no network, a bad run id): re-running the list
        # normally would skip those steps as already scored and push nothing.
        --push-only) PUSH_ONLY=true; shift ;;
        -h|--help) sed -n '2,96p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

# After the loop, because --run is one of the things the loop can set.
RUN_DIR="$REPO/checkpoint/$RUN"

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

# ---------- what one checkpoint costs, and what fits in this allocation ----------
# The same plan run_bench_eval.sh will draw its units from, read once here against
# the WHOLE allocation. Two numbers come out of it and both are needed before any
# eval is started:
#
#   UNIT_MIN_MINUTES  the cheapest unit. Entering a checkpoint costs this plus the
#                     merge before anything at all can be banked, so it is the
#                     threshold below which starting another checkpoint is pure
#                     loss -- the GPUs would be spent on a merge whose results are
#                     thrown away at the wall clock.
#   CKPT_MINUTES      every unit plus the merge, i.e. one checkpoint end to end.
#                     Only used to say how many allocations the list needs.
#
# Against the whole allocation, not the time left: see --alloc-minutes in
# run_bench_eval.sh for why the granularity has to be fixed for the job.
WINDOW=0
(( JOB_MINUTES > 0 )) && WINDOW=$(( JOB_MINUTES - SAFETY_MARGIN ))
read -r UNIT_MIN_MINUTES UNIT_SUM_MINUTES UNIT_COUNT < <(
    "$PY_BIN" "$SCRIPT_DIR/bench_eval.py" --plan --bank "$BANK" \
        --sample-n "$SAMPLE_N_JSON" --num-gpus "$NUM_GPUS" --window "$WINDOW" |
    awk -F'\t' 'NR==1||$3<min{min=$3} {sum+=$3; n++} END{print (n?min:0), sum+0, n+0}'
) || { echo "error: could not plan the banking units" >&2; exit 2; }
(( UNIT_COUNT > 0 )) || { echo "error: could not plan the banking units" >&2; exit 2; }
CKPT_MINUTES=$(( UNIT_SUM_MINUTES + MERGE_MINUTES ))
# The margin is in here because run_bench_eval.sh applies its own to whatever clock
# it is handed. Without it this script would let a step in with exactly enough
# time, the eval would subtract five minutes, decline to start anything and exit --
# a whole invocation spent regenerating task configs and reading its own mind.
MIN_STEP_MINUTES=$(( UNIT_MIN_MINUTES + MERGE_MINUTES + SAFETY_MARGIN ))

# Minutes of this allocation still to come. Unlimited without --job-minutes, which
# is the shell-in-an-salloc case: nothing here decides anything then.
minutes_left() {
    (( JOB_MINUTES > 0 )) || { echo 99999; return; }
    local left=$(( JOB_MINUTES - ( ( $(date +%s) - START_EPOCH ) / 60 ) ))
    (( left < 0 )) && left=0
    echo "$left"
}

# ---------- how many links in a row have banked nothing ----------
# One line, "<progress> <consecutive links with none>", beside the results it
# describes so it travels with them and survives every job. Progress is counted,
# not timestamped: a step file that appeared, or a unit banked in a shadow, is
# something this link earned; a clock that moved on is not.
CHAIN_STATE="$PROFILE_DIR/.steps_chain_progress"

progress_units() {
    local -a found=()
    shopt -s nullglob
    found=( "$PROFILE_DIR"/step-*.json
            "$SHADOW_ROOT/$RUN"-cp*/bench_eval/partial/step-*/*.json
            "$SHADOW_ROOT/$RUN"-cp*/bench_eval/*/partial/step-*/*.json )
    shopt -u nullglob
    echo "${#found[@]}"
}

PREV_PROGRESS=-1
NOPROGRESS=0
if [[ -f "$CHAIN_STATE" ]]; then
    read -r PREV_PROGRESS NOPROGRESS < "$CHAIN_STATE"
    [[ "$PREV_PROGRESS" =~ ^[0-9]+$ ]] || PREV_PROGRESS=-1
    [[ "$NOPROGRESS"    =~ ^[0-9]+$ ]] || NOPROGRESS=0
fi

# Judged at STARTUP, on what the link before this one left behind, and not at the
# end of this one. The difference matters: the exits that most need bounding are
# the ones that never reach the end -- a missing conda env, a run directory that
# is not there, an lmms-eval that dies on import. Those exit non-zero in seconds,
# and with --autoresume_ignore_failure a non-zero exit is a request for another
# job, so nothing downstream of them would ever run. Counted here, a link that
# fails before it does anything still costs the chain one of its lives.
if (( JOB_MINUTES > 0 )) && ! $DRY_RUN && ! $PUSH_ONLY; then
    THIS_PROGRESS=$(progress_units)
    if (( THIS_PROGRESS > PREV_PROGRESS )); then
        NOPROGRESS=0
    else
        NOPROGRESS=$(( NOPROGRESS + 1 ))
    fi
    printf '%s %s\n' "$THIS_PROGRESS" "$NOPROGRESS" > "$CHAIN_STATE"
    if (( NOPROGRESS > MAX_NOPROGRESS )); then
        echo "STOPPING THE CHAIN: the last $NOPROGRESS jobs banked no step and no unit between" >&2
        echo "them. Something is failing the same way every time -- read one of their logs" >&2
        echo "before asking for more GPUs. Exiting 0 so nothing is requeued;" >&2
        echo "launch_bench_eval_steps.sh clears $CHAIN_STATE and starts a fresh chain." >&2
        exit 0
    fi
fi

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
if (( ${#STEPS[@]} == 0 )); then
    echo "nothing to do: every checkpoint in $RUN_DIR is already scored"
    # The launcher reads this line to decide whether submitting a job is worth it,
    # and where the chain counter lives. Keep the two in step: it is a contract
    # between these two files and nothing else.
    $DRY_RUN && echo "PLAN steps=0 per_checkpoint_min=0 jobs_needed=0 chain_state=$CHAIN_STATE"
    (( JOB_MINUTES > 0 )) && rm -f "$CHAIN_STATE"
    exit 0
fi

# How many allocations of this size the whole list needs, rounded up. The units
# pack across job boundaries -- a job that cannot fit another checkpoint still
# banks whatever units it can of the next one -- so this is a count of allocations
# worth of GPU time, not a promise that each job finishes a whole number of steps.
JOBS_NEEDED=0
(( WINDOW > 0 )) && JOBS_NEEDED=$(( ( ${#STEPS[@]} * CKPT_MINUTES + WINDOW - 1 ) / WINDOW ))

echo "=========================================================================="
echo "Run:      $RUN"
echo "Steps:    ${#STEPS[@]} ($STEPS_SOURCE$([[ "$STEPS_SOURCE" == detected ]] && echo ', furthest-first'))   ${STEPS[*]}"
echo "GPUs:     $NUM_GPUS   (serial, one checkpoint at a time)"
echo "Sample:   $PROFILE   (natural=$NATURAL_N, non-natural=$NONNATURAL_N per benchmark)"
echo "Banking:  per $BANK -> $UNIT_COUNT unit(s), the cheapest ${UNIT_MIN_MINUTES}min"
echo "Cost:     ~${CKPT_MINUTES}min per checkpoint (${UNIT_SUM_MINUTES} of units + ${MERGE_MINUTES} to merge), on $NUM_GPUS GPU$( (( NUM_GPUS == 1 )) || echo s)"
if (( JOB_MINUTES > 0 )); then
    echo "Clock:    ${JOB_MINUTES}min allocation, ${WINDOW}min of it usable"
    echo "          this list needs ~$JOBS_NEEDED of them; progress is banked per unit, so a job"
    echo "          killed at the wall clock loses only the unit it was in the middle of"
    echo "          exit $EXIT_MORE = steps still owed (run another), exit 0 = the list is done"
    (( NOPROGRESS > 0 )) && \
        echo "          WARNING: the $NOPROGRESS link(s) before this one banked nothing ($MAX_NOPROGRESS stops the chain)"
else
    echo "Clock:    unlimited (no --job-minutes) -- run to the end of the list or be killed part-way"
fi
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
declare -a DONE=() SKIPPED=() FAILED=() UNPUSHED=() DEFERRED=() PARTIAL=()

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

# The simulated clock --dry-run spends; the real one is minutes_left(). The whole
# allocation, not WINDOW: the margin is already inside MIN_STEP_MINUTES, and
# charging it twice would show a job stopping a checkpoint before it really does.
DRY_LEFT=$JOB_MINUTES

# Steps this allocation will not reach, once the clock says so. Filled by the loop
# below and reported at the end; what the chain actually runs on is the on-disk
# state, not this list.
defer_rest() {
    local from="$1" seen=false s
    for s in "${STEPS[@]}"; do
        [[ "$s" == "$from" ]] && seen=true
        $seen && DEFERRED+=("$s")
    done
}

for STEP in "${STEPS[@]}"; do
    echo ""
    # Not "Starting step": this heading is also what the clock guard below and the
    # skips above print under, and neither of them starts anything.
    echo "Step $STEP"

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
        # With an allocation to fit into, say which steps THIS job would reach and
        # which are a later job's -- charging the full per-checkpoint estimate even
        # though the units pack across the boundary, so the split shown is the
        # pessimistic one.
        if (( JOB_MINUTES > 0 && DRY_LEFT < MIN_STEP_MINUTES )); then
            echo "  ${DRY_LEFT}min left of the allocation -- this step and the rest are a later job's"
            defer_rest "$STEP"
            break
        fi
        DRY_NOTE=""
        if (( JOB_MINUTES > 0 && CKPT_MINUTES > DRY_LEFT )); then
            DRY_NOTE="   (~${CKPT_MINUTES}min against ${DRY_LEFT}min left -- part-scored here, finished by a later job)"
        elif (( JOB_MINUTES > 0 )); then
            DRY_NOTE="   (~${CKPT_MINUTES}min of ${DRY_LEFT}min left)"
        fi
        echo "  would evaluate $CKPT on $NUM_GPUS GPUs -> $RESULT$DRY_NOTE"
        DRY_LEFT=$(( DRY_LEFT - CKPT_MINUTES ))
        (( DRY_LEFT < 0 )) && DRY_LEFT=0
        DONE+=("$STEP (dry-run)")
        continue
    fi

    # Is there room for another checkpoint? Entering one costs the merge plus the
    # cheapest unit before a single result is banked, so below that the GPUs would
    # be spent on a merge that the wall clock throws away. Stop instead, and let
    # the next link start this step with a whole allocation in front of it.
    #
    # A step already part-way through is not re-checked against the merge: its
    # merged model is on disk with a .complete sentinel beside it, and
    # run_bench_eval.sh charges nothing for reusing it. That is deliberately not
    # modelled here -- the estimate is only ever used to decide whether to stop,
    # and stopping one checkpoint early is cheap while overrunning is not.
    LEFT=$(minutes_left)
    if (( JOB_MINUTES > 0 && LEFT < MIN_STEP_MINUTES )); then
        echo "  ${LEFT}min of this allocation left, and the cheapest thing that could be banked"
        echo "  here needs ${MIN_STEP_MINUTES} (merge $MERGE_MINUTES + unit $UNIT_MIN_MINUTES + margin $SAFETY_MARGIN) -- stopping, the next job takes it"
        defer_rest "$STEP"
        break
    fi

    mkdir -p "$SHADOW"
    ln -sfn "$CKPT" "$SHADOW/checkpoint-$STEP"

    # --force means measure it again, and the shadow is reused between jobs, so it
    # may still hold the step file from the last time. run_bench_eval.sh would find
    # it, conclude nothing is pending, and the copy below would report that old
    # number as a fresh measurement.
    $FORCE && rm -f "$(profile_dir_of "$SHADOW")/step-$STEP.json"

    # Two different clocks, and run_bench_eval.sh needs both: --job-minutes is what
    # is left NOW, which decides whether a unit may be started, and --alloc-minutes
    # is the size of the allocation, which decides how big the units are. Without
    # --job-minutes at all (a shell that already holds the GPUs) the guard is off,
    # which is what --min-minutes 0 has always meant here.
    declare -a CLOCK_ARGS=(--min-minutes 0)
    (( JOB_MINUTES > 0 )) && CLOCK_ARGS=(--job-minutes "$LEFT" --alloc-minutes "$JOB_MINUTES")

    bash "$SCRIPT_DIR/run_bench_eval.sh" --run-dir "$SHADOW" --num-gpus "$NUM_GPUS" \
        --every 1 --bank "$BANK" "${CLOCK_ARGS[@]}" "${SIZE_ARGS[@]}"

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
        # No step file is two different situations, and calling both "failed" is
        # how a chain that is working perfectly well gets read as broken. A step
        # with units banked ran out of clock and will be resumed at those units;
        # one with none banked did not get that far.
        shopt -s nullglob
        declare -a BANKED=( "$SHADOW_PROFILE/partial/step-$STEP"/*.json )
        shopt -u nullglob
        if (( ${#BANKED[@]} > 0 )); then
            echo "  step $STEP part-scored: ${#BANKED[@]}/$UNIT_COUNT unit(s) banked, the merge kept -- the next job resumes here"
            PARTIAL+=("$STEP (${#BANKED[@]}/$UNIT_COUNT)")
        else
            echo "  step $STEP produced no result -- see the output above" >&2
            FAILED+=("$STEP")
        fi
    fi
done

# What the list still owes, read off disk rather than off the counters above: a
# step is done when its step file exists, whoever wrote it. Steps whose checkpoint
# is missing are left out -- they cannot be scored by any number of further jobs,
# so counting them as owed would keep a chain alive that has nothing to do.
declare -a REMAINING=()
for s in "${STEPS[@]}"; do
    [[ -f "$PROFILE_DIR/step-$s.json" ]] && continue
    [[ -f "$RUN_DIR/checkpoint-$s/adapter_config.json" \
       && -s "$RUN_DIR/checkpoint-$s/adapter_model.safetensors" ]] || continue
    REMAINING+=("$s")
done

echo ""
echo "=========================================================================="
echo "Evaluated: ${#DONE[@]}   ${DONE[*]:-}"
(( ${#PARTIAL[@]} > 0 )) && \
    echo "Part-done: ${#PARTIAL[@]}   ${PARTIAL[*]}   (units banked, resumed by the next job)"
echo "Skipped:   ${#SKIPPED[@]}   ${SKIPPED[*]:-}"
echo "Failed:    ${#FAILED[@]}   ${FAILED[*]:-}"
(( ${#DEFERRED[@]} > 0 )) && \
    echo "Deferred:  ${#DEFERRED[@]}   ${DEFERRED[*]}   (not started -- out of wall clock)"
if (( ${#UNPUSHED[@]} > 0 )); then
    echo "Unpushed:  ${#UNPUSHED[@]}   ${UNPUSHED[*]}   (scored, but WandB rejected them)"
    echo "           retry:  STEPS=\"${UNPUSHED[*]}\" bash $0 --push-only --wandb-run-id $WANDB_RUN_ID"
fi

# Outside an allocation this is a single run and the old contract holds: non-zero
# means something went wrong.
if (( JOB_MINUTES == 0 )) || $DRY_RUN || $PUSH_ONLY; then
    $DRY_RUN && echo "PLAN steps=${#STEPS[@]} per_checkpoint_min=$CKPT_MINUTES jobs_needed=$JOBS_NEEDED chain_state=$CHAIN_STATE"
    echo "=========================================================================="
    (( ${#FAILED[@]} + ${#UNPUSHED[@]} == 0 ))
    exit $?
fi

# Inside one, the exit status is not a verdict on this job, it is the answer to
# "is there more?" -- which is what submit_job's autoresume requeues on. An
# unpushed step does NOT keep the chain going: the result it took GPU hours for is
# on disk, and --push-only ships it later without an allocation.
echo "Owed:      ${#REMAINING[@]}   ${REMAINING[*]:-}"

if (( ${#REMAINING[@]} == 0 )); then
    echo "The list is complete -- exiting 0, which ends the chain."
    echo "=========================================================================="
    rm -f "$CHAIN_STATE"
    exit 0
fi

echo "Still owed ${#REMAINING[@]} step(s) -- exiting $EXIT_MORE so the next job in the chain runs."
echo "=========================================================================="
exit $EXIT_MORE
