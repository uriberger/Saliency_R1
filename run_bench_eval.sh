#!/bin/bash
# Evaluate a GRPO run's checkpoints on the two mini test suites, oldest first,
# then exit. This is the body of the single-GPU job that watch_bench_evals.sh
# submits.
#
# It drains, it does not idle: when no checkpoint is waiting it returns
# immediately and releases the GPUs. The dispatcher submits a new job the next
# time work appears, so nothing is held while training is still producing the
# next checkpoint.
#
# For each pending checkpoint:
#   1. merge the LoRA adapter into the base model (a checkpoint is 116 MB of
#      adapter; lmms-eval needs a standalone model)
#   2. run each banking unit that fits in the remaining wall clock
#   3. reduce the results to bench_eval/step-<N>.json, which the trainer picks up
#      and logs to WandB, and store the per-item rows beside it
#   4. delete the merged model -- 16 GB each, and /lustre is not roomy
#
# Progress is banked per UNIT, not per checkpoint. A job that runs out of wall
# clock after two of three units records what those produced under
# bench_eval/[<profile>/]partial/step-<N>/ and leaves the merged model in place;
# the next job reuses both and picks up at the units still owed. That is what
# makes a one-hour allocation useful: the unit of work it has to finish is a
# suite or a task, not a whole checkpoint plus the merge.
#
# What a unit is depends on --bank:
#
#   --bank suite   (default)  natural / mme-realworld / non-natural, three units
#                             of 14-40 min at 100 documents per benchmark
#   --bank task               one unit per benchmark, thirteen of 6-66 min
#
# Suite banking is right at 100 documents and wrong above it: the natural suite
# at 300 is ~105 minutes and can never finish inside a one-hour allocation, so it
# would be started, killed and repeated forever, banking nothing. Task banking
# costs four extra model loads (~6 min) and makes every unit finishable. The
# minute figures come from eval_mini/benchmarks.py, measured over the 295
# results.json this has already produced -- not guessed.
#
# Units are attempted largest-first, and a unit that does not fit the remaining
# clock is SKIPPED rather than ending the job: the smaller ones behind it usually
# still fit, and the merge is only paid once. The job stops when nothing left
# fits.
#
# What is NOT banked is a partial step-<N>.json. It is written only once every
# unit is in hand, because a half-measured checkpoint on the benchmark curve is
# indistinguishable from a real one. The merged model is deleted at that same
# moment -- it exists only to serve the units still owed for that checkpoint.
#
# Everything runs through vlm_reasoning's launch_lmms_eval_job.sh --direct, so a
# mini benchmark is evaluated with exactly the recipe the full test suite uses
# (--r1-mode: the Saliency-R1 system prompt, repetition_penalty 1.05,
# max_new_tokens 4096).
#
# Usage:
#   bash run_bench_eval.sh --run-dir CKPT_DIR [--num-gpus 1] [--every 100]
#                          [--natural-n 300] [--nonnatural-n 100]
#                          [--task-n mmstar=200] [--bank task] [--steps 100,200]
#
# Environment:
#   OPENAI_API_KEY / NVIDIA_API_KEY   needed by mathvista's llm_as_judge metric
#   HF_TOKEN
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")" && pwd)
export SCRIPT_DIR
REPO=${REPO:-/home/uberger/scratch/research/saliency_r1}
VLM_REASONING=${VLM_REASONING:-/home/uberger/scratch/research/vlm_reasoning}
LMMS_EVAL_DIR=${LMMS_EVAL_DIR:-/home/uberger/scratch/research/lmms-eval}
CONDA_SH=/home/uberger/scratch/miniconda3/etc/profile.d/conda.sh

RUN_DIR=""
NUM_GPUS=1
EVERY=100
# Sample sizes. --sample-n is the old spelling and sets both suites; the per-suite
# flags exist because the effects being chased are on the natural benchmarks and
# the non-natural suite generates 4x the tokens per document, so raising both
# would spend most of the extra GPU time where it buys nothing.
SAMPLE_N=""
NATURAL_N=""
NONNATURAL_N=""
declare -a TASK_N=()
BANK=suite
# Explicit list of steps to evaluate. Overrides the `% EVERY` rule, which is what
# lets rerun_bench_evals.sh ask for exactly the checkpoints that already have a
# result rather than every checkpoint on disk.
STEPS_FILTER=""
HARVEST=true
# Below this much wall-clock left, start no further unit. Left empty here and
# filled in per unit from the measured cost table -- see the plan below. Setting
# it pins every unit to one flat figure instead; 0 disables the guard entirely.
MIN_MINUTES=""
MAX_CHECKPOINTS=0   # 0 = drain everything that is pending
# Total wall clock this job was given, in minutes. Set by the dispatcher, because
# asking Slurm does not work from inside submit_job's container -- see minutes_left.
JOB_MINUTES=0
DRY_RUN=false
# When the script started, which is what --job-minutes is measured from.
START_EPOCH=$(date +%s)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-dir)         RUN_DIR="$2";         shift 2 ;;
        --num-gpus)        NUM_GPUS="$2";        shift 2 ;;
        --every)           EVERY="$2";           shift 2 ;;
        --sample-n)        SAMPLE_N="$2";        shift 2 ;;
        --natural-n)       NATURAL_N="$2";       shift 2 ;;
        --nonnatural-n)    NONNATURAL_N="$2";    shift 2 ;;
        --task-n)          TASK_N+=("$2");       shift 2 ;;
        --bank)            BANK="$2";            shift 2 ;;
        --steps)           STEPS_FILTER="$2";    shift 2 ;;
        --no-harvest)      HARVEST=false;        shift ;;
        --min-minutes)     MIN_MINUTES="$2";     shift 2 ;;
        --job-minutes)     JOB_MINUTES="$2";     shift 2 ;;
        --max-checkpoints) MAX_CHECKPOINTS="$2"; shift 2 ;;
        --dry-run)         DRY_RUN=true;         shift ;;
        -h|--help)         sed -n '2,60p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

# The merge is a file copy, not GPU work, so it costs the same at any allocation
# size. Only charged against the clock when there is no merged model to reuse.
MERGE_MINUTES=10

[[ -n "$RUN_DIR" ]] || { echo "error: --run-dir is required" >&2; exit 2; }
[[ -d "$RUN_DIR" ]] || { echo "error: no such run dir: $RUN_DIR" >&2; exit 2; }
RUN_DIR=$(cd "$RUN_DIR" && pwd)

# The eval nodes have no internet, so every benchmark is served from the HF cache
# whether or not this is set. Pin it anyway, because the two dispatchers that
# submit this job do NOT agree on it and the difference is not cosmetic:
# watch_bench_evals.sh started on the login node leaves it unset, while the copy
# launch_grpo_qwen3_overlap_colocated_job.sh starts on the compute node inherits
# `export HF_HUB_OFFLINE=1` from the training job and passes it on through
# `#SBATCH --export=ALL`. Unset and set take different code paths inside
# `datasets`, and the offline one aborted the entire non-natural suite on
# visulogic (see `local_data` in eval_mini/benchmarks.py) -- for ~1100 steps of
# two runs the eval jobs recorded natural-only results and nothing said so.
#
# Pinned to 1, not 0: with no route to the hub, offline is the truthful value and
# the one both suites are now built to load under. 0 would only buy connection
# attempts that must fail.
export HF_HUB_OFFLINE=1

BENCH_DIR="$RUN_DIR/bench_eval"
TASK_DIR="$BENCH_DIR/tasks"
MERGE_ROOT="${BENCH_MERGE_ROOT:-$REPO/checkpoint/_bench_eval}"
mkdir -p "$BENCH_DIR" "$MERGE_ROOT"

# ---------- mini task configs ----------
# Regenerated every job rather than committed: they carry the absolute path of the
# lmms-eval clone, and a stale path would silently evaluate the wrong task file.
source "$CONDA_SH"
set +u; conda activate lmms_eval; set -u

declare -a GEN_ARGS=()
[[ -n "$SAMPLE_N" ]]     && GEN_ARGS+=(--n "$SAMPLE_N")
[[ -n "$NATURAL_N" ]]    && GEN_ARGS+=(--natural-n "$NATURAL_N")
[[ -n "$NONNATURAL_N" ]] && GEN_ARGS+=(--nonnatural-n "$NONNATURAL_N")
for t in ${TASK_N[@]+"${TASK_N[@]}"}; do GEN_ARGS+=(--task-n "$t"); done

python "$SCRIPT_DIR/eval_mini/make_mini_tasks.py" \
    --out-dir "$TASK_DIR" --lmms-eval-dir "$LMMS_EVAL_DIR" \
    ${GEN_ARGS[@]+"${GEN_ARGS[@]}"} || exit 1

# The generated configs are the authority on what this job samples. Reading the
# sizes back out of sample_n.json rather than re-deriving them from the flags
# means the banking markers, the step file and the per-item store cannot disagree
# with what was actually generated.
SAMPLE_N_JSON=$(cat "$TASK_DIR/sample_n.json")

# Where results for this sample profile live. The 100/100 default keeps the flat
# layout every existing reader (including the trainer's callback) expects; any
# other profile gets a subdirectory, so the two can never be read as one curve.
PROFILE_DIR=$(python - "$BENCH_DIR" "$SAMPLE_N_JSON" <<'PY'
import json, os, sys
sys.path.insert(0, os.path.join(os.environ["SCRIPT_DIR"], "eval_mini"))
from benchmarks import profile_dir, profile_name
print(profile_dir(sys.argv[1], json.loads(sys.argv[2])))
print(profile_name(json.loads(sys.argv[2])))
PY
) || exit 1
PROFILE=$(sed -n 2p <<< "$PROFILE_DIR")
PROFILE_DIR=$(sed -n 1p <<< "$PROFILE_DIR")

# Where a finished unit is recorded so a later job can skip it. One directory per
# step, deleted the moment that step's step-<N>.json is written. Under the profile
# directory, so units banked at one sample size are not offered to a job asking
# for another.
PARTIAL_DIR="$PROFILE_DIR/partial"
mkdir -p "$PROFILE_DIR" "$PARTIAL_DIR"

# ---------- the banking units ----------
# tag <TAB> tasks <TAB> minutes <TAB> extra args, largest first. The cost table
# and the knowledge that mme-realworld needs its own resolution both live in
# eval_mini/benchmarks.py, next to the benchmark table they describe.
declare -a UNIT_TAG=() UNIT_TASKS=() UNIT_MINUTES=() UNIT_EXTRA=()
while IFS=$'\t' read -r tag tasks minutes extra; do
    [[ -n "$tag" ]] || continue
    UNIT_TAG+=("$tag"); UNIT_TASKS+=("$tasks")
    UNIT_MINUTES+=("${MIN_MINUTES:-$minutes}"); UNIT_EXTRA+=("$extra")
done < <(python "$SCRIPT_DIR/bench_eval.py" --plan --bank "$BANK" \
             --sample-n "$SAMPLE_N_JSON" --num-gpus "$NUM_GPUS")
(( ${#UNIT_TAG[@]} > 0 )) || { echo "error: could not plan the units" >&2; exit 1; }

# ---------- banking a finished unit ----------
# A unit is banked by recording WHERE lmms-eval put its results, not by copying
# them. The per-task sample sizes are recorded alongside because they change what
# the mini task contains without changing lmms-eval's output directory: without
# this, results produced at n=100 would be silently reused for a job asking for
# n=300 and the curve would mix two different benchmarks.
#
# rerun_bench_evals.sh writes markers in this same format to carry a suite over
# from an earlier profile instead of re-generating it, so the shape is a contract,
# not an implementation detail: {"sample_n": {task: n}, "results": path}, with an
# optional "carried_from" naming the profile it came from.
bank_unit() {
    local marker="$1" tasks="$2" results="$3"
    mkdir -p "$(dirname "$marker")"
    python - "$marker" "$tasks" "$results" "$SAMPLE_N_JSON" <<'PY'
import json, sys
marker, tasks, results, sizes = sys.argv[1:5]
sizes = json.loads(sizes)
json.dump({"sample_n": {t: sizes[t] for t in tasks.split(",") if t in sizes},
           "results": results}, open(marker, "w"))
PY
}

# Echo the banked results path if it is still usable, else fail. "Usable" has to
# be checked, not assumed: the referenced file may have been cleaned up, a
# results.json from a run killed mid-write parses as truncated garbage, and a
# marker left by an earlier job may describe a different sample size.
banked_unit() {
    local marker="$1" tasks="$2"
    [[ -f "$marker" ]] || return 1
    python - "$marker" "$tasks" "$SAMPLE_N_JSON" <<'PY' || return 1
import json, os, sys
marker, tasks, sizes = sys.argv[1:4]
sizes = json.loads(sizes)
try:
    banked = json.load(open(marker))
except Exception:
    sys.exit(1)
have = banked.get("sample_n") or {}
for task in tasks.split(","):
    if task in sizes and have.get(task) != sizes[task]:
        sys.exit(1)
path = banked.get("results", "")
if not os.path.isfile(path):
    sys.exit(1)
try:
    results = json.load(open(path)).get("results") or {}
except Exception:
    sys.exit(1)
# The file has to contain THESE tasks, not merely be a valid results.json: a
# marker pointing at a neighbouring unit's output would otherwise be accepted and
# the benchmark it claims to cover would be silently absent from the step file.
if not all(task in results for task in tasks.split(",")):
    sys.exit(1)
print(path)
PY
}

# ---------- what still needs evaluating ----------
# Step 0 is the model the run started from, recorded by the launcher. It is already
# a full model, so it is scored directly -- no adapter to merge, and nothing to
# delete afterwards.
BASE_MODEL_FILE="$BENCH_DIR/base_model.txt"

wanted_step() {
    [[ -z "$STEPS_FILTER" ]] && { (( $1 % EVERY == 0 )); return; }
    [[ ",$STEPS_FILTER," == *",$1,"* ]]
}

pending_steps() {
    local d step
    if [[ -f "$BASE_MODEL_FILE" && ! -f "$PROFILE_DIR/step-0.json" ]] && wanted_step 0; then
        echo 0
    fi
    for d in "$RUN_DIR"/checkpoint-*; do
        [[ -d "$d" ]] || continue
        step=${d##*checkpoint-}
        [[ "$step" =~ ^[0-9]+$ ]] || continue
        wanted_step "$step" || continue
        [[ -f "$PROFILE_DIR/step-$step.json" ]] && continue
        # Both files, not just the config: a checkpoint being written right now
        # would otherwise be picked up and merged from a partial adapter.
        [[ -f "$d/adapter_config.json" && -s "$d/adapter_model.safetensors" ]] || continue
        echo "$step"
    done | sort -n
}

# Minutes of wall clock left in this allocation. Unlimited outside Slurm.
#
# --job-minutes first, and squeue only as a fallback, because squeue is not
# reachable from where this actually runs: submit_job executes the command in a
# container that does not mount /cm/shared, so the slurm client is absent and the
# query silently returns nothing. Believing the 99999 fallback in that situation is
# what let earlier jobs run head-first into the wall clock and TIMEOUT rather than
# stopping at a unit boundary.
#
# The countdown starts when this script does, which is a minute or two after the
# allocation did -- the wrapper has to stage a container first. SAFETY_MARGIN
# covers that gap plus teardown, so the estimate stays on the pessimistic side.
SAFETY_MARGIN=5
minutes_left() {
    local left
    if (( JOB_MINUTES > 0 )); then
        local elapsed=$(( ( $(date +%s) - START_EPOCH ) / 60 ))
        left=$(( JOB_MINUTES - elapsed - SAFETY_MARGIN ))
        (( left < 0 )) && left=0
        echo "$left"
        return
    fi
    [[ -n "${SLURM_JOB_ID:-}" ]] || { echo 99999; return; }
    left=$(squeue -h -j "$SLURM_JOB_ID" -o "%L" 2>/dev/null | tr -d ' ')
    [[ -n "$left" ]] || { echo 99999; return; }
    python - "$left" <<'PY'
import re, sys
# Slurm TIME_LEFT is [[days-]hours:]minutes:seconds
text = sys.argv[1]
days, _, rest = text.partition("-")
if not rest:
    days, rest = 0, text
parts = [int(p) for p in rest.split(":")]
while len(parts) < 3:
    parts.insert(0, 0)
h, m, s = parts
print(int(days) * 1440 + h * 60 + m + s // 60)
PY
}

# ---------- evaluate one unit ----------
# Echoes the results.json lmms-eval produced, or nothing if the run failed. Each
# unit gets its own --tag so its results land in their own directory: attributing
# a unit by "the newest file in a shared directory" would quietly mislabel
# everything the first time one failed and left the previous one newest.
# Exit status is three-valued, because "no results" and "no time" have to be told
# apart by the caller: 0 results echoed, 1 the unit failed, 2 not enough wall
# clock for this one (nothing attempted, nothing lost).
run_unit() {
    local model="$1" tag="$2" tasks="$3" need="$4"; shift 4
    local marker="$PARTIAL_DIR/step-$STEP/$tag.json"
    local banked
    if banked=$(banked_unit "$marker" "$tasks"); then
        echo "[step $STEP] $tag: reusing the results banked by an earlier job" >&2
        printf '%s\n' "$banked"
        return 0
    fi

    local left; left=$(minutes_left)
    if (( need > 0 && left < need )); then
        echo "[step $STEP] $tag: ${left}min left, this unit needs ${need} -- skipping it" >&2
        return 2
    fi

    # An explicit marker for report_bench_evals.sh. The tag also appears inside
    # lmms-eval's output paths, which is why the report cannot simply grep for it:
    # the model directory name contains the tag of whichever job merged it.
    echo "[step $STEP] unit $tag: starting ($tasks), ${left}min left, budgeted $need" >&2

    local -a common=(--model "$model" --tasks "$tasks" --max-new-tokens 4096
                     --num-gpus "$NUM_GPUS" --direct --r1-mode --tag "r1_$tag" "$@")
    local out_dir
    out_dir=$(bash "$VLM_REASONING/scripts/slurm/launch_lmms_eval_job.sh" \
        "${common[@]}" --print-output-dir) || return 1

    bash "$VLM_REASONING/scripts/slurm/launch_lmms_eval_job.sh" \
        "${common[@]}" --include_path "$TASK_DIR" >&2 || return 1

    local results
    results=$(ls -t "$out_dir"/*/*_results.json 2>/dev/null | head -1)
    [[ -n "$results" ]] || return 1
    bank_unit "$marker" "$tasks" "$results"
    printf '%s\n' "$results"
}

# ---------- drain ----------
RUN_NAME=$(basename "$RUN_DIR")
done_count=0
echo "=========================================================================="
echo "Run dir:    $RUN_DIR"
echo "Pending:    $(pending_steps | tr '\n' ' ')"
echo "GPUs:       $NUM_GPUS   cadence: $([[ -n "$STEPS_FILTER" ]] && echo "steps $STEPS_FILTER" || echo "every $EVERY steps")"
echo "Profile:    $PROFILE   ->  $PROFILE_DIR"
echo "Banking:    per $BANK, ${#UNIT_TAG[@]} unit(s), largest first"
for i in "${!UNIT_TAG[@]}"; do
    printf '  %-14s %3s min  %s\n' "${UNIT_TAG[$i]}" "${UNIT_MINUTES[$i]}" "${UNIT_TASKS[$i]}"
done
echo "Banked:     $PARTIAL_DIR"
echo "=========================================================================="

if $DRY_RUN; then
    echo "[dry-run] would evaluate, in order: $(pending_steps | tr '\n' ' ')"
    echo "[dry-run] wall clock left: $(minutes_left) min (+$MERGE_MINUTES to merge a checkpoint)"
    echo "[dry-run] merged models go to $MERGE_ROOT, kept across jobs while a step is"
    echo "[dry-run] unfinished and deleted once its step-<N>.json is written"
    # Each unit is put through the same validation the job itself would apply, not
    # merely checked for a file: a marker can exist and still be rejected because
    # it names a sample size this job is not asking for, or points at a results
    # file that has been cleaned up. "A marker is present" and "this unit will be
    # skipped" are different statements, and only the second one is useful.
    for s in $(pending_steps); do
        STEP="$s"
        reusable=""; owed=""
        for i in "${!UNIT_TAG[@]}"; do
            if banked_unit "$PARTIAL_DIR/step-$s/${UNIT_TAG[$i]}.json" "${UNIT_TASKS[$i]}" >/dev/null 2>&1
            then reusable+="${UNIT_TAG[$i]} "
            else owed+="${UNIT_TAG[$i]}(${UNIT_MINUTES[$i]}m) "
            fi
        done
        echo "[dry-run] step $s"
        echo "[dry-run]   reusable: ${reusable:-none}"
        echo "[dry-run]   to run:   ${owed:-none}"
    done
    exit 0
fi

while true; do
    STEP=$(pending_steps | head -1)
    if [[ -z "$STEP" ]]; then
        echo "[drain] nothing pending -- exiting so the GPUs go back to the pool"
        break
    fi
    LEFT=$(minutes_left)
    # What entering this checkpoint costs before anything can be banked: the
    # cheapest unit still owed, plus the merge if no earlier job already paid for
    # it. Cheapest, not first: units are attempted largest-first but a job with
    # only 20 minutes left can still bank a small one, and refusing to enter the
    # checkpoint at all would waste that.
    MERGED="$MERGE_ROOT/${RUN_NAME}_cp${STEP}_merged"
    MERGE_DONE="$MERGED.complete"
    NEED=${UNIT_MINUTES[$(( ${#UNIT_MINUTES[@]} - 1 ))]}
    if (( STEP != 0 )) && [[ ! -f "$MERGE_DONE" ]]; then
        NEED=$(( NEED + MERGE_MINUTES ))
    fi
    if (( NEED > 0 && LEFT < NEED )); then
        echo "[drain] only ${LEFT}min of wall clock left (need $NEED to bank anything) -- exiting; the dispatcher will resubmit"
        break
    fi
    if (( MAX_CHECKPOINTS > 0 && done_count >= MAX_CHECKPOINTS )); then
        echo "[drain] hit --max-checkpoints $MAX_CHECKPOINTS -- exiting"
        break
    fi

    echo ""
    echo "--------------------------------------------------------------------------"
    if (( STEP == 0 )); then
        # The baseline: already a standalone model, so there is nothing to merge and,
        # crucially, nothing to delete afterwards -- it is the model training started
        # from, not a temporary copy.
        MERGED=$(< "$BASE_MODEL_FILE")
        MERGED_IS_TEMPORARY=false
        echo "[step 0] baseline: $MERGED   (${LEFT}min left)"
        if [[ ! -d "$MERGED" ]]; then
            echo "[step 0] baseline model not found at $MERGED -- skipping" >&2
            break
        fi
    else
        CKPT="$RUN_DIR/checkpoint-$STEP"
        MERGED_IS_TEMPORARY=true
        # Any merged model still lying around for a DIFFERENT step of this run is
        # from a job that died; it will never be reused, and at 16 GB each they
        # accumulate. Only one step of one run is ever in flight (the dispatcher
        # guarantees it), so this is safe to do unconditionally.
        for stale in "$MERGE_ROOT/${RUN_NAME}"_cp*_merged; do
            [[ -d "$stale" && "$stale" != "$MERGED" ]] || continue
            echo "[step $STEP] removing stale merge $(basename "$stale")"
            rm -rf "$stale" "$stale.complete"
        done
        # The sentinel is what makes reuse safe: a merge killed halfway leaves a
        # directory that looks complete but is not, and loading it would fail (or,
        # worse, load a truncated shard). It is written only after the merge
        # returns, and lives beside the directory rather than inside it so nothing
        # in the model dir is unexpected to from_pretrained.
        if [[ -f "$MERGE_DONE" && -d "$MERGED" ]]; then
            echo "[step $STEP] reusing the merge from an earlier job: $MERGED   (${LEFT}min left)"
            echo "--------------------------------------------------------------------------"
        else
            echo "[step $STEP] merging $CKPT -> $MERGED   (${LEFT}min left)"
            echo "--------------------------------------------------------------------------"
            rm -rf "$MERGED" "$MERGE_DONE"
            if ! bash "$REPO/merge_lora_grpo_qwen3.sh" "$CKPT" "$MERGED"; then
                echo "[step $STEP] merge FAILED -- skipping this checkpoint" >&2
                rm -rf "$MERGED" "$MERGE_DONE"
                # Leave no step file: a later job retries it rather than recording a gap
                # as if it had been measured.
                break
            fi
            touch "$MERGE_DONE"
        fi
    fi
    echo "--------------------------------------------------------------------------"

    set +u; conda activate lmms_eval; set -u
    # Each unit is attempted only if the clock allows, and banked the moment it
    # succeeds. A unit that does not fit is skipped and the next one tried: they
    # are ordered largest-first, so what is left behind is usually the one costly
    # unit and the cheap ones can still be banked with the time remaining.
    declare -a RESULTS=() OWED=() FAILED=()
    for i in "${!UNIT_TAG[@]}"; do
        # Read into a temp so the exit status is the function's, not the assignment's.
        unit_out=$(run_unit "$MERGED" "${UNIT_TAG[$i]}" "${UNIT_TASKS[$i]}" \
                            "${UNIT_MINUTES[$i]}" ${UNIT_EXTRA[$i]})
        case $? in
            0) RESULTS+=("$unit_out") ;;
            2) OWED+=("${UNIT_TAG[$i]}") ;;
            *) FAILED+=("${UNIT_TAG[$i]}") ;;
        esac
    done

    if (( ${#OWED[@]} > 0 )); then
        # Deliberately no step file and no cleanup: the units already finished are
        # banked in $PARTIAL_DIR/step-$STEP and the merged model is left in place,
        # so the next job resumes at the units still owed instead of redoing the
        # merge and everything before it.
        echo "[drain] out of wall clock part-way through step $STEP -- still owed: ${OWED[*]}"
        echo "        finished units are banked, the merge is kept, and the dispatcher will resubmit"
        break
    fi

    if (( ${#RESULTS[@]} > 0 )); then
        if (( ${#FAILED[@]} > 0 )); then
            echo "[step $STEP] WARNING: ${#FAILED[@]} unit(s) FAILED (${FAILED[*]}) -- recording a" >&2
            echo "            step file without them; those benchmarks will be absent from the curve" >&2
        fi
        # Which units were reused from another profile rather than generated here.
        # Read before the markers are deleted, and passed on so the step file says
        # so: a carried-over benchmark is a repeated measurement, not a new one,
        # and the only honest thing to do with a repeated measurement is record it.
        CARRIED=$(python "$SCRIPT_DIR/bench_eval.py" --carried-from "$PARTIAL_DIR/step-$STEP")
        python "$SCRIPT_DIR/bench_eval.py" --collect --run-dir "$RUN_DIR" --step "$STEP" \
            --sample-n "$SAMPLE_N_JSON" --carried "$CARRIED" --results "${RESULTS[@]}"
        # Per-item rows, stored beside the step file. Failure here is reported and
        # not fatal: the aggregate is the result that took the GPU hours, and the
        # harvest can be redone at any time from the same files with --retro.
        if $HARVEST; then
            python "$SCRIPT_DIR/bench_samples.py" --harvest --run-dir "$RUN_DIR" \
                --step "$STEP" --sample-n "$SAMPLE_N_JSON" --results "${RESULTS[@]}" \
                || echo "[step $STEP] per-item harvest failed -- rerun bench_samples.py --retro" >&2
        fi
        done_count=$((done_count + 1))
        # The checkpoint is measured, so everything that existed to serve it goes:
        # the banked units have been folded into step-$STEP.json, and the merged
        # model has nothing left to be evaluated against.
        rm -rf "$PARTIAL_DIR/step-$STEP"
    else
        echo "[step $STEP] every unit failed -- not recording a result" >&2
    fi

    if $MERGED_IS_TEMPORARY; then
        rm -rf "$MERGED" "$MERGE_DONE"
    fi
done

echo ""
echo "Evaluated $done_count checkpoint(s); results in $PROFILE_DIR"
