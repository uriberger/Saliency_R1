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
#   2. run the natural suite, then MME-RealWorld at its test-time resolution,
#      then the non-natural suite
#   3. reduce the results to bench_eval/step-<N>.json, which the trainer picks up
#      and logs to WandB
#   4. delete the merged model -- 16 GB each, and /lustre is not roomy
#
# Progress is banked per suite, not per checkpoint. A job that runs out of wall
# clock after two of the three suites records what those two produced under
# bench_eval/partial/step-<N>/ and leaves the merged model in place; the next job
# reuses both and picks up at the third. That is what makes a one-hour allocation
# useful: the unit of work it has to finish is a suite (~15-25 min on 1 GPU), not
# a whole checkpoint (50-86 min) plus the merge.
#
# What is NOT banked is a partial step-<N>.json. It is written only once all three
# suites are in hand, because a half-measured checkpoint on the benchmark curve is
# indistinguishable from a real one. The merged model is deleted at that same
# moment -- it exists only to serve the suites still owed for that checkpoint.
#
# Everything runs through vlm_reasoning's launch_lmms_eval_job.sh --direct, so a
# mini benchmark is evaluated with exactly the recipe the full test suite uses
# (--r1-mode: the Saliency-R1 system prompt, repetition_penalty 1.05,
# max_new_tokens 4096).
#
# Usage:
#   bash run_bench_eval.sh --run-dir CKPT_DIR [--num-gpus 1] [--every 100]
#
# Environment:
#   OPENAI_API_KEY / NVIDIA_API_KEY   needed by mathvista's llm_as_judge metric
#   HF_TOKEN
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")" && pwd)
REPO=${REPO:-/home/uberger/scratch/research/saliency_r1}
VLM_REASONING=${VLM_REASONING:-/home/uberger/scratch/research/vlm_reasoning}
LMMS_EVAL_DIR=${LMMS_EVAL_DIR:-/home/uberger/scratch/research/lmms-eval}
CONDA_SH=/home/uberger/scratch/miniconda3/etc/profile.d/conda.sh

RUN_DIR=""
NUM_GPUS=1
EVERY=100
SAMPLE_N=100
# Below this much wall-clock left, start no further suite. Left empty here and
# filled in after parsing, because the figure depends on how many GPUs this job
# got -- see below. 0 disables the guard entirely.
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
        --min-minutes)     MIN_MINUTES="$2";     shift 2 ;;
        --job-minutes)     JOB_MINUTES="$2";     shift 2 ;;
        --max-checkpoints) MAX_CHECKPOINTS="$2"; shift 2 ;;
        --dry-run)         DRY_RUN=true;         shift ;;
        -h|--help)         sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

# The guard sizes ONE SUITE, because a suite is what gets banked. Measured on 1 GPU
# (gaps between consecutive step-*.json in a draining job): 50-86 min per checkpoint,
# of which ~10 is the merge, leaving ~15-25 for each of the three suites. The
# generation part shrinks with the allocation; loading the model into each worker
# does not, hence the fixed term.
#
# Getting this wrong in the optimistic direction is still the expensive failure --
# a suite started and killed at the wall clock banks nothing and is redone from
# scratch -- but it now costs one suite rather than a whole checkpoint.
SUITE_MINUTES=${MIN_MINUTES:-$(( 5 + 25 / NUM_GPUS ))}
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
# Where a finished suite is recorded so a later job can skip it. One directory per
# step, deleted the moment that step's step-<N>.json is written.
PARTIAL_DIR="$BENCH_DIR/partial"
mkdir -p "$BENCH_DIR" "$MERGE_ROOT" "$PARTIAL_DIR"

# ---------- banking a finished suite ----------
# A suite is banked by recording WHERE lmms-eval put its results, not by copying
# them. The sample size is recorded alongside because --sample-n changes what the
# mini task contains but does not change lmms-eval's output directory: without
# this, results produced at n=100 would be silently reused for a job asking for
# n=500 and the curve would mix two different benchmarks.
bank_suite() {
    local marker="$1" results="$2"
    mkdir -p "$(dirname "$marker")"
    printf '{"sample_n": %s, "results": "%s"}\n' "$SAMPLE_N" "$results" > "$marker"
}

# Echo the banked results path if it is still usable, else fail. "Usable" has to
# be checked, not assumed: the referenced file may have been cleaned up, and a
# results.json from a run killed mid-write parses as truncated garbage.
banked_suite() {
    local marker="$1"
    [[ -f "$marker" ]] || return 1
    python - "$marker" "$SAMPLE_N" <<'PY' || return 1
import json, os, sys
try:
    marker = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(1)
if str(marker.get("sample_n")) != sys.argv[2]:
    sys.exit(1)
path = marker.get("results", "")
if not os.path.isfile(path):
    sys.exit(1)
try:
    if not json.load(open(path)).get("results"):
        sys.exit(1)
except Exception:
    sys.exit(1)
print(path)
PY
}

# ---------- mini task configs ----------
# Regenerated every job rather than committed: they carry the absolute path of the
# lmms-eval clone, and a stale path would silently evaluate the wrong task file.
source "$CONDA_SH"
set +u; conda activate lmms_eval; set -u
python "$SCRIPT_DIR/eval_mini/make_mini_tasks.py" \
    --out-dir "$TASK_DIR" --lmms-eval-dir "$LMMS_EVAL_DIR" --n "$SAMPLE_N" || exit 1

NATURAL_TASKS=$(python "$SCRIPT_DIR/eval_mini/make_mini_tasks.py" --print-tasks natural)
NONNATURAL_TASKS=$(python "$SCRIPT_DIR/eval_mini/make_mini_tasks.py" --print-tasks nonnatural)
# MME-RealWorld's images are ~36MP and the wrapper default downsamples them ~22x,
# which makes the model over-abstain. The test suite runs it at 3.2M pixels; a
# combined invocation cannot, so it gets its own.
NATURAL_TASKS=${NATURAL_TASKS//mmerealworld_mini,/}
NATURAL_TASKS=${NATURAL_TASKS//,mmerealworld_mini/}

# ---------- what still needs evaluating ----------
# Step 0 is the model the run started from, recorded by the launcher. It is already
# a full model, so it is scored directly -- no adapter to merge, and nothing to
# delete afterwards.
BASE_MODEL_FILE="$BENCH_DIR/base_model.txt"

pending_steps() {
    local d step
    if [[ -f "$BASE_MODEL_FILE" && ! -f "$BENCH_DIR/step-0.json" ]]; then
        echo 0
    fi
    for d in "$RUN_DIR"/checkpoint-*; do
        [[ -d "$d" ]] || continue
        step=${d##*checkpoint-}
        [[ "$step" =~ ^[0-9]+$ ]] || continue
        (( step % EVERY == 0 )) || continue
        [[ -f "$BENCH_DIR/step-$step.json" ]] && continue
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
# stopping at a suite boundary.
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

# ---------- evaluate one suite ----------
# Echoes the results.json lmms-eval produced, or nothing if the run failed. Each
# suite gets its own --tag so its results land in their own directory: attributing
# a suite by "the newest file in a shared directory" would quietly mislabel
# everything the first time a suite failed and left the previous one newest.
# Exit status is three-valued, because "no results" and "no time" have to be told
# apart by the caller: 0 results echoed, 1 the suite failed, 2 out of wall clock
# (nothing attempted, nothing lost).
run_suite() {
    local model="$1" tasks="$2" tag="$3"; shift 3
    local marker="$PARTIAL_DIR/step-$STEP/$tag.json"
    local banked
    if banked=$(banked_suite "$marker"); then
        echo "[step $STEP] $tag: reusing the suite banked by an earlier job" >&2
        printf '%s\n' "$banked"
        return 0
    fi

    local left; left=$(minutes_left)
    if (( SUITE_MINUTES > 0 && left < SUITE_MINUTES )); then
        echo "[step $STEP] $tag: ${left}min of wall clock left, a suite needs $SUITE_MINUTES -- stopping here" >&2
        return 2
    fi

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
    bank_suite "$marker" "$results"
    printf '%s\n' "$results"
}

# ---------- drain ----------
RUN_NAME=$(basename "$RUN_DIR")
done_count=0
echo "=========================================================================="
echo "Run dir:    $RUN_DIR"
echo "Pending:    $(pending_steps | tr '\n' ' ')"
echo "GPUs:       $NUM_GPUS   sample: $SAMPLE_N/benchmark   cadence: every $EVERY steps   need ${SUITE_MINUTES}min/suite"
echo "Natural:    $NATURAL_TASKS + mmerealworld_mini (at 3.2M pixels, as on test)"
echo "Non-nat:    $NONNATURAL_TASKS"
echo "Banked:     $PARTIAL_DIR"
echo "=========================================================================="

if $DRY_RUN; then
    echo "[dry-run] would evaluate, in order: $(pending_steps | tr '\n' ' ')"
    echo "[dry-run] wall clock left: $(minutes_left) min (need $SUITE_MINUTES per suite, +$MERGE_MINUTES to merge)"
    echo "[dry-run] merged models go to $MERGE_ROOT, kept across jobs while a step is"
    echo "[dry-run] unfinished and deleted once its step-<N>.json is written"
    for s in $(pending_steps); do
        banked=$(ls "$PARTIAL_DIR/step-$s" 2>/dev/null | sed 's/\.json$//' | tr '\n' ' ')
        echo "[dry-run] step $s: suites already banked: ${banked:-none}"
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
    # What entering this checkpoint costs before anything can be banked: one suite,
    # plus the merge if no earlier job already paid for it.
    MERGED="$MERGE_ROOT/${RUN_NAME}_cp${STEP}_merged"
    MERGE_DONE="$MERGED.complete"
    NEED=$SUITE_MINUTES
    if (( STEP != 0 )) && [[ ! -f "$MERGE_DONE" ]]; then
        NEED=$(( NEED + MERGE_MINUTES ))
    fi
    if (( SUITE_MINUTES > 0 && LEFT < NEED )); then
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
    # Each suite is attempted only if the clock allows, and banked the moment it
    # succeeds. Status 2 means the clock ran out: stop the whole job rather than
    # move on, since the remaining suites have no more time than this one did.
    NAT_RESULTS=""; MMERW_RESULTS=""; NONNAT_RESULTS=""; OUT_OF_TIME=false
    NAT_RESULTS=$(run_suite "$MERGED" "$NATURAL_TASKS" natural)
    (( $? == 2 )) && OUT_OF_TIME=true
    if ! $OUT_OF_TIME; then
        MMERW_RESULTS=$(run_suite "$MERGED" "mmerealworld_mini" mmerw --max-pixels 3211264)
        (( $? == 2 )) && OUT_OF_TIME=true
    fi
    if ! $OUT_OF_TIME; then
        NONNAT_RESULTS=$(run_suite "$MERGED" "$NONNATURAL_TASKS" nonnatural)
        (( $? == 2 )) && OUT_OF_TIME=true
    fi

    if $OUT_OF_TIME; then
        # Deliberately no step file and no cleanup: the suites already finished are
        # banked in $PARTIAL_DIR/step-$STEP and the merged model is left in place,
        # so the next job resumes at the first suite still owed instead of redoing
        # the merge and everything before it.
        echo "[drain] out of wall clock part-way through step $STEP -- finished suites are banked,"
        echo "        the merge is kept, and the dispatcher will resubmit to finish it"
        break
    fi

    if [[ -n "$NAT_RESULTS$MMERW_RESULTS$NONNAT_RESULTS" ]]; then
        python "$SCRIPT_DIR/bench_eval.py" --collect --run-dir "$RUN_DIR" --step "$STEP" \
            --results $NAT_RESULTS $MMERW_RESULTS $NONNAT_RESULTS
        done_count=$((done_count + 1))
        # The checkpoint is measured, so everything that existed to serve it goes:
        # the banked suites have been folded into step-$STEP.json, and the merged
        # model has nothing left to be evaluated against.
        rm -rf "$PARTIAL_DIR/step-$STEP"
    else
        echo "[step $STEP] every suite failed -- not recording a result" >&2
    fi

    if $MERGED_IS_TEMPORARY; then
        rm -rf "$MERGED" "$MERGE_DONE"
    fi
done

echo ""
echo "Evaluated $done_count checkpoint(s); results in $BENCH_DIR"
