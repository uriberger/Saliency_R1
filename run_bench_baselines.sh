#!/bin/bash
# Score a list of baseline models on the mini test suites and publish each one to
# WandB as a horizontal reference line across the bench panels.
#
#   bash run_bench_baselines.sh [--dry-run] [--force] [--publish-only] [--no-publish]
#
# Edit BASELINES and SPAN_RUN below. A baseline is a standalone model, not a
# checkpoint: one number, not a curve. So it is not a point on the training run's
# bench/step axis -- it is a flat line at its score, labelled with the baseline's
# name, that the run's curve is read against.
#
# Scoring reuses run_bench_eval.sh unchanged, through the same shadow-directory
# trick as run_bench_eval_steps.sh. That script treats bench_eval/base_model.txt as
# a pending "step 0" that is already a standalone model and needs no LoRA merge --
# exactly what a baseline is. A shadow directory holding nothing but that file
# therefore leaves it one thing to do, and the baseline is scored by precisely the
# recipe that scores every checkpoint: same suites, same sample of each benchmark,
# same --r1-mode generation settings. A baseline scored any other way would not be
# comparable to the curve it is drawn under.
#
# Publishing is bench_eval.py --publish-baseline: one WandB run per baseline,
# logging the same bench/* keys the training run logs, at bench/step 0 and
# bench/step SPAN, so the panels draw it as a line spanning the run. See that
# function for why it is a separate run rather than extra keys on the training run.
#
# In the WandB report, add these runs to the panel's run set once (they are grouped
# under job_type = bench_baseline) and every bench panel gains the lines.
#
# Note this takes all $NUM_GPUS GPUs for the whole list -- do not start it next to
# a training job on the same node. Roughly two hours per baseline on 8 GPUs.
set -uo pipefail

# label|model -- the label is the legend entry, the model is a directory or a HF
# repo id. Labels must be unique: the WandB run id is derived from them.
BASELINES=(
    "sft-coldstart|checkpoint/coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged"
    "saliency-r1|checkpoint/grpo-coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged-saliency-r1-qwen3_merged"
    "grpo-no-saliency|checkpoint/grpo-qwen3-vl-8b-instruct-no-sal_merged"
    "overlap-8k|checkpoint/grpo-coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged-overlap__wov0.4_2head_trmean_merged"
    "qwen3-vl-8b-instruct|Qwen/Qwen3-VL-8B-Instruct"
)

# The run whose bench panels these lines are drawn on. Only its x-range is read:
# the lines are drawn from bench/step 0 to its last scored checkpoint, so they span
# the curve without extending the axis past it.
SPAN_RUN=grpo-coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged-overlap__wov0.033_2head_trmean_50k_set_a_mean_in_v2_k_proj_beta_004

NUM_GPUS=8

# 300 documents per natural benchmark, 100 per non-natural one -- the same profile
# every other scoring path now uses. A baseline scored at a different size from the
# curve it is drawn under is not a reference line, it is a second measurement of a
# different thing, so this has to match.
NATURAL_N=${NATURAL_N:-300}
NONNATURAL_N=${NONNATURAL_N:-100}
declare -a TASK_N=()
BANK=${BANK:-auto}
# Restrict to some of BASELINES, by label. Repeatable.
declare -a ONLY=()

SCRIPT_DIR=$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")" && pwd)
REPO=${REPO:-/home/uberger/scratch/research/saliency_r1}
CONDA_SH=/home/uberger/scratch/miniconda3/etc/profile.d/conda.sh
ROOT="$REPO/outputs/bench_baselines"

DRY_RUN=false
FORCE=false
PUBLISH=true
SCORE=true
OVERWRITE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)      DRY_RUN=true;  shift ;;
        # Re-score a baseline that already has a result. The step-0 file is what
        # marks one done, so without this a rerun is a cheap no-op rather than a
        # repeat of hours of GPU time.
        --force)        FORCE=true;    shift ;;
        # Publish scores that already exist -- e.g. from a login node, if the GPU
        # node they were scored on could not reach WandB.
        --publish-only) SCORE=false;   shift ;;
        --no-publish)   PUBLISH=false; shift ;;
        # Replace an already-published baseline run of the same name, rather than
        # refusing. Deletes the old run; its URL stops resolving.
        --overwrite)    OVERWRITE="--overwrite"; shift ;;
        --num-gpus)     NUM_GPUS="$2"; shift 2 ;;
        --natural-n)    NATURAL_N="$2";    shift 2 ;;
        --nonnatural-n) NONNATURAL_N="$2"; shift 2 ;;
        --task-n)       TASK_N+=("$2");    shift 2 ;;
        --bank)         BANK="$2";         shift 2 ;;
        # Score/publish only these labels, rather than all of BASELINES.
        --only)         ONLY+=("$2");      shift 2 ;;
        -h|--help)      sed -n '2,31p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

# ---------- which sample profile ----------
# Resolved by the same code the eval job uses, so this script and the job cannot
# disagree about where a result belongs or which WandB keys it becomes.
PY_BIN=$(command -v python || command -v python3) || \
    { echo "error: no python on PATH" >&2; exit 2; }
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
print(profile_name(sizes), os.path.relpath(profile_dir("x", sizes), "x"))
PY
) || { echo "error: could not resolve the sample profile" >&2; exit 2; }

profile_dir_of() {
    [[ "$PROFILE_SUBDIR" == "." ]] && { echo "$1/bench_eval"; return; }
    echo "$1/bench_eval/$PROFILE_SUBDIR"
}

# ---------- how far the lines should reach ----------
# Read from the span run's curve AT THIS PROFILE where it has one, since that is
# the axis these lines will be drawn against. A profile with no curve yet falls
# back to the 100-document one: the span only sets the x-extent of a horizontal
# line, so borrowing it is cosmetic, not a mixing of measurements.
SPAN_DIR=$(profile_dir_of "$REPO/checkpoint/$SPAN_RUN")
SPAN=$(ls "$SPAN_DIR"/step-*.json 2>/dev/null | sed 's|.*/step-||; s|\.json$||' | sort -n | tail -1)
if [[ -z "$SPAN" ]]; then
    SPAN_DIR="$REPO/checkpoint/$SPAN_RUN/bench_eval"
    SPAN=$(ls "$SPAN_DIR"/step-*.json 2>/dev/null | sed 's|.*/step-||; s|\.json$||' | sort -n | tail -1)
    [[ -n "$SPAN" ]] && echo "note: $SPAN_RUN has no $PROFILE curve yet; spanning to its 100-document one (x-extent only)" >&2
fi
[[ -n "$SPAN" ]] || { echo "error: $SPAN_RUN has no scored checkpoints to span" >&2; exit 2; }

mkdir -p "$ROOT/_models"

echo "=========================================================================="
echo "Baselines: $( (( ${#ONLY[@]} > 0 )) && echo "${#ONLY[@]} of ${#BASELINES[@]}   (${ONLY[*]})" || echo "${#BASELINES[@]}")"
echo "Sample:    $PROFILE   (natural=$NATURAL_N, non-natural=$NONNATURAL_N per benchmark)"
echo "Span:      0..$SPAN   (from $SPAN_RUN)"
echo "GPUs:      $NUM_GPUS   (serial, one baseline at a time)"
echo "Results:   $(profile_dir_of "$ROOT/<label>")/step-0.json"
$SCORE   || echo "Mode:      --publish-only, nothing will be evaluated"
$PUBLISH || echo "Mode:      --no-publish, nothing will be sent to WandB"
$DRY_RUN && echo "Mode:      --dry-run"
echo "=========================================================================="

# Reported at the end rather than as they happen: a missing model scrolls off the
# top of an hours-long log, and a list that says it scored five baselines when it
# scored three is the failure that matters here.
declare -a DONE=() SKIPPED=() FAILED=()

for ENTRY in "${BASELINES[@]}"; do
    LABEL=${ENTRY%%|*}
    MODEL=${ENTRY#*|}
    # --only, when given, is the whole list. Matched exactly rather than as a
    # substring: the labels are short and a loose match on "saliency-r1" would
    # also take "grpo-no-saliency".
    if (( ${#ONLY[@]} > 0 )); then
        keep=false
        for want in "${ONLY[@]}"; do [[ "$want" == "$LABEL" ]] && keep=true; done
        $keep || continue
    fi
    echo ""
    echo "--------------------------------------------------------------------------"
    echo "Baseline: $LABEL"

    # A relative path is relative to the repo, not to wherever this was invoked.
    # Tested by existence, not by shape, so that a HF repo id -- which is also a
    # relative-looking "org/name" -- is left alone rather than turned into a path.
    if [[ "$MODEL" != /* && -d "$REPO/$MODEL" ]]; then
        MODEL="$REPO/$MODEL"
    fi

    # Not a directory: treat it as a HF repo id and resolve it in the cache. The
    # snapshot is symlinked under a name of its own because lmms-eval derives its
    # results directory from the model path's basename, and a bare snapshot path
    # would file the results under a commit hash.
    if [[ ! -d "$MODEL" ]]; then
        if $DRY_RUN; then
            echo "  would resolve HF repo id $MODEL from the cache"
        else
            source "$CONDA_SH"; set +u; conda activate lmms_eval; set -u
            SNAPSHOT=$(python -c "
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1]))" "$MODEL" 2>/dev/null)
            if [[ -z "$SNAPSHOT" || ! -d "$SNAPSHOT" ]]; then
                echo "  cannot resolve $MODEL as a directory or a HF repo id -- skipping" >&2
                SKIPPED+=("$LABEL (no model)")
                continue
            fi
            ln -sfn "$SNAPSHOT" "$ROOT/_models/${MODEL##*/}"
            MODEL="$ROOT/_models/${MODEL##*/}"
        fi
    fi
    echo "  model:  $MODEL"

    SHADOW="$ROOT/$LABEL"
    RESULT="$(profile_dir_of "$SHADOW")/step-0.json"

    if $SCORE; then
        if [[ -f "$RESULT" ]] && ! $FORCE; then
            echo "  already scored ($RESULT) -- skipping the eval; pass --force to redo it"
        elif $DRY_RUN; then
            echo "  would evaluate on $NUM_GPUS GPUs -> $RESULT"
        else
            mkdir -p "$SHADOW/bench_eval"
            echo "$MODEL" > "$SHADOW/bench_eval/base_model.txt"
            # --force means re-score, and run_bench_eval.sh decides what is pending
            # from the presence of this file. Removing it is how it is asked again.
            $FORCE && rm -f "$RESULT"

            bash "$SCRIPT_DIR/run_bench_eval.sh" --run-dir "$SHADOW" \
                --num-gpus "$NUM_GPUS" --min-minutes 0 --bank "$BANK" "${SIZE_ARGS[@]}"

            # run_bench_eval.sh reports a suite that failed and moves on, so its exit
            # status does not say whether this model was scored. The step file does.
            if [[ ! -f "$RESULT" ]]; then
                echo "  $LABEL produced no result -- see the output above" >&2
                FAILED+=("$LABEL (eval)")
                continue
            fi
            echo "  scored -> $RESULT"
        fi
    fi

    if $PUBLISH; then
        if $DRY_RUN; then
            echo "  would publish as WandB run baseline/$LABEL@$PROFILE, keys bench_$PROFILE/*, step 0..$SPAN"
        elif [[ ! -f "$RESULT" ]]; then
            echo "  nothing to publish: no $RESULT" >&2
            FAILED+=("$LABEL (no result to publish)")
            continue
        else
            source "$CONDA_SH"; set +u; conda activate lmms_eval; set -u
            if ! python "$SCRIPT_DIR/bench_eval.py" --publish-baseline \
                    --run-dir "$SHADOW" --name "$LABEL" --span "$SPAN" \
                    --sample-n "$SAMPLE_N_JSON" $OVERWRITE; then
                FAILED+=("$LABEL (publish)")
                continue
            fi
        fi
    fi
    DONE+=("$LABEL")
done

echo ""
echo "=========================================================================="
echo "Done:    ${#DONE[@]}   ${DONE[*]:-}"
echo "Skipped: ${#SKIPPED[@]}   ${SKIPPED[*]:-}"
echo "Failed:  ${#FAILED[@]}   ${FAILED[*]:-}"
echo "=========================================================================="
(( ${#FAILED[@]} == 0 ))
