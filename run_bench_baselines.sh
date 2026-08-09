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
        -h|--help)      sed -n '2,31p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

# ---------- how far the lines should reach ----------
SPAN_DIR="$REPO/checkpoint/$SPAN_RUN/bench_eval"
[[ -d "$SPAN_DIR" ]] || { echo "error: no bench_eval under $SPAN_DIR" >&2; exit 2; }
SPAN=$(ls "$SPAN_DIR"/step-*.json 2>/dev/null | sed 's|.*/step-||; s|\.json$||' | sort -n | tail -1)
[[ -n "$SPAN" ]] || { echo "error: $SPAN_RUN has no scored checkpoints to span" >&2; exit 2; }

mkdir -p "$ROOT/_models"

echo "=========================================================================="
echo "Baselines: ${#BASELINES[@]}"
echo "Span:      bench/step 0..$SPAN   (from $SPAN_RUN)"
echo "GPUs:      $NUM_GPUS   (serial, one baseline at a time)"
echo "Results:   $ROOT/<label>/bench_eval/step-0.json"
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
    RESULT="$SHADOW/bench_eval/step-0.json"

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
                --num-gpus "$NUM_GPUS" --min-minutes 0

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
            echo "  would publish as WandB run baseline/$LABEL, bench/step 0..$SPAN"
        elif [[ ! -f "$RESULT" ]]; then
            echo "  nothing to publish: no $RESULT" >&2
            FAILED+=("$LABEL (no result to publish)")
            continue
        else
            source "$CONDA_SH"; set +u; conda activate lmms_eval; set -u
            if ! python "$SCRIPT_DIR/bench_eval.py" --publish-baseline \
                    --run-dir "$SHADOW" --name "$LABEL" --span "$SPAN" $OVERWRITE; then
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
