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
# Below this much wall-clock left, start nothing new: being killed halfway wastes
# the whole checkpoint. Left empty here and filled in after parsing, because the
# figure depends on how many GPUs this job got -- see below.
MIN_MINUTES=""
MAX_CHECKPOINTS=0   # 0 = drain everything that is pending
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-dir)         RUN_DIR="$2";         shift 2 ;;
        --num-gpus)        NUM_GPUS="$2";        shift 2 ;;
        --every)           EVERY="$2";           shift 2 ;;
        --sample-n)        SAMPLE_N="$2";        shift 2 ;;
        --min-minutes)     MIN_MINUTES="$2";     shift 2 ;;
        --max-checkpoints) MAX_CHECKPOINTS="$2"; shift 2 ;;
        --dry-run)         DRY_RUN=true;         shift ;;
        -h|--help)         sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

# A checkpoint costs a fixed ~10 min to merge (a copy, not GPU work, so it does not
# shrink with the allocation) plus ~40 min of generation on 4 GPUs, which does. At
# --num-gpus 4 that reproduces the 50 min this guard used to hardcode; at the
# default 1 it asks for ~170, which still fits in the dispatcher's 4h job with room
# to spare. Getting this wrong in the optimistic direction is the expensive
# failure: the job starts a checkpoint it cannot finish, dies at the wall clock
# having written no step file, and the dispatcher submits another to do the same.
MIN_MINUTES=${MIN_MINUTES:-$(( 10 + 160 / NUM_GPUS ))}

[[ -n "$RUN_DIR" ]] || { echo "error: --run-dir is required" >&2; exit 2; }
[[ -d "$RUN_DIR" ]] || { echo "error: no such run dir: $RUN_DIR" >&2; exit 2; }
RUN_DIR=$(cd "$RUN_DIR" && pwd)

BENCH_DIR="$RUN_DIR/bench_eval"
TASK_DIR="$BENCH_DIR/tasks"
MERGE_ROOT="${BENCH_MERGE_ROOT:-$REPO/checkpoint/_bench_eval}"
mkdir -p "$BENCH_DIR" "$MERGE_ROOT"

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
minutes_left() {
    local left
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
run_suite() {
    local model="$1" tasks="$2" tag="$3"; shift 3
    local -a common=(--model "$model" --tasks "$tasks" --max-new-tokens 4096
                     --num-gpus "$NUM_GPUS" --direct --r1-mode --tag "r1_$tag" "$@")
    local out_dir
    out_dir=$(bash "$VLM_REASONING/scripts/slurm/launch_lmms_eval_job.sh" \
        "${common[@]}" --print-output-dir) || return 1

    bash "$VLM_REASONING/scripts/slurm/launch_lmms_eval_job.sh" \
        "${common[@]}" --include_path "$TASK_DIR" >&2 || return 1

    ls -t "$out_dir"/*/*_results.json 2>/dev/null | head -1
}

# ---------- drain ----------
RUN_NAME=$(basename "$RUN_DIR")
done_count=0
echo "=========================================================================="
echo "Run dir:    $RUN_DIR"
echo "Pending:    $(pending_steps | tr '\n' ' ')"
echo "GPUs:       $NUM_GPUS   sample: $SAMPLE_N/benchmark   cadence: every $EVERY steps   need ${MIN_MINUTES}min/checkpoint"
echo "Natural:    $NATURAL_TASKS + mmerealworld_mini (at 3.2M pixels, as on test)"
echo "Non-nat:    $NONNATURAL_TASKS"
echo "=========================================================================="

if $DRY_RUN; then
    echo "[dry-run] would evaluate, in order: $(pending_steps | tr '\n' ' ')"
    echo "[dry-run] wall clock left: $(minutes_left) min (need $MIN_MINUTES per checkpoint)"
    echo "[dry-run] merged models would go to $MERGE_ROOT and be deleted after each step"
    exit 0
fi

while true; do
    STEP=$(pending_steps | head -1)
    if [[ -z "$STEP" ]]; then
        echo "[drain] nothing pending -- exiting so the GPUs go back to the pool"
        break
    fi
    LEFT=$(minutes_left)
    if (( LEFT < MIN_MINUTES )); then
        echo "[drain] only ${LEFT}min of wall clock left (need $MIN_MINUTES) -- exiting; the dispatcher will resubmit"
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
        MERGED="$MERGE_ROOT/${RUN_NAME}_cp${STEP}_merged"
        MERGED_IS_TEMPORARY=true
        echo "[step $STEP] merging $CKPT -> $MERGED   (${LEFT}min left)"
        echo "--------------------------------------------------------------------------"
        rm -rf "$MERGED"
        if ! bash "$REPO/merge_lora_grpo_qwen3.sh" "$CKPT" "$MERGED"; then
            echo "[step $STEP] merge FAILED -- skipping this checkpoint" >&2
            rm -rf "$MERGED"
            # Leave no step file: a later job retries it rather than recording a gap
            # as if it had been measured.
            break
        fi
    fi
    echo "--------------------------------------------------------------------------"

    set +u; conda activate lmms_eval; set -u
    NAT_RESULTS=$(run_suite "$MERGED" "$NATURAL_TASKS" natural)
    MMERW_RESULTS=$(run_suite "$MERGED" "mmerealworld_mini" mmerw --max-pixels 3211264)
    NONNAT_RESULTS=$(run_suite "$MERGED" "$NONNATURAL_TASKS" nonnatural)

    if [[ -n "$NAT_RESULTS$MMERW_RESULTS$NONNAT_RESULTS" ]]; then
        python "$SCRIPT_DIR/bench_eval.py" --collect --run-dir "$RUN_DIR" --step "$STEP" \
            --results $NAT_RESULTS $MMERW_RESULTS $NONNAT_RESULTS
        done_count=$((done_count + 1))
    else
        echo "[step $STEP] every suite failed -- not recording a result" >&2
    fi

    $MERGED_IS_TEMPORARY && rm -rf "$MERGED"
done

echo ""
echo "Evaluated $done_count checkpoint(s); results in $BENCH_DIR"
