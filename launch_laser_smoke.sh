#!/usr/bin/env bash
# Submit the LASER-on-Qwen3-VL SMOKE run: does the attention capture fire under FSDP?
#
#   bash launch_laser_smoke.sh --name laser-smoke
#   bash launch_laser_smoke.sh --name laser-smoke --dry-run
#
# This is a MECHANICAL CHECK, not a training run. Everything upstream of it is already
# verified without a GPU -- the capture reproduces transformers' own attention at
# max |delta| 0.00e+00 on both transformers 4.57.6 and 5.13 (test_laser_capture_cpu.py),
# and the 45,933-row corpus loads through verl's own RLHFDataset with correct Qwen3-VL
# MRoPE. The one thing CPU cannot answer is whether `AttentionSliceCapturer` still finds
# its modules and fires once verl has FSDP-wrapped the actor. That is what this buys.
#
# WHY 4 GPUs AND NOT 8. Their config sets param_offload=False and optimizer_offload=False,
# so ~105-140 GB of params + grads + Adam state stays resident and FSDP-sharded, while
# gpu_memory_utilization=0.4 hands vLLM ~32 GB of each 80 GB card. 4 ranks puts training
# at 26-35 GB/GPU, which fits; 2 ranks would need 53-70 GB and OOM. 8 would halve the step
# work, but the step here is 128 sequences total and ~20 of the ~40 minutes is fixed
# startup that 8 ranks make WORSE (eight readers pulling the same 17 GB checkpoint, four
# vLLM engines to initialise). 4 is also the shape the paper ran, so a failure here is
# about our port rather than about a rank count nobody has tried. Use 8 for the real 45K
# run, where generation dominates and actually scales.
#
# THE FOUR OVERRIDES THAT MAKE IT A SMOKE RUN
#
#   TRAIN_BATCH_SIZE=16     their default is 512, and the smoke set has 61 usable rows
#   PPO_MINI_BATCH_SIZE=16  after verl's overlong filter. 512 cannot form one step.
#   ATTENTION_START_STEP=1  their default is 20, so a BROKEN CAPTURE WOULD LOOK LIKE A
#                           HEALTHY RUN for twenty steps and then fail. This is the single
#                           most important line in this file.
#   SAVE_FREQ/TEST_FREQ=-1  no checkpointing, no validation: pure cost for a mechanical
#                           check.
#
# MAX_RESPONSE_LENGTH is deliberately left at their 2048. Shrinking it would cut the
# window count, and the windowed reward is the mechanism under test.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

NAME="laser-smoke"
GPUS=4
DURATION=2
STEPS=3
FORK="$REPO/laser_fork"
DATA="$REPO/cold_data/laser/verl"
MODEL="Qwen/Qwen3-VL-8B-Instruct"
OUT_DIR="$REPO/outputs/laser/smoke"
PARTITION_OVERRIDE=""
DRY_RUN=0
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)   DRY_RUN=1;               shift   ;;
        --name)      NAME="$2";               shift 2 ;;
        --gpus)      GPUS="$2";               shift 2 ;;
        --steps)     STEPS="$2";              shift 2 ;;
        --model)     MODEL="$2";              shift 2 ;;
        --data)      DATA="$2";               shift 2 ;;
        --out-dir)   OUT_DIR="$2";            shift 2 ;;
        --duration)  DURATION="$2";           shift 2 ;;
        --partition) PARTITION_OVERRIDE="$2"; shift 2 ;;
        --)          shift; EXTRA+=("$@"); break ;;
        *)           EXTRA+=("$1");           shift ;;
    esac
done

# Resolve through the worktree symlinks to the central tree. A job submitted from a
# worktree that is then merged and deleted would otherwise reference paths that no longer
# exist by the time it reaches the front of the queue -- and it would fail at model load,
# an hour later, looking like an environment problem.
# `|| true` is load-bearing: readlink -f exits non-zero on a path that does not exist,
# and under `set -e` that kills the script before any of the checks below can say WHY.
# It cost a silent no-op submission to find out.
resolve() { readlink -f "$1" 2>/dev/null || true; }
FORK=$(resolve "$FORK"); [ -n "$FORK" ] || FORK="$REPO/laser_fork"
DATA=$(resolve "$DATA"); [ -n "$DATA" ] || DATA="$REPO/cold_data/laser/verl"
OUT_DIR=$(resolve "$OUT_DIR"); [ -n "$OUT_DIR" ] || OUT_DIR="$REPO/outputs/laser/smoke"

TRAIN_FILE="$DATA/train_smoke64.parquet"
VAL_FILE="$DATA/val_smoke64.parquet"
for f in "$TRAIN_FILE" "$VAL_FILE"; do
    [ -f "$f" ] || { echo "MISSING: $f -- run build_laser_data.py --smoke 64 first" >&2; exit 1; }
done
[ -d "$FORK" ] || { echo "MISSING fork: $FORK" >&2; exit 1; }
grep -q "Qwen3VLTextAttention" "$FORK/verl/workers/actor/attention_capture.py" || {
    echo "The fork's attention_capture.py is not the Qwen3-VL one." >&2
    echo "  bash patch_laser_qwen3.sh" >&2; exit 1; }
(( GPUS % 2 == 0 )) || { echo "INFER_TP=2 must divide --gpus ($GPUS)" >&2; exit 1; }

# shellcheck source=/dev/null
source "$REPO/cluster_env.sh"
PARTITION=${PARTITION_OVERRIDE:-${PARTITION:-$(SR1_JOB_HOURS=$DURATION sr1_pick_partition)}}
ACCOUNT=${ACCOUNT:-nvr_israel_rlop}
sr1_find_submit_job || [[ $DRY_RUN -eq 1 ]] || {
    echo "ERROR: submit_job not found." >&2; exit 1; }

LOG_ROOT="$REPO/outputs/logs"
mkdir -p "$LOG_ROOT" "$OUT_DIR"

RUNNER="$LOG_ROOT/$NAME.runner.sh"
{
    echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    printf 'cd %q\n' "$FORK"
    echo 'source /home/uberger/scratch/miniconda3/etc/profile.d/conda.sh'
    echo 'conda activate laser'
    echo "export HF_HOME=${HF_HOME:-/home/uberger/scratch/cache/hf_cache}"
    # The model is cached; staying offline keeps a transient hub outage from looking like
    # a port failure, which is the whole class of confusion this run exists to avoid.
    echo 'export HF_HUB_OFFLINE=1'
    echo 'export TOKENIZERS_PARALLELISM=false'
    echo 'export VLLM_WORKER_MULTIPROC_METHOD=spawn'
    echo
    printf 'env N_GPUS=%q INFER_TP=2 NUM_CPUS=20 \\\n' "$GPUS"
    printf '    TRAIN_BATCH_SIZE=16 PPO_MINI_BATCH_SIZE=16 ROLLOUT_N=8 \\\n'
    printf '    ATTENTION_START_STEP=1 ENABLE_ATTENTION=True APPLY_HOOK_ATTENTION=True \\\n'
    printf '    APPLY_RECTIFICATION=True APPLY_SINK_SUPPRESSION=True \\\n'
    printf '    APPLY_EARLY_WEIGHTED_STABILITY=True \\\n'
    printf '    SAVE_FREQ=-1 TEST_FREQ=-1 TOTAL_EPOCHS=1 VAL_BEFORE_TRAIN=False \\\n'
    printf '    MODEL_PATH=%q \\\n' "$MODEL"
    printf '    TRAIN_FILES=%q VAL_FILES=%q \\\n' "$TRAIN_FILE" "$VAL_FILE"
    printf '    OUTPUT_DIR=%q PROJECT_NAME=laser EXP_NAME=%q \\\n' "$OUT_DIR" "$NAME"
    printf '    bash train.sh \\\n'
    # trainer.logger is hardcoded to ["console","wandb"] in train.sh and is NOT an env
    # var. Without a key wandb blocks on auth, which on a batch node is an invisible hang.
    printf '        trainer.logger=%q \\\n' '["console"]'
    printf '        trainer.total_training_steps=%q' "$STEPS"
    for a in ${EXTRA[@]+"${EXTRA[@]}"}; do printf ' \\\n        %q' "$a"; done
    echo
} > "$RUNNER"
chmod +x "$RUNNER"

echo "=========================================================================="
echo "Job       : $NAME   ($ACCOUNT, $PARTITION, ${DURATION}h, ${GPUS} GPU)"
echo "Model     : $MODEL"
echo "Data      : $TRAIN_FILE"
echo "Steps     : $STEPS   (attention rewards ON from step 1)"
echo "Out dir   : $OUT_DIR"
echo "=========================================================================="
cat "$RUNNER"
echo "=========================================================================="
echo "WHAT TO LOOK FOR, in order:"
echo "  1. no AttentionSliceCapturer import/discovery error -- the FSDP question"
echo "  2. reward extra-info carries attention_score and suppression_attention_score,"
echo "     and they are NOT identically zero (zero == an empty capture, or format 0)"
echo "  3. attention_score ~0.4-0.5 and suppression_attention_score ~0.36 -- the values"
echo "     laser.py measured independently on a different code path"
echo "  4. the FORMAT rate. If it is ~0, that is the SYSTEM_PROMPT in build_laser_data.py"
echo "     (written against their checkers, not copied from them), not the port."
echo "=========================================================================="

[[ $DRY_RUN -eq 1 ]] && { echo "[dry-run] not submitting."; exit 0; }

submit_job \
    --account "$ACCOUNT" \
    --partition "$PARTITION" \
    --name "$NAME" \
    --gpu "$GPUS" \
    --duration "$DURATION" \
    --outfile "$LOG_ROOT/$NAME.%j.out" \
    --logroot "$LOG_ROOT" \
    -c "bash $RUNNER"
