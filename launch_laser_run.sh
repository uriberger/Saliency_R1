#!/usr/bin/env bash
# The LASER baseline run: 45K, their hyperparameters, on Qwen3-VL-8B.
#
#   bash launch_laser_run.sh --name laser-45k                 # submit
#   bash launch_laser_run.sh --name laser-45k --dry-run       # print the runner, submit nothing
#
# Gated on the smoke run (launch_laser_smoke.sh) having passed. The smoke is what proved
# the attention capture fires under FSDP and that the rewards are non-zero; this is the
# same configuration with their batch sizes and their ATTENTION_START_STEP.
#
# THIS IS A ~9-DAY JOB. Sized from the smoke's measured 286 s/step:
#
#   45,421 rows / 512 = 89 steps/epoch x 2 epochs        = 178 optimizer steps
#   4,096 sequences/step (512 prompts x 8) vs the smoke's 128, on 8 GPUs vs 4
#   -> ~76 min/step -> ~226 h
#
# Treat that as an order of magnitude, not a promise: every step-time estimate in this
# project has been wrong at least once, and generation is 75% of it, so the real number
# moves with how long the cold-started model actually writes.
#
# THE LEVERS, IF NINE DAYS IS TOO LONG
#   --max-response 4096   roughly halves generation (the dominant term) -> ~5 days, but
#                         format drops 0.729 -> 0.562, i.e. a third fewer rollouts carry
#                         an attention signal. The measurement is in launch_laser_smoke.sh.
#   --epochs 1            halves it, and deviates from their 2.
#   --gpus 16             two nodes; generation scales, so ~5 days. Longer queue.
#
# AUTORESUME. No partition here runs longer than 8 h, so this MUST survive relaunch.
# verl's `trainer.resume_mode: auto` picks up the latest checkpoint in
# default_local_dir by itself, and submit_job --autoresume_uninstrumented relaunches at
# the wall limit. SAVE_FREQ is their 10, so a relaunch loses at most 10 steps -- which at
# 76 min/step is still ~13 h, so do not lower it casually and do not raise it at all.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

NAME="laser-45k"
GPUS=8
DURATION=8
EPOCHS=2
MAX_RESPONSE=8192
TRAIN_BATCH=512
MINI_BATCH=256
ATTENTION_START=20
FORK="$REPO/laser_fork"
DATA="$REPO/cold_data/laser/verl"
MODEL="/home/uberger/scratch/research/saliency_r1/checkpoint/laser_coldstart_qwen3_vl_8b_revisual"
OUT_DIR="$REPO/outputs/laser/run45k"
PARTITION_OVERRIDE=""
DRY_RUN=0
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)      DRY_RUN=1;               shift   ;;
        --name)         NAME="$2";               shift 2 ;;
        --gpus)         GPUS="$2";               shift 2 ;;
        --epochs)       EPOCHS="$2";             shift 2 ;;
        --max-response) MAX_RESPONSE="$2";       shift 2 ;;
        --model)        MODEL="$2";              shift 2 ;;
        --out-dir)      OUT_DIR="$2";            shift 2 ;;
        --duration)     DURATION="$2";           shift 2 ;;
        --partition)    PARTITION_OVERRIDE="$2"; shift 2 ;;
        --)             shift; EXTRA+=("$@"); break ;;
        *)              EXTRA+=("$1");           shift ;;
    esac
done

resolve() { readlink -f "$1" 2>/dev/null || true; }
FORK=$(resolve "$FORK"); [ -n "$FORK" ] || FORK="$REPO/laser_fork"
DATA=$(resolve "$DATA"); [ -n "$DATA" ] || DATA="$REPO/cold_data/laser/verl"
OUT_DIR=$(resolve "$OUT_DIR"); [ -n "$OUT_DIR" ] || OUT_DIR="$REPO/outputs/laser/run45k"

TRAIN_FILE="$DATA/train_45k.parquet"
VAL_FILE="$DATA/val_45k.parquet"
for f in "$TRAIN_FILE" "$VAL_FILE"; do
    [ -f "$f" ] || { echo "MISSING: $f -- run build_laser_data.py first" >&2; exit 1; }
done
[ -d "$MODEL" ] || { echo "MISSING model: $MODEL" >&2; exit 1; }
grep -q "Qwen3VLTextAttention" "$FORK/verl/workers/actor/attention_capture.py" || {
    echo "The fork is not patched for Qwen3-VL: bash patch_laser_qwen3.sh" >&2; exit 1; }

# The parquet must have NO system turn -- a system prompt the cold start never saw drives
# format to 0.000 on this checkpoint, which silently zeroes both attention rewards.
# Uses the `laser` env's interpreter explicitly: this guard runs BEFORE conda activation
# (which happens inside the job), and the system python3 has no pyarrow.
LASER_PY=/home/uberger/scratch/miniconda3/envs/laser/bin/python
"$LASER_PY" - "$TRAIN_FILE" <<'PY'
import sys, pyarrow.parquet as pq
r = pq.ParquetFile(sys.argv[1]).read_row_group(0).slice(0, 1).to_pylist()[0]
roles = [m["role"] for m in r["prompt"]]
if "system" in roles:
    raise SystemExit(f"FAIL: prompt has a system turn {roles}. This parquet predates the "
                     "cold start and will score format 0. Rebuild with build_laser_data.py.")
print(f"  prompt roles: {roles}  (bare user turn, matches the cold start)")
PY

# ppo_max_token_len_per_gpu must exceed prompt + response, and rollout/ref inherit it.
TOKEN_BUDGET=$(( (2048 + MAX_RESPONSE) * 6 / 5 ))

# shellcheck source=/dev/null
source "$REPO/cluster_env.sh"
PARTITION=${PARTITION_OVERRIDE:-${PARTITION:-$(SR1_JOB_HOURS=$DURATION sr1_pick_partition)}}
ACCOUNT=${ACCOUNT:-nvr_israel_rlop}
sr1_find_submit_job || [[ $DRY_RUN -eq 1 ]] || { echo "ERROR: submit_job not found." >&2; exit 1; }

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
    echo 'export HF_HUB_OFFLINE=1'
    echo 'export TOKENIZERS_PARALLELISM=false'
    echo 'export VLLM_WORKER_MULTIPROC_METHOD=spawn'
    echo
    printf 'env N_GPUS=%q INFER_TP=2 NUM_CPUS=20 \\\n' "$GPUS"
    printf '    TRAIN_BATCH_SIZE=%q PPO_MINI_BATCH_SIZE=%q ROLLOUT_N=8 \\\n' "$TRAIN_BATCH" "$MINI_BATCH"
    printf '    MAX_RESPONSE_LENGTH=%q \\\n' "$MAX_RESPONSE"
    printf '    ATTENTION_START_STEP=%q ENABLE_ATTENTION=True APPLY_HOOK_ATTENTION=True \\\n' "$ATTENTION_START"
    printf '    APPLY_RECTIFICATION=True APPLY_SINK_SUPPRESSION=True \\\n'
    printf '    APPLY_EARLY_WEIGHTED_STABILITY=True \\\n'
    printf '    TOTAL_EPOCHS=%q SAVE_FREQ=10 TEST_FREQ=10 \\\n' "$EPOCHS"
    printf '    MODEL_PATH=%q \\\n' "$MODEL"
    printf '    TRAIN_FILES=%q VAL_FILES=%q \\\n' "$TRAIN_FILE" "$VAL_FILE"
    printf '    OUTPUT_DIR=%q PROJECT_NAME=laser EXP_NAME=%q \\\n' "$OUT_DIR" "$NAME"
    printf '    bash train.sh \\\n'
    printf '        trainer.logger=%q \\\n' '["console"]'
    printf '        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=%q' "$TOKEN_BUDGET"
    for a in ${EXTRA[@]+"${EXTRA[@]}"}; do printf ' \\\n        %q' "$a"; done
    echo
} > "$RUNNER"
chmod +x "$RUNNER"

echo "=========================================================================="
echo "Job        : $NAME   ($ACCOUNT, $PARTITION, ${DURATION}h chunks, ${GPUS} GPU, autoresume)"
echo "Model      : $MODEL"
echo "Data       : $TRAIN_FILE"
echo "Batch      : $TRAIN_BATCH prompts x 8 rollouts, mini $MINI_BATCH, $EPOCHS epochs"
echo "Response   : $MAX_RESPONSE tokens   (token budget/gpu $TOKEN_BUDGET)"
echo "Attention  : on from step $ATTENTION_START (their default; plain GRPO before that)"
echo "Out dir    : $OUT_DIR"
echo "=========================================================================="
cat "$RUNNER"
echo "=========================================================================="
echo "~178 optimizer steps, ~76 min/step from the smoke -> roughly NINE DAYS."
echo "verl resume_mode=auto picks up the latest checkpoint, so relaunches are safe."
echo "Watch: critic/score/max should exceed 1.3 once step >= $ATTENTION_START --"
echo "that is acc 1 + 0.3 format + the attention terms, and is the only evidence in"
echo "the step metrics that the attention rewards are firing at all."
echo "=========================================================================="

[[ $DRY_RUN -eq 1 ]] && { echo "[dry-run] not submitting."; exit 0; }

submit_job \
    --account "$ACCOUNT" \
    --partition "$PARTITION" \
    --name "$NAME" \
    --gpu "$GPUS" \
    --duration "$DURATION" \
    --autoresume_uninstrumented \
    --outfile "$LOG_ROOT/$NAME.%j.out" \
    --logroot "$LOG_ROOT" \
    -c "bash $RUNNER"
