#!/usr/bin/env bash
# EXECUTE side of P2's smoke. launch_deepeyes_smoke_job.sh is the SUBMIT side; this runs
# inside the allocation and is not meant to be called by hand except to debug.
#
#   bash launch_deepeyes_smoke.sh --stage grpo  --out-dir ... --model ... --data-dir ...
#   bash launch_deepeyes_smoke.sh --stage tool  --out-dir ... --model ... --data-dir ...
#
# TWO STAGES, IN THIS ORDER, AND THE ORDER IS THE POINT.
#
#   grpo  Plain verl GRPO on geo3k. No tool, no reward of ours, no agent of ours. If
#         this does not run, nothing after it matters and the arm should be abandoned
#         the same day rather than debugged alongside new code. It is also where the
#         per-step time comes from, which baseline-port-estimates.md section 9 needs
#         because it currently contradicts itself by ~2x.
#
#   tool  One multi-turn rollout through the existing ImageZoomInTool, carrying the
#         three assertions that are silent when wrong. See deepeyes_tool_smoke.py.
#
# `set -e` is what makes stage `grpo` failing stop stage `tool` from running and
# producing a confident-looking answer about a stack that does not work.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

STAGE=""
OUT_DIR=""
DATA_DIR=""
MODEL="Qwen/Qwen3-VL-8B-Instruct"
FORK=""
GPUS=8
TP=2
STEPS=3
TRAIN_BATCH=16
ROLLOUT_N=8
MAX_PROMPT=4096
MAX_RESPONSE=2048
GPU_MEM=0.5
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage)        STAGE="$2";        shift 2 ;;
        --out-dir)      OUT_DIR="$2";      shift 2 ;;
        --data-dir)     DATA_DIR="$2";     shift 2 ;;
        --model)        MODEL="$2";        shift 2 ;;
        --fork)         FORK="$2";         shift 2 ;;
        --gpus)         GPUS="$2";         shift 2 ;;
        --tp)           TP="$2";           shift 2 ;;
        --steps)        STEPS="$2";        shift 2 ;;
        --train-batch)  TRAIN_BATCH="$2";  shift 2 ;;
        --rollout-n)    ROLLOUT_N="$2";    shift 2 ;;
        --max-prompt)   MAX_PROMPT="$2";   shift 2 ;;
        --max-response) MAX_RESPONSE="$2"; shift 2 ;;
        --gpu-mem)      GPU_MEM="$2";      shift 2 ;;
        --)             shift; EXTRA+=("$@"); break ;;
        *)              EXTRA+=("$1");     shift ;;
    esac
done

[[ -n "$STAGE"    ]] || { echo "ERROR: --stage grpo|tool is required." >&2; exit 2; }
[[ -n "$OUT_DIR"  ]] || { echo "ERROR: --out-dir is required." >&2; exit 2; }
[[ -n "$DATA_DIR" ]] || { echo "ERROR: --data-dir is required." >&2; exit 2; }
[[ -n "$FORK"     ]] || { echo "ERROR: --fork is required." >&2; exit 2; }

mkdir -p "$OUT_DIR"
export DEEPEYES_MROPE_DIR="$OUT_DIR/mrope"
export DEEPEYES_ASSERT_MROPE=${DEEPEYES_ASSERT_MROPE:-1}
mkdir -p "$DEEPEYES_MROPE_DIR"

echo "=========================================================================="
echo "stage      : $STAGE"
echo "model      : $MODEL"
echo "fork       : $FORK"
echo "out dir    : $OUT_DIR"
echo "gpus/tp    : $GPUS / $TP"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
echo "=========================================================================="

# The fork is what must execute, not whatever `pip list` thinks is installed. P0 left
# the `deepeyes` env pointing at deepeyes_fork, but a stack proof that proves the wrong
# stack is worse than no proof, so say out loud which tree answered.
python -c "import verl, os; print('verl ->', os.path.dirname(verl.__file__))"
python - "$FORK" <<'PY'
import os
import sys

import verl

want = os.path.realpath(os.path.join(sys.argv[1], "verl"))
got = os.path.realpath(os.path.dirname(verl.__file__))
if want != got:
    raise SystemExit(f"FAIL: verl resolves to {got}, expected {want}. Wrong conda env?")
from verl.utils import deepeyes_mrope_guard  # noqa: F401

print("  guard present; fork confirmed")
PY

case "$STAGE" in
grpo)
    TRAIN="$DATA_DIR/geo3k_train.parquet"
    VAL="$DATA_DIR/geo3k_val.parquet"
    for f in "$TRAIN" "$VAL"; do
        [ -f "$f" ] || { echo "MISSING $f -- run build_deepeyes_smoke_data.py" >&2; exit 1; }
    done

    # Everything below is verl's own defaults plus the four things a smoke run has to
    # change (batch, steps, no checkpointing, console logger). In particular:
    #   - data_source "hiyouga/geometry3k" routes to verl's BUILT-IN reward. There is no
    #     custom_reward_function line here on purpose.
    #   - use_kl_loss=False / use_kl_in_reward=False matches DeepEyes' own config
    #     (KL coefficient 0) and removes the ref policy's forward from the step, which
    #     is also what makes the measured step time comparable to the real run.
    #   - ppo_max_token_len_per_gpu must EXCEED max_prompt+max_response or verl asserts
    #     at the log-prob path, ~11 minutes in. 4096+2048=6144, so 8192.
    START=$(date +%s)
    set -x
    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        algorithm.use_kl_in_reward=False \
        data.train_files="$TRAIN" \
        data.val_files="$VAL" \
        data.train_batch_size="$TRAIN_BATCH" \
        data.max_prompt_length="$MAX_PROMPT" \
        data.max_response_length="$MAX_RESPONSE" \
        data.filter_overlong_prompts=True \
        data.truncation=error \
        data.image_key=images \
        actor_rollout_ref.model.path="$MODEL" \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.ppo_mini_batch_size="$TRAIN_BATCH" \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.mode=async \
        actor_rollout_ref.rollout.tensor_model_parallel_size="$TP" \
        actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEM" \
        actor_rollout_ref.rollout.enforce_eager=True \
        actor_rollout_ref.rollout.free_cache_engine=True \
        actor_rollout_ref.rollout.n="$ROLLOUT_N" \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        trainer.logger='["console"]' \
        trainer.project_name=deepeyes \
        trainer.experiment_name=p2-verl-smoke \
        trainer.default_local_dir="$OUT_DIR/ckpt" \
        trainer.n_gpus_per_node="$GPUS" \
        trainer.nnodes=1 \
        trainer.val_before_train=False \
        trainer.save_freq=-1 \
        trainer.test_freq=-1 \
        trainer.total_epochs=1 \
        trainer.total_training_steps="$STEPS" \
        ray_kwargs.ray_init.num_cpus=32 \
        ${EXTRA[@]+"${EXTRA[@]}"} 2>&1 | tee "$OUT_DIR/grpo.log"
    set +x
    END=$(date +%s)
    echo "WALL_CLOCK_GRPO_SECONDS=$((END - START))" | tee "$OUT_DIR/grpo.wall"
    python "$REPO/deepeyes_smoke_report.py" --stage grpo --out-dir "$OUT_DIR"
    ;;

tool)
    DATA="$DATA_DIR/zoom_smoke.parquet"
    [ -f "$DATA" ] || { echo "MISSING $DATA -- run build_deepeyes_smoke_data.py" >&2; exit 1; }
    python "$REPO/deepeyes_tool_smoke.py" \
        --model "$MODEL" \
        --data "$DATA" \
        --out-dir "$OUT_DIR" \
        --n-gpus "$GPUS" \
        --tp "$TP" \
        --max-prompt-length "$MAX_PROMPT" \
        --max-response-length "$MAX_RESPONSE" \
        --gpu-mem-util "$GPU_MEM" \
        ${EXTRA[@]+"${EXTRA[@]}"} 2>&1 | tee "$OUT_DIR/tool.log"
    ;;

*)
    echo "ERROR: --stage $STAGE is not a stage (grpo|tool)." >&2; exit 2 ;;
esac
