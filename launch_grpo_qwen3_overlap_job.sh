#!/bin/bash
# GRPO training for Qwen3-VL-8B with attention-overlap reward (non-colocated).
# All GPUs go to training; DINO runs locally on each process (or via --dino-api-base).
#
# Default: submits to the cluster via submit_job (SLURM).
# --direct: runs on the current node immediately (no SLURM), e.g. on an interactive GPU node.
#
# Usage:
#   WANDB_API_KEY=... NVIDIA_API_KEY=... bash launch_grpo_qwen3_overlap_job.sh [OPTIONS]
#   bash launch_grpo_qwen3_overlap_job.sh --direct --num-gpus 8 --w-overlap 0.3
#
# THE HACK-RESISTANT REWARD -- one flag, nothing else to set:
#
#   bash launch_grpo_qwen3_overlap_job.sh --overlap-metric auroc
#
# Why: mean_in divides by the map's own PEAK, so a map that merely FLATTENS scores
# higher without attending the box any better -- 32x more movement under flattening
# than under real grounding, and that is what the wov0.2/wov0.4 runs did (their MMStar
# gain vanishes once chance-corrected). auroc depends only on patch ORDER, so that
# route is closed by construction. Derivation + evidence:
# vlm_reasoning/wiki/hack-resistant-overlap-reward-plan.md
#
# --overlap-metric auroc also switches two coupled defaults, because neither
# transfers from mean_in. Both are still overridable, and both appear in the run name,
# so a name always states what actually ran.
#
#   mass_floor_tau  ->  0.0022   Not optional: auroc is rank-based and so blind to a
#                                model withdrawing attention from the image. 0.0022 =
#                                p10 of the reference model's image_mass (stable at
#                                0.0018-0.0029 across all seven offline collections).
#   w_overlap       ->  0.11     GRPO normalises advantages within the group, so what
#                                matters is the reward term's SPREAD, and the auroc
#                                composite has ~3.6x the per-sample sd of mean_in
#                                (0.13 vs 0.036). 0.11 reproduces the pressure of
#                                mean_in's wov0.4 -- the better of the two runs
#                                already trained (pooled 70.80 vs 70.11 over 54,387
#                                items, 9-8 on benchmark wins). Reusing 0.4 here would
#                                apply ~3.6x the intended pressure.
#
# Environment overrides:
#   PARTITION=batch_singlenode   DURATION=4 (hours)
#   SAVE_STEPS=10   CKPT_KEEP_EVERY=200
#   NVIDIA_API_KEY / OPENAI_API_KEY / OPENAI_BASE_URL / JUDGE_MODEL
#   WANDB_API_KEY   (omit -> offline)   HF_TOKEN
set -euo pipefail

SCRIPT_PATH="$(realpath "$0")"
REPO=/home/uberger/scratch/research/saliency_r1
CONDA_SH=/home/uberger/scratch/miniconda3/etc/profile.d/conda.sh
CONDA_ENV=saliency_r1_qwen3
HF_HOME=${HF_HOME:-/home/uberger/scratch/cache/hf_cache}

# ---------- SLURM defaults ----------
ACCOUNT=nvr_israel_rlop
PARTITION=${PARTITION:-batch_singlenode}
DURATION=${DURATION:-4}

# ---------- training defaults ----------
MODEL="$REPO/checkpoint/coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged"
NUM_GPUS=8
OUTPUT_DIR=""
MAX_COMPLETION_LENGTH=1024
SAVE_STEPS=${SAVE_STEPS:-10}
CKPT_KEEP_EVERY=${CKPT_KEEP_EVERY:-200}
EXTRA_ARGS=""
DIRECT=false

# ---------- overlap-reward defaults ----------
# W_OVERLAP and MASS_FLOOR_TAU are metric-dependent; the auroc values are applied
# after arg parsing, only if the user did not set them explicitly (see below).
W_OVERLAP=0.2
TOKEN_REDUCTION=mean
OVERLAP_HEADS="28,31"
OVERLAP_LAYER=22
BOX_THRESHOLD=0.10
MAX_BOX_AREA=0.5
OVERLAP_METRIC=mean_in   # mean_in (incumbent default) | auroc (hack-resistant; see docs/)
MASS_FLOOR_TAU=""        # unset -> off for mean_in, 0.0022 for auroc (see below).
                         # Pass 0 to force it off explicitly.
DINO_API_BASE=""

# ---------- parse args ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --direct)                 DIRECT=true;                  shift ;;
        --model)                  MODEL="$2";                   shift 2 ;;
        --num-gpus)               NUM_GPUS="$2";                shift 2 ;;
        --output-dir)             OUTPUT_DIR="$2";              shift 2 ;;
        --partition)              PARTITION="$2";               shift 2 ;;
        --duration)               DURATION="$2";                shift 2 ;;
        --nvidia-api-key)         NVIDIA_API_KEY="$2";          shift 2 ;;
        --openai-api-key)         OPENAI_API_KEY="$2";          shift 2 ;;
        --wandb-api-key)          WANDB_API_KEY="$2";           shift 2 ;;
        --hf-token)               HF_TOKEN="$2";                shift 2 ;;
        --max-completion-length)  MAX_COMPLETION_LENGTH="$2";   shift 2 ;;
        --w-overlap)              W_OVERLAP="$2"; W_OVERLAP_SET=1; shift 2 ;;
        --token-reduction)        TOKEN_REDUCTION="$2";         shift 2 ;;
        --overlap-heads)          OVERLAP_HEADS="$2";           shift 2 ;;
        --overlap-layer)          OVERLAP_LAYER="$2";           shift 2 ;;
        --box-threshold)          BOX_THRESHOLD="$2";           shift 2 ;;
        --max-box-area)           MAX_BOX_AREA="$2";            shift 2 ;;
        --overlap-metric)         OVERLAP_METRIC="$2";          shift 2 ;;
        --mass-floor-tau)         MASS_FLOOR_TAU="$2";          shift 2 ;;
        --dino-api-base)          DINO_API_BASE="$2";           shift 2 ;;
        *)                        EXTRA_ARGS="$EXTRA_ARGS $1";  shift ;;
    esac
done

REFORWARD_SALIENCY=True

# --overlap-metric auroc is a complete, self-sufficient configuration: it carries the
# two settings that do not transfer from mean_in, so the only change needed from a
# previous run is the metric flag itself. An explicit --w-overlap / --mass-floor-tau
# still wins (pass --mass-floor-tau 0 to force the floor off). mean_in is untouched,
# so a bare invocation is still bit-identical to the runs already trained.
if [[ "$OVERLAP_METRIC" == "auroc" ]]; then
    [[ -z "$MASS_FLOOR_TAU" ]] && MASS_FLOOR_TAU=0.0022
    [[ -z "${W_OVERLAP_SET:-}" ]] && W_OVERLAP=0.11
fi

# ---------- naming: every swept HP appears in the model AND wandb name ----------
N_HEADS=$(echo "$OVERLAP_HEADS" | awk -F, '{print NF}')
SUFFIX="__wov${W_OVERLAP}_${N_HEADS}head_tr${TOKEN_REDUCTION}"
# Only non-default metric settings extend the suffix, so existing mean_in run names
# (and the checkpoints already on disk) stay exactly as they are.
[[ "$OVERLAP_METRIC" != "mean_in" ]] && SUFFIX="${SUFFIX}_${OVERLAP_METRIC}"
[[ -n "$MASS_FLOOR_TAU" ]] && SUFFIX="${SUFFIX}_mf${MASS_FLOOR_TAU}"
MODEL_SLUG=$(echo "$MODEL" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9_]/_/g')
RUN_NAME="grpo-${MODEL_SLUG}-overlap${SUFFIX}"
[[ -z "$OUTPUT_DIR" ]] && OUTPUT_DIR="$REPO/checkpoint/${RUN_NAME}"
mkdir -p "$OUTPUT_DIR"

REWARD_WEIGHTS="1.0 ${W_OVERLAP} 1.0 1.0"

echo "=========================================================================="
echo "Model:          $MODEL"
echo "GPUs:           $NUM_GPUS (all training, no colocated sidecars)"
echo "Overlap reward: layer=$OVERLAP_LAYER heads=[$OVERLAP_HEADS] token_reduction=$TOKEN_REDUCTION w_overlap=$W_OVERLAP"
echo "Metric:         $OVERLAP_METRIC$([[ -n "$MASS_FLOOR_TAU" ]] && echo " mass_floor_tau=$MASS_FLOOR_TAU" || echo " (no mass floor)")"
echo "DINO:           box_threshold=$BOX_THRESHOLD max_box_area=$MAX_BOX_AREA  $([[ -n "$DINO_API_BASE" ]] && echo "served=$DINO_API_BASE" || echo 'local-on-device')"
echo "Run name:       $RUN_NAME"
echo "Output dir:     $OUTPUT_DIR"
echo "Mode:           $($DIRECT && echo 'direct (no SLURM)' || echo "SLURM ($PARTITION, ${DURATION}h)")"
echo "Judge key:      $([[ -n "${NVIDIA_API_KEY:-}${OPENAI_API_KEY:-}" ]] && echo '(set)' || echo '(MISSING - openai_reward will fail)')"
echo "WandB:          $([[ -n "${WANDB_API_KEY:-}" ]] && echo '(online)' || echo '(offline)')"
[[ -n "$EXTRA_ARGS" ]] && echo "Extra args:     $EXTRA_ARGS"
echo "=========================================================================="

# ---------- SLURM path ----------
if ! $DIRECT; then
    if ! command -v submit_job >/dev/null 2>&1; then
        for CI_ROOT in \
            /lustre/fs1/portfolios/adlr/projects/adlr_other_infra/release/cluster-interface \
            /lustre/fsw/portfolios/adlr/projects/adlr_other_infra/release/cluster-interface; do
            for CAND in "$CI_ROOT/latest" $(ls -1dt "$CI_ROOT"/*/ 2>/dev/null); do
                if [ -x "${CAND%/}/submit_job" ]; then export PATH="${CAND%/}:$PATH"; break 2; fi
            done
        done
    fi
    command -v submit_job >/dev/null 2>&1 || {
        echo "ERROR: submit_job not found under cluster-interface paths. Use --direct to run without SLURM." >&2
        exit 1
    }

    LOG_ROOT="$REPO/outputs/logs"
    mkdir -p "$LOG_ROOT"

    submit_job \
        --account "$ACCOUNT" \
        --partition "$PARTITION" \
        --name "$RUN_NAME" \
        --gpu "$NUM_GPUS" \
        --duration "$DURATION" \
        --autoresume_uninstrumented \
        --outfile "$LOG_ROOT/${RUN_NAME}.%j.out" \
        --logroot "$LOG_ROOT" \
        -c "bash -c '
            export HF_HOME=$HF_HOME;
            export WANDB_API_KEY=${WANDB_API_KEY:-};
            ${HF_TOKEN:+export HF_TOKEN=$HF_TOKEN;}
            ${NVIDIA_API_KEY:+export NVIDIA_API_KEY=$NVIDIA_API_KEY;}
            ${OPENAI_API_KEY:+export OPENAI_API_KEY=$OPENAI_API_KEY;}
            ${OPENAI_BASE_URL:+export OPENAI_BASE_URL=$OPENAI_BASE_URL;}
            ${JUDGE_MODEL:+export JUDGE_MODEL=$JUDGE_MODEL;}
            bash $SCRIPT_PATH --direct \
                --num-gpus $NUM_GPUS \
                --model $MODEL \
                --output-dir $OUTPUT_DIR \
                --max-completion-length $MAX_COMPLETION_LENGTH \
                --w-overlap $W_OVERLAP \
                --token-reduction $TOKEN_REDUCTION \
                --overlap-heads $OVERLAP_HEADS \
                --overlap-layer $OVERLAP_LAYER \
                --box-threshold $BOX_THRESHOLD \
                --max-box-area $MAX_BOX_AREA \
                ${DINO_API_BASE:+--dino-api-base $DINO_API_BASE} \
                $EXTRA_ARGS
        '"
    echo "Submitted $RUN_NAME"
    exit 0
fi

# ---------- direct path ----------
source "$CONDA_SH"
# conda activate + activate.d hooks (e.g. cuda-nvcc's, which expands
# $NVCC_PREPEND_FLAGS with no default) assume nounset is OFF; our set -u makes
# that a fatal "unbound variable". Disable nounset for activation only.
set +u
conda activate "$CONDA_ENV"
set -u
# Activating across conda installs -- e.g. when this script is launched (bash)
# from a fish shell that already had a different env active -- can leave a stale
# env's bin/ ahead of ours on PATH, so `python` resolves to the WRONG interpreter
# even though CONDA_DEFAULT_ENV/CONDA_PREFIX are correct. Force this env's bin to
# the front and clear bash's command hash so the right python/torchrun are used.
[ -n "${CONDA_PREFIX:-}" ] || { echo "ERROR: 'conda activate $CONDA_ENV' failed (no CONDA_PREFIX)." >&2; exit 1; }
export PATH="$CONDA_PREFIX/bin:$PATH"
hash -r
if [ "$(command -v python)" != "$CONDA_PREFIX/bin/python" ]; then
    echo "ERROR: python resolves to '$(command -v python)', expected '$CONDA_PREFIX/bin/python' (env '$CONDA_ENV')." >&2
    exit 1
fi

source "$REPO/setup_cuda_home.sh"
bash "$REPO/check_cuda_home.sh" || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DS_BUILD_OPS=0
export HF_HOME
export HF_HUB_OFFLINE=1
export HF_TOKEN=${HF_TOKEN:-}
export WANDB_API_KEY=${WANDB_API_KEY:-}
[ -z "${WANDB_API_KEY:-}" ] && export WANDB_MODE=offline
export WANDB_PROJECT=vlm_reasoning
export WANDB_ENTITY=nvr-israel
export WANDB_RUN_ID=${WANDB_RUN_ID:-$RUN_NAME}
export WANDB_NAME=${WANDB_NAME:-$RUN_NAME}
export WANDB_RESUME=${WANDB_RESUME:-allow}
export OVERLAP_STEPS_DEVICE=${OVERLAP_STEPS_DEVICE:-cpu}
export OVERLAP_STEPS_CKPT=${OVERLAP_STEPS_CKPT:-$REPO/checkpoint/steps_classifier/best}
[ -d "$OVERLAP_STEPS_CKPT/encoder" ] || {
    echo "ERROR: steps-classifier ckpt not found at $OVERLAP_STEPS_CKPT (need encoder/ tokenizer/ head.pt). Set OVERLAP_STEPS_CKPT to a valid path." >&2
    exit 1
}
[ -n "${NVIDIA_API_KEY:-}" ] && export NVIDIA_API_KEY
[ -n "${OPENAI_API_KEY:-}" ] && export OPENAI_API_KEY
[ -n "${OPENAI_BASE_URL:-}" ] && export OPENAI_BASE_URL
[ -n "${JUDGE_MODEL:-}" ] && export JUDGE_MODEL

# Sync the tracked overlap-reward sources into the (gitignored) live trl_repo tree.
# Same copies patch_trl_qwen3.sh makes; repeated here because the launcher does not
# call that script, so without this an edit to trl/ silently does not reach training.
cp "$REPO/trl/overlap_steps.py"           "$REPO/trl_repo/trl/trainer/overlap_steps.py"
cp "$REPO/trl/rewards/overlap_rewards.py" "$REPO/trl_repo/trl/rewards/overlap_rewards.py"
cp "$REPO/trl/scripts/utils.py"           "$REPO/trl_repo/trl/scripts/utils.py"
cp "$REPO/trl/grpo_vlm_qwen3.py"          "$REPO/trl_repo/examples/scripts/grpo_vlm_qwen3.py"
cp "$REPO/trl/grpo_trainer_qwen3.py"      "$REPO/trl_repo/trl/trainer/grpo_trainer_qwen3.py"
echo "Synced overlap-reward sources into trl_repo/"

cd "$REPO/trl_repo"

_cleanup_checkpoints() {
    local output_dir="$1" prev_latest=""
    while true; do
        sleep 30
        local latest
        latest=$(ls -d "$output_dir"/checkpoint-* 2>/dev/null | sed 's|.*/checkpoint-||' | sort -n | tail -1 || true)
        if [[ -n "$latest" && "$latest" != "$prev_latest" ]]; then
            prev_latest="$latest"
            ls -d "$output_dir"/checkpoint-* 2>/dev/null | sed 's|.*/checkpoint-||' | sort -n | while read -r step; do
                if (( step % CKPT_KEEP_EVERY != 0 )) && [[ "$step" != "$latest" ]]; then
                    echo "[checkpoint cleanup] Removing $output_dir/checkpoint-$step"
                    rm -rf "$output_dir/checkpoint-$step"
                fi
            done
        fi
    done
}
_cleanup_checkpoints "$OUTPUT_DIR" &
CLEANUP_PID=$!
trap "kill $CLEANUP_PID 2>/dev/null; wait $CLEANUP_PID 2>/dev/null" EXIT

RESUME_FLAG=""
LATEST_CKPT=$(ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | sed 's|.*/checkpoint-||' | sort -n | tail -1 || true)
[ -n "$LATEST_CKPT" ] && RESUME_FLAG="--resume_from_checkpoint $OUTPUT_DIR/checkpoint-$LATEST_CKPT"

MASTER_PORT=${MASTER_PORT:-$(shuf -i 29500-65000 -n 1)}

DINO_FLAG=""
[[ -n "$DINO_API_BASE" ]] && DINO_FLAG="--dino_api_base $DINO_API_BASE"
# Omitted entirely when unset, so the dataclass default (None = floor off) applies.
# A value of 0 is still passed through: _mass_gate treats tau<=0 as "off", so an
# explicit --mass-floor-tau 0 disables the floor without falling back to the auroc
# default above.
MASS_FLOOR_FLAG=""
[[ -n "$MASS_FLOOR_TAU" ]] && MASS_FLOOR_FLAG="--mass_floor_tau $MASS_FLOOR_TAU"

accelerate launch \
    --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    --num_processes "$NUM_GPUS" \
    --main_process_port "$MASTER_PORT" \
    examples/scripts/grpo_vlm_qwen3.py \
    --model_name_or_path "$MODEL" \
    --attn_implementation sdpa \
    --output_dir "$OUTPUT_DIR" \
    --learning_rate 1e-5 \
    --torch_dtype bfloat16 \
    --max_prompt_length 2048 \
    --max_completion_length "$MAX_COMPLETION_LENGTH" \
    --reforward_saliency "$REFORWARD_SALIENCY" \
    --reward_variant ours \
    --overlap_layer "$OVERLAP_LAYER" \
    --overlap_heads "$OVERLAP_HEADS" \
    --token_reduction "$TOKEN_REDUCTION" \
    --box_threshold "$BOX_THRESHOLD" \
    --max_box_area "$MAX_BOX_AREA" \
    --overlap_metric "$OVERLAP_METRIC" \
    $MASS_FLOOR_FLAG \
    $DINO_FLAG \
    --reward_weights $REWARD_WEIGHTS \
    --use_peft \
    --lora_target_modules q_proj v_proj \
    --log_completions \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_generations 8 \
    --report_to wandb \
    --logging_steps 5 \
    --save_steps "$SAVE_STEPS" \
    --num_train_epochs 3 \
    --temperature 1 \
    $RESUME_FLAG \
    $EXTRA_ARGS

echo "Finished $RUN_NAME"
