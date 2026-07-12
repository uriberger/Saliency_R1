#!/bin/bash
# Submit a Saliency-R1 GRPO training job via ADLR submit_job, with auto-resume so
# it survives the wall-clock limit. Mirrors vlm_reasoning/scripts/slurm/launch_vlm_job.sh
# but runs on a BARE GPU node (no container image): it activates the `saliency_r1`
# conda env on lustre and launches accelerate + DeepSpeed Zero3.
#
# Usage:
#   OPENAI_API_KEY=sk-... WANDB_API_KEY=... bash launch_grpo_job.sh                 # 3B smoke test, 1 GPU
#   ... bash launch_grpo_job.sh --model peterant330/Saliency-R1-CI-v2 --num-gpus 8  # paper 7B config
#
# Any flag not parsed below is forwarded verbatim to grpo_vlm.py via EXTRA_ARGS.
#
# Environment overrides:
#   PARTITION=batch_block1      DURATION=4 (hours)
#   OPENAI_API_KEY / OPENAI_BASE_URL   consumed by trl/rewards/openai_rewards.py (LLM-as-judge)
#   WANDB_API_KEY                       omit -> WANDB_MODE=offline
#   HF_TOKEN                            for gated model/dataset downloads
set -e

# ADLR cluster-interface tools (submit_job, etc.) on PATH.
export PATH="/lustre/fs1/portfolios/adlr/projects/adlr_other_infra/release/cluster-interface/21.1_2026-04-15_21-25-57:$PATH"

# ---------- cluster / project constants ----------
ACCOUNT=nvr_israel_rlop
PARTITION=${PARTITION:-batch_block1}
DURATION=${DURATION:-4}
REPO=/home/uberger/scratch/research/saliency_r1
CONDA_SH=/home/uberger/scratch/miniconda3/etc/profile.d/conda.sh
CONDA_ENV=saliency_r1
HF_HOME=${HF_HOME:-/home/uberger/scratch/cache/hf_cache}

# ---------- training defaults (chosen: 3B smoke test, openai_reward kept) ----------
MODEL="Qwen/Qwen2.5-VL-3B-Instruct"
NUM_GPUS=1
OUTPUT_DIR=""
EXTRA_ARGS=""

# ---------- parse args ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)       MODEL="$2";      shift 2 ;;
        --num-gpus)    NUM_GPUS="$2";   shift 2 ;;
        --output-dir)  OUTPUT_DIR="$2"; shift 2 ;;
        *)             EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

MODEL_SLUG=$(echo "$MODEL" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9_]/_/g')
[[ -z "$OUTPUT_DIR" ]] && OUTPUT_DIR="$REPO/checkpoint/grpo-${MODEL_SLUG}-saliency-r1"
JOB_NAME="grpo_saliency_r1_${MODEL_SLUG}"
LOG_ROOT="$REPO/outputs/logs"
mkdir -p "$LOG_ROOT" "$OUTPUT_DIR"

echo "Model:      $MODEL"
echo "GPUs:       $NUM_GPUS"
echo "Output dir: $OUTPUT_DIR"
echo "Job name:   $JOB_NAME"
echo "Partition:  $PARTITION  (duration ${DURATION}h, auto-resume)"
echo "Judge key:  $([[ -n "${NVIDIA_API_KEY:-}${OPENAI_API_KEY:-}" ]] && echo '(set)' || echo '(MISSING - openai_reward will fail)')  endpoint=${OPENAI_BASE_URL:-https://inference-api.nvidia.com} model=${JUDGE_MODEL:-azure/openai/gpt-4o-mini}"
echo "WandB:      $([[ -n "$WANDB_API_KEY" ]] && echo '(online)' || echo '(offline)')"
[[ -n "$EXTRA_ARGS" ]] && echo "Extra args: $EXTRA_ARGS"
echo ""

# submit_job runs the -c command on the bare node (no --image); /lustre is
# natively mounted. resume_from_checkpoint="" lets the trainer pick up the latest
# checkpoint in OUTPUT_DIR on each auto-resume relaunch.
submit_job \
    --account "$ACCOUNT" \
    --partition "$PARTITION" \
    --name "$JOB_NAME" \
    --gpu "$NUM_GPUS" \
    --duration "$DURATION" \
    --autoresume_uninstrumented \
    --outfile "$LOG_ROOT/${JOB_NAME}.%j.out" \
    --logroot "$LOG_ROOT" \
    -c "bash -c '
        source $CONDA_SH;
        conda activate $CONDA_ENV;
        export CUDA_HOME=\${CUDA_HOME:-/cm/shared/apps/cuda12.4/toolkit/12.4.1};
        export PATH=\$CUDA_HOME/bin:\$PATH;
        export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\${LD_LIBRARY_PATH:-};
        export HF_HOME=$HF_HOME;
        export HF_TOKEN=${HF_TOKEN:-};
        export WANDB_API_KEY=${WANDB_API_KEY:-};
        [ -z \"\$WANDB_API_KEY\" ] && export WANDB_MODE=offline;
        export WANDB_PROJECT=vlm_reasoning;
        export WANDB_ENTITY=nvr-israel;
        ${NVIDIA_API_KEY:+export NVIDIA_API_KEY=$NVIDIA_API_KEY;}
        ${OPENAI_API_KEY:+export OPENAI_API_KEY=$OPENAI_API_KEY;}
        ${OPENAI_BASE_URL:+export OPENAI_BASE_URL=$OPENAI_BASE_URL;}
        ${JUDGE_MODEL:+export JUDGE_MODEL=$JUDGE_MODEL;}
        cd $REPO/trl_repo;
        # Resume only if a checkpoint already exists (auto-resume relaunch); a
        # fresh first run has none and must NOT pass --resume_from_checkpoint.
        RESUME_FLAG=\"\";
        ls -d $OUTPUT_DIR/checkpoint-* >/dev/null 2>&1 && RESUME_FLAG=\"--resume_from_checkpoint True\";
        accelerate launch \
            --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
            --num_processes $NUM_GPUS \
            examples/scripts/grpo_vlm.py \
            --model_name_or_path $MODEL \
            --attn_implementation sdpa \
            --output_dir $OUTPUT_DIR \
            --learning_rate 1e-5 \
            --torch_dtype bfloat16 \
            --max_prompt_length 2048 \
            --max_completion_length 1024 \
            --use_peft \
            --lora_target_modules q_proj v_proj \
            --log_completions \
            --per_device_train_batch_size 8 \
            --gradient_accumulation_steps 2 \
            --num_generations 8 \
            --report_to wandb \
            --logging_steps 5 \
            --save_steps 200 \
            --num_train_epochs 3 \
            --temperature 1 \
            \$RESUME_FLAG \
            $EXTRA_ARGS
    '"

echo "Submitted $JOB_NAME"
