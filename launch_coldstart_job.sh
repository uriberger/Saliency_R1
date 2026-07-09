#!/bin/bash
# Submit the Saliency-R1 COLD-START SFT (Qwen3-VL-8B-Instruct) via ADLR submit_job,
# with auto-resume so it survives the wall-clock limit. Mirrors launch_grpo_job.sh but
# runs LLaMA-Factory (llamafactory-cli train <yaml>) in the `sr1_coldstart` conda env on
# a bare GPU node. DeepSpeed Zero3, LoRA. Rebuilds their peterant330/Saliency-R1-CI-v2
# cold-start on our model.
#
# Usage:
#   bash launch_coldstart_job.sh                          # paper scale: 8 GPUs, 4h chunks, auto-resume
#   bash launch_coldstart_job.sh --num-gpus 4
#   WANDB_API_KEY=... bash launch_coldstart_job.sh        # online WandB (else offline)
#   bash launch_coldstart_job.sh --config /path/other.yaml
#
# Any `key=value` after the flags is forwarded verbatim as a LLaMA-Factory (OmegaConf)
# override, e.g.:  bash launch_coldstart_job.sh per_device_train_batch_size=4
#
# Environment overrides:
#   PARTITION=batch_block1   DURATION=4 (hours)   HF_TOKEN=...   WANDB_API_KEY=...
set -e

# ADLR cluster-interface tools (submit_job, etc.) on PATH.
export PATH="/lustre/fs1/portfolios/adlr/projects/adlr_other_infra/release/cluster-interface/21.1_2026-04-15_21-25-57:$PATH"

# ---------- cluster / project constants ----------
ACCOUNT=nvr_israel_rlop
PARTITION=${PARTITION:-batch_block1}
DURATION=${DURATION:-4}
REPO=/home/uberger/scratch/research/saliency_r1
LF_DIR=/home/uberger/scratch/research/LLaMA-Factory
CONDA_SH=/home/uberger/scratch/miniconda3/etc/profile.d/conda.sh
CONDA_ENV=sr1_coldstart
HF_HOME=${HF_HOME:-/home/uberger/scratch/cache/hf_cache}

# ---------- training defaults ----------
CONFIG="$REPO/train/cold_start/qwen3_vl_8b_instruct_sft/train.yaml"
NUM_GPUS=8
OUTPUT_DIR=""
EXTRA_ARGS=""   # forwarded as OmegaConf key=value overrides

# ---------- parse args ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-gpus)    NUM_GPUS="$2";   shift 2 ;;
        --config)      CONFIG="$2";     shift 2 ;;
        --output-dir)  OUTPUT_DIR="$2"; shift 2 ;;
        *)             EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

# Default output dir = the value baked into the yaml (keep launcher + yaml in sync so
# checkpoint detection below points at the right place).
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR=$(grep -E '^output_dir:' "$CONFIG" | head -1 | sed 's/^output_dir:[[:space:]]*//')
fi

JOB_NAME="coldstart_qwen3_vl_8b_instruct_sft"
LOG_ROOT="$REPO/outputs/logs"
mkdir -p "$LOG_ROOT" "$OUTPUT_DIR"

echo "Config:     $CONFIG"
echo "GPUs:       $NUM_GPUS"
echo "Output dir: $OUTPUT_DIR"
echo "Job name:   $JOB_NAME"
echo "Partition:  $PARTITION  (duration ${DURATION}h, auto-resume)"
echo "WandB:      $([[ -n "$WANDB_API_KEY" ]] && echo '(online)' || echo '(offline)')"
[[ -n "$EXTRA_ARGS" ]] && echo "Overrides:  $EXTRA_ARGS"
echo ""

# submit_job runs the -c command on the bare node (no --image); /lustre is natively
# mounted. On each auto-resume relaunch we detect the latest checkpoint in OUTPUT_DIR
# and pass it as resume_from_checkpoint=<path>; a fresh first run has none and starts
# clean (the yaml sets overwrite_output_dir=false + save_only_model=false so state
# survives across relaunches).
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
        export WANDB_RUN_ID=\$(echo -n "coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5" | md5sum | cut -c1-8);
        export WANDB_RESUME=allow;
        # DeepSpeed under LLaMA-Factory must be launched via torchrun; use all N GPUs.
        export FORCE_TORCHRUN=1;
        export NPROC_PER_NODE=$NUM_GPUS;
        cd $LF_DIR;
        # Resume from the highest-numbered checkpoint if one exists (auto-resume relaunch).
        RESUME=\"\";
        LATEST=\$(ls -d $OUTPUT_DIR/checkpoint-* 2>/dev/null | sort -V | tail -1);
        [ -n \"\$LATEST\" ] && RESUME=\"resume_from_checkpoint=\$LATEST\";
        llamafactory-cli train $CONFIG \
            output_dir=$OUTPUT_DIR \
            \$RESUME \
            $EXTRA_ARGS
    '"

echo "Submitted $JOB_NAME"
