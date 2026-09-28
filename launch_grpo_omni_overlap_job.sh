#!/bin/bash
# GRPO training for Nemotron-3-Nano-Omni-30B-A3B with the attention-overlap reward.
#
# This is `launch_grpo_qwen3_overlap_colocated_job.sh` with the model changed and
# NOTHING ELSE that matters: LoRA r=16 alpha=32 on q/k/v_proj, lr 1e-5, per_device 1,
# grad_accum 8, num_generations 8, max_completion_length 1024, beta 0. Five things differ,
# and each one is here because the Omni forced it:
#
#   the accelerate config   accelerate_omni_plan_a.yaml, not deepspeed_zero3.yaml. It is
#                           TRL's multi_gpu.yaml with mixed_precision 'no' -- the model is
#                           already bf16 and autocast would promote every log-softmax over
#                           a 131k vocabulary to fp32. See PLAN A and the file's header
#   the conda envs          `nemotron` trains, `nemotron_vllm` generates -- see TWO ENVS
#   the harness             trl_repo_nemotron/, so trl_repo is not rewritten under the
#                           Qwen3-VL runs that are using it
#   gradient checkpointing  ON. It is what makes bfloat16 fit on one card
#   the LoRA target scope   the bare names land on an AUDIO tower here -- see THE TRAP
#
# ---------------------------------------------------------------------------
# PLAN A -- one whole copy per card. docs/omni-gpu-layout.md is the argument.
#
#     GPU 0       Grounding-DINO reward server      (127.0.0.1:$DINO_PORT)
#     GPU 1       vLLM generation server            (127.0.0.1:$VLLM_PORT)
#     GPU 2-7     6 training processes, each a WHOLE copy, bfloat16 + recompute
#
# No ZeRO-3 and no sharding. The Qwen3-VL run cuts ONE copy across all six training cards
# and re-collects it 16 times a step; on a 62 GB model that is what makes the step slow.
# A whole copy per card has nothing to re-collect -- the only traffic is averaging 1.22M
# LoRA gradients -- and measured 34.0 s on the training side against an 80 GB card's
# 73.3 GB peak, 5.9 GB spare. Batch stays 8, six processes, 48 rollouts per update.
#
# THE HEADROOM IS SMALL AND THE THING THAT PROTECTS IT IS THE PICTURE, not the question.
# On set_a the questions are 8-24 tokens and the Omni's pictures come to 266-286, so the
# longest real sequence is ~1,358 positions against the 1,327 benchmarked. But the Omni is
# NATIVE-RESOLUTION and climbs to 3,328 picture tokens on a large image. A training set
# with bigger pictures invalidates the memory argument and nothing else. This corpus is
# capped at 512px on the long side by grpo_vlm_qwen3.py's own resize.
#
# --vllm-gpus 2 is PLAN A': if the generation server needs two cards, the trainer drops to
# five processes and grad_accum goes 8 -> 10 to hold 48 rollouts. ~42 s a step. Do not
# reach past it for Plan B (one copy across a PAIR of cards): that halves the process
# count, doubles the batch and roughly doubles the step, to solve a memory problem the
# measurement says does not exist.
#
# DO NOT QUANTIZE. docs/omni-quantization.md measured all four corners: 16-bit with
# recompute is 34.0 s, 8-bit without is 68.1 s, 8-bit with is 85.3 s, and 16-bit without
# does not fit. 8 bits is 2.4x slower per matmul and that is larger than anything it buys.
# (If you ever do quantize anyway, 8 bits is the setting that does not move the attention
# and 4 bits flattens it by 3-5% on every concentration statistic.)
#
# ---------------------------------------------------------------------------
# TWO ENVS, and this is the one structural surprise in the file.
#
# vLLM 0.11 -- what `nemotron` has -- cannot serve this model. Its registry has no
# `NemotronH_Nano_Omni_Reasoning_V3`, and backporting is not a registry entry: its
# `nemotron_h` has no MoE layer type at all, and the Omni's `hybrid_override_pattern` is
# half `E`. vLLM >= 0.20 has it, and pins torch 2.11 (CUDA 13; the nodes are on driver
# 580, so that runs).
#
# Upgrading `nemotron` in place would drag the TRAINING side to torch 2.11 and
# transformers 4.57 -- and the training side is the part that is measured, shimmed against
# transformers 5.13 in seven places, and working. So the server gets its own environment:
#
#     nemotron        torch 2.8.0   transformers 5.13.0.dev0   -> the trainer
#     nemotron_vllm   torch 2.11.0  transformers 4.57.6        -> vLLM 0.20.2, the server
#
# They talk over HTTP and one NCCL communicator. The communicator is the part with no
# precedent here: NCCL 2.27 (torch 2.8) broadcasting to NCCL 2.28 (torch 2.11). If weight
# sync ever hangs on `init_communicator`, that is the first place to look and
# `--sync-all-weights` is not the workaround -- the versions are.
#
# The sync itself is scoped: SR1_VLLM_SYNC_LORA_ONLY pushes the 18 tensors a merged LoRA
# actually changes instead of all 7,349. Under LoRA the base is frozen and the server
# loaded that same base from the same checkpoint, so the rest is re-sending bytes it has.
# At a few ms of HTTP round trip apiece, 7,349 of them would be tens of seconds on a step
# whose whole budget is ~47 s. `--sync-all-weights` turns it off.
#
# ---------------------------------------------------------------------------
# THE TRAP: the audio tower eats your LoRA, silently.
#
# peft matches bare target names by SUFFIX, anywhere in the model. `q_proj,k_proj,v_proj`
# is safe on Qwen3-VL (its vision tower uses a fused `qkv`) and safe on RADIO. It is not
# safe here: the Omni carries a 24-layer AUDIO encoder using exactly those three names,
# which an image-only batch never runs. The first attempt put 144 of its 180 LoRA tensors
# there; they came back with no gradient at all, and every log line looked right.
#
# `vlm_family.NemotronVL.lora_target_modules` rewrites the names into
# `language_model\..*\.(q_proj|k_proj|v_proj)` -- a single string, which peft treats as a
# regex over the whole path -- and the trainer then ASSERTS the adapters landed on the six
# attention layers [5, 12, 19, 26, 33, 42]. It refuses to start otherwise.
#
# ---------------------------------------------------------------------------
# THE SALIENCY LAYER MOVED, and it is the one number that could not be carried over.
#
# The Qwen3-VL runs read layer 22 of 36. **The Omni has an attention matrix at only 6 of
# its 52 layers** -- `hybrid_override_pattern` is MEMEM*EMEM... and the `*` positions are
# [5, 12, 19, 26, 33, 42]. 22 is a Mamba layer here; pointing the reward at it attaches
# the capture hook to nothing, so the trainer refuses rather than training on a reward
# that is silently zero. 33 is the default because it is the nearest attention layer to
# the same RELATIVE DEPTH (63% against Qwen3-VL's 61%), and that is the whole of the
# argument for it -- there is no head-selection probe behind it.
#
# The HEAD PAIR carries over even less. 28 and 31 were chosen on Qwen3-VL-8B by a probe;
# on any other model the same two indices name two arbitrary heads. They are kept so the
# command line differs in as little as possible, and any attention number read off this
# run has to say so. `--overlap-heads` takes a list; passing all 32 is the alternative.
#
# ---------------------------------------------------------------------------
# THE PREFLIGHT, which is the thing to run first -- not the step time.
#
# The learning signal has to travel back through 23 Mamba layers on a torch fallback to
# reach the 6 attention layers the LoRA sits on. It does -- all 36 adapter tensors finite
# and non-zero, gradient norms 3.0e-04 to 6.3e-01 -- but a run that starts without
# checking trains a reward-shaped nothing for an hour before anyone looks. `--preflight`
# (ON by default) runs `omni_train_step_bench.py --grads-only` on GPU 0 before any sidecar
# starts: it loads the model, attaches the scoped LoRA, turns recompute on, takes one
# optimizer step and one backward, and refuses the job if any tensor comes back missing,
# non-finite or exactly zero. It costs ~4 minutes and holds nothing afterwards.
#
# KNOWN AND UNRESOLVED, so it does not surprise anyone: the Omni does not reproduce its
# own gradients run to run, at bfloat16 as well as at 8 bits, so it is not a quantization
# artefact. It is measured as a WORST-PER-TENSOR relative difference over 36 tensors,
# which whichever tensor has the smallest norm dominates; a global metric was never
# computed and would very likely be far smaller. Probably benign for GRPO. Do not chase it
# unless something else points there.
#
# ---------------------------------------------------------------------------
# Usage:
#   WANDB_API_KEY=... NVIDIA_API_KEY=... bash launch_grpo_omni_overlap_job.sh --max-steps 50
#   bash launch_grpo_omni_overlap_job.sh --direct --num-gpus 8        # on an interactive node
#
# Environment overrides:
#   PARTITION  DURATION  DINO_PORT  VLLM_PORT  VLLM_GPU_MEM  VLLM_MAX_MODEL_LEN
#   SAVE_STEPS  CKPT_KEEP_EVERY  OVERLAP_STEPS_CKPT
#   NVIDIA_API_KEY / OPENAI_API_KEY / OPENAI_BASE_URL / JUDGE_MODEL
#   WANDB_API_KEY   (omit -> offline)   HF_TOKEN
set -euo pipefail

SCRIPT_PATH="$(realpath "$0")"
REPO="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
CONDA_SH=/home/uberger/scratch/miniconda3/etc/profile.d/conda.sh
TRAIN_ENV=nemotron            # torch 2.8, transformers 5.13.0.dev0 -- the measured side
VLLM_ENV=nemotron_vllm        # torch 2.11, vLLM 0.20.2 -- the only one that serves the Omni
HARNESS="$REPO/trl_repo_nemotron"
export HF_HOME=${HF_HOME:-/home/uberger/scratch/cache/hf_cache}

# ---------- SLURM defaults ----------
source "$REPO/cluster_env.sh"
ACCOUNT=nvr_israel_rlop
PARTITION=${PARTITION:-}
EXCLUDE_HOSTS=${EXCLUDE_HOSTS:-}
# 1 h, for the same reason the Qwen3-VL launcher asks for 1 h: it is the shortest useful
# chunk, it fits the most backfill windows, and at <= 2 h the job is also eligible for
# batch_short (PriorityTier 40). Warm-up is longer here than there -- the preflight is ~4
# min and vLLM loads 62 GB -- so a 1 h chunk is mostly warm-up on the FIRST allocation and
# mostly training on every requeue after it.
DURATION=${DURATION:-2}

# ---------- training defaults: identical to the Qwen3-VL runs ----------
MODEL=${MODEL:-nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16}
NUM_GPUS=8
OUTPUT_DIR=""
MAX_COMPLETION_LENGTH=1024
NUM_GENERATIONS=8
GRAD_ACCUM=8
PER_DEVICE_BATCH=1
LEARNING_RATE=1e-5
LORA_TARGETS=${LORA_TARGETS:-q_proj,k_proj,v_proj}
BETA=0
MAX_STEPS=${MAX_STEPS:-50}
SAVE_STEPS=${SAVE_STEPS:-10}
CKPT_KEEP_EVERY=${CKPT_KEEP_EVERY:-50}
DATASET=${DATASET:-$REPO/cold_data/grpo_sets/set_a}
DIRECT=false
EXTRA_ARGS=""

# ---------- overlap-reward defaults ----------
W_OVERLAP=0.2
TOKEN_REDUCTION=mean
OVERLAP_HEADS="28,31"
OVERLAP_LAYER=33              # see THE SALIENCY LAYER MOVED
OVERLAP_METRIC=mean_in
BOX_THRESHOLD=0.10
MAX_BOX_AREA=0.5

# ---------- sidecars and layout ----------
DINO_PORT=${DINO_PORT:-8100}
VLLM_PORT=${VLLM_PORT:-8000}
VLLM_GPUS_N=1                 # 2 = Plan A'
# 0.90 is what the Qwen3-VL launcher uses. The Omni is 62 GB of weights on an 80 GB card,
# so 0.90 leaves ~9 GB of KV cache -- enough for 8 rollouts of a ~1,360-token sequence,
# and `omni_vllm_probe.py` is what measured it rather than assuming it.
VLLM_GPU_MEM=${VLLM_GPU_MEM:-0.90}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-4096}
# torch.compile + CUDA-graph capture on a 52-layer mixture of experts is minutes of
# startup that a 1-2 h allocation pays again on every requeue. Eager by default for that
# reason alone; VLLM_ENFORCE_EAGER=False buys back generation throughput on a long run.
VLLM_ENFORCE_EAGER=${VLLM_ENFORCE_EAGER:-True}
# A Mamba hybrid needs ONE state block per concurrently decoding sequence, and this card
# has ~914. vLLM's default max_num_seqs is 1024, so it refuses before anything runs. One
# GRPO step asks for gen_batch sequences at most (48 here); 64 leaves margin.
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-64}
# The FLAN-T5 observe-step classifier. The Qwen3-VL launcher puts it on the training GPU
# because CPU was the dominant per-step cost there; here the training card is at 65-73 GB
# of 79 and the first attempt died in NCCL's allreduce with "Cuda failure 2 'out of
# memory'". A 110M-parameter encoder is not worth one of the ~6 GB that are left. It costs
# step time, and the node has 96 cores for six ranks to spend.
OVERLAP_STEPS_DEVICE=${OVERLAP_STEPS_DEVICE:-cpu}
OVERLAP_STEPS_CKPT=${OVERLAP_STEPS_CKPT:-$REPO/checkpoint/steps_classifier/best}
PREFLIGHT=${PREFLIGHT:-true}
SYNC_LORA_ONLY=true

# ---------- parse args ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --direct)                 DIRECT=true;                  shift ;;
        --model)                  MODEL="$2";                   shift 2 ;;
        --num-gpus)               NUM_GPUS="$2";                shift 2 ;;
        --output-dir)             OUTPUT_DIR="$2";              shift 2 ;;
        --partition)              PARTITION="$2";               shift 2 ;;
        --exclude-hosts|--exclude_hosts)
            EXCLUDE_HOSTS="${EXCLUDE_HOSTS:+$EXCLUDE_HOSTS,}$2"; shift 2 ;;
        --duration)               DURATION="$2";                shift 2 ;;
        --dataset_name|--dataset) DATASET="$2";                 shift 2 ;;
        --max-steps)              MAX_STEPS="$2";               shift 2 ;;
        --max-completion-length)  MAX_COMPLETION_LENGTH="$2";   shift 2 ;;
        --num-generations)        NUM_GENERATIONS="$2";         shift 2 ;;
        --grad-accum)             GRAD_ACCUM="$2"; GRAD_ACCUM_SET=1; shift 2 ;;
        --per-device-batch)       PER_DEVICE_BATCH="$2";        shift 2 ;;
        --learning-rate)          LEARNING_RATE="$2";           shift 2 ;;
        --w-overlap)              W_OVERLAP="$2";               shift 2 ;;
        --token-reduction)        TOKEN_REDUCTION="$2";         shift 2 ;;
        --lora-targets)           LORA_TARGETS="$2";            shift 2 ;;
        --overlap-heads)          OVERLAP_HEADS="$2";           shift 2 ;;
        --overlap-layer)          OVERLAP_LAYER="$2";           shift 2 ;;
        --overlap-metric)         OVERLAP_METRIC="$2";          shift 2 ;;
        --box-threshold)          BOX_THRESHOLD="$2";           shift 2 ;;
        --max-box-area)           MAX_BOX_AREA="$2";            shift 2 ;;
        --beta)                   BETA="$2";                    shift 2 ;;
        --vllm-gpus)              VLLM_GPUS_N="$2";             shift 2 ;;
        --vllm-gpu-mem)           VLLM_GPU_MEM="$2";            shift 2 ;;
        --preflight)              PREFLIGHT=true;               shift 1 ;;
        --no-preflight)           PREFLIGHT=false;              shift 1 ;;
        --sync-all-weights)       SYNC_LORA_ONLY=false;         shift 1 ;;
        --nvidia-api-key)         NVIDIA_API_KEY="$2";          shift 2 ;;
        --openai-api-key)         OPENAI_API_KEY="$2";          shift 2 ;;
        --wandb-api-key)          WANDB_API_KEY="$2";           shift 2 ;;
        --hf-token)               HF_TOKEN="$2";                shift 2 ;;
        --)                       shift; EXTRA_ARGS="$*";       break ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ---------- the GPU layout, and the batch arithmetic that follows from it ----------
# gen_batch = per_device x TRAIN_N x grad_accum, and the reference runs hold it at 48.
# Losing a training card to a two-card generation server is the ONLY thing here that
# changes TRAIN_N, so grad_accum moves with it -- 5 x 10 = 50 is not 48 and 48/5 is not an
# integer, which is exactly what Plan A' costs and why the banner prints the number.
DINO_GPU=0
if (( VLLM_GPUS_N == 2 )); then
    VLLM_GPUS="1,2"
    TRAIN_N=$(( NUM_GPUS - 3 ))
    TRAIN_GPUS=$(seq -s, 3 $(( NUM_GPUS - 1 )))
    [[ -z "${GRAD_ACCUM_SET:-}" ]] && GRAD_ACCUM=10
else
    VLLM_GPUS="1"
    TRAIN_N=$(( NUM_GPUS - 2 ))
    TRAIN_GPUS=$(seq -s, 2 $(( NUM_GPUS - 1 )))
fi
GEN_BATCH=$(( PER_DEVICE_BATCH * TRAIN_N * GRAD_ACCUM ))

if (( TRAIN_N < 1 )); then
    echo "ERROR: --num-gpus $NUM_GPUS leaves $TRAIN_N training processes." >&2
    exit 1
fi
if (( GEN_BATCH % NUM_GENERATIONS != 0 )); then
    echo "ERROR: gen_batch $GEN_BATCH is not a multiple of num_generations $NUM_GENERATIONS." >&2
    exit 1
fi

RUN_NAME="grpo-omni30b-overlap__wov${W_OVERLAP}_L${OVERLAP_LAYER}_h${OVERLAP_HEADS//,/-}_${OVERLAP_METRIC}"
[ -n "$OUTPUT_DIR" ] || OUTPUT_DIR="$REPO/outputs/omni_grpo_plan_a/$RUN_NAME"

# ---------- submit, or run here ----------
if [ "$DIRECT" != true ]; then
    PARTITION=${PARTITION:-$(SR1_JOB_HOURS=$DURATION sr1_pick_partition)}
    sr1_find_submit_job || { echo "ERROR: submit_job not found" >&2; exit 1; }
    ARGS=("--direct" "--num-gpus" "$NUM_GPUS" "--model" "$MODEL" "--output-dir" "$OUTPUT_DIR"
          "--dataset" "$DATASET" "--max-steps" "$MAX_STEPS" "--w-overlap" "$W_OVERLAP"
          "--overlap-layer" "$OVERLAP_LAYER" "--overlap-heads" "$OVERLAP_HEADS"
          "--overlap-metric" "$OVERLAP_METRIC" "--lora-targets" "$LORA_TARGETS"
          "--grad-accum" "$GRAD_ACCUM" "--vllm-gpus" "$VLLM_GPUS_N"
          "--vllm-gpu-mem" "$VLLM_GPU_MEM" "--beta" "$BETA")
    [ "$PREFLIGHT" = true ] || ARGS+=("--no-preflight")
    [ "$SYNC_LORA_ONLY" = true ] || ARGS+=("--sync-all-weights")
    echo "Submitting $RUN_NAME to $PARTITION for ${DURATION}h"
    exec submit_job --account "$ACCOUNT" --partition "$PARTITION" \
        --gpu 8 --nodes 1 --duration "$DURATION" --name "$RUN_NAME" \
        ${EXCLUDE_HOSTS:+--exclude_hosts "$EXCLUDE_HOSTS"} \
        --autoresume_uninstrumented \
        --command "bash $SCRIPT_PATH ${ARGS[*]}"
fi

# =========================================================================
# From here down we are ON the node.
# =========================================================================
mkdir -p "$OUTPUT_DIR"
LOG_DIR="$OUTPUT_DIR/sidecar_logs"
mkdir -p "$LOG_DIR"

export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export OVERLAP_STEPS_DEVICE OVERLAP_STEPS_CKPT
export WANDB_PROJECT=${WANDB_PROJECT:-saliency_r1}
export WANDB_RUN_ID=${WANDB_RUN_ID:-$RUN_NAME}
export WANDB_NAME=${WANDB_NAME:-$RUN_NAME}
export WANDB_RESUME=${WANDB_RESUME:-allow}
export WANDB_DATA_DIR=${WANDB_DATA_DIR:-/home/uberger/scratch/cache/wandb_data}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-/home/uberger/scratch/cache/wandb_cache}
mkdir -p "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR"
[ -n "${WANDB_API_KEY:-}" ] || export WANDB_MODE=offline
[ -n "${NVIDIA_API_KEY:-}" ] && export NVIDIA_API_KEY
[ -n "${OPENAI_API_KEY:-}" ] && export OPENAI_API_KEY
[ -n "${OPENAI_BASE_URL:-}" ] && export OPENAI_BASE_URL
[ -n "${JUDGE_MODEL:-}" ] && export JUDGE_MODEL
[ "$SYNC_LORA_ONLY" = true ] && export SR1_VLLM_SYNC_LORA_ONLY=1
# The vendored layernorm-only mamba_ssm. Without it the Nemotron decoder raises at IMPORT
# -- `MambaRMSNormGated.forward` IS a call to `rmsnorm_fn` -- and having no dist-info is
# deliberate: `is_mamba_2_ssm_available()` keeps reading False, so the fused SSM kernels
# stay off and this run uses the same torch-native Mamba path everything was measured on.
export PYTHONPATH="$REPO/vendor/mamba_ssm_min:${PYTHONPATH:-}"
# So a copy of nemotron_loader.py living inside trl_repo_nemotron can still find
# vendor/mamba_ssm_min: walking up from its own __file__ lands in the TRL clone.
export SR1_REPO="$REPO"
# 73.3 GB of an 80 GB card, and the one allocation that decides it is ~5.5 GB contiguous
# in the backward. The first attempt died with 5.17 GiB free and 5.50 GiB asked for -- not
# short of memory, short of one unfragmented block. Expandable segments let the allocator
# grow a segment instead of needing a new one that size, which is exactly this case.
# Exported before every child, so the preflight and all six training ranks get it.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

[ -d "$OVERLAP_STEPS_CKPT/encoder" ] || {
    echo "ERROR: steps-classifier ckpt not found at $OVERLAP_STEPS_CKPT" >&2; exit 1; }

echo "=============================================================="
echo "Run:              $RUN_NAME"
echo "Node:             $(hostname)"
echo "Model:            $MODEL"
echo "Harness:          $HARNESS"
echo "Envs:             train=$TRAIN_ENV  generate=$VLLM_ENV"
echo "GPUs (total $NUM_GPUS):  DINO=cuda:$DINO_GPU  vLLM=cuda:[$VLLM_GPUS]  train=cuda:[$TRAIN_GPUS] ($TRAIN_N procs)"
echo "Plan:             $([ "$VLLM_GPUS_N" = 2 ] && echo "A' (server on two cards, five trainers)" || echo 'A (one whole copy per training card)')"
echo "Batch:            per_device=$PER_DEVICE_BATCH num_generations=$NUM_GENERATIONS grad_accum=$GRAD_ACCUM  (gen_batch=$GEN_BATCH)"
echo "Reward:           overlap $OVERLAP_METRIC w=$W_OVERLAP  layer=$OVERLAP_LAYER heads=$OVERLAP_HEADS tr=$TOKEN_REDUCTION"
echo "LoRA:             r=16 alpha=32 targets=$LORA_TARGETS (scoped to the decoder by vlm_family)"
echo "Steps:            max_steps=$MAX_STEPS save_steps=$SAVE_STEPS"
echo "Output:           $OUTPUT_DIR"
echo "=============================================================="
# memory.used as well as total: a previous job's vLLM worker can outlive its allocation
# for a few seconds, and starting six 65 GB ranks beside one is an OOM with no other
# symptom.
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv || true

# ---------- 0. the preflight ----------
# Before any sidecar takes a card, so it has a whole one to itself. See THE PREFLIGHT.
if [ "$PREFLIGHT" = true ]; then
    echo "[preflight] LoRA landing + gradient check on cuda:$DINO_GPU (~4 min) ..."
    (
        # shellcheck disable=SC1090
        source "$CONDA_SH"; conda activate "$TRAIN_ENV"
        # CUDA_HOME: see the training block below for why a run that compiles nothing
        # still needs one.
        source "$REPO/setup_cuda_home.sh" >/dev/null
        cd "$REPO"
        CUDA_VISIBLE_DEVICES=$DINO_GPU python omni_train_step_bench.py \
            --bits 16 --grad-ckpt 1 --grads-only 1 --completion "$MAX_COMPLETION_LENGTH"
    ) 2>&1 | tee "$LOG_DIR/preflight.log"
    grep -q "PREFLIGHT PASS" "$LOG_DIR/preflight.log" || {
        echo "[preflight] FAILED -- not starting a run that would train nothing." >&2
        exit 1; }
    echo "[preflight] passed."
fi

# ---------- cleanup ----------
DINO_PID=""; VLLM_PID=""; CLEANUP_PID=""
cleanup() {
    echo "[cleanup] shutting down sidecars ..."
    for pid in "$VLLM_PID" "$DINO_PID" "$CLEANUP_PID"; do
        [ -n "$pid" ] || continue
        pkill -TERM -P "$pid" 2>/dev/null || true
        kill -TERM "$pid" 2>/dev/null || true
    done
    # vLLM spawns a detached EngineCore worker that holds GPU memory and does not match
    # the serve cmdline -- kill it explicitly or it orphans the card.
    pkill -TERM -u "$USER" -f "trl.scripts.vllm_serve" 2>/dev/null || true
    pkill -TERM -u "$USER" -f "VLLM::EngineCore" 2>/dev/null || true
    sleep 2
    pkill -KILL -u "$USER" -f "VLLM::EngineCore" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait_for_health() {
    local url="$1" name="$2" timeout_s="$3" pid="$4" waited=0
    echo "[health] waiting for $name at $url (timeout ${timeout_s}s) ..."
    until curl -sf "$url" >/dev/null 2>&1; do
        if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
            echo "[health] ERROR: $name (pid $pid) died before becoming healthy." >&2
            tail -60 "$LOG_DIR/${name}.log" >&2 2>/dev/null || true
            exit 1
        fi
        if grep -qE "Engine core initialization failed|EngineCore failed to start" "$LOG_DIR/${name}.log" 2>/dev/null; then
            echo "[health] ERROR: $name logged a fatal error; aborting." >&2
            tail -60 "$LOG_DIR/${name}.log" >&2 2>/dev/null || true
            exit 1
        fi
        sleep 5; waited=$(( waited + 5 ))
        if (( waited >= timeout_s )); then
            echo "[health] ERROR: $name not healthy after ${timeout_s}s." >&2
            tail -60 "$LOG_DIR/${name}.log" >&2 2>/dev/null || true
            exit 1
        fi
    done
    echo "[health] $name is up (after ${waited}s)."
}

VLLM_EAGER_FLAG=""
case "$VLLM_ENFORCE_EAGER" in
    True|true|1) VLLM_EAGER_FLAG="--enforce_eager True" ;;
esac

# ---------- 1. Grounding-DINO on GPU 0 ----------
echo "[start] Grounding-DINO on cuda:$DINO_GPU -> 127.0.0.1:$DINO_PORT"
(
    # shellcheck disable=SC1090
    source "$CONDA_SH"; conda activate "$TRAIN_ENV"
    CUDA_VISIBLE_DEVICES=$DINO_GPU DINO_SERVER_BATCH=${DINO_SERVER_BATCH:-8} \
        exec python "$REPO/serve_grounding_dino.py" --host 127.0.0.1 --port "$DINO_PORT"
) > "$LOG_DIR/dino.log" 2>&1 &
DINO_PID=$!

# ---------- 2. vLLM generation server on GPU 1 (or 1-2) ----------
# In `nemotron_vllm`, and from trl_repo_nemotron, whose trl/scripts/vllm_serve.py is the
# version-tolerant copy. Tensor-parallel 2 is what Plan A' means.
echo "[start] vLLM server on cuda:[$VLLM_GPUS] -> 127.0.0.1:$VLLM_PORT ($VLLM_ENV)"
(
    # shellcheck disable=SC1090
    source "$CONDA_SH"; conda activate "$VLLM_ENV"
    cd "$HARNESS"
    # TWO THINGS THAT ARE NOT OPTIONAL ON THIS CLUSTER, both found by omni_vllm_probe.py.
    #
    # VLLM_ENABLE_V1_MULTIPROCESSING=0 runs the engine in the worker process instead of
    # spawning an EngineCore child. That child HANGS here: it gets as far as the worker's
    # memory snapshot and then sits in a futex with 43 sleeping threads, while the parent
    # prints "Waiting for 1 local core engine proc(s) to start" forever. In-process it
    # loads normally. (ptrace is off on these nodes, so py-spy and gdb cannot see into the
    # child at all -- `omni_vllm_probe.py --watchdog` exists because of that.)
    #
    # The triton MoE backend, because the default `auto` picks FlashInfer's CUTLASS path,
    # which JIT-compiles on first use and dies in `get_cuda_path()`: there is no
    # /usr/local/cuda on these nodes and the only system toolkit is CUDA 12.4 against a
    # torch built on 13. Triton ships its own compiler and needs none.
    export VLLM_ENABLE_V1_MULTIPROCESSING=0
    # And kernel warmup calls DeepGEMM's FP8 path on a bfloat16 model, which
    # raises "DeepGEMM backend is not available or outdated".
    export VLLM_USE_DEEP_GEMM=0
    CUDA_VISIBLE_DEVICES=$VLLM_GPUS \
        exec python -m trl.scripts.vllm_serve \
            --model "$MODEL" \
            --host 127.0.0.1 --port "$VLLM_PORT" \
            --tensor_parallel_size "$VLLM_GPUS_N" \
            --gpu_memory_utilization "$VLLM_GPU_MEM" \
            --dtype bfloat16 \
            --max_model_len "$VLLM_MAX_MODEL_LEN" \
            --enable_prefix_caching True \
            --max_num_seqs "$VLLM_MAX_NUM_SEQS" \
            --kernel_config '{"moe_backend": "triton", "enable_flashinfer_autotune": false}' \
            $VLLM_EAGER_FLAG \
            --trust_remote_code True
) > "$LOG_DIR/vllm.log" 2>&1 &
VLLM_PID=$!

wait_for_health "http://127.0.0.1:$DINO_PORT/health"  "dino" 900  "$DINO_PID"
# 62 GB of weights plus a compile: allow half an hour.
wait_for_health "http://127.0.0.1:$VLLM_PORT/health/" "vllm" 2400 "$VLLM_PID"

# ---------- 3. checkpoint housekeeping ----------
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
                    rm -rf "$output_dir/checkpoint-$step"
                fi
            done
        fi
    done
}
_cleanup_checkpoints "$OUTPUT_DIR" &
CLEANUP_PID=$!

RESUME_FLAG=""
LATEST_CKPT=$(ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | sed 's|.*/checkpoint-||' | sort -n | tail -1 || true)
[ -n "$LATEST_CKPT" ] && RESUME_FLAG="--resume_from_checkpoint $OUTPUT_DIR/checkpoint-$LATEST_CKPT"

MASTER_PORT=${MASTER_PORT:-$(shuf -i 29500-65000 -n 1)}
BETA_FLAG=""
[[ "$BETA" != "0" && "$BETA" != "0.0" ]] && BETA_FLAG="--beta $BETA"

# No benchmark eval runs during training. Record what the run started from so a later
# pass can score it as step 0; everything else is watch_bench_evals.sh's job afterwards.
mkdir -p "$OUTPUT_DIR/bench_eval"
echo "$MODEL" > "$OUTPUT_DIR/bench_eval/base_model.txt"

# ---------- 4. GRPO training on GPUs 2..N-1 ----------
echo "[start] training on cuda:[$TRAIN_GPUS] ($TRAIN_N procs, $TRAIN_ENV)"
# shellcheck disable=SC1090
source "$CONDA_SH"; conda activate "$TRAIN_ENV"
[ -n "${CONDA_PREFIX:-}" ] || { echo "ERROR: conda activate $TRAIN_ENV failed." >&2; exit 1; }
export PATH="$CONDA_PREFIX/bin:$PATH"
hash -r
# CUDA_HOME, and it is not optional even though nothing here compiles a kernel.
# `accelerate.utils.is_peft_model` -> `extract_model_from_parallel` does
# `from deepspeed import DeepSpeedEngine` whenever deepspeed is INSTALLED, and importing
# deepspeed probes every op builder, which needs a toolkit. So a run that uses no
# deepspeed at all -- this one: multi_gpu.yaml, no ZeRO -- still dies at the peft check
# with "CUDA_HOME does not exist, unable to compile CUDA op(s)". setup_cuda_home.sh is the
# repo's resolver, and it also re-asserts the active env's bin at the front of PATH
# because the toolkit it picks may itself be a conda env with its own python.
#
# Deliberately NOT sourced in the vLLM subshell above: that environment is a CUDA 13
# build, and putting a 12.4 toolkit's lib64 ahead of its own libraries is a way to break
# a server that currently works.
source "$REPO/setup_cuda_home.sh"
if [ "$(command -v python)" != "$CONDA_PREFIX/bin/python" ]; then
    echo "ERROR: CUDA_HOME='$CUDA_HOME' shadowed the active env's python." >&2
    exit 1
fi
bash "$REPO/check_cuda_home.sh" || exit 1

# BOTH ENDS OF THE WEIGHT-SYNC COMMUNICATOR MUST BE THE SAME NCCL.
# The trainer's torch 2.8 ships NCCL 2.27.3 and the server's torch 2.11 ships 2.28.9, and
# `ncclCommInitRank` rejects the pairing outright -- "NCCL error: invalid usage", raised
# on both sides at once, because the bootstrap protocol changed between the two. That is
# the price of the two-environment split (see TWO ENVS at the top), and this is the whole
# of the fix: vLLM's pynccl is a CTYPES wrapper that honours VLLM_NCCL_SO_PATH, and the
# libraries are self-contained -- they link nothing but libc and dlopen the driver -- so
# loading the server's copy here disturbs nothing. In particular it does NOT touch torch's
# own NCCL, which the six ranks use for DDP and which never talks to the server.
_SERVER_NCCL=$(ls /home/uberger/scratch/miniconda3/envs/"$VLLM_ENV"/lib/python*/site-packages/nvidia/nccl/lib/libnccl.so.2 2>/dev/null | head -1)
if [ -n "$_SERVER_NCCL" ]; then
    export VLLM_NCCL_SO_PATH=${VLLM_NCCL_SO_PATH:-$_SERVER_NCCL}
    echo "[nccl] weight sync will use $VLLM_NCCL_SO_PATH (the SERVER's copy, on both ends)"
else
    echo "[nccl] WARNING: no libnccl under $VLLM_ENV; the weight-sync communicator will" >&2
    echo "[nccl]          try to pair NCCL 2.27 with 2.28 and fail in init_communicator." >&2
fi

cd "$HARNESS"
CUDA_VISIBLE_DEVICES=$TRAIN_GPUS accelerate launch \
    --config_file "$REPO/accelerate_omni_plan_a.yaml" \
    --num_processes "$TRAIN_N" \
    --main_process_port "$MASTER_PORT" \
    examples/scripts/grpo_vlm_qwen3.py \
    --model_name_or_path "$MODEL" \
    --dataset_name "$DATASET" \
    --attn_implementation eager \
    --output_dir "$OUTPUT_DIR" \
    --learning_rate "$LEARNING_RATE" \
    --torch_dtype bfloat16 \
    --max_prompt_length 2048 \
    --max_completion_length "$MAX_COMPLETION_LENGTH" \
    --max_steps "$MAX_STEPS" \
    --gradient_checkpointing \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --reforward_saliency True \
    --reward_variant ours \
    --overlap_metric "$OVERLAP_METRIC" \
    --overlap_layer "$OVERLAP_LAYER" \
    --overlap_heads "$OVERLAP_HEADS" \
    --token_reduction "$TOKEN_REDUCTION" \
    --box_threshold "$BOX_THRESHOLD" \
    --max_box_area "$MAX_BOX_AREA" \
    $BETA_FLAG \
    --reward_weights 1.0 "$W_OVERLAP" 1.0 1.0 \
    --use_vllm \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port "$VLLM_PORT" \
    --use_peft \
    --lora_r 16 --lora_alpha 32 \
    --lora_target_modules ${LORA_TARGETS//,/ } \
    --log_completions \
    --per_device_train_batch_size "$PER_DEVICE_BATCH" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --num_generations "$NUM_GENERATIONS" \
    --report_to wandb \
    --logging_steps 1 \
    --save_steps "$SAVE_STEPS" \
    --temperature 1 \
    --ddp_find_unused_parameters False \
    --val_sets_dir "" \
    $RESUME_FLAG \
    $EXTRA_ARGS

echo "Finished $RUN_NAME"
echo "Score its checkpoints afterwards (holds no GPU itself):"
echo "  bash $REPO/watch_bench_evals.sh --run-dir $OUTPUT_DIR"
