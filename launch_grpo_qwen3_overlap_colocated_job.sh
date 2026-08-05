#!/bin/bash
# GRPO training for Qwen3-VL-8B with attention-overlap reward (colocated).
# Co-locates DINO + vLLM sidecars on the same node as training:
#     GPU 0         Grounding-DINO reward server   (127.0.0.1:$DINO_PORT)
#     GPU 1         vLLM generation server         (127.0.0.1:$VLLM_PORT)
#     GPU 2..N-1    GRPO training, DeepSpeed ZeRO-3  (N-2 processes)
#
# --share-sidecar-gpu puts DINO and vLLM together on GPU 0, giving N-1 training processes:
#     GPU 0         Grounding-DINO + vLLM  (vllm_gpu_mem default drops 0.90 -> 0.85)
#     GPU 1..N-1    GRPO training                    (N-1 processes)
# OFF by default. On a 4-GPU node this is 3 training procs instead of 2 (+50%). Justified by
# job 5627435 on GB200: DINO peaked at 1.6 GB of 185 GB (1%), so a dedicated GPU is waste.
#
# vLLM replaces the slow HF generate() path -- this is the main speedup over the
# non-colocated launcher. The policy is LoRA; the trainer merges the adapter and
# pushes weights to the vLLM server over NCCL every step.
#
# Default: submits to the cluster via submit_job (SLURM).
# --direct: runs on the current node immediately (no SLURM), e.g. on an interactive GPU node.
#
# Usage:
#   WANDB_API_KEY=... NVIDIA_API_KEY=... bash launch_grpo_qwen3_overlap_colocated_job.sh [OPTIONS]
#   bash launch_grpo_qwen3_overlap_colocated_job.sh --direct --num-gpus 8
#
# THE HACK-RESISTANT REWARD -- one flag, nothing else to set:
#
#   bash launch_grpo_qwen3_overlap_colocated_job.sh --overlap-metric auroc
#
# Identical semantics to the non-colocated launcher (only the generation path differs),
# so the same defaults and the same evidence apply. Why: mean_in divides by the map's own
# PEAK, so a map that merely FLATTENS scores higher without attending the box any better
# -- 32x more movement under flattening than under real grounding, and that is what the
# wov0.2/wov0.4 runs did (their MMStar gain vanishes once chance-corrected). auroc depends
# only on patch ORDER, so that route is closed by construction. Derivation + evidence:
# vlm_reasoning/wiki/hack-resistant-overlap-reward-plan.md
#
# --overlap-metric auroc also switches two coupled defaults, because neither transfers
# from mean_in. Both are still overridable, and both appear in the run name, so a name
# always states what actually ran.
#
#   mass_floor_tau  ->  0.0022   Not optional: auroc is rank-based and so blind to a
#                                model withdrawing attention from the image. 0.0022 =
#                                p10 of the reference model's image_mass.
#   w_overlap       ->  0.11     GRPO normalises advantages within the group, so what
#                                matters is the reward term's SPREAD, and the auroc
#                                composite has ~3.6x the per-sample sd of mean_in
#                                (0.13 vs 0.036). 0.11 reproduces the pressure of
#                                mean_in's wov0.4. Reusing 0.2/0.4 here would apply
#                                ~3.6x the intended pressure.
#
# --overlap-metric mean_in_v2 is the third option: the same mean over the box, divided
# by the mean over the WHOLE map instead of by its peak. Chance is 1.0, and it is
# unbounded above only in principle -- measured over 1074 grounded steps of the
# cold-start policy on set_a it runs p10 0.41 / median 0.74 / p99 1.36 / max 2.33,
# because the median DINO box union already covers 56% of the image and the ceiling
# (n_patches/n_in) is ~1.8 there. No clamp needed.
#
# It carries ONE coupled default, and unlike auroc's it was measured here rather than
# taken from the offline screen (overlap_metric_spread.py on a 40-sample probe):
#
#   w_overlap       ->  0.033    Its per-sample sd is 0.105 vs mean_in's 0.0086, i.e.
#                                12x the spread, so 0.4 x 0.0086/0.105 ~ 0.033
#                                reproduces the pressure of mean_in's wov0.4. Reusing
#                                0.2/0.4 would apply ~6-12x the intended pressure. The
#                                same script re-derives auroc's weight as 0.089 against
#                                the documented 0.11, so treat 0.033 as +-25%.
#
# The mass floor stays OFF: a ratio of two means is blind to the model withdrawing
# attention from the image (the hole auroc's floor closes), but auroc's tau=0.0022 does
# not fit this corpus -- p10 of image_mass here is 0.00078 and 0.0022 bites on 30% of
# steps, well past the p25 the reward docstring warns about. If you enable it anyway,
# the floor RAISES the spread (0.105 -> 0.143), so pass --w-overlap 0.024 with it.
#
# Two caveats it does have and mean_in does not: it is more coupled to DINO box size
# (r +0.38 vs mean_in's +0.17), though that pull dies at 1.0 rather than diverging, and
# it has no offline attack/utility screen behind it.
#
# LORA TARGETS -- k_proj is now on by default:
#
#   bash launch_grpo_qwen3_overlap_colocated_job.sh --lora-targets q_proj,v_proj   # old default
#
# This reward pays for WHERE attention lands, and with k_proj frozen the image-token key
# directions are frozen with it: an attention logit here is cos(q, k_j) (Qwen3-VL RMS-
# normalises q and k per head, so a LoRA can only rotate them, never rescale), and two
# patches whose keys point the same way cannot be pulled apart by any query update.
# Adapting k_proj is what opens that route. Cheap under GQA -- 32 query heads over 8 KV
# heads makes k_proj 4096x1024 against q_proj's 4096x4096, so +38% adapter params over
# q+v -- and blunt in one way worth knowing: reward heads 28 and 31 share KV group 7, so
# a key update moves heads 28-31 together. The old q_proj,v_proj default is the LoRA
# paper's, chosen for downstream language quality at a fixed budget, not for attention
# placement. Changing this invalidates --resume from an existing adapter checkpoint.
#
# MIXED CORPORA -- restrict the overlap reward to photographs:
#
#   bash launch_grpo_qwen3_overlap_colocated_job.sh --dataset_name cold_data/grpo_sets/set_b --natural-only
#
# set_b is 80% natural + 20% charts/documents/diagrams. Grounding-DINO grounds the
# observe-step phrase on the image, and it is a photograph detector, so on the non-natural
# fifth its boxes -- and hence the whole overlap score -- are noise. --natural-only masks
# the overlap term on those rows (their `natural` column is False); they keep format +
# accuracy + judge. Adds _natonly to the run name. Off by default.
#
# WATCHING A RUN INSTEAD OF ONLY AUTOPSYING IT
#
# Two independent monitors, both landing in the same WandB run:
#
#   validation   At step 0 and every --eval-steps (100) steps, val_natural and
#                val_nonnatural are scored on ANSWER ACCURACY ONLY -- 256 held-out
#                rows each, whose images appear in neither set_a nor set_b -- and
#                logged as val/<set>/accuracy. Build them once with
#                `python build_grpo_sets.py --build-val`; --no-eval turns it off.
#
#                One greedy completion per prompt in a single batched vLLM call: no
#                DINO, no saliency re-forward, no judge, no log-prob pass, and the
#                policy never runs a forward. Routing this through the Trainer's
#                evaluation loop instead re-runs the entire reward pipeline and
#                measured 21.6 min per set -- ~90% of training throughput at a
#                100-step cadence, which is what made the cheap path necessary.
#
#   benchmarks   13 of the test benchmarks -- all but the three that need an LLM
#                judge to score (see eval_mini/benchmarks.py) -- cut to 100 docs
#                and split into a natural and a non-natural suite, run on every kept
#                checkpoint by a separate 4-GPU job. watch_bench_evals.sh starts
#                with the run, holds no GPUs itself, and submits a job only when a
#                checkpoint is waiting and none is already in flight. It submits
#                with sbatch here on the compute node, because the submit_job
#                wrapper resolves the cluster from $HOSTNAME and has no pool1-*
#                entry. --no-auto-bench turns it off; to run it by hand later:
#
#                  bash watch_bench_evals.sh --run-dir <output-dir>
#
#                Results reach WandB under bench/* within one logging interval;
#                anything finishing after training exits is appended with
#                `python bench_eval.py --backfill --run-dir DIR --wandb-run-id ID`.
#
# Environment overrides:
#   PARTITION=batch_singlenode   DURATION=4 (hours)
#   NATURAL_ONLY=true            (same as --natural-only; --no-natural-only to force off)
#   SAVE_STEPS=10   CKPT_KEEP_EVERY=100
#   EVAL_STEPS=100   VAL_SETS_DIR=<dir>   AUTO_BENCH=true   BENCH_GPUS=4
#   DINO_PORT=8100   VLLM_PORT=8000   VLLM_MAX_MODEL_LEN=4096
#   VLLM_GPU_MEM     (default 0.90, or 0.85 with --share-sidecar-gpu)
#   SHARE_SIDECAR_GPU=true   (same as --share-sidecar-gpu; --no-share-sidecar-gpu to force off)
#   VLLM_ENFORCE_EAGER=False
#   OVERLAP_STEPS_DEVICE=cpu   OVERLAP_STEPS_CKPT=<path>
#   NVIDIA_API_KEY / OPENAI_API_KEY / OPENAI_BASE_URL / JUDGE_MODEL
#   WANDB_API_KEY   (omit -> offline)   HF_TOKEN
set -euo pipefail

SCRIPT_PATH="$(realpath "$0")"
REPO=/home/uberger/scratch/research/saliency_r1
CONDA_SH=/home/uberger/scratch/miniconda3/etc/profile.d/conda.sh
# This colocated job runs both the vLLM server and the trainer in ONE env, so it
# MUST use the vllm-enabled env. Hardcoded (not overridable) to prevent picking up
# a stray CONDA_ENV from the shell, which silently breaks the vLLM sidecar.
CONDA_ENV=saliency_r1_qwen3_vllm
HF_HOME=${HF_HOME:-/home/uberger/scratch/cache/hf_cache}

# ---------- SLURM defaults ----------
source "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/cluster_env.sh"
ACCOUNT=nvr_israel_rlop
PARTITION=${PARTITION:-$(sr1_pick_partition batch_singlenode batch_long batch)}
DURATION=${DURATION:-4}

# ---------- training defaults ----------
MODEL="$REPO/checkpoint/coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged"
NUM_GPUS=8
OUTPUT_DIR=""
MAX_COMPLETION_LENGTH=1024
NUM_GENERATIONS=8
GRAD_ACCUM=8
PER_DEVICE_BATCH=1
LEARNING_RATE=1e-5
# LoRA targets, comma-separated. k_proj is in the default because this run rewards
# WHERE attention lands: Qwen3-VL RMS-normalises q and k per head (q_norm/k_norm in
# modeling_qwen3_vl.py), so a LoRA can only ROTATE key/query directions, never scale
# them, and with k_proj frozen the image-token key directions are fixed -- two patches
# whose keys point the same way cannot be separated by any query update, which is
# exactly the move the overlap reward is asking for. It is cheap under GQA (32 query
# heads over 8 KV heads, so k_proj is 4096x1024 against q_proj's 4096x4096: +38%
# adapter params over q+v). It is also blunt: the reward heads 28 and 31 share KV
# group 7, so a key update moves heads 28-31 together.
# The upstream q_proj,v_proj default comes from the LoRA paper's ablation, which tuned
# downstream language quality at a fixed budget, not attention placement.
# NOTE: changing this invalidates --resume from an existing adapter checkpoint (the
# adapter shapes no longer match). It needs a fresh run.
LORA_TARGETS=${LORA_TARGETS:-q_proj,k_proj,v_proj}
SAVE_STEPS=${SAVE_STEPS:-10}
# Checkpoints are cheap -- 116 MB of LoRA adapter plus ZeRO state, not a full model
# -- and every kept one is a point on the benchmark curve, so keep one per 100
# steps rather than per 500. A 3,000-step run costs ~3.5 GB.
CKPT_KEEP_EVERY=${CKPT_KEEP_EVERY:-100}
# Held-out validation: score val_natural and val_nonnatural every EVAL_STEPS steps.
# Their images are disjoint from set_a and set_b, so this measures generalization
# rather than memorization. Set VAL_SETS_DIR="" to turn validation off entirely.
EVAL_STEPS=${EVAL_STEPS:-100}
VAL_SETS_DIR=${VAL_SETS_DIR:-$REPO/cold_data/grpo_sets}
# The benchmark-eval dispatcher (watch_bench_evals.sh) starts with the run and
# submits a $BENCH_GPUS-GPU job whenever a kept checkpoint is waiting to be scored.
# It holds no GPU itself. --no-auto-bench turns it off.
AUTO_BENCH=${AUTO_BENCH:-true}
BENCH_GPUS=${BENCH_GPUS:-4}
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
MAX_BOX_AREA=0.5         # per-BOX area cap. Pass 0 to disable it and keep every box
                         # above --box-threshold.
MAX_UNION_AREA=""        # unset -> off. Per-STEP cap on the union of the kept boxes:
                         # a step whose union covers more than this fraction of the
                         # image is skipped (not scored 0), like an ungroundable one.
                         # --max-box-area does not bound the union -- N disjoint boxes
                         # each under the per-box cap can cover the whole image, and
                         # the measured median union is already 56% of it.
OVERLAP_METRIC=mean_in   # mean_in (incumbent default) | mean_in_v2 (/mean not /max; see
                         # below) | auroc (hack-resistant; see above)
MASS_FLOOR_TAU=""        # unset -> off for mean_in, 0.0022 for auroc (see below).
                         # Pass 0 to force it off explicitly.
# --natural-only: score the overlap reward only on rows with natural=True, leaving
# charts/documents/diagrams to format + accuracy + judge. Grounding-DINO is a
# photograph detector, so its boxes on non-natural imagery are noise, and a noisy
# overlap term is worse than none. Only meaningful on a mixed corpus with a `natural`
# column (cold_data/grpo_sets/set_b); OFF by default so existing runs are unchanged.
NATURAL_ONLY=${NATURAL_ONLY:-false}

# ---------- sidecar defaults ----------
DINO_PORT=${DINO_PORT:-8100}
VLLM_PORT=${VLLM_PORT:-8000}
# VLLM_GPU_MEM default is resolved AFTER arg parsing -- it depends on whether the two
# sidecars share a GPU (see SHARE_SIDECAR_GPU below). Leaving it empty here preserves
# both override paths: the VLLM_GPU_MEM env var and the --vllm-gpu-mem flag.
VLLM_GPU_MEM=${VLLM_GPU_MEM:-}
# Put Grounding-DINO on the SAME GPU as vLLM, freeing one GPU for training. Measured on
# GB200 (job 5627435): DINO peaked at 1.6 GB of 185 GB (1%), while vLLM's 89% was simply
# gpu_memory_utilization=0.90 taking what it is allowed. Dedicating a whole GPU to DINO is
# therefore mostly waste -- on a 4-GPU node this turns 2 training procs into 3 (+50%).
# OFF by default: it changes GPU placement, so opt in explicitly.
SHARE_SIDECAR_GPU=${SHARE_SIDECAR_GPU:-false}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-4096}
VLLM_ENFORCE_EAGER=${VLLM_ENFORCE_EAGER:-False}
OVERLAP_STEPS_DEVICE=${OVERLAP_STEPS_DEVICE:-cuda}   # T5 step-classifier on the training GPU (CPU was the dominant per-step cost)
OVERLAP_STEPS_CKPT=${OVERLAP_STEPS_CKPT:-$REPO/checkpoint/steps_classifier/best}

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
        --num-generations)        NUM_GENERATIONS="$2";         shift 2 ;;
        --grad-accum)             GRAD_ACCUM="$2";              shift 2 ;;
        --per-device-batch)       PER_DEVICE_BATCH="$2";        shift 2 ;;
        --learning-rate)          LEARNING_RATE="$2";           shift 2 ;;
        --w-overlap)              W_OVERLAP="$2"; W_OVERLAP_SET=1; shift 2 ;;
        --token-reduction)        TOKEN_REDUCTION="$2";         shift 2 ;;
        --lora-targets)           LORA_TARGETS="$2";            shift 2 ;;
        --overlap-heads)          OVERLAP_HEADS="$2";           shift 2 ;;
        --overlap-layer)          OVERLAP_LAYER="$2";           shift 2 ;;
        --box-threshold)          BOX_THRESHOLD="$2";           shift 2 ;;
        --max-box-area)           MAX_BOX_AREA="$2";            shift 2 ;;
        --max-union-area)         MAX_UNION_AREA="$2";          shift 2 ;;
        --overlap-metric)         OVERLAP_METRIC="$2";          shift 2 ;;
        --mass-floor-tau)         MASS_FLOOR_TAU="$2";          shift 2 ;;
        --natural-only)           NATURAL_ONLY=true;            shift ;;
        --no-natural-only)        NATURAL_ONLY=false;           shift ;;
        --eval-steps)             EVAL_STEPS="$2";              shift 2 ;;
        --no-eval)                VAL_SETS_DIR="";              shift ;;
        --val-sets-dir)           VAL_SETS_DIR="$2";            shift 2 ;;
        --auto-bench)             AUTO_BENCH=true;              shift ;;
        --no-auto-bench)          AUTO_BENCH=false;             shift ;;
        --bench-gpus)             BENCH_GPUS="$2";              shift 2 ;;
        --dino-port)              DINO_PORT="$2";               shift 2 ;;
        --vllm-port)              VLLM_PORT="$2";               shift 2 ;;
        --vllm-gpu-mem)           VLLM_GPU_MEM="$2";            shift 2 ;;
        --share-sidecar-gpu)      SHARE_SIDECAR_GPU=true;       shift ;;
        --no-share-sidecar-gpu)   SHARE_SIDECAR_GPU=false;      shift ;;
        --vllm-max-model-len)     VLLM_MAX_MODEL_LEN="$2";      shift 2 ;;
        --vllm-enforce-eager)     VLLM_ENFORCE_EAGER="$2";      shift 2 ;;
        # The training command runs from $REPO/trl_repo, so a relative dataset path
        # given on the command line would be resolved against the wrong directory.
        # Absolutize it here, while we are still in the invocation cwd; anything
        # that is not an existing local path (a Hub id, say) is passed through.
        --dataset_name|--dataset-name)
            _ds="$2"
            [[ -e "$_ds" ]] && _ds="$(cd "$(dirname "$_ds")" && pwd)/$(basename "$_ds")"
            EXTRA_ARGS="$EXTRA_ARGS --dataset_name $_ds"; shift 2 ;;
        *)                        EXTRA_ARGS="$EXTRA_ARGS $1";  shift ;;
    esac
done

if [ "$SHARE_SIDECAR_GPU" = true ]; then MIN_GPUS=2; else MIN_GPUS=3; fi
if (( NUM_GPUS < MIN_GPUS )); then
    if [ "$SHARE_SIDECAR_GPU" = true ]; then
        echo "ERROR: need >=2 GPUs with --share-sidecar-gpu (1 shared DINO+vLLM + >=1 training); got --num-gpus $NUM_GPUS" >&2
    else
        echo "ERROR: need >=3 GPUs (1 DINO + 1 vLLM + >=1 training); got --num-gpus $NUM_GPUS" >&2
    fi
    exit 1
fi
if (( NUM_GPUS > 8 )); then
    echo "ERROR: single-node launcher (max 8 GPUs). localhost DINO/vLLM sidecars cannot serve a second node." >&2
    exit 1
fi

# Sidecar placement. Shared: both servers on GPU 0, training on 1..N-1. Separate (default):
# DINO on 0, vLLM on 1, training on 2..N-1.
DINO_GPU=0
if [ "$SHARE_SIDECAR_GPU" = true ]; then
    VLLM_GPU=0
    TRAIN_N=$(( NUM_GPUS - 1 ))
    TRAIN_GPUS=$(seq -s, 1 $(( NUM_GPUS - 1 )))
else
    VLLM_GPU=1
    TRAIN_N=$(( NUM_GPUS - 2 ))
    TRAIN_GPUS=$(seq -s, 2 $(( NUM_GPUS - 1 )))
fi

# vLLM sizes its KV cache as a fraction of TOTAL GPU memory, but DINO starts first and is
# already resident, so 0.90 can over-subscribe a shared GPU. 0.85 of 185 GB still leaves
# ~28 GB -- ample for DINO's measured 1.6 GB plus fragmentation. An explicit
# --vllm-gpu-mem / VLLM_GPU_MEM always wins over both defaults.
if [ -z "$VLLM_GPU_MEM" ]; then
    if [ "$SHARE_SIDECAR_GPU" = true ]; then VLLM_GPU_MEM=0.85; else VLLM_GPU_MEM=0.90; fi
fi

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
# mean_in_v2 carries only the weight -- 0.033, measured (see the header). No mass floor
# by default: the tau that fits this corpus is not auroc's 0.0022 (p10 of image_mass on
# set_a is 0.00078, and 0.0022 bites on 30% of steps here), and 0.033 was measured with
# the floor OFF. If you pass --mass-floor-tau anyway, drop the weight to ~0.024.
if [[ "$OVERLAP_METRIC" == "mean_in_v2" ]]; then
    [[ -z "${W_OVERLAP_SET:-}" ]] && W_OVERLAP=0.033
fi

# ---------- naming: every swept HP appears in the model AND wandb name ----------
N_HEADS=$(echo "$OVERLAP_HEADS" | awk -F, '{print NF}')
SUFFIX="__wov${W_OVERLAP}_${N_HEADS}head_tr${TOKEN_REDUCTION}"
# Only non-default metric settings extend the suffix, so existing mean_in run names
# (and the checkpoints already on disk) stay exactly as they are.
[[ "$OVERLAP_METRIC" != "mean_in" ]] && SUFFIX="${SUFFIX}_${OVERLAP_METRIC}"
[[ -n "$MASS_FLOOR_TAU" ]] && SUFFIX="${SUFFIX}_mf${MASS_FLOOR_TAU}"
[[ -n "$MAX_UNION_AREA" ]] && SUFFIX="${SUFFIX}_mu${MAX_UNION_AREA}"
[[ "$MAX_BOX_AREA" == "0" ]] && SUFFIX="${SUFFIX}_nobox"
# The reward differs from a plain run, so the checkpoints and the wandb run must not
# share a name with one.
[[ "$NATURAL_ONLY" == true ]] && SUFFIX="${SUFFIX}_natonly"
# A different adapter is a different experiment, so it must not share a name (or a
# checkpoint dir) with a q+v run. q_proj,v_proj is the historical set and stays unmarked,
# which keeps every existing run name and every checkpoint already on disk untouched.
LORA_SLUG=$(echo "$LORA_TARGETS" | sed 's/_proj//g; s/,//g')
[[ "$LORA_TARGETS" != "q_proj,v_proj" ]] && SUFFIX="${SUFFIX}_lora${LORA_SLUG}"
MODEL_SLUG=$(echo "$MODEL" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9_]/_/g')
RUN_NAME="grpo-${MODEL_SLUG}-overlap${SUFFIX}"
[[ -z "$OUTPUT_DIR" ]] && OUTPUT_DIR="$REPO/checkpoint/${RUN_NAME}"
mkdir -p "$OUTPUT_DIR"

REWARD_WEIGHTS="1.0 ${W_OVERLAP} 1.0 1.0"

echo "=========================================================================="
echo "Model:            $MODEL"
echo "GPUs (total $NUM_GPUS):  DINO=cuda:$DINO_GPU  vLLM=cuda:$VLLM_GPU  train=cuda:[$TRAIN_GPUS] ($TRAIN_N procs)$([ "$SHARE_SIDECAR_GPU" = true ] && echo '  [sidecars SHARED on cuda:0]')"
echo "Generation:       vLLM server  127.0.0.1:$VLLM_PORT  gpu_mem=$VLLM_GPU_MEM  max_len=$VLLM_MAX_MODEL_LEN"
echo "DINO reward:      127.0.0.1:$DINO_PORT  box_threshold=$BOX_THRESHOLD max_box_area=$([[ "$MAX_BOX_AREA" == "0" ]] && echo 'off (no per-box cap)' || echo "$MAX_BOX_AREA") max_union_area=$([[ -n "$MAX_UNION_AREA" ]] && echo "$MAX_UNION_AREA" || echo 'off')"
echo "Overlap reward:   layer=$OVERLAP_LAYER heads=[$OVERLAP_HEADS] token_reduction=$TOKEN_REDUCTION w_overlap=$W_OVERLAP"
echo "Metric:           $OVERLAP_METRIC$([[ -n "$MASS_FLOOR_TAU" ]] && echo " mass_floor_tau=$MASS_FLOOR_TAU" || echo " (no mass floor)")"
echo "Overlap rows:     $([ "$NATURAL_ONLY" = true ] && echo 'natural images only (non-natural: format+accuracy+judge)' || echo 'all rows')"
echo "Validation:       $([ -n "$VAL_SETS_DIR" ] && echo "accuracy only, step 0 then every $EVAL_STEPS steps, from $VAL_SETS_DIR" || echo 'off')"
echo "Checkpoints:      save every $SAVE_STEPS, keep every $CKPT_KEEP_EVERY"
echo "Benchmarks:       $([ "$AUTO_BENCH" = true ] && echo "dispatcher auto-started (${BENCH_GPUS}-GPU jobs)" || echo 'off (see --auto-bench)')"
echo "Batch:            per_device=$PER_DEVICE_BATCH num_generations=$NUM_GENERATIONS grad_accum=$GRAD_ACCUM  (gen_batch=$(( PER_DEVICE_BATCH * TRAIN_N * GRAD_ACCUM )))"
echo "LoRA targets:     ${LORA_TARGETS//,/ }"
echo "T5 step clf:      $OVERLAP_STEPS_DEVICE  ckpt=$OVERLAP_STEPS_CKPT"
echo "Run name:         $RUN_NAME"
echo "Output dir:       $OUTPUT_DIR"
echo "Mode:             $($DIRECT && echo 'direct (no SLURM)' || echo "SLURM ($PARTITION, ${DURATION}h)")"
echo "Judge key:        $([[ -n "${NVIDIA_API_KEY:-}${OPENAI_API_KEY:-}" ]] && echo '(set)' || echo '(MISSING - openai_reward will fail)')"
echo "WandB:            $([[ -n "${WANDB_API_KEY:-}" ]] && echo '(online)' || echo '(offline)')"
[[ -n "$EXTRA_ARGS" ]] && echo "Extra args:       $EXTRA_ARGS"
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
            export CONDA_ENV=$CONDA_ENV;
            export HF_HOME=$HF_HOME;
            export WANDB_API_KEY=${WANDB_API_KEY:-};
            export WANDB_DATA_DIR=${WANDB_DATA_DIR:-/home/uberger/scratch/cache/wandb_data};
            export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-/home/uberger/scratch/cache/wandb_cache};
            export VLLM_NO_USAGE_STATS=1;
            export DO_NOT_TRACK=1;
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
                --num-generations $NUM_GENERATIONS \
                --grad-accum $GRAD_ACCUM \
                --per-device-batch $PER_DEVICE_BATCH \
                --learning-rate $LEARNING_RATE \
                --w-overlap $W_OVERLAP \
                --token-reduction $TOKEN_REDUCTION \
                --lora-targets $LORA_TARGETS \
                --overlap-heads $OVERLAP_HEADS \
                --overlap-layer $OVERLAP_LAYER \
                --box-threshold $BOX_THRESHOLD \
                --max-box-area $MAX_BOX_AREA \
                ${MAX_UNION_AREA:+--max-union-area $MAX_UNION_AREA} \
                --overlap-metric $OVERLAP_METRIC \
                ${MASS_FLOOR_TAU:+--mass-floor-tau $MASS_FLOOR_TAU} \
                --dino-port $DINO_PORT \
                --vllm-port $VLLM_PORT \
                --vllm-gpu-mem $VLLM_GPU_MEM \
                --vllm-max-model-len $VLLM_MAX_MODEL_LEN \
                --vllm-enforce-eager $VLLM_ENFORCE_EAGER \
                --eval-steps $EVAL_STEPS \
                --bench-gpus $BENCH_GPUS \
                ${VAL_SETS_DIR:+--val-sets-dir $VAL_SETS_DIR} \
                $([ -z "$VAL_SETS_DIR" ] && echo --no-eval) \
                $([ "$AUTO_BENCH" = true ] && echo --auto-bench) \
                $([ "$SHARE_SIDECAR_GPU" = true ] && echo --share-sidecar-gpu) \
                $EXTRA_ARGS
        '"
    echo "Submitted $RUN_NAME"
    exit 0
fi

# ---------- direct path ----------
source "$CONDA_SH"
echo "Activating conda env $CONDA_ENV"
# conda activate and package activate.d hooks (e.g. cuda-nvcc's, which expands
# $NVCC_PREPEND_FLAGS with no default) assume nounset is OFF. Our `set -u` makes
# any such unguarded expansion a fatal "unbound variable". Disable nounset for
# the duration of activation only, then restore it.
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
if [ "$(command -v python)" != "$CONDA_PREFIX/bin/python" ]; then
    echo "ERROR: after CUDA_HOME setup, python resolves to '$(command -v python)', expected '$CONDA_PREFIX/bin/python' (env '$CONDA_ENV'). CUDA_HOME='$CUDA_HOME' shadowed it." >&2
    exit 1
fi
bash "$REPO/check_cuda_home.sh" || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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
# Keep wandb artifact staging/cache OFF the tiny /home partition (10G quota); it
# otherwise piles up under ~/.local/share/wandb and ~/.cache/wandb and fills /home.
export WANDB_DATA_DIR=${WANDB_DATA_DIR:-/home/uberger/scratch/cache/wandb_data}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-/home/uberger/scratch/cache/wandb_cache}
mkdir -p "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR"
# Don't write vLLM usage stats into ~/.config (also on /home).
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export OVERLAP_STEPS_DEVICE
export OVERLAP_STEPS_CKPT
[ -d "$OVERLAP_STEPS_CKPT/encoder" ] || {
    echo "ERROR: steps-classifier ckpt not found at $OVERLAP_STEPS_CKPT (need encoder/ tokenizer/ head.pt). Set OVERLAP_STEPS_CKPT to a valid path." >&2
    exit 1
}
[ -n "${NVIDIA_API_KEY:-}" ] && export NVIDIA_API_KEY
[ -n "${OPENAI_API_KEY:-}" ] && export OPENAI_API_KEY
[ -n "${OPENAI_BASE_URL:-}" ] && export OPENAI_BASE_URL
[ -n "${JUDGE_MODEL:-}" ] && export JUDGE_MODEL

LOG_DIR="$OUTPUT_DIR/sidecar_logs"
mkdir -p "$LOG_DIR"

# ---------- cleanup: kill sidecars (and their worker children) on any exit ----------
DINO_PID=""
VLLM_PID=""
CLEANUP_PID=""
cleanup() {
    echo "[cleanup] shutting down sidecars ..."
    for pid in "$VLLM_PID" "$DINO_PID" "$CLEANUP_PID" "${BENCH_WATCHER_PID:-}"; do
        [ -n "$pid" ] || continue
        pkill -TERM -P "$pid" 2>/dev/null || true
        kill -TERM "$pid" 2>/dev/null || true
    done
    # vLLM spawns a detached EngineCore worker that holds GPU memory and doesn't
    # match the serve cmdline -- kill it explicitly or it orphans GPU 1.
    pkill -TERM -u "$USER" -f "trl.scripts.vllm_serve --model $MODEL" 2>/dev/null || true
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
            echo "[health] ERROR: $name process (pid $pid) died before becoming healthy." >&2
            tail -40 "$LOG_DIR/${name}.log" >&2 2>/dev/null || true
            exit 1
        fi
        if grep -qE "Engine core initialization failed|EngineCore failed to start|Traceback \(most recent call last\)" "$LOG_DIR/${name}.log" 2>/dev/null; then
            echo "[health] ERROR: $name logged a fatal error (worker died); aborting." >&2
            tail -40 "$LOG_DIR/${name}.log" >&2 2>/dev/null || true
            exit 1
        fi
        sleep 5; waited=$(( waited + 5 ))
        if (( waited >= timeout_s )); then
            echo "[health] ERROR: $name not healthy after ${timeout_s}s." >&2
            tail -40 "$LOG_DIR/${name}.log" >&2 2>/dev/null || true
            exit 1
        fi
    done
    echo "[health] $name is up (after ${waited}s)."
}

# ---------- 1. Grounding-DINO reward server on GPU 0 ----------
echo "[start] Grounding-DINO on cuda:$DINO_GPU -> 127.0.0.1:$DINO_PORT"
CUDA_VISIBLE_DEVICES=$DINO_GPU DINO_SERVER_BATCH=${DINO_SERVER_BATCH:-8} \
    python "$REPO/serve_grounding_dino.py" --host 127.0.0.1 --port "$DINO_PORT" \
    > "$LOG_DIR/dino.log" 2>&1 &
DINO_PID=$!

# ---------- 2. vLLM generation server on GPU 1 ----------
cd "$REPO/trl_repo"
echo "[start] vLLM server on cuda:$VLLM_GPU -> 127.0.0.1:$VLLM_PORT"
which python
VLLM_EAGER_FLAG=""
case "$VLLM_ENFORCE_EAGER" in
    True|true|1) VLLM_EAGER_FLAG="--enforce_eager True" ;;
esac
CUDA_VISIBLE_DEVICES=$VLLM_GPU \
    python -m trl.scripts.vllm_serve \
        --model "$MODEL" \
        --host 127.0.0.1 --port "$VLLM_PORT" \
        --gpu_memory_utilization "$VLLM_GPU_MEM" \
        --dtype bfloat16 \
        --max_model_len "$VLLM_MAX_MODEL_LEN" \
        --enable_prefix_caching True \
        $VLLM_EAGER_FLAG \
    > "$LOG_DIR/vllm.log" 2>&1 &
VLLM_PID=$!

# DINO loads fast (~1-2 min); vLLM must load the 8B + capture CUDA graphs (~5-15 min).
wait_for_health "http://127.0.0.1:$DINO_PORT/health"  "dino" 600  "$DINO_PID"
wait_for_health "http://127.0.0.1:$VLLM_PORT/health/" "vllm" 1800 "$VLLM_PID"

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
                    echo "[checkpoint cleanup] Removing $output_dir/checkpoint-$step"
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

# Omitted entirely when unset, so the dataclass default (None = floor off) applies.
# A value of 0 is still passed through: _mass_gate treats tau<=0 as "off", so an
# explicit --mass-floor-tau 0 disables the floor without falling back to the auroc
# default above.
MASS_FLOOR_FLAG=""
[[ -n "$MASS_FLOOR_TAU" ]] && MASS_FLOOR_FLAG="--mass_floor_tau $MASS_FLOOR_TAU"

# Same shape: omitted when unset, so the dataclass default (None = no union cap) applies
# and an existing run's command line is reproduced byte for byte. _union_mask treats a
# non-positive value as "off", so --max-union-area 0 disables it explicitly.
MAX_UNION_FLAG=""
[[ -n "$MAX_UNION_AREA" ]] && MAX_UNION_FLAG="--max_union_area $MAX_UNION_AREA"

# Omitted when off, so the dataclass default (False) applies and the command line of an
# existing run is reproduced byte for byte.
NATURAL_ONLY_FLAG=""
[[ "$NATURAL_ONLY" == true ]] && NATURAL_ONLY_FLAG="--overlap_natural_only True"

# Held-out validation. Omitted entirely when off, so eval_strategy stays "no" and the
# command line of an existing run is unchanged.
EVAL_FLAGS=""
if [[ -n "$VAL_SETS_DIR" ]]; then
    for _split in val_natural val_nonnatural; do
        [[ -d "$VAL_SETS_DIR/$_split" ]] || {
            echo "ERROR: $VAL_SETS_DIR/$_split not found. Build the validation sets first:" >&2
            echo "         python build_grpo_sets.py --build-val --out-dir $VAL_SETS_DIR" >&2
            echo "       or pass --no-eval to train without validation." >&2
            exit 1
        }
    done
    # Accuracy-only validation, driven by a callback rather than the Trainer's
    # evaluation loop, so it costs one batched vLLM call instead of re-running the
    # whole reward pipeline. A step-0 baseline is always taken.
    EVAL_FLAGS="--val_sets_dir $VAL_SETS_DIR --val_eval_steps $EVAL_STEPS"
fi

# ---------- benchmark-eval dispatcher ----------
# Submits a $BENCH_GPUS-GPU job whenever a kept checkpoint is waiting to be scored on
# the mini test suites. It holds no GPU itself and runs alongside training, here on
# the compute node -- watch_bench_evals.sh picks its own submission backend, falling
# back from submit_job (which cannot resolve a pool1-* hostname) to sbatch (which
# can). Deciding that there rather than here keeps one copy of the rule.
#
# A dispatcher that died on startup looks exactly like one that found no work, so
# confirm it is alive and print the manual command if it is not.
BENCH_WATCHER_PID=""
if [[ "$AUTO_BENCH" == true ]]; then
    # Record what this run starts from, so the benchmark job can score it as step 0.
    # Without a baseline the earliest benchmark point is step 100, and a curve with
    # no origin cannot say whether training helped or hurt.
    mkdir -p "$OUTPUT_DIR/bench_eval"
    echo "$MODEL" > "$OUTPUT_DIR/bench_eval/base_model.txt"
    echo "[bench] starting the benchmark dispatcher for $OUTPUT_DIR"
    bash "$REPO/watch_bench_evals.sh" --run-dir "$OUTPUT_DIR" --num-gpus "$BENCH_GPUS" \
        --every "$CKPT_KEEP_EVERY" > "$LOG_DIR/bench_watcher.log" 2>&1 &
    BENCH_WATCHER_PID=$!
    sleep 5
    if kill -0 "$BENCH_WATCHER_PID" 2>/dev/null; then
        sed -n '/^Submitting:/p' "$LOG_DIR/bench_watcher.log" | sed 's/^/[bench] /'
        echo "[bench] dispatcher running (pid $BENCH_WATCHER_PID), log: $LOG_DIR/bench_watcher.log"
    else
        BENCH_WATCHER_PID=""
        echo "==========================================================================" >&2
        echo "[bench] The dispatcher exited immediately. It said:" >&2
        sed 's/^/        /' "$LOG_DIR/bench_watcher.log" >&2
        echo "" >&2
        echo "        Training continues. To collect benchmarks, run this on the login" >&2
        echo "        node (it needs no GPUs):" >&2
        echo "" >&2
        echo "          bash $REPO/watch_bench_evals.sh --run-dir $OUTPUT_DIR \\" >&2
        echo "               --num-gpus $BENCH_GPUS --every $CKPT_KEEP_EVERY" >&2
        echo "==========================================================================" >&2
    fi
fi

# ---------- 4. GRPO training on GPUs 2..N-1 ----------
echo "[start] training on cuda:[$TRAIN_GPUS] ($TRAIN_N procs)"
CUDA_VISIBLE_DEVICES=$TRAIN_GPUS accelerate launch \
    --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    --num_processes "$TRAIN_N" \
    --main_process_port "$MASTER_PORT" \
    examples/scripts/grpo_vlm_qwen3.py \
    --model_name_or_path "$MODEL" \
    --attn_implementation sdpa \
    --output_dir "$OUTPUT_DIR" \
    --learning_rate "$LEARNING_RATE" \
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
    $MAX_UNION_FLAG \
    --overlap_metric "$OVERLAP_METRIC" \
    $MASS_FLOOR_FLAG \
    $NATURAL_ONLY_FLAG \
    $EVAL_FLAGS \
    --dino_api_base "http://127.0.0.1:$DINO_PORT" \
    --reward_weights $REWARD_WEIGHTS \
    --use_vllm \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port "$VLLM_PORT" \
    --use_peft \
    --lora_target_modules ${LORA_TARGETS//,/ } \
    --log_completions \
    --per_device_train_batch_size "$PER_DEVICE_BATCH" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --num_generations "$NUM_GENERATIONS" \
    --report_to wandb \
    --logging_steps 5 \
    --save_steps "$SAVE_STEPS" \
    --num_train_epochs 3 \
    --temperature 1 \
    $RESUME_FLAG \
    $EXTRA_ARGS

echo "Finished $RUN_NAME"
