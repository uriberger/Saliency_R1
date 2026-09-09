#!/usr/bin/env bash
# Submit launch_saliency_viz.sh to SLURM, for one or MORE models in one allocation.
#
# The launcher it wraps assumes it is already ON a node with GPUs -- it fans one shard per
# visible device and waits -- so every saliency_viz run so far was started by hand from
# inside an allocation. This is the `_job.sh` half, the same split as
# launch_overlap_probe_job.sh / launch_sink_shift_job.sh.
#
#   bash launch_saliency_viz_job.sh --name sviz-3models \
#       --model vanilla=Qwen/Qwen3-VL-8B-Instruct \
#       --model overlap-wov0.4=checkpoint/coldstart_..._merged+checkpoint/grpo-...-trmean \
#       [--gpus 8] [--duration 1] [--n-samples 20] [--out-dir <dir>] \
#       [-- <anything else, forwarded verbatim to launch_saliency_viz.sh>]
#
# --model NAME=BASE[+ADAPTER] may be repeated. Each entry is one full scan+render into
# <out-dir>/NAME, run SEQUENTIALLY inside the single allocation: each scan already uses
# every GPU on the node, so running two at once only halves each one's memory. BASE is
# anything AutoProcessor/from_pretrained accepts (a HF id resolves from the offline cache);
# +ADAPTER is a PEFT LoRA merged in-memory by overlap_probe.load_model.
#
# Give every model the same --n-samples/--seed/--dataset (the default) and the three runs
# draw the SAME rows, so the pictures are of the same images and can be read side by side.
# The chains differ per model, so the per-step maps do not line up -- only the chain-level
# panel (the map averaged over that model's own observe steps) is directly comparable.
#
# Runtime reference: 20 samples, all five maps, 8 GPUs is ~4.5 minutes of scan plus ~2
# minutes of model load (outputs/saliency_viz/glimpse-first). `-- --methods glimpse` is
# roughly half of that. A 1-hour allocation therefore holds several models and buys
# batch_short, which starts in about a minute -- see cluster_env.sh.
#
# Everything after `--` goes to the inner launcher untouched, which is where the viz flags
# live (--methods, --max-new-tokens, --max-steps, --glimpse-*, --norm, --cmap...).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

NAME=""
MODELS=()
GPUS=8
DURATION=1
N_SAMPLES=20
OUT_DIR=""
DRY_RUN=0
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)     DRY_RUN=1;       shift   ;;
        --name)        NAME="$2";       shift 2 ;;
        --model)       MODELS+=("$2");  shift 2 ;;
        --gpus)        GPUS="$2";       shift 2 ;;
        --duration)    DURATION="$2";   shift 2 ;;
        --n-samples)   N_SAMPLES="$2";  shift 2 ;;
        --out-dir)     OUT_DIR="$2";    shift 2 ;;
        --)            shift; EXTRA+=("$@"); break ;;
        *)             EXTRA+=("$1");   shift ;;
    esac
done

[[ -n "$NAME" ]] || { echo "ERROR: --name is required (it names the job and the log)." >&2; exit 2; }
[[ ${#MODELS[@]} -gt 0 ]] || {
    echo "ERROR: at least one --model NAME=BASE[+ADAPTER] is required." >&2; exit 2; }
OUT_DIR=${OUT_DIR:-$REPO/outputs/saliency_viz/$NAME}

# Reject a malformed --model here rather than an hour later inside the allocation, and
# resolve every path against the repo now: the runner cds elsewhere and a relative
# checkpoint/ would then point at nothing.
declare -a TAGS=() BASES=() ADAPTERS=()
for m in "${MODELS[@]}"; do
    [[ "$m" == *=* ]] || { echo "ERROR: --model '$m' is not NAME=BASE[+ADAPTER]" >&2; exit 2; }
    tag="${m%%=*}"; spec="${m#*=}"
    base="${spec%%+*}"; adap=""
    [[ "$spec" == *+* ]] && adap="${spec#*+}"
    [[ -n "$tag" && -n "$base" ]] || { echo "ERROR: --model '$m' has an empty half" >&2; exit 2; }
    # A local directory is made absolute; anything else (a HF id like Qwen/Qwen3-VL-8B-
    # Instruct) is passed through untouched for the offline hub cache to resolve.
    [[ -d "$base" ]] && base="$(cd "$base" && pwd)"
    if [[ -n "$adap" ]]; then
        [[ -d "$adap" ]] || { echo "ERROR: adapter not found: $adap" >&2; exit 2; }
        adap="$(cd "$adap" && pwd)"
        [[ -f "$adap/adapter_config.json" ]] || {
            echo "ERROR: $adap has no adapter_config.json -- is it a merged model? Then " >&2
            echo "       pass it as the BASE half, with no '+'." >&2; exit 2; }
    fi
    TAGS+=("$tag"); BASES+=("$base"); ADAPTERS+=("$adap")
done

# shellcheck source=/dev/null
source "$REPO/cluster_env.sh"
PARTITION=${PARTITION:-$(SR1_JOB_HOURS=$DURATION sr1_pick_partition)}
ACCOUNT=${ACCOUNT:-nvr_israel_rlop}
CONDA_ENV=${CONDA_ENV:-saliency_r1_qwen3_vllm}

sr1_find_submit_job || [[ $DRY_RUN -eq 1 ]] || {
    echo "ERROR: submit_job not found under the cluster-interface paths." >&2; exit 1; }

LOG_ROOT="$REPO/outputs/logs"
mkdir -p "$LOG_ROOT" "$OUT_DIR"

# A worktree does not get cold_data/ copied in -- only the paths in .worktree-links are
# symlinked -- so point the viz at the central tree's validation sets when this checkout
# has none. An explicit --dataset in EXTRA still wins: argparse takes the last one.
DATASET=""
if [[ ! -d "$REPO/cold_data/grpo_sets/val_natural" ]]; then
    DATASET="/lustre/fs1/portfolios/nvr/projects/nvr_israel_rlop/users/uberger/research/saliency_r1/cold_data/grpo_sets/val_natural"
fi

# The inner command as a file rather than a quoted -c string: the viz flags carry commas
# (--methods a,b) and equals signs, and nesting those inside `bash -c '...'` is how a flag
# silently loses half of itself.
RUNNER="$LOG_ROOT/$NAME.runner.sh"
{
    echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    printf 'cd %q\n' "$REPO"
    echo "export CONDA_ENV=$CONDA_ENV"
    echo "export HF_HOME=${HF_HOME:-/home/uberger/scratch/cache/hf_cache}"
    echo "export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}"
    echo "export OVERLAP_STEPS_CKPT=${OVERLAP_STEPS_CKPT:-$REPO/checkpoint/steps_classifier/best}"
    echo "rc=0"
    for i in "${!TAGS[@]}"; do
        echo
        printf 'echo "########## model %s (%s of %s)"\n' "${TAGS[$i]}" "$((i + 1))" "${#TAGS[@]}"
        # `|| rc=1`: one model that OOMs or has a bad path must not cost the other two
        # their allocation. The exit status still reports that something failed.
        printf 'bash %q/launch_saliency_viz.sh --gpus %q --n-samples %q --out-dir %q/%q' \
            "$REPO" "$GPUS" "$N_SAMPLES" "$OUT_DIR" "${TAGS[$i]}"
        printf ' --base-model %q' "${BASES[$i]}"
        [[ -n "${ADAPTERS[$i]}" ]] && printf ' --adapter %q' "${ADAPTERS[$i]}"
        [[ -n "$DATASET" ]] && printf ' --dataset %q' "$DATASET"
        # Only the first model pays for the CPU selftest; it gates the pixel->token
        # regrouping, which is a property of the processor and not of the weights.
        [[ $i -gt 0 ]] && printf ' --skip-selftest'
        for a in ${EXTRA[@]+"${EXTRA[@]}"}; do printf ' %q' "$a"; done
        printf ' || rc=1\n'
    done
    echo
    echo 'exit "$rc"'
} > "$RUNNER"
chmod +x "$RUNNER"

echo "=========================================================================="
echo "Job       : $NAME   ($ACCOUNT, $PARTITION, ${DURATION}h, ${GPUS} GPU)"
echo "Out dir   : $OUT_DIR"
echo "Models    : ${#TAGS[@]}"
for i in "${!TAGS[@]}"; do
    echo "            ${TAGS[$i]}  <- ${BASES[$i]}${ADAPTERS[$i]:+  + ${ADAPTERS[$i]}}"
done
echo "Extra     : ${EXTRA[*]:-(none)}"
echo "Runner    : $RUNNER"
echo "=========================================================================="
cat "$RUNNER"
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
