#!/usr/bin/env bash
# Submit ergo_zoom_gate.py to SLURM, on ONE GPU.
#
#   bash launch_ergo_zoom_gate_job.sh \
#       --model checkpoint/coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged \
#       --reference Qwen/Qwen3-VL-8B-Instruct \
#       [--out-dir outputs/ergo_gate_zoom] [--duration 2] [-- <forwarded to the python>]
#
# ONE GPU, not eight: this is 191 V* rows x 2 turns x 2 models of plain HF `generate`
# at batch 4. There is nothing to shard, no vLLM engine, no detector and no judge, and
# the queue for one card is shorter than the queue for a node.
#
# WHAT THE FIRST SCORED LINES SHOULD SAY, and what a mismatch there means:
#
#   * `processor_resized_under_us  0` -- the gate pre-resizes each image with
#     smart_resize(factor=32) and then asserts the processor left it alone, by reading
#     image_grid_thw back. Anything but 0 means the "resized" coordinate frame is not the
#     frame the model saw, and EVERY number in the frame table is measuring something
#     else. Fix the pixel plumbing before reading further.
#   * `coord hint: 'none'` -- the frame verdict is only valid under ERGO's verbatim
#     prompt. `--coord-hint rel1000` is a separate, named experiment about whether a
#     rescuable prompt exists; it is not the measurement.
#   * `a 1280-token budget really is ~1280 tokens` -- ERGO's code writes every budget as
#     N*28*28. On Qwen3-VL one visual token is 32x32 px, so a literal copy buys 23% fewer
#     tokens than the table says.
#
# NOT OPTIONAL: `--reference Qwen/Qwen3-VL-8B-Instruct`. The cold start's zoom-parse rate
# on its own cannot tell "this recipe does not produce boxes" from "this CHECKPOINT does
# not"; the backbone has grounding in its own post-training and the cold start may have
# trained it out. Those two have completely different fixes and only the pair separates
# them.
#
# The V* snapshot must be on disk first (it is, in HF_HOME, fetched once from the login
# node). HF_HUB_OFFLINE=1 is set in the runner so a compute node cannot stall on the hub.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

NAME="ergo-zoom-gate"
MODEL="checkpoint/coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged"
REFERENCE="Qwen/Qwen3-VL-8B-Instruct"
OUT_DIR="outputs/ergo_gate_zoom"
DURATION=2
GPUS=1
DRY_RUN=0
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)   DRY_RUN=1;       shift   ;;
        --name)      NAME="$2";       shift 2 ;;
        --model)     MODEL="$2";      shift 2 ;;
        --reference) REFERENCE="$2";  shift 2 ;;
        --no-reference) REFERENCE=""; shift   ;;
        --out-dir)   OUT_DIR="$2";    shift 2 ;;
        --duration)  DURATION="$2";   shift 2 ;;
        --gpus)      GPUS="$2";       shift 2 ;;
        --partition) PARTITION="$2";  shift 2 ;;
        --)          shift; EXTRA+=("$@"); break ;;
        *)           EXTRA+=("$1"); shift ;;
    esac
done

# shellcheck source=/dev/null
source "$REPO/cluster_env.sh"
PARTITION=${PARTITION:-$(SR1_JOB_HOURS=$DURATION sr1_pick_partition)}
ACCOUNT=${ACCOUNT:-nvr_israel_rlop}
# The Qwen3-VL training env: transformers 5.13.0.dev0, which is the one that has this
# backbone's processor. The `ergo` env from P0 is a clone of it and would work too, but
# this gate touches neither trl nor vLLM, so it does not need to wait on P0.
CONDA_ENV=${CONDA_ENV:-saliency_r1_qwen3}

sr1_find_submit_job || [[ $DRY_RUN -eq 1 ]] || {
    echo "ERROR: submit_job not found under the cluster-interface paths." >&2; exit 1; }

LOG_ROOT="$REPO/outputs/logs"
mkdir -p "$LOG_ROOT" "$REPO/$OUT_DIR"

RUNNER="$LOG_ROOT/$NAME.runner.sh"
{
    echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    printf 'cd %q\n' "$REPO"
    echo 'source "$(conda info --base)/etc/profile.d/conda.sh"'
    echo "conda activate $CONDA_ENV"
    echo "export HF_HOME=${HF_HOME:-/home/uberger/scratch/cache/hf_cache}"
    echo "export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}"
    printf 'python %q/ergo_zoom_gate.py --model %q' "$REPO" "$MODEL"
    [[ -n "$REFERENCE" ]] && printf ' --reference %q' "$REFERENCE"
    printf ' --out %q --dump-completions %q' \
        "$OUT_DIR/gate.json" "$OUT_DIR/completions.jsonl"
    for a in ${EXTRA[@]+"${EXTRA[@]}"}; do printf ' %q' "$a"; done
    printf '\n'
} > "$RUNNER"
chmod +x "$RUNNER"

echo "=========================================================================="
echo "Job       : $NAME   ($ACCOUNT, $PARTITION, ${DURATION}h, ${GPUS} GPU)"
echo "Model     : $MODEL"
echo "Reference : ${REFERENCE:-(none -- the verdict will not be separable, see header)}"
echo "Out dir   : $OUT_DIR"
echo "Extra     : ${EXTRA[*]:-(none)}"
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
