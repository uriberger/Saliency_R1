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

# ---------- is there any GPU hardware to wait for? ----------
# One second, and it is the difference between a job that is QUEUED and a job that will
# never start. As of 2026-11-06 every GPU partition on oci-nrt-cs-001 -- batch_block1,
# batch_singlenode, batch_short, batch_long, batch_large, backfill -- reports
# TotalNodes=0 / Nodes=(null), and 32 jobs have been pending at Reason=PartitionConfig
# since 2026-10-30. A pending GPU job there is not being patient, it is parked forever,
# and sbatch does not say so: it either accepts the job or rejects it with "More
# processors requested than permitted", which reads like a QoS problem and is not one.
if (( GPUS > 0 )) && command -v sinfo >/dev/null 2>&1; then
    HAVE_GPU=$(sinfo -h -N -o "%G" 2>/dev/null | grep -ci gpu || true)
    if [[ "$HAVE_GPU" == "0" ]]; then
        echo "ERROR: no node on $(scontrol show config 2>/dev/null | awk '/ClusterName/{print $3}')" \
             "advertises a gpu GRES." >&2
        echo "       Every GPU partition here is empty of hardware; a job submitted into" >&2
        echo "       one waits at Reason=PartitionConfig and never starts." >&2
        echo "" >&2
        echo "       Move to the second cluster, which checks out this repo at the SAME" >&2
        echo "       absolute paths, and run there:" >&2
        echo "" >&2
        echo "           ssh cs-oci-ord-login-01" >&2
        echo "           cd $REPO" >&2
        echo "           bash launch_ergo_zoom_gate_job.sh" >&2
        echo "" >&2
        echo "       Nothing needs copying: checkpoint/ and HF_HOME resolve on both." >&2
        echo "       SR1_ALLOW_NO_GPU=1 overrides this, for a cluster whose sinfo lies." >&2
        [[ "${SR1_ALLOW_NO_GPU:-0}" == "1" ]] || exit 1
    fi
fi

# shellcheck source=/dev/null
source "$REPO/cluster_env.sh"
if (( GPUS == 0 )); then
    # The CPU path exists for ONE thing: proving the script runs end to end -- both
    # generate() calls, both passes, the frame table and the JSON -- without waiting on
    # GPU hardware. It is not the measurement; an 8B model on CPU cannot produce 191 rows
    # x 2 turns x 2 models in any useful time. Keep --n and --max-new-tokens tiny.
    PARTITION=${PARTITION:-cpu_short}
fi
PARTITION=${PARTITION:-$(SR1_JOB_HOURS=$DURATION sr1_pick_partition)}
ACCOUNT=${ACCOUNT:-nvr_israel_rlop}
# The Qwen3-VL training env: transformers 5.13.0.dev0, which is the one that has this
# backbone's processor. The `ergo` env from P0 is a clone of it and would work too, but
# this gate touches neither trl nor vLLM, so it does not need to wait on P0.
CONDA_ENV=${CONDA_ENV:-saliency_r1_qwen3}

# ---------- how to submit ----------
# `submit_job` resolves through detect_system.sh, whose host list has no vscode or compute
# nodes on it, so finding it on PATH proves nothing -- it must be ASKED. On
# oci-nrt-cs-001-vscode-01 it is present and prints "Unable to determine target cluster
# system from host name", on stdout, with exit status 0. Captured and matched rather than
# piped into grep -q for the reason `launch_bench_eval_steps.sh` gives: under pipefail,
# grep exits at its first match, SIGPIPEs detect_system.sh and the pipeline reports 141,
# so the negation would choose submit_job on exactly the hosts where it cannot work.
# sbatch reaches slurmctld from anywhere and this job needs none of submit_job's extras
# (no autoresume, no container, one allocation).
submit_job_usable() {
    sr1_find_submit_job || return 1
    local dir out
    dir=$(dirname "$(command -v submit_job)")
    [[ -f "$dir/detect_system.sh" ]] || return 1
    out=$(bash "$dir/detect_system.sh" 2>&1) || true
    [[ "$out" != *"Unable to determine target cluster"* ]]
}

BACKEND=sbatch
submit_job_usable && BACKEND=submit_job
[[ "$BACKEND" == sbatch ]] && ! command -v sbatch >/dev/null 2>&1 && {
    echo "ERROR: neither submit_job nor sbatch can submit from $(hostname)." >&2
    echo "       Run ergo_zoom_gate.py directly instead, inside an allocation you hold." >&2
    exit 1; }

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
echo "Job       : $NAME   ($ACCOUNT, $PARTITION, ${DURATION}h, ${GPUS} GPU, via $BACKEND)"
echo "Model     : $MODEL"
echo "Reference : ${REFERENCE:-(none -- the verdict will not be separable, see header)}"
echo "Out dir   : $OUT_DIR"
echo "Extra     : ${EXTRA[*]:-(none)}"
echo "=========================================================================="
cat "$RUNNER"
echo "=========================================================================="

[[ $DRY_RUN -eq 1 ]] && { echo "[dry-run] not submitting."; exit 0; }

if [[ "$BACKEND" == submit_job ]]; then
    submit_job \
        --account "$ACCOUNT" \
        --partition "$PARTITION" \
        --name "$NAME" \
        --gpu "$GPUS" \
        --duration "$DURATION" \
        --outfile "$LOG_ROOT/$NAME.%j.out" \
        --logroot "$LOG_ROOT" \
        -c "bash $RUNNER"
else
    # The same allocation shape the dispatcher asks for. This cluster rejects a request
    # whose memory-to-GPU ratio would strand GPUs on the node rather than trimming it, so
    # the per-GPU ratios are copied from launch_bench_eval_steps.sh rather than guessed.
    # 16 and not the 28 `launch_bench_eval_steps.sh` uses: this cluster carries a
    # `20_cpus_per_gpu` QoS whose MaxTRESPerJob is cpu=20, so a 1-GPU job asking for 28 is
    # rejected outright with "More processors requested than permitted" -- a submit-time
    # error, not a trim. The 8-GPU launchers never see it because they take whole nodes.
    CPUS_PER_GPU=${CPUS_PER_GPU:-16}
    MEM_PER_GPU_GB=${MEM_PER_GPU_GB:-200}
    SBATCH_FILE="$LOG_ROOT/$NAME.sbatch"
    # `--gres=gpu:0` is not "no GPU", it is a malformed request on some SLURM builds, so
    # the CPU path omits the line entirely rather than asking for zero of something.
    GRES_LINE=""
    (( GPUS > 0 )) && GRES_LINE="#SBATCH --gres=gpu:$GPUS"
    (( GPUS > 0 )) && CPUS=$((CPUS_PER_GPU * GPUS)) || CPUS=32
    (( GPUS > 0 )) && MEM=$((MEM_PER_GPU_GB * GPUS)) || MEM=160
    cat > "$SBATCH_FILE" <<SBATCH_EOF
#!/bin/bash
#SBATCH --account=$ACCOUNT
#SBATCH --partition=$PARTITION
#SBATCH --job-name=$NAME
#SBATCH --nodes=1
#SBATCH --ntasks=1
$GRES_LINE
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem=${MEM}G
#SBATCH --time=${DURATION}:00:00
#SBATCH --export=ALL
#SBATCH --output=$LOG_ROOT/$NAME.%j.out

bash $RUNNER
SBATCH_EOF
    sbatch "$SBATCH_FILE"
fi
