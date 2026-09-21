#!/usr/bin/env bash
# Submit the token-mediation pilot to SLURM. Same shape as launch_sink_location_job.sh:
# this builds a runner and hands it to submit_job.
#
#   bash launch_token_mediation_job.sh --name tm-pilot --limit 150 \
#       --model checkpoint/coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged
#
# The selftest runs FIRST inside the allocation and `set -e` makes it a gate: an arm that
# silently does nothing looks exactly like a null result, so the run is not allowed to
# start until the self-swap identity and the zero-KL identity have both been checked on
# this model. One GPU on purpose -- the smallest request that can run starts soonest, and
# the pilot is ~150 pictures.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

NAME=""
MODEL=""
OUT_DIR="outputs/token_mediation/pilot"
LIMIT=150
STAGE=run
CANON=12
DURATION=2
GPUS=1
SHARDS=1
PARTITION_OVERRIDE=""
DRY_RUN=0
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)   DRY_RUN=1;               shift   ;;
        --name)      NAME="$2";               shift 2 ;;
        --model)     MODEL="$2";              shift 2 ;;
        --out-dir)   OUT_DIR="$2";            shift 2 ;;
        --limit)     LIMIT="$2";              shift 2 ;;
        --stage)     STAGE="$2";              shift 2 ;;
        --canon)     CANON="$2";              shift 2 ;;
        --duration)  DURATION="$2";           shift 2 ;;
        --gpus)      GPUS="$2";               shift 2 ;;
        --shards)    SHARDS="$2";             shift 2 ;;
        --partition) PARTITION_OVERRIDE="$2"; shift 2 ;;
        --)          shift; EXTRA+=("$@"); break ;;
        *)           EXTRA+=("$1");           shift ;;
    esac
done

[[ -n "$NAME"  ]] || { echo "ERROR: --name is required." >&2; exit 2; }
[[ -n "$MODEL" ]] || { echo "ERROR: --model is required." >&2; exit 2; }
[[ "$OUT_DIR" = /* ]] || OUT_DIR="$REPO/$OUT_DIR"

# shellcheck source=/dev/null
source "$REPO/cluster_env.sh"
PARTITION=${PARTITION_OVERRIDE:-${PARTITION:-$(SR1_JOB_HOURS=$DURATION sr1_pick_partition)}}
ACCOUNT=${ACCOUNT:-nvr_israel_rlop}
CONDA_ENV=${CONDA_ENV:-saliency_r1_qwen3_vllm}

CONDA_ROOT=${CONDA_ROOT:-}
if [[ -z "$CONDA_ROOT" ]]; then
    for CAND in /home/uberger/scratch/miniconda3 \
                /lustre/fs12/portfolios/nvr/projects/nvr_israel_rlop/users/uberger/research/miniforge3; do
        [[ -f "$CAND/etc/profile.d/conda.sh" ]] && { CONDA_ROOT="$CAND"; break; }
    done
fi
[[ -n "$CONDA_ROOT" ]] || { echo "ERROR: no conda root found." >&2; exit 1; }

sr1_find_submit_job || [[ $DRY_RUN -eq 1 ]] || {
    echo "ERROR: submit_job not found under the cluster-interface paths." >&2; exit 1; }

LOG_ROOT="$REPO/outputs/logs"
mkdir -p "$LOG_ROOT" "$OUT_DIR"

RUNNER="$LOG_ROOT/$NAME.runner.sh"
{
    echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    printf 'cd %q\n' "$REPO"
    echo "export HF_HOME=${HF_HOME:-/home/uberger/scratch/cache/hf_cache}"
    echo "export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}"
    printf 'source %q\n' "$CONDA_ROOT/etc/profile.d/conda.sh"
    printf 'conda activate %q\n' "$CONDA_ENV"
    echo 'echo "[runner] python: $(which python)"'
    echo 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true'
    printf 'python token_mediation_probe.py selftest --model %q\n' "$MODEL"
    for ((s = 0; s < SHARDS; s++)); do
        printf 'CUDA_VISIBLE_DEVICES=%d python token_mediation_probe.py %q' \
            "$((s % GPUS))" "$STAGE"
        printf ' --model %q --out %q --limit %q --shard %d --shards %d' \
            "$MODEL" "$OUT_DIR" "$LIMIT" "$s" "$SHARDS"
        for a in ${EXTRA[@]+"${EXTRA[@]}"}; do printf ' %q' "$a"; done
        [[ $SHARDS -gt 1 ]] && printf ' &'
        echo
    done
    [[ $SHARDS -gt 1 ]] && echo "wait"
    if [[ "$STAGE" == map ]]; then
        printf 'python token_mediation_probe.py mapreport --out %q --canon %q\n' \
            "$OUT_DIR" "$CANON"
    else
        printf 'python token_mediation_probe.py report --out %q\n' "$OUT_DIR"
    fi
} > "$RUNNER"
chmod +x "$RUNNER"

echo "=========================================================================="
echo "Job     : $NAME   ($ACCOUNT, $PARTITION, ${DURATION}h, ${GPUS} GPU)"
echo "Model   : $MODEL"
echo "Stage   : $STAGE"
echo "Out dir : $OUT_DIR   (limit $LIMIT, $SHARDS shard(s))"
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
