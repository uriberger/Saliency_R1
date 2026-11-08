#!/usr/bin/env bash
# Submit the DeepEyes go/no-go gate to SLURM: three models x two benchmarks, one job.
#
#   NVIDIA_API_KEY=... bash launch_deepeyes_gate_job.sh --name deepeyes-gate
#   NVIDIA_API_KEY=... bash launch_deepeyes_gate_job.sh --name deepeyes-gate-vstar \
#       --bench vstar --duration 2
#
# Same `_job.sh` / `_run.sh` split as launch_laser_job.sh: this builds a runner and hands
# it to submit_job; launch_deepeyes_gate.sh is what serves and scores once the allocation
# exists. One model at a time, serially, inside one allocation -- vLLM wants the whole
# card and three 7-8B servers do not share one usefully.
#
# THE THREE ARMS, and why these three. The gate's question is not "is DeepEyes good", it
# is "is there headroom LEFT on the backbone we would actually train":
#
#   deepeyes-7b   ChenShawn/DeepEyes-7B -- their released checkpoint, the upper end
#   qwen3vl-8b    Qwen/Qwen3-VL-8B-Instruct -- the untouched backbone
#   coldstart     checkpoint/coldstart_..._merged -- what every arm of ours starts from
#
# The last two are scored THROUGH THEIR ZOOM LOOP, not through lmms-eval. That is the
# comparison that matters: DeepEyes' claim is about what the loop buys, so a baseline
# measured without the loop concedes the question before it is asked.
#
# NOT OPTIONAL, in the order they bite:
#
#   * GPU hardware has to exist. `sinfo -h -N -o "%G" | grep -ci gpu` returning 0 means
#     the partitions are there and the NODES ARE NOT, and a job submitted into one never
#     starts and never errors -- it sits at Reason=PartitionConfig forever. This script
#     refuses rather than queueing into that; see docs/deepeyes-gate.md §4.
#   * build_deepeyes_eval_data.py must have run (CPU, cpu_short). It patches three faults
#     into their clone and converts HR-Bench parquet -> TSV. ~6.6 GB, a few minutes.
#   * NVIDIA_API_KEY must be exported. It is the judge, and without it the gate reads low
#     instead of failing.
#   * ChenShawn/DeepEyes-7B (16.6 GB) must be in HF_HOME already. The compute nodes run
#     HF_HUB_OFFLINE=1; pre-stage from a login node.
#
# COST. 191 V* rows + 1,600 HR-Bench rows per model, ~2 generate turns each, one 7-8B on
# one card. Budget ~1 h per model for both benchmarks at --workers 8, so 4 h and 1 GPU
# holds all three. V* alone is ~10 min per model and is the arm the decision turns on --
# `--bench vstar --duration 2` is the cheap version if the queue is tight.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

NAME=""
OUT_DIR=""
BENCH="vstar,hrbench"
MODELS="deepeyes-7b,qwen3vl-8b,coldstart"
DURATION=4
GPUS=1
TP=1
WORKERS=8
PARTITION_OVERRIDE=""
DRY_RUN=0
FORCE=0
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --name)      NAME="$2";               shift 2 ;;
        --out-dir)   OUT_DIR="$2";            shift 2 ;;
        --bench)     BENCH="$2";              shift 2 ;;
        --models)    MODELS="$2";             shift 2 ;;
        --duration)  DURATION="$2";           shift 2 ;;
        --gpus)      GPUS="$2";               shift 2 ;;
        --tp)        TP="$2";                 shift 2 ;;
        --workers)   WORKERS="$2";            shift 2 ;;
        --partition) PARTITION_OVERRIDE="$2"; shift 2 ;;
        --dry-run)   DRY_RUN=1;               shift   ;;
        --force)     FORCE=1;                 shift   ;;   # submit even with no GPU nodes
        --)          shift; EXTRA+=("$@"); break ;;
        *)           EXTRA+=("$1");           shift   ;;
    esac
done

[[ -n "$NAME" ]] || { echo "ERROR: --name is required (it names the job and the log)." >&2; exit 2; }
OUT_DIR=${OUT_DIR:-outputs/deepeyes_gate}
[[ "$OUT_DIR" = /* ]] || OUT_DIR="$REPO/$OUT_DIR"

# Resolve each arm to something vLLM can load. A relative checkpoint path is made
# absolute here, because the runner cd's to $REPO but submit_job's own cwd is not $REPO.
model_path() {
    case "$1" in
        deepeyes-7b) echo "ChenShawn/DeepEyes-7B" ;;
        qwen3vl-8b)  echo "Qwen/Qwen3-VL-8B-Instruct" ;;
        coldstart)   echo "$REPO/checkpoint/coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged" ;;
        *)           echo "" ;;
    esac
}

IFS=',' read -r -a ARMS <<< "$MODELS"
for m in "${ARMS[@]}"; do
    [[ -n "$(model_path "$m")" ]] || { echo "ERROR: --models $m is not an arm (deepeyes-7b|qwen3vl-8b|coldstart)." >&2; exit 2; }
done

if [[ -z "${NVIDIA_API_KEY:-}${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: no NVIDIA_API_KEY/OPENAI_API_KEY in the environment. That is the judge;" >&2
    echo "       without it every non-trivial answer is ungraded and the gate reads low." >&2
    exit 2
fi

DATA_DIR="$OUT_DIR/data"
[[ -f "$DATA_DIR/manifest.json" ]] || {
    echo "ERROR: $DATA_DIR/manifest.json missing. Run first (CPU, no GPU needed):" >&2
    echo "       python build_deepeyes_eval_data.py --out-dir $DATA_DIR" >&2; exit 2; }

# shellcheck source=/dev/null
source "$REPO/cluster_env.sh"
PARTITION=${PARTITION_OVERRIDE:-${PARTITION:-$(SR1_JOB_HOURS=$DURATION sr1_pick_partition)}}
ACCOUNT=${ACCOUNT:-nvr_israel_rlop}

# The one-second test from docs/deepeyes-gate.md §4. A partition can exist, accept the
# submission, and have no node behind it; the job then sits at Reason=PartitionConfig
# with StartTime=Unknown and nothing ever changes. Checking the GRES of actual NODES is
# what tells that apart from ordinary queueing.
GPU_NODES=$(sinfo -h -N -o "%G" 2>/dev/null | grep -ci gpu || true)
if [[ "${GPU_NODES:-0}" -eq 0 ]]; then
    echo "==========================================================================" >&2
    echo "REFUSING TO SUBMIT: no node on this cluster advertises a gpu GRES." >&2
    echo "  The GPU partitions still exist and sbatch would accept this job, but it" >&2
    echo "  would park at Reason=PartitionConfig and never start." >&2
    echo "  Move to cs-oci-ord-login-01, or pass --force if you know better." >&2
    echo "==========================================================================" >&2
    [[ $FORCE -eq 1 ]] || exit 3
    echo "[--force] submitting anyway." >&2
fi

sr1_find_submit_job || [[ $DRY_RUN -eq 1 ]] || {
    echo "ERROR: submit_job not found under the cluster-interface paths." >&2; exit 1; }

LOG_ROOT="$REPO/outputs/logs"
mkdir -p "$LOG_ROOT" "$OUT_DIR"

RUNNER="$LOG_ROOT/$NAME.runner.sh"
{
    echo "#!/usr/bin/env bash"
    echo "set -uo pipefail"
    printf 'cd %q\n' "$REPO"
    echo "export CONDA_ENV=${CONDA_ENV:-laser}"
    echo "export HF_HOME=${HF_HOME:-/home/uberger/scratch/cache/hf_cache}"
    echo "export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}"
    [[ -n "${NVIDIA_API_KEY:-}" ]] && printf 'export NVIDIA_API_KEY=%q\n' "$NVIDIA_API_KEY"
    [[ -n "${OPENAI_API_KEY:-}" ]] && printf 'export OPENAI_API_KEY=%q\n' "$OPENAI_API_KEY"
    [[ -n "${OPENAI_BASE_URL:-}" ]] && printf 'export OPENAI_BASE_URL=%q\n' "$OPENAI_BASE_URL"
    [[ -n "${JUDGE_MODEL:-}" ]] && printf 'export JUDGE_MODEL=%q\n' "$JUDGE_MODEL"
    echo "rc=0"
    # Serially, and `|| rc=1` rather than `set -e`: an arm that OOMs should not cost the
    # arms after it. The gate is readable with two of three rows; it is not readable at
    # all if the first failure takes the allocation down.
    for m in "${ARMS[@]}"; do
        echo
        echo "echo '########## arm $m'"
        printf 'bash launch_deepeyes_gate.sh --tag %q --model %q --out-dir %q --bench %q --tp %q --workers %q --data-dir %q' \
            "$m" "$(model_path "$m")" "$OUT_DIR" "$BENCH" "$TP" "$WORKERS" "$DATA_DIR"
        for a in ${EXTRA[@]+"${EXTRA[@]}"}; do printf ' %q' "$a"; done
        echo " || rc=1"
    done
    echo
    echo 'exit "$rc"'
} > "$RUNNER"
chmod +x "$RUNNER"

echo "=========================================================================="
echo "Job       : $NAME   ($ACCOUNT, $PARTITION, ${DURATION}h, ${GPUS} GPU)"
echo "Arms      : $MODELS"
echo "Bench     : $BENCH"
echo "Out dir   : $OUT_DIR"
echo "Data dir  : $DATA_DIR"
echo "GPU nodes : $GPU_NODES advertising a gpu GRES"
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
