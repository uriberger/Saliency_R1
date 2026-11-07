#!/usr/bin/env bash
# Submit Gate A of the ERGO port -- Qwen3-VL's low-resolution penalty -- to SLURM.
# Same `_job.sh` split as launch_laser_job.sh: this script only builds a runner and
# hands it to submit_job; ergo_gate_lowres.py --run is what walks the grid once the
# allocation exists.
#
#   bash launch_ergo_gate_lowres_job.sh --name ergo-gate-lowres \
#       --out-dir outputs/ergo_gate_lowres --gpus 8 --duration 4
#
# WHAT IT MEASURES. V*, HR-Bench 4K and HR-Bench 8K on vanilla Qwen3-VL-8B-Instruct
# and on our cold start, at visual-token caps of 640, 1280 and 16384, plus a fourth
# V* arm at 640 with the ground-truth crop appended as a second image. 14 units, one
# `accelerate launch` each. See ergo_gate_lowres.py's docstring for why the caps are
# N*32*32 and why --force_simple is not optional.
#
# COST AND SIZE. 8 GPUs for 4 hours. The driver is data-parallel through
# `accelerate launch`, so the GPUs shard documents rather than the model, and 8 is
# where this stops being a queue problem. The run RESUMES: a unit with a harvested
# json is skipped, so a job that dies at hour 3 is resubmitted, not restarted. It
# exits 17 while units are still owed and 0 when the grid is complete -- the same
# convention run_bench_eval_steps.sh uses -- so an autoresume chain terminates by
# itself.
#
# ORDER. Every V* arm runs before any HR-Bench arm. V* is 191 documents against
# HR-Bench's 2x300 and it is the benchmark ERGO's Table 1 is quoted from, so if the
# allocation runs short the gate still has the numbers it is actually decided on.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

NAME="ergo-gate-lowres"
OUT_DIR=""
DURATION=4
GPUS=8
HRBENCH_N=""
PARTITION_OVERRIDE=""
DRY_RUN=0
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)    DRY_RUN=1;               shift   ;;
        --name)       NAME="$2";               shift 2 ;;
        --out-dir)    OUT_DIR="$2";            shift 2 ;;
        --duration)   DURATION="$2";           shift 2 ;;
        --gpus)       GPUS="$2";               shift 2 ;;
        --hrbench-n)  HRBENCH_N="$2";          shift 2 ;;
        --partition)  PARTITION_OVERRIDE="$2"; shift 2 ;;
        --)           shift; EXTRA+=("$@"); break ;;
        *)            EXTRA+=("$1");           shift ;;
    esac
done

[[ -n "$OUT_DIR" ]] || OUT_DIR="$REPO/outputs/ergo_gate_lowres"
[[ "$OUT_DIR" = /* ]] || OUT_DIR="$REPO/$OUT_DIR"

# Generated here rather than in the job. The configs carry absolute paths into the
# shared lmms-eval clone, and a login node can fail that loudly in a second; a
# compute node fails it 2 minutes in, after the model has loaded, once per unit.
PY="${GATE_PYTHON:-/home/uberger/scratch/miniconda3/envs/lmms_eval/bin/python}"
"$PY" "$REPO/ergo_gate_lowres.py" --make-tasks --out-dir "$OUT_DIR" \
    ${HRBENCH_N:+--hrbench-n "$HRBENCH_N"}

# shellcheck source=/dev/null
source "$REPO/cluster_env.sh"
PARTITION=${PARTITION_OVERRIDE:-${PARTITION:-$(SR1_JOB_HOURS=$DURATION sr1_pick_partition)}}
ACCOUNT=${ACCOUNT:-nvr_israel_rlop}
CONDA_ENV=${CONDA_ENV:-lmms_eval}

sr1_find_submit_job || [[ $DRY_RUN -eq 1 ]] || {
    echo "ERROR: submit_job not found under the cluster-interface paths." >&2; exit 1; }

LOG_ROOT="$REPO/outputs/logs"
mkdir -p "$LOG_ROOT" "$OUT_DIR"

# Leave a quarter of an hour at the end: --job-minutes stops the driver STARTING a
# unit it cannot finish, so the allocation ends with a harvested grid rather than a
# truncated lmms-eval run whose partial output dir the next job would have to
# distinguish from a complete one.
JOB_MINUTES=$(( DURATION * 60 - 15 ))

RUNNER="$LOG_ROOT/$NAME.runner.sh"
{
    echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    printf 'cd %q\n' "$REPO"
    echo 'source /home/uberger/scratch/miniconda3/etc/profile.d/conda.sh'
    echo "conda activate ${CONDA_ENV}"
    echo "export HF_HOME=${HF_HOME:-/home/uberger/scratch/cache/hf_cache}"
    # The eval nodes have no route to the hub, and the offline and online
    # `datasets` paths resolve differently -- run_bench_eval.sh pins this for the
    # same reason, after an online/offline disagreement silently killed a suite.
    echo "export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}"
    # Without this lmms-eval reroutes the datasets cache to node-local scratch
    # (HF_HOME is on a remote filesystem), finds it empty, and then cannot reach
    # the hub to fill it -- every task dies on a ConnectionError before the model
    # loads. /nonexistent means "write straight to HF_HOME".
    echo 'export LMMS_LOCAL_CACHE=/nonexistent'
    echo 'export TOKENIZERS_PARALLELISM=false'
    # Nothing here should reach a judge: the V* parser is local and the HR-Bench
    # scorer was replaced precisely to remove the GPT fallback. Unsetting the key
    # makes that a fact rather than a hope.
    echo 'unset OPENAI_API_KEY'
    printf '%q %q --run --out-dir %q --num-gpus %q --job-minutes %q' \
        "$PY" "$REPO/ergo_gate_lowres.py" "$OUT_DIR" "$GPUS" "$JOB_MINUTES"
    [[ -n "$HRBENCH_N" ]] && printf ' --hrbench-n %q' "$HRBENCH_N"
    for a in ${EXTRA[@]+"${EXTRA[@]}"}; do printf ' %q' "$a"; done
    echo
} > "$RUNNER"
chmod +x "$RUNNER"

echo "=========================================================================="
echo "Job       : $NAME   ($ACCOUNT, $PARTITION, ${DURATION}h, ${GPUS} GPU)"
echo "Out dir   : $OUT_DIR"
echo "Units     : $("$PY" "$REPO/ergo_gate_lowres.py" --plan --out-dir "$OUT_DIR" | grep -c '^[a-z]')"
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
