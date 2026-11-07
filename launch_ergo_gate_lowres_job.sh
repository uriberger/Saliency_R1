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

# Does this cluster have GPUs AT ALL? oci-nrt-cs-001 lost every GPU node around
# 2026-10-27: each GPU partition still reports State=UP, so it accepts the job and
# parks it at Reason=PartitionConfig -- which reads exactly like ordinary queueing
# and is normally something to poll rather than resubmit. It is not queueing. The
# partitions have TotalNodes=0 and nothing anywhere advertises a gpu GRES, so a 4h
# allocation submitted here never starts and nobody finds out for a day.
if [[ $DRY_RUN -eq 0 ]] && command -v sinfo >/dev/null 2>&1; then
    if [[ "$(sinfo -h -N -o '%G' | grep -ci gpu)" -eq 0 ]]; then
        echo "ERROR: no node on $(hostname) advertises a gpu GRES." >&2
        echo "       Every GPU partition here reports TotalNodes=0, so this job would be" >&2
        echo "       accepted, parked at (PartitionConfig), and never scheduled." >&2
        echo "       Run it from cs-oci-ord-login-01 instead -- same repo, same paths." >&2
        echo "       Override with GATE_SKIP_GPU_CHECK=1 if the partitions have come back." >&2
        [[ -n "${GATE_SKIP_GPU_CHECK:-}" ]] || exit 1
    fi
fi

# shellcheck source=/dev/null
source "$REPO/cluster_env.sh"
PARTITION=${PARTITION_OVERRIDE:-${PARTITION:-$(SR1_JOB_HOURS=$DURATION sr1_pick_partition)}}
ACCOUNT=${ACCOUNT:-nvr_israel_rlop}
CONDA_ENV=${CONDA_ENV:-lmms_eval}

# `submit_job` on PATH proves nothing: it sources detect_system.sh, whose host
# case list does not cover every login node, and on a host it does not know it
# exits with "Unable to determine target cluster system from host name". Probe it
# and fall back to raw sbatch, the way watch_bench_evals.sh does. Capture the
# output rather than piping into `grep -q`: under pipefail grep's early exit
# SIGPIPEs detect_system.sh, the pipeline reports 141, and the negation would read
# a detected failure as success -- choosing submit_job on exactly the hosts where
# it cannot work.
BACKEND=""
if sr1_find_submit_job; then
    CI_DIR="$(dirname "$(command -v submit_job)")"
    if [[ -f "$CI_DIR/detect_system.sh" ]]; then
        PROBE="$(bash "$CI_DIR/detect_system.sh" 2>&1 || true)"
        [[ "$PROBE" != *"Unable to determine target cluster"* ]] && BACKEND=submit_job
    fi
fi
if [[ -z "$BACKEND" ]] && command -v sbatch >/dev/null 2>&1; then
    BACKEND=sbatch
fi
[[ -n "$BACKEND" || $DRY_RUN -eq 1 ]] || {
    echo "ERROR: neither submit_job nor sbatch can submit from $(hostname)." >&2; exit 1; }

# Only the sbatch backend needs these; submit_job sizes the allocation itself.
# 229 GiB per GPU is the ceiling, not a preference: slurm rejects outright any
# request whose memory-to-GPU ratio would strand GPUs on the node, so an over-ask
# means nothing is submitted at all.
CPUS_PER_GPU=${CPUS_PER_GPU:-28}
MEM_PER_GPU_GB=${MEM_PER_GPU_GB:-229}

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
echo "Backend   : ${BACKEND:-(none)}   (from $(hostname))"
echo "Runner    : $RUNNER"
echo "=========================================================================="
cat "$RUNNER"
echo "=========================================================================="

[[ $DRY_RUN -eq 1 ]] && { echo "[dry-run] not submitting."; exit 0; }

if [[ "$BACKEND" == "submit_job" ]]; then
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
    SBATCH="$LOG_ROOT/$NAME.sbatch"
    cat > "$SBATCH" <<SBATCH_EOF
#!/bin/bash
#SBATCH --account=$ACCOUNT
#SBATCH --partition=$PARTITION
#SBATCH --job-name=$NAME
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:$GPUS
#SBATCH --cpus-per-task=$((CPUS_PER_GPU * GPUS))
#SBATCH --mem=$((MEM_PER_GPU_GB * GPUS))G
#SBATCH --time=${DURATION}:00:00
#SBATCH --export=ALL
#SBATCH --output=$LOG_ROOT/$NAME.%j.out

bash $RUNNER
SBATCH_EOF
    sbatch "$SBATCH"
fi
