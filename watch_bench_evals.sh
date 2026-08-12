#!/bin/bash
# Watch a GRPO run's checkpoint directory and submit a single-GPU eval job whenever
# there is work for one. Run this on a node that can submit jobs; it uses no GPUs
# of its own and can sit in a tmux window for the length of an experiment.
#
#   bash watch_bench_evals.sh --run-dir $REPO/checkpoint/<run-name>
#
# The rule is: submit only when a checkpoint is waiting AND no eval job of ours is
# already queued or running. So there is never more than one in flight, the queue
# is not flooded with thirty jobs that then compete with each other for fairshare,
# and no GPU is held while training is still producing the next checkpoint. The
# submitted job drains whatever has piled up and exits.
#
# State lives entirely on disk -- a checkpoint counts as done once its
# bench_eval/step-<N>.json exists -- so killing and restarting this loses nothing,
# and running it after training has finished backfills the remaining checkpoints.
#
# Usage:
#   bash watch_bench_evals.sh --run-dir DIR [--num-gpus 1] [--every 100]
#                             [--interval 60] [--duration 4] [--once]
#
# Environment: PARTITION, OPENAI_API_KEY / NVIDIA_API_KEY, HF_TOKEN
set -uo pipefail

SCRIPT_PATH="$(realpath "$0")"
SCRIPT_DIR=$(cd "$(dirname "$SCRIPT_PATH")" && pwd)
REPO=${REPO:-/home/uberger/scratch/research/saliency_r1}
source "$SCRIPT_DIR/cluster_env.sh"

ACCOUNT=nvr_israel_rlop
PARTITION=${PARTITION:-$(sr1_pick_partition batch_singlenode batch_long batch)}

RUN_DIR=""
# One GPU per eval job. The mini suites are 100 docs per benchmark, so this is a
# small job, and a single-GPU allocation gets scheduled while a 4-GPU one is still
# queued behind the 8-GPU training run it is supposed to be tracking. The trade is
# wall clock, which run_bench_eval.sh's --min-minutes guard scales with this number.
NUM_GPUS=1
EVERY=100
SAMPLE_N=100
INTERVAL=60
DURATION=4
ONCE=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-dir)   RUN_DIR="$2";   shift 2 ;;
        --num-gpus)  NUM_GPUS="$2";  shift 2 ;;
        --every)     EVERY="$2";     shift 2 ;;
        --sample-n)  SAMPLE_N="$2";  shift 2 ;;
        --interval)  INTERVAL="$2";  shift 2 ;;
        --duration)  DURATION="$2";  shift 2 ;;
        --partition) PARTITION="$2"; shift 2 ;;
        --once)      ONCE=true;      shift ;;
        --dry-run)   DRY_RUN=true; ONCE=true; shift ;;
        -h|--help)   sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

[[ -n "$RUN_DIR" ]] || { echo "error: --run-dir is required" >&2; exit 2; }
[[ -d "$RUN_DIR" ]] || { echo "error: no such run dir: $RUN_DIR" >&2; exit 2; }
# `pwd -P`, not `pwd`: /home/uberger/scratch/research/saliency_r1 and
# /lustre/fs1/.../saliency_r1 are the same directory through a symlink, and the
# in-flight job token is derived from this string. Two dispatchers started with
# different spellings of one run would hash differently, fail to see each other's
# job, and both submit.
RUN_DIR=$(cd "$RUN_DIR" && pwd -P)

RUN_NAME=$(basename "$RUN_DIR")
# submit_job rewrites the name it is given -- it appends a _<date>-<time> stamp and
# replaces dots with underscores (a run named ...wov0.11... comes back as
# ...wov0_11..._20260802-144252). An exact `squeue -n` match would therefore never
# fire, and this loop would submit a fresh job every poll, forever. So the check
# looks for a token that no such rewriting can touch: letters and digits only,
# derived from the run directory, hence unique per run.
JOB_TOKEN="bencheval$(printf '%s' "$RUN_DIR" | md5sum | cut -c1-8)"
JOB_NAME="${JOB_TOKEN}_${RUN_NAME}"
BENCH_DIR="$RUN_DIR/bench_eval"
LOG_ROOT="$REPO/outputs/logs"
mkdir -p "$BENCH_DIR" "$LOG_ROOT"

# ---------- how to submit ----------
# Two backends, because submit_job does not work everywhere:
#
#   submit_job  the ADLR wrapper. It sources detect_system.sh, which maps
#               $HOSTNAME to a cluster through a case list -- and that list has no
#               pool1-* entry, so it resolves on the login node and fails on every
#               compute node. The binary lives on /lustre and is readable from
#               both, so its PRESENCE proves nothing; the probe below asks
#               detect_system.sh whether it can actually place this host. Note it
#               reports the failure on stdout and still exits 0, so the output has
#               to be inspected rather than the status.
#
#   sbatch      stock Slurm. Works from anywhere that reaches slurmctld, compute
#               nodes included, which is what lets --auto-bench start this from
#               inside a training job.
CI_DIR=""
if command -v submit_job >/dev/null 2>&1; then
    CI_DIR=$(dirname "$(command -v submit_job)")
else
    for CI_ROOT in \
        /lustre/fs1/portfolios/adlr/projects/adlr_other_infra/release/cluster-interface \
        /lustre/fsw/portfolios/adlr/projects/adlr_other_infra/release/cluster-interface; do
        for CAND in "$CI_ROOT/latest" $(ls -1dt "$CI_ROOT"/*/ 2>/dev/null); do
            if [ -x "${CAND%/}/submit_job" ]; then
                CI_DIR="${CAND%/}"; export PATH="$CI_DIR:$PATH"; break 2
            fi
        done
    done
fi

submit_job_usable() {
    [ -n "$CI_DIR" ] && [ -f "$CI_DIR/detect_system.sh" ] || return 1
    # Capture and match, rather than piping into grep -q. Under `set -o pipefail`
    # the pipeline reports the rightmost non-zero status, and `grep -q` exits on
    # its first match -- which SIGPIPEs detect_system.sh, making the pipeline 141
    # instead of grep's 0. The negation would then read a *detected failure* as
    # success, i.e. choose submit_job on exactly the hosts where it cannot work.
    local out
    out=$(bash "$CI_DIR/detect_system.sh" 2>&1) || true
    [[ "$out" != *"Unable to determine target cluster"* ]]
}

if submit_job_usable; then
    BACKEND=submit_job
elif command -v sbatch >/dev/null 2>&1; then
    BACKEND=sbatch
else
    echo "ERROR: neither submit_job nor sbatch can submit from $(hostname)." >&2
    echo "       Run this script on a host that reaches slurmctld." >&2
    exit 1
fi

# Only used by the sbatch backend; submit_job sizes the allocation itself. The
# defaults match what this cluster actually hands out (an 8-GPU job gets 240 CPUs
# and 1878736 MB, i.e. 30 CPUs and 234842 MB per GPU).
#
# Asking for more than that per GPU is not merely large, it is rejected outright:
# slurm refuses any request whose memory-to-GPU ratio would strand GPUs on the
# node ("For N GPUs, please only request MAX ... RAM"), so an over-ask means the
# dispatcher never submits anything at all. 234842 MB is 229.4 GiB, so the ceiling
# in the GiB units --mem=<n>G speaks is 229 -- the 240 that stood here read the
# node's ~1970 GB as GiB and overshot by 4%.
CPUS_PER_GPU=${CPUS_PER_GPU:-28}
MEM_PER_GPU_GB=${MEM_PER_GPU_GB:-229}

pending_steps() {
    local d step
    # Step 0 is the model the run started from, recorded by the launcher as
    # bench_eval/base_model.txt. It has to appear here too: this is what decides
    # whether a job is submitted at all, so a baseline the job would happily score
    # would otherwise never get one.
    if [[ -f "$BENCH_DIR/base_model.txt" && ! -f "$BENCH_DIR/step-0.json" ]]; then
        echo 0
    fi
    for d in "$RUN_DIR"/checkpoint-*; do
        [[ -d "$d" ]] || continue
        step=${d##*checkpoint-}
        [[ "$step" =~ ^[0-9]+$ ]] || continue
        (( step % EVERY == 0 )) || continue
        [[ -f "$BENCH_DIR/step-$step.json" ]] && continue
        [[ -f "$d/adapter_config.json" && -s "$d/adapter_model.safetensors" ]] || continue
        echo "$step"
    done | sort -n
}

job_in_flight() {
    squeue -u "$USER" -h -o '%j' 2>/dev/null | grep -qF "$JOB_TOKEN"
}

# Second line of defence. If squeue is ever unreachable, or a job is rejected in a
# way that leaves nothing in the queue, the check above stops protecting anything --
# and the cost of getting this wrong is a queue full of duplicate eval jobs. A
# cooldown bounds that to one job per COOLDOWN seconds no matter what.
COOLDOWN=${COOLDOWN:-600}
last_submit=0
cooling_down() {
    local now; now=$(date +%s)
    (( now - last_submit < COOLDOWN ))
}

# No judge key is needed: every benchmark in the mini suites scores itself, which
# is why the three judged ones were left out (see eval_mini/benchmarks.py). A key
# is still forwarded if the environment happens to carry one, so that re-adding a
# judged benchmark does not also require remembering to plumb it through.
#
# It travels in this process's environment rather than inside the job script --
# sbatch propagates the submitting environment (--export=ALL is Slurm's default),
# and quoting a secret into a `bash -c '...'` string is how you get a value that
# silently truncates at the first space or quote. The previous form did not even
# parse: `${NVIDIA_API_KEY:+... ${OPENAI_API_KEY:-...} ...}` is not valid nesting,
# and the stray `}` left a bare `;}` that killed the job before it ran anything.
BENCH_JUDGE_KEY=${OPENAI_API_KEY:-${NVIDIA_API_KEY:-}}
[[ -n "$BENCH_JUDGE_KEY" ]] && export OPENAI_API_KEY="$BENCH_JUDGE_KEY"

INNER_CMD="bash $SCRIPT_DIR/run_bench_eval.sh \
        --run-dir $RUN_DIR --num-gpus $NUM_GPUS --every $EVERY --sample-n $SAMPLE_N"

submit_eval_job() {
    if [[ "$BACKEND" == "submit_job" ]]; then
        submit_job \
            --account "$ACCOUNT" \
            --partition "$PARTITION" \
            --name "$JOB_NAME" \
            --gpu "$NUM_GPUS" \
            --duration "$DURATION" \
            --outfile "$LOG_ROOT/${JOB_NAME}.%j.out" \
            --logroot "$LOG_ROOT" \
            -c "bash -c '$INNER_CMD'"
        return $?
    fi

    # No requeue trap here, unlike the eval-suite launcher: this job drains what is
    # pending and exits, so a wall-clock kill loses at most the checkpoint in
    # progress -- whose step file was never written, so the next submission simply
    # picks it up again.
    local script="$LOG_ROOT/${JOB_TOKEN}.sbatch"
    cat > "$script" <<SBATCH_EOF
#!/bin/bash
#SBATCH --account=$ACCOUNT
#SBATCH --partition=$PARTITION
#SBATCH --job-name=$JOB_NAME
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:$NUM_GPUS
#SBATCH --cpus-per-task=$((CPUS_PER_GPU * NUM_GPUS))
#SBATCH --mem=$((MEM_PER_GPU_GB * NUM_GPUS))G
#SBATCH --time=${DURATION}:00:00
#SBATCH --export=ALL
#SBATCH --output=$LOG_ROOT/${JOB_TOKEN}.%j.out

bash -c '$INNER_CMD'
SBATCH_EOF
    sbatch "$script"
}

echo "=========================================================================="
echo "Watching:   $RUN_DIR"
echo "Job name:   $JOB_NAME   ($NUM_GPUS GPUs, ${DURATION}h, $PARTITION)"
echo "Submitting: $BACKEND   (from $(hostname))"
echo "Benchmarks: $(python "$SCRIPT_DIR/eval_mini/make_mini_tasks.py" --print-tasks natural | tr ',' ' ' | wc -w) natural + $(python "$SCRIPT_DIR/eval_mini/make_mini_tasks.py" --print-tasks nonnatural | tr ',' ' ' | wc -w) non-natural, all self-scoring (no API key needed)"
echo "Cadence:    every $EVERY steps, $SAMPLE_N samples/benchmark"
echo "Results:    $BENCH_DIR/step-<N>.json"
echo "Poll:       every ${INTERVAL}s   $($ONCE && echo '(--once: one pass)' || echo '(Ctrl-C to stop)')"
echo "=========================================================================="

while true; do
    steps=$(pending_steps | tr '\n' ' ')
    if [[ -z "$steps" ]]; then
        status="nothing pending"
    elif job_in_flight; then
        status="pending [$steps] -- a job is already queued/running, not submitting another"
    elif cooling_down; then
        status="pending [$steps] -- submitted less than ${COOLDOWN}s ago, waiting"
    elif $DRY_RUN; then
        status="pending [$steps] -- would submit $JOB_NAME ($NUM_GPUS GPUs, $PARTITION, ${DURATION}h)"
    else
        echo "[$(date '+%F %T')] pending [$steps] -- submitting $JOB_NAME"
        if submit_eval_job; then
            last_submit=$(date +%s)
            status="submitted"
        else
            status="SUBMIT FAILED -- will retry next poll"
        fi
    fi
    echo "[$(date '+%F %T')] $status"

    $ONCE && break
    sleep "$INTERVAL"
done
