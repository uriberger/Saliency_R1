#!/bin/bash
# Watch a GRPO run's checkpoint directory and submit a 4-GPU eval job whenever
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
#   bash watch_bench_evals.sh --run-dir DIR [--num-gpus 4] [--every 100]
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
NUM_GPUS=4
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
RUN_DIR=$(cd "$RUN_DIR" && pwd)

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

# ADLR cluster-interface tools (submit_job) on PATH.
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
    echo "ERROR: submit_job not found. This host cannot submit jobs -- run this script" >&2
    echo "       somewhere that can (the login/vscode node)." >&2
    exit 1
}

pending_steps() {
    local d step
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
# and the cost of getting this wrong is a queue full of duplicate 4-GPU jobs. A
# cooldown bounds that to one job per COOLDOWN seconds no matter what.
COOLDOWN=${COOLDOWN:-600}
last_submit=0
cooling_down() {
    local now; now=$(date +%s)
    (( now - last_submit < COOLDOWN ))
}

submit_eval_job() {
    submit_job \
        --account "$ACCOUNT" \
        --partition "$PARTITION" \
        --name "$JOB_NAME" \
        --gpu "$NUM_GPUS" \
        --duration "$DURATION" \
        --outfile "$LOG_ROOT/${JOB_NAME}.%j.out" \
        --logroot "$LOG_ROOT" \
        -c "bash -c '
            ${HF_TOKEN:+export HF_TOKEN=$HF_TOKEN;}
            ${OPENAI_API_KEY:+export OPENAI_API_KEY=$OPENAI_API_KEY;}
            ${NVIDIA_API_KEY:+export OPENAI_API_KEY=\${OPENAI_API_KEY:-$NVIDIA_API_KEY};}
            bash $SCRIPT_DIR/run_bench_eval.sh \
                --run-dir $RUN_DIR --num-gpus $NUM_GPUS --every $EVERY --sample-n $SAMPLE_N
        '"
}

echo "=========================================================================="
echo "Watching:   $RUN_DIR"
echo "Job name:   $JOB_NAME   ($NUM_GPUS GPUs, ${DURATION}h, $PARTITION)"
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
