#!/bin/bash
# Score a named list of checkpoints from one GRPO run as a SLURM job, chained
# across as many allocations as the list needs.
#
#   bash launch_bench_eval_steps.sh                          # detect the steps, submit
#   bash launch_bench_eval_steps.sh --run <run-name> --duration 1
#   bash launch_bench_eval_steps.sh --steps "3990 2000 900"  # exactly these, in this order
#   bash launch_bench_eval_steps.sh --dry-run                # print the plan, submit nothing
#
# This is the non-direct counterpart of run_bench_eval_steps.sh: the same list,
# the same recipe, the same results, run on an allocation this script asks for
# instead of on the node you are standing on. Everything it knows about the run --
# which steps are owed, which WandB run they belong to, what a checkpoint costs --
# is read by run_bench_eval_steps.sh itself, here on the login node with no GPUs
# held, and printed before anything is submitted.
#
# THE POINT OF IT IS THE CHAIN. One hour is not enough for a list of any length:
# at 8 GPUs and 300 documents a checkpoint is ~38 minutes, so an allocation
# finishes one and part of the next. What makes that useful rather than futile is
# that every intermediate result is on /lustre the moment it is earned -- a
# benchmark suite as a banked unit, a checkpoint as a step file, a merged model as
# a directory with a .complete sentinel. A job killed at the wall clock loses the
# unit it was in the middle of and nothing else, and the next job resumes at that
# unit rather than at the beginning of the checkpoint.
#
# Two ways to get the next job, and the default needs nothing running anywhere:
#
#   autoresume (default)   submit_job --autoresume_uninstrumented. The job is
#                          killed 3 minutes before its time limit and requeued,
#                          and requeued again on any non-zero exit
#                          (--autoresume_ignore_failure). run_bench_eval_steps.sh
#                          exits 17 while steps are owed and 0 when they are not,
#                          so the chain runs exactly as long as there is work and
#                          then stops by itself. Needs submit_job, which only
#                          resolves on the login node.
#
#   --chain N              N allocations submitted up front, each depending on the
#                          previous one with afterany. Fixed length: if the list
#                          outlives it, run this again; if the list finishes early,
#                          the leftover jobs start, find nothing pending and exit
#                          in seconds. This is the only mode the sbatch backend
#                          has, and sbatch is the only backend that works from a
#                          compute node.
#
# Neither mode holds a process anywhere: unlike watch_bench_evals.sh there is no
# poll loop to keep alive in a tmux window. The state that drives all of it is the
# set of files under bench_eval/, which is also what makes stopping the chain
# (scancel) and restarting it (run this again) cost nothing.
#
# Environment:
#   WANDB_API_KEY   to push each step to the run's curve as it is scored
#   HF_TOKEN, PARTITION, REPO
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")" && pwd)
REPO=${REPO:-/home/uberger/scratch/research/saliency_r1}
source "$SCRIPT_DIR/cluster_env.sh"

ACCOUNT=nvr_israel_rlop
HF_HOME=${HF_HOME:-/home/uberger/scratch/cache/hf_cache}
LOG_ROOT="$REPO/outputs/logs"

# Same default as run_bench_eval_steps.sh, and the only thing that normally has to
# change to point this at another run.
RUN=${RUN:-grpo-coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged-overlap__wov0.11_2head_trmean_saliency_r1_8k_auroc}
STEPS=${STEPS:-}
NUM_GPUS=${NUM_GPUS:-8}
# One hour, and not because it is generous. It is the length that actually starts:
# batch_short caps at MaxTime=2h and carries PriorityTier=40, the highest open to
# this account, so a 1h request is eligible for it while a 4h one is not. See
# SR1_JOB_HOURS in cluster_env.sh. Raising this past 2 gives batch_short up, which
# is usually a worse trade than running one more link of the chain.
DURATION=${DURATION:-1}
CHAIN=0
NATURAL_N=""
NONNATURAL_N=""
declare -a TASK_N=()
BANK=""
WANDB_RUN_ID=${WANDB_RUN_ID:-}
FORCE=false
FORCE_SUBMIT=false
DRY_RUN=false
PARTITION=${PARTITION:-}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run)          RUN="$2";          shift 2 ;;
        --steps)        STEPS="$2";        shift 2 ;;
        --num-gpus)     NUM_GPUS="$2";     shift 2 ;;
        --duration)     DURATION="$2";     shift 2 ;;
        # N allocations up front instead of an autoresume chain. See the header.
        --chain)        CHAIN="$2";        shift 2 ;;
        --natural-n)    NATURAL_N="$2";    shift 2 ;;
        --nonnatural-n) NONNATURAL_N="$2"; shift 2 ;;
        --task-n)       TASK_N+=("$2");    shift 2 ;;
        --bank)         BANK="$2";         shift 2 ;;
        --wandb-run-id) WANDB_RUN_ID="$2"; shift 2 ;;
        # --force re-scores steps that already have a result; --force-submit
        # ignores a chain that is already in the queue. Different questions.
        --force)        FORCE=true;        shift ;;
        --force-submit) FORCE_SUBMIT=true; shift ;;
        --partition)    PARTITION="$2";    shift 2 ;;
        --dry-run)      DRY_RUN=true;      shift ;;
        -h|--help)      sed -n '2,52p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

# After the loop: which partitions can hold the job depends on how long it asks
# for, and batch_short is in the list at 1-2h and out of it at 4.
PARTITION=${PARTITION:-$(SR1_JOB_HOURS=$DURATION sr1_pick_partition)}

RUN_DIR="$REPO/checkpoint/$RUN"
[[ -d "$RUN_DIR" ]] || { echo "error: no such run dir: $RUN_DIR" >&2; exit 2; }
mkdir -p "$LOG_ROOT"

JOB_MINUTES=$(( DURATION * 60 ))

# Everything the job will be told, assembled once so the preview below and the
# submitted command cannot describe different work.
declare -a STEPS_ARGS=(--run "$RUN" --num-gpus "$NUM_GPUS" --job-minutes "$JOB_MINUTES")
[[ -n "$STEPS" ]]        && STEPS_ARGS+=(--steps "$STEPS")
[[ -n "$NATURAL_N" ]]    && STEPS_ARGS+=(--natural-n "$NATURAL_N")
[[ -n "$NONNATURAL_N" ]] && STEPS_ARGS+=(--nonnatural-n "$NONNATURAL_N")
for t in ${TASK_N[@]+"${TASK_N[@]}"}; do STEPS_ARGS+=(--task-n "$t"); done
[[ -n "$BANK" ]]         && STEPS_ARGS+=(--bank "$BANK")
[[ -n "$WANDB_RUN_ID" ]] && STEPS_ARGS+=(--wandb-run-id "$WANDB_RUN_ID")
$FORCE                   && STEPS_ARGS+=(--force)

# ---------- what there is to do ----------
# Asked of the script that will do it, rather than worked out again here. It reads
# the pending steps, the sample profile and the WandB run off the run directory in
# a second or two and holds no GPUs to do it, so there is no reason for this
# launcher to own a second copy of that logic -- and every reason not to, since a
# copy that drifts submits jobs for work that does not exist.
PREVIEW=$(bash "$SCRIPT_DIR/run_bench_eval_steps.sh" "${STEPS_ARGS[@]}" --dry-run 2>&1)
PREVIEW_STATUS=$?
echo "$PREVIEW"
if (( PREVIEW_STATUS != 0 )); then
    echo "" >&2
    echo "error: the plan above could not be read -- not submitting anything" >&2
    exit 1
fi

# The machine-readable line run_bench_eval_steps.sh prints under --dry-run. Keep
# the two in step: it is a contract between these two files and nothing else.
PLAN_LINE=$(grep -m1 '^PLAN ' <<< "$PREVIEW")
PENDING=$(sed -n 's/.*[ ]steps=\([0-9]*\).*/\1/p' <<< "$PLAN_LINE")
JOBS_NEEDED=$(sed -n 's/.*[ ]jobs_needed=\([0-9]*\).*/\1/p' <<< "$PLAN_LINE")
CHAIN_STATE=$(sed -n 's/.*[ ]chain_state=\([^ ]*\).*/\1/p' <<< "$PLAN_LINE")
PENDING=${PENDING:-0}
JOBS_NEEDED=${JOBS_NEEDED:-0}
(( PENDING > 0 )) || { echo ""; echo "Nothing to submit."; exit 0; }

# ---------- one chain per run and profile ----------
# submit_job rewrites the name it is given (dots to underscores, a timestamp
# appended), so the guard cannot be an exact name match. It looks for a token that
# no such rewriting touches, derived from the run directory and the sample sizes,
# and distinct from watch_bench_evals.sh's `bencheval` prefix: that dispatcher and
# this chain would otherwise be invisible to each other while working the same
# checkpoints from opposite ends of the list.
TOKEN_INPUT="$RUN_DIR@${NATURAL_N:-default}_${NONNATURAL_N:-default}_${TASK_N[*]:-}"
JOB_TOKEN="benchsteps$(printf '%s' "$TOKEN_INPUT" | md5sum | cut -c1-8)"
JOB_NAME="${JOB_TOKEN}_${RUN}"

IN_FLIGHT=$(squeue -u "$USER" -h -o '%j' 2>/dev/null | grep -cF "$JOB_TOKEN")
if (( IN_FLIGHT > 0 )) && ! $FORCE_SUBMIT; then
    echo "" >&2
    echo "error: $IN_FLIGHT job(s) carrying $JOB_TOKEN are already queued or running." >&2
    echo "" >&2
    echo "Two chains over one run is not merely wasteful, it corrupts: both would pick the" >&2
    echo "same furthest-first step, merge it into the same directory under" >&2
    echo "checkpoint/_bench_eval, and evaluate whichever half-written model the other one" >&2
    echo "left. Wait for the chain to end, or stop it with" >&2
    echo "" >&2
    echo "    scancel -n $JOB_NAME" >&2
    echo "" >&2
    echo "and run this again. --force-submit overrides, for the case where you know the" >&2
    echo "queued jobs are for a different list." >&2
    # Under --dry-run this is something to know about, not a reason to stop
    # printing the plan.
    $DRY_RUN || exit 1
fi

# ---------- how to submit ----------
# submit_job resolves through detect_system.sh, whose host list has no compute
# nodes on it, so its presence on PATH proves nothing -- ask it. sbatch works
# anywhere that reaches slurmctld, but has no autoresume, so from a compute node
# only --chain is available.
#
# Captured and matched rather than piped into `grep -q`: under `set -o pipefail`
# grep exits on its first match, SIGPIPEs detect_system.sh, and the pipeline
# reports 141 -- so the negation would read a DETECTED FAILURE as success and pick
# submit_job on exactly the hosts where it cannot work. It also reports the
# failure on stdout and exits 0, so the status says nothing either way.
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
if [[ "$BACKEND" == sbatch ]] && (( CHAIN == 0 )); then
    CHAIN=$JOBS_NEEDED
    (( CHAIN < 1 )) && CHAIN=1
    CHAIN_REASON="   (sbatch has no autoresume, so the chain length is fixed here)"
fi
[[ "$BACKEND" == sbatch ]] && ! command -v sbatch >/dev/null 2>&1 && {
    echo "ERROR: neither submit_job nor sbatch can submit from $(hostname)." >&2
    echo "       Run run_bench_eval_steps.sh directly instead, inside an allocation you hold." >&2
    exit 1
}

echo ""
echo "=========================================================================="
echo "Submitting: $JOB_NAME"
echo "Allocation: $NUM_GPUS GPUs, ${DURATION}h, $PARTITION   (via $BACKEND)"
if (( CHAIN > 0 )); then
    echo "Chain:      $CHAIN allocation(s), each depending on the one before${CHAIN_REASON:-}"
    (( CHAIN < JOBS_NEEDED )) && \
        echo "            the list needs ~$JOBS_NEEDED -- run this again when they are spent"
else
    echo "Chain:      autoresume -- requeued at the wall clock and on exit 17, and stopped"
    echo "            the first time the list comes back complete (~$JOBS_NEEDED jobs)"
fi
echo "Logs:       $LOG_ROOT/${JOB_TOKEN}.*.out"
[[ -z "${WANDB_API_KEY:-}" ]] && \
    echo "NOTE:       no WANDB_API_KEY -- steps will be scored but not pushed; ship them"
[[ -z "${WANDB_API_KEY:-}" ]] && \
    echo "            afterwards with run_bench_eval_steps.sh --push-only"
echo "=========================================================================="

$DRY_RUN && { echo "--dry-run: nothing submitted."; exit 0; }

# A fresh chain starts with a clean slate: the no-progress counter left behind by
# a chain that gave up would otherwise stop this one after a single link.
[[ -n "$CHAIN_STATE" ]] && rm -f "$CHAIN_STATE"

# The command travels as one string, through `submit_job -c "bash -c '...'"` for
# one backend and through a heredoc for the other, so it is wrapped in SINGLE
# quotes by the time a shell on the node parses it. Every argument inside is
# therefore quoted with DOUBLE quotes -- `${STEPS_ARGS[*]@Q}` would emit single
# ones and close the wrapper at the first step list. Nothing that reaches here is
# supposed to contain a quote, a dollar or a backslash (they are run names, step
# numbers and benchmark sizes), so anything that does is a mistake worth stopping
# for rather than quoting around.
quoted_args() {
    local a
    for a in "$@"; do
        case "$a" in
            *[\'\"\$\\\`]*)
                echo "error: cannot pass $a through the job command -- it contains a" >&2
                echo "       quote, a dollar or a backslash" >&2
                return 1 ;;
        esac
        printf '"%s" ' "$a"
    done
}
STEPS_ARGS_Q=$(quoted_args "${STEPS_ARGS[@]}") || exit 2

# The environment the job needs, exported inside the command rather than relied on
# to be inherited: submit_job runs it in a container, and a secret quoted into a
# `bash -c '...'` string is how you get a value that truncates at the first space.
INNER_CMD="export HF_HOME=$HF_HOME;
    export WANDB_API_KEY=${WANDB_API_KEY:-};
    ${HF_TOKEN:+export HF_TOKEN=$HF_TOKEN;}
    bash $SCRIPT_DIR/run_bench_eval_steps.sh $STEPS_ARGS_Q"

if [[ "$BACKEND" == submit_job ]]; then
    declare -a CHAIN_ARGS=(--autoresume_uninstrumented --autoresume_ignore_failure)
    # --dependent_clones N submits N MORE jobs after this one, each afterany on the
    # previous, so a chain of N is this job plus N-1 clones. Autoresume is left out
    # of that mode on purpose: the flags are copied into every clone, and a clone
    # that also requeues itself turns a fixed chain into an open-ended one.
    (( CHAIN > 0 )) && CHAIN_ARGS=(--dependent_clones $(( CHAIN - 1 )))
    submit_job \
        --account "$ACCOUNT" \
        --partition "$PARTITION" \
        --name "$JOB_NAME" \
        --gpu "$NUM_GPUS" \
        --duration "$DURATION" \
        "${CHAIN_ARGS[@]}" \
        --outfile "$LOG_ROOT/${JOB_TOKEN}.%j.out" \
        --logroot "$LOG_ROOT" \
        -c "bash -c '$INNER_CMD'"
    exit $?
fi

# ---------- sbatch ----------
# The same allocation shape the dispatcher asks for: this cluster hands an 8-GPU
# job 240 CPUs and 1878736 MB, and a request whose memory-to-GPU ratio would
# strand GPUs on the node is rejected outright rather than merely trimmed.
CPUS_PER_GPU=${CPUS_PER_GPU:-28}
MEM_PER_GPU_GB=${MEM_PER_GPU_GB:-229}

SBATCH_FILE="$LOG_ROOT/${JOB_TOKEN}.sbatch"
cat > "$SBATCH_FILE" <<SBATCH_EOF
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

# afterany, not afterok: a link that exits 17 has done its job and said there is
# more to do, and afterok would read that as a reason to abandon the rest of the
# chain -- which is exactly backwards.
PREV=""
for (( i = 0; i < CHAIN; i++ )); do
    if [[ -n "$PREV" ]]; then
        OUT=$(sbatch --dependency="afterany:$PREV" "$SBATCH_FILE")
    else
        OUT=$(sbatch "$SBATCH_FILE")
    fi
    status=$?
    echo "$OUT"
    (( status == 0 )) || { echo "submit failed at link $(( i + 1 ))/$CHAIN" >&2; exit 1; }
    PREV=$(awk '{print $NF}' <<< "$OUT")
done
