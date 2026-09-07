#!/usr/bin/env bash
# Submit launch_sink_shift.sh to SLURM. The launcher it wraps assumes it is already ON a
# node with GPUs -- it fans one shard per visible device and waits -- so this is the
# `_job.sh` half, the same split as launch_overlap_probe_job.sh / launch_overlap_probe.sh.
#
#   bash launch_sink_shift_job.sh --name <jobname> --model <path> \
#       [--gpus 8] [--duration 1] [--stages survey,selftest,run] [--out-dir <dir>] \
#       [--dry-run] [-- <anything else, forwarded verbatim to launch_sink_shift.sh>]
#
# The three stages run in sequence inside one allocation, because `run` is gated on a
# passing selftest and the gate reads a log the selftest writes into the out-dir.
#
# RESUME IS THE POINT OF THE 1-HOUR DEFAULT. `run` is append-only JSONL keyed by
# (split, arm, alpha, row_index), so a job killed at the wall clock loses only the row in
# flight. Re-submit the identical command and it picks up where it stopped -- and the
# second submission SKIPS survey and selftest, because their outputs are already in the
# out-dir, so it spends the whole hour generating. One hour also buys batch_short, which
# starts in about a minute where the longer pools queue for hours (see cluster_env.sh).
#
# WHAT FITS IN AN HOUR, at ~6 s a row narrow / ~10 s broad on 8 GPUs, minus ~10 minutes
# for the survey and selftest on the FIRST submission:
#
#   scope trained, 128 rows/split, 4 arms x 2 alphas   ~35 min   fits, first try
#   scope all,     128 rows/split, 4 arms x 2 alphas   ~55 min   fits on the resume
#   either,        256 rows/split, 5 arms x 4 alphas   2.5-4 h   3 to 5 submissions
#
# Everything after `--` goes to the inner launcher untouched, which is where the probe's
# own flags live (--scope, --arms, --alphas, --rows-per-split, --best-of, ...).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

NAME=""
MODEL=""
GPUS=8
DURATION=1
STAGES="survey,selftest,run"
OUT_DIR=""
DRY_RUN=0
FORCE_STAGES=0
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)       DRY_RUN=1;        shift   ;;
        --force-stages)  FORCE_STAGES=1;   shift   ;;
        --name)          NAME="$2";        shift 2 ;;
        --model)         MODEL="$2";       shift 2 ;;
        --gpus)          GPUS="$2";        shift 2 ;;
        --duration)      DURATION="$2";    shift 2 ;;
        --stages)        STAGES="$2";      shift 2 ;;
        --out-dir)       OUT_DIR="$2";     shift 2 ;;
        --)              shift; EXTRA+=("$@"); break ;;
        *)               EXTRA+=("$1");    shift   ;;
    esac
done

[[ -n "$NAME" ]]  || { echo "ERROR: --name is required (it names the job and the log)." >&2; exit 2; }
[[ -n "$MODEL" ]] || { echo "ERROR: --model is required." >&2; exit 2; }
[[ -d "$MODEL" || -f "$MODEL/config.json" ]] || echo "WARNING: --model '$MODEL' is not a directory here; assuming the node can see it." >&2

# WHICH TREE THE JOB RUNS FROM. A job outlives the worktree it was submitted from --
# ./worktree.sh done deletes the directory while the job is still queued or running, and
# the runner would then point at nothing. So when this script lives in a worktree, the
# runner is pinned to the central tree instead, and says so. Override with
# SINK_SHIFT_RUN_REPO when the worktree's own copy is the one that must run (a launcher
# change that is not merged yet), and then do not merge until the job has finished.
RUN_REPO="$REPO"
if [[ "$REPO" == */.worktrees/* ]]; then
    RUN_REPO=$(cd "$REPO/../.." && pwd)
    echo "NOTE: submitted from a worktree; the job will run from the central tree" >&2
    echo "      $RUN_REPO" >&2
    echo "      (set SINK_SHIFT_RUN_REPO to override -- see the comment in this file)" >&2
fi
RUN_REPO=${SINK_SHIFT_RUN_REPO:-$RUN_REPO}
[[ -f "$RUN_REPO/launch_sink_shift.sh" ]] || {
    echo "ERROR: no launch_sink_shift.sh under $RUN_REPO" >&2; exit 2; }

OUT_DIR=${OUT_DIR:-$RUN_REPO/outputs/sink_shift/$NAME}

# Partition list, filtered for what this cluster has and what a $DURATION-hour job is
# eligible for. Same helper every other launcher here uses.
# shellcheck source=/dev/null
source "$REPO/cluster_env.sh"
PARTITION=${PARTITION:-$(SR1_JOB_HOURS=$DURATION sr1_pick_partition)}
ACCOUNT=${ACCOUNT:-nvr_israel_rlop}
CONDA_ENV=${CONDA_ENV:-saliency_r1_qwen3_vllm}

if ! command -v submit_job >/dev/null 2>&1; then
    for CI_ROOT in \
        /lustre/fs1/portfolios/adlr/projects/adlr_other_infra/release/cluster-interface \
        /lustre/fsw/portfolios/adlr/projects/adlr_other_infra/release/cluster-interface; do
        for CAND in "$CI_ROOT/latest" $(ls -1dt "$CI_ROOT"/*/ 2>/dev/null); do
            if [ -x "${CAND%/}/submit_job" ]; then export PATH="${CAND%/}:$PATH"; break 2; fi
        done
    done
fi
command -v submit_job >/dev/null 2>&1 || [[ $DRY_RUN -eq 1 ]] || {
    echo "ERROR: submit_job not found under the cluster-interface paths." >&2; exit 1; }

LOG_ROOT="$RUN_REPO/outputs/logs"
mkdir -p "$LOG_ROOT" "$OUT_DIR"

# The inner command as a file rather than a quoted -c string: the probe's flags are
# comma-separated lists (--arms centre,outward --alphas 0.5,1.0) and nesting those inside
# `bash -c '...'` is how an arm silently goes missing. Same reason
# launch_overlap_probe_job.sh writes a runner.
RUNNER="$LOG_ROOT/$NAME.runner.sh"
{
    echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    printf 'export CONDA_ENV=%q\n' "$CONDA_ENV"
    printf 'export HF_HOME=%q\n' "${HF_HOME:-/home/uberger/scratch/cache/hf_cache}"
    printf 'export HF_HUB_OFFLINE=%q\n' "${HF_HUB_OFFLINE:-1}"
    printf 'REPO=%q\n' "$RUN_REPO"
    printf 'OUT=%q\n' "$OUT_DIR"
    printf 'MODEL=%q\n' "$MODEL"
    printf 'GPUS=%q\n' "$GPUS"
    printf 'FORCE=%q\n' "$FORCE_STAGES"
    echo 'EXTRA=()'
    for a in ${EXTRA[@]+"${EXTRA[@]}"}; do printf 'EXTRA+=(%q)\n' "$a"; done
    echo 'cd "$REPO"'
    echo ''
    echo 'run_stage() {'
    echo '    local stage="$1" gpus="$2"; shift 2'
    echo '    echo "=== stage $stage ($(date -Is)) ==="'
    echo '    bash "$REPO/launch_sink_shift.sh" --stage "$stage" --gpus "$gpus" \'
    echo '        --out-dir "$OUT" --model "$MODEL" "$@" ${EXTRA[@]+"${EXTRA[@]}"}'
    echo '}'
    echo ''
    # Skipping an already-finished stage is what makes a re-submission spend its whole
    # hour on the part that is not done. --force-stages runs them anyway.
    for stage in ${STAGES//,/ }; do
        case "$stage" in
            survey)
                echo 'if [[ $FORCE -eq 0 && -s "$OUT/survey.json" ]]; then'
                echo '    echo "=== stage survey: already done, skipping ==="'
                echo 'else'
                echo '    run_stage survey 1'
                echo 'fi'
                ;;
            selftest)
                echo 'if [[ $FORCE -eq 0 ]] && grep -q "SELFTEST PASS" "$OUT/logs/selftest.log" 2>/dev/null; then'
                echo '    echo "=== stage selftest: already passed, skipping ==="'
                echo 'else'
                echo '    run_stage selftest 1'
                echo 'fi'
                ;;
            run)
                # The gate is repeated here rather than left to the inner launcher, so a
                # job that will refuse to run says so in its first seconds instead of
                # after two model loads.
                echo 'grep -q "SELFTEST PASS" "$OUT/logs/selftest.log" 2>/dev/null || {'
                echo '    echo "no passing selftest in $OUT/logs/selftest.log -- refusing to run" >&2; exit 2; }'
                echo 'run_stage run "$GPUS"'
                ;;
            report)
                echo 'run_stage report 1'
                ;;
            *)
                echo "ERROR: unknown stage '$stage'" >&2; exit 2 ;;
        esac
    done
    echo 'echo "=== all stages finished ($(date -Is)) ==="'
} > "$RUNNER"
chmod +x "$RUNNER"

echo "=========================================================================="
echo "Job       : $NAME   ($ACCOUNT, $PARTITION, ${DURATION}h, ${GPUS} GPU)"
echo "Run repo  : $RUN_REPO"
echo "Out dir   : $OUT_DIR"
echo "Model     : $MODEL"
echo "Stages    : $STAGES$( [[ $FORCE_STAGES -eq 1 ]] && echo ' (forced)')"
echo "Extra     : ${EXTRA[*]:-(none)}"
echo "Runner    : $RUNNER"
echo "Log       : $LOG_ROOT/$NAME.<jobid>.out"
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
