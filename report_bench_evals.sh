#!/bin/bash
# Report on the eval jobs watch_bench_evals.sh has in flight: which run each one
# belongs to, which checkpoint it is scoring right now, and how far into it it is.
#
#   bash report_bench_evals.sh [--watch 60] [--every 100]
#
# The job name is the only thing squeue knows about an eval job, and it does not
# contain a usable run path: the dispatcher builds it as
# bencheval<md5(RUN_DIR)>_<run name>, and submit_job then rewrites the tail --
# dots become underscores and a _<date>-<time> stamp is appended. So the name's
# tail cannot be turned back into a directory, and this script maps the md5 token
# instead, hashing every checkpoint/<run>/ the same way the dispatcher does.
# That works for a job that is still PENDING, before any log exists.
#
# Everything else comes from the job's log and the run's bench_eval/ directory:
#   [step N]                     the checkpoint being worked on
#   r1_<suite>                   which of the three suites is running
#   bench_eval/partial/step-N/   the suites already banked for it
#   Model Responding: X/Y        progress inside the current suite
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")" && pwd)
REPO=${REPO:-/home/uberger/scratch/research/saliency_r1}
LOG_ROOT="$REPO/outputs/logs"
CKPT_ROOT="$REPO/checkpoint"

# Only used to say what is still owed; the dispatcher's own --every is not
# recorded anywhere, so this is the default it is almost always run with.
EVERY=100
WATCH=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --every) EVERY="$2"; shift 2 ;;
        --watch) WATCH="$2";  shift 2 ;;
        --repo)  REPO="$2"; LOG_ROOT="$REPO/outputs/logs"; CKPT_ROOT="$REPO/checkpoint"; shift 2 ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

# ---------- md5 token -> run directory ----------
# Hash the physical path, as the dispatcher does (it resolves RUN_DIR with
# `pwd -P` first): /home/uberger/scratch/research/saliency_r1 is a symlink to the
# /lustre spelling, and the two hash differently.
declare -A RUN_OF_TOKEN=()
build_token_map() {
    local d real token
    for d in "$CKPT_ROOT"/*/; do
        [[ -d "$d" ]] || continue
        real=$(cd "$d" && pwd -P) || continue
        token=$(printf '%s' "$real" | md5sum | cut -c1-8)
        RUN_OF_TOKEN[$token]="$real"
    done
}

# ---------- what a run still owes ----------
scored_steps() {
    local bench="$1/bench_eval" f
    for f in "$bench"/step-*.json; do
        [[ -f "$f" ]] || continue
        f=${f##*/step-}; echo "${f%.json}"
    done | sort -n
}

pending_steps() {
    local run_dir="$1" bench="$1/bench_eval" d step
    [[ -f "$bench/base_model.txt" && ! -f "$bench/step-0.json" ]] && echo 0
    for d in "$run_dir"/checkpoint-*; do
        [[ -d "$d" ]] || continue
        step=${d##*checkpoint-}
        [[ "$step" =~ ^[0-9]+$ ]] || continue
        (( step % EVERY == 0 )) || continue
        [[ -f "$bench/step-$step.json" ]] && continue
        [[ -f "$d/adapter_config.json" && -s "$d/adapter_model.safetensors" ]] || continue
        echo "$step"
    done | sort -n
}

# ---------- the log ----------
# Both backends embed the job id before .out -- sbatch as <token>.<id>.out,
# submit_job as <job name>.<id>.out -- so one glob finds either.
log_for_job() { ls -t "$LOG_ROOT"/*."$1".out 2>/dev/null | head -1; }

report_one() {
    local jobid="$1" name="$2" state="$3" elapsed="$4" left="$5" node="$6" reason="$7"

    local token=""
    [[ "$name" =~ ^bencheval([0-9a-f]{8}) ]] && token="${BASH_REMATCH[1]}"
    local run_dir="${RUN_OF_TOKEN[$token]:-}"
    local log; log=$(log_for_job "$jobid")
    # Fall back to the log if the token matches no checkpoint directory -- which
    # happens for a run that has since been renamed or moved away.
    if [[ -z "$run_dir" && -n "$log" ]]; then
        run_dir=$(grep -am1 '^Run dir:' "$log" | awk '{print $3}')
    fi
    local run_name=${run_dir:+$(basename "$run_dir")}

    echo "job $jobid  [$state]  ${elapsed} elapsed, ${left} left  ${node:-$reason}"
    echo "  run:      ${run_name:-<unknown: token $token>}"

    if [[ -z "$log" ]]; then
        echo "  step:     not started yet (no log; queued as ${reason:-pending})"
    else
        # The last [step N] line is the checkpoint in hand: the merge (or the
        # baseline banner) prints one as soon as a checkpoint is entered, so this
        # never lags behind what is actually being evaluated.
        local step; step=$(grep -aoE '^\[step [0-9]+\]' "$log" | tail -1 | tr -dc '0-9')
        local suite; suite=$(grep -aoE 'r1_(natural|mmerw|nonnatural)' "$log" | tail -1)
        suite=${suite#r1_}
        # Progress is written with carriage returns, so the whole bar is one
        # "line"; splitting on \r is what makes the last update visible. Only the
        # tail of the log is scanned -- these files reach tens of MB.
        local prog; prog=$(tail -c 200000 "$log" | tr '\r' '\n' |
                           grep -aoE 'Model Responding: +[0-9]+%\|[^|]*\| *[0-9]+/[0-9]+ \[[^]]*\]' | tail -1)
        prog=${prog//|/ }; prog=$(echo "$prog" | tr -s ' ')

        if [[ -z "$step" ]]; then
            echo "  step:     starting up (merging/loading, nothing scored yet)"
        else
            local banked=""
            [[ -n "$run_dir" ]] && banked=$(ls "$run_dir/bench_eval/partial/step-$step" 2>/dev/null |
                                            sed 's/\.json$//' | tr '\n' ' ')
            echo "  step:     $step   suite: ${suite:-<not started>}   banked: ${banked:-none}"
            [[ -n "$prog" ]] && echo "  progress: $prog"
        fi
        # A finished or stalled job says so in its own words; worth surfacing.
        local last_drain; last_drain=$(grep -a '^\[drain\]' "$log" | tail -1)
        [[ -n "$last_drain" ]] && echo "  drain:    ${last_drain#\[drain\] }"
    fi

    if [[ -n "$run_dir" ]]; then
        local done_steps pending
        done_steps=$(scored_steps "$run_dir" | tr '\n' ' ')
        pending=$(pending_steps "$run_dir" | tr '\n' ' ')
        echo "  scored:   ${done_steps:-none}"
        echo "  pending:  ${pending:-none}"
    fi
    echo ""
}

# Runs the dispatcher has work for but nothing in flight. Either no watcher is
# running for them, or one is and it is in its cooldown -- both worth seeing,
# since a run silently going unevaluated looks exactly like one that is up to date.
report_idle_runs() {
    local d run_dir pending header=false
    for d in "$CKPT_ROOT"/*/; do
        [[ -d "$d/bench_eval" ]] || continue
        run_dir=$(cd "$d" && pwd -P)
        # ${a[@]+...} rather than a bare "${a[@]}": under `set -u` an empty array
        # is an unbound expansion on older bash, i.e. exactly the no-jobs case.
        printf '%s\n' ${IN_FLIGHT[@]+"${IN_FLIGHT[@]}"} | grep -qxF "$run_dir" && continue
        pending=$(pending_steps "$run_dir" | tr '\n' ' ')
        [[ -n "$pending" ]] || continue
        $header || { echo "Pending, nothing in flight:"; header=true; }
        printf '  %-70s %s\n' "$(basename "$run_dir")" "[$pending]"
    done
    $header && echo ""
}

report() {
    build_token_map
    IN_FLIGHT=()
    local found=false line jobid name state elapsed left node reason token
    echo "=========================================================================="
    echo "bench evals for $USER   $(date '+%F %T')"
    echo "=========================================================================="
    while IFS='|' read -r jobid name state elapsed left node reason; do
        [[ "$name" == bencheval* ]] || continue
        found=true
        [[ "$name" =~ ^bencheval([0-9a-f]{8}) ]] && token="${BASH_REMATCH[1]}" || token=""
        [[ -n "${RUN_OF_TOKEN[$token]:-}" ]] && IN_FLIGHT+=("${RUN_OF_TOKEN[$token]}")
        report_one "$jobid" "$name" "$state" "$elapsed" "$left" "$node" "$reason"
    done < <(squeue -u "$USER" -h -o '%i|%j|%T|%M|%L|%N|%R' 2>/dev/null)
    $found || echo "no eval jobs queued or running"$'\n'
    report_idle_runs
}

if (( WATCH > 0 )); then
    while true; do clear; report; sleep "$WATCH"; done
else
    report
fi
