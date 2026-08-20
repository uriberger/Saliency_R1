#!/bin/bash
# How likely am I to get a NON-INTERACTIVE GPU allocation on this cluster?
#
# Run it on each cluster you are choosing between and diff the outputs. It is READ-ONLY:
# it submits nothing. The only thing it asks the scheduler to do is `sbatch --test-only`,
# which validates a job and reports when it WOULD start without ever queueing it.
#
#   bash cluster_capacity_report.sh                              # sensible defaults
#   bash cluster_capacity_report.sh --gpus 8,4 --hours 4         # the shapes you care about
#   bash cluster_capacity_report.sh --account nvr_israel_rlop --days 14 --allhours 24
#
# Standalone on purpose -- no repo, no conda, no python, no gawk extensions. scp it,
# paste it, whatever is easiest.
#
# WHAT TO READ, in the order it decides the answer:
#
#   [5b] CLUSTER-WIDE WAITS is the one to read first on a cluster you have never used.
#        It is every user's GPU job over the last --allhours: how long they actually
#        waited, split by partition and by GPUs-per-job. No opinion, no simulation.
#   [4]  THE SCHEDULER'S OWN ANSWER: for each (partition x GPUs x hours), when SLURM
#        says a job submitted right now would start. Decisive about PARTITIONS. Read the
#        caveat printed under it before reading anything into the GPU-count column.
#   [3]  How deep the queue you would join is, and how long the jobs already in it have
#        been waiting. A partition whose pending 8-GPU backlog has a mean age in DAYS is
#        a partition you do not get into, whatever [4] says.
#   [5]  YOUR history, if you have any here. Empty is informative: no fair-share debt.
#   [6]  Why [5] looks the way it does. Fair-share falls as you consume, so interactive
#        shells you are holding right now depress every batch job you submit afterwards.
#
# ONE TRAP, and it is the reason this script prints GPUs-per-node so prominently. If every
# node has 8 GPUs, a 4-GPU request is HALF A NODE: too large to fit the gaps a 1-GPU job
# slides into, too small to be handed a whole node that just freed. On such a cluster 4
# can be far harder to schedule than either 8 or 1, and no amount of "there are idle GPUs"
# changes that. Section [1] tells you which regime you are in.

set -uo pipefail

ACCOUNT="${SLURM_ACCOUNT:-}"
DAYS=14           # how far back to look at YOUR jobs
ALLHOURS=24       # how far back to look at EVERYONE's jobs (cluster-wide waits)
GPUS="8,4,1"
HOURS="4"
USERNAME="${USER:-$(id -un)}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --account)  ACCOUNT="$2";  shift 2 ;;
        --user)     USERNAME="$2"; shift 2 ;;
        --days)     DAYS="$2";     shift 2 ;;
        --allhours) ALLHOURS="$2"; shift 2 ;;
        --gpus)     GPUS="$2";     shift 2 ;;
        --hours)    HOURS="$2";    shift 2 ;;
        -h|--help)  sed -n '2,36p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done

command -v sinfo >/dev/null 2>&1 || { echo "no SLURM here (sinfo not found)" >&2; exit 1; }

# Default the account to whatever the association says, so the script needs no arguments
# on a cluster where the account is named differently.
if [[ -z "$ACCOUNT" ]]; then
    ACCOUNT=$(sacctmgr -nP show assoc where user="$USERNAME" format=Account 2>/dev/null \
              | grep -v '^$' | sort -u | head -1)
fi

NOW_ISO=$(date +%Y-%m-%dT%H:%M:%S)

# Shared awk helpers, prepended to every awk program below.
#
#  * iso2epoch parses SLURM's timestamps ARITHMETICALLY. The obvious `date -d` per row
#    is two subprocesses per job and this script reads ~20k of them cluster-wide, which
#    turns a 2-second report into a 10-minute one. It is also portable: mktime() is a
#    gawk extension and some clusters ship mawk.
#  * It treats local time as UTC, which is exact here because every number below is a
#    DIFFERENCE between two timestamps in the same zone (a DST boundary costs 1 h once).
#  * gpus_of reads GPUs out of both spellings SLURM uses: squeue's TRES_PER_NODE
#    ("gres/gpu:8", "gres/gpu:h100:8") and sacct's AllocTRES ("gres/gpu=8").
read -r -d '' AWKLIB <<'AWKEOF'
function days_from_civil(y, m, d,   era, yoe, doy, doe) {
    if (m <= 2) y -= 1
    era = int((y >= 0 ? y : y - 399) / 400)
    yoe = y - era * 400
    doy = int((153 * (m + (m > 2 ? -3 : 9)) + 2) / 5) + d - 1
    doe = yoe * 365 + int(yoe / 4) - int(yoe / 100) + doy
    return era * 146097 + doe - 719468
}
function iso2epoch(s,   a, b) {
    if (s !~ /^[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9]/) return -1
    split(substr(s, 1, 10), a, "-")
    split(substr(s, 12, 8), b, ":")
    return days_from_civil(a[1] + 0, a[2] + 0, a[3] + 0) * 86400 + b[1] * 3600 + b[2] * 60 + b[3]
}
function gpus_of(s,   n, parts) {
    if (s ~ /gpu[:=][a-zA-Z0-9_]+:[0-9]+/) { n = split(s, parts, ":"); return parts[n] + 0 }
    if (match(s, /gpu[:=][0-9]+/))         { return substr(s, RSTART + 4, RLENGTH - 4) + 0 }
    return 0
}
function human(sec) {
    if (sec <= 60)    return "now"
    if (sec < 3600)   return sprintf("%d min", sec / 60)
    if (sec < 86400)  return sprintf("%.1f h", sec / 3600)
    return sprintf("%.1f DAYS", sec / 86400)
}
AWKEOF

hr()  { printf '%s\n' "--------------------------------------------------------------------------------"; }
sec() { printf '\n[%s] %s\n' "$1" "$2"; hr; }

echo "================================================================================"
echo " NON-INTERACTIVE GPU ALLOCATION REPORT"
echo "================================================================================"
CLUSTER=$(scontrol show config 2>/dev/null | awk -F= '/^ClusterName/{gsub(/ /,"",$2); print $2}')
printf ' cluster   %s\n' "${CLUSTER:-<unknown>}"
printf ' host      %s\n' "$(hostname)"
printf ' user      %s   account %s\n' "$USERNAME" "${ACCOUNT:-<unknown>}"
printf ' when      %s\n' "$(date -Is)"
printf ' scheduler %s   priority %s\n' \
    "$(scontrol show config 2>/dev/null | awk -F= '/^SchedulerType/{gsub(/ /,"",$2); print $2}')" \
    "$(scontrol show config 2>/dev/null | awk -F= '/^PriorityType/{gsub(/ /,"",$2); print $2}')"
printf ' probing   %s GPUs x %s h   (your history %s d, cluster-wide %s h)\n' \
    "$GPUS" "$HOURS" "$DAYS" "$ALLHOURS"

# ---------------------------------------------------------------------------
sec 1 "CAPACITY -- GPU partitions, and how many GPUs a node has"
GPU_PARTS=$(sinfo -h -o '%P|%G' 2>/dev/null | awk -F'|' '$2 ~ /gpu:/ {gsub(/\*/,"",$1); print $1}' | sort -u)
if [[ -z "$GPU_PARTS" ]]; then
    echo "  no partition advertises a gpu GRES -- nothing to compare here"
else
    printf '  %-22s %-11s %-9s %6s %6s %6s %6s %9s\n' \
        PARTITION TIMELIMIT GPUS/NODE ALLOC IDLE OTHER TOTAL "IDLE GPUs"
    for p in $GPU_PARTS; do
        tl=$(sinfo -h -p "$p" -o '%l' 2>/dev/null | head -1)
        aiot=$(sinfo -h -p "$p" -o '%F' 2>/dev/null | head -1)
        gpn=$(sinfo -h -p "$p" -o '%G' 2>/dev/null | head -1 | sed 's/(.*//; s/.*gpu://')
        IFS=/ read -r a i o t <<<"$aiot"
        printf '  %-22s %-11s %-9s %6s %6s %6s %6s %9s\n' \
            "$p" "${tl:-?}" "${gpn:-?}" "${a:-?}" "${i:-?}" "${o:-?}" "${t:-?}" "$(( ${i:-0} * ${gpn:-0} ))"
    done
    echo
    echo "  Node sizes present:"
    sinfo -h -N -o '%G' 2>/dev/null | sed 's/(.*//' | grep 'gpu:' | sort | uniq -c \
        | awk '{printf "    %6d nodes  %s\n", $1, $2}'
    echo "  If that is 8 everywhere, a 4-GPU request is half a node -- see the trap at the top."
fi

# ---------------------------------------------------------------------------
sec 2 "TIME LIMITS -- which GPU partitions can hold a ${HOURS%%,*}h job at all"
for p in $GPU_PARTS; do
    printf '  %-22s %s\n' "$p" "$(sinfo -h -p "$p" -o '%l' 2>/dev/null | head -1)"
done
echo "  A run needing 20h+ needs a long partition, or requeue-on-timeout plus checkpoints."

# ---------------------------------------------------------------------------
sec 3 "QUEUE -- the backlog you would be joining, per partition and GPU count"
printf '  %-22s %-6s %5s %5s %14s %14s\n' PARTITION GPUs RUN PEND "PEND MEAN AGE" "PEND MAX AGE"
for p in $GPU_PARTS; do
    squeue -h -p "$p" -o '%t|%b|%V' 2>/dev/null \
    | awk -F'|' -v P="$p" -v NOWS="$NOW_ISO" "$AWKLIB"'
        BEGIN { NOW = iso2epoch(NOWS) }
        {
            g = gpus_of($2); if (g == 0) next
            if ($1 == "R") { run[g]++; seen[g] = 1; next }
            if ($1 != "PD") next
            pend[g]++; seen[g] = 1
            s = iso2epoch($3)
            if (s > 0) { age = NOW - s; sum[g] += age; cnt[g]++; if (age > mx[g]) mx[g] = age }
        }
        END {
            for (g in seen)
                printf "  %-22s %-6s %5d %5d %12.1f h %12.1f h\n", P, g, run[g] + 0, pend[g] + 0,
                       (cnt[g] ? sum[g] / cnt[g] / 3600 : 0), (mx[g] ? mx[g] / 3600 : 0)
        }' | sort -k2,2rn
done
echo "  A pending backlog whose MEAN AGE is in days is a partition you do not get into."

# ---------------------------------------------------------------------------
sec 4 "THE SCHEDULER'S OWN ANSWER -- sbatch --test-only (submits nothing)"
echo "  For each shape: when SLURM says a job submitted right now would start."
echo
printf '  %-22s %-5s %-6s %-22s %s\n' PARTITION GPUs HOURS "WOULD START" DELAY
ACCT_FLAG=()
[[ -n "$ACCOUNT" ]] && ACCT_FLAG=(-A "$ACCOUNT")
PROBE_ROWS=""          # "delay_seconds|partition|gpus|hours", reused by [7]
for p in $GPU_PARTS; do
    for g in ${GPUS//,/ }; do
        for h in ${HOURS//,/ }; do
            out=$(sbatch --test-only "${ACCT_FLAG[@]}" -p "$p" --gres=gpu:"$g" -N1 \
                         -t "${h}:00:00" --wrap=true 2>&1)
            when=$(printf '%s' "$out" | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}' | head -1)
            if [[ -n "$when" ]]; then
                d=$(printf '%s\n' "$when $NOW_ISO" | awk "$AWKLIB"'{ d = iso2epoch($1) - iso2epoch($2); print (d < 0 ? 0 : d) }')
                printf '  %-22s %-5s %-6s %-22s %s\n' "$p" "$g" "$h" "$when" \
                    "$(awk "$AWKLIB"'BEGIN{ print human('"$d"') }')"
                PROBE_ROWS+="$d|$p|$g|$h"$'\n'
            else
                # A refusal is as informative as a time: it says the shape is not merely
                # slow here, it is not allowed -- wrong account, wrong QoS, over the
                # partition's time limit, or larger than any node.
                msg=$(printf '%s' "$out" | tr '\n' ' ' | sed 's/^ *//; s/  */ /g' | cut -c1-84)
                printf '  %-22s %-5s %-6s %-22s %s\n' "$p" "$g" "$h" "REFUSED" "${msg:-no answer}"
            fi
        done
    done
done
cat <<'CAVEAT'

  TWO CAVEATS, both of which matter more than the table looks:
   * It is optimistic. It does not model the jobs that will be scheduled ahead of yours
     over the next hour, only the state right now. Read it next to [3] and [5b].
   * If every node has the same GPU count, the GPUs column will often be IDENTICAL across
     1/4/8 -- that is correct, not a bug: a whole free node satisfies all three at the
     same instant. It means this probe discriminates between PARTITIONS, not between GPU
     counts. What makes a half-node request slow is CONTENTION for the gaps, and only
     [3] and [5b] can see that.
CAVEAT

# ---------------------------------------------------------------------------
sec 5 "YOUR HISTORY -- what your jobs actually waited, last ${DAYS} days"
hist=$(sacct -u "$USERNAME" --starttime "now-${DAYS}days" -X -P \
             -o Partition,AllocTRES,Submit,Start,State 2>/dev/null)
if [[ -z "$hist" || $(printf '%s\n' "$hist" | wc -l) -le 1 ]]; then
    echo "  No job history for $USERNAME here in the last ${DAYS} days."
    echo "  On a cluster you have not used that is not a gap: it means no fair-share debt,"
    echo "  which is often the single biggest reason a queue moves for you. See [5b]."
else
    printf '  %-22s %-5s %6s %13s %13s %9s\n' PARTITION GPUs JOBS "MEAN WAIT" "MAX WAIT" ">10 min"
    printf '%s\n' "$hist" | awk -F'|' "$AWKLIB"'
        NR > 1 {
            g = gpus_of($2); if (g == 0) next
            s = iso2epoch($3); t = iso2epoch($4)
            if (s <= 0 || t <= 0) next
            w = t - s; if (w < 0) w = 0
            k = $1 "|" g
            n[k]++; sum[k] += w; if (w > mx[k]) mx[k] = w
            if (w > 600) slow[k]++
        }
        END {
            for (k in n) {
                split(k, a, "|")
                printf "  %-22s %-5s %6d %11.1f m %11.1f m %9d\n",
                       a[1], a[2], n[k], sum[k] / n[k] / 60, mx[k] / 60, slow[k] + 0
            }
        }' | sort -k1,1 -k2,2rn
fi

# ---------------------------------------------------------------------------
sec 5b "CLUSTER-WIDE WAITS -- every user's GPU jobs, last ${ALLHOURS} h"
echo "  The measurement that needs no history of your own. If this is denied, the"
echo "  accounting DB restricts other users' jobs and only [3] and [4] are available."
echo
allhist=$(sacct -a --starttime "now-${ALLHOURS}hours" -X -P \
                -o Partition,AllocTRES,Submit,Start,State 2>/dev/null)
if [[ -z "$allhist" || $(printf '%s\n' "$allhist" | wc -l) -le 1 ]]; then
    echo "  denied or empty -- skipping"
else
    printf '  %-22s %-5s %7s %11s %11s %11s %8s\n' \
        PARTITION GPUs JOBS "MEDIAN" "MEAN" "P90" "<5 min"
    printf '%s\n' "$allhist" | awk -F'|' "$AWKLIB"'
        NR > 1 {
            g = gpus_of($2); if (g == 0) next
            s = iso2epoch($3); t = iso2epoch($4)
            if (s <= 0 || t <= 0) next
            w = t - s; if (w < 0) w = 0
            k = $1 "|" g
            n[k]++; sum[k] += w
            waits[k, n[k]] = w
            if (w < 300) fast[k]++
        }
        END {
            for (k in n) {
                c = n[k]
                # insertion sort: the per-key groups are small and this avoids depending
                # on gawk asort(), which mawk does not have.
                for (i = 1; i <= c; i++) v[i] = waits[k, i]
                for (i = 2; i <= c; i++) {
                    x = v[i]; j = i - 1
                    while (j >= 1 && v[j] > x) { v[j + 1] = v[j]; j-- }
                    v[j + 1] = x
                }
                med = v[int((c + 1) / 2)]
                p90 = v[int(c * 0.9) < 1 ? 1 : int(c * 0.9)]
                split(k, a, "|")
                printf "  %-22s %-5s %7d %9s %9s %9s %7.0f%%\n", a[1], a[2], c,
                       human(med), human(sum[k] / c), human(p90), 100 * (fast[k] + 0) / c
                delete v
            }
        }' | sort -k1,1 -k2,2rn
    echo
    echo "  MEDIAN is the number to compare across clusters. MEAN is dragged by a few"
    echo "  very old jobs; P90 is what a bad day looks like; <5 min is your odds of"
    echo "  simply walking on."
fi

# ---------------------------------------------------------------------------
sec 6 "PRIORITY -- fair-share, and what your account is holding right now"
if command -v sshare >/dev/null 2>&1; then
    sshare -U -u "$USERNAME" -o Account,User,RawUsage,EffectvUsage,FairShare -P 2>/dev/null \
        | sed 's/^/  /' | head -10
else
    echo "  sshare not available"
fi
if [[ -n "$ACCOUNT" ]]; then
    echo
    echo "  GPUs the account holds right now -- interactive shells count, and they cost"
    echo "  fair-share for every batch job submitted afterwards:"
    squeue -h -A "$ACCOUNT" -t R -o '%u|%b' 2>/dev/null \
    | awk -F'|' "$AWKLIB"'{ s[$1] += gpus_of($2) } END { for (u in s) printf "    %5d GPUs  %s\n", s[u], u }' \
    | sort -rn
fi
echo
echo "  Association limits (blank GrpTRES/MaxTRES = no hard cap; the queue is priority-driven):"
sacctmgr -nP show assoc where user="$USERNAME" format=Account,Partition,GrpTRES,MaxTRES,MaxJobs,QOS 2>/dev/null \
    | sed 's/^/    /' | head -8

# ---------------------------------------------------------------------------
sec 7 "SUMMARY -- paste this block back to compare clusters"
printf ' cluster=%s  user=%s  at=%s\n' "${CLUSTER:-?}" "$USERNAME" "$(date -Is)"
printf ' node size: %s\n' "$(sinfo -h -N -o '%G' 2>/dev/null | sed 's/(.*//' | grep 'gpu:' | sort | uniq -c | tr '\n' ' ')"
echo " fastest predicted start, ${HOURS%%,*}h job (from [4]):"
if [[ -n "$PROBE_ROWS" ]]; then
    printf '%s' "$PROBE_ROWS" | sort -n | head -6 \
        | awk -F'|' "$AWKLIB"'{ printf "   %-22s %2s GPUs  %s\n", $2, $3, human($1) }'
else
    echo "   no partition would accept any of the probed shapes"
fi
echo " median observed wait, last ${ALLHOURS} h, for the shapes you would use:"
if [[ -n "${allhist:-}" ]]; then
    printf '%s\n' "$allhist" | awk -F'|' -v want="$GPUS" "$AWKLIB"'
        BEGIN { split(want, W, ","); for (i in W) keep[W[i] + 0] = 1 }
        NR > 1 {
            g = gpus_of($2); if (!(g in keep)) next
            s = iso2epoch($3); t = iso2epoch($4); if (s <= 0 || t <= 0) next
            w = t - s; if (w < 0) w = 0
            k = $1 "|" g; n[k]++; waits[k, n[k]] = w
        }
        END {
            for (k in n) {
                c = n[k]
                for (i = 1; i <= c; i++) v[i] = waits[k, i]
                for (i = 2; i <= c; i++) { x = v[i]; j = i - 1
                    while (j >= 1 && v[j] > x) { v[j + 1] = v[j]; j-- }; v[j + 1] = x }
                split(k, a, "|")
                printf "   %-22s %2s GPUs  %-9s (n=%d)\n", a[1], a[2], human(v[int((c + 1) / 2)]), c
                delete v
            }
        }' | sort -k1,1 -k2,2rn | head -12
else
    echo "   cluster-wide accounting not available here"
fi
echo
hr
echo "Diff the [3], [4], [5b] and [7] blocks between clusters. [5b] is the honest one."
