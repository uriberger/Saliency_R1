#!/bin/bash
# Cluster-portable resolution of the two settings that differ between the machines this
# repo runs on. SOURCE this (do not execute it) from a launcher, after REPO is set:
#
#     source "$REPO/cluster_env.sh"
#     PARTITION=${PARTITION:-$(sr1_pick_partition)}
#     sr1_find_submit_job || { echo "ERROR: submit_job not found" >&2; exit 1; }
#
# Why this exists: the origin cluster and oci-hsg-cs-001 (GB200/aarch64) disagree on
#
#   * SLURM partition names -- origin has batch_singlenode / batch_block1; oci-hsg-cs-001
#     has batch (4h) / batch_long (7d) / cpu / cpu_datamover. Submitting to a partition that
#     does not exist is not a clean failure: the job is simply never schedulable.
#   * ADLR cluster-interface release -- the launchers pinned
#     21.1_2026-04-15_21-25-57, which exists on the origin but not on oci-hsg-cs-001
#     (which has 20.0, 21.0, 21.2, 21.3 and a 'latest' symlink).
#
# Both are resolved by asking the machine rather than branching on a cluster name, so a
# third cluster works with no edit. An explicit PARTITION=... in the environment always
# wins -- these only supply the default.

# The preference list used when sr1_pick_partition is called with no arguments. Every
# launcher in this repo submits ONE node with 8 GPUs for 4h, so every name here must be able
# to hold that shape. Ordered least-contended-first, but the order is only a tie-break: SLURM
# considers all of them and takes whichever can start the job soonest.
#
# Why this is a list and not one name: pinning a single name is what left 8-GPU jobs queued
# for a day behind a pool the scheduler had no reason to prefer. The clusters get there by
# different routes, and the list has to cover both:
#
#   * polar -- the pools differ by orders of magnitude in SIZE, and the small ones are
#     shared with the interactive pool.
#   * oci-nrt-cs-001 -- size says nothing, because every GPU partition is backed by the SAME
#     554 nodes. What differs is QUEUE DEPTH. batch_block1 is Default=YES and so collects
#     nearly every job on the cluster (~880 pending); batch_singlenode is the same hardware
#     with ~3. Measured with `sbatch --test-only`, that was a 13.5-hour difference in
#     predicted start for one identical 8-GPU job. Both are listed and both earn their
#     place: batch_singlenode is usually empty, and batch_block1 is PriorityTier=20 against
#     batch_singlenode's 10, so it wins outright whenever it does have room.
#
# Names NOT added for oci-nrt-cs-001, so nobody re-derives this: batch_short (MaxTime=2h,
# under the 4h these jobs ask for), interactive (its QoS caps a user at 16 GPUs, and two
# interactive shells already spend exactly that -- see sr1__drop_user_capped for why a
# per-user cap parks the whole job), backfill (PreemptMode=CANCEL at PriorityTier=1, so a
# 4h training job is killed rather than requeued), and admin / batch_large / sniff-gpu-nodes
# (AllowAccounts excludes us). batch_long is likewise closed to this account here, but it
# stays for oci-hsg-cs-001; alongside a partition that does work it is simply ignored,
# because this cluster sets EnforcePartLimits=ANY.
SR1_DEFAULT_PARTITIONS=${SR1_DEFAULT_PARTITIONS:-"polar4 polar3 polar batch_singlenode batch_block1 batch_long batch"}

# sr1_pick_partition [<preferred> ...]
# Echo a comma-separated list of the named partitions that actually exist here, in the
# order given -- the form `sbatch --partition` and `submit_job --partition` both accept.
# With no arguments, use $SR1_DEFAULT_PARTITIONS. If none of the names match, echo the
# cluster's default partition (the one sinfo marks with '*'), because any real partition
# beats a name that cannot be scheduled.
#
# Two names are filtered out of a multi-partition result; see the helpers below for why.
# Both filters are skipped when only one partition survives, and both fail open.
sr1_pick_partition() {
    local avail want default list=
    [ $# -eq 0 ] && set -- $SR1_DEFAULT_PARTITIONS
    avail=$(sinfo -h -o '%P' 2>/dev/null | tr -d '*' | sort -u)
    if [ -z "$avail" ]; then echo "${1:-batch}"; return 0; fi   # no SLURM here (e.g. a login shell without it)
    for want in "$@"; do
        printf '%s\n' "$avail" | grep -qx -- "$want" && list="${list:+$list }$want"
    done
    if [ -z "$list" ]; then
        default=$(sinfo -h -o '%P' 2>/dev/null | grep -m1 '\*' | tr -d '*')
        echo "${default:-$(printf '%s\n' "$avail" | head -1)}"
        return 0
    fi
    list=$(sr1__drop_backfill_mix "$list")
    list=$(sr1__drop_user_capped "$list")
    printf '%s\n' "$list" | tr ' ' ','
}

# sr1__drop_backfill_mix <space-separated partitions>
# SLURM here rejects a submission that names both backfill and non-backfill partitions
# ("Submitting to a mix of backfill and non-backfill partitions is not allowed"), drops the
# backfill ones and warns. Do it ourselves so the warning does not scroll past on every
# launch. A backfill-only list is a legal submission, so that is left alone.
sr1__drop_backfill_mix() {
    local list="$1" kept= dropped= p
    case "$list" in *backfill*) ;; *) printf '%s\n' "$list"; return 0 ;; esac
    for p in $list; do
        case "$p" in backfill*) dropped="${dropped:+$dropped }$p" ;;
                     *)         kept="${kept:+$kept }$p" ;; esac
    done
    [ -z "$kept" ] && { printf '%s\n' "$list"; return 0; }
    echo "sr1_pick_partition: dropped $dropped (SLURM refuses to mix backfill and non-backfill)" >&2
    printf '%s\n' "$kept"
}

# sr1__drop_user_capped <space-separated partitions>
# Drop partitions whose QoS sets a per-user running-job cap (MaxJobsPerUser).
#
# This one is not obvious and it bites hard. QOSMaxJobsPerUserLimit is a JOB-level hold, not
# a per-partition one: if you are at the cap for ONE partition in the list, the job parks
# across ALL of them and stops being considered anywhere -- Reason flips from Priority to
# QOSMaxJobsPerUserLimit and StartTime goes Unknown. A cap of one running job is the
# pathological case, since a single shell you already hold is enough to trigger it.
#
# Such a partition is still perfectly good on its own; reach it with an explicit
# PARTITION=<name>, where there is no other partition to park.
sr1__drop_user_capped() {
    local list="$1" kept= dropped= p capped
    case "$list" in *" "*) ;; *) printf '%s\n' "$list"; return 0 ;; esac   # single partition: no parking risk
    capped=$(
        { scontrol show partition -o 2>/dev/null |
              sed -n 's/^PartitionName=\([^ ]*\).*[^A-Za-z]QoS=\([^ ]*\).*/\1 \2/p'
          echo '--'
          sacctmgr -n -P show qos format=Name,MaxJobsPU 2>/dev/null | awk -F'|' '$2 != "" {print $1}'
        } | awk '/^--$/{s=1;next} s{cap[$1]=1;next} {pq[$1]=$2}
                 END{for (p in pq) if (cap[pq[p]]) print p}'
    )
    [ -z "$capped" ] && { printf '%s\n' "$list"; return 0; }   # no sacctmgr, or no caps anywhere: fail open
    for p in $list; do
        if printf '%s\n' "$capped" | grep -qx -- "$p"
        then dropped="${dropped:+$dropped }$p"
        else kept="${kept:+$kept }$p"; fi
    done
    [ -z "$kept" ] && { printf '%s\n' "$list"; return 0; }     # would empty the list: leave it alone
    [ -n "$dropped" ] && echo "sr1_pick_partition: dropped $dropped (per-user QoS job cap parks a multi-partition job)" >&2
    printf '%s\n' "$kept"
}

# sr1_find_submit_job
# Put the ADLR submit_job wrapper on PATH. Prefers an already-visible submit_job, then the
# 'latest' symlink, then the newest versioned release; the tree is mounted under /lustre/fs1
# on some clusters and /lustre/fsw on others. `[ -x ]` follows symlinks, so a broken
# cross-filesystem symlink is skipped rather than selected. Returns non-zero if not found.
sr1_find_submit_job() {
    command -v submit_job >/dev/null 2>&1 && return 0
    local root cand
    for root in \
        /lustre/fs1/portfolios/adlr/projects/adlr_other_infra/release/cluster-interface \
        /lustre/fsw/portfolios/adlr/projects/adlr_other_infra/release/cluster-interface; do
        for cand in "$root/latest" $(ls -1dt "$root"/*/ 2>/dev/null); do
            if [ -x "${cand%/}/submit_job" ]; then
                export PATH="${cand%/}:$PATH"
                return 0
            fi
        done
    done
    return 1
}
