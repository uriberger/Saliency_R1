#!/bin/bash
# Cluster-portable resolution of the two settings that differ between the machines this
# repo runs on. SOURCE this (do not execute it) from a launcher, after REPO is set:
#
#     source "$REPO/cluster_env.sh"
#     PARTITION=${PARTITION:-$(sr1_pick_partition batch_singlenode batch_long batch)}
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

# sr1_pick_partition <preferred> [<fallback> ...]
# Echo the first named partition that actually exists here. If none match, echo the
# cluster's default partition (the one sinfo marks with '*'), because any real partition
# beats a name that cannot be scheduled. Preference order should start with the origin's
# name so behaviour there is unchanged.
sr1_pick_partition() {
    local avail want default
    avail=$(sinfo -h -o '%P' 2>/dev/null | tr -d '*' | sort -u)
    if [ -z "$avail" ]; then echo "${1:-batch}"; return 0; fi   # no SLURM here (e.g. a login shell without it)
    for want in "$@"; do
        if printf '%s\n' "$avail" | grep -qx -- "$want"; then echo "$want"; return 0; fi
    done
    default=$(sinfo -h -o '%P' 2>/dev/null | grep -m1 '\*' | tr -d '*')
    echo "${default:-$(printf '%s\n' "$avail" | head -1)}"
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
