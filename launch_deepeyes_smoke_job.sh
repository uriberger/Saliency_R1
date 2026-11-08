#!/usr/bin/env bash
# Submit P2: does plain verl GRPO run on Qwen3-VL-8B in the `deepeyes` env, and does one
# multi-turn ImageZoomInTool rollout come out with the right position ids, the right
# mask and its image grids intact?
#
#   bash launch_deepeyes_smoke_job.sh --name deepeyes-p2
#   bash launch_deepeyes_smoke_job.sh --name deepeyes-p2 --dry-run
#   bash launch_deepeyes_smoke_job.sh --name deepeyes-p2 --stage tool   # just stage B
#
# WHAT IS NOT OPTIONAL
#
#   the `deepeyes` env, not `laser`.  They are different conda envs pointing at different
#       verl trees. `laser` resolves verl to laser_fork, which this probe must not touch
#       and whose agent_loop.py does not carry the Qwen3-VL get_rope_index branch. The
#       runner asserts which tree answered before it starts; a mismatch is a hard stop,
#       not a warning, because every number after it would be about the wrong code.
#
#   DEEPEYES_ASSERT_MROPE=1.  Default. With it off, a 1-D position-id fall-through is
#       invisible: transformers broadcasts 2-D ids into three identical rows and trains
#       happily. This is the ONE failure in the whole arm that costs a week if found
#       late, and the only defence is to make it loud.
#
#   8 GPUs, TP 2.  FSDP keeps ~96 GB of params+grads+Adam resident and sharded (~12 GB a
#       card) while vLLM takes 0.5 of each 80 GB card. 4 GPUs would fit too but halves
#       nothing useful here: the step is 128 sequences and most of the clock is model
#       load and engine init.
#
#   Qwen/Qwen3-VL-8B-Instruct, NOT our cold start.  P2 asks whether the STACK runs, so
#       the fewest local modifications in the question, the better. The cold start is
#       also not loadable here as it stands: it was written by transformers 5.7.0 and
#       serialises `extra_special_tokens` as a list, which 4.57.6 (the pin that ships
#       Qwen3-VL) calls .keys() on. fix_coldstart_tokenizer.py is the fix and is
#       idempotent, but it rewrites a SHARED checkpoint and so belongs to whoever is
#       actually training, not to a probe. Pass --model to override.
#
# WHAT THE FIRST STEP SHOULD PRINT
#
#   `verl -> .../deepeyes_fork/verl`, then `guard present; fork confirmed`.
#   `[DEEPEYES-MROPE] get_rope_index=qwen3_vl pid=...` -- and NEVER `=qwen2_vl`.
#   `[DEEPEYES-MROPE] forward qwen3_vl_base_forward: position_ids (4, 1, N) OK`.
#   Then verl's own `step:1 - ... - timing_s/step:...`.
#
#   A `[DEEPEYES-MROPE]` line naming qwen2_vl means a Qwen3VLProcessor reached the
#   Qwen2.5-VL MRoPE. A RuntimeError carrying that tag means the same thing and the job
#   stopped, which is the designed behaviour -- do not set DEEPEYES_ASSERT_MROPE=0 to
#   get past it.
#
# COST AND SIZE. 4 hours, 8 GPUs. Stage A is 3 steps of a 16x8 batch; stage B is one
# rollout batch. Both pay ~10-15 min of model load plus vLLM engine init, and stage B
# pays it a second time because it stands up its own ray cluster. If the allocation is
# the binding constraint, `--stage tool` runs stage B alone in well under an hour -- but
# stage A is the GATE and skipping it inverts the point of the probe.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

NAME=""
STAGE="grpo,tool"
GPUS=8
TP=2
STEPS=3
DURATION=4
MODEL="Qwen/Qwen3-VL-8B-Instruct"
OUT_DIR=""
DATA_DIR=""
FORK=""
PARTITION_OVERRIDE=""
DRY_RUN=0
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)   DRY_RUN=1;               shift   ;;
        --name)      NAME="$2";               shift 2 ;;
        --stage)     STAGE="$2";              shift 2 ;;
        --gpus)      GPUS="$2";               shift 2 ;;
        --tp)        TP="$2";                 shift 2 ;;
        --steps)     STEPS="$2";              shift 2 ;;
        --duration)  DURATION="$2";           shift 2 ;;
        --model)     MODEL="$2";              shift 2 ;;
        --out-dir)   OUT_DIR="$2";            shift 2 ;;
        --data-dir)  DATA_DIR="$2";           shift 2 ;;
        --fork)      FORK="$2";               shift 2 ;;
        --partition) PARTITION_OVERRIDE="$2"; shift 2 ;;
        --)          shift; EXTRA+=("$@"); break ;;
        *)           EXTRA+=("$1");           shift ;;
    esac
done

[[ -n "$NAME" ]] || { echo "ERROR: --name is required (it names the job and the log)." >&2; exit 2; }
IFS=',' read -r -a STAGES <<< "$STAGE"
for s in "${STAGES[@]}"; do
    case "$s" in grpo|tool) ;; *) echo "ERROR: --stage $s is not a stage (grpo|tool)." >&2; exit 2 ;; esac
done
(( GPUS % TP == 0 )) || { echo "ERROR: --tp $TP must divide --gpus $GPUS." >&2; exit 2; }

# Resolve through the worktree symlinks to the CENTRAL tree. A job submitted from a
# worktree that is later merged and deleted would otherwise reference paths that no
# longer exist by the time it reaches the front of the queue, and fail at model load an
# hour later looking like an environment problem. `|| true` is load-bearing: readlink -f
# exits non-zero on a path that does not exist and set -e would kill us before any check
# could say why.
resolve() { readlink -f "$1" 2>/dev/null || true; }

# deepeyes_fork is NOT in .worktree-links on main (that entry is on the unmerged
# feat/deepeyes-env branch), so a worktree has no symlink for it; find the central tree.
if [ -z "$FORK" ]; then
    CENTRAL="$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)"
    CENTRAL="${CENTRAL%/.git}"
    FORK="${CENTRAL:-$REPO}/deepeyes_fork"
fi
FORK=$(resolve "$FORK")
[ -n "$FORK" ] && [ -d "$FORK" ] || { echo "MISSING deepeyes_fork -- see docs/deepeyes-build-prompts.md Prompt 0" >&2; exit 1; }
case "$FORK" in */laser_fork) echo "REFUSING: that is laser_fork." >&2; exit 1 ;; esac
grep -q "Qwen3VLProcessor" "$FORK/verl/experimental/agent_loop/agent_loop.py" || {
    echo "The fork's agent_loop.py has no Qwen3-VL get_rope_index branch." >&2
    echo "  bash patch_deepeyes_fork.sh" >&2; exit 1; }
[ -f "$FORK/verl/utils/deepeyes_mrope_guard.py" ] || {
    echo "The fork has no MRoPE guard.  bash patch_deepeyes_fork.sh" >&2; exit 1; }

# Per-branch subdirectories under the SHARED outputs/, so two sessions never clobber
# each other (CLAUDE.md: the symlinked trees are shared, not copies).
OUT_DIR=${OUT_DIR:-"$REPO/outputs/deepeyes/probe-verl-smoke/$NAME"}
DATA_DIR=${DATA_DIR:-"$REPO/outputs/deepeyes/probe-verl-smoke/data"}
OUT_DIR=$(resolve "$(dirname "$OUT_DIR")")/$(basename "$OUT_DIR")
DATA_DIR=$(resolve "$DATA_DIR")
[ -n "$DATA_DIR" ] && [ -d "$DATA_DIR" ] || {
    echo "MISSING data dir -- run: python build_deepeyes_smoke_data.py --out-dir <dir>" >&2; exit 1; }
for f in geo3k_train.parquet geo3k_val.parquet zoom_smoke.parquet; do
    [ -f "$DATA_DIR/$f" ] || { echo "MISSING $DATA_DIR/$f" >&2; exit 1; }
done

# shellcheck source=/dev/null
source "$REPO/cluster_env.sh"
PARTITION=${PARTITION_OVERRIDE:-${PARTITION:-$(SR1_JOB_HOURS=$DURATION sr1_pick_partition)}}
ACCOUNT=${ACCOUNT:-nvr_israel_rlop}
CONDA_ENV=${CONDA_ENV:-deepeyes}

# ---------------------------------------------------------------------------
# IS THERE ANY GPU HARDWARE AT ALL? Ask before queueing, because a partition with no
# nodes is not a queue -- it accepts the job and never schedules it, and the job sits
# there looking like contention. On 2026-11-07 every GPU partition on oci-nrt-cs-001
# reported `Nodes=(null) TotalNodes=0` (backfill, batch_block1, batch_large,
# batch_long, batch_short, batch_singlenode, interactive, sniff-gpu-nodes), while the
# eight cpu_* partitions had 50 nodes each. One scontrol call tells a dead partition
# from a busy one; park nothing.
LIVE=""
for p in ${PARTITION//,/ }; do
    n=$(scontrol show partition "$p" 2>/dev/null | grep -o 'TotalNodes=[0-9]*' | head -1 | cut -d= -f2)
    [ -n "${n:-}" ] && [ "$n" -gt 0 ] 2>/dev/null && LIVE="${LIVE:+$LIVE,}$p"
done
if [ -z "$LIVE" ]; then
    echo "==========================================================================" >&2
    echo "NO GPU HARDWARE. Every partition in '$PARTITION' has TotalNodes=0 on"       >&2
    echo "$(hostname). This cluster currently has no GPU nodes at all; only the"      >&2
    echo "cpu_* partitions have hardware. Submitting would park the job forever."     >&2
    echo                                                                              >&2
    echo "  sinfo -a -o '%P %D'    # count nodes per partition"                       >&2
    echo                                                                              >&2
    echo "What CAN run here today, and answers A1/A2/A3 on the real agent-loop code:" >&2
    echo "  bash launch_deepeyes_agentloop_cpu_job.sh --name deepeyes-p2-cpu"         >&2
    echo "  python test_deepeyes_rope_cpu.py"                                         >&2
    echo "Pass --partition <name> to override if you know better."                    >&2
    echo "==========================================================================" >&2
    [[ $DRY_RUN -eq 1 ]] || exit 1
fi
[ -n "$LIVE" ] && PARTITION="$LIVE"

# ---------------------------------------------------------------------------
# WHICH SUBMITTER. The repo's launchers all use the ADLR `submit_job` wrapper, and on
# 2026-11-07 it cannot run on this cluster at all: its NSS config roots every path at
# `/scratch/fsw/portfolios/adlr/...` and there is no `/scratch` on the login node OR on
# the compute nodes (checked on cpu-0010), so it dies at
#     ERROR: Request to mount path that does not exist: /scratch
# before reaching sbatch. Plain sbatch works, and everything this job needs -- conda,
# the fork, the HF cache -- is on /lustre and runs bare-metal. Prefer the wrapper when
# it is usable so the house convention still holds wherever it does.
SUBMITTER="${SUBMITTER:-auto}"
if [ "$SUBMITTER" = auto ]; then
    if [ -d /scratch ] && sr1_find_submit_job; then SUBMITTER=submit_job; else SUBMITTER=sbatch; fi
fi
[ "$SUBMITTER" = submit_job ] && { sr1_find_submit_job || [[ $DRY_RUN -eq 1 ]] || {
    echo "ERROR: submit_job not found under the cluster-interface paths." >&2; exit 1; }; }

# ---------------------------------------------------------------------------
# SUBMITTING FROM A vscode LOGIN NODE. Two shims, both for the same reason: the
# ADLR cluster-interface identifies the cluster from $HOSTNAME, and
# `oci-nrt-cs-001-vscode-01` is not a name it knows.
#
#   1. detect_system.sh's case arm is `oci-nrt-slurm-1-*`. No arm matches a vscode
#      node, so it exits with
#          ERROR: Unable to determine target cluster system from host name ...
#      which reads like a broken launcher and is not one. Spoofing HOSTNAME for the
#      duration of the submit_job call selects the correct arm -- which is also where
#      NSS_CONFIGFILE and CLUSTER_NAME come from, so this is the one-line fix rather
#      than setting three variables by hand.
#
#   2. That arm then resolves ADLR_PYTHON through the NSS config's
#      NSSSUB_ADLR_UTILS_ENV_ROOT, which is `/scratch/fsw/portfolios/adlr/...`.
#      There is no /scratch on this node at all, so submit_job dies on its shebang
#      line with "No such file or directory". detect_system.sh honours a PRE-SET
#      ADLR_PYTHON (`if [ -z "${ADLR_PYTHON:-}" ]`), and the same interpreter exists
#      under /lustre/fs1, so point at that.
#
# This belongs in cluster_env.sh next to sr1_find_submit_job eventually -- every
# launcher in the repo hits it from here, and the other DeepEyes prompts will. It is
# local to this file for now because cluster_env.sh is shared with live sessions and a
# probe should not be the thing that edits it.
SUBMIT_HOSTNAME="$(hostname)"
case "$SUBMIT_HOSTNAME" in
    oci-nrt-cs-*vscode*) SUBMIT_HOSTNAME="oci-nrt-slurm-1-login-01" ;;
esac
if [ -z "${ADLR_PYTHON:-}" ] && [ ! -x "/scratch/fsw/portfolios/adlr/projects/adlr_other_infra/python/latest/bin/python" ]; then
    for p in /lustre/fs1/portfolios/adlr/projects/adlr_other_infra/python/latest/bin/python \
             /lustre/fsw/portfolios/adlr/projects/adlr_other_infra/python/latest/bin/python; do
        [ -x "$p" ] && { export ADLR_PYTHON="$p"; break; }
    done
fi

LOG_ROOT="$REPO/outputs/logs"
mkdir -p "$LOG_ROOT" "$OUT_DIR"

RUNNER="$LOG_ROOT/$NAME.runner.sh"
{
    echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    printf 'cd %q\n' "$REPO"
    echo 'source /home/uberger/scratch/miniconda3/etc/profile.d/conda.sh'
    printf 'conda activate %q\n' "$CONDA_ENV"
    echo "export HF_HOME=${HF_HOME:-/home/uberger/scratch/cache/hf_cache}"
    # Everything is cached. Staying offline keeps a transient hub outage from looking
    # like a port failure, which is the class of confusion this probe exists to remove.
    echo 'export HF_HUB_OFFLINE=1'
    echo 'export TOKENIZERS_PARALLELISM=false'
    echo 'export VLLM_WORKER_MULTIPROC_METHOD=spawn'
    echo 'export DEEPEYES_ASSERT_MROPE=1'
    echo
    for s in "${STAGES[@]}"; do
        printf 'bash launch_deepeyes_smoke.sh --stage %q --out-dir %q --data-dir %q \\\n' \
            "$s" "$OUT_DIR/$s" "$DATA_DIR"
        printf '    --model %q --fork %q --gpus %q --tp %q --steps %q' \
            "$MODEL" "$FORK" "$GPUS" "$TP" "$STEPS"
        for a in ${EXTRA[@]+"${EXTRA[@]}"}; do printf ' %q' "$a"; done
        echo
    done
    echo
    printf 'python deepeyes_smoke_report.py --stage grpo --out-dir %q || true\n' "$OUT_DIR/grpo"
    printf 'python deepeyes_smoke_report.py --stage tool --out-dir %q || true\n' "$OUT_DIR/tool"
} > "$RUNNER"
chmod +x "$RUNNER"

echo "=========================================================================="
echo "Job       : $NAME   ($ACCOUNT, $PARTITION, ${DURATION}h, ${GPUS} GPU, TP $TP)"
echo "Stages    : $STAGE"
echo "Model     : $MODEL"
echo "Fork      : $FORK"
echo "Env       : $CONDA_ENV"
echo "Data      : $DATA_DIR"
echo "Out dir   : $OUT_DIR"
echo "Runner    : $RUNNER"
echo "=========================================================================="
cat "$RUNNER"
echo "=========================================================================="
echo "WHAT TO LOOK FOR, in order:"
echo "  1. 'verl -> .../deepeyes_fork/verl' and 'guard present; fork confirmed'"
echo "  2. '[DEEPEYES-MROPE] get_rope_index=qwen3_vl'  -- and never '=qwen2_vl'"
echo "  3. '[DEEPEYES-MROPE] forward qwen3_vl_base_forward: position_ids (4, 1, N) OK'"
echo "  4. verl's 'step:1 - ... timing_s/step:...'  -- THE number P2 is asked for"
echo "  5. stage tool: A1/A2/A3 PASS lines and the decoded rollouts under them"
echo "=========================================================================="

echo "Submitter : $SUBMITTER"
[[ $DRY_RUN -eq 1 ]] && { echo "[dry-run] not submitting."; exit 0; }

if [ "$SUBMITTER" = submit_job ]; then
    env HOSTNAME="$SUBMIT_HOSTNAME" submit_job \
        --account "$ACCOUNT" \
        --partition "$PARTITION" \
        --name "$NAME" \
        --gpu "$GPUS" \
        --duration "$DURATION" \
        --outfile "$LOG_ROOT/$NAME.%j.out" \
        --logroot "$LOG_ROOT" \
        -c "bash $RUNNER"
else
    SB="$LOG_ROOT/$NAME.sbatch"
    {
        echo "#!/bin/bash"
        echo "#SBATCH --account=$ACCOUNT"
        echo "#SBATCH --job-name=$NAME"
        echo "#SBATCH --partition=$PARTITION"
        echo "#SBATCH --nodes=1"
        echo "#SBATCH --ntasks=1"
        echo "#SBATCH --gpus-per-node=$GPUS"
        echo "#SBATCH --cpus-per-task=32"
        echo "#SBATCH --time=${DURATION}:00:00"
        echo "#SBATCH --output=$LOG_ROOT/$NAME.%j.out"
        # Ray writes thousands of small files; keep them off the shared filesystem.
        echo 'export RAY_TMPDIR=/tmp/ray_$SLURM_JOB_ID'
        echo 'mkdir -p "$RAY_TMPDIR"'
        echo "bash $RUNNER"
    } > "$SB"
    chmod +x "$SB"
    echo "sbatch script: $SB"
    sbatch "$SB"
fi
