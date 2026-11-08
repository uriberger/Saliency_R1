#!/usr/bin/env bash
# Build the DeepEyes 47k verl corpus on a CPU node.
#
#   bash launch_deepeyes_data_job.sh                       # the full build + verify
#   bash launch_deepeyes_data_job.sh --stats-only          # measure, write no parquet
#   bash launch_deepeyes_data_job.sh --smoke 64            # a mechanical check
#   bash launch_deepeyes_data_job.sh --dry-run             # print the sbatch, submit nothing
#
# WHY cpu_short. This is pure IO and zstd: read 4.84 GB of parquet, write 4.84 GB
# back with the tool wiring attached. It wants no GPU, and this cluster has none to
# give -- every batch_* partition reports TotalNodes=0 while the eight cpu_* pools
# have 50 nodes each and start instantly. cpu_short caps at MaxTime=04:00:00, which
# is several times what the build needs.
#
# PLAIN sbatch, NOT submit_job: the ADLR cluster-interface wrapper roots its mounts at
# /scratch/fsw/portfolios/adlr/... and there is no /scratch on this cluster's login or
# compute nodes, so it fails before it reaches sbatch. Same reasoning, same words, as
# launch_deepeyes_agentloop_cpu_job.sh.
#
# THE DOWNLOAD IS NOT PART OF THIS JOB, DELIBERATELY. The three parquet files come
# from the login node, which has egress; the compute nodes are assumed not to. The job
# runs with HF_HUB_OFFLINE=1 and resolves the snapshot out of the shared HF cache, so a
# missing download is a clean early failure rather than a silent 4.84 GB re-fetch:
#
#   hf download ChenShawn/DeepEyes-Datasets-47k --repo-type dataset
#
# WHAT THE FIRST LINES SHOULD PRINT, AND WHAT A MISMATCH MEANS
#
#   HF snapshot : .../datasets--ChenShawn--DeepEyes-Datasets-47k/snapshots/<sha>
#       If this is missing, the download never happened on this filesystem.
#   [vstar]  22362 rows / [chart]  13659 rows / [thinklite]  11031 rows
#       Any other count means a file is short or partly unreadable. The build asserts
#       data_source and env_name row by row and exits on the first disagreement,
#       because P5's reward routes on those exact strings.
#   pixels median / p99
#       This is the number that sizes data.max_prompt_length. If it comes back near
#       262,144 (512x512) on every corpus, something applied a resize cap and the zoom
#       tool has nothing left to recover -- stop and find it.
#
# OUTPUT lands in cold_data/deepeyes/verl, which is shared like cold_data/laser: one
# copy for every session. It is NOT per-branch, because a 5 GB corpus duplicated per
# worktree is how a filesystem fills up. Two sessions building at once will clobber
# each other -- pass --out-dir if that is a risk.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

NAME="deepeyes-data"
PARTITION="cpu_short"
DURATION="02:00:00"
CPUS=16
OUT_DIR=""
HF_DIR=""
IMAGE_MODE="file"
EXTRA=()
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)    DRY_RUN=1;         shift   ;;
        --name)       NAME="$2";         shift 2 ;;
        --partition)  PARTITION="$2";    shift 2 ;;
        --duration)   DURATION="$2";     shift 2 ;;
        --cpus)       CPUS="$2";         shift 2 ;;
        --out-dir)    OUT_DIR="$2";      shift 2 ;;
        --hf-dir)     HF_DIR="$2";       shift 2 ;;
        --image-mode) IMAGE_MODE="$2";   shift 2 ;;
        --stats-only) EXTRA+=(--stats-only); shift ;;
        --verify-only) EXTRA+=(--verify-only); shift ;;
        --smoke)      EXTRA+=(--smoke "$2"); shift 2 ;;
        --max-rows)   EXTRA+=(--max-rows "$2"); shift 2 ;;
        --val-size)   EXTRA+=(--val-size "$2"); shift 2 ;;
        -h|--help)    sed -n '2,40p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

resolve() { readlink -f "$1" 2>/dev/null || true; }

OUT_DIR=${OUT_DIR:-"$REPO/cold_data/deepeyes/verl"}
OUT_DIR=$(resolve "$(dirname "$OUT_DIR")")/$(basename "$OUT_DIR")

export HF_HOME=${HF_HOME:-/home/uberger/scratch/cache/hf_cache}
SNAP_GLOB="$HF_HOME/hub/datasets--ChenShawn--DeepEyes-Datasets-47k/snapshots"/*
if [[ -z "$HF_DIR" ]]; then
    # shellcheck disable=SC2086
    HF_DIR=$(ls -d $SNAP_GLOB 2>/dev/null | head -1 || true)
fi
[[ -n "$HF_DIR" && -f "$HF_DIR/data_thinklite_reasoning_acc.parquet" ]] || {
    echo "MISSING the DeepEyes-Datasets-47k snapshot under $HF_HOME." >&2
    echo "Run this on the LOGIN node first (the compute nodes have no egress):" >&2
    echo "  hf download ChenShawn/DeepEyes-Datasets-47k --repo-type dataset" >&2
    exit 1; }

# P3's corpus verifier, which the build is not allowed to grade itself with: it decodes
# create_kwargs.image through the TOOL's own decoder and compares against the raw
# `images` column as an independent witness, then reports the downscale distribution --
# the one number that says whether a zoom on this corpus recovers anything at all.
# The second path is the unmerged feat/deepeyes-zoom-tool worktree and disappears when
# P3 lands on main.
ZOOM_CFG=""
for c in "$REPO/deepeyes/zoom_tool_config.py" \
         "$REPO/../../.worktrees/feat-deepeyes-zoom-tool/deepeyes/zoom_tool_config.py" \
         "$REPO/.worktrees/feat-deepeyes-zoom-tool/deepeyes/zoom_tool_config.py"; do
    [[ -f "$c" ]] && { ZOOM_CFG=$(resolve "$c"); break; }
done
[[ -n "$ZOOM_CFG" ]] || { echo "MISSING deepeyes/zoom_tool_config.py (P3's)" >&2; exit 1; }

LOG_ROOT="$(resolve "$REPO/outputs")/logs"
mkdir -p "$LOG_ROOT" "$OUT_DIR"
SBATCH_FILE="$LOG_ROOT/$NAME.sbatch"

cat > "$SBATCH_FILE" <<EOF
#!/bin/bash
#SBATCH --account=nvr_israel_rlop
#SBATCH --job-name=$NAME
#SBATCH --partition=$PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$CPUS
#SBATCH --time=$DURATION
#SBATCH --output=$LOG_ROOT/$NAME.%j.out
set -euo pipefail
cd $REPO
source /home/uberger/scratch/miniconda3/etc/profile.d/conda.sh
conda activate deepeyes
export HF_HOME=$HF_HOME
# Offline on purpose: a compute node that cannot reach the hub must fail loudly on a
# missing snapshot, not silently try to pull 4.84 GB through a node with no egress.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
# datasets' parquet loader copies every shard into its cache during --verify. Keep
# that ~5 GB of churn off the shared filesystem and out of everyone else's cache.
export HF_DATASETS_CACHE=/tmp/hf_datasets_\$SLURM_JOB_ID
mkdir -p "\$HF_DATASETS_CACHE"
trap 'rm -rf "\$HF_DATASETS_CACHE"' EXIT

echo "HOST=\$(hostname)  CPUS=$CPUS  MEM=\$(free -g | awk '/^Mem:/{print \$2}')G"
python -c "import pyarrow, datasets, transformers; print('pyarrow', pyarrow.__version__, '| datasets', datasets.__version__, '| transformers', transformers.__version__)"

python build_deepeyes_data.py \\
    --hf-dir $HF_DIR \\
    --out-dir $OUT_DIR \\
    --image-mode $IMAGE_MODE \\
    --strict-template \\
    ${EXTRA[@]+"${EXTRA[@]}"}

echo
echo "=== P3's corpus verifier (independent witness + downscale distribution) ==="
# Not gated on \$?: a failure here is a finding to read, not a reason to hide the rest.
#
# The thinklite shard is EXPECTED to fail with "no tools_kwargs" on every row. That is
# the math split, the zoom tool is disabled on it, and this verifier reporting anything
# else would mean the tool had leaked onto 23% of the corpus.
for shard in $OUT_DIR/train_vstar_*.parquet $OUT_DIR/train_chart_*.parquet; do
    echo "--- \$(basename \$shard)   (tool ON -- must pass)"
    python $ZOOM_CFG verify "\$shard" --n 50 || true
done
for shard in $OUT_DIR/train_thinklite_*.parquet; do
    echo "--- \$(basename \$shard)   (tool OFF -- must report 'no tools_kwargs' on EVERY row)"
    python $ZOOM_CFG verify "\$shard" --n 50 || true
done

echo "=== output ==="
ls -lh $OUT_DIR
EOF
chmod +x "$SBATCH_FILE"

echo "=========================================================================="
echo "Job        : $NAME   ($PARTITION, $DURATION, ${CPUS} CPU, no GPU)"
echo "HF snapshot: $HF_DIR"
echo "Out dir    : $OUT_DIR   (SHARED -- not per-branch)"
echo "Image mode : $IMAGE_MODE"
echo "Extra      : ${EXTRA[*]-<none>}"
echo "Script     : $SBATCH_FILE"
echo "=========================================================================="

[[ $DRY_RUN -eq 1 ]] && { cat "$SBATCH_FILE"; echo "[dry-run] not submitting."; exit 0; }
sbatch "$SBATCH_FILE"
