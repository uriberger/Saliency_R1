#!/usr/bin/env bash
# Build a question-box cache for one corpus, as a SLURM job. One shard per visible GPU,
# then a merge -- the same fan-out launch_overlap_probe.sh uses, and the same `_job.sh`
# split as launch_overlap_probe_job.sh.
#
#   bash launch_precompute_question_boxes_job.sh \
#       --dataset cold_data/grpo_sets/set_a \
#       [--out <file>] [--gpus 8] [--duration 1] [--box-threshold 0.10] \
#       [--direct] [--dry-run]
#
# --out defaults to outputs/question_boxes/<dataset>_bt<threshold>.json, which is
# gitignored and symlinked into every worktree, so the cache is built once and shared.
#
# --direct runs the shard fan-out HERE instead of submitting it, for when you are already
# inside an allocation (or on a GPU box with no SLURM). Same runner script either way, so
# the two paths cannot diverge. With --direct and no explicit --gpus, the shard count comes
# from the GPUs actually visible rather than the 8 a submitted job would ask for -- fanning
# 8 shards at 2 visible cards would put four processes on each and OOM.
#
# The output feeds `launch_grpo_qwen3_overlap_job.sh --question-boxes <out>`, which then
# runs with no Grounding-DINO on the training device at all.
#
# Runtime: ~20 groundings/second/GPU, so 8 cards do the 8k in under a minute and a 50k
# set in about five. An hour is generous and buys batch_short.
#
# --box-threshold must match the training run's --box-threshold: it is applied inside the
# detector, so it cannot be re-applied to cached boxes, and the trainer refuses a
# mismatched file rather than training on the wrong ones.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Deliberately NOT `cd "$REPO"` here: a relative --dataset / --out given on the command
# line has to be resolved against the cwd the user typed it in, and the runner cds to
# $REPO itself. Everything this script touches before then is an absolute path.

DATASET=""
OUT=""
# GPUS doubles as the shard count. Left unset so --direct can tell "the user asked for N"
# from "nobody said", and default to the visible cards in the second case only.
GPUS=""
DIRECT=false
DURATION=1
BOX_THRESHOLD=0.10
# The detector's own batch. 32 OOMs on an 80GB card at 512px and halves itself, which
# works but costs a retry on most calls -- measured on val_natural, job 6542363.
BATCH_SIZE=16
NAME=""
DRY_RUN=0
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)        DRY_RUN=1;            shift   ;;
        --direct)         DIRECT=true;          shift   ;;
        --dataset)        DATASET="$2";         shift 2 ;;
        --out)            OUT="$2";             shift 2 ;;
        --gpus)           GPUS="$2";            shift 2 ;;
        --duration)       DURATION="$2";        shift 2 ;;
        --box-threshold)  BOX_THRESHOLD="$2";   shift 2 ;;
        --batch-size)     BATCH_SIZE="$2";      shift 2 ;;
        --name)           NAME="$2";            shift 2 ;;
        --)               shift; EXTRA+=("$@"); break ;;
        *)                EXTRA+=("$1");        shift ;;
    esac
done

[[ -n "$DATASET" ]] || { echo "ERROR: --dataset is required." >&2; exit 2; }

# How many shards, and therefore how many cards the fan-out expects. A submitted job asks
# SLURM for the number, so 8 is simply the request; running here, the number is a FACT
# about this machine, and getting it wrong stacks shards onto the same card.
VISIBLE=""
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    VISIBLE=$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
elif command -v nvidia-smi >/dev/null 2>&1; then
    VISIBLE=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)
fi
if $DIRECT; then
    if [[ -z "$GPUS" ]]; then
        [[ -n "$VISIBLE" && "$VISIBLE" -gt 0 ]] || {
            echo "ERROR: --direct could not see any GPU (no CUDA_VISIBLE_DEVICES, and" >&2
            echo "       nvidia-smi found nothing). Pass --gpus N if you know better." >&2
            exit 1; }
        GPUS=$VISIBLE
        echo "[note] --direct: fanning $GPUS shard(s) over the $GPUS visible GPU(s)."
    elif [[ -n "$VISIBLE" && "$VISIBLE" -gt 0 && "$GPUS" -gt "$VISIBLE" ]]; then
        echo "ERROR: --gpus $GPUS but only $VISIBLE GPU(s) are visible. The fan-out gives" >&2
        echo "       shard i to GPU i, so the extra shards would stack onto cards that are" >&2
        echo "       already busy and OOM. Lower --gpus, or drop it to use all $VISIBLE." >&2
        exit 1
    fi
fi
GPUS=${GPUS:-8}

# A local corpus must be absolutised, because the runner cds to $REPO. Look for it in the
# invocation cwd first and in $REPO second -- a worktree does not have cold_data/grpo_sets
# (it is gitignored and not in .worktree-links), so a path typed there resolves against
# neither unless we try both. Anything that is not a directory here is taken to be a Hub
# id (peterant330/saliency-r1-8k) and passed through, which is the only other thing
# --dataset_name accepts.
if [[ -d "$DATASET" ]]; then
    DATASET="$(cd "$DATASET" && pwd)"
elif [[ -d "$REPO/$DATASET" ]]; then
    DATASET="$(cd "$REPO/$DATASET" && pwd)"
elif [[ "$DATASET" != */* || "$DATASET" == */*/* || "$DATASET" == .* || "$DATASET" == /* ]]; then
    echo "ERROR: --dataset $DATASET is not a directory here or under $REPO, and does not" >&2
    echo "       look like a Hub id either." >&2
    exit 2
else
    echo "[note] --dataset $DATASET is not a local directory; treating it as a Hub id."
fi

DS_SLUG=$(basename "$DATASET" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9_]/_/g')
NAME=${NAME:-qbox_$DS_SLUG}
# outputs/ is gitignored and symlinked into every worktree, so one cache is shared rather
# than rebuilt per branch. The name carries the threshold because a file built at another
# one cannot be reused -- the trainer refuses it -- and two of them must not collide.
OUT=${OUT:-$REPO/outputs/question_boxes/${DS_SLUG}_bt${BOX_THRESHOLD}.json}
mkdir -p "$(dirname "$OUT")"
OUT="$(cd "$(dirname "$OUT")" && pwd)/$(basename "$OUT")"

# shellcheck source=/dev/null
source "$REPO/cluster_env.sh"
# Neither is read on the --direct path, and resolving a partition there would ask SLURM a
# question about a job that is not being submitted.
if ! $DIRECT; then
    PARTITION=${PARTITION:-$(SR1_JOB_HOURS=$DURATION sr1_pick_partition)}
    ACCOUNT=${ACCOUNT:-nvr_israel_rlop}
fi
CONDA_ENV=${CONDA_ENV:-saliency_r1_qwen3_vllm}

CONDA_ROOT=${CONDA_ROOT:-}
if [[ -z "$CONDA_ROOT" ]]; then
    for CAND in /home/uberger/scratch/miniconda3 \
                /lustre/fs12/portfolios/nvr/projects/nvr_israel_rlop/users/uberger/research/miniforge3; do
        [[ -f "$CAND/etc/profile.d/conda.sh" ]] && { CONDA_ROOT="$CAND"; break; }
    done
fi
[[ -n "$CONDA_ROOT" ]] || { echo "ERROR: no conda root found." >&2; exit 1; }

if ! $DIRECT; then
    sr1_find_submit_job || [[ $DRY_RUN -eq 1 ]] || {
        echo "ERROR: submit_job not found under the cluster-interface paths. Use --direct" >&2
        echo "       to run here instead." >&2; exit 1; }
fi

LOG_ROOT="$REPO/outputs/logs"
mkdir -p "$LOG_ROOT"

# The shard loop lives in the runner rather than in a -c string: it has a wait/fail
# discipline, and a partial cache must never be merged into a file a run would then
# silently train on.
RUNNER="$LOG_ROOT/$NAME.runner.sh"
cat > "$RUNNER" <<RUNNER_EOF
#!/usr/bin/env bash
set -euo pipefail
set +u
source $CONDA_ROOT/etc/profile.d/conda.sh
conda activate $CONDA_ENV
set -u
export HF_HOME=${HF_HOME:-/home/uberger/scratch/cache/hf_cache}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false
cd $REPO

SHARD_DIR="\$(dirname "$OUT")/.$NAME.shards"
mkdir -p "\$SHARD_DIR"

pids=()
for ((i = 0; i < $GPUS; i++)); do
    CUDA_VISIBLE_DEVICES="\$i" python precompute_question_boxes.py \\
        --dataset "$DATASET" \\
        --out "\$SHARD_DIR/shard\$i.json" \\
        --shard "\$i" --num-shards "$GPUS" \\
        --box-threshold "$BOX_THRESHOLD" \\
        --batch-size "$BATCH_SIZE" \\
        ${EXTRA[@]+${EXTRA[@]}} \\
        > "\$SHARD_DIR/shard\$i.log" 2>&1 &
    pids+=(\$!)
    echo "[launch] shard \$i -> GPU \$i (pid \${pids[-1]})"
done

fail=0
for i in "\${!pids[@]}"; do
    if wait "\${pids[\$i]}"; then echo "[done] shard \$i ok"
    else echo "[FAIL] shard \$i -- see \$SHARD_DIR/shard\$i.log" >&2; fail=1; fi
done
if [[ \$fail -ne 0 ]]; then
    echo "ERROR: a shard failed; NOT merging. A partial cache would be a silently" >&2
    echo "       smaller reward on the rows it is missing." >&2
    tail -30 "\$SHARD_DIR"/shard*.log >&2
    exit 1
fi

python precompute_question_boxes.py --merge "\$SHARD_DIR"/shard*.json --out "$OUT"
RUNNER_EOF
chmod +x "$RUNNER"

echo "=========================================================================="
if $DIRECT; then
echo "Job       : $NAME   (direct, no SLURM, ${GPUS} GPU)"
else
echo "Job       : $NAME   ($ACCOUNT, $PARTITION, ${DURATION}h, ${GPUS} GPU)"
fi
echo "Dataset   : $DATASET"
echo "Out       : $OUT"
echo "DINO      : box_threshold=$BOX_THRESHOLD batch_size=$BATCH_SIZE"
echo "Runner    : $RUNNER"
echo "=========================================================================="
cat "$RUNNER"
echo "=========================================================================="

[[ $DRY_RUN -eq 1 ]] && { echo "[dry-run] not $($DIRECT && echo running || echo submitting)."; exit 0; }

# The SAME runner the submitted path would have handed to the node, so the two modes
# cannot drift. exec, so this script's exit status IS the build's -- a failed shard must
# not look like a success to whatever called this.
if $DIRECT; then exec bash "$RUNNER"; fi

submit_job \
    --account "$ACCOUNT" \
    --partition "$PARTITION" \
    --name "$NAME" \
    --gpu "$GPUS" \
    --duration "$DURATION" \
    --outfile "$LOG_ROOT/$NAME.%j.out" \
    --logroot "$LOG_ROOT" \
    -c "bash $RUNNER"
