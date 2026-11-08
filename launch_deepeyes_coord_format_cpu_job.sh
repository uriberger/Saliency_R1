#!/usr/bin/env bash
# Submit the DeepEyes coordinate/format probe as a CPU job array.
#
#   bash launch_deepeyes_coord_format_cpu_job.sh --name de-coordfmt \
#       --out-dir outputs/deepeyes_coord_format/probe-deepeyes-coord-format
#
# WHY A CPU LAUNCHER EXISTS FOR A GENERATION PROBE.
#
# This probe wants one GPU for an hour. On 2026-11-07 `oci-nrt-cs-001` had NO GPU
# HARDWARE AT ALL -- every GPU partition reported `TotalNodes=0` and no node advertised a
# `gpu` GRES, a state it had been in since 2026-10-30. The one-line test:
#
#     sinfo -h -N -o "%G" | grep -ci gpu       # 0 means there is nothing to queue behind
#
# A GPU job submitted into that sits at `Reason=PartitionConfig` forever and looks exactly
# like ordinary queueing. Meanwhile `cpu_short` has 50 idle 96-core nodes and a 4 h cap.
# So the probe runs there instead: an 8B forward at batch size 1 is slow on a CPU but it
# is not impossible, and the measurement is 764 short generations, not a training run.
#
# USE launch_deepeyes_coord_format_job.sh INSTEAD the moment GPUs come back. This file is
# a workaround and should be read as one; the two produce the same JSONL in the same
# directory and the same `--stage report` reduces either.
#
# SHAPE, AND WHY IT IS NOT A SQUARE GRID.
#
# A CPU forward is dominated by PREFILL, which scales with the image, not by decode. At
# V*'s native resolution one row takes over 20 minutes on 32 cores -- measured, job
# 7742357, which produced no completed row in that time. So:
#
#   * `--cpus 96`, one worker per node. Prefill is compute-bound and takes all the cores
#     it is given; decode is bandwidth-bound and does not. Two workers sharing a node is
#     slower per worker and no faster overall.
#   * BOTH arms get a pixel cap (`--max-pixels-a`, `--max-pixels-b`), not just arm B.
#     That is a deviation from the GPU design and has to be said out loud in the writeup:
#     arm A is supposed to be the processor's own config, which is what verl trains at.
#     The three readings stay separable because the two CAPS differ, which is all the
#     measurement needs -- but the resolution is no longer the training resolution.
#   * Arm A gets more shards than arm B (`--shards-a`, `--shards-b`) because its rows
#     cost more. An equal split leaves the arm-A workers running an hour past the others.
#
# `cpu_short`'s QoS caps this account at 10 CPU nodes; the default shape asks for 8 and
# leaves room for another session.
#
# Resuming: re-run the identical command. Shard results are append-only JSONL keyed by
# uid and skipped on a second pass, so a job that hits the 4 h wall loses nothing.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

NAME=""
OUT_DIR=""
PROCS=1
CPUS_PER_PROC=96
DURATION="4:00:00"
PARTITION="cpu_short"
# 1280 and 320 visual tokens. Qwen3-VL is patch-16 / merge-2, so one visual token is
# 32x32 px -- these are written N*32*32 rather than the N*28*28 of every Qwen2.5-VL
# script so that the number in front means "tokens" on THIS backbone.
MAXPIX_A=$(( 1280 * 32 * 32 ))
MAXPIX_B=$(( 320 * 32 * 32 ))
N_A=64
N_B=32
SHARDS_A=3
SHARDS_B=1
MAX_NEW_TOKENS=320
DRY_RUN=0
MODELS=()
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)      DRY_RUN=1;          shift   ;;
        --name)         NAME="$2";          shift 2 ;;
        --out-dir)      OUT_DIR="$2";       shift 2 ;;
        --procs)        PROCS="$2";         shift 2 ;;
        --cpus)         CPUS_PER_PROC="$2"; shift 2 ;;
        --duration)     DURATION="$2";      shift 2 ;;
        --partition)    PARTITION="$2";     shift 2 ;;
        --max-pixels-a) MAXPIX_A="$2";      shift 2 ;;
        --max-pixels-b) MAXPIX_B="$2";      shift 2 ;;
        --n-a)          N_A="$2";           shift 2 ;;
        --n-b)          N_B="$2";           shift 2 ;;
        --shards-a)     SHARDS_A="$2";      shift 2 ;;
        --shards-b)     SHARDS_B="$2";      shift 2 ;;
        --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
        --model)        MODELS+=("$2");     shift 2 ;;
        --)             shift; EXTRA+=("$@"); break ;;
        *)              EXTRA+=("$1");      shift   ;;
    esac
done

[[ -n "$NAME"    ]] || { echo "ERROR: --name is required." >&2; exit 2; }
[[ -n "$OUT_DIR" ]] || { echo "ERROR: --out-dir is required." >&2; exit 2; }
[[ "$OUT_DIR" = /* ]] || OUT_DIR="$REPO/$OUT_DIR"

if [[ ${#MODELS[@]} -eq 0 ]]; then
    MODELS=(
        "coldstart=checkpoint/coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged"
        "instruct=Qwen/Qwen3-VL-8B-Instruct"
    )
fi

LOG_ROOT="$REPO/outputs/logs"
mkdir -p "$LOG_ROOT" "$OUT_DIR/logs" "$OUT_DIR/raw"

# The worker manifest. One line per (model, arm, shard), carrying that arm's row budget;
# the array task picks its own PROCS lines out of it. Written to disk rather than
# recomputed inside the job so that what ran is recoverable from the output alone.
MANIFEST="$OUT_DIR/workers.tsv"
: > "$MANIFEST"
for spec in "${MODELS[@]}"; do
    for arm in A B; do
        if [[ "$arm" == A ]]; then NS=$SHARDS_A; NN=$N_A; else NS=$SHARDS_B; NN=$N_B; fi
        for ((s = 0; s < NS; s++)); do
            printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
                "${spec%%=*}" "${spec#*=}" "$arm" "$s" "$NS" "$NN" >> "$MANIFEST"
        done
    done
done
NWORK=$(wc -l < "$MANIFEST")
NARRAY=$(( (NWORK + PROCS - 1) / PROCS ))

SBATCH="$LOG_ROOT/$NAME.sbatch"
cat > "$SBATCH" <<EOF
#!/bin/bash
#SBATCH --account=${ACCOUNT:-nvr_israel_rlop}
#SBATCH --partition=$PARTITION
#SBATCH --job-name=$NAME
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$(( PROCS * CPUS_PER_PROC ))
#SBATCH --mem=160G
#SBATCH --time=$DURATION
#SBATCH --array=0-$(( NARRAY - 1 ))
#SBATCH --export=ALL
#SBATCH --output=$LOG_ROOT/$NAME.%A_%a.out
set -uo pipefail
cd $REPO
source /home/uberger/scratch/miniconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV:-saliency_r1_qwen3_vllm}
export HF_HOME=${HF_HOME:-/home/uberger/scratch/cache/hf_cache}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false
export DEEPEYES_REF=${DEEPEYES_REF:-/home/uberger/scratch/deepeyes_ref/DeepEyes}
# One worker per \$CPUS_PER_PROC cores. Left unset, every process grabs all 96 and they
# fight: measured, that is slower than giving each a third of the machine.
export OMP_NUM_THREADS=$CPUS_PER_PROC
export MKL_NUM_THREADS=$CPUS_PER_PROC

MANIFEST=$MANIFEST
START=\$(( SLURM_ARRAY_TASK_ID * $PROCS + 1 ))
END=\$(( START + $PROCS - 1 ))
pids=()
for LN in \$(seq \$START \$END); do
    LINE=\$(sed -n "\${LN}p" "\$MANIFEST") || true
    [[ -z "\$LINE" ]] && continue
    IFS=\$'\t' read -r TAG MPATH ARM SHARD NSHARD NROWS <<< "\$LINE"
    LOG="$OUT_DIR/logs/\${TAG}_\${ARM}_\${SHARD}.log"
    echo "[task \$SLURM_ARRAY_TASK_ID] worker \$LN: \$TAG arm \$ARM shard \$SHARD/\$NSHARD n=\$NROWS -> \$LOG"
    python -u deepeyes_coord_format_probe.py --stage generate \\
        --out-dir $OUT_DIR --model "\$MPATH" --model-tag "\$TAG" \\
        --arm "\$ARM" --shard "\$SHARD" --num-shards "\$NSHARD" --n "\$NROWS" \\
        --arm-a-max-pixels $MAXPIX_A --arm-b-max-pixels $MAXPIX_B \\
        --max-new-tokens $MAX_NEW_TOKENS \\
        ${EXTRA[@]+"${EXTRA[@]}"} > "\$LOG" 2>&1 &
    pids+=(\$!)
done
fail=0
for p in \${pids[@]+"\${pids[@]}"}; do wait "\$p" || fail=1; done
exit \$fail
EOF
chmod +x "$SBATCH"

echo "=========================================================================="
echo "Job       : $NAME   ($PARTITION, $DURATION, array 0-$(( NARRAY - 1 )))"
echo "Out dir   : $OUT_DIR"
echo "Arm A     : max_pixels $MAXPIX_A ($(( MAXPIX_A / 1024 )) vision tokens), n=$N_A, $SHARDS_A shards"
echo "Arm B     : max_pixels $MAXPIX_B ($(( MAXPIX_B / 1024 )) vision tokens), n=$N_B, $SHARDS_B shards"
echo "Tokens    : max_new_tokens $MAX_NEW_TOKENS"
echo "Workers   : $NWORK, $PROCS per node x $CPUS_PER_PROC cores"
echo "Manifest  : $MANIFEST"
echo "sbatch    : $SBATCH"
echo "=========================================================================="
cat "$MANIFEST"
echo "=========================================================================="

[[ $DRY_RUN -eq 1 ]] && { echo "[dry-run] not submitting."; exit 0; }
sbatch "$SBATCH"
echo
echo "When the array is done:"
echo "  python deepeyes_coord_format_probe.py --stage report --out-dir $OUT_DIR"
