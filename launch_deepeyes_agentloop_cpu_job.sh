#!/usr/bin/env bash
# Submit the CPU replay of P2's three assertions to cpu_short.
#
#   bash launch_deepeyes_agentloop_cpu_job.sh --name deepeyes-p2-cpu
#   bash launch_deepeyes_agentloop_cpu_job.sh --name deepeyes-p2-cpu --dry-run
#
# PLAIN sbatch, NOT submit_job, and that is not a style choice. The ADLR
# cluster-interface wrapper cannot run on this cluster at all right now: its NSS config
# roots everything at `/scratch/fsw/portfolios/adlr/...`, and there is no `/scratch` on
# the vscode login node OR on the compute nodes (checked on cpu-0010). It fails at
#     ERROR: Request to mount path that does not exist: /scratch
# before it ever reaches sbatch. Everything this job needs -- conda, the fork, the HF
# cache -- is on /lustre and works bare-metal; `sbatch` is on PATH and submits fine.
#
# WHY CPU. oci-nrt-cs-001 has no GPU hardware: backfill, batch_block1, batch_large,
# batch_long, batch_short, batch_singlenode, interactive and sniff-gpu-nodes all report
# `Nodes=(null) TotalNodes=0`, while the eight cpu_* partitions have 50 nodes. So the
# GPU half of P2 cannot run here today and the CPU half is what is left. See
# deepeyes_agentloop_cpu.py for exactly what that does and does not establish.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

NAME="deepeyes-p2-cpu"
PARTITION="cpu_short"
DURATION="01:00:00"
MODEL="Qwen/Qwen3-VL-8B-Instruct"
OUT_DIR=""
DATA_DIR=""
ROWS=4
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)   DRY_RUN=1;     shift   ;;
        --name)      NAME="$2";     shift 2 ;;
        --partition) PARTITION="$2";shift 2 ;;
        --duration)  DURATION="$2"; shift 2 ;;
        --model)     MODEL="$2";    shift 2 ;;
        --out-dir)   OUT_DIR="$2";  shift 2 ;;
        --data-dir)  DATA_DIR="$2"; shift 2 ;;
        --rows)      ROWS="$2";     shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

resolve() { readlink -f "$1" 2>/dev/null || true; }

OUT_DIR=${OUT_DIR:-"$REPO/outputs/deepeyes/probe-verl-smoke/$NAME"}
DATA_DIR=${DATA_DIR:-"$REPO/outputs/deepeyes/probe-verl-smoke/data"}
OUT_DIR=$(resolve "$(dirname "$OUT_DIR")")/$(basename "$OUT_DIR")
DATA_DIR=$(resolve "$DATA_DIR")
[ -n "$DATA_DIR" ] && [ -f "$DATA_DIR/zoom_smoke.parquet" ] || {
    echo "MISSING $DATA_DIR/zoom_smoke.parquet -- run build_deepeyes_smoke_data.py" >&2; exit 1; }

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
#SBATCH --cpus-per-task=16
#SBATCH --time=$DURATION
#SBATCH --output=$LOG_ROOT/$NAME.%j.out
set -euo pipefail
cd $REPO
source /home/uberger/scratch/miniconda3/etc/profile.d/conda.sh
conda activate deepeyes
export HF_HOME=${HF_HOME:-/home/uberger/scratch/cache/hf_cache}
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export DEEPEYES_ASSERT_MROPE=1
export PYTHONUNBUFFERED=1
# Ray writes a lot of small files; keep it off the shared filesystem.
export RAY_TMPDIR=/tmp/ray_\$SLURM_JOB_ID
mkdir -p "\$RAY_TMPDIR"

echo "HOST=\$(hostname)"
python -c "import verl, os; print('verl ->', os.path.dirname(verl.__file__))"

python deepeyes_agentloop_cpu.py \\
    --model $MODEL \\
    --data $DATA_DIR/zoom_smoke.parquet \\
    --out-dir $OUT_DIR \\
    --rows $ROWS
EOF
chmod +x "$SBATCH_FILE"

echo "=========================================================================="
echo "Job     : $NAME   ($PARTITION, $DURATION, CPU only)"
echo "Model   : $MODEL"
echo "Data    : $DATA_DIR/zoom_smoke.parquet"
echo "Out dir : $OUT_DIR"
echo "Script  : $SBATCH_FILE"
echo "=========================================================================="
cat "$SBATCH_FILE"
echo "=========================================================================="

[[ $DRY_RUN -eq 1 ]] && { echo "[dry-run] not submitting."; exit 0; }
sbatch "$SBATCH_FILE"
