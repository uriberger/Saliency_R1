#!/usr/bin/env bash
# Run the P3 zoom tool through verl's real ToolAgentLoop on a CPU node.
#
#   bash launch_deepeyes_zoom_agentloop_cpu_job.sh --name deepeyes-p3-cpu
#   bash launch_deepeyes_zoom_agentloop_cpu_job.sh --name deepeyes-p3-cpu --frame abs_original
#   bash launch_deepeyes_zoom_agentloop_cpu_job.sh --name deepeyes-p3-cpu --dry-run
#
# WHAT IS NOT OPTIONAL
#   - `bash patch_deepeyes_zoom_tool.sh` must have been run, or the fork still carries
#     verl's tool: MIN_DIMENSION 28, no coordinate frame, and a create() that
#     smart-resizes the image it is about to crop.
#   - --frame. There is no default here and none in the tool. P4
#     (deepeyes_coord_format_probe.py) measures which of the three is right; until it
#     reports, every run of this states its assumption on the command line.
#
# WHAT THE FIRST STEP SHOULD PRINT
#       [DEEPEYES-ZOOM] ImageZoomInTool: bbox_frame=<yours> min_dimension=32
#       (patch 16 x merge 2 = 32px/token) view_factor=32 crop_source_check=strict
#   then, once per tool call:
#       [DEEPEYES-ZOOM] call=0 frame=... raw=[...] -> converted=[...] -> cropped=[...]
#       original=[W,H] view=[vW,vH] crop=[w,h] status=ok
#
#   min_dimension that is not 32, or view_factor that is not 32, means the fork was not
#   patched or `patch_size` was overridden -- a 28 there is Qwen2.5-VL's geometry on a
#   Qwen3-VL model. `raw == converted` on a frame other than abs_original means the
#   conversion did not happen. A `downscale` of 1.0 means the corpus has no detail for
#   a zoom to recover and the arm is decoration on that data.
#
# PLAIN sbatch, NOT submit_job: the ADLR wrapper cannot run on this cluster (its NSS
# config roots everything at /scratch, which exists on no node here). P2 found this;
# see launch_deepeyes_agentloop_cpu_job.sh for the full note.
#
# CPU, because this cluster has zero GPU nodes -- every GPU partition reports
# TotalNodes=0 -- and because generation is scripted here anyway. The eight cpu_*
# partitions have 50 nodes each and schedule instantly.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

NAME="deepeyes-p3-cpu"
PARTITION="cpu_short"
DURATION="01:00:00"
MODEL="Qwen/Qwen3-VL-8B-Instruct"
FRAME="rel1000"
OUT_DIR=""
DATA=""
P2_DIR=""
ROWS=4
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)   DRY_RUN=1;      shift   ;;
        --name)      NAME="$2";      shift 2 ;;
        --partition) PARTITION="$2"; shift 2 ;;
        --duration)  DURATION="$2";  shift 2 ;;
        --model)     MODEL="$2";     shift 2 ;;
        --frame)     FRAME="$2";     shift 2 ;;
        --out-dir)   OUT_DIR="$2";   shift 2 ;;
        --data)      DATA="$2";      shift 2 ;;
        --p2-dir)    P2_DIR="$2";    shift 2 ;;
        --rows)      ROWS="$2";      shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

resolve() { readlink -f "$1" 2>/dev/null || true; }

# Per-BRANCH output directory. outputs/ is a symlink shared with every other session
# and every running job; writing to a common name there is how two sessions clobber
# each other's results without either noticing.
OUT_DIR=${OUT_DIR:-"$REPO/outputs/deepeyes/feat-zoom-tool/$NAME"}
DATA=${DATA:-"$REPO/outputs/deepeyes/probe-verl-smoke/data/zoom_smoke.parquet"}
OUT_DIR=$(resolve "$(dirname "$OUT_DIR")")/$(basename "$OUT_DIR")
DATA=$(resolve "$DATA")
[ -n "$DATA" ] && [ -f "$DATA" ] || {
    echo "MISSING zoom_smoke.parquet -- it is P2's; run build_deepeyes_smoke_data.py" >&2; exit 1; }

# The fork must carry OUR tool, not verl's. Checking here costs nothing and saves a
# scheduling round trip to discover MIN_DIMENSION is still 28.
bash "$REPO/patch_deepeyes_zoom_tool.sh" --check | sed 's/^/  /'
if ! bash "$REPO/patch_deepeyes_zoom_tool.sh" --check | grep -q "^  applied      image_zoom_in_tool.py"; then
    echo "REFUSING: deepeyes_fork does not carry the P3 tool. Run:" >&2
    echo "    bash patch_deepeyes_zoom_tool.sh" >&2
    exit 1
fi

LOG_ROOT="$(resolve "$REPO/outputs")/logs"
mkdir -p "$LOG_ROOT" "$OUT_DIR"
SBATCH_FILE="$LOG_ROOT/$NAME.sbatch"

P2_ARG=""
[ -n "$P2_DIR" ] && P2_ARG="--p2-dir $P2_DIR"

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
export HF_HOME=\${HF_HOME:-/home/uberger/scratch/cache/hf_cache}
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export DEEPEYES_ASSERT_MROPE=1
export PYTHONUNBUFFERED=1
# Ray writes a lot of small files; keep it off the shared filesystem.
export RAY_TMPDIR=/tmp/ray_\$SLURM_JOB_ID
mkdir -p "\$RAY_TMPDIR"

echo "HOST=\$(hostname)"
python -c "import verl, os; print('verl ->', os.path.dirname(verl.__file__))"

# The unit tests first: they are seconds, they need nothing, and a failure here makes
# every later line of the log meaningless.
python test_deepeyes_zoom_tool_cpu.py

python deepeyes_zoom_agentloop_cpu.py \\
    --frame $FRAME \\
    --model $MODEL \\
    --data $DATA \\
    --out-dir $OUT_DIR \\
    --rows $ROWS $P2_ARG
EOF
chmod +x "$SBATCH_FILE"

echo "=========================================================================="
echo "Job     : $NAME   ($PARTITION, $DURATION, CPU only)"
echo "Frame   : $FRAME   <-- stated, not defaulted; P4 measures the right one"
echo "Model   : $MODEL"
echo "Data    : $DATA"
echo "Out dir : $OUT_DIR"
echo "Script  : $SBATCH_FILE"
echo "=========================================================================="

[[ $DRY_RUN -eq 1 ]] && { cat "$SBATCH_FILE"; echo "[dry-run] not submitting."; exit 0; }
sbatch "$SBATCH_FILE"
