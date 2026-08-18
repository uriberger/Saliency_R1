#!/usr/bin/env bash
# Re-scan the same 1,157 prepared cases with the box-free sharpness columns, then race
# them against the DINO grounding columns.
#
# Nothing new is computed on the GPU: `saliency_sharpness.py` reads the maps the two
# existing scans already build and costs one sort over the patch axis, so this is the
# same work as the 2026-08-06/08-07 runs plus a few seconds. The reason it has to run
# at all is that those scans only ever persisted the DINO scores, never the maps.
#
#   bash launch_sharpness.sh --gpus 8 [--maps grad,glimpse,rollout_wnorm] \
#        [--out-root outputs/sharpness] [--cases-dir outputs/intervene_probe/coldstart_setA_v2]
#
# Runtime on 8 GPUs, from the 2026-08 logs: heads ~1 min, grad ~5 min, each rollout
# ~5 min, glimpse ~28 min. Drop glimpse with --maps grad,rollout_wnorm for a 12-minute
# pass; it is the only expensive one, being a backward per target token.
#
# It writes to a NEW tree (--out-root) rather than back into outputs/head_corr and
# outputs/flow_corr. The npz schema gained columns, and overwriting the scans every
# published number on this corpus was computed from, to add a column, is not a trade
# worth making.
#
# Resuming: re-run the identical command. A shard whose scan/shardNN.npz exists is
# skipped by both underlying launchers (--overwrite to redo).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

GPUS=8
OUT_ROOT="outputs/sharpness"
CASES_DIR="outputs/intervene_probe/coldstart_setA_v2"
MAPS="grad,rollout_wnorm,glimpse"
DO_HEADS=1
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)       GPUS="$2";      shift 2 ;;
        --out-root)   OUT_ROOT="$2";  shift 2 ;;
        --cases-dir)  CASES_DIR="$2"; shift 2 ;;
        --maps)       MAPS="$2";      shift 2 ;;
        --no-heads)   DO_HEADS=0;     shift   ;;
        *)            EXTRA+=("$1");  shift   ;;
    esac
done

if [[ ! -d "$CASES_DIR/cases" ]]; then
    echo "no $CASES_DIR/cases -- point --cases-dir at an intervene_probe out-dir" >&2
    exit 2
fi

echo "=========================================================================="
echo "Out root  : $OUT_ROOT"
echo "Cases     : $CASES_DIR"
echo "Heads     : $DO_HEADS"
echo "Flow maps : $MAPS"
echo "Shards    : $GPUS"
echo "=========================================================================="

SCANS=()

if [[ "$DO_HEADS" -eq 1 ]]; then
    bash launch_head_correlation.sh --gpus "$GPUS" --out-dir "$OUT_ROOT/heads" \
        --cases-dir "$CASES_DIR" "${EXTRA[@]+"${EXTRA[@]}"}"
    SCANS+=("--scan" "heads=$OUT_ROOT/heads")
fi

if [[ -n "$MAPS" ]]; then
    bash launch_flow_correlation.sh --gpus "$GPUS" --out-dir "$OUT_ROOT" \
        --cases-dir "$CASES_DIR" --maps "$MAPS" "${EXTRA[@]+"${EXTRA[@]}"}"
    IFS=',' read -r -a MAP_LIST <<< "$MAPS"
    for MAP in "${MAP_LIST[@]}"; do
        SCANS+=("--scan" "$MAP=$OUT_ROOT/$MAP")
    done
fi

CONDA_ENV=${CONDA_ENV:-saliency_r1_qwen3_vllm}
set +u
source "/home/uberger/scratch/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u

echo
echo "##### report -> $OUT_ROOT/report.txt"
python sharpness_report.py "${SCANS[@]}" --json "$OUT_ROOT/sharpness.json" \
    | tee "$OUT_ROOT/report.txt"

echo
echo "[next] the same report restricted to steps whose DINO union stays localised,"
echo "       which is where every map here reads highest:"
echo "       python sharpness_report.py ${SCANS[*]} --max-union 0.5"
