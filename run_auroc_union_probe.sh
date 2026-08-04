#!/usr/bin/env bash
# Probe the auroc 50k/set_a run at steps 1000/2000/2500 and report the DINO box-UNION
# size distribution, to pick a --max_union_area default.
#
#   bash run_auroc_union_probe.sh                 # 4 models, 40 prompts, 8 GPUs
#   bash run_auroc_union_probe.sh --n-samples 20 --gpus 4
#   bash run_auroc_union_probe.sh --report-only outputs/overlap_probe/<dir>
#
# Runs on whatever cluster holds the checkpoints; needs no network. Output is one text
# report (a few KB) -- the probe JSON itself can stay where it was produced.
#
# WHY these settings:
#   --overlap-metric auroc   matches the run being diagnosed, so `score` is the reward
#                            that run actually optimised. (auroc_raw is recorded on
#                            every step regardless, so the report works either way.)
#   --max-box-area 0.5       the value that run trained with. The union is measured
#                            AFTER this per-box filter, i.e. exactly the region the
#                            reward scored.
#   NO --max-union-area      nothing may be dropped: we are measuring the distribution
#                            the cap would act on, so the cap must be off.
#   --store-maps (default)   records boxes_raw + grid per step. REQUIRED: _union_mask
#                            returns None at 100% coverage, so without the raw boxes
#                            the fullest unions -- the ones this whole exercise is
#                            about -- are invisible.
#   base_coldstart included  the reference point. "Did the union grow during training?"
#                            is unanswerable from the trained checkpoints alone.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

RUN=checkpoint/grpo-coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged-overlap__wov0.11_2head_trmean_50k_set_a_auroc
N_SAMPLES=40
GPUS=8
OUT_DIR="$REPO/outputs/overlap_probe/auroc_union_$(date +%Y%m%d-%H%M%S)"
REPORT_ONLY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --n-samples)   N_SAMPLES="$2";  shift 2 ;;
        --gpus)        GPUS="$2";       shift 2 ;;
        --out-dir)     OUT_DIR="$2";    shift 2 ;;
        --run)         RUN="$2";        shift 2 ;;
        # Skip the GPU pass and just re-report an existing probe directory.
        --report-only) REPORT_ONLY="$2"; shift 2 ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "$REPORT_ONLY" ]]; then
    # Fail before burning 8 GPUs if a checkpoint is missing or is not a LoRA adapter.
    for step in 1000 2000 2500; do
        [[ -f "$RUN/checkpoint-$step/adapter_config.json" ]] || {
            echo "missing LoRA adapter: $RUN/checkpoint-$step/adapter_config.json" >&2
            echo "(pass --run <training output dir> if the run lives elsewhere here)" >&2
            exit 1
        }
    done

    # NAME=PATH labels: these directories differ only by a trailing number, and a
    # mislabelled column would invert the conclusion about which way the union drifted.
    ADAPTERS="cp1000=$RUN/checkpoint-1000,cp2000=$RUN/checkpoint-2000,cp2500=$RUN/checkpoint-2500"

    echo "=========================================================================="
    echo "Probing base_coldstart + cp1000 + cp2000 + cp2500 on set_a"
    echo "  run      : $RUN"
    echo "  samples  : $N_SAMPLES x 8 generations   GPUs: $GPUS"
    echo "  out dir  : $OUT_DIR"
    echo "=========================================================================="

    bash launch_overlap_probe.sh \
        --n-samples "$N_SAMPLES" \
        --gpus "$GPUS" \
        --out-dir "$OUT_DIR" \
        --no-judge \
        --trained-adapter "$ADAPTERS" \
        --overlap-metric auroc \
        --max-box-area 0.5 \
        --box-threshold 0.10
    REPORT_ONLY="$OUT_DIR"
fi

echo
echo "=========================================================================="
echo "Union-size report -> $REPORT_ONLY/union_size_report.txt"
echo "=========================================================================="
python union_size_report.py "$REPORT_ONLY/probe_merged.json" \
    --caps 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 \
    | tee "$REPORT_ONLY/union_size_report.txt"

echo
echo "Send back just this file (a few KB):"
echo "  $REPORT_ONLY/union_size_report.txt"
