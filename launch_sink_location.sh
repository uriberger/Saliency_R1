#!/usr/bin/env bash
# Run sink_location_probe.py sharded across an interactive node's GPUs.
#
#   bash launch_sink_location.sh --stage corpus   --out-dir DIR
#   bash launch_sink_location.sh --stage selftest --gpus 1 --out-dir DIR --model M
#   bash launch_sink_location.sh --stage scan     --gpus 8 --out-dir DIR --model M
#   bash launch_sink_location.sh --stage arms     --gpus 8 --out-dir DIR --model M
#   python sink_location_probe.py --stage report --out-dir DIR
#
# SELFTEST GATES scan AND arms and is not optional. It checks that the scan reproduces
# stock SDPA (it edits nothing, and that has to be measured), that the ring's area is what
# the formula says, that the negative controls come back at 1.0 -- and, above all, that
# EVERY TRANSFORM'S PIXEL-TO-PATCH MAPPING decodes where `patch_correspondence` claims. An
# off-by-one there answers the content-versus-position question confidently and backwards,
# and no later table would look wrong.
#
# RUN corpus FIRST, on CPU. It is the only stage that touches `datasets`, so the GPU
# stages are deterministic, offline and shard trivially. It is also where the grid census
# comes from, and the grid census is what stops a cross-type comparison of raw ring
# percentages: the ring is 23% of a 16x16 grid and 50% of a 6x8 one.
#
# COST. One picture is one prefill at batch size 1, well under a second, so the whole scan
# is minutes rather than hours -- this experiment needs no generation, no Grounding-DINO
# and no judge. Budget roughly:
#     corpus  ~1 h on a CPU node          scan  ~10 min on 8 GPUs (1,800 pictures)
#     arms    ~30 min on 8 GPUs           report instant
#
# Resuming: re-run the identical command. Results are append-only JSONL keyed by unit, and
# the bulk arrays are flushed in parts, so a killed shard loses at most one part.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

GPUS=8
STAGE=scan
OUT_DIR=""
MODEL=""
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)     GPUS="$2";    shift 2 ;;
        --stage)    STAGE="$2";   shift 2 ;;
        --out-dir)  OUT_DIR="$2"; shift 2 ;;
        --model)    MODEL="$2";   shift 2 ;;
        *)          EXTRA+=("$1"); shift  ;;
    esac
done

[[ -n "$OUT_DIR" ]] || { echo "--out-dir is required" >&2; exit 2; }
if [[ "$STAGE" == "selftest" || "$STAGE" == "scan" || "$STAGE" == "arms" ]]; then
    [[ -n "$MODEL" ]] || { echo "--model is required for stage $STAGE" >&2; exit 2; }
fi

CONDA_ENV=${CONDA_ENV:-saliency_r1_qwen3_vllm}
set +u
source "/home/uberger/scratch/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u

export HF_HOME=${HF_HOME:-/home/uberger/scratch/cache/hf_cache}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/progress"

echo "=========================================================================="
echo "Stage     : $STAGE"
echo "Out dir   : $OUT_DIR"
echo "Model     : ${MODEL:-(none)}"
echo "Shards    : $GPUS"
echo "Extra     : ${EXTRA[*]:-(none)}"
echo "=========================================================================="

COMMON=(--out-dir "$OUT_DIR")
[[ -n "$MODEL" ]] && COMMON+=(--model "$MODEL")

if [[ "$STAGE" == "corpus" || "$STAGE" == "selftest" ]]; then
    # corpus is CPU-only; selftest deliberately runs on ONE GPU, because a check that
    # passed on some shards and not others is not a gate.
    CUDA_VISIBLE_DEVICES=0 python sink_location_probe.py --stage "$STAGE" \
        "${COMMON[@]}" "${EXTRA[@]+"${EXTRA[@]}"}" 2>&1 \
        | tee "$OUT_DIR/logs/$STAGE.log"
    exit "${PIPESTATUS[0]}"
fi

if [[ "$STAGE" == "report" ]]; then
    python sink_location_probe.py --stage report --out-dir "$OUT_DIR" \
        "${EXTRA[@]+"${EXTRA[@]}"}" | tee "$OUT_DIR/report.txt"
    exit 0
fi

if [[ ! -f "$OUT_DIR/logs/selftest.log" ]] || \
   ! grep -q "SELFTEST PASS" "$OUT_DIR/logs/selftest.log"; then
    echo "refusing to run: no passing selftest in $OUT_DIR/logs/selftest.log" >&2
    echo "  bash launch_sink_location.sh --stage selftest --gpus 1 --out-dir $OUT_DIR --model $MODEL" >&2
    exit 2
fi

if [[ ! -f "$OUT_DIR/corpus/manifest.jsonl" ]]; then
    echo "refusing to run: no corpus at $OUT_DIR/corpus/manifest.jsonl" >&2
    echo "  bash launch_sink_location.sh --stage corpus --out-dir $OUT_DIR" >&2
    exit 2
fi

# Drop the previous attempt's heartbeats: a shard writes its first only after the model
# loads, so on a resume the monitor reads the dead run's files, calls them stale and exits
# while the shards it was watching are fine. Resume state is in the results files.
rm -f "$OUT_DIR"/progress/*.json

pids=()
for ((i = 0; i < GPUS; i++)); do
    CUDA_VISIBLE_DEVICES="$i" python sink_location_probe.py \
        --stage "$STAGE" --shard "$i" --num-shards "$GPUS" \
        "${COMMON[@]}" "${EXTRA[@]+"${EXTRA[@]}"}" \
        >"$OUT_DIR/logs/${STAGE}_shard${i}.log" 2>&1 &
    pids+=($!)
    echo "[launch] shard $i -> GPU $i (pid ${pids[-1]})"
done

sleep 5
python sink_location_probe.py --stage monitor --out-dir "$OUT_DIR" &
mon=$!

fail=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "[done] shard $i ok"
        grep -v "Loading weights" "$OUT_DIR/logs/${STAGE}_shard${i}.log" | tail -2 | sed 's/^/        /'
    else
        echo "[FAIL] shard $i -- see $OUT_DIR/logs/${STAGE}_shard${i}.log" >&2
        tail -20 "$OUT_DIR/logs/${STAGE}_shard${i}.log" >&2 || true
        fail=1
    fi
done
kill "$mon" 2>/dev/null || true
wait "$mon" 2>/dev/null || true

if [[ $fail -eq 0 ]]; then
    python sink_location_probe.py --stage report --out-dir "$OUT_DIR" \
        | tee "$OUT_DIR/report.txt" || true
else
    echo "WARNING: a shard failed; re-run the identical command to resume." >&2
    exit 1
fi
