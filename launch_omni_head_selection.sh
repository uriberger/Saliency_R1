#!/usr/bin/env bash
# Select the Omni's saliency LAYER and HEAD PAIR, end to end, on one 8-GPU node.
#
# The reward the Omni GRPO run trains is `--overlap-layer 33 --overlap-heads 28,31`, and
# NEITHER number was selected: 33 is the attention layer nearest Qwen3-VL's layer 22 in
# relative depth, and 28,31 were chosen on Qwen3-VL-8B, where they name two different
# heads of a different model. This is what replaces both.
#
#   bash launch_omni_head_selection.sh                       # submit, 4 h, one node
#   bash launch_omni_head_selection.sh --direct              # on a node already held
#   STAGE=scan bash launch_omni_head_selection.sh            # just the scan, cases exist
#
# THE SEARCH IS 192 CELLS, NOT 1,152. `hybrid_override_pattern` gives the Omni an
# attention matrix at only 6 of its 52 decoder layers -- [5, 12, 19, 26, 33, 42] -- so
# 6 x 32 heads is the whole space. That also dissolves the layer question: the scan covers
# every attention layer there is, so the LAYER comes out of the same parity split as the
# heads instead of being argued for from depth.
#
# THE LAYOUT, and why the detector gets a card of its own:
#
#   GPU 0-6   7 prepare shards, one whole 62 GB copy of the Omni each
#   GPU 7     Grounding-DINO
#
# A per-shard detector costs ~8 GB on a card already holding 62, and the GRPO run measured
# what that does: peak 74.1 GB against 66.4, and 123 `[dino] CUDA OOM; retrying batch`
# lines in half an hour with the detector's own card idle. The scan needs no detector, so
# it uses all 8.
#
# --max-union IS PRE-REGISTERED AT 0.5, before any Omni number exists. Every map measured
# so far reads lower the larger the DINO union gets (r(union, auroc) = -0.55 over all 1,152
# Qwen3-VL heads) and the median step's union covers 54% of the grid, so the level is not
# comparable across union sizes -- and a threshold chosen after seeing the result is a
# researcher degree of freedom on a confirmation set that is single use. 0.5 is what the
# Qwen3-VL report used and what launch_head_correlation.sh's own `[next]` line names.
#
# READ THE PARITY SPLIT, NOT THE RANKING. Heads are ranked on odd-indexed rows and
# re-scored on even ones. A head that survives is a candidate; one that does not is
# selection noise -- and on 192 cells that is a far stronger statement than on 1,152.
set -euo pipefail

SCRIPT_PATH="$(realpath "$0")"
REPO="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
CONDA_SH=/home/uberger/scratch/miniconda3/etc/profile.d/conda.sh
ENV_NAME=${CONDA_ENV:-nemotron}     # transformers 5.13 + the shims; NOT nemotron_vllm
MODEL=${MODEL:-nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16}
DATASET=${DATASET:-$REPO/cold_data/grpo_sets/set_a}
N_SAMPLES=${N_SAMPLES:-1000}        # the Qwen3-VL corpus size, so the rows are comparable
MAX_UNION=${MAX_UNION:-0.5}         # PRE-REGISTERED -- see above; do not move after a look
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-768}
DINO_PORT=${DINO_PORT:-8137}
STAGE=${STAGE:-all}                 # all | prepare | scan | report
RUN=${RUN:-setA}
OUT_DIR=${OUT_DIR:-$REPO/outputs/omni_head_select/$RUN}
CASES_DIR="$OUT_DIR/cases"
SCAN_DIR="$OUT_DIR/scan"
DIRECT=false
[ "${1:-}" = "--direct" ] && DIRECT=true

if [ "$DIRECT" != true ]; then
    source "$REPO/cluster_env.sh"
    # 4 h, not 2: `prepare` is resumable only at SHARD granularity, so a wall that lands
    # mid-shard throws that shard's generation away. batch_short's 2 h cap is exactly the
    # wrong size for ~1.6 h of generation plus a 62 GB load.
    DURATION=${DURATION:-4}
    PARTITION=${PARTITION:-$(SR1_JOB_HOURS=$DURATION sr1_pick_partition)}
    sr1_find_submit_job || { echo "ERROR: submit_job not found" >&2; exit 1; }
    echo "Submitting omni-head-select/$RUN to $PARTITION for ${DURATION}h"
    exec submit_job --account nvr_israel_rlop --partition "$PARTITION" \
        --gpu 8 --nodes 1 --duration "$DURATION" --name "omni-head-select-$RUN" \
        --command "STAGE=$STAGE N_SAMPLES=$N_SAMPLES RUN=$RUN bash $SCRIPT_PATH --direct"
fi

# =========================================================================
# on the node
# =========================================================================
export HF_HOME=${HF_HOME:-/home/uberger/scratch/cache/hf_cache}
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OVERLAP_STEPS_CKPT=${OVERLAP_STEPS_CKPT:-$REPO/checkpoint/steps_classifier/best}
# Without vendor/mamba_ssm_min the Nemotron decoder raises at IMPORT, and the search for it
# walks up from nemotron_loader.py's own path -- which is right here, but SR1_REPO is the
# answer that does not depend on that.
export SR1_REPO=$REPO
export CONDA_ENV=$ENV_NAME          # what the two sharding launchers read

mkdir -p "$OUT_DIR/logs"
cd "$REPO"
echo "=========================================================================="
echo "node       $(hostname)   $(nvidia-smi -L | wc -l) GPUs"
echo "model      $MODEL"
echo "dataset    $DATASET   $N_SAMPLES samples"
echo "stage      $STAGE"
echo "out        $OUT_DIR"
echo "max-union  $MAX_UNION  (pre-registered)"
echo "=========================================================================="

step() { echo; echo "=========== $* ==========="; date "+%F %H:%M:%S"; }

start_dino() {
    step "Grounding-DINO on GPU 7, port $DINO_PORT"
    (
        # shellcheck disable=SC1090
        set +u; source "$CONDA_SH"; conda activate "$ENV_NAME"; set -u
        CUDA_VISIBLE_DEVICES=7 DINO_SERVER_BATCH=${DINO_SERVER_BATCH:-8} \
            exec python "$REPO/serve_grounding_dino.py" --host 127.0.0.1 --port "$DINO_PORT"
    ) >"$OUT_DIR/logs/dino.log" 2>&1 &
    DINO_PID=$!
    trap 'kill $DINO_PID 2>/dev/null || true' EXIT
    for _ in $(seq 1 90); do
        curl -sf "http://127.0.0.1:$DINO_PORT/health" >/dev/null && return 0
        kill -0 $DINO_PID 2>/dev/null || {
            echo "ERROR: the detector died:" >&2
            tail -30 "$OUT_DIR/logs/dino.log" >&2; exit 1; }
        sleep 5
    done
    echo "ERROR: the detector never became healthy" >&2
    tail -30 "$OUT_DIR/logs/dino.log" >&2
    exit 1
}

# ---------- 1. the cases: chains, observe steps, per-step DINO unions ----------
if [ "$STAGE" = all ] || [ "$STAGE" = prepare ]; then
    start_dino
    step "prepare: $N_SAMPLES samples, 7 shards on GPUs 0-6"
    bash "$REPO/launch_intervene_probe.sh" --stage prepare \
        --gpus 7 --n-samples "$N_SAMPLES" --out-dir "$CASES_DIR" \
        --dataset "$DATASET" --base-model "$MODEL" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --dino-api-base "http://127.0.0.1:$DINO_PORT"
    kill "$DINO_PID" 2>/dev/null || true
    trap - EXIT
    python - "$CASES_DIR" <<'PY'
import json, sys
from collections import Counter
from pathlib import Path
tot, dropped = 0, Counter()
for f in sorted((Path(sys.argv[1]) / "cases").glob("shard*.json")):
    d = json.loads(f.read_text())
    tot += len(d["cases"])
    dropped.update(d["dropped"])
steps = sum(len(c["steps"]) for f in sorted((Path(sys.argv[1]) / "cases").glob("shard*.json"))
            for c in json.loads(f.read_text())["cases"])
print(f"\n[cases] {tot} cases, {steps} grounded observe steps, dropped {dict(dropped)}")
if not tot:
    raise SystemExit(
        "FAIL: zero cases and no error -- read the `dropped` histogram. `bad_format` on "
        "every row means the <think>-opening prompt is not being handled, which is the "
        "one failure of this stage that looks like a working run.")
PY
fi

# ---------- 2. the scan: every attention layer, every head ----------
if [ "$STAGE" = all ] || [ "$STAGE" = scan ]; then
    step "scan: 8 shards, all 8 GPUs (no detector needed here)"
    bash "$REPO/launch_head_correlation.sh" --gpus 8 \
        --out-dir "$SCAN_DIR" --cases-dir "$CASES_DIR" --base-model "$MODEL"
fi

# ---------- 3. the report ----------
step "report at the pre-registered --max-union $MAX_UNION"
set +u; source "$CONDA_SH"; conda activate "$ENV_NAME"; set -u
python "$REPO/head_correlation_probe.py" --stage report --out-dir "$SCAN_DIR" \
    --max-union "$MAX_UNION" --incumbent-layer 33 --incumbent-heads 28,31 \
    2>&1 | tee "$OUT_DIR/logs/report_maxunion${MAX_UNION}.log"

step "DONE"
echo "The pick is the head whose r(select) sign SURVIVES on r(HELD OUT), not the top of"
echo "the ranking. Then re-run the training arm:"
echo "  bash launch_grpo_omni_overlap_job.sh --overlap-layer <L> --overlap-heads <h,h>"
