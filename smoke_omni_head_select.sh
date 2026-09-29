#!/usr/bin/env bash
# End-to-end smoke test of the Omni head-selection path, on ONE node, in ~40 minutes.
#
# Proves the three things the port changed, on the real model and the real corpus, before
# a full run commits a node for two hours:
#
#   1. `intervene_probe --stage prepare` builds cases at all. The failure it is guarding
#      against is silent: the Omni's chat template opens `<think>` in the PROMPT, so an
#      unported `judge_format` scores every completion of a well-behaved model as
#      malformed and writes a cases file with zero cases and no error.
#   2. `head_correlation_probe --stage scan` finds SIX attention layers, not 52 and not 0,
#      and reads their weights off the module's own output.
#   3. `--stage report` runs with no incumbent row -- layer 22 is a Mamba layer here.
#
#   PARTITION=batch_short DURATION=1 bash smoke_omni_head_select.sh            # submit
#   bash smoke_omni_head_select.sh --direct                                    # on a node
#
# GPU 0 the policy + the step classifier, GPU 1 Grounding-DINO. The detector is served
# rather than local for the reason the GRPO run learned the hard way: a per-process
# Grounding-DINO costs ~8 GB on a card already holding 62 GB of Omni, and the symptom is
# `[dino] CUDA OOM; retrying batch` rather than an error.
set -euo pipefail

SCRIPT_PATH="$(realpath "$0")"
REPO="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
CONDA_SH=/home/uberger/scratch/miniconda3/etc/profile.d/conda.sh
ENV_NAME=nemotron
MODEL=${MODEL:-nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16}
N_SAMPLES=${N_SAMPLES:-24}
DINO_PORT=${DINO_PORT:-8137}
OUT_DIR=${OUT_DIR:-$REPO/outputs/omni_head_select/smoke}
CASES_DIR="$OUT_DIR/cases_probe"
SCAN_DIR="$OUT_DIR/scan_probe"
DIRECT=false
[ "${1:-}" = "--direct" ] && DIRECT=true

if [ "$DIRECT" != true ]; then
    source "$REPO/cluster_env.sh"
    DURATION=${DURATION:-1}
    PARTITION=${PARTITION:-$(SR1_JOB_HOURS=$DURATION sr1_pick_partition)}
    sr1_find_submit_job || { echo "ERROR: submit_job not found" >&2; exit 1; }
    echo "Submitting omni-head-smoke to $PARTITION for ${DURATION}h"
    exec submit_job --account nvr_israel_rlop --partition "$PARTITION" \
        --gpu 2 --nodes 1 --duration "$DURATION" --name omni-head-smoke \
        --command "bash $SCRIPT_PATH --direct"
fi

# =========================================================================
# on the node
# =========================================================================
export HF_HOME=${HF_HOME:-/home/uberger/scratch/cache/hf_cache}
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OVERLAP_STEPS_CKPT=${OVERLAP_STEPS_CKPT:-$REPO/checkpoint/steps_classifier/best}
# nemotron_loader walks up from its own file to find vendor/mamba_ssm_min; SR1_REPO is the
# explicit answer, and without that vendored layernorm-only mamba_ssm the Nemotron decoder
# raises at IMPORT.
export SR1_REPO=$REPO

mkdir -p "$OUT_DIR/logs"
cd "$REPO"
set +u; source "$CONDA_SH"; conda activate "$ENV_NAME"; set -u
[ -n "${CONDA_PREFIX:-}" ] || { echo "ERROR: conda activate $ENV_NAME failed" >&2; exit 1; }
echo "=== node $(hostname), env $CONDA_PREFIX, $(nvidia-smi -L | wc -l) GPUs ==="

step() { echo; echo "=========== $* ==========="; date "+%H:%M:%S"; }

# ---------- the detector, on its own card ----------
step "Grounding-DINO on GPU 1, port $DINO_PORT"
CUDA_VISIBLE_DEVICES=1 DINO_SERVER_BATCH=${DINO_SERVER_BATCH:-8} \
    python serve_grounding_dino.py --host 127.0.0.1 --port "$DINO_PORT" \
    >"$OUT_DIR/logs/dino.log" 2>&1 &
DINO_PID=$!
trap 'kill $DINO_PID 2>/dev/null || true' EXIT
for i in $(seq 1 90); do
    curl -sf "http://127.0.0.1:$DINO_PORT/health" >/dev/null && break
    kill -0 $DINO_PID 2>/dev/null || { echo "ERROR: DINO died"; tail -30 "$OUT_DIR/logs/dino.log"; exit 1; }
    sleep 5
done
curl -sf "http://127.0.0.1:$DINO_PORT/health" || { echo "ERROR: DINO never healthy"; exit 1; }
echo

# ---------- 1. the cases ----------
step "prepare: $N_SAMPLES samples of set_a through the Omni"
CUDA_VISIBLE_DEVICES=0 python intervene_probe.py --stage prepare \
    --out-dir "$CASES_DIR" --base-model "$MODEL" \
    --n-samples "$N_SAMPLES" --shard 0 --num-shards 1 --device cuda:0 \
    --max-new-tokens 768 --log-every 2 \
    --dino-api-base "http://127.0.0.1:$DINO_PORT" 2>&1 | tee "$OUT_DIR/logs/prepare.log"

python - "$CASES_DIR" <<'PY'
import json, sys
from pathlib import Path
d = json.loads(sorted((Path(sys.argv[1]) / "cases").glob("shard*.json"))[0].read_text())
n = len(d["cases"])
print(f"\n[check] {n} cases kept, dropped {d['dropped']}")
if not n:
    raise SystemExit("FAIL: zero cases. That is the silent failure this test exists for "
                     "-- read the `dropped` histogram above: bad_format means the "
                     "<think>-opening prompt is not being handled.")
c = d["cases"][0]
print(f"[check] grid {c['grid']}, {len(c['steps'])} grounded steps, "
      f"chain {len(c['chain_ids'])} tokens")
print(f"[check] answer_text {c.get('answer_text')!r}   gold {c['gold']!r}")
if c.get("answer_text") is None:
    raise SystemExit("FAIL: prepare did not store the model's own answer")
PY

# ---------- 2. the scan ----------
step "scan: every attention layer, every head"
CUDA_VISIBLE_DEVICES=0 python head_correlation_probe.py --stage scan \
    --out-dir "$SCAN_DIR" --cases-dir "$CASES_DIR" --base-model "$MODEL" \
    --shard 0 --num-shards 1 --device cuda:0 --log-every 2 \
    2>&1 | tee "$OUT_DIR/logs/scan.log"

python - "$SCAN_DIR" <<'PY'
import numpy as np, sys
from pathlib import Path
f = sorted((Path(sys.argv[1]) / "scan").glob("shard*.npz"))[0]
d = np.load(f)
L = [int(x) for x in d["layers"]]
print(f"\n[check] layers {L}")
print(f"[check] auroc {d['auroc'].shape} = (steps, layers, heads)")
if L != [5, 12, 19, 26, 33, 42]:
    raise SystemExit(f"FAIL: expected the Omni's six attention layers, got {L}")
au = d["auroc"]
print(f"[check] auroc finite {np.isfinite(au).mean():.3f}, "
      f"mean {np.nanmean(au):.4f} (chance 0.5)")
if not np.isfinite(au).any():
    raise SystemExit("FAIL: every auroc is NaN -- the capture read nothing")
PY

# ---------- 3. the report ----------
step "report: no incumbent row, because 22 is a Mamba layer here"
python head_correlation_probe.py --stage report --out-dir "$SCAN_DIR" \
    --incumbent-layer 19 --incumbent-heads 4,9 2>&1 | tee "$OUT_DIR/logs/report.log"

step "SMOKE PASSED"
echo "cases  $CASES_DIR"
echo "scan   $SCAN_DIR"
