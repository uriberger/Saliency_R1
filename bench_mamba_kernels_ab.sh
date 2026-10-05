#!/bin/bash
# Do the fused Mamba kernels speed up the Omni's training side? A/B, same node, same job.
#
# THE SETUP, because the switch is not where you would expect it. 23 of this model's 52
# layers are Mamba, and `modeling_nemotron_h.py` prints at every load:
#
#   The fast path is not available because one of `(selective_state_update,
#   causal_conv1d_fn, causal_conv1d_update)` is None. Falling back to the naive
#   implementation.
#
# That is deliberate. `vendor/mamba_ssm_min` is a layernorm-only stub with no dist-info,
# vendored because the decoder raises at IMPORT without `rmsnorm_fn`, and the launcher
# PREPENDS it to PYTHONPATH -- so `import mamba_ssm` finds the stub and the fused ops
# resolve to None.
#
# `causal_conv1d` and `mamba_ssm` are now pip-installed for real (2026-10-04, prebuilt
# wheels for cu12/torch2.8/cp310/cxx11abiTRUE). That install is INERT on its own: the
# remote code reaches the kernels through `transformers.integrations.lazy_load_kernel`,
# which -- because `is_kernels_available()` is False here, so it takes the plain-import
# branch rather than the Hub one -- ends at `importlib.import_module("mamba_ssm")`, and
# the prepended stub still wins. The stub also declares `__version__ =
# "0.0.0+layernorm.only"` so every version gate keeps failing.
#
# So PYTHONPATH IS THE SWITCH, and that is what this script flips:
#
#   arm A  PYTHONPATH=<repo>/vendor/mamba_ssm_min   -> stub wins  -> naive path (today)
#   arm B  PYTHONPATH unset                          -> real wins  -> fused path
#
# One gate is invisible from a login node and worth knowing about: both
# `is_causal_conv1d_available()` and `is_mamba_ssm_available()` are
# `is_torch_cuda_available() and _is_package_available(...)`. On a machine with no GPU
# they are False whatever is installed, so this question can only be asked on a node.
#
# WHAT IS MEASURED. `omni_train_step_bench.py`, which is the training side and nothing
# else: a no-grad saliency re-forward, a forward, a backward, x8 micro-steps, an optimizer
# step, x N. Generation, the judge and the detector are not in it and are not affected by
# these kernels -- vLLM has its own. So the number here maps onto the parts of the
# 222.5 s step that run on a training card: the saliency capture (48.8 s), its straggler
# wait (48.3 s), `compute_loss` (15.7 s) and the backward (~23 s).
#
# Both arms run `--check-grads` (on by default). A fused kernel that is fast and produces
# a non-finite or zero gradient is not a win, and the 23 Mamba layers are exactly what the
# learning signal has to cross to reach the LoRA.
#
# Usage:
#   bash bench_mamba_kernels_ab.sh                 # submit, 1 GPU, 1 h
#   bash bench_mamba_kernels_ab.sh --direct        # on a node you already hold
set -euo pipefail

SCRIPT_PATH="$(realpath "$0")"
REPO="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
CONDA_SH=/home/uberger/scratch/miniconda3/etc/profile.d/conda.sh
export HF_HOME=${HF_HOME:-/home/uberger/scratch/cache/hf_cache}

STEPS=${STEPS:-10}
COMPLETION=${COMPLETION:-768}
DURATION=${DURATION:-1}
DIRECT=false
OUT_DIR=${OUT_DIR:-$REPO/outputs/omni_grpo_plan_a/mamba_kernel_ab}

# FLAGS, NOT AN ENV PREFIX, and this is not style. `submit_job` **execs** the string it is
# given rather than running it through a shell, so `STEPS=10 bash script` dies with
# `exec: STEPS=10: not found` and exit 127 before anything loads (job 7184083). Every
# launcher in this repo passes settings as flags for that reason.
while [[ $# -gt 0 ]]; do
    case "$1" in
        --direct)     DIRECT=true;      shift ;;
        --steps)      STEPS="$2";       shift 2 ;;
        --completion) COMPLETION="$2";  shift 2 ;;
        --out-dir)    OUT_DIR="$2";     shift 2 ;;
        --duration)   DURATION="$2";    shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [ "$DIRECT" != true ]; then
    source "$REPO/cluster_env.sh"
    PARTITION=${PARTITION:-$(SR1_JOB_HOURS=$DURATION sr1_pick_partition)}
    sr1_find_submit_job || { echo "ERROR: submit_job not found" >&2; exit 1; }
    echo "Submitting mamba-kernel A/B to $PARTITION for ${DURATION}h (steps=$STEPS, completion=$COMPLETION)"
    exec submit_job --account nvr_israel_rlop --partition "$PARTITION" \
        --gpu 1 --nodes 1 --duration "$DURATION" --name "omni-mamba-kernel-ab" \
        --command "bash $SCRIPT_PATH --direct --steps $STEPS --completion $COMPLETION --out-dir $OUT_DIR"
fi

# ---------------------------------------------------------------- on the node
mkdir -p "$OUT_DIR"
# shellcheck disable=SC1090
source "$CONDA_SH"; conda activate nemotron
source "$REPO/setup_cuda_home.sh" >/dev/null
cd "$REPO"

export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export SR1_REPO="$REPO"

echo "=============================================================="
echo "node $(hostname)   steps=$STEPS   completion=$COMPLETION"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "=============================================================="

run_arm() {
    local arm="$1" pp="$2" log="$OUT_DIR/$arm.log"
    echo
    echo "########## arm $arm  (PYTHONPATH='${pp:-<unset>}') ##########"
    (
        if [ -n "$pp" ]; then export PYTHONPATH="$pp"; else unset PYTHONPATH; fi
        # Report what the model resolved BEFORE the benchmark, so a run that silently
        # took the other path cannot be mistaken for a speed-up or the lack of one.
        python - <<'PY'
import torch
from transformers.integrations import lazy_load_kernel
from transformers.utils.import_utils import resolve_internal_import
import importlib.util as u
print("  torch.cuda.is_available :", torch.cuda.is_available())
print("  mamba_ssm resolves to   :", (u.find_spec("mamba_ssm") or type("x", (), {"origin": "<none>"})).origin)
cc, ms = lazy_load_kernel("causal-conv1d"), lazy_load_kernel("mamba-ssm")
need = {
    "causal_conv1d_fn": getattr(cc, "causal_conv1d_fn", None),
    "causal_conv1d_update": getattr(cc, "causal_conv1d_update", None),
    "selective_state_update": resolve_internal_import(ms, chained_path="ops.triton.selective_state_update.selective_state_update"),
    "mamba_chunk_scan_combined": resolve_internal_import(ms, chained_path="ops.triton.ssd_combined.mamba_chunk_scan_combined"),
    "mamba_split_conv1d_scan_combined": resolve_internal_import(ms, chained_path="ops.triton.ssd_combined.mamba_split_conv1d_scan_combined"),
}
for k, v in need.items():
    print(f"    {k:36} {'OK' if v else 'None'}")
print("  IS_FAST_PATH_AVAILABLE  :", all(need.values()))
PY
        python omni_train_step_bench.py \
            --bits 16 --grad-ckpt 1 --reforward 1 \
            --completion "$COMPLETION" --steps "$STEPS" \
            --out "$OUT_DIR/$arm.json"
    ) 2>&1 | tee "$log"
    # The model's own verdict, which is the one that cannot be argued with.
    if grep -q "fast path is not available" "$log"; then
        echo "  >>> arm $arm ran the NAIVE Mamba path"
    else
        echo "  >>> arm $arm ran the FUSED Mamba path (no fallback warning)"
    fi
}

# A first: it is the incumbent, so if the node is odd the baseline is the number that
# shows it, rather than the new arm being blamed.
run_arm baseline_naive "$REPO/vendor/mamba_ssm_min"
run_arm fused_kernels  ""

echo
echo "=============================== SUMMARY ==============================="
python - "$OUT_DIR" <<'PY'
import json, pathlib, sys
d = pathlib.Path(sys.argv[1])
rows = {}
for arm in ("baseline_naive", "fused_kernels"):
    f = d / f"{arm}.json"
    if f.exists():
        rows[arm] = json.loads(f.read_text())
    else:
        print(f"  {arm}: NO JSON -- see {arm}.log")
if len(rows) == 2:
    a, b = rows["baseline_naive"], rows["fused_kernels"]
    print(f"  {'':28} {'naive':>10} {'fused':>10} {'speed-up':>10}")
    # median first: a 10-step sample's mean is pulled around by the first step, which is
    # still warming caches. peak_gb matters as much as the time here -- the GRPO run has
    # 0.0 GB of driver memory free at its worst, so a kernel that is fast AND lighter is
    # worth more than the seconds alone say.
    for k in ("median_s", "mean_s", "sd_s", "peak_gb", "load_s"):
        if k in a and k in b and b[k]:
            print(f"  {k:28} {a[k]:>10.3f} {b[k]:>10.3f} {a[k] / b[k]:>9.2f}x")
    print(f"\n  per-step times, naive: {[round(x, 1) for x in a['times']]}")
    print(f"  per-step times, fused: {[round(x, 1) for x in b['times']]}")
PY
echo "logs and json in $OUT_DIR"
