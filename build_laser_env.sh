#!/usr/bin/env bash
# Build the `laser` conda env for LASER's verl fork (laser_fork/, branch qwen3-vl).
#
#   bash build_laser_env.sh                 # creates/updates the `laser` env
#   bash build_laser_env.sh --name mylaser  # under a different name
#   bash build_laser_env.sh --check         # report what is installed, install nothing
#
# THIS IS A SEPARATE ENV ON PURPOSE. LASER pins transformers==4.57.6 where
# saliency_r1_qwen3_vllm runs 5.13.0.dev0, and the whole reason their attention capture
# was pinned is that transformers moves. Installing LASER's pins into a shared env would
# break every other thing this repo runs -- CLAUDE.md's "treat any pip install as global"
# applies to the ENV, so the answer is a new one, not a careful one.
#
# WHAT IS ALREADY KNOWN TO WORK HERE. saliency_r1_qwen3_vllm already runs torch
# 2.8.0+cu128, vllm 0.11.0 and xformers 0.0.32.post1 -- LASER's exact pins for all three
# -- so those wheels install cleanly on this cluster. The two unknowns are flash-attn
# (installed best-effort, see below) and verl's own `pip install -e .`.
#
# NO vLLM PATCH IS NEEDED. patch_vllm_qwen3.sh exists to make vllm 0.11.0 tolerate
# transformers *5.x*; LASER's env is on 4.x, which is what that vllm was built against.
# Do not run it against this env.
#
# FLASH-ATTN IS OPTIONAL AND IS NOT ALLOWED TO FAIL THE BUILD. requirements_laser.txt
# pins flash-attn==2.8.1 as "prebuilt CUDA 12.8 wheel", and pip will fall back to a
# source build that needs nvcc and the better part of an hour on a login node. verl runs
# on SDPA without it; the attention capture is backend-agnostic by construction (it wraps
# whatever ALL_ATTENTION_FUNCTIONS entry the model is configured with). So it is tried
# last, with a short timeout, and a failure is reported rather than fatal.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORK=${LASER_FORK:-$REPO/laser_fork}
ENV_NAME=laser
CHECK_ONLY=0
CONDA_ROOT=${CONDA_ROOT:-/home/uberger/scratch/miniconda3}
PY_VERSION=3.10          # matches saliency_r1_qwen3_vllm, which is the known-good pair
                         # for torch 2.8.0+cu128 on this cluster

while [[ $# -gt 0 ]]; do
    case "$1" in
        --name)  ENV_NAME="$2"; shift 2 ;;
        --check) CHECK_ONLY=1;  shift   ;;
        --fork)  FORK="$2";     shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

# shellcheck source=/dev/null
source "$CONDA_ROOT/etc/profile.d/conda.sh"

report() {
    conda activate "$ENV_NAME" 2>/dev/null || { echo "env '$ENV_NAME' does not exist"; return 1; }
    python - <<'PY'
import importlib, sys
print(f"  python                {sys.version.split()[0]}")
for name, attr in (("torch", "__version__"), ("transformers", "__version__"),
                   ("vllm", "__version__"), ("verl", "__version__"),
                   ("ray", "__version__"), ("flash_attn", "__version__"),
                   ("datasets", "__version__"), ("numpy", "__version__"),
                   # cv2 is here because pip WILL warn that opencv wants numpy>=2 while
                   # verl pins numpy<2. The warning is about metadata; this line is about
                   # whether it actually imports, which is the question that matters.
                   ("cv2", "__version__"), ("math_verify", None)):
    try:
        m = importlib.import_module(name)
        v = getattr(m, attr, "(installed)") if attr else "(installed)"
    except Exception as exc:
        v = f"MISSING ({type(exc).__name__})"
    print(f"  {name:<21} {v}")
try:
    import torch
    print(f"  torch cuda            {torch.version.cuda}")
except Exception:
    pass
# The one import that proves the port is wired in, not merely present on disk.
try:
    from verl.workers.actor.attention_capture import (   # noqa: F401
        AttentionSliceCapturer, compute_slice_for_sample, find_text_attention_modules,
        TEXT_ATTENTION_CLASSES)
    print(f"  attention capture     OK, targets {TEXT_ATTENTION_CLASSES}")
except Exception as exc:
    print(f"  attention capture     MISSING ({type(exc).__name__}: {exc})")
PY
}

if [[ $CHECK_ONLY -eq 1 ]]; then
    echo "=== $ENV_NAME ==="
    report
    exit $?
fi

[ -d "$FORK" ] || { echo "MISSING fork: $FORK (see docs/laser-fork-qwen3.md)" >&2; exit 1; }
REQ="$FORK/requirements_laser.txt"
[ -f "$REQ" ] || { echo "MISSING: $REQ" >&2; exit 1; }

echo "=========================================================================="
echo "env      : $ENV_NAME  (python $PY_VERSION)"
echo "fork     : $FORK"
echo "conda    : $CONDA_ROOT"
echo "=========================================================================="

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[0/4] env '$ENV_NAME' exists -- reusing it (this script is idempotent)"
else
    echo "[0/4] creating env"
    conda create -y -n "$ENV_NAME" "python=$PY_VERSION" || exit 1
fi
conda activate "$ENV_NAME" || exit 1
python -m pip install --upgrade pip -q

# 1. torch first, from the cu128 index, exactly as their README instructs. Installing it
#    via requirements_laser.txt instead would pull the default PyPI build, which is
#    cu12.x-generic and not what vllm 0.11.0 and xformers were compiled against.
echo "[1/4] torch 2.8.0 / torchvision 0.23.0 / torchaudio 2.8.0 (cu128)"
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128 || exit 1

# 2. everything else, minus two lines.
#
#    flash-attn is deferred to step 4 because it may compile.
#
#    THE NUMPY PIN IS DROPPED HERE AND COMES BACK IN STEP 3, ON PURPOSE.
#    requirements_laser.txt cannot be resolved as written:
#
#        numpy==1.26.4
#        opencv-python-headless==4.12.0.88   ->  metadata says numpy>=2, <2.3.0 on py>=3.9
#
#    pip calls that ResolutionImpossible and refuses the whole file. But numpy 1.26.4 is
#    the RIGHT answer, not the wrong one: verl's own setup.py:32 and requirements.txt:8
#    both say `numpy<2.0.0`, so the framework doing the training is the authority here and
#    their pin agrees with it. Step 3's `pip install -e laser_fork` restores 1.26.4, which
#    is why this only has to get out of the resolver's way rather than pick a side.
#
#    opencv's constraint is metadata, not ABI: `import cv2` works at 4.12.0.88 against
#    numpy 1.26.4 (verified, and re-verified by the report below on every build). pip
#    prints a "dependency conflicts" warning about it at the end of step 3 and that
#    warning is expected and cosmetic. Downgrading to opencv 4.10.0.84 would silence it
#    at the cost of a version nobody has run this stack on.
#
#    `cupy-cuda12x` shows up in the same warning and is an orphan -- `pip show` reports no
#    dependents, nothing in requirements_laser.txt asks for it, and its import failure on
#    a login node is "no GPU", not "wrong numpy".
echo "[2/4] the rest of requirements_laser.txt (flash-attn deferred; numpy pin deferred to verl)"
TMP_REQ=$(mktemp)
grep -v "^flash-attn" "$REQ" | grep -v "^numpy==" > "$TMP_REQ"
python -m pip install -r "$TMP_REQ" || { rm -f "$TMP_REQ"; exit 1; }
rm -f "$TMP_REQ"
echo "      numpy resolved to $(python -c 'import numpy; print(numpy.__version__)')"

# 3. verl itself, editable, so the fork's checkout IS what executes -- the same
#    arrangement trl_repo/ has, and the reason patch_laser_qwen3.sh has anything to patch.
echo "[3/4] pip install -e $FORK"
python -m pip install -e "$FORK" --no-build-isolation || \
    python -m pip install -e "$FORK" || exit 1

# 4. flash-attn, best effort. See the header: not required, and not allowed to fail the
#    build. --no-build-isolation lets it see the torch we just installed if it does have
#    to compile, which is the difference between "slow" and "impossible".
echo "[4/4] flash-attn 2.8.1 (optional)"
if timeout 900 python -m pip install flash-attn==2.8.1 --no-build-isolation 2>&1 | tail -3; then
    echo "      flash-attn installed"
else
    echo "      flash-attn NOT installed -- this is fine. verl runs on SDPA, and the"
    echo "      attention capture wraps whichever backend the model is configured with."
fi

echo
echo "=========================================================================="
echo "installed:"
report
echo "=========================================================================="
echo "Next:"
echo "  bash patch_laser_qwen3.sh                 # if the fork was re-cloned"
echo "  conda activate $ENV_NAME; and python -c 'import verl, transformers; print(verl.__version__, transformers.__version__)'"
echo "  then the GPU smoke run -- docs/laser-fork-qwen3.md §6"
