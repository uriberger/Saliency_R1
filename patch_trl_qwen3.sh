#!/usr/bin/env bash
# Install the Qwen3-VL-compatible TRL patches into trl_repo/.
#
# trl_repo/ is gitignored (it's a full external repo); this script re-applies
# the three in-tree changes needed to run GRPO with Qwen3-VL on transformers 5.x:
#
#   1. trl_repo/trl/import_utils.py
#      -- Add _pkg_available() shim so _is_package_available() (which changed
#         from returning bool to (bool, version) in transformers 5.x) doesn't make
#         every availability flag truthy.
#
#   2. trl_repo/trl/trainer/grpo_trainer_qwen3.py  (new file, no original to patch)
#      -- Copy from trl/grpo_trainer_qwen3.py (tracked source).
#
#   3. trl_repo/trl/trainer/__init__.py  +  trl_repo/trl/__init__.py
#      -- Register GRPOTrainerQwen3 in both lazy-import structures.
#
# Idempotent: backs up *.orig on first run for import_utils.py;
# the __init__ edits are guarded by a grep check.
#
# Usage:
#   bash patch_trl_qwen3.sh                   # uses saliency_r1_qwen3 env
#   bash patch_trl_qwen3.sh saliency_r1_qwen3
set -euo pipefail

REPO=/home/uberger/scratch/research/saliency_r1
TRL_REPO=$REPO/trl_repo
ENV=${1:-saliency_r1_qwen3}

echo "=== patch_trl_qwen3.sh: env=$ENV  trl_repo=$TRL_REPO ==="

# ── 1. import_utils.py ──────────────────────────────────────────────────────
IU="$TRL_REPO/trl/import_utils.py"
[ -f "$IU" ] || { echo "MISSING: $IU"; exit 1; }

if grep -q '_pkg_available' "$IU"; then
    echo "  (import_utils.py already patched – skipping)"
else
    [ -f "$IU.orig" ] || cp "$IU" "$IU.orig"
    # Insert the _pkg_available shim after the _is_package_available import line,
    # then replace all scalar assignments.
    python3 - "$IU" <<'PYEOF'
import sys, re
path = sys.argv[1]
src = open(path).read()

SHIM = '''

def _pkg_available(name: str) -> bool:
    """Return a plain bool from _is_package_available, compatible with both
    transformers <=4.x (returns bool) and >=5.x (returns (bool, version))."""
    result = _is_package_available(name)
    return result[0] if isinstance(result, tuple) else result
'''

# Insert shim after the _is_package_available import
src = src.replace(
    'from transformers.utils.import_utils import _is_package_available\n',
    'from transformers.utils.import_utils import _is_package_available\n' + SHIM,
    1
)

# Replace scalar assignments (not the tuple-unpack line)
def replace_scalar(m):
    pkg = m.group(1)
    return f'_pkg_available("{pkg}")'

src = re.sub(r'_is_package_available\("([^"]+)"\)(?!\s*,\s*return_version)',
             replace_scalar, src)

open(path, 'w').write(src)
print(f'  patched {path}')
PYEOF
fi

# ── 2. grpo_trainer_qwen3.py ────────────────────────────────────────────────
SRC="$REPO/trl/grpo_trainer_qwen3.py"
DST="$TRL_REPO/trl/trainer/grpo_trainer_qwen3.py"
[ -f "$SRC" ] || { echo "MISSING source: $SRC"; exit 1; }
cp "$SRC" "$DST"
echo "  copied grpo_trainer_qwen3.py -> $DST"

# ── 2b. attention-overlap reward support (reward_variant=ours) ──────────────
# Keep these tracked-source files in sync with the live trl_repo tree.
cp "$REPO/trl/overlap_steps.py"           "$TRL_REPO/trl/trainer/overlap_steps.py"
cp "$REPO/trl/rewards/overlap_rewards.py" "$TRL_REPO/trl/rewards/overlap_rewards.py"
cp "$REPO/trl/rewards/openai_rewards.py"  "$TRL_REPO/trl/rewards/openai_rewards.py"
cp "$REPO/trl/rewards/__init__.py"        "$TRL_REPO/trl/rewards/__init__.py"
cp "$REPO/trl/scripts/utils.py"           "$TRL_REPO/trl/scripts/utils.py"
cp "$REPO/trl/grpo_vlm_qwen3.py"          "$TRL_REPO/examples/scripts/grpo_vlm_qwen3.py"
echo "  copied overlap-reward files (overlap_steps, overlap_rewards, openai_rewards, rewards/__init__, scripts/utils, grpo_vlm_qwen3)"

# ── 3a. trl/trainer/__init__.py ─────────────────────────────────────────────
TINIT="$TRL_REPO/trl/trainer/__init__.py"
[ -f "$TINIT" ] || { echo "MISSING: $TINIT"; exit 1; }
if grep -q 'grpo_trainer_qwen3' "$TINIT"; then
    echo "  (trainer/__init__.py already has grpo_trainer_qwen3 – skipping)"
else
    sed -i 's|"grpo_trainer": \["GRPOTrainer"\],|"grpo_trainer": ["GRPOTrainer"],\n    "grpo_trainer_qwen3": ["GRPOTrainerQwen3"],|' "$TINIT"
    echo "  patched $TINIT"
fi

# ── 3b. trl/__init__.py ─────────────────────────────────────────────────────
TINIT2="$TRL_REPO/trl/__init__.py"
[ -f "$TINIT2" ] || { echo "MISSING: $TINIT2"; exit 1; }
if grep -q 'GRPOTrainerQwen3' "$TINIT2"; then
    echo "  (trl/__init__.py already has GRPOTrainerQwen3 – skipping)"
else
    # Add to _import_structure list
    sed -i 's|"GRPOTrainer",|"GRPOTrainer",\n        "GRPOTrainerQwen3",|' "$TINIT2"
    # Add to TYPE_CHECKING import block
    sed -i 's|        GRPOTrainer,|        GRPOTrainer,\n        GRPOTrainerQwen3,|' "$TINIT2"
    echo "  patched $TINIT2"
fi

echo "=== Done. Verify with: python -c 'from trl import GRPOTrainerQwen3; print(GRPOTrainerQwen3.__module__)' ==="
