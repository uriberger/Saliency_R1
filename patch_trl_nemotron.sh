#!/usr/bin/env bash
# Install the same in-tree TRL changes into trl_repo_nemotron/ -- the SECOND clone, the
# one the Nemotron-Omni run executes.
#
# WHY A SECOND CLONE. trl_repo/ is shared by every session and every running job, and
# `patch_trl_qwen3.sh` rewrites it in place: re-patching it to try a Nemotron change would
# hand that change to six in-flight Qwen3-VL ranks mid-step. trl_repo_nemotron/ is the
# same fork at the same branch, patched from the same tracked sources, and nothing but the
# Omni launcher points at it. See docs/omni-training-harness.md.
#
# It is still SHARED between worktrees (it is in .worktree-links), so running this from a
# worktree patches it for every session that uses it -- exactly the caution CLAUDE.md
# gives for trl_repo. Today nothing else does.
#
# Unlike patch_trl_qwen3.sh this takes REPO from its own location rather than hardcoding
# the central tree, so running it from a worktree installs THAT worktree's sources. That
# is the point: the Omni port has to be testable before it is merged.
#
# Two files land here that the Qwen3-VL patcher does not copy, because the trainer now
# asks them for the geometry instead of reading `image_grid_thw`:
#
#   vlm_family.py       the per-family seam (grids, image token, decoder layout)
#   nemotron_loader.py  the shims a remote-code Nemotron needs to load at all
#
# Usage:
#   bash patch_trl_nemotron.sh                 # patch trl_repo_nemotron/
#   bash patch_trl_nemotron.sh --verify        # patch, then import-check in `nemotron`
set -euo pipefail

REPO="$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")" && pwd)"
TRL_REPO=${TRL_REPO_NEMOTRON:-$REPO/trl_repo_nemotron}
ENV=${CONDA_ENV:-nemotron}

if [ ! -d "$TRL_REPO" ]; then
    echo "MISSING: $TRL_REPO" >&2
    echo "Create it once, from the central tree, with:" >&2
    echo "  git clone --branch saliency-r1-integration git@github.com:uriberger/trl.git trl_repo_nemotron" >&2
    exit 1
fi

echo "=== patch_trl_nemotron.sh: repo=$REPO  trl_repo=$TRL_REPO ==="
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
cp "$REPO/trl/rewards/roll_null.py"      "$TRL_REPO/trl/rewards/roll_null.py"
cp "$REPO/trl/rewards/overlap_rewards.py" "$TRL_REPO/trl/rewards/overlap_rewards.py"
cp "$REPO/trl/rewards/openai_rewards.py"  "$TRL_REPO/trl/rewards/openai_rewards.py"
cp "$REPO/trl/rewards/__init__.py"        "$TRL_REPO/trl/rewards/__init__.py"
cp "$REPO/trl/scripts/utils.py"           "$TRL_REPO/trl/scripts/utils.py"
cp "$REPO/trl/grpo_vlm_qwen3.py"          "$TRL_REPO/examples/scripts/grpo_vlm_qwen3.py"
echo "  copied overlap-reward files (overlap_steps, roll_null, overlap_rewards, openai_rewards, rewards/__init__, scripts/utils, grpo_vlm_qwen3)"

# ── 2c. roll-null gradient reward support (reward_variant=grad) ─────────────
# grad_maps.py is imported by the trainer as `from .grad_maps import ...`, so it
# lands next to it in trl/trainer/; grad_rewards.py does `from .overlap_rewards
# import ...`, so it lands next to that in trl/rewards/.
cp "$REPO/trl/grad_maps.py"               "$TRL_REPO/trl/trainer/grad_maps.py"
cp "$REPO/trl/rewards/grad_rewards.py"    "$TRL_REPO/trl/rewards/grad_rewards.py"
echo "  copied gradient-reward files (grad_maps, grad_rewards)"

# ── 2d. GLIMPSE grounding reward support (reward_variant=glimpse) ───────────
# Same placement rule as 2c: glimpse_maps.py is imported by the trainer as
# `from .glimpse_maps import ...` and does `from .grad_maps import ...` itself, so
# it lands beside both in trl/trainer/; glimpse_rewards.py imports from
# .overlap_rewards and .grad_rewards, so it lands beside those in trl/rewards/.
cp "$REPO/trl/glimpse_maps.py"            "$TRL_REPO/trl/trainer/glimpse_maps.py"
cp "$REPO/trl/rewards/glimpse_rewards.py" "$TRL_REPO/trl/rewards/glimpse_rewards.py"
echo "  copied GLIMPSE-reward files (glimpse_maps, glimpse_rewards)"

# ── 2e. placebo controls (--placebo roll|random|length) ─────────────────────
# placebo_rewards.py does `from . import overlap_rewards` and `from . import roll_null`
# -- it takes the scored/unscored decision from the real reward rather than reproducing
# it -- so it lands beside both in trl/rewards/. No map module: it scores the attention
# map the trainer already builds.
cp "$REPO/trl/rewards/placebo_rewards.py" "$TRL_REPO/trl/rewards/placebo_rewards.py"
echo "  copied placebo-control file (placebo_rewards)"

# ── 2f. mask-free rewards (--maskfree flatness|mass) ────────────────────────
# Same placement rule as 2e: maskfree_rewards.py does `from . import overlap_rewards`
# (for the --overlap_natural_only gate and the optional --maskfree-parity path), so it
# lands beside it in trl/rewards/. No map module and no DINO: it scores the attention map
# the trainer already builds, and it takes the same reward_funcs slot as --placebo.
#
# THIS LINE IS LOAD-BEARING FOR EVERY `--reward_variant ours` RUN, not just --maskfree
# ones. grpo_trainer_qwen3.py imports is_active/pop_diagnostics from this module at the
# top of its `reward_variant == "ours"` metrics block, BEFORE the is_active() guard --
# the same shape as the placebo block above. Omitting the copy is therefore not "the new
# flag is unavailable", it is an ImportError on the first metrics log of a plain mean_in
# run. It was omitted once (069bd32 added the module and not this line) and the failure
# surfaced only on a second cluster, because the first had the file copied by hand.
# test_import_layout_cpu.py now checks absolute `from trl.<mod> import` targets against
# this script's copy list, so a repeat fails on CPU instead of mid-run.
cp "$REPO/trl/rewards/maskfree_rewards.py" "$TRL_REPO/trl/rewards/maskfree_rewards.py"
echo "  copied mask-free reward file (maskfree_rewards)"

# ── 2i. mismatched-box control (--mismatch_bank) ────────────────────────────
# Same placement rule as 2e and 2f: mismatch_rewards.py does `from . import
# overlap_rewards` -- it reuses that module's _union_mask, _step_score and every knob
# they read, so the control differs from its reference in the boxes and nothing else --
# so it lands beside it in trl/rewards/. No map module and no DINO: the boxes come from
# a bank built offline by build_mismatch_bank.py, and it takes the same reward_funcs slot
# as --placebo and --maskfree.
#
# LOAD-BEARING FOR EVERY `--reward_variant ours` RUN for the same reason 2f is:
# grpo_trainer_qwen3.py imports is_active/pop_diagnostics from this module in its "ours"
# metrics block, BEFORE the is_active() guard. Omitting the copy is not "the new flag is
# unavailable", it is an ImportError on the first metrics log of a plain mean_in run.
cp "$REPO/trl/rewards/mismatch_rewards.py" "$TRL_REPO/trl/rewards/mismatch_rewards.py"
echo "  copied mismatched-box control file (mismatch_rewards)"

# ── 2h. length guard (--length-guard REF_TOKENS) ────────────────────────────
# An ADDITIONAL reward term, not a slot replacement: it applies under every
# --saliency-method and under --reward_variant none, so grpo_vlm_qwen3.py appends it and
# its weight rather than putting it in the overlap reward's position. It imports nothing
# from the other reward modules (it reads completion_ids and no map, no boxes), so it
# lands in trl/rewards/ purely to sit beside them.
#
# LOAD-BEARING FOR EVERY RUN, not just --length-guard ones, for the same reason 2f is:
# grpo_trainer_qwen3.py imports is_active/pop_diagnostics from this module in its metrics
# block, OUTSIDE every reward_variant branch and BEFORE the is_active() guard. Omitting
# the copy is not "the new flag is unavailable", it is an ImportError on the first metrics
# log of any run at all. test_import_layout_cpu.py parses this script's copy list, so a
# missing line here fails on CPU rather than on the second cluster at step 5.
cp "$REPO/trl/rewards/length_guard_rewards.py" "$TRL_REPO/trl/rewards/length_guard_rewards.py"
echo "  copied length-guard reward file (length_guard_rewards)"

# ── 2g. ZeRO-3 generation hooks ─────────────────────────────────────────────
# unwrap_model_for_generation() strips the ZeRO-3 fetch/partition hooks, yields, then
# re-registers them. Upstream re-registers on the normal path only, so anything raised
# inside the yielded block -- a vLLM timeout, an OOM, a reward that throws -- escapes
# with the hooks still off and the policy keeps training with unsharded parameter
# access. The corruption is silent: no error names this function, and the run continues.
# The local copy wraps the yield in try/finally. It is a whole-file copy of an upstream
# module rather than a sed, same as trl/scripts/utils.py, so the next TRL bump shows up
# as a conflict here instead of being silently reverted.
cp "$REPO/trl/models/utils.py"            "$TRL_REPO/trl/models/utils.py"
echo "  copied ZeRO-3 generation-hook fix (models/utils)"


# ── 2j. the per-family geometry seam ────────────────────────────────────────
# grpo_trainer_qwen3.py imports these as `from .vlm_family import family_for` and
# `from .nemotron_loader import ...`, so they land beside it in trl/trainer/. They are
# the reason a third copy of the trainer was not needed: the Omni has no
# `image_grid_thw`, its grid is native-resolution and per-picture, and the trainer asks
# the family for it instead of reading the field. The same two files at the REPO ROOT are
# what the measuring side imports -- one source, two copies, kept in step here.
# The generation server, which has to run under vLLM 0.20 here and 0.11 in trl_repo. Its
# header says what the three differences are; nothing in it is Nemotron-specific. Copied
# only by THIS patcher, so the shared clone's server is left exactly as upstream shipped
# it while in-flight Qwen3-VL runs are using it.
cp "$REPO/trl/scripts/vllm_serve.py"      "$TRL_REPO/trl/scripts/vllm_serve.py"
echo "  copied the version-tolerant generation server (scripts/vllm_serve)"

cp "$REPO/vlm_family.py"                  "$TRL_REPO/trl/trainer/vlm_family.py"
cp "$REPO/nemotron_loader.py"             "$TRL_REPO/trl/trainer/nemotron_loader.py"
echo "  copied the geometry seam (vlm_family, nemotron_loader)"

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

echo "=== Done. trl_repo_nemotron patched. ==="

if [ "${1:-}" = "--verify" ]; then
    echo "=== verifying in conda env '$ENV' ==="
    # shellcheck disable=SC1091
    source /home/uberger/scratch/miniconda3/etc/profile.d/conda.sh
    conda activate "$ENV"
    python - <<'PYEOF'
import trl, pathlib
p = pathlib.Path(trl.__file__).resolve().parent
print("trl package:", p)
assert "trl_repo_nemotron" in str(p), (
    f"`import trl` resolved to {p}, not trl_repo_nemotron. The editable install still "
    "points at the shared clone -- run:\n"
    "  pip install -e <repo>/trl_repo_nemotron --no-deps")
from trl import GRPOTrainerQwen3
from trl.trainer import vlm_family, nemotron_loader
print("GRPOTrainerQwen3:", GRPOTrainerQwen3.__module__)
print("families:", sorted(vlm_family._REGISTRY))
print("loader shims:", [n for n in dir(nemotron_loader) if n.startswith(("_shim", "_vendor", "_repair"))])
PYEOF
fi
