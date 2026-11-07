#!/usr/bin/env bash
# Install the same in-tree TRL changes into trl_repo_ergo/ -- the THIRD clone, the one the
# ERGO baseline executes.
#
# WHY A THIRD CLONE. ERGO's turn 2 conditions on TWO images (the downsampled original and
# the crop its <zoom> box selects). That is a change to grpo_trainer_qwen3.py's
# `_signature_columns`/`split_pixel_values_by_grid` and to trl/scripts/vllm_serve.py, which
# today takes exactly one base64 image per prompt. trl_repo/ is shared by every session and
# every running job, and `patch_trl_qwen3.sh` rewrites it in place: making the two-image
# change there would hand it to in-flight Qwen3-VL ranks mid-step, and a single-image row
# that silently reshapes is the quietest possible way to corrupt a run. trl_repo_ergo/ is
# the same fork at the same branch, patched from the same tracked sources, and nothing but
# the `ergo` / `ergo_vllm` envs resolves `trl` to it. See docs/ergo-build-prompts.md § P0.
#
# RUNNING THIS RE-PATCHES A SHARED CLONE. trl_repo_ergo is in .worktree-links, so every
# worktree sees the same directory -- exactly the caution CLAUDE.md gives for trl_repo.
# Running this from a worktree installs THAT worktree's sources for every session using the
# `ergo` env, including any ERGO job already in an allocation. Today only the ERGO arm does.
#
# Like patch_trl_nemotron.sh and unlike patch_trl_qwen3.sh, REPO comes from this script's
# own location rather than being hardcoded to the central tree. That is the point: the ERGO
# port has to be testable before it is merged.
#
# THE TWO ENVS, AND THE SEAM BETWEEN THEM.
#   ergo       cloned from saliency_r1_qwen3       torch 2.7.1, transformers 5.13.0.dev0,
#                                                  peft 0.19.1 -- runs the trainer
#   ergo_vllm  cloned from saliency_r1_qwen3_vllm  vLLM 0.11.0 -- runs `trl vllm-serve`
# Both have `pip install -e trl_repo_ergo --no-deps`. The second one is not optional and is
# easy to forget: the generation server imports `trl` in its OWN env, so leaving ERGO on
# saliency_r1_qwen3_vllm would run trl_repo's single-image vllm_serve.py while the trainer
# ran trl_repo_ergo's two-image one. Nothing errors -- the server just never sees the second
# image. vLLM stays at 0.11.0: vllm-project/vllm#29595 reports corrupted Qwen3-VL grounding
# output above it, and <zoom> boxes are grounding output.
#
# WHAT IT COPIES. Our tracked sources into trl_repo_ergo/ and nothing else -- the same list
# patch_trl_nemotron.sh uses, so the three clones hold the same files and one copy list can
# be diffed against another. Steps 1 and 3 are not copies and cannot be: they are in-place
# edits to upstream files (the transformers-5.x availability shim, and registering
# GRPOTrainerQwen3 in the two lazy-import structures). Without them `from trl import
# GRPOTrainerQwen3` fails and the clone is inert.
#
# Usage:
#   bash patch_trl_ergo.sh                 # patch trl_repo_ergo/
#   bash patch_trl_ergo.sh --verify        # patch, then import-check `ergo` AND the two
#                                          # live envs, which must NOT have moved
set -euo pipefail

REPO="$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")" && pwd)"
TRL_REPO=${TRL_REPO_ERGO:-$REPO/trl_repo_ergo}
ENV=${CONDA_ENV:-ergo}

if [ ! -d "$TRL_REPO" ]; then
    echo "MISSING: $TRL_REPO" >&2
    echo "Create it once, from the central tree, with:" >&2
    echo "  git clone --branch saliency-r1-integration git@github.com:uriberger/trl.git trl_repo_ergo" >&2
    echo "NOTE: the fork's saliency-r1-integration head on GitHub is 2 commits BEHIND the" >&2
    echo "local trl_repo (32405e0 vs 8d8de8b -- overlap_steps and the DINO ZeRO-3 guard were" >&2
    echo "never pushed). The existing trl_repo_ergo was seeded from the local trl_repo so it" >&2
    echo "holds the code that actually runs, then had origin pointed back at the fork. A" >&2
    echo "fresh clone from GitHub would silently be those 2 commits short." >&2
    exit 1
fi

echo "=== patch_trl_ergo.sh: repo=$REPO  trl_repo=$TRL_REPO ==="

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
cp "$REPO/trl/rewards/roll_null.py"       "$TRL_REPO/trl/rewards/roll_null.py"
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
# LOAD-BEARING FOR EVERY `--reward_variant ours` RUN, not just --maskfree ones.
# grpo_trainer_qwen3.py imports is_active/pop_diagnostics from this module at the top of
# its `reward_variant == "ours"` metrics block, BEFORE the is_active() guard. Omitting the
# copy is not "the new flag is unavailable", it is an ImportError on the first metrics log
# of a plain mean_in run. See patch_trl_qwen3.sh § 2f for the incident.
cp "$REPO/trl/rewards/maskfree_rewards.py" "$TRL_REPO/trl/rewards/maskfree_rewards.py"
echo "  copied mask-free reward file (maskfree_rewards)"

# ── 2i. mismatched-box control (--mismatch_bank) ────────────────────────────
# LOAD-BEARING FOR EVERY `--reward_variant ours` RUN for the same reason 2f is: the
# trainer imports is_active/pop_diagnostics from it BEFORE the is_active() guard.
cp "$REPO/trl/rewards/mismatch_rewards.py" "$TRL_REPO/trl/rewards/mismatch_rewards.py"
echo "  copied mismatched-box control file (mismatch_rewards)"

# ── 2h. length guard (--length-guard REF_TOKENS) ────────────────────────────
# LOAD-BEARING FOR EVERY RUN AT ALL, not just --length-guard ones: the trainer imports
# is_active/pop_diagnostics from it in its metrics block, OUTSIDE every reward_variant
# branch and BEFORE the is_active() guard.
cp "$REPO/trl/rewards/length_guard_rewards.py" "$TRL_REPO/trl/rewards/length_guard_rewards.py"
echo "  copied length-guard reward file (length_guard_rewards)"

# ── 2g. ZeRO-3 generation hooks ─────────────────────────────────────────────
# unwrap_model_for_generation() strips the ZeRO-3 fetch/partition hooks, yields, then
# re-registers them. Upstream re-registers on the normal path only, so anything raised
# inside the yielded block -- a vLLM timeout, an OOM, a reward that throws -- escapes with
# the hooks still off and the policy keeps training with unsharded parameter access. The
# corruption is silent. The local copy wraps the yield in try/finally.
cp "$REPO/trl/models/utils.py"            "$TRL_REPO/trl/models/utils.py"
echo "  copied ZeRO-3 generation-hook fix (models/utils)"

# ── 2k. the generation server ───────────────────────────────────────────────
# THE FILE ERGO HAS TO CHANGE (docs/ergo-build-prompts.md § P6): ~lines 572-578 take one
# base64 image per prompt and build `multi_modal_data={"image": img}`; ERGO needs
# list[list[str]]. patch_trl_qwen3.sh deliberately does NOT copy this file, so trl_repo's
# server stays exactly as upstream shipped it while in-flight Qwen3-VL runs use it -- which
# is precisely why the ERGO version must land here and nowhere else.
#
# It is served by the `ergo_vllm` env, not `ergo`. If ERGO is ever launched against
# saliency_r1_qwen3_vllm by accident, that env resolves `trl` to trl_repo and the server
# silently reverts to one image per prompt with no error anywhere.
cp "$REPO/trl/scripts/vllm_serve.py"      "$TRL_REPO/trl/scripts/vllm_serve.py"
echo "  copied the generation server (scripts/vllm_serve)"

# ── 2j. the per-family geometry seam ────────────────────────────────────────
# LOAD-BEARING FOR EVERY RUN, Qwen3-VL ones included. grpo_trainer_qwen3.py imports
# `from .vlm_family import family_for` at module scope -- it asks the family for a sample's
# token grid and for which multimodal inputs are row-aligned, instead of reading
# `image_grid_thw` and assuming a 2x2 merge. That seam is where ERGO's second image belongs
# too: "how many grid rows does this sample have" is already a family question here.
#
# nemotron_loader.py is imported lazily (only a `trust_remote_code` checkpoint reaches it)
# and is copied anyway, so the three clones hold the same files and one copy list can be
# checked against another.
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

echo "=== Done. trl_repo_ergo patched. ==="

if [ "${1:-}" = "--verify" ]; then
    echo "=== verifying: '$ENV' points at trl_repo_ergo, and the live envs have NOT moved ==="
    # shellcheck disable=SC1091
    source /home/uberger/scratch/miniconda3/etc/profile.d/conda.sh

    # cd out of the repo first. The tracked source directory is named `trl/`, so with the
    # repo root as cwd `import trl` picks IT up and every env answers the same wrong thing.
    cd /tmp

    check_env() {   # check_env <env> <substring that must be in the trl path>
        local env="$1" want="$2" got
        conda activate "$env"
        got=$(python -c 'import trl, os; print(os.path.dirname(trl.__file__))')
        conda deactivate
        printf '  %-26s %s\n' "$env" "$got"
        case "$got" in
            */"$want"/trl) ;;
            *) echo "FAIL: '$env' resolves trl to $got, expected a path under $want/" >&2
               return 1 ;;
        esac
    }

    check_env "$ENV"                   trl_repo_ergo
    check_env ergo_vllm                trl_repo_ergo
    # The two that every in-flight Qwen3-VL job uses. If either of these moved, a pip
    # install leaked out of the ERGO envs and the running jobs changed underneath them.
    check_env saliency_r1_qwen3        trl_repo
    check_env saliency_r1_qwen3_vllm   trl_repo

    conda activate "$ENV"
    python - <<'PYEOF'
import pathlib
import trl
from trl import GRPOTrainerQwen3
from trl.trainer import vlm_family
print("trl package:", pathlib.Path(trl.__file__).resolve().parent)
print("GRPOTrainerQwen3:", GRPOTrainerQwen3.__module__)
print("families:", sorted(vlm_family._REGISTRY))
PYEOF
    echo "=== verified ==="
fi
