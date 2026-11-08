#!/usr/bin/env bash
# Install the DeepEyes zoom tool into deepeyes_fork/.
#
#   bash patch_deepeyes_zoom_tool.sh            # default paths
#   bash patch_deepeyes_zoom_tool.sh --check    # report state, change nothing
#   bash patch_deepeyes_zoom_tool.sh /path/to/fork
#
# deepeyes_fork/ is SHARED between the DeepEyes agents (P2/P3/P5), though NOT with the
# LASER arm -- it is a --no-hardlinks clone of laser_fork with its own object store.
# Running this re-patches it for every DeepEyes session. It does not touch laser_fork,
# and it refuses to run if the path it is given resolves there.
#
# ---------------------------------------------------------------------------
# ONE FILE, REPLACED WHOLE
# ---------------------------------------------------------------------------
#
#   deepeyes/image_zoom_in_tool.py  ->  verl/tools/image_zoom_in_tool.py
#
# Not an anchored in-place edit like patch_deepeyes_fork.sh's four, because this is not
# a tripwire inserted into someone else's logic -- it is the whole tool, rewritten for
# Qwen3-VL's coordinate frame, its 32 px visual token, and a crop taken from the
# ORIGINAL image. See the module docstring for which three changes and why each one is
# silent when it is wrong.
#
# ---------------------------------------------------------------------------
# WHY THIS IS A SECOND SCRIPT AND NOT A FIFTH EDIT IN patch_deepeyes_fork.sh
# ---------------------------------------------------------------------------
# patch_deepeyes_fork.sh lives on P2's branch (probe/deepeyes-verl-smoke) and this lives
# on P3's (feat/deepeyes-zoom-tool); both are unmerged. Adding the same path on both
# branches buys an add/add conflict for whoever merges second, over a file whose whole
# job is to be run without thinking about it.
#
# So: separate file, no overlap in what they touch, and THIS ONE CHAINS TO THAT ONE when
# it is present. `bash patch_deepeyes_zoom_tool.sh` is therefore the single command that
# leaves the fork fully patched, whichever branches have landed. Once both are on main
# they can be folded into one script; until then, running this is sufficient.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CHECK=0
FORK=""
for a in "$@"; do
    case "$a" in
        --check) CHECK=1 ;;
        *)       FORK="$a" ;;
    esac
done

# The fork lives in the CENTRAL tree; a worktree gets a symlink for it via
# .worktree-links. Resolve upward anyway, so this works from a tree that predates
# that entry.
if [ -z "$FORK" ]; then
    if [ -e "$REPO/deepeyes_fork" ]; then
        FORK="$REPO/deepeyes_fork"
    else
        CENTRAL="$(cd "$REPO" && git rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)"
        CENTRAL="${CENTRAL%/.git}"
        FORK="$CENTRAL/deepeyes_fork"
    fi
fi

SRC="$REPO/deepeyes/image_zoom_in_tool.py"
DST="$FORK/verl/tools/image_zoom_in_tool.py"

echo "=== patch_deepeyes_zoom_tool.sh: fork=$FORK ==="
[ -f "$SRC" ]  || { echo "MISSING tracked source: $SRC" >&2; exit 1; }
[ -d "$FORK" ] || { echo "MISSING fork: $FORK -- see docs/deepeyes-build-prompts.md Prompt 0" >&2; exit 1; }
[ -f "$DST" ]  || { echo "MISSING $DST -- is this really the verl fork?" >&2; exit 1; }
# The one mistake that is unrecoverable from here: patching the copy the LASER arm and
# any in-flight LASER job execute. `|| true` on readlink is load-bearing -- it exits
# non-zero on a path that does not exist, and set -e would kill us before the message.
RESOLVED="$(readlink -f "$FORK" 2>/dev/null || true)"
case "$RESOLVED" in
    */laser_fork) echo "REFUSING: $FORK resolves to laser_fork, which is the LASER arm's copy." >&2; exit 1 ;;
esac

# --- P2's installer, if it has landed. Deliberately before ours: its four edits are in
# --- other files, so order does not matter for correctness, only for the report order.
if [ -f "$REPO/patch_deepeyes_fork.sh" ]; then
    if [ "$CHECK" = "1" ]; then
        bash "$REPO/patch_deepeyes_fork.sh" --check "$FORK" || true
    else
        bash "$REPO/patch_deepeyes_fork.sh" "$FORK"
    fi
    echo
else
    echo "  note: patch_deepeyes_fork.sh (P2's MRoPE guard) is not on this branch; run it too."
fi

if [ "$CHECK" = "1" ]; then
    if cmp -s "$SRC" "$DST"; then
        echo "  applied      image_zoom_in_tool.py (fork matches tracked source)"
    else
        echo "  NOT applied  image_zoom_in_tool.py (fork differs from tracked source)"
        echo "               diff -u $DST $SRC"
    fi
    # What the installed copy would actually do. Printed rather than asserted: an
    # unset bbox_frame is a CONFIG fault, not an install fault, and it is the tool
    # config -- not this file -- that carries it.
    grep -n "^FRAMES\|^DEFAULT_PATCH_SIZE\|^DEFAULT_MERGE_SIZE\|    MIN_DIMENSION" "$DST" | sed 's/^/    /' || true
    exit 0
fi

if [ ! -f "$DST.orig" ]; then
    cp -p "$DST" "$DST.orig"
    echo "  backed up    verl/tools/image_zoom_in_tool.py -> .orig"
fi

if cmp -s "$SRC" "$DST"; then
    echo "  already applied: image_zoom_in_tool.py"
else
    cp -p "$SRC" "$DST"
    echo "  installed    deepeyes/image_zoom_in_tool.py -> verl/tools/image_zoom_in_tool.py"
fi

python - "$DST" <<'PY'
import py_compile
import sys

py_compile.compile(sys.argv[1], doraise=True)
print("  py_compile: verl/tools/image_zoom_in_tool.py parses")
PY

echo
echo "Next:"
echo "  python test_deepeyes_zoom_tool_cpu.py     # CPU, no GPU, no ray; the five cases"
echo
echo "Reminder: the tool config MUST set bbox_frame (rel1000 | abs_original |"
echo "abs_smartresize). There is no default and construction fails without it --"
echo "that is deliberate. P4 (deepeyes_coord_format_probe.py) measures which."
