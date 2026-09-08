#!/usr/bin/env bash
# Register the sink-shift model wrapper in an lmms-eval checkout. Idempotent, reversible.
#
#   bash install_lmms_sinkshift.sh [--lmms-eval-dir DIR] [--copy] [--uninstall] [--check]
#
# WHY THIS EXISTS. The wrapper first went in by hand, into one clone, uncommitted. That
# worked until the A100 cluster turned out to have its own checkout on its own filesystem
# (/lustre/fs12, which the H100 cluster cannot even see), and `--model-type
# qwen3_vl_sinkshift` died there with "not found in available models". A change that lives
# only in an untracked clone is a change that exists on exactly one machine.
#
# So the wrapper's source of truth is lmms_eval_plugin/qwen3_vl_sinkshift.py in THIS repo,
# and this script wires a checkout up to it. It does two things, both additive:
#
#   1. Links (or copies) the wrapper to lmms_eval/models/chat/qwen3_vl_sinkshift.py.
#      A symlink by default, so the two clusters cannot drift: editing the file here
#      changes what both of them run.
#   2. Adds ONE key to AVAILABLE_CHAT_TEMPLATE_MODELS in lmms_eval/models/__init__.py.
#
# NOTHING ELSE IS TOUCHED, and that is deliberate. lmms-eval imports only the model it is
# asked for, so a job running `--model qwen3_vl` never opens the new file. Adding a key to
# a dict cannot change what another key resolves to. The registry edit is written to a
# temporary file and renamed into place, so a process reading it mid-write sees the old
# whole file or the new whole file -- never half of one. `--check` re-verifies afterwards
# that `qwen3_vl` still resolves to exactly the class it did before.
#
# The wrapper is inert unless asked for: SINK_SHIFT_ALPHA defaults to 0, which is the
# identity, and it says so loudly in the eval log.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# A symlink into a worktree is a symlink into a directory built to be deleted:
# ./worktree.sh done removes it and every eval on every cluster then fails on a dangling
# path, long after the change that caused it. So the link is always to the central tree,
# which means the wrapper has to be merged before it can be installed from a worktree.
if [[ "$REPO" == */.worktrees/* ]]; then
    CENTRAL=$(cd "$REPO/../.." && pwd)
    echo "NOTE: running from a worktree; linking to the central tree $CENTRAL" >&2
    REPO="$CENTRAL"
fi
SRC="$REPO/lmms_eval_plugin/qwen3_vl_sinkshift.py"

LMMS_EVAL_DIR=${LMMS_EVAL_DIR:-/home/uberger/scratch/research/lmms-eval}
MODE=install
LINK=symlink

while [[ $# -gt 0 ]]; do
    case "$1" in
        --lmms-eval-dir) LMMS_EVAL_DIR="$2"; shift 2 ;;
        --copy)          LINK=copy;          shift ;;
        --uninstall)     MODE=uninstall;     shift ;;
        --check)         MODE=check;         shift ;;
        -h|--help)       sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

[[ -f "$SRC" ]] || { echo "ERROR: no wrapper source at $SRC" >&2; exit 2; }
INIT="$LMMS_EVAL_DIR/lmms_eval/models/__init__.py"
DEST="$LMMS_EVAL_DIR/lmms_eval/models/chat/qwen3_vl_sinkshift.py"
[[ -f "$INIT" ]] || { echo "ERROR: $INIT not found -- is --lmms-eval-dir right?" >&2; exit 2; }

KEY='    "qwen3_vl_sinkshift": "Qwen3_VL_SinkShift",'

# ---------------------------------------------------------------- check
if [[ "$MODE" == check ]]; then
    python - "$LMMS_EVAL_DIR" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from lmms_eval.models import get_model
stock = get_model("qwen3_vl")
print(f"  qwen3_vl           -> {stock.__module__}.{stock.__name__}")
try:
    ours = get_model("qwen3_vl_sinkshift")
except Exception as exc:
    print(f"  qwen3_vl_sinkshift -> NOT REGISTERED ({type(exc).__name__})")
    sys.exit(1)
print(f"  qwen3_vl_sinkshift -> {ours.__module__}.{ours.__name__}")
# The one regression that would matter: the stock wrapper must be untouched.
assert stock.__module__ == "lmms_eval.models.chat.qwen3_vl", stock.__module__
assert issubclass(ours, stock), "the variant must subclass the stock wrapper"
assert ours.is_simple is False, "must resolve as a chat model, like qwen3_vl"
print("  OK: qwen3_vl unchanged, and the variant subclasses it")
PY
    exit $?
fi

# ---------------------------------------------------------------- uninstall
if [[ "$MODE" == uninstall ]]; then
    rm -f "$DEST" && echo "removed $DEST"
    python - "$INIT" "$KEY" <<'PY'
import os, sys, tempfile
init, key = sys.argv[1], sys.argv[2] + "\n"
src = open(init).read()
if key not in src:
    print("  registry key was not present"); raise SystemExit(0)
d = os.path.dirname(os.path.abspath(init))
fd, tmp = tempfile.mkstemp(dir=d, prefix=".init_", suffix=".py")
with os.fdopen(fd, "w") as fh:
    fh.write(src.replace(key, ""))
os.chmod(tmp, os.stat(init).st_mode & 0o777)
os.replace(tmp, init)
print("  removed the registry key")
PY
    exit 0
fi

# ---------------------------------------------------------------- install
echo "lmms-eval : $LMMS_EVAL_DIR"
echo "wrapper   : $SRC"

if [[ "$LINK" == symlink ]]; then
    ln -sfn "$SRC" "$DEST"
    echo "  linked  $DEST -> $SRC"
else
    cp -f "$SRC" "$DEST"
    echo "  copied  $DEST   (a copy can drift from the repo; --copy was asked for)"
fi

python - "$INIT" "$KEY" <<'PY'
import os, sys, tempfile
init, key = sys.argv[1], sys.argv[2] + "\n"
src = open(init).read()
if key in src:
    print("  registry key already present"); raise SystemExit(0)
anchor = "AVAILABLE_CHAT_TEMPLATE_MODELS"
i = src.index(anchor)
mark = '    "qwen3_vl": "Qwen3_VL",\n'
j = src.index(mark, i) + len(mark)
d = os.path.dirname(os.path.abspath(init))
fd, tmp = tempfile.mkstemp(dir=d, prefix=".init_", suffix=".py")
with os.fdopen(fd, "w") as fh:
    fh.write(src[:j] + key + src[j:])
os.chmod(tmp, os.stat(init).st_mode & 0o777)
# Atomic within a filesystem: a concurrent eval reads the old whole file or the new
# whole file. This is a shared clone and other jobs import it while we write.
os.replace(tmp, init)
print("  added the registry key next to qwen3_vl")
PY

echo "verifying:"
bash "$0" --lmms-eval-dir "$LMMS_EVAL_DIR" --check
