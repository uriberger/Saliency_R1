#!/usr/bin/env bash
# Install the Qwen3-VL attention capture into laser_fork/.
#
# laser_fork/ is gitignored (it is a full external repo, forked from
# github.com/KeViNYuAn0314/LASER onto branch `qwen3-vl`). This script applies the ONE
# change that repo needs to train Qwen3-VL instead of Qwen2.5-VL:
#
#   laser_fork/verl/workers/actor/attention_capture.py
#     <- laser/attention_capture.py   (tracked source, this repo)
#
# That is the whole diff. Everything else in their fork is already Qwen3-VL-aware:
# verl/models/transformers/qwen3_vl.py provides get_rope_index (interleaved MRoPE), the
# input embeds and three backend forwards, and registry.py / monkey_patch.py /
# flops_counter.py / utils/vllm/patch.py / dataset/rl_dataset.py all reference it. Their
# transformers==4.57.6 pin ships Qwen3-VL -- their own qwen3_vl.py imports from
# transformers.models.qwen3_vl.modeling_qwen3_vl, so the pin was never the blocker the
# wiki page thought it was.
#
# WHAT THE REPLACEMENT DOES DIFFERENTLY. Upstream mirrors Qwen2_5_VLAttention.forward
# byte-for-byte to get post-RoPE Q and K, which is what the pin exists for. An attention
# interface function already receives them -- post-RoPE and, on Qwen3-VL, post-QK-norm --
# so the mirror is unnecessary. The replacement wraps the function behind the model's
# EXISTING _attn_implementation entry instead of renaming the implementation, because verl
# branches on that string (padding-free packing, Ulysses SP, FlashAttention kwargs) and
# renaming it would change the forward's semantics while looking like a no-op.
#
# Idempotent: backs up the original as *.orig on first run, then overwrites.
#
# Usage:
#   bash patch_laser_qwen3.sh                       # default paths
#   bash patch_laser_qwen3.sh /path/to/laser_fork
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORK=${1:-$REPO/laser_fork}
SRC="$REPO/laser/attention_capture.py"
DST="$FORK/verl/workers/actor/attention_capture.py"

echo "=== patch_laser_qwen3.sh: fork=$FORK ==="

[ -f "$SRC" ] || { echo "MISSING tracked source: $SRC" >&2; exit 1; }
[ -d "$FORK" ] || {
    echo "MISSING fork: $FORK" >&2
    echo "  git clone https://github.com/KeViNYuAn0314/LASER $FORK" >&2
    echo "  cd $FORK; and git checkout -b qwen3-vl" >&2
    exit 1
}
[ -f "$DST" ] || { echo "MISSING target: $DST -- is this really the LASER fork?" >&2; exit 1; }

if [ ! -f "$DST.orig" ]; then
    cp "$DST" "$DST.orig"
    echo "  backed up upstream -> $(basename "$DST").orig"
fi

cp "$SRC" "$DST"
echo "  installed laser/attention_capture.py -> verl/workers/actor/attention_capture.py"

# The guard that matters: dp_actor.py imports these three names and nothing else from the
# module, so a rename in the replacement would surface as an ImportError only once a
# training job had already claimed its GPUs.
python - "$DST" <<'PY'
import ast, sys
tree = ast.parse(open(sys.argv[1]).read())
top = {b.name for b in tree.body if isinstance(b, (ast.FunctionDef, ast.ClassDef))}
need = {"find_text_attention_modules", "AttentionSliceCapturer", "compute_slice_for_sample"}
missing = need - top
if missing:
    raise SystemExit(f"FAIL: replacement is missing {sorted(missing)}; dp_actor.py imports them")
print("  API check: find_text_attention_modules, AttentionSliceCapturer, "
      "compute_slice_for_sample all present")

# Check what the file DISPATCHES on, not what it mentions. The docstring names
# Qwen2_5_VLAttention on purpose -- saying what is being replaced is its job -- so a
# grep for that string fails on a correct file.
targets = None
for node in tree.body:
    if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == "TEXT_ATTENTION_CLASSES" for t in node.targets):
        targets = [e.value for e in node.value.elts]
if not targets or not all(str(t).startswith("Qwen3VL") for t in targets):
    raise SystemExit(f"FAIL: TEXT_ATTENTION_CLASSES is {targets}; expected Qwen3-VL classes")
print(f"  dispatch check: TEXT_ATTENTION_CLASSES = {targets}")
PY

echo
echo "Next:"
echo "  python test_laser_capture_cpu.py          # CPU, no GPU, no checkpoint"
echo "  then a GPU smoke run -- see docs/laser-fork-qwen3.md"
