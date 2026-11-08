#!/usr/bin/env bash
# Install the MRoPE guard into deepeyes_fork/, and fix the one dispatch bug it found.
#
#   bash patch_deepeyes_fork.sh                     # default paths
#   bash patch_deepeyes_fork.sh --check             # report state, change nothing
#   bash patch_deepeyes_fork.sh /path/to/fork
#
# deepeyes_fork/ is SHARED between the DeepEyes agents (P2/P3/P5), though NOT with the
# LASER arm -- it is a --no-hardlinks clone of laser_fork with its own object store.
# Running this re-patches it for every DeepEyes session. It does not touch laser_fork.
#
# Idempotent: every edit is skipped if already present, and the first run backs each
# target up as *.orig.
#
# ---------------------------------------------------------------------------
# FOUR EDITS, THREE FILES. The first is a BUG FIX; the other three are tripwires.
# ---------------------------------------------------------------------------
#
# [1] verl/experimental/agent_loop/agent_loop.py  -- THE BUG
#
#     Qwen3-VL's processor_config.json declares image_processor_type
#     "Qwen2VLImageProcessor". So verl's recurring gate
#
#         "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__
#
#     is TRUE on Qwen3-VL -- the gate is not the discriminator, the PROCESSOR class is.
#     verl/utils/dataset/rl_dataset.py:380 knows this and branches on Qwen3VLProcessor.
#     agent_loop.py's `_postprocess` -- the MULTI-TURN path, the one a tool run uses --
#     did not, and imported qwen2_vl's get_rope_index unconditionally. This edit adds
#     the same branch, copied from rl_dataset.py.
#
#     Measured consequence, so the fix is not sold as bigger than it is: for STILL
#     IMAGES the two implementations agree bit-for-bit (test_deepeyes_rope_cpu.py
#     checks this on a real two-image tool turn). They diverge on video, where
#     Qwen3-VL splits video_grid_thw per frame and drops Qwen2.5-VL's
#     tokens_per_second scaling. So this was latent, not live -- but it was latent in
#     exactly the file the whole arm runs through, and nothing would have reported it.
#
# [2] verl/models/transformers/qwen2_vl.py :: get_rope_index
# [3] verl/models/transformers/qwen3_vl.py :: get_rope_index
#
#     One line each, calling `rope_tripwire(processor, <who>)`. [2] RAISES if a
#     Qwen3VLProcessor ever reaches the Qwen2-VL implementation again -- including
#     from verl/workers/rollout/schemas.py:273, the sglang multi-turn path, which is
#     NOT patched here because the vLLM rollout never reaches it and a tripwire is a
#     better guard than a second edit. [3] prints the `[DEEPEYES-MROPE]` line that
#     makes "Qwen3-VL's get_rope_index is what ran" a fact in the log.
#
# [4] verl/models/transformers/qwen3_vl.py :: qwen3_vl_base_forward
#
#     `assert_mrope_position_ids` on every actor and ref forward. This is the one that
#     matters most, because transformers ACCEPTS a 1-D position-id tensor: both
#     Qwen3VLTextModel.forward and Qwen3VLTextRotaryEmbedding.forward do
#     `if position_ids.ndim == 2: position_ids[None, ...].expand(3, ...)`. RoPE then
#     degenerates to text-positional with no error and a healthy loss curve.
#
# All three tripwires are ON by default; DEEPEYES_ASSERT_MROPE=0 downgrades them to a
# warning. Set DEEPEYES_MROPE_DIR=<dir> to collect per-process counts.
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

# The fork lives in the CENTRAL tree. It is not in .worktree-links on main yet (that
# entry is on the unmerged feat/deepeyes-env branch), so a worktree has no symlink for
# it and we resolve upward instead of assuming $REPO/deepeyes_fork exists.
if [ -z "$FORK" ]; then
    if [ -d "$REPO/deepeyes_fork" ]; then
        FORK="$REPO/deepeyes_fork"
    else
        CENTRAL="$(cd "$REPO" && git rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)"
        CENTRAL="${CENTRAL%/.git}"
        FORK="$CENTRAL/deepeyes_fork"
    fi
fi

SRC="$REPO/deepeyes/mrope_guard.py"
DST_GUARD="$FORK/verl/utils/deepeyes_mrope_guard.py"

echo "=== patch_deepeyes_fork.sh: fork=$FORK ==="
[ -f "$SRC" ]  || { echo "MISSING tracked source: $SRC" >&2; exit 1; }
[ -d "$FORK" ] || { echo "MISSING fork: $FORK -- see docs/deepeyes-build-prompts.md Prompt 0" >&2; exit 1; }
[ -f "$FORK/verl/models/transformers/qwen3_vl.py" ] || {
    echo "MISSING $FORK/verl/models/transformers/qwen3_vl.py -- is this really the verl fork?" >&2; exit 1; }
# The one mistake that is unrecoverable from here: patching the copy the LASER arm and
# any in-flight LASER job execute. `|| true` on readlink is load-bearing -- it exits
# non-zero on a path that does not exist, and set -e would kill us before the message.
RESOLVED="$(readlink -f "$FORK" 2>/dev/null || true)"
case "$RESOLVED" in
    */laser_fork) echo "REFUSING: $FORK resolves to laser_fork, which is the LASER arm's copy." >&2; exit 1 ;;
esac

python - "$FORK" "$SRC" "$CHECK" <<'PY'
import shutil
import sys
from pathlib import Path

fork, src, check = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3] == "1"

GUARD_IMPORT = "from verl.utils.deepeyes_mrope_guard import"

EDITS = [
    # (relative path, anchor, replacement, label)
    (
        "verl/experimental/agent_loop/agent_loop.py",
        '        if self.processor is not None and "Qwen2VLImageProcessor" in '
        "self.processor.image_processor.__class__.__name__:\n"
        "            from verl.models.transformers.qwen2_vl import get_rope_index\n",
        '        if self.processor is not None and "Qwen2VLImageProcessor" in '
        "self.processor.image_processor.__class__.__name__:\n"
        "            # Qwen3-VL declares image_processor_type Qwen2VLImageProcessor, so the\n"
        "            # gate above is True for it and cannot be the discriminator. Branch on\n"
        "            # the PROCESSOR class, exactly as verl/utils/dataset/rl_dataset.py:380\n"
        "            # already does -- this call site is the multi-turn one, and it did not.\n"
        '            if "Qwen3VLProcessor" in self.processor.__class__.__name__:\n'
        "                from verl.models.transformers.qwen3_vl import get_rope_index\n"
        "            else:\n"
        "                from verl.models.transformers.qwen2_vl import get_rope_index\n",
        "agent_loop: Qwen3-VL get_rope_index branch",
    ),
    (
        "verl/models/transformers/qwen2_vl.py",
        "    spatial_merge_size = processor.image_processor.merge_size\n",
        "    from verl.utils.deepeyes_mrope_guard import rope_tripwire\n\n"
        '    rope_tripwire(processor, "qwen2_vl")\n\n'
        "    spatial_merge_size = processor.image_processor.merge_size\n",
        "qwen2_vl.get_rope_index: tripwire",
    ),
    (
        "verl/models/transformers/qwen3_vl.py",
        "    spatial_merge_size = processor.image_processor.merge_size\n",
        "    from verl.utils.deepeyes_mrope_guard import rope_tripwire\n\n"
        '    rope_tripwire(processor, "qwen3_vl")\n\n'
        "    spatial_merge_size = processor.image_processor.merge_size\n",
        "qwen3_vl.get_rope_index: marker",
    ),
    (
        "verl/models/transformers/qwen3_vl.py",
        "    input_kwargs = _get_input_embeds(\n",
        "    from verl.utils.deepeyes_mrope_guard import assert_mrope_position_ids\n\n"
        '    assert_mrope_position_ids(kwargs.get("position_ids"), "qwen3_vl_base_forward")\n\n'
        "    input_kwargs = _get_input_embeds(\n",
        "qwen3_vl_base_forward: 4-row position_ids assertion",
    ),
]

dst_guard = fork / "verl/utils/deepeyes_mrope_guard.py"

if check:
    print(f"  guard module : {'present' if dst_guard.exists() else 'ABSENT'}")
    for rel, _anchor, repl, label in EDITS:
        text = (fork / rel).read_text()
        key = repl.split("\n")[0].strip() if GUARD_IMPORT in repl else "Qwen3VLProcessor"
        state = "applied" if key in text else "NOT applied"
        print(f"  {state:<12} {label}")
    raise SystemExit(0)

shutil.copyfile(src, dst_guard)
print(f"  installed deepeyes/mrope_guard.py -> verl/utils/deepeyes_mrope_guard.py")

for rel, anchor, repl, label in EDITS:
    p = fork / rel
    text = p.read_text()
    orig = p.with_suffix(p.suffix + ".orig")
    if not orig.exists():
        shutil.copyfile(p, orig)
    if repl in text:
        print(f"  already applied: {label}")
        continue
    n = text.count(anchor)
    if n != 1:
        raise SystemExit(f"FAIL: anchor for '{label}' appears {n} times in {rel}; refusing to guess")
    p.write_text(text.replace(anchor, repl))
    print(f"  applied: {label}  ({rel})")

# The guards are useless if they cannot be imported. Compile every file we touched.
import py_compile

for rel in sorted({rel for rel, *_ in EDITS} | {"verl/utils/deepeyes_mrope_guard.py"}):
    py_compile.compile(str(fork / rel), doraise=True)
print("  py_compile: all touched files parse")
PY

if [ "$CHECK" = "1" ]; then exit 0; fi

echo
echo "Next:"
echo "  python test_deepeyes_rope_cpu.py        # CPU, no GPU; proves the branch and the shapes"
echo "  bash launch_deepeyes_smoke_job.sh --name deepeyes-smoke   # 8 GPUs, the real thing"
