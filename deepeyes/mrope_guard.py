"""Make a wrong-MRoPE Qwen3-VL run fail loudly instead of training fine and being wrong.

Installed into `deepeyes_fork/verl/utils/deepeyes_mrope_guard.py` by
`patch_deepeyes_fork.sh`. Four call sites import from it; see that script for the
exact anchors.

WHY THIS FILE EXISTS
--------------------
DeepEyes' own fork has `verl/workers/agent/parallel_env.py:282` gating position-id
computation on

    processor.image_processor.__class__.__name__ == "Qwen2VLImageProcessor"

and falling through to 1-D LM position ids when it is False. We are not porting that
file, but the same shape of bug is reachable here, and the reason it is dangerous is
upstream of verl entirely. `transformers/models/qwen3_vl/modeling_qwen3_vl.py`:

    Qwen3VLTextModel.forward:
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

    Qwen3VLTextRotaryEmbedding.forward:
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

A 1-D position-id tensor is therefore ACCEPTED: it is broadcast into three identical
rows and RoPE degenerates to plain text-positional, with no error, no warning and a
perfectly healthy loss curve. Nothing downstream can tell you it happened. The only
defence is to check at the boundary, which is what this module is.

WHAT IT CHECKS, AND WHERE
-------------------------
1. `rope_tripwire(processor, who)` -- in BOTH `get_rope_index` implementations.
   Qwen3-VL's `processor_config.json` declares `image_processor_type:
   "Qwen2VLImageProcessor"`, so every `"Qwen2VLImageProcessor" in <class name>` gate
   in verl is True for Qwen3-VL. The gate is not the discriminator; the PROCESSOR
   class is. `rl_dataset.py` branches on `Qwen3VLProcessor` and gets it right;
   `agent_loop.py` did not, and `workers/rollout/schemas.py` (sglang) still does not.
   This raises if a Qwen3-VL processor reaches the Qwen2-VL implementation.

2. `assert_mrope_position_ids(position_ids, where)` -- in `qwen3_vl_base_forward`,
   i.e. on every actor and ref forward. Requires a 3-D `(4, *, *)` tensor: verl builds
   `cat([text_row, t, h, w])`, so 4 rows is the signature of the real thing and 2-D is
   the signature of `compute_position_id_with_mask` having won.

Both are ON by default. `DEEPEYES_ASSERT_MROPE=0` downgrades them to a one-line
warning, which is the right setting only if one of them is ever shown to
false-positive. The cost is two attribute reads per micro-batch.

3. `mark_rope(who)` -- records which implementation actually ran, so "Qwen3-VL's
   get_rope_index is what ran" is a thing the log SAYS rather than a thing the code
   review concluded. Prints one `[DEEPEYES-MROPE]` line per process per kind, and if
   `DEEPEYES_MROPE_DIR` is set, drops `mrope.<pid>.json` there at exit with the
   counts. Ray actors are separate processes, hence per-pid files.
"""

from __future__ import annotations

import atexit
import json
import os
import threading
from pathlib import Path

_TAG = "[DEEPEYES-MROPE]"
_lock = threading.Lock()
_counts: dict[str, int] = {}
_announced: set[str] = set()
_dumped = False


def _hard() -> bool:
    """Hard-fail unless explicitly downgraded. Default ON -- see the module docstring."""
    return os.environ.get("DEEPEYES_ASSERT_MROPE", "1") != "0"


def _fail(msg: str) -> None:
    if _hard():
        raise RuntimeError(f"{_TAG} {msg}")
    print(f"{_TAG} WARNING (DEEPEYES_ASSERT_MROPE=0): {msg}", flush=True)


def mark_rope(who: str) -> None:
    """Record that `who`'s get_rope_index ran in this process."""
    with _lock:
        _counts[who] = _counts.get(who, 0) + 1
        first = who not in _announced
        if first:
            _announced.add(who)
    if first:
        print(f"{_TAG} get_rope_index={who} pid={os.getpid()}", flush=True)
        _arm_dump()


def rope_tripwire(processor, who: str) -> None:
    """Raise if a Qwen3-VL processor reached a non-Qwen3-VL `get_rope_index`.

    `who` is the module the call site lives in: "qwen3_vl", "qwen2_vl", "glm4v".
    """
    mark_rope(who)
    if who == "qwen3_vl":
        return
    name = type(processor).__name__
    if "Qwen3VL" in name:
        _fail(
            f"{name} dispatched to verl.models.transformers.{who}.get_rope_index. "
            "Qwen3-VL reuses Qwen2VLImageProcessor, so an image-processor-name gate "
            "cannot tell the two apart -- branch on the PROCESSOR class the way "
            "verl/utils/dataset/rl_dataset.py does. Left alone this produces "
            "Qwen2.5-VL MRoPE for a Qwen3-VL model: identical for a single still "
            "image, wrong the moment a video or a timestamped grid appears, and "
            "silent either way."
        )


def assert_mrope_position_ids(position_ids, where: str) -> None:
    """Require the 4-row interleaved-MRoPE layout at a Qwen3-VL forward."""
    if position_ids is None:
        _fail(f"{where}: position_ids is None -- transformers would synthesise 1-D ids")
        return
    dim = getattr(position_ids, "dim", None)
    if dim is None:
        return  # not a tensor; nothing to say
    shape = tuple(position_ids.shape)
    if position_ids.dim() != 3 or shape[0] != 4:
        _fail(
            f"{where}: position_ids has shape {shape}, expected (4, *, *). "
            "verl builds cat([text_row, t, h, w]); a 2-D tensor here is "
            "compute_position_id_with_mask having won the branch, and "
            "Qwen3VLTextRotaryEmbedding will broadcast it into three identical rows "
            "WITHOUT COMPLAINING. The run would look healthy and be wrong."
        )
        return
    with _lock:
        _counts[f"forward:{where}"] = _counts.get(f"forward:{where}", 0) + 1
        first = where not in _announced
        if first:
            _announced.add(where)
    if first:
        print(f"{_TAG} forward {where}: position_ids {shape} OK (4-row MRoPE)", flush=True)
        _arm_dump()


# --------------------------------------------------------------------------- dump


def _arm_dump() -> None:
    """Dump NOW, and again at exit.

    Dumping only at exit was not enough: the processes that matter are ray actors, and
    a ray actor killed at the end of a rollout does not reliably run atexit handlers.
    Announcements are rare (once per kind per process), so writing eagerly costs
    nothing and makes the file exist while the job is still alive to read it.
    """
    global _dumped
    if os.environ.get("DEEPEYES_MROPE_DIR") and not _dumped:
        _dumped = True
        atexit.register(dump_counts)
    dump_counts()


def dump_counts() -> None:
    d = os.environ.get("DEEPEYES_MROPE_DIR")
    if not d:
        return
    try:
        p = Path(d)
        p.mkdir(parents=True, exist_ok=True)
        with _lock:
            payload = {"pid": os.getpid(), "counts": dict(_counts), "hard": _hard()}
        (p / f"mrope.{os.getpid()}.json").write_text(json.dumps(payload))
    except Exception as e:  # never let bookkeeping kill a training job
        print(f"{_TAG} could not dump counts: {e}", flush=True)


def snapshot() -> dict[str, int]:
    with _lock:
        return dict(_counts)
