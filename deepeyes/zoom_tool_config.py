#!/usr/bin/env python
"""One source of truth for how the DeepEyes zoom tool is configured and fed.

Three things live here, and they live together because each of them is a number that
would otherwise be re-decided, differently, in the data builder (P6), the launcher
(P7) and the smoke script (P9):

  1. `build_tool_config`  -- the tool YAML verl's `initialize_tools_from_config` reads.
  2. `build_create_kwargs`-- the `extra_info.tools_kwargs` block, built so that the
                             tool's crop-source assertion passes BECAUSE it is true,
                             not because the two fields were copied from one variable.
  3. the turn limits      -- `max_turns` is 5, see below.

This module imports nothing from verl and nothing from torch, so a data-conversion job
on `cpu_short` can import it.

---------------------------------------------------------------------------
THE max_turns DECISION, SETTLED HERE SO THE LAUNCHER DOES NOT RE-LITIGATE IT
---------------------------------------------------------------------------
The paper (arXiv:2505.14362) says "up to a maximum of 6 times of active perceptions".
Their released training script `examples/agent/final_merged_v1v8_thinklite.sh` sets
`actor_rollout_ref.rollout.agent.max_turns=5`, and their rollout loop counts GENERATE
rounds, not tool calls:

    verl/workers/agent/parallel_env.py:181   for step in range(config.agent.max_turns):
    verl/workers/agent/parallel_env.py:226   if done or step == config.agent.max_turns - 1:
                                                 active_mask[idx] = False   # no tool

so the last generate round's tool call is never executed. `max_turns=5` is therefore
5 generate rounds and AT MOST 4 tool calls -- not 6 perceptions.

WE FOLLOW THE CODE, NOT THE SENTENCE: 5 generate rounds, 4 tool calls. It is what the
released checkpoint was trained under, and P1 scores that checkpoint, so the two arms
of the comparison stay on the same budget. The deviation from the paper's prose is
stated in docs/deepeyes-zoom-tool.md.

Modern verl spells the same budget with two keys rather than one
(`verl/experimental/agent_loop/tool_agent_loop.py:239,241`):

    actor_rollout_ref.rollout.multi_turn.max_assistant_turns = 5   # generate rounds
    actor_rollout_ref.rollout.multi_turn.max_user_turns      = 4   # tool rounds

`max_assistant_turns` alone reproduces their loop exactly -- the check runs AFTER the
generate and BEFORE the tool call, same as theirs. `max_user_turns` is set to the
implied 4 as a belt-and-braces bound; it binds at the same place.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any, Optional

# --------------------------------------------------------------------------------------
# Turn budget. See the module docstring.
# --------------------------------------------------------------------------------------
MAX_ASSISTANT_TURNS = 5  # generate rounds; == DeepEyes' agent.max_turns=5
MAX_USER_TURNS = 4  # tool rounds; the implied bound, stated rather than inferred
MAX_TOOL_CALLS = MAX_ASSISTANT_TURNS - 1  # 4. What the paper calls "active perceptions".

# --------------------------------------------------------------------------------------
# Qwen3-VL geometry. patch 16 * merge 2 = 32 px per visual token (Qwen2.5-VL: 14*2 = 28).
#
# IMAGE_PATCH_SIZE is ALSO what `data.image_patch_size` must be set to in the launcher.
# verl defaults that key to 14 (verl/utils/dataset/rl_dataset.py:108,
# verl/trainer/config/data/legacy_data.yaml:71) and derives it from nothing, so on
# Qwen3-VL the dataloader silently smart-resizes the prompt image on a 28 px grid unless
# the launcher says otherwise. The tool's crop-source check compares its own view
# computation against the one here and fails the run if they disagree.
# --------------------------------------------------------------------------------------
IMAGE_PATCH_SIZE = 16
MERGE_SIZE = 2
PATCH_FACTOR = IMAGE_PATCH_SIZE * MERGE_SIZE  # 32
#: qwen_vl_utils' token bounds; pixel bounds are these times PATCH_FACTOR**2.
VIEW_MIN_TOKENS = 4
VIEW_MAX_TOKENS = 16384

TOOL_NAME = "image_zoom_in_tool"
TOOL_CLASS = "verl.tools.image_zoom_in_tool.ImageZoomInTool"

#: DeepEyes' own schema, which is what the released checkpoint saw. Two faults of
#: theirs are preserved deliberately -- `"required": ["bbox_2d"]` is ours and correct,
#: but the wording is theirs -- because the prompt the model was trained under is not
#: a thing to improve halfway through a replication.
TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": (
            "Zoom in on a specific region of an image by cropping it based on a bounding box "
            "(bbox) and an optional object label."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "bbox_2d": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 4,
                    "maxItems": 4,
                    "description": (
                        "The bounding box of the region to zoom in, as [x1, y1, x2, y2], where "
                        "(x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner."
                    ),
                },
                "label": {
                    "type": "string",
                    "description": "The name or label of the object in the specified bounding box (optional).",
                },
            },
            "required": ["bbox_2d"],
        },
    },
}


def tool_config(
    *,
    bbox_frame: str,
    log_dir: Optional[str] = None,
    log_first_n_calls: int = 20,
    log_crops: int = 0,
    crop_source_check: str = "strict",
    min_expected_downscale: Optional[float] = None,
    num_workers: int = 8,
    rate_limit: int = 32,
    **extra: Any,
) -> dict:
    """The `tools:` block. `bbox_frame` is keyword-only and has NO default anywhere in
    this codebase -- see `verl/tools/image_zoom_in_tool.py` [1]. P4 measures it."""
    cfg = {
        "type": "native",
        "bbox_frame": bbox_frame,
        "patch_size": IMAGE_PATCH_SIZE,
        "merge_size": MERGE_SIZE,
        "view_patch_size": IMAGE_PATCH_SIZE,
        "view_merge_size": MERGE_SIZE,
        "view_min_tokens": VIEW_MIN_TOKENS,
        "view_max_tokens": VIEW_MAX_TOKENS,
        "crop_source_check": crop_source_check,
        "log_first_n_calls": log_first_n_calls,
        "log_crops": log_crops,
        "num_workers": num_workers,
        "rate_limit": rate_limit,
    }
    if log_dir:
        cfg["log_dir"] = str(log_dir)
    if min_expected_downscale is not None:
        cfg["min_expected_downscale"] = float(min_expected_downscale)
    cfg.update(extra)
    return {"tools": [{"class_name": TOOL_CLASS, "config": cfg, "tool_schema": TOOL_SCHEMA}]}


def build_tool_config(path: str | Path, **kwargs: Any) -> Path:
    """Write the tool YAML and return its path."""
    import yaml

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(tool_config(**kwargs), sort_keys=False))
    return path


# --------------------------------------------------------------------------------------
# The data side.
# --------------------------------------------------------------------------------------
def smart_resize_wh(width: int, height: int, factor: int = PATCH_FACTOR,
                    min_tokens: int = VIEW_MIN_TOKENS, max_tokens: int = VIEW_MAX_TOKENS) -> tuple[int, int]:
    """`qwen_vl_utils.smart_resize`, returned as (width, height).

    Imported lazily so this module stays importable without qwen_vl_utils; falls back
    to a local transcription if it is missing, which is what lets a data job on a
    plain CPU node compute the view size.
    """
    try:
        from qwen_vl_utils.vision_process import smart_resize
    except Exception:  # pragma: no cover - only on a node without qwen_vl_utils
        import math

        def smart_resize(h, w, factor, min_pixels=None, max_pixels=None):  # type: ignore[misc]
            h_bar = max(factor, round(h / factor) * factor)
            w_bar = max(factor, round(w / factor) * factor)
            if max_pixels and h_bar * w_bar > max_pixels:
                beta = math.sqrt((h * w) / max_pixels)
                h_bar = math.floor(h / beta / factor) * factor
                w_bar = math.floor(w / beta / factor) * factor
            elif min_pixels and h_bar * w_bar < min_pixels:
                beta = math.sqrt(min_pixels / (h * w))
                h_bar = math.ceil(h * beta / factor) * factor
                w_bar = math.ceil(w * beta / factor) * factor
            return h_bar, w_bar

    h_bar, w_bar = smart_resize(
        height, width, factor=factor,
        min_pixels=min_tokens * factor**2, max_pixels=max_tokens * factor**2,
    )
    return int(w_bar), int(h_bar)


def build_create_kwargs(
    raw_image: Any,
    *,
    image_patch_size: int = IMAGE_PATCH_SIZE,
    merge_size: int = MERGE_SIZE,
    view_min_tokens: int = VIEW_MIN_TOKENS,
    view_max_tokens: int = VIEW_MAX_TOKENS,
    as_data_uri: bool = True,
) -> dict:
    """Build `tools_kwargs[TOOL_NAME]["create_kwargs"]` from the RAW image.

    `raw_image` must be the un-processed image -- the PIL object decoded straight from
    the parquet `images` column's bytes, or those bytes, or a path. NOT the output of
    `verl/utils/dataset/vision_utils.py::process_image`, which is already smart-resized
    and is what the model is shown.

    The point of this helper is that `original_size` and `view_size` are derived HERE,
    from the raw image, by two different computations. A hand-rolled conversion that
    assigns both from the same already-resized variable passes its own assertions and
    silently makes the zoom a no-op; that is the failure this exists to make impossible.

        >>> import PIL.Image
        >>> ck = build_create_kwargs(PIL.Image.new("RGB", (2000, 1500)))
        >>> ck["original_size"], ck["view_size"]
        ([2000, 1500], [1984, 1504])

    Returns a dict with `image`, `original_size`, `view_size`.
    """
    from PIL import Image

    if isinstance(raw_image, dict) and raw_image.get("bytes") is not None:
        img = Image.open(io.BytesIO(raw_image["bytes"]))
    elif isinstance(raw_image, bytes | bytearray):
        img = Image.open(io.BytesIO(bytes(raw_image)))
    elif isinstance(raw_image, str):
        img = Image.open(raw_image)
    elif isinstance(raw_image, Image.Image):
        img = raw_image
    else:
        raise TypeError(f"build_create_kwargs: unsupported raw_image of type {type(raw_image).__name__}")

    img.load()
    W, H = img.size
    factor = image_patch_size * merge_size
    view = smart_resize_wh(W, H, factor=factor, min_tokens=view_min_tokens, max_tokens=view_max_tokens)

    if as_data_uri:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        spec: Any = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    else:
        spec = img

    return {"image": spec, "original_size": [W, H], "view_size": [view[0], view[1]]}


def build_tools_kwargs(raw_image: Any, **kwargs: Any) -> dict:
    """`extra_info.tools_kwargs`, ready to go into a verl parquet row."""
    return {TOOL_NAME: {"create_kwargs": build_create_kwargs(raw_image, **kwargs)}}


if __name__ == "__main__":
    import json

    print(json.dumps({
        "max_assistant_turns": MAX_ASSISTANT_TURNS,
        "max_user_turns": MAX_USER_TURNS,
        "max_tool_calls": MAX_TOOL_CALLS,
        "data.image_patch_size": IMAGE_PATCH_SIZE,
        "px_per_visual_token": f"{PATCH_FACTOR}x{PATCH_FACTOR}",
    }, indent=2))
