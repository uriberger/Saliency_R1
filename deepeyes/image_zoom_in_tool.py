# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ImageZoomInTool, adapted from verl's own to DeepEyes-on-Qwen3-VL.

TRACKED SOURCE. The copy that executes is
`deepeyes_fork/verl/tools/image_zoom_in_tool.py`; install this one over it with

    bash patch_deepeyes_zoom_tool.sh            # --check reports, changes nothing

and never hand-edit the fork. `deepeyes_fork/` is shared by every DeepEyes agent.

---------------------------------------------------------------------------
WHAT CHANGED FROM verl's VERSION, AND WHY EACH ONE IS SILENT WHEN WRONG
---------------------------------------------------------------------------

[1] THE COORDINATE FRAME IS NOW EXPLICIT, AND HAS NO DEFAULT.

    verl's `execute` read `bbox_2d` and cropped with it as absolute pixels. That is
    right for Qwen2.5-VL, whose grounding output is absolute pixels in the
    POST-`smart_resize` frame. Qwen3-VL's cookbook documents a 0-1000 relative grid
    against the ORIGINAL image. Those two readings differ by roughly a factor of two
    in each linear dimension on a typical V* photo, so getting it wrong crops the
    wrong region on every single call -- and nothing complains, because a crop of the
    wrong region is still a valid image and the loss curve stays healthy.

    So the frame is a config key with THREE readings and NO silent default:

        bbox_frame: rel1000         # x * W/1000,  y * H/1000     (Qwen3-VL cookbook)
        bbox_frame: abs_original    # x, y                        (DeepEyes' assumption)
        bbox_frame: abs_smartresize # x * W/rW,    y * H/rH       (Qwen2.5-VL's frame)

    Unset or unrecognised raises at construction. P4
    (`deepeyes_coord_format_probe.py`) measures which one Qwen3-VL actually emits
    under DeepEyes' prompt; when it reports, the answer is a one-line config change
    here and nothing else. P4's shorter spellings `abs_orig` / `abs_resized` are
    accepted as aliases so its verdict can be pasted in verbatim.

    Every call converts raw -> original-image pixels at the tool boundary, and the
    first `log_first_n_calls` calls per process log BOTH boxes, so the frame is a fact
    in the run log rather than an assumption in a config file.

[2] MIN_DIMENSION IS 32, NOT 28.

    Qwen2.5-VL is patch-14 / merge-2: one visual token covers 28x28 px, and their
    28 px floor is a one-visual-token floor. Qwen3-VL is patch-16 / merge-2: one token
    covers 32x32 px, so 28 is SUB-token here and the floor stops meaning what it
    meant. It is restated as `patch_size * merge_size` and both are config.

    On "restate any N*28*28 budget as N*32*32": there is no such budget in this file
    upstream -- the only 28 was the floor -- and DeepEyes never caps the crop either.
    The knobs exist anyway (`max_crop_tokens` / `min_crop_tokens`), denominated in
    TOKENS rather than pixels precisely so the 28-vs-32 question cannot come back.
    Both default to None, which is DeepEyes' behaviour: do not resize the crop, let
    the processor do it.

[3] THE CROP SOURCE IS THE ORIGINAL IMAGE, AND THAT IS ASSERTED.

    This is the whole method. DeepEyes' own tool
    (`verl/workers/agent/envs/mm_process_engine/visual_toolbox_v2.py:reset`) crops
    `origin_multi_modal_data`, which their dataloader builds with `process_raw_image`
    -- a bare `Image.open(...).convert("RGB")` with NO `smart_resize` -- held
    alongside the `process_image` copy the model is shown. A crop taken from the
    already-downsampled copy recovers NO detail, produces a perfectly plausible
    image, and turns the entire arm into expensive plain GRPO.

    verl's version called `qwen_vl_utils.fetch_image` inside `create`, which
    smart-resizes. This one never resizes on load, and refuses an image spec that
    carries resize instructions (`max_pixels`, `resized_height`, ...).

    Because "is this image the original?" is not answerable from the pixels, the
    data builder must say so, and the tool checks the statement:

        create_kwargs = {
            "image":         <original bytes / path / PIL, un-resized>,
            "original_size": [W, H],    # read from the RAW bytes at build time
            "view_size":     [vW, vH],  # optional: what the dataloader showed the model
        }

    `original_size` must equal the loaded size or the run dies. Use
    `deepeyes/zoom_tool_config.py::build_create_kwargs` and it is right by
    construction. `crop_source_check: warn|off` downgrades this, and exists for
    smoke data only.

---------------------------------------------------------------------------
FIDELITY: max_turns IS 5, MEANING 4 TOOL CALLS
---------------------------------------------------------------------------
Not set here -- it belongs to the rollout config -- but decided here and recorded in
`deepeyes/zoom_tool_config.py` so the launcher does not re-litigate it. The paper says
"up to a maximum of 6 times of active perceptions"; their released script
`examples/agent/final_merged_v1v8_thinklite.sh` sets `agent.max_turns=5`, and their
loop (`verl/workers/agent/parallel_env.py:181,226`) counts GENERATE rounds and skips
the tool on the last one. We follow the code, not the sentence: 5 generate rounds,
at most 4 tool calls.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import threading
from contextlib import ExitStack
from enum import Enum
from io import BytesIO
from math import ceil, floor, sqrt
from typing import Any, Callable, Optional, TypeVar
from uuid import uuid4

import ray
import ray.actor
from PIL import Image
from qwen_vl_utils.vision_process import smart_resize, to_rgb

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

T = TypeVar("T")

#: Prefix on every stdout line this file emits, so a smoke log can be grepped the same
#: way P2's `[DEEPEYES-MROPE]` lines are.
TAG = "[DEEPEYES-ZOOM]"

#: The three readings of `bbox_2d`, each a map from the emitted numbers to ORIGINAL
#: image pixels -- the frame the crop is taken in.
#:
#:   rel1000         (x/1000)*W, (y/1000)*H     Qwen3-VL's documented grounding grid
#:   abs_original    x, y                       DeepEyes' tool's assumption
#:   abs_smartresize (x/rW)*W,   (y/rH)*H       post-smart_resize pixels, Qwen2.5-VL's
#:
#: There is deliberately no fourth entry for "whatever looks sensible". See [1] above.
FRAMES = ("rel1000", "abs_original", "abs_smartresize")

#: P4 (`deepeyes_coord_format_probe.py`) names the same three readings slightly
#: differently. Accept its spellings so its verdict can be pasted into the config
#: without a translation step -- a translation step is where a frame gets flipped.
FRAME_ALIASES = {
    "abs_orig": "abs_original",
    "abs_resized": "abs_smartresize",
    "abs_resize": "abs_smartresize",
    "rel_1000": "rel1000",
    "relative1000": "rel1000",
}

#: Qwen3-VL geometry. patch 16 * merge 2 = one visual token per 32x32 px.
#: Qwen2.5-VL was patch 14 * merge 2 = 28. Changing these is how this file would be
#: pointed at another backbone; nothing else in it hardcodes a number of pixels.
DEFAULT_PATCH_SIZE = 16
DEFAULT_MERGE_SIZE = 2

#: qwen_vl_utils' own token bounds, restated here so the derived "view" frame matches
#: what `verl/utils/dataset/vision_utils.py::process_image` produced. They are token
#: counts, not pixels: the pixel bounds are these times (patch*merge)**2.
DEFAULT_VIEW_MIN_TOKENS = 4
DEFAULT_VIEW_MAX_TOKENS = 16384


# Adapted from verl/tools/sandbox_fusion_tools.py
class PoolMode(Enum):
    """Execution pool mode enumeration."""

    ThreadMode = 1
    ProcessMode = 2


@ray.remote(concurrency_groups={"acquire": 1, "release": 10})
class TokenBucketWorker:
    """Ray actor for rate limiting using token bucket algorithm."""

    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        self.current_count = 0  # For observability
        self._semaphore = threading.Semaphore(rate_limit)

    @ray.method(concurrency_group="acquire")
    def acquire(self):
        """Acquire a token from the bucket."""
        self._semaphore.acquire()
        self.current_count += 1

    @ray.method(concurrency_group="release")
    def release(self):
        """Release a token back to the bucket."""
        self._semaphore.release()
        self.current_count -= 1

    def get_current_count(self):
        """Get current number of acquired tokens."""
        return self.current_count


class VisualExecutionWorker:
    """Worker for executing visual processing operations with optional rate limiting."""

    def __init__(self, enable_global_rate_limit=True, rate_limit=10):
        self.rate_limit_worker = self._init_rate_limit(rate_limit) if enable_global_rate_limit else None

    def _init_rate_limit(self, rate_limit):
        """Initialize singleton rate limiter."""
        return TokenBucketWorker.options(name="rate-limiter", get_if_exists=True).remote(rate_limit)

    def ping(self):
        """Health check method."""
        return True

    def execute(self, fn: Callable[..., T], *fn_args, **fn_kwargs) -> T:
        """Execute function with optional rate limiting."""
        if self.rate_limit_worker:
            with ExitStack() as stack:
                stack.callback(self.rate_limit_worker.release.remote)
                ray.get(self.rate_limit_worker.acquire.remote())
                try:
                    return fn(*fn_args, **fn_kwargs)
                except Exception as e:
                    # TODO we should make this available to the tool caller
                    logger.warning(f"Error when executing visual processing: {e}")
        else:
            return fn(*fn_args, **fn_kwargs)


def init_visual_execution_pool(
    num_workers: int, enable_global_rate_limit=True, rate_limit=10, mode: PoolMode = PoolMode.ThreadMode
):
    """Initialize visual execution pool."""
    if mode == PoolMode.ThreadMode:
        return (
            ray.remote(VisualExecutionWorker)
            .options(max_concurrency=num_workers)
            .remote(enable_global_rate_limit=enable_global_rate_limit, rate_limit=rate_limit)
        )
    else:
        raise NotImplementedError("Process mode is not implemented yet")


class ImageZoomInTool(BaseTool):
    """Crop a region out of the ORIGINAL image and hand it back as an observation.

    Config keys (all under the tool's `config:` block; see the module docstring):

        bbox_frame        REQUIRED. rel1000 | abs_original | abs_smartresize.
        patch_size        16 on Qwen3-VL (was 14 on Qwen2.5-VL).
        merge_size        2.
        min_dimension     defaults to patch_size*merge_size = 32.
        max_crop_tokens   None. A crop budget, in visual TOKENS (32x32 px each).
        min_crop_tokens   None.
        crop_source_check strict | warn | off. strict requires create_kwargs to
                          declare `original_size`.
        min_expected_downscale
                          None. If set, a corpus whose images are not at least this
                          much bigger than what the model was shown fails the run --
                          that corpus cannot be recovering any detail.
        view_patch_size   the dataloader's `data.image_patch_size` (DEFAULTS TO 14 IN
                          verl AND MUST BE SET TO 16 FOR Qwen3-VL). Used only to
                          derive the `abs_smartresize` frame and the "view" size when
                          the data does not state it.
        view_merge_size / view_min_tokens / view_max_tokens
                          the rest of the dataloader's smart_resize parameters.
        log_first_n_calls 20. Raw-vs-converted boxes, per process.
        log_dir           where to append zoom_calls.<pid>.jsonl. Falls back to
                          $DEEPEYES_ZOOM_LOG_DIR.
        log_crops         0. Also dump that many crops (plus the source with the box
                          drawn) as PNGs, for P9's eyeball check.
        num_workers / rate_limit / enable_global_rate_limit
                          unchanged; see `execution_pool` below.
    """

    #: Restated from Qwen2.5-VL's 28. One visual token on Qwen3-VL is 32x32 px, so a
    #: 28 px floor would accept a crop smaller than a single token. Instances read
    #: `self.min_dimension`; this class attribute is the documented default only.
    MIN_DIMENSION = DEFAULT_PATCH_SIZE * DEFAULT_MERGE_SIZE  # 32

    #: Process-wide, because verl constructs one tool object per agent-loop worker and
    #: runs every trajectory through it. Guarded by a lock: `_call_tool` fans out over
    #: asyncio tasks and the loop runs tool calls concurrently.
    _log_lock = threading.Lock()
    _log_count = 0
    _crop_count = 0

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        _tool_schema = OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "image_zoom_in_tool",
                "description": (
                    "Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) and an "
                    "optional object label."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "bbox_2d": {
                            "type": "array",
                            "items":{"type":"number"},
                            "minItems":4,
                            "maxItems":4,
                            "description": (
                                "The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is "
                                "the top-left corner and (x2, y2) is the bottom-right corner."
                            ),
                        },
                        "label": {
                            "type": "string",
                            "description": "The name or label of the object in the specified bounding box (optional).",
                        },
                    },
                    "required": ["bbox_2d"],
                },
            }
        })
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        # ---- [1] the coordinate frame: explicit, or nothing -----------------------
        # `config.get("bbox_frame")` with a default would be the single most expensive
        # line in this file. A wrong frame is invisible: the crops are valid images,
        # the tokens are well formed, the loss falls. The only thing that moves is
        # where the model is looking, and nothing in the run reports that.
        self.bbox_frame = self._resolve_frame(config.get("bbox_frame", None))

        # ---- [2] geometry ---------------------------------------------------------
        self.patch_size = int(config.get("patch_size", DEFAULT_PATCH_SIZE))
        self.merge_size = int(config.get("merge_size", DEFAULT_MERGE_SIZE))
        self.patch_factor = self.patch_size * self.merge_size  # 32 on Qwen3-VL
        self.min_dimension = int(config.get("min_dimension", self.patch_factor))
        if self.min_dimension < self.patch_factor:
            raise ValueError(
                f"min_dimension={self.min_dimension} is below one visual token "
                f"({self.patch_factor}x{self.patch_factor} px on patch-{self.patch_size} / "
                f"merge-{self.merge_size}). That is the Qwen2.5-VL 28 showing through; the "
                f"floor exists to keep a crop worth at least one token."
            )
        self.max_crop_tokens = config.get("max_crop_tokens", None)
        self.min_crop_tokens = config.get("min_crop_tokens", None)

        # The dataloader's smart_resize parameters, needed to say what the model SAW.
        # verl's `data.image_patch_size` defaults to 14 and nothing derives it from the
        # processor -- see verl/utils/dataset/rl_dataset.py:108 -- so on Qwen3-VL the
        # launcher must set it to 16 and this must agree with it.
        self.view_patch_size = int(config.get("view_patch_size", self.patch_size))
        self.view_merge_size = int(config.get("view_merge_size", self.merge_size))
        self.view_factor = self.view_patch_size * self.view_merge_size
        self.view_min_tokens = int(config.get("view_min_tokens", DEFAULT_VIEW_MIN_TOKENS))
        self.view_max_tokens = int(config.get("view_max_tokens", DEFAULT_VIEW_MAX_TOKENS))

        # ---- [3] crop source ------------------------------------------------------
        self.crop_source_check = str(config.get("crop_source_check", "strict")).lower()
        if self.crop_source_check not in ("strict", "warn", "off"):
            raise ValueError(
                f"crop_source_check={self.crop_source_check!r}; expected one of strict|warn|off"
            )
        self.min_expected_downscale = config.get("min_expected_downscale", None)

        # ---- logging --------------------------------------------------------------
        self.log_first_n_calls = int(config.get("log_first_n_calls", 20))
        log_dir = config.get("log_dir", None) or os.environ.get("DEEPEYES_ZOOM_LOG_DIR", "")
        self.log_dir = str(log_dir) if log_dir else ""
        self.log_crops = int(config.get("log_crops", 0))
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)

        # Worker and rate limiting configuration
        self.num_workers = config.get("num_workers", 20)
        self.rate_limit = config.get("rate_limit", 50)
        self.timeout = config.get("timeout", 30)
        self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)
        self._execution_pool = None

        banner = (
            f"{TAG} ImageZoomInTool: bbox_frame={self.bbox_frame} "
            f"min_dimension={self.min_dimension} (patch {self.patch_size} x merge {self.merge_size} "
            f"= {self.patch_factor}px/token) view_factor={self.view_factor} "
            f"crop_source_check={self.crop_source_check} log_first_n_calls={self.log_first_n_calls}"
        )
        print(banner, flush=True)
        logger.info(banner)
        logger.info(f"Initialized ImageZoomInTool with config: {config}")

    # ------------------------------------------------------------------ pool
    @property
    def execution_pool(self):
        """Created on first use, which in this version is never.

        `execute` does not route through the pool upstream either -- the pool is built
        and dropped. Keeping it lazy costs nothing, and buys the one thing that
        matters here: `ImageZoomInTool(...)` can be constructed on a box with no ray
        cluster, which is what makes `test_deepeyes_zoom_tool_cpu.py` a unit test
        rather than a job submission.
        """
        if self._execution_pool is None:
            self._execution_pool = init_visual_execution_pool(
                num_workers=self.num_workers,
                enable_global_rate_limit=self.enable_global_rate_limit,
                rate_limit=self.rate_limit,
                mode=PoolMode.ThreadMode,
            )
        return self._execution_pool

    # ------------------------------------------------------- [1] the frame
    @staticmethod
    def _resolve_frame(raw: Any) -> str:
        """Map a config value to one of FRAMES, or raise. Never guesses."""
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            raise ValueError(
                "ImageZoomInTool: `bbox_frame` is not set, and it has no default.\n"
                "  Qwen2.5-VL emits absolute pixels in the post-smart_resize frame;\n"
                "  Qwen3-VL's cookbook documents a 0-1000 relative grid against the\n"
                "  original image. Those differ by ~2x per axis on a V* photo, and a\n"
                "  wrong frame crops the wrong region on EVERY call with no symptom.\n"
                f"  Set one of {list(FRAMES)} in the tool config.\n"
                "  P4 (deepeyes_coord_format_probe.py) measures which one applies."
            )
        key = str(raw).strip().lower()
        key = FRAME_ALIASES.get(key, key)
        if key not in FRAMES:
            raise ValueError(
                f"ImageZoomInTool: bbox_frame={raw!r} is not a frame this tool knows. "
                f"Expected one of {list(FRAMES)} (aliases: {sorted(FRAME_ALIASES)})."
            )
        return key

    def convert_bbox(
        self,
        bbox_2d,
        original_size: tuple[int, int],
        view_size: tuple[int, int],
    ) -> list[float]:
        """Map the emitted box into ORIGINAL-image pixels, in the configured frame.

        No clamping and no validation here on purpose: `_maybe_resize_bbox` owns those,
        and the log wants the converted box BEFORE it is clamped so that "the model
        aimed off the edge" and "the frame is wrong" stay distinguishable.
        """
        x1, y1, x2, y2 = (float(v) for v in bbox_2d)
        W, H = original_size

        if self.bbox_frame == "abs_original":
            return [x1, y1, x2, y2]
        if self.bbox_frame == "rel1000":
            return [x1 * W / 1000.0, y1 * H / 1000.0, x2 * W / 1000.0, y2 * H / 1000.0]
        if self.bbox_frame == "abs_smartresize":
            vW, vH = view_size
            if vW <= 0 or vH <= 0:
                raise ValueError(f"view_size={view_size} is degenerate; cannot map abs_smartresize")
            return [x1 * W / vW, y1 * H / vH, x2 * W / vW, y2 * H / vH]
        raise AssertionError(f"unreachable frame {self.bbox_frame!r}")  # pragma: no cover

    # --------------------------------------------- [3] the crop source
    def view_size_for(self, width: int, height: int) -> tuple[int, int]:
        """What `verl/utils/dataset/vision_utils.py::process_image` would have shown
        the model for an original of this size: `smart_resize` at the DATALOADER's
        patch factor and token bounds.

        Returned as (width, height) -- `smart_resize` returns (height, width), and
        that transposition is a classic way to get a square-ish image "working" and a
        portrait one silently wrong.
        """
        h_bar, w_bar = smart_resize(
            height,
            width,
            factor=self.view_factor,
            min_pixels=self.view_min_tokens * self.view_factor**2,
            max_pixels=self.view_max_tokens * self.view_factor**2,
        )
        return int(w_bar), int(h_bar)

    def _load_original_image(self, spec: Any) -> Image.Image:
        """Decode an image WITHOUT resizing it. The whole of change [3] is here.

        verl's version called `qwen_vl_utils.fetch_image`, which ends in
        `image.resize(...)`. DeepEyes' own dataloader keeps a second, un-resized copy
        (`process_raw_image`) for exactly this reason, and crops that.

        Accepts what `fetch_image` accepts minus the resize controls: a PIL image, a
        `{"bytes": ...}` dict (what parquet round-trips), a local path, `file://`,
        `data:image/...;base64,...`, or an http(s) URL.
        """
        if isinstance(spec, dict):
            # `{"max_pixels": ...}` and friends are instructions to downsample. They
            # are legitimate for the PROMPT image and fatal for the crop source, so
            # refuse rather than silently honour or silently ignore them.
            forbidden = sorted(
                k for k in ("max_pixels", "min_pixels", "resized_height", "resized_width") if k in spec
            )
            if forbidden:
                raise ValueError(
                    f"create_kwargs['image'] carries resize instructions {forbidden}. The zoom "
                    "tool must crop the ORIGINAL image; a crop of a downsampled image recovers "
                    "nothing and the arm becomes plain GRPO with a format penalty."
                )
            if "bytes" in spec and spec["bytes"] is not None:
                img = Image.open(BytesIO(spec["bytes"]))
            elif spec.get("image") is not None:
                return self._load_original_image(spec["image"])
            elif spec.get("path"):
                img = Image.open(spec["path"])
            else:
                raise ValueError(f"unrecognised image spec dict with keys {sorted(spec)}")
        elif isinstance(spec, Image.Image):
            img = spec
        elif isinstance(spec, bytes | bytearray):
            img = Image.open(BytesIO(bytes(spec)))
        elif isinstance(spec, str):
            if spec.startswith("data:image"):
                if "base64," not in spec:
                    raise ValueError("data: URI is not base64-encoded")
                img = Image.open(BytesIO(base64.b64decode(spec.split("base64,", 1)[1])))
            elif spec.startswith("file://"):
                img = Image.open(spec[7:])
            elif spec.startswith("http://") or spec.startswith("https://"):
                import requests

                with requests.get(spec, stream=True, timeout=self.timeout) as r:
                    r.raise_for_status()
                    img = Image.open(BytesIO(r.content))
            else:
                img = Image.open(spec)
        else:
            raise ValueError(f"unrecognised image spec of type {type(spec).__name__}")

        img.load()
        # `to_rgb`, not `.convert("RGB")`: it is what qwen_vl_utils uses for the prompt
        # image, and it white-backgrounds RGBA instead of compositing onto black. The
        # crop and the prompt image must not disagree about what a transparent pixel is.
        return to_rgb(img)

    def _check_crop_source(self, img: Image.Image, create_kwargs: dict) -> dict:
        """Assert -- at runtime, fatally -- that what we are about to crop is the
        original, pre-processor-resize image.

        It is not answerable from the pixels: an image that has already been
        smart-resized looks exactly like an original that happened to need no resize.
        So the data builder states the original's size, read from the RAW bytes at
        build time, and this compares. `deepeyes/zoom_tool_config.py::build_create_kwargs`
        produces the block correctly; a hand-rolled conversion that reuses the
        `process_image` output for both fields is the mistake this catches.
        """
        W, H = img.size
        declared = create_kwargs.get("original_size", None)
        stated_view = create_kwargs.get("view_size", None)
        derived_view = self.view_size_for(W, H)
        info: dict[str, Any] = {
            "original_size": [W, H],
            "view_size": list(derived_view),
            "view_size_source": "derived",
        }

        def fail(msg: str):
            if self.crop_source_check == "strict":
                raise ValueError(msg)
            if self.crop_source_check == "warn":
                print(f"{TAG} WARNING crop-source: {msg}", flush=True)
                logger.warning(msg)

        if declared is None:
            fail(
                "create_kwargs has no `original_size`, so this tool cannot verify that it is "
                "cropping the ORIGINAL image rather than the one the model was already shown. "
                "That difference is the entire method: a crop of a downsampled image recovers no "
                "detail, looks plausible, and keeps the loss curve healthy. Build the block with "
                "deepeyes/zoom_tool_config.py::build_create_kwargs, or set crop_source_check: warn "
                "if this is smoke data you do not care about."
            )
        else:
            dw, dh = (int(v) for v in declared)
            info["declared_original_size"] = [dw, dh]
            if (dw, dh) != (W, H):
                fail(
                    f"create_kwargs declares original_size={[dw, dh]} but the image handed to the "
                    f"tool is {[W, H]}. Something between the parquet and here resized it -- most "
                    f"likely the `process_image`/`fetch_image` copy was passed instead of the raw "
                    f"bytes. The crop would be taken from a downsampled image."
                )

        if stated_view is not None:
            vw, vh = (int(v) for v in stated_view)
            info["view_size"] = [vw, vh]
            info["view_size_source"] = "stated"
            info["derived_view_size"] = list(derived_view)
            if (vw, vh) != derived_view:
                fail(
                    f"the data says the model was shown {[vw, vh]}, but this tool's smart_resize "
                    f"parameters (factor {self.view_factor}, tokens "
                    f"{self.view_min_tokens}..{self.view_max_tokens}) give {list(derived_view)}. "
                    f"They must agree or the abs_smartresize frame is computed against the wrong "
                    f"view. verl's data.image_patch_size DEFAULTS TO 14; on Qwen3-VL it must be 16."
                )

        vw, vh = info["view_size"]
        downscale = sqrt((W * H) / float(max(vw * vh, 1)))
        info["downscale"] = round(downscale, 4)
        if self.min_expected_downscale is not None and downscale < float(self.min_expected_downscale):
            fail(
                f"the original is only {downscale:.3f}x the linear size of what the model was "
                f"shown ({[W, H]} vs {[vw, vh]}), below min_expected_downscale="
                f"{self.min_expected_downscale}. On this corpus a zoom recovers no detail the "
                f"model did not already have, so the tool is decoration."
            )
        return info

    # --------------------------------------------- [2] the floor, restated
    def _validate_bbox(self, left: float, top: float, right: float, bottom: float) -> bool:
        """Validate the bounding box dimensions and aspect ratio."""
        try:
            if not (left < right and top < bottom):
                logger.warning(f"Invalid bbox shape: left={left}, top={top}, right={right}, bottom={bottom}")
                return False

            height = bottom - top
            width = right - left

            # Prevent division by zero for zero-sized boxes
            if min(height, width) == 0:
                logger.warning(f"Bbox has zero width or height: left={left}, top={top}, right={right}, bottom={bottom}")
                return False

            if max(height, width) / min(height, width) > 100:
                logger.warning(f"Bbox aspect ratio > 100: left={left}, top={top}, right={right}, bottom={bottom}")
                return False

            return True
        except Exception as e:
            logger.warning(f"Bbox validation error: {e}")
            return False

    def _maybe_resize_bbox(self, bbox_2d: list[float], image_width: int, image_height: int) -> Optional[list[float]]:
        """
        Clamp, validate, and potentially resize a bounding box.

        This function ensures the final bounding box is within image bounds and meets the minimum
        dimension requirements. If the initial box is too small, it attempts to expand it
        from its center. It performs a final check to guarantee the output dimensions are valid.

        The box is expected to ALREADY be in original-image pixels: `execute` converts
        at the tool boundary, before this is called.

        Returns:
            A valid bounding box as a list of coordinates, or None if validation fails.
        """
        left, top, right, bottom = bbox_2d

        # 1. Clamp the initial bounding box to the image dimensions.
        left = max(0.0, float(left))
        top = max(0.0, float(top))
        right = min(float(image_width), float(right))
        bottom = min(float(image_height), float(bottom))

        # 2. If clamped bbox is invalid, return immediately.
        if not self._validate_bbox(left, top, right, bottom):
            return None

        current_bbox = [left, top, right, bottom]
        height = bottom - top
        width = right - left

        # 3. If the box is too small, attempt to resize it.
        if height < self.min_dimension or width < self.min_dimension:
            logger.info(f"Bbox {width}x{height} is smaller than {self.min_dimension}, attempting resize.")
            center_x = (left + right) / 2.0
            center_y = (top + bottom) / 2.0

            min_dim = min(height, width)
            if min_dim == 0:  # Safeguard for zero-area boxes
                return None

            # 1. Calculate the target dimensions to make the smallest side min_dimension.
            ratio = self.min_dimension / min_dim
            target_width = width * ratio
            target_height = height * ratio

            # 2. If the target size is larger than the image, scale it down to fit.
            #    This preserves the aspect ratio while respecting image boundaries.
            if target_width > image_width:
                scale_down = image_width / target_width
                target_width = image_width
                target_height *= scale_down

            if target_height > image_height:
                scale_down = image_height / target_height
                target_height = image_height
                target_width *= scale_down

            # 3. Determine the coordinates for the box centered on the original center.
            new_half_width = target_width / 2.0
            new_half_height = target_height / 2.0
            new_left = center_x - new_half_width
            new_top = center_y - new_half_height

            # 4. Shift the box if it extends beyond the image boundaries to keep its size.
            if new_left < 0:
                new_left = 0
            if new_top < 0:
                new_top = 0
            if new_left + target_width > image_width:
                new_left = image_width - target_width
            if new_top + target_height > image_height:
                new_top = image_height - target_height

            new_right = new_left + target_width
            new_bottom = new_top + target_height

            # Use floor and ceil for final integer coordinates.
            current_bbox = [floor(new_left), floor(new_top), ceil(new_right), ceil(new_bottom)]

        # 4. Final validation on the resulting bounding box (either original or resized).
        final_left, final_top, final_right, final_bottom = current_bbox
        if not self._validate_bbox(final_left, final_top, final_right, final_bottom):
            logger.warning(f"Final bbox is invalid after processing: {current_bbox}")
            return None

        final_height = floor(final_bottom) - floor(final_top)
        final_width = floor(final_right) - floor(final_left)

        if final_height < self.min_dimension or final_width < self.min_dimension:
            logger.warning(
                f"Final bbox size ({final_width}x{final_height}) are still smaller than minimum "
                f"({self.min_dimension}). An image smaller than one visual token "
                f"({self.patch_factor}x{self.patch_factor}) cannot be shown to the model. "
                f"Original bbox: {bbox_2d}, original image size: {image_width}x{image_height}"
            )
            return None

        return current_bbox

    def _maybe_budget_crop(self, crop: Image.Image) -> Image.Image:
        """Optional crop budget, denominated in VISUAL TOKENS.

        DeepEyes imposes none, and neither did verl's version, so both bounds default
        to None and this is a no-op. It exists so that if a budget is ever wanted it is
        written as a token count and converted here at `patch_factor**2` px per token
        -- 32*32 on Qwen3-VL, where the Qwen2.5-VL number would have been 28*28 and
        would have meant ~23% fewer pixels for the same stated token count.
        """
        if self.max_crop_tokens is None and self.min_crop_tokens is None:
            return crop
        w, h = crop.size
        max_px = int(self.max_crop_tokens) * self.patch_factor**2 if self.max_crop_tokens else None
        min_px = int(self.min_crop_tokens) * self.patch_factor**2 if self.min_crop_tokens else None
        h_bar, w_bar = smart_resize(h, w, factor=self.patch_factor, min_pixels=min_px, max_pixels=max_px)
        if (w_bar, h_bar) == (w, h):
            return crop
        return crop.resize((int(w_bar), int(h_bar)))

    # ------------------------------------------------------------- logging
    def _log_call(self, record: dict, src: Optional[Image.Image] = None, crop: Optional[Image.Image] = None) -> None:
        """Raw box, converted box, and the frame that did it -- for the first N calls.

        The point is P9 step 3: "eyeball ten of them against the image". A frame that
        is assumed is a frame nobody checks; a frame that prints both numbers next to
        the image size is one somebody can falsify in ten seconds.
        """
        with self._log_lock:
            n = ImageZoomInTool._log_count
            if n >= self.log_first_n_calls:
                return
            ImageZoomInTool._log_count = n + 1
            dump_crop = src is not None and crop is not None and ImageZoomInTool._crop_count < self.log_crops
            if dump_crop:
                k = ImageZoomInTool._crop_count
                ImageZoomInTool._crop_count = k + 1

        record = {"call": n, "pid": os.getpid(), "frame": self.bbox_frame, **record}
        print(
            f"{TAG} call={n} frame={self.bbox_frame} raw={record.get('bbox_raw')} "
            f"-> converted={record.get('bbox_converted')} -> cropped={record.get('bbox_final')} "
            f"original={record.get('original_size')} view={record.get('view_size')} "
            f"crop={record.get('crop_size')} status={record.get('status')}",
            flush=True,
        )
        if not self.log_dir:
            return
        try:
            with open(os.path.join(self.log_dir, f"zoom_calls.{os.getpid()}.jsonl"), "a") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
            if dump_crop:
                from PIL import ImageDraw

                crop.save(os.path.join(self.log_dir, f"crop_{n:04d}.png"))
                marked = src.copy()
                box = record.get("bbox_final")
                if box:
                    ImageDraw.Draw(marked).rectangle(
                        [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                        outline=(255, 0, 0),
                        width=max(3, min(src.size) // 200),
                    )
                marked.thumbnail((1024, 1024))
                marked.save(os.path.join(self.log_dir, f"source_{n:04d}.png"))
        except Exception as e:  # logging must never take a rollout down
            logger.warning(f"zoom call logging failed: {e}")

    # ------------------------------------------------------------- BaseTool
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """
        Creates a new instance for image zoom-in tool.

        This method initializes a new session for an image, which can then be used
        for operations like zooming. It loads the image WITHOUT resizing it and
        records what the model was shown, so `execute` can convert coordinates and
        assert its crop source.

        Args:
            instance_id: An optional unique identifier for the instance. If not
                provided, a new UUID will be generated.
            **kwargs: Should contain 'image' key with image data, or 'create_kwargs'
                containing {'image': ..., 'original_size': [W, H], 'view_size': [W, H]}.
                Image can be one of the following:
                - A PIL.Image.Image object.
                - A {"bytes": ...} dict, which is what parquet round-trips.
                - A string containing an HTTP or HTTPS URL.
                - A string containing a local file path.
                - A string containing a file URI (e.g., "file:///path/to/image.jpg").
                - A string containing a base64-encoded image in the format of "data:image/jpeg;base64,..."

        Returns:
            Tuple of (instance_id, ToolResponse)
        """
        if instance_id is None:
            instance_id = str(uuid4())

        # Handle create_kwargs parameter if passed
        create_kwargs = dict(kwargs.get("create_kwargs", {}) or {})
        if create_kwargs:
            kwargs.update(create_kwargs)

        # Get image from kwargs
        image = kwargs.get("image")
        if image is None:
            raise ValueError("Missing required 'image' parameter in kwargs")

        img = self._load_original_image(image)
        source = self._check_crop_source(img, create_kwargs if create_kwargs else kwargs)

        self._instance_dict[instance_id] = {
            "image": img,
            "source": source,
            "response": "",
            "reward": 0.0,
        }
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        bbox_raw = parameters.get("bbox_2d")
        label = parameters.get("label", "")

        if not bbox_raw or len(bbox_raw) != 4:
            return (
                ToolResponse(text="Error: bbox_2d parameter is missing or not a list of 4 numbers."),
                -0.05,
                {"success": False},
            )

        instance_data = self._instance_dict[instance_id]
        image = instance_data["image"]
        source = instance_data["source"]
        image_width, image_height = image.size
        view = tuple(source["view_size"])

        base = {
            "bbox_raw": list(bbox_raw),
            "original_size": [image_width, image_height],
            "view_size": list(view),
            "downscale": source.get("downscale"),
            "label": label,
        }

        try:
            bbox_converted = self.convert_bbox(bbox_raw, (image_width, image_height), view)
        except Exception as e:
            self._log_call({**base, "status": f"convert-failed: {e}"})
            logger.warning(f"Could not convert bbox {bbox_raw} under frame {self.bbox_frame}: {e}")
            return (
                ToolResponse(text=f"Error: could not interpret bounding box {list(bbox_raw)}."),
                -0.05,
                {"success": False},
            )
        base["bbox_converted"] = [round(v, 2) for v in bbox_converted]

        try:
            resized_bbox = self._maybe_resize_bbox(
                bbox_converted, image_width=image_width, image_height=image_height
            )

            if resized_bbox is None:
                self._log_call({**base, "bbox_final": None, "status": "invalid"})
                error_msg = (
                    f"Error: The specified bounding box {list(bbox_raw)} is invalid or results in a crop "
                    f"smaller than the minimum size of {self.min_dimension}x{self.min_dimension}."
                )
                logger.warning(f"Tool execution failed: {error_msg}")
                return ToolResponse(text=error_msg), -0.05, {"success": False}

            cropped_image = self._maybe_budget_crop(image.crop(tuple(resized_bbox)))
            logger.info(f"Cropped image size: {cropped_image.size}")
        except Exception as e:
            self._log_call({**base, "status": f"crop-failed: {e}"})
            logger.error(f"Error processing image zoom-in: {e}")
            return ToolResponse(text=f"Error processing image zoom-in: {e}"), -0.05, {"success": False}

        self._log_call(
            {
                **base,
                "bbox_final": [float(v) for v in resized_bbox],
                "crop_size": list(cropped_image.size),
                "crop_tokens": (cropped_image.size[0] // self.patch_factor)
                * (cropped_image.size[1] // self.patch_factor),
                "status": "ok",
            },
            src=image,
            crop=cropped_image,
        )

        # The text echoes the box the model EMITTED, not the converted one: the model
        # asked in its own frame and an answer in a different frame is a second,
        # untrained coordinate convention appearing mid-conversation.
        response_text = f"Zoomed in on the image to the region {list(bbox_raw)}."
        if label:
            response_text = f"Zoomed in on the image to the region {list(bbox_raw)} with label {label}."

        return (
            ToolResponse(
                image=[cropped_image],
                text=response_text,
            ),
            0.0,
            {
                "success": True,
                "frame": self.bbox_frame,
                "bbox_raw": list(bbox_raw),
                "bbox_final": [float(v) for v in resized_bbox],
                "crop_size": list(cropped_image.size),
            },
        )

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
