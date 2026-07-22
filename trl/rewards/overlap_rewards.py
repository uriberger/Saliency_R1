# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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

"""Attention-overlap reward (reward_variant="ours"), the flag-selectable alternative
to Saliency-R1's think_saliency_reward.

The trainer (grpo_trainer_qwen3.py, reward_variant="ours" branch) does the attention
surgery and hands this reward, per completion, a list of per-observe-step saliency
maps + the step text:

    saliency_map[c] = [{"map": np.ndarray (grid_h, grid_w) float32, "text": str}, ...]

Each map is raw observe-token -> image-patch attention at LAYER 22, mean of the
configured heads (default (22,28)+(22,31)), ReLU, token-reduced over the step's tokens.
This reward grounds each step's text with Grounding-DINO (per step, in the loop),
builds the union mask of boxes >= box_threshold with area <= max_box_area, and scores

    mean_in = mean over the box of the MAX-normalized (/max -> [0,1]) map,

then averages mean_in over the completion's grounded observe steps (steps DINO can't
ground are SKIPPED, not scored 0). The result is gated by format validity
(multiplicative, like their valid_list). Zero grounded observe steps -> None (masked,
neutral in the GRPO advantage — NaN is nan-summed out).

This mirrors the offline reference metric in the vlm_reasoning repo
(analysis/aggregation_correlation.py:_score_saliency_flat, _filtered_boxes) so the
online reward matches the fit the design was validated on. w_overlap is applied by the
trainer via --reward_weights, not here.
"""

from __future__ import annotations

import contextlib
import os

import numpy as np

GROUNDING_DINO_HF_ID = "IDEA-Research/grounding-dino-base"


@contextlib.contextmanager
def _no_deepspeed_zero3_init():
    """Temporarily hide HF's global ZeRO-3 config from ``from_pretrained``.

    The trainer runs under DeepSpeed ZeRO-3 (accelerate ``zero3_init_flag: true``),
    which registers a process-global HfDeepSpeedConfig. Every subsequent
    ``from_pretrained`` — including this auxiliary Grounding-DINO model — would then
    be wrapped in ``deepspeed.zero.Init`` and have its parameters partitioned into
    1-D shards, so a sharded weight is no longer 2-D and the forward pass raises
    ``RuntimeError: 'weight' must be 2-D``. DINO is a small, frozen, single-device
    model that must be fully materialised, so we null the weakref for the duration
    of the load and restore it (no-op if transformers lacks deepspeed integration).
    Mirrors overlap_steps._no_deepspeed_zero3_init (kept local to avoid importing
    the trainer package from the rewards package).
    """
    try:
        import transformers.integrations.deepspeed as _ds
    except Exception:
        yield
        return
    saved = getattr(_ds, "_hf_deepspeed_config_weak_ref", None)
    _ds._hf_deepspeed_config_weak_ref = None
    try:
        yield
    finally:
        _ds._hf_deepspeed_config_weak_ref = saved

# Config, set by grpo_vlm_qwen3.py via configure() from the CLI flags. box_threshold /
# max_box_area default to the flagship offline filter (honest |r|~0.22 combo).
_CFG = {
    "box_threshold": 0.10,
    "max_box_area": 0.5,
    "dino_api_base": None,   # if set, hit a served batched DINO endpoint; else local
    "dino_device": None,     # local device override; default cuda if available
    "dino_batch_size": 32,
}

# Lazily-loaded local Grounding-DINO singleton (one per training process).
_DINO = {"proc": None, "model": None, "device": None}


def configure(**kwargs):
    """Set reward config from the CLI flags. None values are ignored (keep defaults)."""
    for k, v in kwargs.items():
        if v is not None:
            _CFG[k] = v


# ---------------------------------------------------------------------------
# Grounding-DINO (batched)
# ---------------------------------------------------------------------------

def _load_dino_local():
    if _DINO["model"] is None:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        device = _CFG.get("dino_device") or ("cuda" if torch.cuda.is_available() else "cpu")
        proc = AutoProcessor.from_pretrained(GROUNDING_DINO_HF_ID)
        # Load fully materialised: never let DeepSpeed ZeRO-3 partition this auxiliary
        # detector (would 1-D-shard its weights -> "'weight' must be 2-D" at forward).
        with _no_deepspeed_zero3_init():
            model = AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_DINO_HF_ID).to(device).eval()
        _DINO.update(proc=proc, model=model, device=device)
    return _DINO["proc"], _DINO["model"], _DINO["device"]


def _dino_boxes_local(images, texts):
    """Batched local Grounding-DINO. Returns list (per item) of [x1,y1,x2,y2] rel boxes.

    box_threshold is applied here; area filtering is applied by the caller.
    """
    import torch

    proc, model, device = _load_dino_local()
    prompts = [(t.strip() + (".") if not t.strip().endswith(".") else t.strip()) for t in texts]
    out_boxes = [None] * len(images)
    bs = int(_CFG["dino_batch_size"])
    for start in range(0, len(images), bs):
        imgs = images[start:start + bs]
        txts = prompts[start:start + bs]
        inputs = proc(
            images=imgs, text=txts, return_tensors="pt",
            padding=True, truncation=True, max_length=256,
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        target_sizes = [(im.size[1], im.size[0]) for im in imgs]  # (h, w)
        results = proc.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            threshold=float(_CFG["box_threshold"]),
            text_threshold=float(_CFG["box_threshold"]),
            target_sizes=target_sizes,
        )
        for j, res in enumerate(results):
            w, h = imgs[j].size
            boxes = []
            for box in res["boxes"].tolist():
                x1, y1, x2, y2 = box
                boxes.append([x1 / w, y1 / h, x2 / w, y2 / h])
            out_boxes[start + j] = boxes
    return out_boxes


def _dino_boxes_served(images, texts):
    """Batched served Grounding-DINO endpoint (preferred layout: DINO on a GPU outside
    the training allocation; see grpo-reward-port-plan memory). Posts base64 images +
    texts + thresholds, expects per-item relative-coord box lists back.

    Kept minimal on purpose; the local path is the tested one. Enable by setting
    --dino_api_base (OVERLAP_DINO_API_BASE). Falls back to local on any error so
    training never dies on a reward-server hiccup.
    """
    import base64
    import io

    import requests

    payload_images = []
    for im in images:
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        payload_images.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    resp = requests.post(
        _CFG["dino_api_base"].rstrip("/") + "/ground",
        json={
            "images": payload_images,
            "texts": list(texts),
            "box_threshold": float(_CFG["box_threshold"]),
            "text_threshold": float(_CFG["box_threshold"]),
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["boxes"]


def _dino_boxes(images, texts):
    if not images:
        return []
    if _CFG.get("dino_api_base"):
        try:
            return _dino_boxes_served(images, texts)
        except Exception as e:  # noqa: BLE001
            print(f"[overlap_reward] served DINO failed ({e}); falling back to local")
    return _dino_boxes_local(images, texts)


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------

def _box_area(b):
    x1, y1, x2, y2 = b
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _union_mask(boxes, grid_h, grid_w):
    """Boolean (grid_h, grid_w) union of area-filtered boxes; None if degenerate.

    Rasterisation matches analysis/aggregation_correlation.py exactly.
    """
    max_area = _CFG["max_box_area"]
    boxes = [b for b in boxes if max_area is None or _box_area(b) <= max_area]
    if not boxes:
        return None
    mask = np.zeros((grid_h, grid_w), dtype=bool)
    for x1, y1, x2, y2 in boxes:
        r0 = max(0, int(y1 * grid_h))
        r1 = min(grid_h, max(r0 + 1, round(y2 * grid_h)))
        c0 = max(0, int(x1 * grid_w))
        c1 = min(grid_w, max(c0 + 1, round(x2 * grid_w)))
        mask[r0:r1, c0:c1] = True
    n_in = int(mask.sum())
    if n_in == 0 or n_in == grid_h * grid_w:
        return None
    return mask


def _mean_in(step_map, mask):
    """mean of MAX-normalized (/max -> [0,1]) saliency inside the mask."""
    vmax = float(step_map.max())
    m = step_map / vmax if vmax > 0 else step_map
    inside = m[mask]
    return float(inside.mean()) if inside.size > 0 else None


def think_overlap_reward(completions=None, saliency_map=None, valid_list=None, image=None, **kwargs):
    """Per-completion overlap reward. See module docstring.

    Returns a list (len == n completions) of floats, or None where there is no grounded
    observe step (masked -> neutral in GRPO). w_overlap is applied by --reward_weights.
    """
    n = len(saliency_map)
    if valid_list is None:
        valid_list = [True] * n

    # Flatten every (completion, observe-step) into one batched DINO call.
    flat_images, flat_texts, flat_owner = [], [], []
    for c, steps in enumerate(saliency_map):
        if not steps:
            continue
        img = image[c]
        for si, st in enumerate(steps):
            flat_images.append(img)
            flat_texts.append(st["text"])
            flat_owner.append((c, si))

    boxes_per_item = _dino_boxes(flat_images, flat_texts) if flat_images else []

    # Gather grounded mean_in per completion.
    per_completion = [[] for _ in range(n)]
    for (c, si), boxes in zip(flat_owner, boxes_per_item):
        step_map = saliency_map[c][si]["map"]
        gh, gw = step_map.shape
        mask = _union_mask(boxes, gh, gw)
        if mask is None:
            continue  # DINO couldn't ground this step -> skip (do NOT score 0)
        mi = _mean_in(step_map, mask)
        if mi is not None:
            per_completion[c].append(mi)

    rewards = []
    for c in range(n):
        vals = per_completion[c]
        if not vals:
            rewards.append(None)  # zero grounded observe steps -> mask (neutral)
            continue
        overlap = float(np.mean(vals))
        rewards.append(overlap * (1.0 if valid_list[c] else 0.0))  # format gate (multiplicative)
    return rewards
