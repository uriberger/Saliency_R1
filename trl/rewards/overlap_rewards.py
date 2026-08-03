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
each step with one of three metrics (--overlap_metric), then averages over the
completion's grounded observe steps (steps DINO can't ground are SKIPPED, not scored
0). The result is gated by format validity (multiplicative, like their valid_list).
Zero grounded observe steps -> None (masked, neutral in the GRPO advantage — NaN is
nan-summed out).

  mean_in  (DEFAULT, the incumbent)
      mean over the box of the MAX-normalized (/max -> [0,1]) map. Mirrors the offline
      reference in the vlm_reasoning repo (analysis/aggregation_correlation.py:
      _score_saliency_flat) so the online reward matches the fit it was validated on.

      KNOWN WEAKNESS: it divides by the map's own PEAK, not by its surroundings, so a
      map that merely FLATTENS scores higher inside the box without attending it any
      better. Measured offline, mean_in moves 32x further under pure flattening than
      under a genuine 5% transfer of attention mass into the box, and the wov0.2 /
      wov0.4 runs did exactly that: their MMStar gain disappears once chance-corrected.
      See wiki/lmms-eval-overlap-comparison.md in the vlm_reasoning repo.

  mean_in_v2  (--overlap_metric mean_in_v2)
      mean over the box divided by the mean over the WHOLE map, i.e. the same numerator
      as mean_in but normalized by the map's average instead of its peak. Chance = 1.0
      (a map with no preference for the box scores 1.0); the ceiling is
      n_patches / n_in_box, so the scale depends on how large the box union is.

      It closes mean_in's flattening hole from the other side than auroc does: both the
      numerator and the denominator are means, so any rescale m -> c*m cancels exactly,
      and a uniform flattening moves the score TOWARD 1.0 rather than up. Unlike auroc
      it still sees magnitudes, so a map that concentrates more mass in the box (not
      just ranks it higher) is rewarded for it. Unlike mean_in it is NOT bounded by 1,
      so its spread differs — retune --reward_weights rather than reusing the mean_in
      w_overlap. Not covered by the offline attack/utility screen that produced the
      mean_in and auroc numbers below; treat it as untested.

  auroc  (--overlap_metric auroc)
      P(a random in-box patch outranks a random out-box patch), average ranks for ties.
      Chance = 0.5. Depends only on the ORDER of the patches, so it is exactly
      invariant to m -> m**gamma: the flattening route is closed by construction, not
      by tuning. Scored 0.00 on every reshaping attack in the offline simulation, and
      predicts correctness more stably than mean_in (mean |r| 0.238 vs 0.181 over four
      powered datasets, sd 0.028 vs 0.089, and mean_in flips sign on Visual-CoT/DINO).

Optional mass floor (--mass_floor_tau, applies to any metric; off by default):

      score *= min(1, image_mass / tau)      image_mass = step_map.sum()

  Because attention rows are a softmax over ALL keys and only the image columns are
  kept, image_mass is the fraction of the row spent on the image. AUROC is rank-based
  and therefore blind to a model that withdraws attention from the image toward text
  tokens while keeping a good ranking; mean_in_v2 is a ratio of two means and is blind
  to it for the same reason (any rescale cancels). This floor closes that for both. It
  is not a pure guard:
  image_mass is itself predictive of correctness (r +0.22..+0.29), so the floor also
  RAISES the correlation (0.227 -> 0.238) rather than costing anything. Recommended
  tau = 0.0022, the 10th percentile of the reference model's image_mass (stable at
  0.0018-0.0029 across all seven offline collections). Keep tau near p10: much above
  p25 it stops being a floor and "raise image attention uniformly" becomes its own
  exploitable direction.

There is deliberately NO step-count term. The observe-step count carries essentially
no correctness signal (r -0.004..-0.022), so an anti-brevity multiplier costs 24% of
the reward's predictive value and a hard gate costs 50-70%, to close a step-dropping
hole that is already ~5x smaller under auroc (0.20) than under mean_in (1.07). Monitor
the observe-step count as a training diagnostic instead.

Natural-images-only gating (--overlap_natural_only, OFF by default):

      With a mixed corpus (cold_data/grpo_sets/set_b = 80% natural + 20% charts /
      documents / diagrams), Grounding-DINO is being asked to localise phrases on
      imagery it was never trained for, so on the non-natural rows the box union --
      and therefore the whole overlap score -- is noise. Turning this on returns
      None for every row whose `natural` column is False: those rows keep exactly
      the other three rewards (format, accuracy, judge) and contribute nothing to
      the overlap term. It is a masking, not a zeroing: a zero would be identical
      for the advantage (a per-group constant cancels in reward - group_mean) but
      would drag the logged rewards/think_overlap_reward/mean down with rows the
      reward was never evaluated on.

w_overlap is applied by the trainer via --reward_weights, not here.
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
    # "mean_in" (incumbent, default) | "mean_in_v2" (/mean instead of /max) | "auroc"
    "metric": "mean_in",
    "mass_floor_tau": None,  # None/0 disables the image-mass floor; recommended 0.0022
    "dino_api_base": None,   # if set, hit a served batched DINO endpoint; else local
    "dino_device": None,     # local device override; default cuda if available
    "dino_batch_size": 32,
    "natural_only": False,   # True -> mask (None) the reward on rows with natural=False
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


def _mean_in_v2(step_map, mask):
    """mean of the saliency inside the mask, divided by its mean over the whole map.

    Chance = 1.0; unbounded above (ceiling n_patches / n_in). Both terms are means of
    the SAME map, so the normalisation constant cancels -- no /max, no separate peak
    to inflate, and the value is invariant to m -> c*m (see the module docstring).
    """
    v = np.asarray(step_map, dtype=np.float64)
    inside = v[np.asarray(mask, dtype=bool)]
    if inside.size == 0:
        return None
    denom = float(v.mean())
    if denom <= 0:
        return None  # all-zero map: the ratio is undefined -> skip this step
    return float(inside.mean()) / denom


def _auroc(step_map, mask):
    """P(random in-box patch outranks a random out-box patch); 0.5 == chance.

    Average ranks for ties -- attention maps have many near-identical near-zero
    patches and argsort would break those ties arbitrarily, biasing the estimate.
    Pure numpy (no scipy) to avoid adding a dependency to the training env; this is
    the same computation as the offline screen, so the offline attack simulation and
    utility screen predict this reward exactly.
    """
    v = np.asarray(step_map, dtype=np.float64).ravel()
    m = np.asarray(mask, dtype=bool).ravel()
    n_in = int(m.sum())
    n_out = v.size - n_in
    if n_in == 0 or n_out == 0:
        return None
    order = np.argsort(v, kind="stable")
    ranks = np.empty(v.size, dtype=np.float64)
    ranks[order] = np.arange(1, v.size + 1, dtype=np.float64)
    _uniq, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
    sums = np.zeros(cnt.size, dtype=np.float64)
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    u = ranks[m].sum() - n_in * (n_in + 1) / 2.0
    return float(u / (n_in * n_out))


def _mass_gate(step_map):
    """min(1, image_mass / tau); 1.0 when the floor is disabled.

    image_mass = the map's total, i.e. the fraction of the softmax row spent on image
    tokens (the row is a softmax over all keys; only image columns were kept).
    """
    tau = _CFG.get("mass_floor_tau")
    if not tau or float(tau) <= 0:
        return 1.0
    return min(1.0, float(np.asarray(step_map).sum()) / float(tau))


def _step_score(step_map, mask):
    """Per-step reward: the configured metric, times the optional mass floor."""
    metric = _CFG.get("metric")
    if metric == "auroc":
        v = _auroc(step_map, mask)
    elif metric == "mean_in_v2":
        v = _mean_in_v2(step_map, mask)
    else:
        v = _mean_in(step_map, mask)
    if v is None:
        return None
    return v * _mass_gate(step_map)


def think_overlap_reward(
    completions=None, saliency_map=None, valid_list=None, image=None, natural=None, **kwargs
):
    """Per-completion overlap reward. See module docstring.

    Returns a list (len == n completions) of floats, or None where there is no grounded
    observe step, or where --overlap_natural_only masks a non-natural row (masked ->
    neutral in GRPO). w_overlap is applied by --reward_weights.
    """
    n = len(saliency_map)
    if valid_list is None:
        valid_list = [True] * n

    # --overlap_natural_only: score only the photographic rows. `natural` arrives as a
    # per-row dataset column (the trainer forwards every column as a reward kwarg).
    if _CFG.get("natural_only"):
        if natural is None:
            raise KeyError(
                "--overlap_natural_only requires a boolean 'natural' column in the dataset, "
                "but none reached the reward function. Use a corpus built by "
                "build_grpo_sets.py (cold_data/grpo_sets/*), or drop the flag."
            )
        scored = [bool(x) for x in natural]
    else:
        scored = [True] * n

    # Flatten every (completion, observe-step) into one batched DINO call. Masked rows
    # never reach DINO -- the trainer normally hands them no maps anyway, but a row
    # masked here must not cost a grounding call even if it does.
    flat_images, flat_texts, flat_owner = [], [], []
    for c, steps in enumerate(saliency_map):
        if not steps or not scored[c]:
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
        s = _step_score(step_map, mask)
        if s is not None:
            per_completion[c].append(s)

    rewards = []
    for c in range(n):
        if not scored[c]:
            rewards.append(None)  # non-natural under --overlap_natural_only -> mask
            continue
        vals = per_completion[c]
        if not vals:
            rewards.append(None)  # zero grounded observe steps -> mask (neutral)
            continue
        overlap = float(np.mean(vals))
        rewards.append(overlap * (1.0 if valid_list[c] else 0.0))  # format gate (multiplicative)
    return rewards
