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
builds the union mask of boxes >= box_threshold with area <= max_box_area (and, with
--max_union_area, drops steps whose union covers too much of the image -- see the
coverage section below), and scores
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

      CONFIRMED ONLINE: the wov0.4 / trmean / 50k set_a run took this route at ~step
      1200. It doubled the overlap reward (0.044 -> 0.085) by repeating observe steps,
      and its cp_2000 scores BELOW the cold-start parent on the benchmarks (p3 60.7 ->
      34.3, i.e. chance; mme 715 -> 642) while the training accuracy reward barely moved
      -- the hack costs 0.006 accuracy on set_a and lives inside <think>, which
      accuracy_reward never parses. See wiki/overlap-reward-hack-set-a.md (same repo)
      for the full mechanism, and note the enabling condition is not the metric alone:
      with scale_rewards=True, groups whose accuracy has saturated hand the overlap term
      the entire, std-renormalized advantage (75% of groups by step 1500).

  mean_in_v2  (--overlap_metric mean_in_v2)
      mean over the box divided by the mean over the WHOLE map, i.e. the same numerator
      as mean_in but normalized by the map's average instead of its peak. Chance = 1.0
      (a map with no preference for the box scores 1.0); the ceiling is
      n_patches / n_in_box, so the scale depends on how large the box union is.

      It closes mean_in's flattening hole from the other side than auroc does: both the
      numerator and the denominator are means, so any rescale m -> c*m cancels exactly,
      and a uniform flattening moves the score TOWARD 1.0 rather than up. Unlike auroc
      it still sees magnitudes, so a map that concentrates more mass in the box (not
      just ranks it higher) is rewarded for it.

      MEASURED (overlap_metric_spread.py over 1074 grounded steps of the cold-start
      policy on set_a, 40 samples x 8 generations):

        range      p10 0.41 / median 0.74 / p99 1.36 / max 2.33. Unbounded in principle,
                   but the median box union covers 56% of the image, which puts the
                   ceiling n/k at ~1.8 -- no clamp is needed in practice.
        spread     per-sample sd 0.105, i.e. 12x mean_in's 0.0086, so w_overlap 0.033
                   reproduces mean_in's wov0.4 pressure (the launchers apply this).
                   The same script re-derives auroc's documented 0.11 as 0.089, so
                   treat 0.033 as +-25%.
        box size   r +0.38 with the box area fraction, vs +0.17 for mean_in and -0.11
                   for auroc. Growing the box union does raise the score while the map
                   is BELOW chance, but that pull dies at 1.0 instead of diverging: full
                   coverage gives exactly 1.0. (Dividing by the outside SUM rather than
                   the overall mean removes that limit -- it equals
                   (mean_in/mean_out)/(n-k), which diverges as the union grows and has
                   no fixed chance level. That is why the denominator is the whole map.)

      Not covered by the offline attack/utility screen that produced the mean_in and
      auroc correlation numbers below.

  auroc  (--overlap_metric auroc)
      P(a random in-box patch outranks a random out-box patch), average ranks for ties.
      Chance = 0.5. Depends only on the ORDER of the patches, so it is exactly
      invariant to m -> m**gamma: the flattening route is closed by construction, not
      by tuning. Scored 0.00 on every reshaping attack in the offline simulation, and
      predicts correctness more stably than mean_in (mean |r| 0.238 vs 0.181 over four
      powered datasets, sd 0.028 vs 0.089, and mean_in flips sign on Visual-CoT/DINO).

      NOT immune to the union-growth hack, despite all of the above. The offline screen
      covers RESHAPING the map for a FIXED box; it says nothing about the policy changing
      the TEXT so that DINO returns a bigger box. The wov0.11 / auroc / 50k set_a run did
      exactly that: the overlap reward jumped around step 2200 and every observe step had
      become a description of the BACKGROUND, which grounds to huge boxes. Rank-invariance
      does not help when the ranking is taken over a different, much larger in-box set.
      So do not read auroc's r -0.11 with the box-area fraction (cited below) as evidence
      that it needs no union cap -- that number is from static offline collections and it
      did not predict this. See --max_union_area.

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

Box-coverage caps (two independent filters, both in _union_mask):

  --box_threshold (default 0.10)
      DINO confidence floor. Applied server-side too when --dino_api_base is set.

  --max_box_area (default 0.5, set to 0 to DISABLE)
      Drops any INDIVIDUAL box whose area exceeds this fraction of the image, before
      rasterisation. Note this is a per-box filter and says nothing about the union:
      ten disjoint boxes at 0.1 each pass it and cover the whole image between them.

  --max_union_area (default None = OFF)
      Skips the whole step when the RASTERISED union covers more than this fraction
      of the patch grid. Returns None from _union_mask, so the step takes the same
      path as an ungroundable one: SKIPPED, not scored 0, exactly like the existing
      degenerate-union guard (which only fires at 100% coverage).

      Why it is worth having: the per-box cap leaves the union unbounded, and the
      measured median union already covers 56% of the image (overlap_metric_spread.py,
      1074 grounded steps of the cold-start policy on set_a). A near-full union makes
      the score meaningless -- everything is "inside the box".

      CONFIRMED ONLINE, under AUROC. The wov0.11 / auroc / 50k set_a run jumped in
      overlap reward around step 2200, and its observe steps had all turned into
      descriptions of the BACKGROUND -- background phrases ground to huge boxes, and a
      huge scored region is easier to rank well against. This is the metric-INDEPENDENT
      hole: the offline area-fraction correlations (mean_in +0.17, mean_in_v2 +0.38,
      auroc -0.11) are about reshaping the map for a fixed box and did not predict it.
      auroc's negative number in particular is not protection.

      Why it is still OFF by default (2026-08-04): the right value is not yet known.
      The cold-start policy's median union is ALREADY 0.562, so any cap tight enough to
      look principled masks a large share of completions before training does anything
      -- and a step the cap drops leaves the mean entirely, so an aggressive cap turns
      the reward off for whole completions rather than merely trimming it. Size it from
      the measured distribution first: union_size_report.py over an overlap_probe run
      (which must itself be run with --max_union_area 0, or it cannot see the tail it
      is being used to measure).

      Turning it on also changes WHICH steps are scored, so it shifts the reward's
      scale -- re-check w_overlap against a probe run rather than assuming the
      incumbent weight transfers, and note the launchers add _mu<x> to the run name so
      a capped run never shares a checkpoint dir or wandb name with an uncapped one.

There is deliberately NO step-count term. The observe-step count carries essentially
no correctness signal (r -0.004..-0.022), so an anti-brevity multiplier costs 24% of
the reward's predictive value and a hard gate costs 50-70%, to close a step-dropping
hole that is already ~5x smaller under auroc (0.20) than under mean_in (1.07). Monitor
the observe-step count as a training diagnostic instead.

The hole that actually opened online was the opposite one: DUPLICATED steps. The score
is a mean over grounded steps, so re-quoting one trivially-groundable generic sentence
pulls it up and dilutes the genuine, hard perception steps -- which is what the wov0.4 /
set_a run learned (duplicate-sentence fraction 0.00 -> 0.19 over steps 1000-2000, mean
completion length 163 -> 356 tokens). Log the duplicate fraction alongside the step
count, and consider deduping identical steps before the mean.
See wiki/overlap-reward-hack-set-a.md.

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

from . import roll_null as _RN

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
    "max_box_area": 0.5,     # per-box area cap; None or <= 0 disables it
    "max_union_area": None,  # per-step union coverage cap; None or <= 0 disables it
    # "mean_in" (incumbent, default) | "mean_in_v2" (/mean instead of /max) | "auroc"
    # | "logratio" (the roll-null, chance 0 -- see roll_null.py and the knobs below)
    "metric": "mean_in",
    # Roll-null knobs. Read ONLY when metric == "logratio"; the gradient reward keeps its
    # own copies under --grad_* because there the roll-null is the reward, not a choice.
    "null_offsets": 16,      # K translates of the union forming the null
    "logratio_clip": 1.0,    # +-c on log(N(U)/N_0); 1.0 == a ratio of e ~ 2.7
    "inframe_rolls": True,   # keep the translate inside the grid (no border wrap)
    "min_inframe": 4,        # below this many in-frame offsets, fall back to toroidal
    "roll_seed": 0,
    "mass_floor_tau": None,  # None/0 disables the image-mass floor; recommended 0.0022
    "dino_api_base": None,   # if set, hit a served batched DINO endpoint; else local
    "dino_device": None,     # local device override; default cuda if available
    "dino_batch_size": 32,
    "natural_only": False,   # True -> mask (None) the reward on rows with natural=False
}

# Lazily-loaded local Grounding-DINO singleton (one per training process).
_DINO = {"proc": None, "model": None, "device": None}


# Roll-null by-products, logged only when metric == "logratio". FIXED key set and always
# all of it, for the same NCCL reason grad_rewards.DIAG_KEYS documents: the trainer
# gathers these across ranks, and a key set that depended on what a rank happened to see
# would mean a rank-dependent number of collectives, which hangs rather than fails.
#
# `toroidal_frac` is the one to watch. It says the in-frame control pool was too small --
# a near-full-frame union -- so the null wrapped across the image border and stopped being
# the same question. It rising means the scores are no longer comparable to earlier ones.
ROLL_DIAG_KEYS = ("logratio_raw", "clip_frac", "toroidal_frac", "n_offsets",
                  "union_frac", "ecc", "n_image")
_DIAG: dict[str, list[float]] = {}
_ROLL_RNG = np.random.default_rng(0)


def _diag(key: str, value: float):
    _DIAG.setdefault(key, []).append(float(value))


def pop_diagnostics() -> dict[str, float]:
    """Mean of each roll-null diagnostic since the last call, then clear.

    Always all of ROLL_DIAG_KEYS; NaN for a key nothing was recorded under, including
    every key when the configured metric is not "logratio".
    """
    out = {k: (float(np.mean(_DIAG[k])) if _DIAG.get(k) else float("nan"))
           for k in ROLL_DIAG_KEYS}
    _DIAG.clear()
    return out


def configure(**kwargs):
    """Set reward config from the CLI flags. None values are ignored (keep defaults)."""
    global _ROLL_RNG
    for k, v in kwargs.items():
        if v is not None:
            _CFG[k] = v
    _ROLL_RNG = np.random.default_rng(int(_CFG["roll_seed"]))


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

    def _run_chunk(start, n):
        imgs = images[start:start + n]
        txts = prompts[start:start + n]
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

    # Deformable attention materialises one contiguous
    # (batch, queries, heads, levels, points) tensor, so peak memory scales with the
    # batch AND with the images' native resolution: a batch that fits on one caller
    # can OOM on the next. Callers that co-reside with a big model on the same GPU
    # (the probe: 8B VLM + attention re-forward) cannot pick a batch size that is
    # both safe and fast, so halve on OOM instead of dying. A single item that still
    # OOMs is a real failure and is re-raised.
    bs = int(_CFG["dino_batch_size"])
    start = 0
    while start < len(images):
        n = min(bs, len(images) - start)
        while True:
            try:
                _run_chunk(start, n)
                break
            except torch.cuda.OutOfMemoryError:
                if n == 1:
                    raise
                torch.cuda.empty_cache()
                n = max(1, n // 2)
                print(f"[dino] CUDA OOM; retrying batch at offset {start} with size {n}",
                      flush=True)
        start += n
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


def _union_mask(boxes, grid_h, grid_w, apply_union_cap=True):
    """Boolean (grid_h, grid_w) union of area-filtered boxes; None if degenerate.

    Rasterisation matches analysis/aggregation_correlation.py exactly.

    Two independent filters, both disabled by None or a non-positive value:

      max_box_area    per-box, applied to the raw relative-coordinate box BEFORE
                      rasterisation. Drops individual boxes; the surviving ones
                      still form a union.
      max_union_area  per-step, applied to the RASTERISED union. Rejects the whole
                      step (-> None -> skipped, not scored 0) when the union covers
                      more than this fraction of the patch grid. The per-box cap
                      does not bound the union: N disjoint boxes each under the cap
                      can cover the image between them (measured median union
                      coverage on set_a is 56%), and under `mean_in` a growing union
                      raises the score (r +0.17 with the area fraction). This is the
                      only filter that closes that.

    The union is measured on the patch grid, not on the box geometry, because the
    grid mask is what the metric actually scores -- and rasterisation inflates it:
    every surviving box claims at least one patch row and column, so many small
    boxes cover more grid than their summed area suggests.

    apply_union_cap=False skips only the max_union_area check, so a caller can see the
    mask the cap rejected (overlap_probe uses this to distinguish "the cap dropped this
    step" from "DINO grounded nothing"). The reward path always leaves it True.
    """
    max_area = _CFG.get("max_box_area")
    if max_area is not None and float(max_area) > 0:
        boxes = [b for b in boxes if _box_area(b) <= max_area]
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
    max_union = _CFG.get("max_union_area")
    if apply_union_cap and max_union is not None and float(max_union) > 0:
        if n_in > float(max_union) * grid_h * grid_w:
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


def _roll_logratio(step_map, mask):
    """The roll-null score with THIS module's knobs, logging THIS module's diagnostics.

    `roll_null.py` holds the definition and the reasoning; it is shared with the gradient
    reward, where the same score is the only scoring mode rather than one metric of four.
    Chance is exactly 0, so unlike the other three this metric is already centred.

    Random by construction: it draws control placements. A caller that needs the value
    twice must keep it, not call twice, or the step gets two different scores.
    """
    r, info = _RN.logratio(step_map, mask, _ROLL_RNG,
                           n_offsets=int(_CFG["null_offsets"]),
                           clip=float(_CFG["logratio_clip"]),
                           inframe=bool(_CFG["inframe_rolls"]),
                           min_inframe=int(_CFG["min_inframe"]))
    if r is None:
        return None
    for key in ROLL_DIAG_KEYS:
        _diag(key, info[key])
    return r


def _step_score(step_map, mask):
    """Per-step reward: the configured metric, times the optional mass floor."""
    metric = _CFG.get("metric")
    if metric == "auroc":
        v = _auroc(step_map, mask)
    elif metric == "mean_in_v2":
        v = _mean_in_v2(step_map, mask)
    elif metric == "logratio":
        v = _roll_logratio(step_map, mask)
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
