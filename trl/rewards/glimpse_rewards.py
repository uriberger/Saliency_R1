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

"""GLIMPSE grounding reward (reward_variant="glimpse"), in two metric variants.

The trainer hands this the same per-completion structure the attention-overlap and
gradient rewards get -- a list of `{"map": (gh, gw) float32, "text": str}` per observe
step -- except the map is `trl/glimpse_maps.py`'s GLIMPSE map (docs/saliency-maps.md
map 6), gradient-weighted attention propagated with adaptive layer weights. Each step's
text is grounded with Grounding-DINO into a box union `U` over the patch grid and scored
with the SAME metric implementations the incumbent overlap reward uses -- this module
calls `overlap_rewards._step_score` rather than reimplementing either:

    mean_in_v2   mean of the map inside U / mean over the whole map.  chance 1.0
    auroc        P(a random in-U patch outranks a random out-of-U one). chance 0.5

The completion's reward is the mean of the per-step score over its grounded observe
steps, times the format gate; no grounded step -> None (masked, neutral in the GRPO
advantage), exactly as in overlap_rewards and grad_rewards. `w_glimpse` is applied by the
trainer via --reward_weights, not here.

READ THIS BEFORE TRAINING ON IT. A 3,471-step / 1,157-completion screen of this exact map
(`outputs/flow_corr/glimpse_screen/glimpse/report.txt`) found:

  * GLIMPSE is the first map in this repo that is GROUNDED at all. auroc level 0.567
    against a chance of 0.5, and 0.712 on the smallest union decile -- every earlier map
    sat at or below chance.
  * Its correlation with the model being RIGHT is null to slightly negative at every
    level: `mean_in_v2` r = -0.031 (step) / -0.020 (completion); `auroc` r = -0.056 /
    -0.064. Nothing clears Bonferroni (|r| >= 0.0735 for the 4 tests reported).
  * The level decays hard with union area: 0.712 at mean union 0.11 down to 0.471 at
    0.89, r(union) = -0.487.

So the honest prior is that rewarding either variant moves the policy by an amount
indistinguishable from zero, and if it moves it at all the sign is wrong. Two consequences
are built into this module rather than left to the caller:

  * `union_frac` and `ceiling` are ALWAYS logged. `mean_in_v2`'s ceiling is
    `n_patches / n_in`, so union area moves it mechanically and a reward that can grow the
    union gets a free ride -- the level rising is not evidence of grounding. Watch
    `glimpse/score_raw` against `glimpse/union_frac`, not `score_raw` alone.
  * `--max_union_area` is the knob that decides which regime the run is in, and given
    r(union) = -0.487 it is the first thing to set, not an afterthought. It is shared with
    the overlap reward and configured through `overlap_rewards.configure`.

What this does NOT close, and what to watch (logged as `glimpse/*`):

  the union-area free ride  `union_frac` rising with `score_raw`. The single most likely
                            hack direction for `mean_in_v2`; `auroc` is rank-based and
                            immune to the ceiling but not to the level decay above.
  the positional prior      `ecc`, the union centroid's distance from the grid centre. If
                            the map has radial structure a centred box wins for free.
  a hollow reward           `n_image`, the map's total mass. A ratio is blind to the image
                            mattering less overall.
  duplicate / pruned steps  the score is a mean over GROUNDED steps, so re-quoting one
                            easily-grounded sentence pulls it up and dropping hard steps
                            raises it. `dup_frac`, `n_steps`, `grounded_frac`.
  a blanked aggregation     `unweighted_frac`. eq 18's beta can be zero for every token of
                            a step, in which case the map falls back to a plain mean and
                            the step is no longer the quantity that was screened.
  the bill                  `n_target_tokens`. Cost is linear in it, and it is the only
                            per-step quantity that says what a step actually cost.
"""

from __future__ import annotations

import numpy as np

from .grad_rewards import _centroid_eccentricity, _norm_text
from .overlap_rewards import _dino_boxes, _step_score, _union_mask

# Config, set by grpo_vlm_qwen3.py via configure() from the CLI flags. The METRIC and the
# DINO-side knobs (box_threshold, max_box_area, max_union_area, ...) live in
# overlap_rewards._CFG and are configured there -- this module calls its grounding and
# scoring helpers unchanged, which is what keeps "mean_in_v2 here" and "mean_in_v2 there"
# the same number.
_CFG = {
    "dedupe_steps": True,
    "natural_only": False,
}

# Drained by the trainer once per step and appended to self._metrics as glimpse/<key>.
#
# The key list is FIXED and pop_diagnostics always returns all of it. The trainer gathers
# these across ranks, and a gather is a collective: if the key set depended on what a rank
# happened to see -- a rank with no grounded step has no union_frac to report -- the ranks
# would issue different numbers of collectives and the run would hang in NCCL, not fail.
# Missing keys come back NaN and are dropped after the gather instead.
DIAG_KEYS = (
    "score_raw", "union_frac", "ceiling", "ecc", "n_image",
    "dup_frac", "n_steps", "grounded_frac", "unweighted_frac", "n_target_tokens",
)
_DIAG: dict[str, list[float]] = {}


def configure(**kwargs):
    """Set reward config from the CLI flags. None values are ignored (keep defaults)."""
    for k, v in kwargs.items():
        if v is not None:
            _CFG[k] = v


def _diag(key: str, value: float):
    _DIAG.setdefault(key, []).append(float(value))


def record_map_info(info: dict):
    """Fold one case's map-producer `info` into the diagnostics.

    `unweighted_steps` and `n_target_tokens` are properties of building the map, not of
    scoring it, so the trainer reports them here rather than the reward inferring them.
    """
    n = int(info.get("n_steps_built") or 0)
    fell = len(info.get("unweighted_steps") or ())
    if n:
        _diag("unweighted_frac", fell / n)
    if info.get("n_target_tokens") is not None:
        _diag("n_target_tokens", float(info["n_target_tokens"]))


def pop_diagnostics() -> dict[str, float]:
    """Mean of each diagnostic since the last call, then clear. Always all DIAG_KEYS.

    NaN for a key nothing was recorded under -- see the note on DIAG_KEYS for why the
    shape must not depend on what this rank saw.
    """
    out = {k: (float(np.mean(_DIAG[k])) if _DIAG.get(k) else float("nan")) for k in DIAG_KEYS}
    _DIAG.clear()
    return out


def _dedupe(steps: list[dict]) -> tuple[list[dict], float]:
    """-> (steps with repeated texts dropped, duplicate fraction).

    The mean over steps is what the wov0.4 run hacked: re-quoting one trivially-groundable
    sentence pulls the mean up and dilutes the hard perception steps (duplicate-sentence
    fraction 0.00 -> 0.19 over steps 1000-2000). Deduping removes the payoff; the fraction
    is logged either way so the behaviour stays visible if this is turned off.

    `_norm_text` is shared with grad_rewards -- it is the subtle half (the grounding call
    appends its own period, so "a cat." and "a cat" are one step) -- while the enable flag
    is read from this module's own _CFG, so --glimpse_dedupe_steps means what it says.
    """
    if not steps:
        return steps, 0.0
    seen, kept = set(), []
    for st in steps:
        key = _norm_text(st["text"])
        if key in seen:
            continue
        seen.add(key)
        kept.append(st)
    dup_frac = 1.0 - len(kept) / len(steps)
    return (kept if _CFG["dedupe_steps"] else steps), dup_frac


def think_glimpse_reward(
    completions=None, saliency_map=None, valid_list=None, image=None, natural=None, **kwargs
):
    """Per-completion GLIMPSE grounding reward. See module docstring.

    Returns a list (len == n completions) of floats, or None where there is no grounded
    observe step, or where --glimpse_natural_only masks a non-natural row.
    """
    n = len(saliency_map)
    if valid_list is None:
        valid_list = [True] * n

    if _CFG.get("natural_only"):
        if natural is None:
            raise KeyError(
                "--glimpse_natural_only requires a boolean 'natural' column in the "
                "dataset, but none reached the reward function. Use a corpus built by "
                "build_grpo_sets.py (cold_data/grpo_sets/*), or drop the flag."
            )
        scored = [bool(x) for x in natural]
    else:
        scored = [True] * n

    # One batched DINO call over every (completion, step) that will actually be scored.
    flat_images, flat_texts, flat_owner = [], [], []
    per_completion_steps = [[] for _ in range(n)]
    for c, steps in enumerate(saliency_map):
        if not steps or not scored[c]:
            continue
        kept, dup_frac = _dedupe(list(steps))
        _diag("dup_frac", dup_frac)
        _diag("n_steps", len(steps))
        per_completion_steps[c] = kept
        for si, st in enumerate(kept):
            flat_images.append(image[c])
            flat_texts.append(st["text"])
            flat_owner.append((c, si))

    boxes_per_item = _dino_boxes(flat_images, flat_texts) if flat_images else []

    per_completion = [[] for _ in range(n)]
    n_grounded = 0
    for (c, si), boxes in zip(flat_owner, boxes_per_item):
        step_map = per_completion_steps[c][si]["map"]
        gh, gw = step_map.shape
        mask = _union_mask(boxes, gh, gw)
        if mask is None:
            continue  # DINO couldn't ground this step -> skip (do NOT score 0)
        s = _step_score(step_map, mask)
        if s is None:
            continue
        n_grounded += 1
        per_completion[c].append(s)
        n_in = max(int(mask.sum()), 1)
        # union_frac and ceiling are not optional colour: mean_in_v2's ceiling IS
        # n_patches/n_in, so a score that rose because the union grew is indistinguishable
        # from one that rose because the map improved unless both are on the dashboard.
        _diag("score_raw", s)
        _diag("union_frac", mask.mean())
        _diag("ceiling", mask.size / n_in)
        _diag("ecc", _centroid_eccentricity(mask))
        _diag("n_image", float(np.asarray(step_map, dtype=np.float64).sum()))
    if flat_owner:
        _diag("grounded_frac", n_grounded / len(flat_owner))

    rewards = []
    for c in range(n):
        if not scored[c]:
            rewards.append(None)
            continue
        vals = per_completion[c]
        if not vals:
            rewards.append(None)  # zero scorable observe steps -> mask (neutral)
            continue
        rewards.append(float(np.mean(vals)) * (1.0 if valid_list[c] else 0.0))
    return rewards
