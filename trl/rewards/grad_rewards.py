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

"""Roll-null gradient reward (reward_variant="grad").

The trainer hands this the same per-completion structure the attention-overlap reward
gets -- a list of `{"map": (gh, gw) float32, "text": str}` per observe step -- except the
map is `trl/grad_maps.py`'s pixel-gradient map `G`, not attention. Each step's text is
grounded with Grounding-DINO into a box union `U` over the patch grid, and scored:

    N(A)^2 = sum_{j in A} G_j^2                     gradient energy over a token set
    N_0^2  = (1/K) sum_k N(U'_k)^2                  K translates of U, same shape and area
    r_S    = clip( log N(U) - log N_0 , -c, +c )

`U'_k` is `U` translated to a random offset. The completion's reward is the mean of `r_S`
over its grounded observe steps, times the format gate; no grounded step -> None (masked,
neutral in the GRPO advantage), exactly as in overlap_rewards.

Why this shape and not the obvious one, in the order the alternatives were rejected:

  ||g_U|| alone.  The literal reading of "how much do the boxed pixels move this step's
  tokens", and it has three scales in it that the policy controls and that are not
  grounding: how confident the model is (the centered logit in grad_maps removes this
  one), how large the union is (||g_U|| ~ sqrt(|U|) for ANY map, so "name things that
  ground to bigger boxes" raises it mechanically), and the image's overall gradient
  level. Only the last of the three cancels within a GRPO group, and only because all
  generations in a group share one image.

  ||g_U|| - ||g_control||.  Fixes the chance level, but subtraction removes an ADDITIVE
  offset while the confounds here are MULTIPLICATIVE: scale the whole map by c and the
  difference scales by c too. Raw norms also make it mostly a box-size meter -- for a
  flat map the difference is sqrt(|U|) - sqrt(n - |U|), monotone in |U|.

  ||g_U|| / ||g_out||.  Cancels the multiplicative factor, but the denominator's support
  shrinks as the union grows: unbounded, noisy in the tail, undefined at full coverage --
  and union growth is the known hack direction.

  ||g_U|| / ||g_U'||, THIS.  The control has the same shape and area as `U` by
  construction, so it never degenerates, and it is self-limiting in exactly the direction
  that needs limiting: under a uniform translation the expected shared area is
  E|U n U'| / |U| = |U|/n, so a union covering most of the image is compared against
  itself and the ratio collapses toward 1. Size and confidence are closed; what remains
  open is listed at the bottom of this docstring.

Three details that are not cosmetic:

  * The null pools the SQUARED norms over K offsets before the log. Averaging log-ratios
    instead lets one control landing on a dead region (sky, uniform background) dominate
    the null and blow up that step's score.
  * Scoring in log space makes chance exactly 0, makes the per-completion aggregate a
    geometric mean of ratios, and makes the clip symmetric. This matters more than usual
    under `scale_rewards=True`: a single outlier completion otherwise takes most of its
    group's normalised advantage.
  * Offsets are drawn IN-FRAME (the translate stays inside the grid) rather than
    toroidally, so a control never wraps across the image border. With too few in-frame
    positions -- a near-full-frame union -- it falls back to toroidal and says so in
    `grad/toroidal_frac`.

What this does NOT close, and what to watch instead (`pop_diagnostics`, logged as
`grad/*`):

  the positional prior     If `G` has radial structure, a centred box beats its
                           translates for free and "describe central things" is a hack
                           the reward cannot see. `grad/ecc` and its correlation with the
                           score is the tell; `saliency_viz.py --stage null` measures the
                           prior itself.
  sensitivity != use       ||dz/dx|| is large where the function is locally jagged, which
                           a LoRA can manufacture without relying on the region. Only an
                           occlusion probe separates them.
  duplicate / pruned steps The score is a mean over GROUNDED steps, so re-quoting one
                           easily-grounded sentence pulls it up and dropping hard steps
                           raises it. Identical step texts are deduped here (see
                           `dedupe_steps`); `grad/dup_frac`, `grad/n_steps` and
                           `grad/grounded_frac` are the monitors.
  a hollow reward          A ratio is blind to the image mattering less overall.
                           `grad/n_image` is that magnitude; if it collapses while the
                           reward rises, the reward has gone hollow.

w_grad is applied by the trainer via --reward_weights, not here, and it does NOT transfer
from the attention reward's 0.033/0.11: set it from the measured spread of `r_S`
(overlap_metric_spread.py) so that w * sd matches the incumbent's pressure.
"""

from __future__ import annotations

import re

import numpy as np

from . import overlap_rewards as _OR
from . import roll_null as _RN
from .overlap_rewards import _dino_boxes, _union_mask

# Config, set by grpo_vlm_qwen3.py via configure() from the CLI flags. The DINO-side
# knobs (box_threshold, max_box_area, dino_api_base, ...) live in overlap_rewards._CFG
# and are configured there -- this module calls its grounding helpers unchanged.
_CFG = {
    # Which metric scores a grounded step. Defaults to this reward's historical mode --
    # `--overlap_metric` is one flag across three maps whose defaults differ, so an unset
    # metric must still give THIS map the roll-null it has always used.
    "metric": "logratio",
    "null_offsets": 16,      # K translates per step
    "logratio_clip": 1.0,    # +-c on log(N(U)/N_0); 1.0 == a ratio of e ~ 2.7
    "inframe_rolls": True,   # keep the translate inside the grid (no border wrap)
    "min_inframe": 4,        # below this many in-frame offsets, fall back to toroidal
    "dedupe_steps": True,    # drop repeated step texts before the mean over steps
    "natural_only": False,
    "seed": 0,
}

_RNG = np.random.default_rng(0)

# Drained by the trainer once per step and appended to self._metrics as grad/<key>.
#
# The key list is FIXED and pop_diagnostics always returns all of it. The trainer gathers
# these across ranks, and a gather is a collective: if the key set depended on what a rank
# happened to see -- a rank with no grounded step has no union_frac to report -- the ranks
# would issue different numbers of collectives and the run would hang in NCCL, not fail.
# Missing keys come back NaN and are dropped after the gather instead.
DIAG_KEYS = (
    "union_frac", "ecc", "n_image", "logratio_raw", "clip_frac", "toroidal_frac",
    "n_offsets", "dup_frac", "n_steps", "grounded_frac",
)
_DIAG: dict[str, list[float]] = {}


def configure(**kwargs):
    """Set reward config from the CLI flags. None values are ignored (keep defaults)."""
    global _RNG
    for k, v in kwargs.items():
        if v is not None:
            _CFG[k] = v
    _RNG = np.random.default_rng(int(_CFG["seed"]))


def _diag(key: str, value: float):
    _DIAG.setdefault(key, []).append(float(value))


def pop_diagnostics() -> dict[str, float]:
    """Mean of each diagnostic since the last call, then clear. Always all DIAG_KEYS.

    NaN for a key nothing was recorded under -- see the note on DIAG_KEYS for why the
    shape must not depend on what this rank saw.
    """
    out = {k: (float(np.mean(_DIAG[k])) if _DIAG.get(k) else float("nan")) for k in DIAG_KEYS}
    _DIAG.clear()
    return out


# ---------------------------------------------------------------------------
# The roll-null
#
# The score itself now lives in `roll_null.py`, shared with the attention and GLIMPSE
# rewards, which offer it as `--overlap_metric logratio` / `--glimpse_metric logratio`.
# What stays here is this reward's CONFIG and its DIAGNOSTICS -- the two things that are
# per-reward -- so `--grad_null_offsets` and friends keep meaning exactly what they did
# and a grad run is byte-identical to one from before the split. The names below are
# re-exported because test_grad_reward_cpu.py and the probes import them from here.
# ---------------------------------------------------------------------------

inframe_offsets = _RN.inframe_offsets
_centroid_eccentricity = _RN.centroid_eccentricity


def sample_offsets(mask: np.ndarray, k: int, rng) -> tuple[list[tuple[int, int]], bool]:
    """-> (offsets, fell_back_to_toroidal). Without replacement; identity excluded."""
    return _RN.sample_offsets(mask, k, rng, inframe=bool(_CFG["inframe_rolls"]),
                              min_inframe=int(_CFG["min_inframe"]))


def step_logratio(step_map: np.ndarray, mask: np.ndarray, rng=None) -> float | None:
    """log N(U) - log N_0, clipped, with this reward's knobs and diagnostics."""
    r, info = _RN.logratio(step_map, mask, _RNG if rng is None else rng,
                           n_offsets=int(_CFG["null_offsets"]),
                           clip=float(_CFG["logratio_clip"]),
                           inframe=bool(_CFG["inframe_rolls"]),
                           min_inframe=int(_CFG["min_inframe"]))
    if r is None:
        return None
    for key in ("union_frac", "ecc", "n_image", "logratio_raw", "clip_frac",
                "toroidal_frac", "n_offsets"):
        _diag(key, info[key])
    return r


# ---------------------------------------------------------------------------
# The reward
# ---------------------------------------------------------------------------

def _norm_text(s: str) -> str:
    """Collapse whitespace, lowercase, drop trailing punctuation.

    Trailing punctuation has to go: the steps come from a sentence splitter and the
    grounding call appends its own period, so "a cat." and "a cat" are the same step and
    would otherwise both survive the dedupe -- which is the whole hack.
    """
    return re.sub(r"[\s.;,:!?]+$", "", re.sub(r"\s+", " ", str(s)).strip().lower())


def _dedupe(steps: list[dict]) -> tuple[list[dict], float]:
    """-> (steps with repeated texts dropped, duplicate fraction).

    The mean over steps is what the wov0.4 run hacked: re-quoting one trivially-groundable
    sentence pulls the mean up and dilutes the hard perception steps (duplicate-sentence
    fraction 0.00 -> 0.19 over steps 1000-2000). Deduping removes the payoff; the fraction
    is logged either way so the behaviour stays visible if this is turned off.
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


def _score_step(step_map, mask):
    """One grounded step's score: the roll-null, or any of the three overlap metrics.

    The gradient map used to have exactly one scoring mode. It now shares the metric
    dispatcher with the attention and GLIMPSE maps, so `--overlap_metric` means the same
    thing whichever `--saliency_method` produced the map.

    'logratio' keeps this module's OWN path rather than going through
    `overlap_rewards._step_score`, and that is deliberate: the roll-null is the gradient
    reward's historical mode, its knobs are `--grad_null_offsets` and friends, and its
    by-products are logged as `grad/*`. Routing it through the shared dispatcher would
    silently move it onto `--rollnull_*` and change what an existing command line means.
    The other three have no such history here, so they use the shared implementations
    unchanged -- there is no second copy of mean_in_v2 in this file.
    """
    metric = _CFG.get("metric", "logratio")
    if metric == "logratio":
        return step_logratio(step_map, mask)
    # The union monitors come free with the roll-null; on the other metrics nothing else
    # records them, and they are exactly what says whether a rising score is the union
    # growing rather than the map improving.
    _diag("union_frac", mask.mean())
    _diag("ecc", _centroid_eccentricity(mask))
    _diag("n_image", float(np.sqrt((np.asarray(step_map, dtype=np.float64) ** 2).sum())))
    return _OR._step_score(step_map, mask, metric=metric)


def think_grad_reward(
    completions=None, saliency_map=None, valid_list=None, image=None, natural=None, **kwargs
):
    """Per-completion roll-null gradient reward. See module docstring.

    Returns a list (len == n completions) of floats, or None where there is no grounded
    observe step, or where --grad_natural_only masks a non-natural row.
    """
    n = len(saliency_map)
    if valid_list is None:
        valid_list = [True] * n

    if _CFG.get("natural_only"):
        if natural is None:
            raise KeyError(
                "--grad_natural_only requires a boolean 'natural' column in the dataset, "
                "but none reached the reward function. Use a corpus built by "
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
        r = _score_step(step_map, mask)
        if r is not None:
            n_grounded += 1
            per_completion[c].append(r)
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
