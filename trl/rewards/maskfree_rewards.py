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

"""Two MASK-FREE rewards (--maskfree flatness|mass). No boxes, no Grounding-DINO.

WHY THESE EXIST. `mean_in`, the reward this project has spent the most time on, is

    mean_in = mean_U(m) / max(m)

and the denominator is taken over the WHOLE map. It knows nothing about `U`. Measured on
the val_natural probe, on the quantity GRPO actually sees -- the per-completion reward
with the generation group's mean removed -- the mask contributes far less than the name
suggests:

    mask fed to mean_in                      all unions    union < 0.30
    the real DINO union                        1.000          1.000
    in-frame roll (what --placebo roll did)   +0.538         +0.449
    uniform toroidal roll (real relocation)   +0.353         -0.015
    random box of the same area, anywhere     +0.505         +0.457
    mean_all(m)/max(m)  -- NO MASK AT ALL     +0.690         +0.678

Two readings. Once a mask is genuinely relocated its score stops predicting the reward
(-0.015). And the best single predictor of `mean_in` is a statistic that never sees a
box. So `mean_in` behaves like a flatness reward -- an inverse peak-to-mean ratio -- and
the grounding is close to decoration.

That is a hypothesis about a mechanism, and these two rewards are the experiment that
can kill it. Both drop Grounding-DINO entirely.

    flatness   mean(m) / max(m) over the whole patch grid. `mean_in` with the union
               replaced by the image. Scale-invariant (m -> c*m is a no-op), so it is
               purely the map's SHAPE: 1.0 for a flat map, ->0 for a single spike.
    mass       log(sum(m)) + anchor. The map is a slice of a softmax row (see below), so
               sum(m) is the literal probability mass the step's think-tokens put on
               image patches. docs/sharpness-results.md is the reason this is the second
               variant and not an afterthought: across four map families computed by four
               unrelated methods, total image mass is the ONLY map property that predicts
               correctness under controls -- "DINO survives in 0 of 4 families. SHARP in
               2. MASS in 3." The measured runs move it: the mean_in-trained 8k policy
               puts 2.3x the cold start's mass on the image (0.00393 -> 0.00893).

`flatness` and `mass` are not the same knob and are deliberately shipped as a pair.
`flatness` is scale-INVARIANT and `mass` is scale-ONLY, so between them they span the
two directions `mean_in` moved in, and a run of each says which one carried the benefit.
Within a group at the cold start they correlate +0.38, so they are not redundant either.

WHY log FOR mass, AND WHAT THE ANCHOR IS FOR. Image mass varies multiplicatively and its
LEVEL moves a lot during training (0.0039 -> 0.0089 over the mean_in run). A linear
reward's within-group sd would grow with that level, so the effective pressure
`w x sd_within` would silently drift ~2x over a run -- the calibration failure that made
the three placebo arms hard to compare. In log space the sd is stable and the weight
means one thing from step 0 to step 4000. The anchor is a constant, invisible to GRPO
(the advantage subtracts the group mean), and exists only so a scored completion never
reads NEGATIVE: under the pre-8489767 `.nansum(dim=1)` fold an UNSCORED reward is read as
0, which would make "have no gradeable observe step" the best available move on the
auxiliary dimension. `trl_repo/` now carries the imputing fold and the trap is closed,
but the anchor costs nothing and the failure it prevents is silent. Same reasoning, and
the same fix, as placebo_rewards' `length_anchor`.

The guarantee it buys is EMPIRICAL, not structural: no finite anchor makes a log positive
for an arbitrarily small mass. 18.0 is set against the measured minimum over 13,648
observe steps and all ten models in the val_natural probe --

    min 1.40e-04   p1 4.22e-04   median 5.72e-03   p90 1.62e-02   max 8.49e-02

-- whose log needs 8.87, so the default clears the worst observed step by four orders of
magnitude (exp(-18) = 1.5e-08). On a corpus whose maps are far dimmer than any seen here, check
`maskfree/mass` in the logs and raise --maskfree_mass_anchor rather than assuming.

WHAT IS NOT DROPPED, and why this is still comparable to the reference. The observe-step
segmentation (sentence split + the FLAN-T5 classifier) and the layer-22 attention
re-forward both stay. The step maps handed to this function are byte-for-byte the ones
`think_overlap_reward` would score; only the box union and the metric differ. Dropping
segmentation too would change WHICH tokens the map covers and make this a third
experiment rather than a control on the second.

THE SCORED SET, which is the one place this is NOT a single-variable change. The overlap
reward returns None -- unscored, imputed to the group mean -- for a completion DINO could
ground nowhere. Without DINO that question cannot be asked, so a step here counts when it
has a map with a positive maximum, and a completion counts when it has at least one such
step. That is a SUPERSET of the reference's scored set in principle.

In practice, on the val_natural probe (240 completions, 874 observe steps with a map):

    steps DINO grounded                859 / 874   98.3%
    completions the DINO reward scored 231 / 240   96.2%
    completions scored MASK-FREE       231 / 240   96.2%    +0

Identical. Every completion with a gradeable observe step had at least one groundable
one, so on this corpus the superset is not larger. `--maskfree-parity` re-imposes the
DINO gate anyway, at the full DINO cost, for anyone who wants the guarantee rather than
the measurement; it is off by default because the measurement says it buys nothing and
Grounding-DINO is 16.6 s of a 40.5 s optimizer step.

CALIBRATION, measured the same way every other weight in this repo was -- pooled
within-group sd on the cold-start policy, val_natural, temperature 1, 8 generations:

    variant     level      sd_within   w = 0.4 x sd_within(mean_in) / sd_within(variant)
    flatness    0.0513     0.0064      0.45
    mass       12.1621     0.4586      0.006

(the `mass` LEVEL moves with the anchor and nothing else; its sd_within, which is what
the weight is set from, does not.)

with sd_within(mean_in) = 0.0071 on the same completions. The launcher resolves both;
see its --maskfree block. Re-measure on the corpus you train on with
`overlap_metric_spread.py`, which computes both rows by importing THIS module.
"""

from __future__ import annotations

import math

import numpy as np

KINDS = ("flatness", "mass")

_CFG = {
    # None = disabled; think_maskfree_reward is then never installed by the launcher.
    "kind": None,
    # Constant added to log(sum(m)). Advantage-invariant; see the module docstring for
    # the one thing it is not invisible to. 18.0 against a MEASURED minimum image mass of
    # 1.402e-04 over 13,648 observe steps and all ten models in the val_natural probe,
    # which needs 8.87 -- so this covers three further orders of magnitude below anything
    # ever observed (exp(-18) = 1.5e-08). The margin is free (a constant cancels in the advantage) and the
    # first draft's 8.0 was NOT enough: it went negative on the lowest ~1% of steps.
    "mass_anchor": 18.0,
    # Re-impose the DINO scored/unscored gate. Off by default: measured to change nothing
    # on val_natural, and it costs the full grounding call, which is the point of this
    # reward. See think_maskfree_reward.
    "parity": False,
}

# FIXED key set, always all of it, for the reason grad_rewards.DIAG_KEYS documents: the
# trainer gathers these across ranks, so a key set that depended on what a rank happened
# to see would mean a rank-dependent number of collectives, which hangs rather than fails.
#
# `peak` and `mean` are logged next to the score because the two variants move them in
# different combinations, and the pair is what says which one a run is actually riding.
# `mass` is logged under BOTH kinds on purpose -- a `flatness` run that raises mass is the
# headline result this experiment exists to find, and it is invisible if only the scored
# quantity is recorded.
DIAG_KEYS = ("flatness", "mass", "peak", "mean", "n_steps")
_DIAG: dict[str, list[float]] = {}


def _diag(key: str, value: float):
    _DIAG.setdefault(key, []).append(float(value))


def pop_diagnostics() -> dict[str, float]:
    """Mean of each mask-free diagnostic since the last call, then clear.

    Always all of DIAG_KEYS; NaN for a key nothing was recorded under.
    """
    out = {k: (float(np.mean(_DIAG[k])) if _DIAG.get(k) else float("nan")) for k in DIAG_KEYS}
    _DIAG.clear()
    return out


def is_active() -> bool:
    """True when a mask-free reward replaces the overlap reward. Rank-uniform (it is set
    from the CLI on every process), so the trainer may branch logging collectives on it."""
    return _CFG["kind"] is not None


def needs_dino() -> bool:
    """Whether this reward will call Grounding-DINO at all.

    The launcher reads this decision (as --maskfree-parity) to decide whether to start the
    DINO server, so "no boxes" is enforced by not having a server rather than by trusting
    the reward not to call one.
    """
    return bool(_CFG["kind"]) and bool(_CFG["parity"])


def configure(**kwargs):
    """Set the mask-free config from the CLI flags. None values are ignored."""
    for k, v in kwargs.items():
        if v is not None:
            _CFG[k] = v
    kind = _CFG["kind"]
    if kind is None:
        return
    if kind not in KINDS:
        raise ValueError(f"--maskfree must be one of {'|'.join(KINDS)} (got {kind!r})")


# ---------------------------------------------------------------------------
# The two values
# ---------------------------------------------------------------------------

def flatness(step_map) -> float | None:
    """mean(m) / max(m) over the whole patch grid; None on a degenerate map.

    This is exactly `overlap_rewards._mean_in` with the union replaced by the image, so
    the two are on the same scale and their levels are directly comparable (0.051 vs
    0.039 at the cold start). In (0, 1]: 1.0 iff the map is constant, -> 0 for a spike.
    Invariant to m -> c*m, so it cannot be raised by simply attending to the image more --
    that is `mass`, and keeping them separable is the point of shipping both.
    """
    m = np.asarray(step_map, dtype=np.float64)
    mx = float(m.max()) if m.size else 0.0
    if not np.isfinite(mx) or mx <= 0:
        return None
    return float(m.mean()) / mx


def mass(step_map, anchor: float = 18.0) -> float | None:
    """log(sum(m)) + anchor; None on an all-zero or non-finite map.

    sum(m) is the fraction of the step's softmax row spent on image patches: the map is
    that row sliced to the image columns and never renormalised, which is what makes this
    a probability mass and not an arbitrary scale. Same quantity `--mass-floor-tau` gates
    on and `overlap_rewards._mass_gate` divides by, and the same one
    docs/sharpness-results.md finds is the only map property predicting correctness.
    """
    m = np.asarray(step_map, dtype=np.float64)
    if not m.size:
        return None
    total = float(m.sum())
    if not np.isfinite(total) or total <= 0:
        return None
    return math.log(total) + float(anchor)


def _step_value(step_map, kind: str, anchor: float):
    """The configured mask-free value for one observe step, plus its diagnostics."""
    m = np.asarray(step_map, dtype=np.float64)
    f = flatness(m)
    g = mass(m, anchor)
    if f is None or g is None:
        return None
    # Recorded on every scored step regardless of which kind is being trained: a
    # `flatness` run that moves mass is the finding, and vice versa.
    _diag("flatness", f)
    _diag("mass", g)
    _diag("peak", float(m.max()))
    _diag("mean", float(m.mean()))
    return f if kind == "flatness" else g


# ---------------------------------------------------------------------------
# The reward
# ---------------------------------------------------------------------------

def think_maskfree_reward(
    completions=None, saliency_map=None, valid_list=None, image=None, natural=None, **kwargs
):
    """Per-completion mask-free reward. See module docstring.

    Structurally identical to `think_overlap_reward` -- same `--overlap_natural_only`
    gate, same per-completion mean over its scored observe steps, same multiplicative
    format gate, same None for a completion with nothing to score -- with the box union,
    the Grounding-DINO call and the metric removed.
    """
    kind = _CFG["kind"]
    if kind not in KINDS:
        raise ValueError(
            f"think_maskfree_reward was called with --maskfree {kind!r}; "
            f"expected one of {'|'.join(KINDS)}."
        )
    anchor = float(_CFG["mass_anchor"])

    n = len(saliency_map)
    if valid_list is None:
        valid_list = [True] * n

    # --overlap_natural_only, read from the overlap reward's config so the flag means the
    # same thing here as there and there is only one switch to set.
    from . import overlap_rewards as _ORW

    if _ORW._CFG.get("natural_only"):
        if natural is None:
            raise KeyError(
                "--overlap_natural_only requires a boolean 'natural' column in the dataset, "
                "but none reached the reward function. Use a corpus built by "
                "build_grpo_sets.py (cold_data/grpo_sets/*), or drop the flag."
            )
        scored = [bool(x) for x in natural]
    else:
        scored = [True] * n

    # --maskfree-parity: re-impose the reference's scored/unscored set by running the real
    # grounding pipeline and using it ONLY as a boolean, exactly as placebo_rewards does.
    # Off by default -- measured to change nothing, and it is the entire cost of the
    # reward. When off, nothing in this function touches DINO.
    groundable = None
    if _CFG["parity"]:
        groundable = set()
        flat_images, flat_texts, flat_owner = [], [], []
        for c, steps in enumerate(saliency_map):
            if not steps or not scored[c]:
                continue
            for si, st in enumerate(steps):
                flat_images.append(image[c])
                flat_texts.append(st["text"])
                flat_owner.append((c, si))
        boxes_per_item = _ORW._dino_boxes(flat_images, flat_texts) if flat_images else []
        for (c, si), boxes in zip(flat_owner, boxes_per_item):
            gh, gw = saliency_map[c][si]["map"].shape
            mask = _ORW._union_mask(boxes, gh, gw)
            if mask is None:
                continue
            if _ORW._step_score(saliency_map[c][si]["map"], mask) is None:
                continue
            groundable.add((c, si))

    per_completion = [[] for _ in range(n)]
    for c, steps in enumerate(saliency_map):
        if not steps or not scored[c]:
            continue
        for si, st in enumerate(steps):
            if groundable is not None and (c, si) not in groundable:
                continue
            v = _step_value(st["map"], kind, anchor)
            if v is not None:
                per_completion[c].append(v)

    rewards = []
    for c in range(n):
        if not scored[c]:
            rewards.append(None)  # non-natural under --overlap_natural_only -> mask
            continue
        vals = per_completion[c]
        if not vals:
            rewards.append(None)  # no gradeable observe step -> mask (neutral, NOT 0)
            continue
        _diag("n_steps", float(len(vals)))
        # The format gate is multiplicative and kept identical to the overlap reward's.
        # It is unreachable in training -- the trainer builds no observe-step maps for a
        # format-invalid completion, so such a row arrives with no steps and is masked
        # above -- and `mass_anchor` is what keeps the gated 0 from being the BEST score
        # if it ever becomes reachable.
        rewards.append(float(np.mean(vals)) * (1.0 if valid_list[c] else 0.0))
    return rewards
