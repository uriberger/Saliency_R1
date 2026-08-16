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

"""The roll-null score: a step's mass inside its box union, against the SAME union moved.

    N(A)^2 = sum_{j in A} m_j^2                 the map's squared mass over a patch set
    N_0^2  = (1/K) sum_k N(U'_k)^2              K translates of U, same shape and area
    r      = clip( log N(U) - log N_0, -c, +c )

`U'_k` is `U` translated to a random offset. Chance is exactly 0.

This file is the ONE implementation, shared by all three map types. It was written for
the gradient reward (`grad_rewards.py`, `--reward_variant grad`, where it is the only
scoring mode) and is now also selectable as `--overlap_metric logratio` /
`--glimpse_metric logratio` for the attention and GLIMPSE maps. Nothing here reads config
or writes diagnostics: every knob is an argument and every by-product is returned, so the
three callers can own their own `_CFG` and their own metric buffers without this drifting
into three subtly different roll-nulls.

WHY THIS SHAPE, in the order the alternatives were rejected -- the reasoning is the
gradient reward's and applies unchanged to any non-negative map:

  N(U) alone.        Has three scales in it the policy controls and that are not
                     grounding: how confident the model is, how large the union is
                     (N(U) ~ sqrt(|U|) for ANY map, so "name things that ground to bigger
                     boxes" raises it mechanically), and the map's overall level.
  N(U) - N(control). Fixes the chance level, but subtraction removes an ADDITIVE offset
                     while the confounds are MULTIPLICATIVE. Also mostly a box-size meter.
  N(U) / N(outside). Cancels the multiplicative factor, but the denominator's support
                     shrinks as the union grows: unbounded, noisy, undefined at full
                     coverage -- and union growth is the known hack direction.
  N(U) / N(U'), THIS. The control has the same shape and area as `U` by construction, so
                     it never degenerates, and it is self-limiting exactly where it needs
                     to be: under a uniform translation E|U n U'| / |U| = |U|/n, so a
                     union covering most of the image is compared against itself and the
                     ratio collapses toward 1.

Three details that are not cosmetic:

  * The null pools the SQUARED masses over K offsets BEFORE the log. Averaging log-ratios
    instead lets one control landing on a dead region dominate the null and blow up that
    step's score.
  * Scoring in log space makes chance exactly 0, makes the per-completion aggregate a
    geometric mean of ratios, and makes the clip symmetric. That matters more than usual
    under `scale_rewards=True`, where one outlier completion otherwise takes most of its
    group's normalised advantage.
  * Offsets are drawn IN-FRAME (the translate stays inside the grid) rather than
    toroidally, so a control never wraps across the image border. With too few in-frame
    positions -- a near-full-frame union -- it falls back to toroidal and SAYS SO in the
    returned info, because that fallback changes what the null means.

ONE PROPERTY TO KNOW WHEN USING THIS ON A NON-GRADIENT MAP. The mass is SQUARED, which is
what makes `N` a euclidean norm and is the natural reading for a gradient (the map is
already a per-patch gradient norm). On an attention or GLIMPSE map, whose patches are a
mass-like quantity rather than a norm, squaring weights concentrated peaks more heavily
than a plain sum would: a map with all its mass on one in-box patch scores higher than one
spreading the same mass over the whole box. That is a real difference from `mean_in_v2`,
not a bug, and it is why the two can disagree on the same map -- but it means "the
roll-null" is the same FORMULA across map types and not automatically the same QUESTION.
"""

from __future__ import annotations

import numpy as np


def inframe_offsets(mask: np.ndarray) -> list[tuple[int, int]]:
    """Every translation that keeps all of `mask` inside the grid, excluding identity.

    The mask is a union of boxes and need not be rectangular, so the constraint is on its
    bounding box: shifting by (dy, dx) moves every True cell, and no cell leaves the grid
    exactly when the bounding box does not.
    """
    gh, gw = mask.shape
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return []
    r0, r1, c0, c1 = int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1])
    return [(dy, dx)
            for dy in range(-r0, gh - r1)
            for dx in range(-c0, gw - c1)
            if (dy, dx) != (0, 0)]


def sample_offsets(mask: np.ndarray, k: int, rng, *, inframe: bool = True,
                   min_inframe: int = 4) -> tuple[list[tuple[int, int]], bool]:
    """-> (offsets, fell_back_to_toroidal). Without replacement; identity excluded."""
    pool = inframe_offsets(mask) if inframe else []
    toroidal = len(pool) < int(min_inframe)
    if toroidal:
        gh, gw = mask.shape
        pool = [(dy, dx) for dy in range(gh) for dx in range(gw) if (dy, dx) != (0, 0)]
    if not pool:
        return [], toroidal
    idx = rng.choice(len(pool), size=min(k, len(pool)), replace=False)
    return [pool[int(i)] for i in idx], toroidal


def centroid_eccentricity(mask: np.ndarray) -> float:
    """Distance of the union's centroid from the grid centre, 0 at centre, 1 at a corner.

    The centre hack -- naming central objects because a centre-heavy map beats its
    translates for free -- shows up as this rising, and as its correlation with the score.
    """
    gh, gw = mask.shape
    ys, xs = np.nonzero(mask)
    cy = (ys.mean() + 0.5) / gh - 0.5
    cx = (xs.mean() + 0.5) / gw - 0.5
    return float(np.hypot(cy, cx) / np.hypot(0.5, 0.5))


def logratio(step_map: np.ndarray, mask: np.ndarray, rng, *, n_offsets: int = 16,
             clip: float = 1.0, inframe: bool = True, min_inframe: int = 4):
    """-> (score, info) or (None, info). See the module docstring.

    None (rather than 0.0) on a degenerate step is deliberate: a 0 here is not a low
    score, it is an absent measurement, and the caller drops it from the mean over steps
    exactly as it drops an ungroundable one.

    `info` always carries the same keys so a caller can log a FIXED diagnostic set,
    whatever this step turned out to be; the values are None when nothing was computed.
    """
    info = {"logratio_raw": None, "clip_frac": None, "toroidal_frac": None,
            "n_offsets": None, "union_frac": None, "ecc": None, "n_image": None}
    m2 = np.asarray(step_map, dtype=np.float64) ** 2
    mask = np.asarray(mask, dtype=bool)
    e_in = float(m2[mask].sum())
    if not np.isfinite(e_in) or e_in <= 0:
        return None, info

    offsets, toroidal = sample_offsets(mask, int(n_offsets), rng, inframe=inframe,
                                       min_inframe=min_inframe)
    if not offsets:
        return None, info
    e_null = float(np.mean([m2[np.roll(mask, off, axis=(0, 1))].sum() for off in offsets]))
    if not np.isfinite(e_null) or e_null <= 0:
        return None, info

    r = 0.5 * (np.log(e_in) - np.log(e_null))
    c = float(clip)
    info.update(logratio_raw=r, clip_frac=1.0 if abs(r) > c else 0.0,
                toroidal_frac=1.0 if toroidal else 0.0, n_offsets=len(offsets),
                union_frac=float(mask.mean()), ecc=centroid_eccentricity(mask),
                n_image=float(np.sqrt(m2.sum())))
    return float(np.clip(r, -c, c)), info
