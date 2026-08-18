#!/usr/bin/env python
"""How CONCENTRATED is a saliency map, irrespective of where the mass sits?

Every grounding measurement in this repo asks the same question -- does the map put
its weight on the boxes the step names -- and every one of them needs Grounding-DINO
to answer it. This module asks the rival question that needs no boxes at all: is the
map *sharp*? The hypothesis it exists to test is that the model does better when it
attends to ANY small part of the image rather than smearing over all of it, in which
case a grounding reward would be picking up a concentration effect and mislabelling
it as "looked at the right place".

Six of the seven statistics below are INVARIANT UNDER ANY PERMUTATION OF THE PATCHES.
That is the point, and it is what makes the comparison clean: they cannot encode a
location, only a shape of the value distribution, so a correlation they show with
correctness is by construction not a grounding effect. The seventh, `sconc`, is the
deliberate exception -- it measures whether the mass sits in one compact blob or is
scattered across the frame, which is "focused *somewhere*" without ever asking where
that somewhere is.

    cv     coefficient of variation, std(x)/mean(x) = sqrt(P*sum(p^2) - 1).
           The user-facing version of "the standard deviation is larger". Scale-free
           by construction, 0 for a flat map, sqrt(P-1) for a delta.
    gini   Gini coefficient of the P values. 0 flat, ->1 delta. Cares about the whole
           Lorenz curve rather than the tail alone.
    nent   1 - H(p)/ln(P), normalised negentropy. 0 flat, 1 delta. Divided by ln(P)
           because the grids here are not one size (144-256 patches), and raw entropy
           would then be mostly a readout of image resolution.
    top1   mass fraction in the single largest patch.
    top5   mass fraction in the largest ceil(0.05*P) patches.
    top20  mass fraction in the largest ceil(0.20*P) patches.
           A tail family: three points on the same curve, kept because "sharp" could
           mean one spike (top1) or one object's worth of patches (top20), and those
           are different claims about what the model is doing.
    sconc  spatial concentration: 1 - rms spread of the mass about its own centroid,
           over the rms spread of a flat map. 0 flat, ->1 all mass in one place, and
           NEGATIVE when the mass sits in two far-apart clumps -- a map can be very
           sharp in value and very spread in space, and only this column sees that.

ALL SEVEN ARE ORIENTED SO THAT HIGHER MEANS SHARPER. Read every sign in the report
against that convention and nothing else; the DINO metrics it is compared with have
their own orientation (higher = more mass inside the union) and the two must not be
mixed up.

Scale invariance. Every column is computed on p = x / sum(x), so multiplying a map by
a constant cannot move any of them. This matters more than it looks: the maps being
compared span nine orders of magnitude in raw scale (rollout_mean lives at 1e-10,
grad at 1e0), and the magnitude of a map is a separate, already-measured quantity --
`mass` in the flow scan, and the strongest single correlate of correctness anyone has
found here at +0.22..+0.29. The whole value of a scale-free concentration statistic is
that it is orthogonal to that by construction, so a correlation it shows is not the
mass finding in another coat.

Negative entries. The rollout `incL` columns are differences and go negative. There is
no distribution to speak of then, so the map is rectified (max(x, 0)) before anything
else and `neg_frac` reports what share of the absolute mass that discarded. Read a
column with a large `neg_frac` as a statistic about the positive part only.
"""

from __future__ import annotations

import numpy as np

SHARP_NAMES = ("cv", "gini", "nent", "top1", "top5", "top20", "sconc")

#: Columns invariant under an arbitrary permutation of the patch axis, i.e. those that
#: provably carry no information about WHERE the mass is. `sconc` is not among them.
PERM_INVARIANT = ("cv", "gini", "nent", "top1", "top5", "top20")

TOP_FRACS = {"top1": None, "top5": 0.05, "top20": 0.20}   # None = exactly one patch


def sharpness(maps, grid):
    """Concentration statistics of every map in `maps`.

    maps  [..., S, P] float, P = gh*gw in image-token (row-major) order
    grid  (gh, gw)

    -> (sharp [..., S, len(SHARP_NAMES)], neg_frac [..., S])

    NaN in every column of a map whose rectified sum is zero -- there is no
    distribution to describe, and 0 would be a made-up answer.
    """
    gh, gw = int(grid[0]), int(grid[1])
    x = np.asarray(maps, dtype=np.float64)
    P = x.shape[-1]
    if P != gh * gw:
        raise ValueError(f"maps have {P} patches but grid {gh}x{gw} wants {gh * gw}")
    lead = x.shape[:-1]                       # (..., S)
    flat = x.reshape(-1, P)
    n = flat.shape[0]

    absmass = np.abs(flat).sum(axis=1)
    pos = np.maximum(flat, 0.0)
    tot = pos.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        neg_frac = np.where(absmass > 0, 1.0 - tot / absmass, np.nan)
    ok = tot > 0
    p = np.zeros_like(pos)
    np.divide(pos, tot[:, None], out=p, where=ok[:, None])

    out = np.full((n, len(SHARP_NAMES)), np.nan)
    idx = {nm: i for i, nm in enumerate(SHARP_NAMES)}

    # --- value-distribution columns (all permutation-invariant) ------------------
    # cv = std/mean = sqrt(P*sum(p^2) - 1) algebraically, but that form subtracts two
    # nearly equal numbers on a near-flat map and the sqrt then amplifies the
    # cancellation into ~1e-8. Summing the deviations from the known mean 1/P instead
    # is the same quantity with no cancellation at all.
    dev = p - 1.0 / P
    out[:, idx["cv"]] = np.sqrt(np.maximum(P * (dev * dev).sum(axis=1), 0.0))

    with np.errstate(invalid="ignore", divide="ignore"):
        ent = -(np.where(p > 0, p * np.log(p), 0.0)).sum(axis=1)
    out[:, idx["nent"]] = 1.0 - ent / np.log(P)

    srt = np.sort(p, axis=1)                                  # ascending
    ranks = np.arange(1, P + 1, dtype=np.float64)
    # G = 2*sum(i*x_(i))/(P*sum(x)) - (P+1)/P, with sum(x) == 1 here.
    out[:, idx["gini"]] = 2.0 * (srt * ranks).sum(axis=1) / P - (P + 1.0) / P

    csum = np.cumsum(srt[:, ::-1], axis=1)                    # descending cumulative
    for nm, frac in TOP_FRACS.items():
        k = 1 if frac is None else max(1, int(np.ceil(frac * P)))
        out[:, idx[nm]] = csum[:, k - 1]

    # --- spatial column (the one that is NOT permutation-invariant) --------------
    # Both axes are put on 0..1 so that a 10x16 grid and a 16x16 grid are comparable;
    # the reference is the exact discrete-uniform variance on the same grid, not the
    # continuous 1/12, so a flat map reads exactly 0 at any resolution.
    yy = ((np.arange(gh, dtype=np.float64) + 0.5) / gh)[:, None].repeat(gw, 1).ravel()
    xx = ((np.arange(gw, dtype=np.float64) + 0.5) / gw)[None, :].repeat(gh, 0).ravel()
    cy = p @ yy
    cx = p @ xx
    var = (p @ (yy * yy) - cy * cy) + (p @ (xx * xx) - cx * cx)
    var_flat = (1.0 - 1.0 / gh ** 2) / 12.0 + (1.0 - 1.0 / gw ** 2) / 12.0
    out[:, idx["sconc"]] = 1.0 - np.sqrt(np.maximum(var, 0.0) / var_flat)

    out[~ok] = np.nan
    return out.reshape(*lead, len(SHARP_NAMES)), neg_frac.reshape(lead)


def describe(sharp, names=SHARP_NAMES, prefix="   "):
    """One line per column: mean, sd and the finite count. For a scan's log."""
    a = np.asarray(sharp, dtype=np.float64).reshape(-1, len(names))
    lines = []
    for j, nm in enumerate(names):
        v = a[:, j]
        f = np.isfinite(v)
        lines.append(f"{prefix}{nm:>6}  mean {np.nanmean(v):+8.4f}  sd "
                     f"{np.nanstd(v):8.4f}  finite {int(f.sum())}/{len(v)}")
    return "\n".join(lines)
