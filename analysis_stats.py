#!/usr/bin/env python
"""Small exact statistics, no scipy: Wilson, Fisher, Mann-Whitney, Mantel-Haenszel.

These four keep turning up in this project's analyses and the cluster's analysis envs do
not all carry scipy, so they live here rather than being re-derived per probe. Everything
is exact or a named approximation, and n is a few thousand at most, which is what makes
the exact forms affordable.

NOTE ON DUPLICATION. `human_box_vs_correct.py`, on the unmerged branch
`analysis/human-box-vs-correct`, carries copies of all four. It should import from here
when that branch merges.
"""

from __future__ import annotations

import math

import numpy as np


def wilson(k, n, z=1.959964):
    """95% Wilson interval, so a 0/12 cell does not read as a hard 0%."""
    if n == 0:
        return (float("nan"), float("nan"))
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - half), min(1.0, c + half))


def fisher_exact_2x2(a, b, c, d):
    """Two-sided Fisher exact p for [[a, b], [c, d]], summing every table at most as
    likely as the observed one."""
    n = a + b + c + d
    if n == 0:
        return None
    r1, r2, c1 = a + b, c + d, a + c

    def prob(x):
        return (math.comb(r1, x) * math.comb(r2, c1 - x)) / math.comb(n, c1)

    lo, hi = max(0, c1 - r2), min(r1, c1)
    p_obs = prob(a)
    # 1e-9 relative slack: the observed table must count itself despite float rounding.
    return float(min(1.0, sum(prob(x) for x in range(lo, hi + 1)
                              if prob(x) <= p_obs * (1 + 1e-9))))


def mannwhitney(x, y):
    """(P(a random x beats a random y), two-sided p), tie-corrected normal approximation.

    Reported next to a 2x2 because a threshold throws away how far from the cut each unit
    sits, and a conclusion should not hinge on where the cut went.
    """
    x, y = list(x), list(y)
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return None, None
    v = np.array(x + y, dtype=np.float64)
    order = np.argsort(v, kind="stable")
    ranks = np.empty(v.size, dtype=np.float64)
    ranks[order] = np.arange(1, v.size + 1, dtype=np.float64)
    uniq, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
    sums = np.zeros(cnt.size, dtype=np.float64)
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    u = ranks[:nx].sum() - nx * (nx + 1) / 2.0
    auc = u / (nx * ny)
    n = nx + ny
    tie = float(((cnt ** 3 - cnt).sum()) / (n * (n - 1)))
    sd = math.sqrt(nx * ny / 12.0 * ((n + 1) - tie))
    if sd == 0:
        return float(auc), None
    z = (u - nx * ny / 2.0) / sd
    return float(auc), float(math.erfc(abs(z) / math.sqrt(2.0)))


def mantel_haenszel(tables):
    """Common odds ratio across strata, with its continuity-corrected chi-square p (1 df).

    Each table is (a, b, c, d) = exposed&right, exposed&wrong, unexposed&right,
    unexposed&wrong. Used to hold a nuisance variable fixed -- here PICTURE DIFFICULTY,
    because an easy picture is easy whatever its geometry, and a raw association between
    a geometric property and correctness is also what a difficulty imbalance produces.
    """
    num = den = 0.0
    sa = se = sv = 0.0
    for a, b, c, d in tables:
        n = a + b + c + d
        if n < 2:
            continue
        num += a * d / n
        den += b * c / n
        sa += a
        se += (a + b) * (a + c) / n
        sv += ((a + b) * (c + d) * (a + c) * (b + d)) / (n * n * (n - 1))
    if den == 0 or sv == 0:
        return None, None
    chi2 = (abs(sa - se) - 0.5) ** 2 / sv
    return float(num / den), float(math.erfc(math.sqrt(chi2 / 2.0)))


def spearman(x, y):
    """(rho, two-sided p) by the normal approximation on Fisher's z. Ties averaged."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = x.size
    if n < 4:
        return None, None

    def rank(v):
        order = np.argsort(v, kind="stable")
        r = np.empty(v.size, dtype=np.float64)
        r[order] = np.arange(1, v.size + 1, dtype=np.float64)
        uniq, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
        s = np.zeros(cnt.size, dtype=np.float64)
        np.add.at(s, inv, r)
        return (s / cnt)[inv]

    rx, ry = rank(x), rank(y)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    if not np.isfinite(rho) or abs(rho) >= 1.0:
        return rho, 0.0
    z = math.atanh(rho) * math.sqrt((n - 3) / 1.06)
    return rho, float(math.erfc(abs(z) / math.sqrt(2.0)))


def n_per_group_for(p1, p2, power=0.80, alpha=0.05):
    """Two-proportion, two-sided: n per group to detect p1 vs p2. -> int, or None.

    Printed by the power section so "Arm 0 is underpowered" is a number rather than an
    adjective, and so the interventional arm is sized from a measured effect rather than
    from a guess.
    """
    d = abs(p1 - p2)
    if d <= 0:
        return None
    za, zb = 1.959964, {0.80: 0.8416, 0.90: 1.2816, 0.95: 1.6449}.get(power, 0.8416)
    pbar = (p1 + p2) / 2.0
    n = ((za * math.sqrt(2 * pbar * (1 - pbar))
          + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2) / (d * d)
    return int(math.ceil(n))
