#!/usr/bin/env python
"""Does a SHARP saliency map predict correctness as well as a WELL-PLACED one?

The project's premise is that the model does better when it looks in the right place,
and every measurement of "the right place" here runs through Grounding-DINO boxes.
The rival explanation this report exists to test needs no boxes at all: maybe the model
does better whenever its saliency map is *concentrated* -- on anything -- and the
grounding metrics have been reading concentration through a box-shaped window.

The test is a like-for-like race, run on the same scans, the same 1,157 completions and
the same statistical machinery as `head_correlation_probe.py --stage report` and
`flow_correlation_probe.py --stage report`:

    DINO block    mean_in_v2 and auroc of every column against its step's DINO union
    SHARP block   the seven box-free concentration columns of `saliency_sharpness.py`
    MASS block    the column's total weight on the image -- magnitude, not shape

Six of the seven SHARP columns are invariant under an arbitrary permutation of the
patches, so they are mathematically incapable of encoding a location. If they match or
beat the DINO block, "look at the right place" is not what the correlation is made of.

WHY EACH BLOCK NEEDS ITS OWN MULTIPLICITY CORRECTION. The head family offers 1,152
columns; scored under 7 sharpness metrics that is 8,064 tests against the DINO block's
2,304, and the largest of 8,064 draws beats the largest of 2,304 with no effect
present at all. Bonferroni over-corrects in the other direction, because neighbouring
heads are strongly correlated and the tests are nothing like independent. So the
headline number for every block is a MAX-|r| PERMUTATION THRESHOLD: shuffle the
completion labels, take the largest |r| anywhere in the block, repeat, and read the
95th percentile. That threshold is exact under the null whatever the correlation
structure, and it is what makes 8,064 sharpness tests comparable with 2,304 DINO ones.

THE UNIT IS THE COMPLETION. Steps of one completion share a label, so step-level
significance is anti-conservative; step-level rows are printed for continuity with the
earlier reports and are not what any conclusion rests on.

    python sharpness_report.py \
        --scan heads=outputs/sharpness/heads \
        --scan grad=outputs/sharpness/grad \
        --scan glimpse=outputs/sharpness/glimpse \
        --scan rollout_wnorm=outputs/sharpness/rollout_wnorm

Anything a scan wrote can be restricted with --max-union, exactly as in the two probe
reports; the default (off) is what every published number on this corpus used.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent


def _load_module(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


SHARP = _load_module("_sr_sharpness", "saliency_sharpness.py")
M = len(SHARP.SHARP_NAMES)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
class Family:
    """One map family's scan, flattened to a single column axis.

    head_correlation_probe writes [N, L, H]; flow_correlation_probe writes [N, K] with
    a `names` array. Both become [N, K] here with a list of column names, so every
    statistic below is written once.
    """

    def __init__(self, name, path, max_union=0.0):
        self.name, self.path = name, Path(path)
        files = sorted((self.path / "scan").glob("shard*.npz"))
        if not files:
            raise SystemExit(f"[{name}] no scan output under {self.path / 'scan'}")
        d = [np.load(f) for f in files]
        need = {"sharp", "npatch", "ntok"}
        if not need <= set(d[0].files):
            raise SystemExit(
                f"[{name}] {files[0]} predates the sharpness columns "
                f"(has {sorted(d[0].files)}). Re-run the scan with --overwrite.")

        cat = lambda k: np.concatenate([x[k] for x in d])       # noqa: E731
        v2, au, sh = cat("v2"), cat("auroc"), cat("sharp")
        mass = cat("mass") if "mass" in d[0].files else None
        if v2.ndim == 3:                                   # head scan [N, L, H]
            layers = d[0]["layers"]
            _, Lc, Hc = v2.shape
            self.cols = [f"L{int(layers[l])}H{h}" for l in range(Lc) for h in range(Hc)]
            v2, au = v2.reshape(len(v2), -1), au.reshape(len(au), -1)
            sh = sh.reshape(len(sh), Lc * Hc, M)
            mass = mass.reshape(len(mass), -1) if mass is not None else None
        else:                                              # flow scan [N, K]
            self.cols = [str(s) for s in d[0]["names"]]
        self.v2, self.auroc, self.sharp, self.mass = v2, au, sh, mass
        self.sharp_names = tuple(str(s) for s in d[0]["sharp_names"])
        self.row, self.step = cat("row"), cat("step")
        self.correct, self.union = cat("correct"), cat("union")
        self.npatch, self.ntok = cat("npatch"), cat("ntok")
        self.neg = cat("neg_frac") if "neg_frac" in d[0].files else None
        if self.neg is not None and self.neg.ndim == 3:
            self.neg = self.neg.reshape(len(self.neg), -1)
        self.dataset = (cat("dataset") if "dataset" in d[0].files
                        else np.array([""] * len(self.row)))

        if max_union and max_union > 0:
            keep = self.union <= float(max_union)
            if keep.sum() < 12:
                raise SystemExit(f"[{name}] --max-union {max_union} keeps "
                                 f"{int(keep.sum())} steps")
            for a in ("v2", "auroc", "sharp", "mass", "row", "step", "correct",
                      "union", "npatch", "ntok", "neg", "dataset"):
                val = getattr(self, a)
                if val is not None:
                    setattr(self, a, val[keep])
            self.capped = (int(keep.sum()), len(keep))
        else:
            self.capped = None

        # --- completion-level aggregation; the label is constant within a completion
        self.uniq = np.unique(self.row)
        self.idx = np.searchsorted(self.uniq, self.row)
        self.ccor = np.zeros(len(self.uniq))
        np.maximum.at(self.ccor, self.idx, self.correct)
        self.nsteps = np.zeros(len(self.uniq))
        np.add.at(self.nsteps, self.idx, 1.0)

        self.c_v2 = self.by_completion(self.v2)
        self.c_au = self.by_completion(self.auroc)
        self.c_sharp = np.stack(
            [self.by_completion(self.sharp[:, :, m]) for m in range(M)], axis=-1)
        self.c_mass = self.by_completion(self.mass) if self.mass is not None else None
        self.c_union = self.by_completion(self.union)
        self.c_ntok = self.by_completion(self.ntok.astype(np.float64))
        self.c_npatch = self.by_completion(self.npatch.astype(np.float64))
        # row is NOT sorted -- the shards interleave row_index -- so this is a lookup,
        # not a searchsorted.
        seen = dict(zip(self.row.tolist(), self.dataset.tolist()))
        self.c_dataset = np.array([seen[int(r)] for r in self.uniq])

    def by_completion(self, arr):
        """NaN-aware mean over each completion's steps. [N] or [N,K] -> [C] or [C,K]."""
        arr = np.asarray(arr, dtype=np.float64)
        shape = (len(self.uniq),) if arr.ndim == 1 else (len(self.uniq), arr.shape[1])
        s, cnt = np.zeros(shape), np.zeros(shape)
        np.add.at(s, self.idx, np.nan_to_num(arr, nan=0.0))
        np.add.at(cnt, self.idx, np.isfinite(arr).astype(float))
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(cnt > 0, s / cnt, np.nan)

    def covariates(self):
        """The completion-level control set, as a design matrix and its column names.

        Everything here is a candidate reason a map could look sharp AND the answer be
        right without one causing the other: a bigger grid has more patches to spread
        over, a longer step averages over more attention rows and smooths, more steps
        is a longer chain, a larger DINO union is a more cluttered image, and the
        source datasets differ in both difficulty and image statistics.
        """
        cols = [np.log(self.c_npatch), self.c_ntok, self.nsteps, self.c_union]
        names = ["log_npatch", "ntok", "nsteps", "union"]
        ds = sorted({d for d in self.c_dataset if d})
        for name in ds[1:]:                       # one held out as the reference level
            cols.append((self.c_dataset == name).astype(np.float64))
            names.append(f"ds:{name}")
        return np.column_stack(cols), names


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def _masked_corr(A, B, ok):
    """Column-wise r between matching columns of A and B [N,K], over rows `ok` [N,K].

    One pass of sums rather than one call per column: the head family asks for this
    8,064 columns at a time and a Python loop over them is the difference between a
    report that runs in seconds and one that runs in minutes.
    """
    n = ok.sum(0).astype(np.float64)
    a = np.where(ok, A, 0.0)
    b = np.where(ok, B, 0.0)
    sa, sb = a.sum(0), b.sum(0)
    saa, sbb, sab = (a * a).sum(0), (b * b).sum(0), (a * b).sum(0)
    num = n * sab - sa * sb
    den = (np.sqrt(np.maximum(n * saa - sa ** 2, 0))
           * np.sqrt(np.maximum(n * sbb - sb ** 2, 0)))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where((den > 0) & (n >= 8), num / den, np.nan)


def col_corr(X, y):
    """Pearson r of every column of X [N, ...] against y [N], NaN-aware. -> [...]."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    shape = X.shape[1:]
    X2 = X.reshape(len(X), -1)
    if np.isfinite(X2).all() and np.isfinite(y).all():
        xc = X2 - X2.mean(0)
        yc = y - y.mean()
        den = np.sqrt((xc * xc).sum(0)) * np.sqrt((yc * yc).sum())
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(den > 0, (xc.T @ yc) / den, np.nan)
        return r.reshape(shape)
    ok = np.isfinite(X2) & np.isfinite(y)[:, None]
    return _masked_corr(X2, np.broadcast_to(y[:, None], X2.shape), ok).reshape(shape)


def paired_corr(A, B):
    """r between column k of A and column k of B, for every k. -> [K]."""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    return _masked_corr(A, B, np.isfinite(A) & np.isfinite(B))


def _impute(X):
    """NaN -> that column's mean. -> (filled X [N,K], n_imputed).

    Only defensible because it is rare: a sharpness cell is NaN only when a map's
    rectified sum is exactly zero, and a DINO cell only when a step's union is empty
    or the whole grid. The count is printed so "rare" is never assumed.
    """
    X = np.array(X, dtype=np.float64)
    bad = ~np.isfinite(X)
    if bad.any():
        mu = np.nanmean(np.where(bad, np.nan, X), axis=0)
        mu = np.where(np.isfinite(mu), mu, 0.0)
        X[bad] = np.take(mu, np.where(bad)[1])
    return X, int(bad.sum())


def residualize(X, Z):
    """Least-squares residuals of every column of X [N,K] on [1, Z]. -> [N,K].

    Via the pseudo-inverse of the (tall, thin) design rather than lstsq on 8,064
    right-hand sides at once, which is the same answer for a fraction of the work.
    Constant covariates are dropped rather than left to make the design rank-deficient.
    """
    des = np.column_stack([np.ones(len(X))]
                          + [Z[:, j] for j in range(Z.shape[1]) if Z[:, j].std() > 0])
    return X - des @ (np.linalg.pinv(des) @ X)


def partial_col_corr(X, y, Z):
    """Partial r of every column of X [N,K] with y, holding Z fixed. -> [K]."""
    ok = np.isfinite(y) & np.isfinite(Z).all(axis=1)
    Xf, _ = _impute(np.asarray(X, dtype=np.float64)[ok])
    return col_corr(residualize(Xf, Z[ok]), residualize(y[ok, None], Z[ok])[:, 0])


def pairwise_partial(X, y, C):
    """Partial r of X[:,k] with y holding C[:,k] fixed, column by column. -> [K].

    Not the same as partial_col_corr: the covariate is the column's OWN partner (its
    grounding score, or its sharpness), which differs per column. Computed from the
    three pairwise correlations,

        r(x,y|c) = (r_xy - r_xc * r_yc) / sqrt((1 - r_xc^2)(1 - r_yc^2))

    which is exact as long as all three are taken over the SAME rows -- hence the one
    shared mask below rather than three pairwise-complete ones.
    """
    X, y, C = (np.asarray(a, dtype=np.float64) for a in (X, y, C))
    Y = np.broadcast_to(y[:, None], X.shape)
    ok = np.isfinite(X) & np.isfinite(C) & np.isfinite(Y)
    rxy = _masked_corr(X, Y, ok)
    rxc = _masked_corr(X, C, ok)
    ryc = _masked_corr(C, Y, ok)
    den = np.sqrt(np.maximum((1 - rxc ** 2) * (1 - ryc ** 2), 0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (rxy - rxc * ryc) / den, np.nan)


def max_abs_null(X, y, n_perm, seed=0):
    """Family-wise |r| threshold from shuffled labels. -> (thr95, null [B], n_imputed).

    Shuffling y across completions destroys any association while leaving the entire
    correlation structure of X intact, so the 95th percentile of max_k |r_k| is an
    exact 5% family-wise threshold whatever the columns' mutual dependence -- which is
    the whole problem with 1,152 neighbouring attention heads.
    """
    Xf, nimp = _impute(X)
    Xz = (Xf - Xf.mean(0)) / np.where(Xf.std(0) > 0, Xf.std(0), np.inf)
    Xz = np.ascontiguousarray(Xz, dtype=np.float32)
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=np.float64)
    chunk = max(1, min(n_perm, int(4e7 // max(Xz.shape[1], 1))))
    done = 0
    while done < n_perm:
        b = min(chunk, n_perm - done)
        Y = np.stack([rng.permutation(y) for _ in range(b)], axis=1)
        Y = (Y - Y.mean(0)) / np.where(Y.std(0) > 0, Y.std(0), np.inf)
        R = (Xz.T @ np.ascontiguousarray(Y, dtype=np.float32)) / n
        null[done:done + b] = np.abs(R).max(axis=0)
        done += b
    return float(np.percentile(null, 95)), null, nimp


def fw_p(null, obs):
    """Family-wise p for an observed max |r| against the permutation null."""
    return (1.0 + float((null >= abs(obs)).sum())) / (len(null) + 1.0)


# ---------------------------------------------------------------------------
# per-family blocks
# ---------------------------------------------------------------------------
def block_stats(fam, X, y, sel, tag, n_perm, seed, resid=False):
    """r(all/select/held-out), the covariate-partial r, and the family-wise threshold.

    X is [C, K] at completion level; `sel` is the odd-row_index half the columns are
    ranked on, its complement the half they are re-scored on.

    With `resid`, X and y are replaced by their residuals on the covariate design
    FIRST, so every number after that -- including the permutation threshold and the
    held-out split -- is a partial correlation. Without it the report can say a raw r
    clears its threshold and a partial r is smaller, but not whether the partial itself
    survives the 8,064 tests it was chosen from, which is the question that decides
    this experiment.
    """
    level = np.nanmean(X, axis=0)                 # the level is of the RAW column
    Z, _ = fam.covariates()
    if resid:
        Xi, _ = _impute(np.asarray(X, dtype=np.float64))
        ok = np.isfinite(y) & np.isfinite(Z).all(axis=1)
        Xr = np.full(Xi.shape, np.nan)
        yr = np.full(len(y), np.nan)
        Xr[ok] = residualize(Xi[ok], Z[ok])
        yr[ok] = residualize(np.asarray(y, dtype=np.float64)[ok, None], Z[ok])[:, 0]
        X, y = Xr, yr
    out = {
        "tag": tag,
        "r_all": col_corr(X, y),
        "r_sel": col_corr(X[sel], y[sel]),
        "r_held": col_corr(X[~sel], y[~sel]),
        "r_partial": (col_corr(X, y) if resid else partial_col_corr(X, y, Z)),
        "level": level,
    }
    thr, null, nimp = max_abs_null(X, y, n_perm, seed)
    out["thr95"], out["null"], out["n_imputed"] = thr, null, nimp
    return out


def ref_corr(fam, X, y, resid):
    """r of columns that are not part of a block, matched to the block's convention.

    The synthetic "mean over all heads" column and the corpus-by-corpus table are
    computed outside block_stats, so under --residualize they have to be residualised
    here too. Reporting one row raw next to a table of partials, with the same heading
    and no marking, is how a report ends up being read backwards.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if not resid:
        return X, y
    Z, _ = fam.covariates()
    Xi, _ = _impute(X if X.ndim == 2 else X[:, None])
    ok = np.isfinite(y) & np.isfinite(Z).all(axis=1)
    Xr = np.full(Xi.shape, np.nan)
    yr = np.full(len(y), np.nan)
    Xr[ok] = residualize(Xi[ok], Z[ok])
    yr[ok] = residualize(y[ok, None], Z[ok])[:, 0]
    return (Xr if X.ndim == 2 else Xr[:, 0]), yr


def by_dataset(fam, x, y, acc):
    """r(x, y) inside each source corpus. -> [(name, n, r, accuracy), ...].

    `acc` is the RAW label, so the accuracy column stays readable when x and y have
    been residualised (residualised labels average to zero by construction).

    The blunter, more readable form of the dataset dummies in `covariates()`: accuracy
    on this corpus runs from one source dataset to another by far more than any map
    statistic moves, so a correlation pooled across all six can be nothing but that
    difference. A column that holds its sign and size inside every dataset is measuring
    something about the map.
    """
    out = []
    for name in sorted(set(fam.c_dataset)):
        m = fam.c_dataset == name
        if m.sum() < 30:
            continue
        out.append((name, int(m.sum()), float(col_corr(x[m, None], y[m])[0]),
                    float(acc[m].mean())))
    return out


def best_of(st, cols, names, key="r_all"):
    """The single largest |r| in a block. -> dict."""
    r = np.asarray(st[key], dtype=np.float64)
    flat = r.reshape(-1)
    if not np.isfinite(flat).any():
        return None
    k = int(np.nanargmax(np.abs(flat)))
    if r.ndim == 2:                       # [K, M] -> (column, metric)
        ci, mi = divmod(k, r.shape[1])
        label = f"{cols[ci]}/{names[mi]}"
    else:
        ci, mi, label = k, None, cols[k]
    g = lambda a: float(np.asarray(a).reshape(-1)[k])          # noqa: E731
    return {"label": label, "col": ci, "metric": mi,
            "r_all": g(st["r_all"]), "r_sel": g(st["r_sel"]),
            "r_held": g(st["r_held"]), "r_partial": g(st["r_partial"]),
            "level": g(st["level"]),
            "thr95": st["thr95"], "p_fw": fw_p(st["null"], g(st["r_all"])),
            "n_tests": int(flat.size)}


def fmt_best(b):
    if b is None:
        return "        (nothing finite)"
    star = "*" if abs(b["r_all"]) >= b["thr95"] else " "
    return (f"{b['label']:>14}  r {b['r_all']:+7.4f}{star} held {b['r_held']:+7.4f} "
            f"part {b['r_partial']:+7.4f}  thr95 {b['thr95']:.4f}  "
            f"p_fw {b['p_fw']:.4f}  ({b['n_tests']} tests)")


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def run_family(fam, args):
    print("\n" + "=" * 78)
    print(f"=== {fam.name}   {fam.path}")
    print("=" * 78)
    C = len(fam.uniq)
    acc = fam.ccor.mean()
    print(f"steps {len(fam.row)}   completions {C}   columns {len(fam.cols)}   "
          f"accuracy {acc:.3f}")
    if fam.capped:
        print(f"--max-union {args.max_union}: {fam.capped[0]}/{fam.capped[1]} steps kept")
    print(f"single-test |r| at n={C}: {1.96 / np.sqrt(C - 3):.4f}")
    if fam.neg is not None:
        nf = np.nanmean(fam.neg, axis=0)
        bad = [(fam.cols[k], nf[k]) for k in np.argsort(-nf)[:3] if nf[k] > 0.01]
        if bad:
            print("rectified before scoring (share of |mass| that was negative): "
                  + ", ".join(f"{c} {v:.2f}" for c, v in bad))

    sel = (fam.uniq % 2 == 1)
    y = fam.ccor
    sharp_names = list(fam.sharp_names)

    rz = args.residualize
    if rz:
        print("\n[--residualize] every r below is a PARTIAL correlation: X and the "
              "label are\n  residualised on log(patches), step tokens, step count, "
              "union area and dataset\n  dummies before anything is computed, so the "
              "permutation thresholds and the\n  held-out split apply to the partial "
              "rather than to the raw r. `level` is still\n  the raw column's mean. "
              "The covariate fit uses all completions, so the held-out\n  half is held "
              "out of the correlation, not of the 9-parameter residualisation.")
    blocks = {}
    blocks["SHARP"] = block_stats(fam, fam.c_sharp.reshape(C, -1), y, sel, "SHARP",
                                  args.perm, args.seed, rz)
    for k in ("r_all", "r_sel", "r_held", "r_partial", "level"):
        blocks["SHARP"][k] = blocks["SHARP"][k].reshape(len(fam.cols), M)
    dino = np.concatenate([fam.c_v2, fam.c_au], axis=1)
    blocks["DINO"] = block_stats(fam, dino, y, sel, "DINO", args.perm, args.seed, rz)
    if fam.c_mass is not None:
        blocks["MASS"] = block_stats(fam, fam.c_mass, y, sel, "MASS", args.perm,
                                     args.seed, rz)

    dino_cols = [f"{c}/v2" for c in fam.cols] + [f"{c}/auroc" for c in fam.cols]
    best = {
        "SHARP": best_of(blocks["SHARP"], fam.cols, sharp_names),
        "DINO": best_of(blocks["DINO"], dino_cols, None),
    }
    if "MASS" in blocks:
        best["MASS"] = best_of(blocks["MASS"], fam.cols, None)

    print("\n--- best column in each block, and whether it clears its own "
          "permutation threshold ---")
    for k in ("DINO", "SHARP", "MASS"):
        if k in best:
            print(f"  {k:>5}  {fmt_best(best[k])}")
    print("  '*' = |r| at or above that block's max-|r| 95th percentile under shuffled "
          "labels.\n  thr95 differs between blocks because they run different numbers "
          "of correlated tests;\n  comparing raw |r| across blocks without it is the "
          "mistake this line exists to prevent.\n  These columns were chosen using ALL "
          "completions, so their `held` is not a clean confirmation.\n  The next table "
          "is.")

    # --- the honest split: choose on one half, score on the other
    n_even = int((~sel).sum())
    thr_1 = 1.96 / np.sqrt(max(n_even - 3, 1))
    print(f"\n--- ranked on the ODD half, re-scored on the EVEN half (n={n_even}) ---")
    print(f"   {'block':>5}  {'column':>16}  {'r(odd, chosen on)':>18}  "
          f"{'r(EVEN, held out)':>18}")
    honest = {}
    for k in ("DINO", "SHARP", "MASS"):
        if k not in blocks:
            continue
        cols_k = (fam.cols if k != "DINO" else dino_cols)
        b = best_of(blocks[k], cols_k, sharp_names if k == "SHARP" else None,
                    key="r_sel")
        if b is None:
            continue
        honest[k] = b
        mark = "*" if abs(b["r_held"]) >= thr_1 else " "
        print(f"   {k:>5}  {b['label']:>16}  {b['r_sel']:>+18.4f}  "
              f"{b['r_held']:>+17.4f}{mark}")
    print(f"   Only one column per block is scored on the even half, so no multiplicity "
          f"correction is\n   owed there: '*' is the plain single-test threshold "
          f"{thr_1:.4f}. This is the number to quote.")

    # --- every sharpness metric, on the family's un-selected reference columns
    prim = primary_columns(fam)
    print("\n--- sharpness by metric, completion level "
          "(reference columns, fixed before looking) ---")
    print(f"   {'column':>14}  " + "  ".join(f"{n:>7}" for n in sharp_names))
    for label, ci in prim:
        if ci is None:                          # the synthetic mean over all columns
            Xr, yr = ref_corr(fam, np.nanmean(fam.c_sharp, axis=1), y, rz)
            r = col_corr(Xr, yr)
        else:
            r = blocks["SHARP"]["r_all"][ci]
        print(f"   {label:>14}  " + "  ".join(f"{v:>+7.4f}" for v in r) + "   r")
    for label, ci in prim:
        lev = (np.nanmean(np.nanmean(fam.c_sharp, axis=1), axis=0) if ci is None
               else blocks["SHARP"]["level"][ci])
        print(f"   {label:>14}  " + "  ".join(f"{v:>7.3f}" for v in lev) + "   level")
    print("   level is the column's mean value, 0 = as flat as the grid allows and "
          "1 = a delta\n   (cv is unbounded above; sconc can go negative for mass in "
          "two distant clumps).")

    # --- the same columns' DINO scores, for the side-by-side the whole report is for
    print(f"\n--- the same reference columns under the DINO metrics ---")
    print(f"   {'column':>14}  {'v2 r':>9} {'v2 level':>9}  {'auroc r':>9} "
          f"{'auroc level':>11}")
    for label, ci in prim:
        if ci is None:
            Xr, yr = ref_corr(fam, np.column_stack(
                [np.nanmean(fam.c_v2, axis=1), np.nanmean(fam.c_au, axis=1)]), y, rz)
            rv, ra = col_corr(Xr, yr)
            lv = np.nanmean(fam.c_v2)
            la = np.nanmean(fam.c_au)
        else:
            rv, ra = blocks["DINO"]["r_all"][ci], blocks["DINO"]["r_all"][ci + len(fam.cols)]
            lv, la = blocks["DINO"]["level"][ci], blocks["DINO"]["level"][ci + len(fam.cols)]
        print(f"   {label:>14}  {rv:>+9.4f} {lv:>9.3f}  {ra:>+9.4f} {la:>11.3f}")
    print("   chance is 1.0 for mean_in_v2 and 0.5 for auroc.")

    # --- confound audit: what else does each sharpness metric track?
    print("\n--- what the sharpness columns are correlated with, besides correctness "
          "(mean |r| over columns) ---")
    Z, znames = fam.covariates()
    print(f"   {'metric':>7}  " + "  ".join(f"{n[:9]:>9}" for n in znames[:4])
          + f"  {'auroc':>9}  {'mass':>9}")
    for mi, nm in enumerate(sharp_names):
        Xm = fam.c_sharp[:, :, mi]
        with np.errstate(invalid="ignore"):
            cells = [np.nanmean(np.abs(col_corr(Xm, Z[:, j]))) for j in range(4)]
            ra = np.nanmean(np.abs(paired_corr(Xm, fam.c_au)))
            rm = (np.nanmean(np.abs(paired_corr(Xm, fam.c_mass)))
                  if fam.c_mass is not None else np.nan)
        print(f"   {nm:>7}  " + "  ".join(f"{c:>9.3f}" for c in cells)
              + f"  {ra:>9.3f}  {rm:>9.3f}")
    print("   These are |r| against the covariate, not against correctness. A metric "
          "that tracks\n   log_npatch or ntok hard is partly a readout of image size "
          "or step length, which is\n   why r(PARTIAL) above holds all of them, plus "
          "dataset dummies, fixed.")

    # --- the horse race, on the family's best sharpness metric
    hb = best["SHARP"]
    if hb is not None and hb["metric"] is not None:
        mi = hb["metric"]
        Xm = fam.c_sharp[:, :, mi]
        r_sharp_given_dino = pairwise_partial(Xm, y, fam.c_au)
        r_dino_given_sharp = pairwise_partial(fam.c_au, y, Xm)
        ci = hb["col"]
        print(f"\n--- horse race on {sharp_names[mi]} vs auroc, same column, same "
              f"completions ---")
        print(f"   {'column':>14}  {'r(sharp)':>9} {'| auroc':>9}   {'r(auroc)':>9} "
              f"{'| sharp':>9}")
        for label, cc in prim + [(f"BEST {fam.cols[ci]}", ci)]:
            if cc is None:
                continue
            print(f"   {label:>14}  {blocks['SHARP']['r_all'][cc, mi]:>+9.4f} "
                  f"{r_sharp_given_dino[cc]:>+9.4f}   "
                  f"{blocks['DINO']['r_all'][cc + len(fam.cols)]:>+9.4f} "
                  f"{r_dino_given_sharp[cc]:>+9.4f}")
        print("   '| x' is the partial holding x fixed. If sharpness survives its "
              "partial and\n   grounding does not, the correlation was never about "
              "location.")

    # --- the best columns, corpus by corpus
    print("\n--- the winning columns inside each source corpus ---")
    picks = []
    if best["SHARP"] is not None and best["SHARP"]["metric"] is not None:
        picks.append((f"SHARP {best['SHARP']['label']}",
                      fam.c_sharp[:, best["SHARP"]["col"], best["SHARP"]["metric"]]))
    if best["DINO"] is not None:
        bi = best["DINO"]["col"]
        picks.append((f"DINO {best['DINO']['label']}",
                      (fam.c_v2 if bi < len(fam.cols) else fam.c_au)
                      [:, bi % len(fam.cols)]))
    rows_ds = []
    for _, x in picks:
        xr, yr = ref_corr(fam, x, y, rz)
        rows_ds.append(by_dataset(fam, xr, yr, y))
    if rows_ds and rows_ds[0]:
        names_ds = [d[0] for d in rows_ds[0]]
        print(f"   {'':>22}" + "".join(f"{n[:9]:>10}" for n in names_ds))
        print(f"   {'n completions':>22}"
              + "".join(f"{d[1]:>10d}" for d in rows_ds[0]))
        print(f"   {'accuracy':>22}" + "".join(f"{d[3]:>10.3f}" for d in rows_ds[0]))
        for (label, _), ds in zip(picks, rows_ds):
            print(f"   {label:>22}" + "".join(f"{d[2]:>+10.3f}" for d in ds))
        print("   Accuracy varies more across these corpora than any map statistic "
              "varies within one,\n   so a column that changes sign between them is "
              "reading the corpus, not the map.")

    # --- step level, for continuity with the earlier reports only
    s_sharp = np.stack([col_corr(fam.sharp[:, :, m], fam.correct)
                        for m in range(M)], axis=1)          # [K, M], one metric at a
    s_au = col_corr(fam.auroc, fam.correct)                  # time to bound the temps
    k1 = int(np.nanargmax(np.abs(s_sharp)))
    print(f"\n--- step level (n={len(fam.row)}), anti-conservative, not a finding ---")
    print(f"   best sharpness {fam.cols[k1 // M]}/{sharp_names[k1 % M]} "
          f"r {s_sharp.reshape(-1)[k1]:+.4f}   "
          f"best auroc {fam.cols[int(np.nanargmax(np.abs(s_au)))]} "
          f"r {s_au[int(np.nanargmax(np.abs(s_au)))]:+.4f}")

    return {"name": fam.name, "path": str(fam.path), "n_completions": C,
            "n_steps": int(len(fam.row)), "accuracy": float(acc),
            "sharp_names": sharp_names,
            "best": {k: {kk: vv for kk, vv in v.items() if kk != "null"}
                     for k, v in best.items() if v is not None},
            "honest_split": {k: {"label": v["label"], "r_sel": v["r_sel"],
                                 "r_held": v["r_held"], "thr_single": thr_1}
                             for k, v in honest.items()},
            "reference": {
                label: {"r_sharp": blocks["SHARP"]["r_all"][ci].tolist(),
                        "level_sharp": blocks["SHARP"]["level"][ci].tolist(),
                        "r_v2": float(blocks["DINO"]["r_all"][ci]),
                        "r_auroc": float(blocks["DINO"]["r_all"][ci + len(fam.cols)])}
                for label, ci in prim if ci is not None},
            }, blocks, best


def primary_columns(fam):
    """Reference columns fixed by the earlier reports, not chosen from this one.

    Selecting a column on the same data that scores it is the failure mode the odd/even
    split exists to catch, so every family also gets columns nobody picked: the mean
    over all of a head scan's 1,152 heads, the two heads the reward actually trained
    on, and each flow map's pre-registered readout.
    """
    if fam.cols and fam.cols[0].startswith("L") and "H" in fam.cols[0]:
        out = [("mean all heads", None)]
        for want in ("L22H28", "L22H31", "L1H4"):     # the two rewarded, and result 2's
            if want in fam.cols:                      # one surviving positive
                out.append((want, fam.cols.index(want)))
        return out
    base = [i for i, n in enumerate(fam.cols) if not n.startswith("inc")]
    inc = [i for i, n in enumerate(fam.cols) if n.startswith("inc")]
    out = ([(fam.cols[i], i) for i in base] if len(base) <= 6
           else [(fam.cols[base[-1]], base[-1])])      # a rollout reads at the last layer
    if inc:
        out.append((fam.cols[inc[-1]], inc[-1]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", action="append", default=[], metavar="NAME=DIR",
                    help="a scan directory to score; repeatable")
    ap.add_argument("--max-union", type=float, default=0.0,
                    help="drop steps whose DINO union exceeds this (0 = off)")
    ap.add_argument("--perm", type=int, default=2000,
                    help="label shuffles for the max-|r| family-wise threshold")
    ap.add_argument("--residualize", action="store_true",
                    help="run everything on covariate residuals, so the permutation "
                         "threshold and the held-out split apply to the PARTIAL r")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default="", help="also write every number here")
    args = ap.parse_args()
    if not args.scan:
        raise SystemExit("--scan NAME=DIR is required (repeatable)")

    fams, summary = [], []
    for spec in args.scan:
        if "=" not in spec:
            raise SystemExit(f"--scan wants NAME=DIR, got {spec!r}")
        name, path = spec.split("=", 1)
        fams.append(Family(name, path, args.max_union))

    rows = []
    for fam in fams:
        js, blocks, best = run_family(fam, args)
        summary.append(js)
        rows.append((fam.name, len(fam.uniq), best, js["honest_split"]))

    print("\n" + "=" * 78)
    print("=== HEADLINE: the best column of each block, per family "
          "(completion level) ===")
    print("=" * 78)
    print(f"{'family':>14} {'n':>5}  {'block':>5}  {'best column':>16} "
          f"{'r':>8} {'thr95':>7} {'p_fw':>7} {'tests':>6}   "
          f"{'odd-half pick':>16} {'r(EVEN)':>8}")
    for name, n, best, hon in rows:
        for k in ("DINO", "SHARP", "MASS"):
            b = best.get(k)
            if b is None:
                continue
            star = "*" if abs(b["r_all"]) >= b["thr95"] else " "
            h = hon.get(k, {})
            hstar = ("*" if h and abs(h["r_held"]) >= h["thr_single"] else " ")
            print(f"{name:>14} {n:>5}  {k:>5}  {b['label']:>16} "
                  f"{b['r_all']:>+8.4f}{star}{b['thr95']:>6.4f} {b['p_fw']:>7.4f} "
                  f"{b['n_tests']:>6}   {h.get('label', ''):>16} "
                  f"{h.get('r_held', float('nan')):>+8.4f}{hstar}")
    print("\nDINO  = the box metrics (mean_in_v2, auroc): does the map sit on the "
          "objects the step names.")
    print("SHARP = the box-free concentration metrics: how peaked the map is, six of "
          "the seven\n        provably blind to WHERE the peak is.")
    print("MASS  = the map's total weight on the image: magnitude, not shape.")
    print("A SHARP entry that clears its threshold while the DINO entry does not is "
          "the result\nthe hypothesis predicts: the model does better when it focuses "
          "on anything.")

    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=1, default=_j))
        print(f"\n-> {args.json}")


def _j(o):
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


if __name__ == "__main__":
    main()
