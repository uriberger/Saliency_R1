#!/usr/bin/env python
"""Do the reasoning steps of one chain get DIFFERENT boxes out of DINO?

The overlap reward runs DINO once per observe step, on that step's own sentence, and
scores the step's attention map against the boxes that come back. That only makes sense
if the boxes actually change from step to step. This script tests whether they do,
entirely from data already on disk -- `overlap_probe.py --store-maps` wrote, for every
step of every completion it generated, the DINO boxes and the rasterised union mask that
the reward scored (`boxes_kept`, `mask_q`), plus the attention map itself (`map_q`).

Three questions, in order:

  1. HOW SIMILAR ARE TWO STEPS' MASKS?  Overlap between the union masks of two steps of
     the same completion, measured as intersection over union ("IoU": shared patches
     divided by patches in either one; 1.0 means identical, 0.0 means disjoint).

     Raw IoU is not enough. These unions are huge -- the median step covers over half the
     patch grid -- and two huge blobs overlap a lot no matter what they contain. So every
     IoU is reported next to two reference points:

       chance IoU  -- what two INDEPENDENT masks of exactly these two sizes would score.
       best IoU    -- what these two sizes score when the smaller sits entirely inside
                      the larger, i.e. as identical as two masks of these sizes can be.

     and then as one number on a 0-1 scale between them:

       closeness = (observed IoU - chance IoU) / (best IoU - chance IoU)

     0.0 means the two steps' boxes are unrelated once you know how big they are, 1.0
     means they are as identical as their sizes permit.

  2. IS THAT JUST THE IMAGE?  The same measurement on two control pairings:
       same image, different completion -- two steps written by two different rollouts of
         the same prompt, so different sentences about the same picture.
       different image -- a step's boxes rasterised onto another image's patch grid.
         This is the floor: it says what the numbers look like when there is genuinely
         no shared content.
     If within-chain closeness is no higher than same-image-different-chain closeness,
     the mask is a property of the picture, not of the sentence.

  3. DOES IT CHANGE THE REWARD?  For every step, rescore its attention map against every
     OTHER step's mask, and rebuild the per-completion overlap reward as it would have
     been if DINO had been run once per completion instead of once per step. GRPO only
     ever sees the within-group ranking of that reward (8 completions of one prompt), so
     the test is whether the ranking survives.

Usage:

    python step_box_similarity.py outputs/overlap_probe/*/probe_merged.json \
        --out-dir outputs/step_box_similarity

Everything is CPU and takes a couple of minutes. No GPU, no model, no DINO call.
"""

from __future__ import annotations

import argparse
import base64
import itertools
import json
import os
import sys
from collections import defaultdict

import numpy as np

# Reproducible everywhere a control pairing is drawn.
SEED = 20260903


# ---------------------------------------------------------------------------
# decoding what overlap_probe stored
# ---------------------------------------------------------------------------
def decode_mask(b64: str, gh: int, gw: int) -> np.ndarray:
    """The rasterised DINO union, as overlap_rewards._union_mask returned it."""
    a = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
    if a.size != gh * gw:
        raise ValueError(f"mask has {a.size} bytes, grid is {gh}x{gw}")
    return a.reshape(gh, gw).astype(bool)


def decode_map(b64: str, gh: int, gw: int) -> np.ndarray:
    """The step's saliency map, peak-normalised to [0,1].

    overlap_probe quantised it by its own peak, so q/255 is exactly the map that
    `_mean_in` divides by its max -- the metric is unchanged by the quantisation apart
    from the 1/255 rounding, which `--verify` checks against the stored values.
    """
    a = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
    if a.size != gh * gw:
        raise ValueError(f"map has {a.size} bytes, grid is {gh}x{gw}")
    return a.reshape(gh, gw).astype(np.float64) / 255.0


def rasterise(boxes, gh: int, gw: int) -> np.ndarray | None:
    """Union of already-area-filtered boxes on a (gh, gw) grid.

    Byte-for-byte the loop in overlap_rewards._union_mask, minus the two caps -- the
    caller passes `boxes_kept`, which the probe already filtered by max_box_area, and the
    union cap was off in every run analysed here. Needed only for the different-image
    control, where a step's boxes have to land on a grid they were not drawn for.
    """
    if not boxes:
        return None
    mask = np.zeros((gh, gw), dtype=bool)
    for x1, y1, x2, y2 in boxes:
        r0 = max(0, int(y1 * gh))
        r1 = min(gh, max(r0 + 1, round(y2 * gh)))
        c0 = max(0, int(x1 * gw))
        c1 = min(gw, max(c0 + 1, round(x2 * gw)))
        mask[r0:r1, c0:c1] = True
    n_in = int(mask.sum())
    if n_in == 0 or n_in == gh * gw:
        return None
    return mask


# ---------------------------------------------------------------------------
# the similarity measures
# ---------------------------------------------------------------------------
def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return inter / union if union > 0 else float("nan")


def iou_chance(na: int, nb: int, n: int) -> float:
    """IoU of two INDEPENDENT masks covering na and nb of n patches.

    Expected intersection is na*nb/n, expected union is na+nb minus that. Using the ratio
    of expectations rather than the expectation of the ratio: on a 160-patch grid with
    masks this large the two agree to well under a percent, and the closed form keeps the
    reference point exact and free.
    """
    if n <= 0:
        return float("nan")
    inter = na * nb / n
    union = na + nb - inter
    return inter / union if union > 0 else float("nan")


def iou_best(na: int, nb: int) -> float:
    """IoU when the smaller mask sits entirely inside the larger one."""
    hi = max(na, nb)
    return (min(na, nb) / hi) if hi > 0 else float("nan")


def closeness(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, float]:
    """(observed IoU, chance IoU, best IoU, closeness on the 0-1 scale between them)."""
    na, nb, n = int(a.sum()), int(b.sum()), a.size
    o, c, m = iou(a, b), iou_chance(na, nb, n), iou_best(na, nb)
    span = m - c
    z = (o - c) / span if span > 1e-12 else float("nan")
    return o, c, m, z


def box_iou(p, q) -> float:
    ix = max(0.0, min(p[2], q[2]) - max(p[0], q[0]))
    iy = max(0.0, min(p[3], q[3]) - max(p[1], q[1]))
    inter = ix * iy
    ap = max(0.0, p[2] - p[0]) * max(0.0, p[3] - p[1])
    aq = max(0.0, q[2] - q[0]) * max(0.0, q[3] - q[1])
    den = ap + aq - inter
    return inter / den if den > 0 else 0.0


STOP = frozenset("""
a an the is are was were be been being this that these those there here it its it's of in
on at to for with from by as and or but not no yes so then than which what who whom whose
where when why how i we you he she they them us our your their his her me my mine also
can could may might must shall should will would do does did done have has had if while
about into over under between among across up down out off very more most much many some
any all both each other another such same own just only even still yet also because since
""".split())


def content_words(text: str) -> set[str]:
    """Lower-cased alphabetic words minus a small stop list.

    Crude on purpose. It only has to separate `two steps that talk about the same things`
    from `two steps that talk about different things`; DINO is being fed the whole
    sentence, so the exact tokenisation is not the interesting variable.
    """
    w = "".join(ch if ch.isalpha() or ch.isspace() else " " for ch in text.lower()).split()
    return {t for t in w if len(t) > 2 and t not in STOP}


def text_overlap(a: str, b: str) -> float:
    """Share of content words the two sentences have in common (0 = none, 1 = the same)."""
    A, B = content_words(a), content_words(b)
    if not A or not B:
        return float("nan")
    return len(A & B) / len(A | B)


def box_set_match(A, B, thr: float = 0.9) -> tuple[float, float]:
    """How well two BOX LISTS line up, before rasterisation blurs them together.

    Returns (mean best-match IoU, fraction of boxes with a >=thr partner), each
    symmetrised over the two directions so neither list is privileged. The union mask
    can look identical while the underlying boxes differ, so this is the finer check.
    """
    if not A or not B:
        return float("nan"), float("nan")
    M = np.array([[box_iou(p, q) for q in B] for p in A], dtype=np.float64)
    best_ab, best_ba = M.max(axis=1), M.max(axis=0)
    allbest = np.concatenate([best_ab, best_ba])
    return float(allbest.mean()), float((allbest >= thr).mean())


# ---------------------------------------------------------------------------
# the metrics the reward actually uses (copies, so this script needs no GPU env)
# ---------------------------------------------------------------------------
def m_mean_in(smap, mask):
    vmax = float(smap.max())
    m = smap / vmax if vmax > 0 else smap
    inside = m[mask]
    return float(inside.mean()) if inside.size else None


def m_mean_in_v2(smap, mask):
    inside = smap[mask]
    if inside.size == 0:
        return None
    den = float(smap.mean())
    return float(inside.mean()) / den if den > 0 else None


def m_auroc(smap, mask):
    v = np.asarray(smap, dtype=np.float64).ravel()
    m = np.asarray(mask, dtype=bool).ravel()
    n_in = int(m.sum())
    n_out = v.size - n_in
    if n_in == 0 or n_out == 0:
        return None
    order = np.argsort(v, kind="stable")
    ranks = np.empty(v.size, dtype=np.float64)
    ranks[order] = np.arange(1, v.size + 1, dtype=np.float64)
    _u, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
    sums = np.zeros(cnt.size, dtype=np.float64)
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    u = ranks[m].sum() - n_in * (n_in + 1) / 2.0
    return float(u / (n_in * n_out))


METRICS = {"mean_in": m_mean_in, "mean_in_v2": m_mean_in_v2, "auroc": m_auroc}


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
class Step:
    __slots__ = ("si", "text", "gh", "gw", "mask", "smap", "boxes", "grounded",
                 "stored", "union_frac")

    def __init__(self, rec):
        self.si = rec["step_index"]
        self.text = rec.get("text", "")
        self.gh, self.gw = rec["grid"]
        self.mask = decode_mask(rec["mask_q"], self.gh, self.gw)
        self.smap = decode_map(rec["map_q"], self.gh, self.gw) if rec.get("map_q") else None
        self.boxes = rec.get("boxes_kept") or []
        self.grounded = bool(rec.get("grounded"))
        self.stored = {k: rec.get(k + "_raw") for k in METRICS}
        self.union_frac = float(self.mask.sum()) / self.mask.size


def load_model(model_rec, min_steps=2):
    """-> list of samples; each sample is (question, [completion, ...]).

    A completion is a list of Step. Steps without a stored mask are dropped: DINO
    grounded nothing there, or the union swallowed the whole grid, and neither was scored
    by the reward either. Dropping the whole-grid ones works AGAINST the hypothesis
    being tested -- they are the most duplicated masks there are.
    """
    out = []
    for s in model_rec["samples"]:
        comps = []
        for c in s["completions"]:
            steps = []
            for rec in c.get("observe_steps", []):
                if not rec.get("mask_q") or not rec.get("grid"):
                    continue
                try:
                    steps.append(Step(rec))
                except ValueError:
                    continue
            if len(steps) >= min_steps:
                comps.append(steps)
        if comps:
            out.append({
                "question": s.get("question", ""),
                "image": s.get("image_file", ""),
                "sample_index": s.get("sample_index"),
                "comps": comps,
            })
    return out


# ---------------------------------------------------------------------------
# question 1 + 2: how similar are the masks, within a chain and in the controls
# ---------------------------------------------------------------------------
def pair_stats(a: Step, b: Step, want_boxes=True):
    o, c, m, z = closeness(a.mask, b.mask)
    d = {"iou": o, "iou_chance": c, "iou_best": m, "closeness": z,
         "identical": bool(np.array_equal(a.mask, b.mask)),
         "text_overlap": text_overlap(a.text, b.text),
         "frac_a": a.union_frac, "frac_b": b.union_frac}
    if want_boxes:
        bm, bh = box_set_match(a.boxes, b.boxes)
        d["box_match"] = bm
        d["box_hit"] = bh
        d["n_boxes_a"] = len(a.boxes)
        d["n_boxes_b"] = len(b.boxes)
    return d


def similarity_pass(samples, rng, max_steps_per_comp=40):
    """Every within-chain pair, plus size-matched control pairs.

    Controls are drawn to the same count as the within-chain pairs of the same completion,
    so the three populations are weighted by the same completions and a chain with many
    steps does not dominate one comparison and not another.
    """
    within, cross_comp, cross_img = [], [], []
    per_comp = []

    # flat index of every step, for the different-image control
    flat = [(si, ci, k) for si, s in enumerate(samples)
            for ci, c in enumerate(s["comps"]) for k in range(len(c))]

    for si, s in enumerate(samples):
        comps = s["comps"]
        for ci, comp in enumerate(comps):
            steps = comp[:max_steps_per_comp]
            pairs = list(itertools.combinations(range(len(steps)), 2))
            if not pairs:
                continue
            rows = [pair_stats(steps[i], steps[j]) for i, j in pairs]
            within.extend(rows)

            zs = [r["closeness"] for r in rows if np.isfinite(r["closeness"])]
            per_comp.append({
                "sample_index": si,
                "n_steps": len(steps),
                "mean_iou": float(np.mean([r["iou"] for r in rows])),
                "min_iou": float(np.min([r["iou"] for r in rows])),
                "mean_closeness": float(np.mean(zs)) if zs else float("nan"),
                "all_identical": all(r["identical"] for r in rows),
                "mean_union_frac": float(np.mean([st.union_frac for st in steps])),
            })

            # control A: same image, different completion
            others = [c for cj, c in enumerate(comps) if cj != ci]
            if others:
                for _ in pairs:
                    a = steps[rng.integers(len(steps))]
                    oc = others[rng.integers(len(others))]
                    b = oc[rng.integers(len(oc))]
                    if a.mask.shape == b.mask.shape:
                        cross_comp.append(pair_stats(a, b))

            # control B: different image -- the other step's BOXES, re-rasterised onto
            # this image's grid, because the two grids are generally different sizes.
            for _ in pairs:
                a = steps[rng.integers(len(steps))]
                for _try in range(8):
                    tsi, tci, tk = flat[rng.integers(len(flat))]
                    if tsi == si:
                        continue
                    other = samples[tsi]["comps"][tci][tk]
                    bm = rasterise(other.boxes, a.gh, a.gw)
                    if bm is None:
                        continue
                    o, c, m, z = closeness(a.mask, bm)
                    bmatch, bhit = box_set_match(a.boxes, other.boxes)
                    cross_img.append({"iou": o, "iou_chance": c, "iou_best": m,
                                      "closeness": z,
                                      "identical": bool(np.array_equal(a.mask, bm)),
                                      "box_match": bmatch, "box_hit": bhit})
                    break
    return within, cross_comp, cross_img, per_comp


# ---------------------------------------------------------------------------
# question 3: does the reward notice?
# ---------------------------------------------------------------------------
def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    x, y = x[ok], y[ok]

    def rank(v):
        o = np.argsort(v, kind="stable")
        r = np.empty(v.size)
        r[o] = np.arange(v.size, dtype=float)
        _u, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
        sums = np.zeros(cnt.size)
        np.add.at(sums, inv, r)
        return (sums / cnt)[inv]

    rx, ry = rank(x), rank(y)
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def swap_pass(samples, rng, metric_name, max_steps_per_comp=25):
    """Rescore each step against other steps' masks, and rebuild the chain reward.

    Per step:
      own_rank  -- where the step's OWN mask ranks among the other steps' masks, scaled
                   so 1.0 = own mask scores highest, 0.5 = chance. This is the plan's
                   `ID accuracy`, recomputed here on the same maps, and it is the check
                   that this script agrees with what is already known.
      swap_gap  -- score on the own mask minus the mean score on the other masks, in
                   units of the spread of scores ACROSS steps of the chain. Small means
                   which mask you use barely moves the number.

    Per completion, three versions of the reward the trainer would have seen:
      true   -- each step on its own mask (what actually happened)
      first  -- every step on the chain's FIRST step's mask (DINO run once per chain)
      other  -- every step on a mask taken from a DIFFERENT completion of the same image
                (DINO run once per image, on somebody else's sentence)
    and the within-group rank correlation of each against `true`. That correlation is the
    quantity that matters: GRPO subtracts the group mean, so only the ranking survives.
    """
    fn = METRICS[metric_name]
    steps_rows, comp_rows, group_rows = [], [], []

    for s in samples:
        comps = s["comps"]
        true_r, first_r, other_r = [], [], []
        for ci, comp in enumerate(comps):
            steps = [st for st in comp if st.smap is not None][:max_steps_per_comp]
            if len(steps) < 2:
                continue
            n = len(steps)
            S = np.full((n, n), np.nan)
            for i, a in enumerate(steps):
                for j, b in enumerate(steps):
                    v = fn(a.smap, b.mask)
                    if v is not None:
                        S[i, j] = v
            diag = np.diag(S).copy()
            spread = float(np.nanstd(diag))
            for i in range(n):
                off = np.delete(S[i], i)
                off = off[np.isfinite(off)]
                if off.size == 0 or not np.isfinite(diag[i]):
                    continue
                better = float((off < diag[i]).sum() + 0.5 * (off == diag[i]).sum())
                steps_rows.append({
                    "own_rank": better / off.size,
                    "own": float(diag[i]),
                    "off_mean": float(off.mean()),
                    "swap_gap": (float(diag[i] - off.mean()) / spread) if spread > 1e-12 else float("nan"),
                })

            other_comps = [c for cj, c in enumerate(comps) if cj != ci]
            om = None
            if other_comps:
                oc = other_comps[rng.integers(len(other_comps))]
                cand = [st for st in oc if st.mask.shape == steps[0].mask.shape]
                if cand:
                    om = cand[0].mask

            v_true = np.nanmean(diag)
            v_first = np.nanmean(S[:, 0])
            v_other = float("nan")
            if om is not None:
                vals = [fn(a.smap, om) for a in steps]
                vals = [v for v in vals if v is not None]
                if vals:
                    v_other = float(np.mean(vals))

            comp_rows.append({"true": float(v_true), "first": float(v_first),
                              "other": v_other, "n_steps": n})
            true_r.append(float(v_true))
            first_r.append(float(v_first))
            other_r.append(v_other)

        if len(true_r) >= 3:
            group_rows.append({
                "n_comps": len(true_r),
                "rho_first": spearman(true_r, first_r),
                "rho_other": spearman(true_r, other_r),
                "top1_first": bool(np.argmax(true_r) == np.argmax(np.nan_to_num(first_r, nan=-1e18))),
                "top1_other": bool(np.argmax(true_r) == np.argmax(np.nan_to_num(other_r, nan=-1e18))),
                "spread_true": float(np.std(true_r)),
            })
    return steps_rows, comp_rows, group_rows


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def boot_ci(values, groups, rng, n=1000):
    """95% interval for a mean, resampling IMAGES not chains.

    Chains of one image are anything but independent -- same picture, same question, and
    the reward's own group -- so an interval that treated them as separate observations
    would be far too narrow. `groups` gives the image index of each value.
    """
    v = np.asarray(values, float)
    g = np.asarray(groups)
    ok = np.isfinite(v)
    v, g = v[ok], g[ok]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    uniq = np.unique(g)
    per_image = [v[g == u] for u in uniq]
    means = np.empty(n)
    for b in range(n):
        pick = rng.integers(0, len(per_image), len(per_image))
        means[b] = np.concatenate([per_image[i] for i in pick]).mean()
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def q(v, ps=(10, 25, 50, 75, 90)):
    a = np.asarray([x for x in v if np.isfinite(x)], float)
    if a.size == 0:
        return {f"p{p}": float("nan") for p in ps}
    return {f"p{p}": float(np.percentile(a, p)) for p in ps}


def fmt(x, nd=3):
    return "  n/a" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def analyse_model(name, samples, rng, metric_name, verify=False):
    within, cross_comp, cross_img, per_comp = similarity_pass(samples, rng)
    res = {"model": name,
           "n_images": len(samples),
           "n_completions": sum(len(s["comps"]) for s in samples),
           "n_steps": sum(len(c) for s in samples for c in s["comps"]),
           "n_within_pairs": len(within)}

    if not within:
        return res, None

    gi = [p["sample_index"] for p in per_comp]

    def col(rows, k):
        return [r.get(k, float("nan")) for r in rows]

    res["union_frac"] = q(col(within, "frac_a") + col(within, "frac_b"))
    res["n_boxes"] = q(col(within, "n_boxes_a") + col(within, "n_boxes_b"))

    for tag, rows in (("within", within), ("cross_comp", cross_comp), ("cross_img", cross_img)):
        if not rows:
            res[tag] = None
            continue
        ious = col(rows, "iou")
        zs = col(rows, "closeness")
        res[tag] = {
            "n_pairs": len(rows),
            "iou": q(ious),
            "iou_mean": float(np.nanmean(ious)),
            "iou_chance_mean": float(np.nanmean(col(rows, "iou_chance"))),
            "iou_best_mean": float(np.nanmean(col(rows, "iou_best"))),
            "closeness": q(zs),
            "closeness_mean": float(np.nanmean(zs)),
            "frac_iou_ge_90": float(np.mean([x >= 0.90 for x in ious if np.isfinite(x)])),
            "frac_iou_ge_75": float(np.mean([x >= 0.75 for x in ious if np.isfinite(x)])),
            "frac_iou_ge_50": float(np.mean([x >= 0.50 for x in ious if np.isfinite(x)])),
            "frac_identical": float(np.mean(col(rows, "identical"))),
            "box_match_mean": float(np.nanmean(col(rows, "box_match"))),
            "box_hit_mean": float(np.nanmean(col(rows, "box_hit"))),
        }

    # Does DINO react to the sentence at all? Split the within-chain pairs by how many
    # content words the two steps share and look at the mask overlap in each bin. If the
    # boxes were driven by the words, the bin with no shared words would sit well below
    # the bin that repeats itself.
    tov = np.asarray(col(within, "text_overlap"), float)
    ious = np.asarray(col(within, "iou"), float)
    zsw = np.asarray(col(within, "closeness"), float)
    ok = np.isfinite(tov) & np.isfinite(ious)
    bins = []
    if ok.sum() >= 20:
        edges = [(-0.001, 0.0), (0.0, 0.10), (0.10, 0.25), (0.25, 1.01)]
        labels = ["no shared word", "0-10%", "10-25%", ">25%"]
        for (lo, hi), lab in zip(edges, labels):
            sel = ok & (tov > lo) & (tov <= hi)
            if sel.sum() < 5:
                continue
            bins.append({"label": lab, "n": int(sel.sum()),
                         "iou": float(np.nanmean(ious[sel])),
                         "closeness": float(np.nanmean(zsw[sel]))})
        res["text_corr_iou"] = float(np.corrcoef(tov[ok], ious[ok])[0, 1])
        res["text_overlap_median"] = float(np.median(tov[ok]))
    res["text_bins"] = bins

    res["per_completion"] = {
        "n": len(per_comp),
        "mean_iou": q(col(per_comp, "mean_iou")),
        "min_iou": q(col(per_comp, "min_iou")),
        "frac_all_identical": float(np.mean(col(per_comp, "all_identical"))),
        "frac_min_iou_ge_90": float(np.mean([x >= 0.90 for x in col(per_comp, "min_iou")])),
        "frac_mean_iou_ge_90": float(np.mean([x >= 0.90 for x in col(per_comp, "mean_iou")])),
        "frac_mean_iou_ge_75": float(np.mean([x >= 0.75 for x in col(per_comp, "mean_iou")])),
        "n_steps": q(col(per_comp, "n_steps")),
    }
    m, lo, hi = boot_ci(col(per_comp, "mean_iou"), gi, rng)
    res["per_completion"]["mean_iou_ci"] = [m, lo, hi]
    m, lo, hi = boot_ci(col(per_comp, "mean_closeness"), gi, rng)
    res["per_completion"]["mean_closeness_ci"] = [m, lo, hi]

    steps_rows, comp_rows, group_rows = swap_pass(samples, rng, metric_name)
    res["swap"] = {"metric": metric_name, "n_steps": len(steps_rows), "n_groups": len(group_rows)}
    if steps_rows:
        res["swap"].update({
            "own_rank_mean": float(np.nanmean([r["own_rank"] for r in steps_rows])),
            "own_rank": q([r["own_rank"] for r in steps_rows]),
            "swap_gap_mean": float(np.nanmean([r["swap_gap"] for r in steps_rows])),
        })
    if group_rows:
        res["swap"].update({
            "rho_first_mean": float(np.nanmean([r["rho_first"] for r in group_rows])),
            "rho_other_mean": float(np.nanmean([r["rho_other"] for r in group_rows])),
            "top1_first": float(np.mean([r["top1_first"] for r in group_rows])),
            "top1_other": float(np.mean([r["top1_other"] for r in group_rows])),
        })

    check = None
    if verify:
        errs = {k: [] for k in METRICS}
        for s in samples:
            for c in s["comps"]:
                for st in c:
                    if st.smap is None:
                        continue
                    for k, fn in METRICS.items():
                        got, want = fn(st.smap, st.mask), st.stored.get(k)
                        if got is not None and want is not None:
                            errs[k].append(abs(got - want))
        check = {k: (float(np.max(v)) if v else float("nan")) for k, v in errs.items()}
        res["verify_max_abs_err"] = check
    return res, check


def render(res_list, metric_name) -> str:
    L = []
    P = L.append
    P("=" * 96)
    P("DO THE STEPS OF ONE REASONING CHAIN GET DIFFERENT BOXES OUT OF DINO?")
    P("=" * 96)
    P("")
    P("All numbers are computed from boxes and masks already stored by overlap_probe.")
    P("")
    P("  IoU        overlap of two steps' union masks: shared patches / patches in either.")
    P("  chance     the IoU two INDEPENDENT masks of the same two sizes would get.")
    P("  best       the IoU those two sizes get when one sits entirely inside the other.")
    P("  closeness  (IoU - chance) / (best - chance).  0 = unrelated given the sizes,")
    P("             1 = as identical as masks of those sizes can be.")
    P("")
    P("Three pairings, same completions weighted the same way in each:")
    P("  within      two steps of the SAME chain              <- the hypothesis")
    P("  same-image  two steps of two DIFFERENT chains, same picture   <- is it the image?")
    P("  diff-image  a step's boxes drawn on ANOTHER picture's grid    <- the floor")
    P("")

    for res in res_list:
        P("-" * 96)
        P(f"{res['model']}   ({res['n_images']} images, {res['n_completions']} chains, "
          f"{res['n_steps']} steps, {res['n_within_pairs']} within-chain step pairs)")
        P("-" * 96)
        if not res.get("within"):
            P("  no chain had two grounded steps.")
            P("")
            continue
        uf, nb = res["union_frac"], res["n_boxes"]
        P(f"  a step's mask covers {uf['p50']:.0%} of the patch grid at the median "
          f"({uf['p10']:.0%} / {uf['p90']:.0%} at the deciles); "
          f"DINO returns {nb['p50']:.0f} boxes at the median.")
        P("")
        P("                  pairs      IoU  (p10   p50   p90)   chance   best  closeness  IoU>=.9  identical")
        for tag, label in (("within", "within"), ("cross_comp", "same-image"), ("cross_img", "diff-image")):
            r = res.get(tag)
            if not r:
                continue
            P(f"  {label:11s} {r['n_pairs']:8d}   {fmt(r['iou_mean'])}  "
              f"({fmt(r['iou']['p10'])} {fmt(r['iou']['p50'])} {fmt(r['iou']['p90'])})   "
              f"{fmt(r['iou_chance_mean'])}  {fmt(r['iou_best_mean'])}    "
              f"{fmt(r['closeness_mean'])}     {r['frac_iou_ge_90']:.1%}     {r['frac_identical']:.1%}")
        P("")
        P("  box lists, before rasterisation (mean best-match IoU per box / share of boxes")
        P("  with a >=0.9 partner in the other step):")
        for tag, label in (("within", "within"), ("cross_comp", "same-image"), ("cross_img", "diff-image")):
            r = res.get(tag)
            if not r:
                continue
            P(f"    {label:11s} {fmt(r['box_match_mean'])}   {r['box_hit_mean']:.1%}")
        if res.get("text_bins"):
            P("")
            P("  within-chain pairs split by how many content words the two steps share")
            P(f"  (median share {res.get('text_overlap_median', float('nan')):.2f}, "
              f"correlation with IoU {fmt(res.get('text_corr_iou'))}):")
            P("               pairs    IoU   closeness")
            for b in res["text_bins"]:
                P(f"    {b['label']:14s} {b['n']:5d}  {fmt(b['iou'])}   {fmt(b['closeness'])}")
        P("")
        pc = res["per_completion"]
        m, lo, hi = pc["mean_iou_ci"]
        cm, clo, chi = pc["mean_closeness_ci"]
        P(f"  per chain ({pc['n']} chains, median {pc['n_steps']['p50']:.0f} steps):")
        P(f"    mean IoU over its own step pairs   {m:.3f}  [{lo:.3f}, {hi:.3f}]   "
          f"(deciles {pc['mean_iou']['p10']:.3f} / {pc['mean_iou']['p50']:.3f} / {pc['mean_iou']['p90']:.3f})")
        P(f"    mean closeness                     {cm:.3f}  [{clo:.3f}, {chi:.3f}]")
        P(f"    every step pair IoU >= 0.9         {pc['frac_min_iou_ge_90']:.1%} of chains")
        P(f"    every step's mask bit-identical    {pc['frac_all_identical']:.1%} of chains")
        P(f"    mean IoU >= 0.9 / >= 0.75          {pc['frac_mean_iou_ge_90']:.1%} / "
          f"{pc['frac_mean_iou_ge_75']:.1%} of chains")
        P("")
        sw = res.get("swap") or {}
        if sw.get("n_steps"):
            P(f"  swapping masks, scored with `{sw['metric']}` ({sw['n_steps']} steps, "
              f"{sw['n_groups']} groups of chains):")
            P(f"    step scores higher on its OWN mask than on another step's   "
              f"{sw['own_rank_mean']:.3f}   (0.5 = coin flip)")
            P(f"    own-mask score minus other-mask mean, in units of the")
            P(f"    spread of scores across the chain's steps                   "
              f"{fmt(sw['swap_gap_mean'])}")
            if "rho_first_mean" in sw:
                P(f"    rank correlation of the chain reward against the real one,")
                P(f"    if DINO had been run ONCE per chain (first step's boxes)     "
                  f"{fmt(sw['rho_first_mean'])}   top-1 kept {sw['top1_first']:.1%}")
                P(f"    ... once per IMAGE, on another chain's sentence              "
                  f"{fmt(sw['rho_other_mean'])}   top-1 kept {sw['top1_other']:.1%}")
        if res.get("verify_max_abs_err"):
            v = res["verify_max_abs_err"]
            P("")
            P("  recomputed-vs-stored max abs error: " +
              ", ".join(f"{k} {fmt(x, 5)}" for k, x in v.items()))
        P("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("merged", nargs="+", help="probe_merged.json file(s) from overlap_probe")
    ap.add_argument("--out-dir", default="outputs/step_box_similarity")
    ap.add_argument("--models", default=None,
                    help="comma-separated model keys to keep (default: all)")
    ap.add_argument("--metric", default="mean_in", choices=sorted(METRICS),
                    help="metric for the mask-swap section (default mean_in)")
    ap.add_argument("--verify", action="store_true",
                    help="recompute each step's stored metric from the quantised map "
                         "and report the largest disagreement")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    keep = set(args.models.split(",")) if args.models else None
    out = []

    for path in args.merged:
        d = json.load(open(path))
        cfg = d["config"]
        if not cfg.get("store_maps"):
            print(f"[skip] {path}: store_maps was off, no boxes stored", file=sys.stderr)
            continue
        for mk, mv in d["models"].items():
            if keep and mk not in keep:
                continue
            samples = load_model(mv)
            if not samples:
                continue
            rng = np.random.default_rng(args.seed)
            tag = f"{os.path.basename(os.path.dirname(path))}::{mk}"
            print(f"[run] {tag} ...", file=sys.stderr, flush=True)
            res, _ = analyse_model(tag, samples, rng, args.metric, verify=args.verify)
            res["source"] = path
            res["dataset"] = cfg.get("dataset")
            res["map"] = cfg.get("map") or "attn"
            res["max_box_area"] = cfg.get("max_box_area")
            res["max_union_area"] = cfg.get("max_union_area")
            out.append(res)

    if not out:
        print("nothing to analyse", file=sys.stderr)
        return 1

    txt = render(out, args.metric)
    with open(os.path.join(args.out_dir, "report.txt"), "w") as f:
        f.write(txt + "\n")
    with open(os.path.join(args.out_dir, "report.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(txt)
    print(f"\nwritten to {args.out_dir}/report.txt and report.json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
