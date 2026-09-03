#!/usr/bin/env python
"""Where does `mean_in`'s DENOMINATOR sit, and how flat is the map around it?

`mean_in = mean_U(m) / max(m)`. Every reading of that metric so far has been about
the numerator -- does the map put mass inside the union DINO drew for the step. This
asks the two questions about the denominator that were never asked:

  1. Is the argmax patch -- the single patch the whole score is divided by -- inside
     the union or outside it? At the cold start, and after training on `mean_in` and
     on `mean_in_v2` (which divides by the map's MEAN instead, and is the variant that
     did worse).
  2. Did the map get flatter -- smaller variance -- inside the union, outside it, and
     over the whole grid?

Nothing is computed on a GPU. `overlap_probe.py --store-maps` already persists, per
observe step: the patch map quantised to its own peak (`map_q`, uint8 row-major), the
absolute peak (`map_max`), the DINO union raster (`mask_q`), the grid, and the scores.
The absolute value of a patch is exactly `q/255 * map_max`, so both the shape questions
(scale-free) and the scale questions (absolute sd, image mass) are answerable from the
stored bytes. `--selftest` re-derives the stored `mean_in_raw` / `mean_in_v2_raw` from
the decode and reports the worst disagreement; run it before believing anything else.

Three things this measurement has to control for, all of them union area:

  - A bigger union catches the peak more often for free. Every peak-location number is
    therefore reported against its own chance level, `mean(union_frac)` -- the
    probability a uniformly-placed peak lands inside -- and the LIFT over it, and the
    union-stratified table repeats the comparison inside area bins so a model that only
    changed its box sizes cannot look like a model that moved its attention.
  - Absolute variance conflates flatness with mass: the `mean_in`-trained policy puts
    2.3x the cold start's softmax mass on the image, which raises every absolute sd
    without changing the shape at all. Both are reported -- absolute sd, and the
    coefficient of variation sd/mean, which is scale-free.
  - The models generate their own completions, so the steps and the boxes differ
    between them. The images do not: all models ran the same prompts, so every CI here
    is a bootstrap over IMAGES (the cluster), paired across models by common random
    numbers, and the differences are image-paired.

    python peak_location_probe.py --probe outputs/overlap_probe/<dir> [--selftest]
"""

from __future__ import annotations

import argparse
import base64
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# Order matters only for the report; unknown models are appended in file order.
PREFERRED_ORDER = [
    "base_coldstart",
    "mean_in_set_a_1000",
    "mean_in_set_a_2000",
    "mean_in_saliency_r1_8k",
    "mean_in_v2_set_a_1000",
    "mean_in_v2_set_a_1500",
    "mean_in_v2_set_a_1700",
    "mean_in_v2_beta_004",
    "auroc_set_a_1000",
    "auroc_set_a_2000",
    "auroc_set_a_2500",
]

UNION_BINS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0001)]


def _u8(b64: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(b64), dtype=np.uint8)


def step_stats(step: dict) -> dict | None:
    """Every per-step number this probe reports, from one stored observe step.

    Returns None for a step that has no map, no union, or a degenerate union (all
    patches in or none in) -- "inside vs outside" is not a question there, and
    `_union_mask` already refuses those, so this only catches decode surprises.
    """
    if not step.get("grounded") or not step.get("map_q") or not step.get("mask_q"):
        return None
    gh, gw = step["grid"]
    q = _u8(step["map_q"]).astype(np.float64)
    mask = _u8(step["mask_q"]).astype(bool)
    if q.size != gh * gw or mask.size != gh * gw:
        return None
    n_in = int(mask.sum())
    if n_in == 0 or n_in == q.size:
        return None
    qmax = q.max()
    if qmax <= 0:
        return None

    # Absolute scale. q/255*map_max is the patch's attention weight; keeping both the
    # scale-free and the absolute view is the whole point of the stored pair.
    scale = float(step["map_max"]) / 255.0
    m = q * scale
    m_in, m_out = m[mask], m[~mask]

    # The peak. Ties are a quantisation artefact (two patches within 1/510 of the peak
    # land on the same byte), so `peak_in` is the FRACTION of tied peak patches that
    # are inside -- 0 or 1 whenever the peak is unique, which is the normal case.
    peak_idx = np.flatnonzero(q == qmax)
    peak_in = float(mask[peak_idx].mean())
    rows, cols = np.divmod(peak_idx, gw)
    r0, c0 = int(rows[0]), int(cols[0])

    # Where the peak sits in the frame, independent of the box: an attention sink
    # pinned to a corner would make `max(m)` a constant of the model, not a property
    # of the step, and that changes what `mean_in` is measuring.
    border = (r0 == 0) or (r0 == gh - 1) or (c0 == 0) or (c0 == gw - 1)
    row_frac = (r0 + 0.5) / gh
    col_frac = (c0 + 0.5) / gw

    # How far the union itself reaches into the ring the peak lives on. A box drawn
    # around an object almost never covers the outermost patches, so "the peak is
    # outside the union" and "the peak is on the border" are not independent claims,
    # and the border-conditional table below needs the union's own border coverage as
    # its chance level.
    ring = np.zeros((gh, gw), dtype=bool)
    ring[0, :] = ring[-1, :] = True
    ring[:, 0] = ring[:, -1] = True
    ring = ring.reshape(-1)
    mask2d_border = mask[ring]
    mask2d_inner = mask[~ring]

    # Top-k location: is it only the single argmax that sits outside, or the whole
    # head of the distribution? k=5% of the grid, plus a fixed k=5.
    order = np.argsort(-q, kind="stable")
    k5 = max(1, int(np.ceil(0.05 * q.size)))
    top5pct_in = float(mask[order[:k5]].mean())
    top5_in = float(mask[order[:5]].mean())

    def _cv(x):
        mu = x.mean()
        return float(x.std() / mu) if mu > 0 else float("nan")

    # A grounded step whose union happens to fall entirely on quantised-to-zero
    # patches has mean_in == 0 exactly; the logs are undefined there, so they go NaN
    # and the decomposition table drops that step rather than the whole record.
    q_in = q[mask].mean()
    _log = (lambda x: float(np.log(x))) if q_in > 0 else (lambda x: float("nan"))

    return {
        "union_frac": n_in / q.size,
        "n_patches": int(q.size),
        "peak_in": peak_in,
        "peak_border": float(border),
        "peak_row_frac": row_frac,
        "peak_col_frac": col_frac,
        "peak_top_row": float(r0 == 0),
        "peak_corner": float((r0 in (0, gh - 1)) and (c0 in (0, gw - 1))),
        "union_on_border": float(mask2d_border.mean()),
        "union_on_inner": float(mask2d_inner.mean()) if mask2d_inner.size else float("nan"),
        "top5_in": top5_in,
        "top5pct_in": top5pct_in,
        # Shape, scale-free: recomputed here rather than read back, so the selftest
        # has something independent to compare the stored score against.
        "mean_in": float(q[mask].mean() / 255.0),
        "mean_in_v2": float(q[mask].mean() / q.mean()),
        # The identity the whole comparison turns on:
        #     mean_in = mean_U(m)/max(m) = [mean_U(m)/mean(m)] * [mean(m)/max(m)]
        #             = mean_in_v2 * flatness
        # exactly, step by step. `flatness` is the mask-free statistic
        # (maskfree_rewards.py's `flatness`): it never sees a box, it is 1 for a
        # perfectly flat map and ~1/P for a delta. So the two metrics differ by
        # exactly one box-blind factor, and taking logs makes the split additive --
        # which is what lets the report say WHICH factor each run moved.
        "flatness": float(q.mean() / 255.0),
        "log_mean_in": _log(q_in / 255.0),
        "log_mean_in_v2": _log(q_in / q.mean()),
        "log_flatness": _log(q.mean() / 255.0),
        # Flatness. sd_* are absolute attention units; cv_* divide by the region's own
        # mean and so cannot move when the model merely attends to the image harder.
        "sd_all": float(m.std()),
        "sd_in": float(m_in.std()),
        "sd_out": float(m_out.std()),
        "cv_all": _cv(m),
        "cv_in": _cv(m_in),
        "cv_out": _cv(m_out),
        # sd over the peak: the denominator `mean_in` actually uses, so this is the
        # variance in the units the reward sees.
        "sdmax_all": float(m.std() / m.max()),
        "sdmax_in": float(m_in.std() / m.max()),
        "sdmax_out": float(m_out.std() / m.max()),
        "mu_all": float(m.mean()),
        "mu_in": float(m_in.mean()),
        "mu_out": float(m_out.mean()),
        "map_max": float(step["map_max"]),
        "image_mass": float(step["image_mass"]),
        # Quantisation guard: if the byte-scale sd of a region is a couple of counts,
        # the uniform +-0.5 rounding error (variance 1/12) is no longer negligible.
        "sd_q_in": float(q[mask].std()),
        "sd_q_out": float(q[~mask].std()),
        "_stored_mean_in": step.get("mean_in_raw"),
        "_stored_mean_in_v2": step.get("mean_in_v2_raw"),
    }


def collect(probe_json: Path) -> tuple[dict, dict]:
    """-> ({model: {image_id: [step stats]}}, config)"""
    d = json.loads(probe_json.read_text())
    out = {}
    for name, mdl in d["models"].items():
        by_image = defaultdict(list)
        for smp in mdl["samples"]:
            img = smp["row_index"]
            for comp in smp["completions"]:
                for si, step in enumerate(comp.get("observe_steps") or []):
                    s = step_stats(step)
                    if s is None:
                        continue
                    s["completion"] = (img, comp["index"])
                    by_image[img].append(s)
        out[name] = dict(by_image)
    return out, d["config"]


COLUMNS = ["union_frac", "peak_in", "peak_border", "peak_row_frac", "peak_col_frac",
           "peak_top_row", "peak_corner", "union_on_border", "union_on_inner",
           "top5_in", "top5pct_in", "mean_in", "mean_in_v2", "flatness",
           "log_mean_in", "log_mean_in_v2", "log_flatness",
           "sd_all", "sd_in", "sd_out", "cv_all", "cv_in", "cv_out",
           "sdmax_all", "sdmax_in", "sdmax_out", "mu_all", "mu_in", "mu_out",
           "map_max", "image_mass", "n_patches"]


def pooled(by_image: dict, images) -> dict:
    """Per-step mean of every column over the given images (with repetition)."""
    steps = [s for im in images for s in by_image.get(im, ())]
    if not steps:
        return {c: float("nan") for c in COLUMNS} | {"n_steps": 0}
    arr = {c: np.array([s[c] for s in steps], dtype=float) for c in COLUMNS}
    res = {c: float(np.nanmean(v)) for c, v in arr.items()}
    res["n_steps"] = len(steps)
    # Chance-corrected peak location: a uniformly placed peak lands inside the union
    # with probability union_frac, so this is the only reading of `peak_in` that a
    # change in box size cannot manufacture.
    res["peak_lift"] = res["peak_in"] - res["union_frac"]
    res["top5pct_lift"] = res["top5pct_in"] - res["union_frac"]
    return res


def per_completion(by_image: dict, images) -> dict:
    """Same means, but each COMPLETION weighted equally instead of each step.

    Robustness only: the runs differ several-fold in steps per completion (3.6 at the
    cold start vs 13.7 for mean_in cp2000), so a per-step mean silently re-weights the
    corpus toward the chattiest policy.
    """
    groups = defaultdict(list)
    for im in images:
        for s in by_image.get(im, ()):
            groups[s["completion"]].append(s)
    if not groups:
        return {c: float("nan") for c in COLUMNS}
    rows = [{c: float(np.nanmean([s[c] for s in g])) for c in COLUMNS}
            for g in groups.values()]
    res = {c: float(np.nanmean([r[c] for r in rows])) for c in COLUMNS}
    res["n_completions"] = len(rows)
    res["peak_lift"] = res["peak_in"] - res["union_frac"]
    return res


def bootstrap(models: dict, n_boot: int, seed: int):
    """Image-clustered bootstrap. Common resamples across models, so the model-minus-
    coldstart differences are paired on the image and not just on the corpus."""
    images = sorted({im for by in models.values() for im in by})
    rng = np.random.default_rng(seed)
    draws = [rng.choice(len(images), size=len(images), replace=True) for _ in range(n_boot)]
    boots = {}
    for name, by in models.items():
        rows = []
        for idx in draws:
            rows.append(pooled(by, [images[i] for i in idx]))
        boots[name] = rows
    return boots, images


def ci(vals, lo=2.5, hi=97.5):
    v = np.array([x for x in vals if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return float("nan"), float("nan")
    return float(np.percentile(v, lo)), float(np.percentile(v, hi))


def fmt(x, nd=3):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--probe", required=True,
                   help="an overlap_probe out-dir (uses its probe_merged.json)")
    p.add_argument("--out", default="", help="write the full numbers as JSON here")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--baseline", default="base_coldstart")
    p.add_argument("--selftest", action="store_true",
                   help="re-derive the stored mean_in / mean_in_v2 from the decode")
    args = p.parse_args()

    probe = Path(args.probe)
    merged = probe if probe.is_file() else probe / "probe_merged.json"
    models, cfg = collect(merged)
    names = [n for n in PREFERRED_ORDER if n in models] + \
            [n for n in models if n not in PREFERRED_ORDER]

    print(f"probe      : {merged}")
    print(f"dataset    : {cfg.get('dataset')}  split={cfg.get('split')} "
          f"n_samples={cfg.get('n_samples')} x {cfg.get('num_generations')} gens")
    print(f"map        : {cfg.get('map', 'attn')}  L{cfg.get('overlap_layer')} "
          f"heads {cfg.get('overlap_heads')} token_reduction={cfg.get('token_reduction')}")
    print(f"grounding  : box_threshold={cfg.get('box_threshold')} "
          f"max_box_area={cfg.get('max_box_area')} max_union_area={cfg.get('max_union_area')}")

    if args.selftest:
        worst_in = worst_v2 = 0.0
        n = 0
        for by in models.values():
            for steps in by.values():
                for s in steps:
                    if s["_stored_mean_in"] is not None:
                        worst_in = max(worst_in, abs(s["mean_in"] - s["_stored_mean_in"]))
                        worst_v2 = max(worst_v2, abs(s["mean_in_v2"] - s["_stored_mean_in_v2"]))
                        n += 1
        print(f"\n[selftest] {n} steps: worst |recomputed - stored| "
              f"mean_in {worst_in:.2e}  mean_in_v2 {worst_v2:.2e}  "
              f"(both are pure uint8 rounding; >1e-2 means the decode is wrong)")

    boots, images = bootstrap(models, args.n_boot, args.seed)
    point = {n: pooled(models[n], images) for n in names}
    pcomp = {n: per_completion(models[n], images) for n in names}
    print(f"\nimages     : {len(images)}   bootstrap: {args.n_boot} image-clustered resamples\n")

    # ---- Q1-Q3: where is the peak? -------------------------------------------------
    print("=" * 104)
    print("PEAK LOCATION -- is the patch that mean_in divides by inside the DINO union?")
    print("=" * 104)
    print(f"{'model':<26} {'steps':>6} {'union':>7} {'P(peak in)':>11} {'[95% CI]':>17} "
          f"{'chance':>7} {'lift':>7} {'[95% CI]':>17}")
    for n in names:
        pt, bt = point[n], boots[n]
        lo, hi = ci([b["peak_in"] for b in bt])
        llo, lhi = ci([b["peak_lift"] for b in bt])
        print(f"{n:<26} {pt['n_steps']:>6d} {fmt(pt['union_frac']):>7} "
              f"{fmt(pt['peak_in']):>11} {'[' + fmt(lo) + ', ' + fmt(hi) + ']':>17} "
              f"{fmt(pt['union_frac']):>7} {fmt(pt['peak_lift']):>7} "
              f"{'[' + fmt(llo) + ', ' + fmt(lhi) + ']':>17}")

    base = args.baseline
    if base in models:
        print(f"\nvs {base}, image-paired (CI excluding 0 = the peak moved):")
        print(f"{'model':<26} {'d P(peak in)':>13} {'[95% CI]':>19} {'d lift':>9} "
              f"{'[95% CI]':>19} {'d union':>9}")
        for n in names:
            if n == base:
                continue
            dif = [b["peak_in"] - a["peak_in"] for a, b in zip(boots[base], boots[n])]
            dlf = [b["peak_lift"] - a["peak_lift"] for a, b in zip(boots[base], boots[n])]
            lo, hi = ci(dif)
            llo, lhi = ci(dlf)
            print(f"{n:<26} {fmt(point[n]['peak_in'] - point[base]['peak_in']):>13} "
                  f"{'[' + fmt(lo) + ', ' + fmt(hi) + ']':>19} "
                  f"{fmt(point[n]['peak_lift'] - point[base]['peak_lift']):>9} "
                  f"{'[' + fmt(llo) + ', ' + fmt(lhi) + ']':>19} "
                  f"{fmt(point[n]['union_frac'] - point[base]['union_frac']):>9}")

    # ---- union-stratified, so box size cannot explain it ---------------------------
    print("\n" + "=" * 104)
    print("P(peak in union) WITHIN union-area bins -- the same question at matched box size")
    print("=" * 104)
    hdr = "  ".join(f"{lo:.1f}-{hi:.1f}".rjust(11) for lo, hi in UNION_BINS)
    print(f"{'model':<26} {hdr}")
    for n in names:
        cells = []
        for lo, hi in UNION_BINS:
            steps = [s for st in models[n].values() for s in st
                     if lo <= s["union_frac"] < hi]
            if len(steps) < 15:
                cells.append(f"{'-':>11}")
            else:
                pin = float(np.mean([s["peak_in"] for s in steps]))
                ch = float(np.mean([s["union_frac"] for s in steps]))
                cells.append(f"{pin:.2f}/{ch:.2f}({len(steps):>3d})".rjust(11))
        print(f"{n:<26} " + "  ".join(cells))
    print("  cells are  P(peak in) / chance (n steps); '-' = fewer than 15 steps")

    # ---- is the peak a step property or a fixed sink? ------------------------------
    print("\n" + "=" * 104)
    print("WHERE the peak sits in the frame, box-blind")
    print("=" * 104)
    print(f"{'model':<26} {'border':>8} {'row':>7} {'col':>7} {'top5 in':>9} "
          f"{'top5% in':>9} {'top5% lift':>11}")
    for n in names:
        pt = point[n]
        print(f"{n:<26} {fmt(pt['peak_border']):>8} {fmt(pt['peak_row_frac']):>7} "
              f"{fmt(pt['peak_col_frac']):>7} {fmt(pt['top5_in']):>9} "
              f"{fmt(pt['top5pct_in']):>9} {fmt(pt['top5pct_lift']):>11}")
    print("  border = share of peaks on the outer ring of the patch grid; row/col are the")
    print("  peak's normalised position (0.5 = centre); top5% is the top 5% of patches.")

    # ---- is "peak outside the box" just "peak on the border"? ----------------------
    print("\n" + "=" * 104)
    print("BORDER vs UNION -- the peak is on the outer ring, and the union rarely reaches it")
    print("=" * 104)
    print(f"{'model':<26} {'top row':>8} {'corner':>7} | {'union@ring':>11} {'union@inner':>12} | "
          f"{'P(in|ring)':>11} {'P(in|inner)':>12}")
    for n in names:
        pt = point[n]
        ring = [s for st in models[n].values() for s in st if s["peak_border"] > 0.5]
        inner = [s for st in models[n].values() for s in st if s["peak_border"] <= 0.5]
        p_ring = float(np.mean([s["peak_in"] for s in ring])) if len(ring) >= 15 else float("nan")
        p_inner = float(np.mean([s["peak_in"] for s in inner])) if len(inner) >= 15 else float("nan")
        print(f"{n:<26} {fmt(pt['peak_top_row']):>8} {fmt(pt['peak_corner']):>7} | "
              f"{fmt(pt['union_on_border']):>11} {fmt(pt['union_on_inner']):>12} | "
              f"{fmt(p_ring):>11} {fmt(p_inner):>12}")
    print("  union@ring / union@inner = share of ring / interior patches the union covers,")
    print("  i.e. the chance level for a peak that sits there. P(in|ring) is the observed rate.")

    # ---- Q4: flatness --------------------------------------------------------------
    print("\n" + "=" * 104)
    print("FLATNESS -- sd of the patch values, absolute (x1e4) and scale-free (cv = sd/mean)")
    print("=" * 104)
    print(f"{'model':<26} {'sd_all':>8} {'sd_in':>8} {'sd_out':>8} | {'cv_all':>7} "
          f"{'cv_in':>7} {'cv_out':>7} | {'mass':>8} {'max x1e4':>9} {'mean_in':>8}")
    for n in names:
        pt = point[n]
        print(f"{n:<26} {fmt(pt['sd_all'] * 1e4, 2):>8} {fmt(pt['sd_in'] * 1e4, 2):>8} "
              f"{fmt(pt['sd_out'] * 1e4, 2):>8} | {fmt(pt['cv_all'], 3):>7} "
              f"{fmt(pt['cv_in'], 3):>7} {fmt(pt['cv_out'], 3):>7} | "
              f"{fmt(pt['image_mass'], 5):>8} {fmt(pt['map_max'] * 1e4, 2):>9} "
              f"{fmt(pt['mean_in'], 4):>8}")

    if base in models:
        print(f"\nflatness vs {base}, image-paired, as a RATIO (<1 = flatter):")
        print(f"{'model':<26} " + " ".join(f"{c:>22}" for c in
                                           ("cv_all", "cv_in", "cv_out")))
        for n in names:
            if n == base:
                continue
            cells = []
            for c in ("cv_all", "cv_in", "cv_out"):
                r = point[n][c] / point[base][c]
                rl, rh = ci([b[c] / a[c] for a, b in zip(boots[base], boots[n])
                             if np.isfinite(a[c]) and a[c] > 0])
                cells.append(f"{fmt(r)} [{fmt(rl)}, {fmt(rh)}]".rjust(22))
            print(f"{n:<26} " + " ".join(cells))

        print(f"\nabsolute sd vs {base}, image-paired, as a RATIO (<1 = smaller spread):")
        print(f"{'model':<26} " + " ".join(f"{c:>22}" for c in
                                           ("sd_all", "sd_in", "sd_out")))
        for n in names:
            if n == base:
                continue
            cells = []
            for c in ("sd_all", "sd_in", "sd_out"):
                r = point[n][c] / point[base][c]
                rl, rh = ci([b[c] / a[c] for a, b in zip(boots[base], boots[n])
                             if np.isfinite(a[c]) and a[c] > 0])
                cells.append(f"{fmt(r)} [{fmt(rl)}, {fmt(rh)}]".rjust(22))
            print(f"{n:<26} " + " ".join(cells))

    # ---- what actually raised the score --------------------------------------------
    print("\n" + "=" * 104)
    print("DECOMPOSITION -- mean_in = mean_in_v2 x flatness, exactly, step by step")
    print("=" * 104)
    print(f"{'model':<26} {'mean_in':>9} {'mean_in_v2':>11} {'flatness':>9} | "
          f"{'d log mean_in':>14} {'= d log v2':>11} {'+ d log flat':>13} {'flat share':>11}")
    for n in names:
        pt = point[n]
        cells = ""
        if base in models and n != base:
            d_all = pt["log_mean_in"] - point[base]["log_mean_in"]
            d_v2 = pt["log_mean_in_v2"] - point[base]["log_mean_in_v2"]
            d_fl = pt["log_flatness"] - point[base]["log_flatness"]
            share = d_fl / d_all if abs(d_all) > 1e-9 else float("nan")
            cells = (f" {fmt(d_all):>14} {fmt(d_v2):>11} {fmt(d_fl):>13} "
                     f"{fmt(share, 2):>11}")
        print(f"{n:<26} {fmt(pt['mean_in'], 4):>9} {fmt(pt['mean_in_v2'], 3):>11} "
              f"{fmt(pt['flatness'], 4):>9} |{cells}")
    print("  flatness = mean(m)/max(m) over the WHOLE grid -- the mask-free statistic; it")
    print("  never sees a box. 'flat share' is how much of the change in log mean_in came")
    print("  from it rather than from the part that looks at the union.")

    # ---- weighting robustness ------------------------------------------------------
    print("\n" + "=" * 104)
    print("PER-COMPLETION weighting (each completion counts once, not each step)")
    print("=" * 104)
    print(f"{'model':<26} {'comps':>6} {'P(peak in)':>11} {'chance':>7} {'lift':>7} "
          f"{'cv_all':>8} {'cv_in':>8} {'cv_out':>8}")
    for n in names:
        pc = pcomp[n]
        print(f"{n:<26} {pc.get('n_completions', 0):>6d} {fmt(pc['peak_in']):>11} "
              f"{fmt(pc['union_frac']):>7} {fmt(pc['peak_lift']):>7} "
              f"{fmt(pc['cv_all']):>8} {fmt(pc['cv_in']):>8} {fmt(pc['cv_out']):>8}")

    # ---- quantisation guard --------------------------------------------------------
    worst = min((float(np.mean([s[c] for st in models[n].values() for s in st]))
                 for n in names for c in ("sd_q_in", "sd_q_out")), default=float("nan"))
    print(f"\n[quantisation] smallest mean byte-scale sd over all models/regions: "
          f"{worst:.1f} counts. The uint8 rounding adds sd sqrt(1/12)=0.29 counts, so "
          f"this inflates a variance by ~{100 * (1 / 12) / max(worst, 1e-9) ** 2:.2f}%.")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "probe": str(merged), "config": cfg, "images": images,
            "point": point, "per_completion": pcomp,
            "ci": {n: {k: ci([b[k] for b in boots[n]])
                       for k in list(COLUMNS) + ["peak_lift"]} for n in names},
        }, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
