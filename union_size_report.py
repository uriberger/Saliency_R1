#!/usr/bin/env python
"""Distribution of the DINO box-UNION size per observe step, from a probe JSON.

Input is overlap_probe.py's probe_merged.json (or the per-shard files). Pure numpy +
stdlib, no GPU, no torch, no transformers -- so it runs wherever the probe output is.

WHY: --max_box_area is a per-BOX filter and bounds nothing about the union that the
reward actually scores. Under auroc the observed failure mode is a policy that drifts
toward describing the BACKGROUND, because background phrases ground to huge boxes and a
huge scored region is easier to rank well against. This report measures that drift and
sizes the --max_union_area cap that would close it.

Union coverage is read in this order, most to least faithful:

  1. union_frac_uncapped     recorded by the feat/overlap-union-cap branch.
  2. RECOMPUTED from boxes_raw + grid (needs the probe's default --store-maps). This is
     the only source that sees FULL-COVERAGE steps: _union_mask returns None when the
     union is 100% of the grid, so those steps are recorded as ungrounded and their
     coverage is lost from both of the other two sources. They matter most here -- they
     are the extreme of exactly the behaviour we are trying to cap.
  3. box_area_frac           always present, but only on grounded steps, so the 100%
     cases are silently missing and the distribution is biased LOW at the top end.

With source 2 the script self-checks its rasteriser against every recorded
box_area_frac; a mismatch means this copy has drifted from _union_mask and the run
aborts rather than reporting numbers computed a different way than the reward.

Usage:
    python union_size_report.py outputs/overlap_probe/<run>/probe_merged.json
    python union_size_report.py <run>/probe_merged.json --caps 0.3,0.4,0.5,0.6
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# rasterisation -- must stay byte-identical to overlap_rewards._union_mask
# ---------------------------------------------------------------------------
def _box_area(b):
    x1, y1, x2, y2 = b
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _union_frac(boxes, gh, gw, max_box_area):
    """Fraction of the (gh, gw) patch grid covered by the area-filtered box union.

    Returns (frac, n_kept). frac is 0.0 when nothing survives the per-box filter, and
    can be exactly 1.0 -- unlike _union_mask this does NOT collapse the degenerate case
    to None, because seeing it is the entire point of this report.
    """
    kept = [b for b in boxes if max_box_area is None or max_box_area <= 0
            or _box_area(b) <= max_box_area]
    if not kept:
        return 0.0, 0
    mask = np.zeros((gh, gw), dtype=bool)
    for x1, y1, x2, y2 in kept:
        r0 = max(0, int(y1 * gh))
        r1 = min(gh, max(r0 + 1, round(y2 * gh)))
        c0 = max(0, int(x1 * gw))
        c1 = min(gw, max(c0 + 1, round(x2 * gw)))
        mask[r0:r1, c0:c1] = True
    return float(mask.sum()) / float(mask.size), len(kept)


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------
def collect(model, max_box_area):
    """-> (steps, completions, source, n_selfcheck).

    steps: one dict per observe step with union coverage attached.
    completions: list of lists of step-indices, so a cap's effect on the per-completion
                 mean (which IS the reward) can be recomputed, not just guessed from
                 the step counts.
    """
    steps, completions = [], []
    sources, checked, mismatch = set(), 0, []
    for smp in model["samples"]:
        for comp in smp["completions"]:
            idxs = []
            for st in comp.get("observe_steps", []):
                frac, n_kept, src = None, st.get("n_boxes_kept"), None
                if st.get("union_frac_uncapped") is not None:
                    frac, src = st["union_frac_uncapped"], "union_frac_uncapped"
                elif st.get("boxes_raw") is not None and st.get("grid"):
                    gh, gw = st["grid"]
                    frac, n_kept = _union_frac(st["boxes_raw"], gh, gw, max_box_area)
                    src = "recomputed"
                    # self-check against what the probe recorded for grounded steps
                    if st.get("box_area_frac") is not None:
                        checked += 1
                        if abs(frac - st["box_area_frac"]) > 1e-9:
                            mismatch.append((frac, st["box_area_frac"]))
                elif st.get("box_area_frac") is not None:
                    frac, src = st["box_area_frac"], "box_area_frac"
                if src:
                    sources.add(src)
                if frac is None:
                    # ungrounded: DINO returned nothing at all. Not a coverage datapoint.
                    steps.append(dict(frac=None, score=None, auroc=None, text=st["text"],
                                      grounded=False, n_boxes_raw=st.get("n_boxes_raw", 0)))
                else:
                    steps.append(dict(
                        frac=frac,
                        score=st.get("score"),
                        auroc=st.get("auroc_raw"),
                        text=st["text"],
                        grounded=bool(st.get("grounded")),
                        n_boxes_raw=st.get("n_boxes_raw", 0),
                        n_boxes_kept=n_kept,
                    ))
                idxs.append(len(steps) - 1)
            completions.append(idxs)
    if mismatch:
        raise SystemExit(
            f"rasteriser mismatch on {len(mismatch)}/{checked} steps (e.g. recomputed "
            f"{mismatch[0][0]:.6f} vs recorded {mismatch[0][1]:.6f}). This script's "
            f"_union_frac has drifted from overlap_rewards._union_mask -- fix before trusting."
        )
    # Report the WEAKEST source in play, so a partially-degraded run is never described
    # as better than it is.
    for s in ("box_area_frac", "recomputed", "union_frac_uncapped"):
        if s in sources:
            return steps, completions, s, checked
    return steps, completions, "none", checked


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def bar(x, width=34, lo=0.0, hi=1.0):
    n = 0 if not np.isfinite(x) else int(round(width * (x - lo) / (hi - lo)))
    return "#" * max(0, min(width, n))


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def report(name, steps, completions, caps, score_field):
    scored = [s for s in steps if s["frac"] is not None]
    fr = np.array([s["frac"] for s in scored], dtype=float)
    n_all, n_ungrounded = len(steps), sum(1 for s in steps if s["frac"] is None)
    n_full = int((fr >= 1.0 - 1e-12).sum())

    print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
    print(f"observe steps: {n_all}   with boxes: {len(scored)}   "
          f"DINO returned nothing: {n_ungrounded}   union == 100% of grid: {n_full}"
          + ("  <- currently skipped by the degenerate guard" if n_full else ""))
    if not len(fr):
        print("no steps with boxes; nothing to report")
        return None

    print("\nunion coverage (fraction of the image the scored region covers)")
    print(f"  mean {fr.mean():.3f}   sd {fr.std():.3f}")
    qs = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print("  " + "  ".join(f"p{q}={pct(fr, q):.3f}" for q in qs) + f"  max={fr.max():.3f}")

    print("\n  histogram (share of steps per coverage decile)")
    hist, _ = np.histogram(fr, bins=10, range=(0.0, 1.0))
    for i, h in enumerate(hist):
        share = h / len(fr)
        print(f"    {i / 10:.1f}-{(i + 1) / 10:.1f}  {share * 100:5.1f}%  "
              f"{bar(share, 30, 0, max(0.001, hist.max() / len(fr)))}  ({h})")

    # Does a bigger scored region actually score better? This is the hypothesis under
    # test: if the correlation is strongly positive, growing the union IS the gradient.
    sc = np.array([(s[score_field] if s[score_field] is not None else np.nan)
                   for s in scored], dtype=float)
    ok = np.isfinite(sc)
    if ok.sum() > 2:
        rp = float(np.corrcoef(fr[ok], sc[ok])[0, 1])
        ranks = lambda v: np.argsort(np.argsort(v)).astype(float)
        rs = float(np.corrcoef(ranks(fr[ok]), ranks(sc[ok]))[0, 1])
        print(f"\n  corr(union coverage, {score_field}):  pearson {rp:+.3f}   spearman {rs:+.3f}")
        print(f"  mean {score_field} by coverage quartile:")
        edges = [0, 25, 50, 75, 100]
        for a, b in zip(edges[:-1], edges[1:]):
            lo, hi = pct(fr[ok], a), pct(fr[ok], b)
            sel = (fr[ok] >= lo) & (fr[ok] <= hi)
            if sel.sum():
                print(f"    coverage {lo:.2f}-{hi:.2f}   n={int(sel.sum()):5d}   "
                      f"{score_field} {sc[ok][sel].mean():.4f}")

    # What each candidate cap would actually do -- to the steps AND to the reward.
    print(f"\n  effect of --max_union_area (a capped step is SKIPPED, not scored 0):")
    print(f"    {'cap':>5}  {'steps kept':>11}  {'dropped':>8}  "
          f"{'mean {} kept'.format(score_field):>16}  {'dropped':>9}  "
          f"{'completions masked':>19}  {'mean reward':>12}")
    base_rewards = comp_rewards(steps, completions, score_field, None)
    base_mean = np.nanmean(base_rewards) if len(base_rewards) else float("nan")
    print(f"    {'off':>5}  {len(scored):11d}  {0:7.1f}%  {np.nanmean(sc):16.4f}  "
          f"{'-':>9}  {fmt_masked(base_rewards):>19}  {base_mean:12.4f}")
    rows = []
    for cap in caps:
        keep = fr <= cap + 1e-12
        kept_sc, drop_sc = sc[ok & keep], sc[ok & ~keep]
        rw = comp_rewards(steps, completions, score_field, cap)
        m = np.nanmean(rw) if np.isfinite(rw).any() else float("nan")
        print(f"    {cap:5.2f}  {int(keep.sum()):11d}  {(~keep).mean() * 100:7.1f}%  "
              f"{(kept_sc.mean() if len(kept_sc) else float('nan')):16.4f}  "
              f"{(drop_sc.mean() if len(drop_sc) else float('nan')):9.4f}  "
              f"{fmt_masked(rw):>19}  {m:12.4f}")
        rows.append((cap, float((~keep).mean()), m))
    return rows


def fmt_masked(rw):
    n = int(np.isnan(rw).sum()) if len(rw) else 0
    return f"{n}/{len(rw)} ({100 * n / max(1, len(rw)):.1f}%)"


def comp_rewards(steps, completions, score_field, cap):
    """Per-completion reward = mean over surviving steps; NaN when none survive (masked)."""
    out = []
    for idxs in completions:
        vals = []
        for i in idxs:
            s = steps[i]
            if s["frac"] is None or s[score_field] is None:
                continue
            if cap is not None and s["frac"] > cap + 1e-12:
                continue
            vals.append(s[score_field])
        out.append(float(np.mean(vals)) if vals else np.nan)
    return np.array(out, dtype=float)


_STOP = set("the a an of in on at to is are was were and or with this that it its for "
            "as by from be been has have had there which what where".split())


def wordiness(steps, frac_lo, label, top=12):
    """Most distinctive words among high-coverage steps -- the 'is it background?' check."""
    hi = [s for s in steps if s["frac"] is not None and s["frac"] >= frac_lo]
    lo = [s for s in steps if s["frac"] is not None and s["frac"] < frac_lo]
    if not hi or not lo:
        return
    def counts(g):
        c = Counter()
        for s in g:
            c.update(w for w in re.findall(r"[a-z]+", s["text"].lower())
                     if len(w) > 2 and w not in _STOP)
        return c, sum(c.values())
    ch, nh = counts(hi)
    cl, nl = counts(lo)
    # Add-one smoothing: without it a word absent from the low group divides by ~0 and
    # reports a millionfold lift, which buries the words that actually separate them.
    lift = [(w, ((n + 1) / (nh + 1)) / ((cl.get(w, 0) + 1) / (nl + 1)), n)
            for w, n in ch.items() if n >= max(3, len(hi) // 40)]
    lift.sort(key=lambda t: -t[1])
    print(f"\n  words over-represented in steps with coverage >= {frac_lo:.2f} "
          f"({len(hi)} steps) vs below ({len(lo)}):")
    print("    " + ", ".join(f"{w} ({l:.1f}x, n={n})" for w, l, n in lift[:top]))
    print(f"  {label} examples (highest coverage):")
    for s in sorted(hi, key=lambda s: -s["frac"])[:3]:
        t = s["text"].strip().replace("\n", " ")
        print(f"    [{s['frac']:.2f}] {t[:150]}{'...' if len(t) > 150 else ''}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("probe_json", nargs="+", help="probe_merged.json (or probe_shard*.json)")
    p.add_argument("--caps", default="0.2,0.3,0.4,0.5,0.6,0.7,0.8",
                   help="candidate --max_union_area values to table")
    p.add_argument("--max-box-area", type=float, default=None,
                   help="per-box filter to apply when recomputing from boxes_raw; "
                        "default: read from the probe config (the value the run used)")
    p.add_argument("--score-field", default="auroc", choices=["auroc", "score"],
                   help="'auroc' uses auroc_raw, recorded on every step whatever the "
                        "configured metric; 'score' is the reward as actually configured")
    p.add_argument("--hi", type=float, default=0.75,
                   help="coverage threshold for the over-represented-words check")
    args = p.parse_args()

    caps = [float(c) for c in args.caps.split(",") if c.strip()]
    models, cfg = {}, None
    for f in args.probe_json:
        d = json.loads(Path(f).read_text())
        cfg = cfg or d.get("config", {})
        for name, m in d["models"].items():
            models.setdefault(name, {"samples": []})["samples"] += m["samples"]

    mba = args.max_box_area if args.max_box_area is not None else cfg.get("max_box_area", 0.5)
    print(f"probe config: metric={cfg.get('overlap_metric')} box_threshold={cfg.get('box_threshold')} "
          f"max_box_area={mba} n_samples={cfg.get('n_samples')} "
          f"num_generations={cfg.get('num_generations')} dataset={Path(str(cfg.get('dataset'))).name}")
    print(f"scoring field: {args.score_field}")

    summary = {}
    for name in models:
        steps, comps, source, checked = collect(models[name], mba)
        print(f"\n[{name}] union coverage source: {source}"
              + (f" (self-checked against {checked} recorded box_area_frac values)"
                 if source == "recomputed" else ""))
        if source == "box_area_frac":
            print("  WARNING: no boxes_raw in this probe run (--no-store-maps) and no "
                  "union_frac_uncapped. Steps whose union covered 100% of the grid were "
                  "recorded as ungrounded, so they are MISSING here and the top of the "
                  "distribution is biased low. Re-run the probe with --store-maps for "
                  "the full picture.")
        rows = report(name, steps, comps, caps, args.score_field)
        wordiness(steps, args.hi, name)
        if rows:
            summary[name] = rows

    if len(summary) > 1:
        print(f"\n{'=' * 78}\nCROSS-CHECKPOINT: % of steps a cap would drop\n{'=' * 78}")
        names = list(summary)
        print(f"  {'cap':>5}  " + "  ".join(f"{n[:22]:>22}" for n in names))
        for i, cap in enumerate(caps):
            print(f"  {cap:5.2f}  " + "  ".join(f"{summary[n][i][1] * 100:21.1f}%" for n in names))
    return 0


if __name__ == "__main__":
    sys.exit(main())
