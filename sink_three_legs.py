#!/usr/bin/env python
"""Where the model LOOKS, where the human says the answer IS, and what the model TALKS about.

    python sink_three_legs.py --dirs A,B,C --out DIR

Three quantities, all of them the same statistic -- a region's share of the picture
divided by its share of the patches, on the picture's OWN grid, 1.0 being a fair share --
so they can be read against each other directly:

  attention   where the model's attention went, over the observe-step tokens
  human box   the corpus's answer-region box, from Visual-CoT
  referent    Grounding-DINO on the sentences the model wrote (sink_observe_boxes.py)

THE NULL IS NOT OPTIONAL. A region's ring enrichment depends on its SIZE and nothing
else, if it is placed at random: the border is a thin frame, so a large blob dropped
anywhere covers proportionally little of it and scores below 1 without preferring the
centre at all. The model's referents are large -- a median 46% of the grid -- so their
0.58 has to be read against what a randomly placed box of the SAME AREA would score on
the SAME grid, or the centre claim is an artefact of box size. Every row therefore
carries its own matched null, and the number that means something is the ratio to it.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def mask_from_box(box, gh, gw):
    """A normalised box -> the patches whose CENTRES it covers. Never empty."""
    x0, y0, x1, y1 = box
    c = (np.arange(gw) + 0.5) / gw
    r = (np.arange(gh) + 0.5) / gh
    m = ((c[None, :] >= x0) & (c[None, :] <= x1)
         & (r[:, None] >= y0) & (r[:, None] <= y1))
    if not m.any():
        m[min(gh - 1, max(0, int((y0 + y1) / 2 * gh))),
          min(gw - 1, max(0, int((x0 + x1) / 2 * gw)))] = True
    return m


def ring_enrich(mask, gh, gw):
    import sink_location as SL
    p = mask.astype(float) / mask.sum()
    ring = SL.ring_set(gh, gw)
    return float(p[ring].sum() / ring.mean())


def random_null(area_frac, gh, gw, rng, n=64):
    """Ring enrichment of a randomly placed axis-aligned box of this area. -> mean.

    The shape is matched too, not just the area: a wide flat box touches the top and
    bottom borders where a square of equal area might not, and referent boxes are not
    square. Sampled per call because the grid differs per picture.
    """
    side = float(np.sqrt(max(area_frac, 1e-6)))
    out = []
    for _ in range(n):
        ar = float(np.exp(rng.uniform(-0.7, 0.7)))          # aspect jitter, ~0.5x to 2x
        w = min(1.0, side * ar)
        h = min(1.0, side / ar)
        x0 = rng.uniform(0.0, max(1e-9, 1.0 - w))
        y0 = rng.uniform(0.0, max(1e-9, 1.0 - h))
        out.append(ring_enrich(mask_from_box((x0, y0, x0 + w, y0 + h), gh, gw), gh, gw))
    return float(np.mean(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--min-mass", type=float, default=0.002)
    ap.add_argument("--seed", type=int, default=20260918)
    args = ap.parse_args()

    P = _load("_3l_probe", "sink_location_probe.py")
    import sink_location as SL

    rng = np.random.default_rng(args.seed)
    I = SL.STAT_INDEX
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("Ring enrichment: a region's share of the picture over its share of the patches.")
    emit("1.00 is a fair share. `null` is a randomly placed box of the SAME area and a")
    emit("similar aspect on the SAME grid -- the value the region would score with no")
    emit("preference at all -- and `ratio` is the column beside it divided by that null.")
    emit("")
    emit(f"{'model':<16} {'n':>5} {'ATTENTION':>10} {'HUMAN box':>10} {'null':>7} "
         f"{'ratio':>7} {'REFERENT':>9} {'null':>7} {'ratio':>7}")

    for d in [x for x in args.dirs.split(",") if x]:
        meta, arrays = P.read_stage(d, "scan")
        if not meta:
            emit(f"{Path(d).name:<16} (no scan results)")
            continue
        man = {r["key"]: r for r in P.read_manifest(d)}
        boxes = {}
        bpath = Path(d) / "observe_boxes.jsonl"
        if bpath.exists():
            for line in bpath.read_text().splitlines():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                boxes[r["key"]] = r

        att, hum, hum0, ref, ref0 = [], [], [], [], []
        for m in meta:
            row = man.get(m["key"])
            if row is None:
                continue
            gh, gw = m["grid"]
            # 1. attention, over the observe-step tokens, all heads clearing the floor
            a = arrays.get(m["unit"], {}).get("stats_obs")
            if a is not None:
                a = np.asarray(a, dtype=float)
                live = a[..., I["image_mass"]] >= args.min_mass
                if live.any():
                    ring = float(np.nanmean(np.where(live, a[..., I["ring_share"]],
                                                     np.nan)))
                    att.append(ring / SL.ring_area_frac(gh, gw))
            # 2. the human box
            bb = row.get("bbox")
            if bb:
                mk = mask_from_box([float(v) for v in bb], gh, gw)
                hum.append(ring_enrich(mk, gh, gw))
                hum0.append(random_null(float(mk.mean()), gh, gw, rng))
            # 3. the model's own referents, one entry per grounded observe step
            rec = boxes.get(m["key"])
            for st in (rec or {}).get("steps", []):
                s = st.get("stats")
                if s and np.isfinite(s.get("ring_enrich", np.nan)):
                    ref.append(s["ring_enrich"])
                    ref0.append(random_null(float(s["area"]), gh, gw, rng))

        f = lambda v: float(np.nanmean(v)) if v else float("nan")   # noqa: E731
        emit(f"{Path(d).name:<16} {len(meta):>5} {f(att):>10.2f} {f(hum):>10.2f} "
             f"{f(hum0):>7.2f} {f(hum)/f(hum0):>7.2f} {f(ref):>9.2f} {f(ref0):>7.2f} "
             f"{f(ref)/f(ref0):>7.2f}   ({len(ref)} steps)")

    emit("")
    emit("A ratio below 1 is a real centre preference: the region sits nearer the middle")
    emit("than its own size explains. A ratio at 1 means the apparent centre bias was")
    emit("nothing but the region's area, which is the artefact the null exists to catch.")
    if args.out:
        Path(args.out).write_text("\n".join(lines) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
