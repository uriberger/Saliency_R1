#!/usr/bin/env python
"""Draw one `fig1_multistep.py` candidate as a Figure-1 panel.

The search writes JSON; this turns one row of it into pictures. Every model in the scan
gets a row, and every row is the SAME two regions -- the two disjoint referents our
candidate's two steps named -- so the rows differ only in where the attention went, which
is the only thing a Figure 1 is allowed to be about.

Per row, three panels:

    original + both boxes | GLIMPSE of the step about region A | GLIMPSE of the step about region B

Our model's two steps are the candidate's own; every other model's are the pair that
maximises its crossover on the same two regions (`others_crossover`), i.e. the best
showing that model can make here, not the one that happens to line up by index.

Output:

    <out>/panel.png     the labelled grid, for reading
    <out>/panel.html    the grid plus the questions, the chains and every number
    <out>/parts/*.png   the same overlays with no captions, for LaTeX

CPU only. `--norm`/`--cmap`/`--alpha` are the saliency_viz render knobs and mean the same
thing here; the default `percentile` 1-99 is what every other picture in this repo uses.

`--smooth SIGMA`, `--upsample` and `--overlay-mode` are `fig1_steps_figure.py`'s and mean
the same thing too, so the same sample drawn by either script is on one scale. A 32x32
glimpse map over a 1024px photograph reads as speckle: `--smooth 1.0` merges the isolated
hot patches into the regions they belong to (past ~1.5 the regions bleed together), and
`--upsample map` -- the default since the RGB order was retired -- interpolates the scalar
field and colours it afterwards instead of interpolating along a straight line between two
colours of a ramp that is not straight, which invented purples jet does not contain.

**None of the three reaches a number.** The blur is a smoother applied to the very
statistic being scored and it flatters in both directions; every v2 in the captions and in
the HTML tables, and every step this script picks for another model, is computed on the
raw grid by `fig1_multistep.py`'s own reward code. `--upsample rgb --smooth 0` reproduces
a panel drawn before 2026-09-22.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


REPO = Path(__file__).resolve().parent


def rerooted(p):
    """A path the scan recorded, made readable from whichever tree is running now.

    `run_dirs` and `sdir_i` go into the JSON as absolute paths, so they name the worktree
    the scan was launched from -- and worktrees here are disposable, so by the time the
    panel is redrawn that prefix is usually gone. `outputs/` is a symlink every tree
    shares, so the file itself never moved: re-root on the last `outputs` component
    instead of trusting the recorded prefix.
    """
    p = Path(p)
    if p.exists():
        return p
    parts = p.parts
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "outputs":
            cand = REPO.joinpath(*parts[i:])
            if cand.exists():
                return cand
    return p


def normalize_map(m, mode, lo, hi):
    """Identical to saliency_viz.normalize_map -- the panel must not be on a new scale."""
    m = np.asarray(m, dtype=np.float64)
    if mode == "rank":
        flat = m.ravel()
        order = flat.argsort(kind="stable")
        ranks = np.empty(flat.size, dtype=np.float64)
        ranks[order] = np.arange(flat.size, dtype=np.float64)
        return (ranks / max(flat.size - 1, 1)).reshape(m.shape)
    if mode == "minmax":
        a, b = float(m.min()), float(m.max())
    else:
        a, b = float(np.percentile(m, lo)), float(np.percentile(m, hi))
    return np.zeros_like(m) if not b > a else np.clip((m - a) / (b - a), 0.0, 1.0)


def gaussian_blur(m, sigma):
    """Separable Gaussian on the patch grid; `sigma` is in PATCHES, not pixels.

    Edge-padded rather than zero-padded. The outer ring carries real mass in every
    Qwen3-VL map -- the encoder stamps it -- and zero padding would dim exactly the ring
    that the border numbers are about, turning a render knob into a silent edit of the
    thing being shown.
    """
    m = np.asarray(m, dtype=np.float64)
    if sigma <= 0:
        return m
    r = int(np.ceil(3 * sigma))
    k = np.exp(-np.arange(-r, r + 1, dtype=np.float64) ** 2 / (2 * sigma ** 2))
    k /= k.sum()
    p = np.pad(m, r, mode="edge")
    p = np.apply_along_axis(np.convolve, 1, p, k, "valid")
    return np.apply_along_axis(np.convolve, 0, p, k, "valid")


def upsample_map(x, size, mode):
    """A normalised [0,1] patch grid resampled to `size`, still as a scalar field."""
    resample = Image.NEAREST if mode == "nearest" else Image.BICUBIC
    big = Image.fromarray(x.astype(np.float32), mode="F").resize(size, resample)
    # bicubic overshoots at a sharp edge; the colormap's domain is [0, 1]
    return np.clip(np.asarray(big, dtype=np.float64), 0.0, 1.0)


def overlay(img, m, cmap, args):
    """The map as colour over the picture. Cosmetic end to end -- every number in the
    captions and in the tables is computed by the caller on the raw `maps`, never here."""
    x = normalize_map(gaussian_blur(m, args.smooth), args.norm, args.norm_lo, args.norm_hi)
    if args.upsample == "rgb":
        rgb = (np.asarray(cmap(x))[..., :3] * 255).astype(np.uint8)
        heat = Image.fromarray(rgb).resize(img.size, Image.BILINEAR)
    else:
        x = upsample_map(x, img.size, args.upsample)
        heat = Image.fromarray((np.asarray(cmap(x))[..., :3] * 255).astype(np.uint8))
    if args.overlay_mode == "alpha":
        a = Image.fromarray((x * 255 * args.alpha).astype(np.uint8))
        if a.size != img.size:
            a = a.resize(img.size, Image.BILINEAR)
        out = img.convert("RGB").copy()
        out.paste(heat, (0, 0), a)
        return out
    return Image.blend(img.convert("RGB"), heat, args.alpha)


def draw_boxes(img, boxes, colour, width=3):
    """Boxes are relative [x1,y1,x2,y2]; the overlay is already at image resolution."""
    out = img.copy()
    d = ImageDraw.Draw(out)
    w, h = out.size
    for x1, y1, x2, y2 in boxes:
        d.rectangle([x1 * w, y1 * h, x2 * w, y2 * h], outline=colour, width=width)
    return out


def caption(img, lines, size=15, pad=5):
    """A dark title bar above the panel, one row per string in `lines`.

    The bar is sized from the rendered text rather than a constant: the labels carry two
    numbers each and a fixed 22 px bar at the default font clipped them, which is how a
    panel ends up mislabelled in a figure.
    """
    from PIL import ImageDraw as _D, ImageFont

    try:
        font = ImageFont.load_default(size=size)
    except TypeError:                                  # Pillow < 10.1
        font = ImageFont.load_default()
        size = 11
    lines = [lines] if isinstance(lines, str) else list(lines)
    bar = pad * 2 + size * len(lines) + 2 * (len(lines) - 1)
    canvas = Image.new("RGB", (img.size[0], img.size[1] + bar), (16, 16, 16))
    canvas.paste(img, (0, bar))
    d = _D.Draw(canvas)
    for i, line in enumerate(lines):
        d.text((pad, pad + i * (size + 2)), line, fill=(238, 238, 238), font=font)
    return canvas


def grid(rows, pad=8):
    w = max(sum(p.size[0] for p in r) + pad * (len(r) + 1) for r in rows)
    h = sum(max(p.size[1] for p in r) for r in rows) + pad * (len(rows) + 1)
    sheet = Image.new("RGB", (w, h), (16, 16, 16))
    y = pad
    for r in rows:
        x = pad
        for p in r:
            sheet.paste(p, (x, y))
            x += p.size[0] + pad
        y += max(p.size[1] for p in r) + pad
    return sheet


def load_step_map(sdir: Path, step: int, key: str):
    z = np.load(sdir / "maps.npz")
    if key not in z.files:
        raise SystemExit(f"{sdir}/maps.npz has no `{key}` map (has {z.files})")
    return np.clip(z[key][step], 0, None).astype(np.float64)


PALETTE = ["#00ff66", "#ff2fd0", "#ffd400", "#00d2ff", "#ff7a3c", "#b18cff"]


def chain_panel(blob, c, run_dir, base_img, out, cmap, args):
    """N steps of our chain, each with its own tight referent, in chain order.

    The pair mode exists for the crossover claim, which is a statement about exactly two
    regions swapping. A chain that enumerates four objects is a different picture and a
    better one, and it needs the whole N x N table -- step k's map scored in every
    region, not just in two -- or "it moved to the right object" is being asserted from
    the diagonal alone.

    Every other model gets, per region, the step of ITS OWN chain that scores highest
    there: the best it can do on this picture, for the same reason the pair mode
    maximises over its step pairs.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("_fig1ms_raster",
                                                  Path(__file__).resolve().parent / "fig1_multistep.py")
    FMS = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(FMS)

    want = [int(x) for x in args.chain.split(",") if x != ""]
    ours = blob["ours"]
    by_model = {}
    for s in blob["steps"]:
        if s["run"] == c["run"] and s["sample"] == c["sample"]:
            by_model.setdefault(s["model"], {})[s["step"]] = s

    picked = []
    for k, si in enumerate(want):
        s = by_model.get(ours, {}).get(si)
        if s is None or not s.get("tight_boxes"):
            raise SystemExit(f"step {si} of {ours} has no tight referent")
        gh, gw = s["grid"]
        picked.append({"step": si, "text": s["text"], "boxes": s["tight_boxes"],
                       "mask": FMS.raster(s["tight_boxes"], gh, gw),
                       "colour": PALETTE[k % len(PALETTE)]})
    if any(p["mask"] is None for p in picked):
        raise SystemExit("a requested step rasterises to an empty referent")

    all_boxes = Image.fromarray(np.asarray(base_img))
    for p in picked:
        all_boxes = draw_boxes(all_boxes, p["boxes"], p["colour"])
    all_boxes.save(out / "parts" / "regions.png")

    rows, tables = [], []
    for tag in [ours] + [t for t in blob["models"] if t != ours]:
        sdir = run_dir / blob["models"].get(tag, tag) / "samples" / c["sample"]
        if not (sdir / "maps.npz").exists():
            print(f"[skip] {tag}: no maps at {sdir}")
            continue
        z = np.load(sdir / "maps.npz")
        if args.map not in z.files:
            print(f"[skip] {tag}: no `{args.map}` map")
            continue
        maps = np.clip(z[args.map], 0, None).astype(np.float64)
        panels = [caption(all_boxes, [tag, "the chain's referents, in order"])]
        table = []
        for k, p in enumerate(picked):
            if tag == ours:
                step = p["step"]
            else:
                # its own best step for this region, not the one with the same index
                scores = [(FMS.OREW._mean_in_v2(maps[t], p["mask"]) or -9e9, t)
                          for t in range(maps.shape[0])]
                step = max(scores)[1]
            row = [FMS.OREW._mean_in_v2(maps[step], q["mask"]) for q in picked]
            table.append((step, row))
            ov = draw_boxes(overlay(base_img, maps[step], cmap, args), p["boxes"], p["colour"])
            ov.save(out / "parts" / f"{tag}_region{k}_step{step:02d}.png")
            others = [v for j, v in enumerate(row) if j != k and v is not None]
            panels.append(caption(ov, [
                f"{tag}  step {step}  -> region {k + 1}",
                f"{args.map} v2 {row[k]:.2f} here, {max(others):.2f} at best elsewhere"]))
        rows.append(panels)
        tables.append((tag, table))

    if not rows:
        raise SystemExit("nothing to draw")
    grid(rows).save(out / "panel.png")

    e = html.escape
    parts = [
        "<!doctype html><meta charset='utf-8'><title>figure 1 chain</title>",
        "<style>body{background:#111;color:#ddd;font:13px/1.55 -apple-system,sans-serif;"
        "margin:24px;max-width:1200px}img{max-width:100%;border:1px solid #333}"
        "pre{white-space:pre-wrap;background:#181818;padding:8px;border-radius:4px}"
        "td,th{padding:3px 10px;text-align:left}code{color:#9cf}"
        "td.d{background:#243; font-weight:bold}</style>",
        f"<h1>{e(c['sample'])} &mdash; {e(str(c['dataset']))}</h1>",
        f"<p><b>Q.</b> <pre>{e(c['question'])}</pre><b>gold:</b> {e(str(c['gt_answer']))}</p>",
        "<p>" + "<br>".join(
            f"<span style='color:{p['colour']}'>region {k + 1}</span> = step {p['step']}: "
            f"<i>{e(p['text'])}</i>" for k, p in enumerate(picked)) + "</p>",
        "<p><img src='panel.png'></p>",
    ]
    for tag, table in tables:
        parts.append(f"<h2>{e(tag)} &mdash; {e(args.map)} v2 of each step's map in each region "
                     "(chance = 1.0; the diagonal is the step's own object)</h2>")
        parts.append("<table><tr><th>step</th>"
                     + "".join(f"<th>region {k + 1}</th>" for k in range(len(picked)))
                     + "</tr>")
        for k, (step, row) in enumerate(table):
            cells = "".join(
                f"<td class='{'d' if j == k else ''}'>"
                f"{'n/a' if v is None else format(v, '.2f')}</td>"
                for j, v in enumerate(row))
            parts.append(f"<tr><td>{step}</td>{cells}</tr>")
        parts.append("</table>")
        sdir = run_dir / blob["models"].get(tag, tag) / "samples" / c["sample"]
        meta = json.loads((sdir / "meta.json").read_text())
        parts.append(f"<pre>{e(meta.get('generation', ''))}</pre>")
    (out / "panel.html").write_text("\n".join(parts))
    print(f"[out] {out/'panel.png'}\n[out] {out/'panel.html'}\n[out] {out/'parts'}/")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", required=True, help="a fig1_multistep.py output")
    ap.add_argument("--rank", type=int, default=0, help="which candidate, 0 = best")
    ap.add_argument("--sample", default=None,
                    help="pick by sample directory name instead of by rank")
    ap.add_argument("--chain", default=None, metavar="i,j,k,...",
                    help="draw these steps of OUR chain in order, each with its own tight "
                         "referent, instead of the candidate's two. This is the panel for a "
                         "chain that enumerates several objects; the pair mode is for the "
                         "crossover claim, which needs exactly two regions to swap between")
    ap.add_argument("--out", required=True)
    ap.add_argument("--map", default="glimpse")
    ap.add_argument("--norm", default="percentile", choices=["percentile", "minmax", "rank"])
    ap.add_argument("--norm-lo", type=float, default=1.0)
    ap.add_argument("--norm-hi", type=float, default=99.0)
    ap.add_argument("--smooth", type=float, default=0.0, metavar="SIGMA",
                    help="Gaussian blur on the patch grid before normalising, sigma in "
                         "PATCHES. Cosmetic only -- every number in the captions and the "
                         "tables is scored on the raw grid. 1.0 is the value the other "
                         "figures use; 0 = off")
    ap.add_argument("--upsample", default="map", choices=["map", "rgb", "nearest"],
                    help="`map` interpolates the scalar field and colours it after "
                         "(default); `rgb` colours the patch grid first and interpolates "
                         "the colours, which is the older order and invents off-ramp "
                         "hues; `nearest` does not interpolate at all")
    ap.add_argument("--cmap", default="jet")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--overlay-mode", default="blend", choices=["blend", "alpha"],
                    help="`blend` tints the whole picture by `--alpha`; `alpha` paints "
                         "the heat in proportion to the map, so the quiet parts stay the "
                         "photograph")
    ap.add_argument("--colour-a", default="#00ff66")
    ap.add_argument("--colour-b", default="#ff2fd0")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    cmap = matplotlib.colormaps[args.cmap]

    blob = json.loads(Path(args.json).read_text())
    cands = blob["candidates"]
    if args.sample:
        cands = [c for c in cands if c["sample"] == args.sample]
        if not cands:
            raise SystemExit(f"no candidate for sample {args.sample}")
    if args.rank >= len(cands):
        raise SystemExit(f"--rank {args.rank} but only {len(cands)} candidates")
    c = cands[args.rank]

    run_by_name = {Path(r).name: rerooted(r) for r in blob["run_dirs"]}
    run_dir = run_by_name[c["run"]]
    ours = blob["ours"]
    out = Path(args.out)
    (out / "parts").mkdir(parents=True, exist_ok=True)

    # Every row is drawn on the candidate's own picture: the scans share `--seed` and
    # `--n-samples`, so the sample directory of the same name under another model is the
    # same row of the same dataset -- but reading the image once makes that explicit
    # rather than trusting it.
    base_img = Image.open(rerooted(c["sdir_i"]) / "original.png").convert("RGB")
    boxes = {"A": c["boxes_i"], "B": c["boxes_j"]}
    colours = {"A": args.colour_a, "B": args.colour_b}

    if args.chain:
        chain_panel(blob, c, run_dir, base_img, out, cmap, args)
        return

    plan = [(ours, c["step_i"], c["step_j"], c["crossover"].get(blob["rank_map"]))]
    for tag, best in (c.get("others_crossover") or {}).items():
        if best is None:
            print(f"[skip] {tag}: no step pair of its own to score on these two regions")
            continue
        plan.append((tag, best["step_i"], best["step_j"], best))

    rows, cards = [], []
    for tag, si, sj, cross in plan:
        sdir = run_dir / blob["models"].get(tag, tag) / "samples" / c["sample"]
        if not (sdir / "maps.npz").exists():
            print(f"[skip] {tag}: no maps at {sdir}")
            continue
        meta = json.loads((sdir / "meta.json").read_text())
        ref = draw_boxes(draw_boxes(base_img, boxes["A"], args.colour_a),
                         boxes["B"], args.colour_b)
        ref.save(out / "parts" / f"{tag}_regions.png")
        panels = [caption(ref, [f"{tag}", "A (green) and B (magenta)"])]
        for role, step in (("A", si), ("B", sj)):
            ov = draw_boxes(overlay(base_img, load_step_map(sdir, step, args.map), cmap, args),
                            boxes[role], colours[role])
            ov.save(out / "parts" / f"{tag}_step{step:02d}_{role}.png")
            v_self = cross[f"self_{'i' if role == 'A' else 'j'}"]
            v_other = cross[f"other_{'i' if role == 'A' else 'j'}"]
            panels.append(caption(ov, [
                f"{tag}  step {step}  (the step about {role})",
                f"{args.map} v2: {v_self:.2f} in {role}, {v_other:.2f} in the other"]))
        rows.append(panels)
        cards.append((tag, si, sj, cross, meta))

    if not rows:
        raise SystemExit("nothing to draw")
    grid(rows).save(out / "panel.png")

    e = html.escape
    parts = [
        "<!doctype html><meta charset='utf-8'><title>figure 1 candidate</title>",
        "<style>body{background:#111;color:#ddd;font:13px/1.55 -apple-system,sans-serif;"
        "margin:24px;max-width:1100px}img{max-width:100%;border:1px solid #333}"
        "pre{white-space:pre-wrap;background:#181818;padding:8px;border-radius:4px}"
        "td,th{padding:3px 10px;text-align:left}code{color:#9cf}</style>",
        f"<h1>{e(c['sample'])} &mdash; {e(str(c['dataset']))}</h1>",
        f"<p><b>Q.</b> {e(c['question'])}<br><b>gold:</b> {e(str(c['gt_answer']))}</p>",
        f"<p>region <span style='color:{args.colour_a}'>A</span> = <i>{e(c['text_i'])}</i><br>"
        f"region <span style='color:{args.colour_b}'>B</span> = <i>{e(c['text_j'])}</i><br>"
        f"grid IoU between them {c['iou']:.3f}; they cover {c['area_i']:.1%} and "
        f"{c['area_j']:.1%} of the patch grid "
        f"(the reward's own union for the same two steps: "
        f"{('n/a' if c['rew_area_i'] is None else format(c['rew_area_i'], '.1%'))} and "
        f"{('n/a' if c['rew_area_j'] is None else format(c['rew_area_j'], '.1%'))}).</p>",
        f"<p><img src='panel.png'></p>",
        "<table><tr><th>model</th><th>steps</th><th>margin</th><th>answer</th>"
        "<th>strict</th><th>soft</th></tr>",
    ]
    grades = {ours: c["ours_grade"], **(c.get("others_grade") or {})}
    answers = {ours: c["ours_answer"], **(c.get("others_answer") or {})}
    for tag, si, sj, cross, _meta in cards:
        g = grades.get(tag) or {}
        parts.append(f"<tr><td><code>{e(tag)}</code></td><td>{si} + {sj}</td>"
                     f"<td>{cross['margin']:+.2f}</td>"
                     f"<td>{e(str(answers.get(tag)))[:160]}</td>"
                     f"<td>{g.get('strict')}</td><td>{g.get('soft')}</td></tr>")
    parts.append("</table>")
    for tag, si, sj, cross, meta in cards:
        parts.append(f"<h2>{e(tag)}</h2><pre>{e(meta.get('generation', ''))}</pre>")
        for k, st in enumerate(meta.get("steps", [])):
            mark = " &larr; A" if k == si else (" &larr; B" if k == sj else "")
            parts.append(f"<div>step {k}{mark}: {e(st['text'])}</div>")
    (out / "panel.html").write_text("\n".join(parts))
    print(f"[out] {out/'panel.png'}\n[out] {out/'panel.html'}\n[out] {out/'parts'}/")


if __name__ == "__main__":
    main()
