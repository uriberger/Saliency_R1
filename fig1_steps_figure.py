#!/usr/bin/env python
"""One chain as a figure: the picture, the question, and one heatmap per reasoning step.

`fig1_panel.py` draws a *comparison* -- several models on two or four fixed regions, with
the boxes and the crossover numbers, which is what an argument needs. This draws the thing
itself: the input, what was asked, and then the same picture once per observe step with
that step's attention over it, in chain order. No detector, no boxes unless asked for, no
second model.

    python fig1_steps_figure.py --run-dir outputs/saliency_viz/fig1ms-bench \
        --model ours --sample sample_007_row000007 --out outputs/fig1-multistep/figure-clevr

    <out>/figure.png        the composed figure: header, input, one panel per step
    <out>/figure.html       the same, with the full chain text under it
    <out>/parts/input.png   the picture, unscaled and unlabelled
    <out>/parts/step00.png  one uncaptioned overlay per step, for LaTeX

`--boxes <fig1_multistep json>` adds each step's tight Grounding-DINO referent to its own
panel; without it nothing is drawn over the heatmap, which is what a figure that is about
the attention rather than about the grounding wants.

The render knobs are `saliency_viz.py`'s and mean the same thing: `--norm percentile`
(1-99) is the default everywhere in this repo, so a panel here is on the same scale as one
from `--stage render`. `--scale` only resamples for output -- the map is always normalised
on the patch grid first.
"""

from __future__ import annotations

import argparse
import html
import json
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BG = (17, 17, 17)
FG = (238, 238, 238)
DIM = (150, 150, 150)


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                              # Pillow < 10.1 has no sized default
        return ImageFont.load_default()


def normalize_map(m, mode, lo, hi):
    """Identical to saliency_viz.normalize_map, so this panel is on the same scale."""
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


def overlay(img, m, cmap, args):
    x = normalize_map(m, args.norm, args.norm_lo, args.norm_hi)
    rgb = (np.asarray(cmap(x))[..., :3] * 255).astype(np.uint8)
    heat = Image.fromarray(rgb).resize(img.size, Image.BILINEAR)
    if args.overlay_mode == "alpha":
        a = Image.fromarray((x * 255 * args.alpha).astype(np.uint8)).resize(
            img.size, Image.BILINEAR)
        out = img.convert("RGB").copy()
        out.paste(heat, (0, 0), a)
        return out
    return Image.blend(img.convert("RGB"), heat, args.alpha)


def draw_boxes(img, boxes, colour, width=2):
    out = img.copy()
    d = ImageDraw.Draw(out)
    w, h = out.size
    for x1, y1, x2, y2 in boxes:
        d.rectangle([x1 * w, y1 * h, x2 * w, y2 * h], outline=colour, width=width)
    return out


def wrap(text, f, width_px):
    """Greedy wrap measured in PIXELS, not characters.

    A character count is wrong by a factor of two between "1." and "metallic", and the
    step texts here are exactly that mix, so a fixed textwrap width either overflows the
    panel or wastes half of it.
    """
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if cur and f.getlength(trial) > width_px:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


def text_block(lines, f, width, pad, line_h, colour=FG, bg=BG):
    h = pad * 2 + line_h * max(len(lines), 1)
    img = Image.new("RGB", (width, h), bg)
    d = ImageDraw.Draw(img)
    for i, ln in enumerate(lines):
        d.text((pad, pad + i * line_h), ln, fill=colour, font=f)
    return img


def captioned(panel, title, body_lines, f_title, f_body, pad, line_h):
    """Panel with a one-line title above it and a wrapped caption below."""
    w = panel.size[0]
    top = text_block([title], f_title, w, pad, line_h)
    bottom = text_block(body_lines, f_body, w, pad, line_h, colour=DIM)
    out = Image.new("RGB", (w, top.size[1] + panel.size[1] + bottom.size[1]), BG)
    out.paste(top, (0, 0))
    out.paste(panel, (0, top.size[1]))
    out.paste(bottom, (0, top.size[1] + panel.size[1]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="a saliency_viz output root")
    ap.add_argument("--model", required=True, help="the subdirectory under it")
    ap.add_argument("--sample", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--map", default="glimpse")
    ap.add_argument("--steps", default=None, help="comma-separated; default every observe step")
    ap.add_argument("--question", default=None,
                    help="header text, replacing the prompt the model was actually given. "
                         "A benchmark prompt carries its own scaffolding ('Answer with the "
                         "option letter only.') that a paper figure does not want; the real "
                         "prompt stays in figure.html either way, so nothing is hidden")
    ap.add_argument("--cols", type=int, default=0,
                    help="panels per row, counting the input picture; 0 = one row")
    ap.add_argument("--scale", type=float, default=2.0, help="output resampling only")
    ap.add_argument("--boxes", default=None,
                    help="a fig1_multistep.py json; adds each step's tight referent")
    ap.add_argument("--box-colour", default="#00ff66")
    ap.add_argument("--caption-lines", type=int, default=3)
    ap.add_argument("--norm", default="percentile", choices=["percentile", "minmax", "rank"])
    ap.add_argument("--norm-lo", type=float, default=1.0)
    ap.add_argument("--norm-hi", type=float, default=99.0)
    ap.add_argument("--cmap", default="jet")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--overlay-mode", default="blend", choices=["blend", "alpha"])
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    cmap = matplotlib.colormaps[args.cmap]

    sdir = Path(args.run_dir) / args.model / "samples" / args.sample
    if not (sdir / "maps.npz").exists():
        raise SystemExit(f"no maps.npz under {sdir}")
    meta = json.loads((sdir / "meta.json").read_text())
    z = np.load(sdir / "maps.npz")
    if args.map not in z.files:
        raise SystemExit(f"{sdir}/maps.npz has no `{args.map}` map (has {z.files})")
    maps = np.clip(z[args.map], 0, None).astype(np.float64)
    img = Image.open(sdir / "original.png").convert("RGB")

    want = ([int(x) for x in args.steps.split(",") if x != ""] if args.steps
            else list(range(len(meta["steps"]))))
    bad = [s for s in want if not 0 <= s < maps.shape[0]]
    if bad:
        raise SystemExit(f"step(s) {bad} outside 0..{maps.shape[0] - 1}")

    boxes_for = {}
    if args.boxes:
        blob = json.loads(Path(args.boxes).read_text())
        for s in blob["steps"]:
            if s["sample"] == args.sample and s["model"] == args.model:
                boxes_for[s["step"]] = s.get("tight_boxes") or []

    out = Path(args.out)
    (out / "parts").mkdir(parents=True, exist_ok=True)
    img.save(out / "parts" / "input.png")

    size = (int(img.size[0] * args.scale), int(img.size[1] * args.scale))
    big = img.resize(size, Image.LANCZOS)
    pad = max(4, int(4 * args.scale))
    f_title = font(max(11, int(8 * args.scale)))
    f_body = font(max(10, int(7 * args.scale)))
    line_h = int(f_title.size * 1.35) if hasattr(f_title, "size") else 14

    panels = [captioned(big, "input", wrap(f"gold answer: {meta.get('gt_answer')}",
                                           f_body, size[0] - 2 * pad),
                        f_title, f_body, pad, line_h)]
    for si in want:
        step = meta["steps"][si]
        ov = overlay(img, maps[si], cmap, args)
        if boxes_for.get(si):
            ov = draw_boxes(ov, boxes_for[si], args.box_colour,
                            width=max(1, int(args.scale)))
        ov.save(out / "parts" / f"step{si:02d}.png")
        body = wrap(step["text"], f_body, size[0] - 2 * pad)[: args.caption_lines]
        panels.append(captioned(ov.resize(size, Image.LANCZOS),
                                f"step {si}  ({step['n_tokens']} tokens)",
                                body, f_title, f_body, pad, line_h))

    # every panel the same height, so the rows line up even where a caption is shorter
    ph = max(p.size[1] for p in panels)
    panels = [p if p.size[1] == ph else
              (lambda c: (c.paste(p, (0, 0)), c)[1])(Image.new("RGB", (p.size[0], ph), BG))
              for p in panels]

    cols = args.cols if args.cols > 0 else len(panels)
    rows = [panels[i:i + cols] for i in range(0, len(panels), cols)]

    q = str(meta.get("question", "")).strip()
    shown_q = (args.question or q).strip()
    body_w = max(sum(p.size[0] for p in r) + pad * (len(r) + 1) for r in rows)
    header_lines = []
    for para in shown_q.splitlines():
        header_lines += wrap(para, f_title, body_w - 2 * pad) or [""]
    header = text_block(header_lines, f_title, body_w, pad, line_h)

    total_h = header.size[1] + sum(ph + pad for _ in rows) + pad
    sheet = Image.new("RGB", (body_w, total_h), BG)
    sheet.paste(header, (0, 0))
    y = header.size[1]
    for r in rows:
        x = pad
        for p in r:
            sheet.paste(p, (x, y))
            x += p.size[0] + pad
        y += ph + pad
    sheet.save(out / "figure.png")

    e = html.escape
    page = [
        "<!doctype html><meta charset='utf-8'><title>chain figure</title>",
        "<style>body{background:#111;color:#ddd;font:13px/1.55 -apple-system,sans-serif;"
        "margin:24px;max-width:1300px}img{max-width:100%;border:1px solid #333}"
        "pre{white-space:pre-wrap;background:#181818;padding:8px;border-radius:4px}</style>",
        f"<h1>{e(args.sample)} &mdash; {e(str(meta.get('dataset')))} "
        f"&mdash; {e(args.model)}, {e(args.map)}</h1>",
        (f"<p><b>header shown in the figure:</b> {e(shown_q)}</p>" if args.question else ""),
        f"<p><b>the prompt the model was given:</b></p><pre>{e(q)}</pre>"
        f"<p><b>gold:</b> {e(str(meta.get('gt_answer')))}</p>",
        "<p><img src='figure.png'></p>",
        f"<h2>the chain</h2><pre>{e(meta.get('generation', ''))}</pre>",
    ]
    (out / "figure.html").write_text("\n".join(page))
    print(f"[out] {out/'figure.png'}  ({len(want)} step(s), {len(rows)} row(s))")
    print(f"[out] {out/'figure.html'}\n[out] {out/'parts'}/")


if __name__ == "__main__":
    main()
