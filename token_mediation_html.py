#!/usr/bin/env python
"""Render the token-mediation result as one self-contained HTML page.

    python token_mediation_html.py --map outputs/token_mediation/map \
        --arms outputs/token_mediation/norm --out docs/token-mediation.html

Everything is inlined: no CDN, no fonts, no build step. The eval nodes run with
`HF_HUB_OFFLINE=1` and the page has to open on a laptop with no network, so a script tag
pointing at a chart library would be a page that works on exactly one machine. Same rule
`sink_location_html.py` follows, and for the same reason.

The numbers are not retyped. This reads the same `.npy` and JSONL the probe wrote and
calls the probe's own paired-contrast functions, so the page and the run cannot drift.
"""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# palette -- the dataviz reference instance, roles only
# ---------------------------------------------------------------------------
#: Diverging blue <-> red with a neutral grey midpoint. The maps are POLARITY data with a
#: real zero ("this square matters more / less than this picture's average square"), which
#: is the one job a diverging scale is for. Never a rainbow, never a hue at the midpoint.
DIVERGING = {
    "light": ["#184f95", "#2a78d6", "#86b6ef", "#cde2fb", "#f0efec",
              "#f7cfc9", "#ef9d93", "#de5f56", "#a82a24"],
    "dark": ["#184f95", "#2a78d6", "#5598e7", "#86b6ef", "#383835",
             "#c98d86", "#d97068", "#cf4a42", "#8c221d"],
}
#: Categorical slots 1-3. Validated all-pairs in both modes; aqua is sub-3:1 on the light
#: surface, so every bar carries a visible label (the relief rule).
SERIES = {"light": ["#2a78d6", "#eb6834", "#1baf7a"],
          "dark": ["#3987e5", "#d95926", "#199e70"]}

CLIP = 1.0            # the diverging scale's end; cells past it keep their printed value


def ramp_color(z, mode="light"):
    if not np.isfinite(z):
        return "transparent"
    steps = DIVERGING[mode]
    t = (max(-CLIP, min(CLIP, float(z))) + CLIP) / (2 * CLIP)     # 0..1
    i = int(round(t * (len(steps) - 1)))
    return steps[i]


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def heatmap_svg(m, label, cell=27, pad=30):
    """One canonical map as an SVG grid. Colour + hover title + selective labels.

    A number on all 144 cells would be noise, so only cells at |z| >= 0.5 are labelled
    directly; the rest carry their value in a hover tooltip, and the table view below the
    figure has every one of them.
    """
    g = m.shape[0]
    w = h = g * cell + pad
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{html.escape(label)}" '
           f'class="hm">']
    out.append(f'<text x="{pad}" y="12" class="hm-ax">left edge of picture &#8594;</text>')
    out.append(f'<text x="10" y="{pad + 4}" class="hm-ax" '
               f'transform="rotate(-90 10 {pad + 4})">top &#8594; bottom</text>')
    for r in range(g):
        for c in range(g):
            v = m[r, c]
            x, y = pad + c * cell, pad + r * cell
            tip = f"row {r}, column {c}: {v:+.2f}" if np.isfinite(v) else "no data"
            out.append(
                f'<rect x="{x}" y="{y}" width="{cell - 2}" height="{cell - 2}" rx="2" '
                f'fill="{ramp_color(v)}" class="cell" data-v="{v:+.3f}">'
                f'<title>{tip}</title></rect>')
            if np.isfinite(v) and abs(v) >= 0.5:
                ink = "#fcfcfb" if abs(v) > 0.62 else "#0b0b0b"
                out.append(
                    f'<text x="{x + (cell - 2) / 2}" y="{y + (cell - 2) / 2 + 3.5}" '
                    f'class="hm-lab" fill="{ink}">{v:+.1f}</text>')
    out.append("</svg>")
    return "".join(out)


def colorbar_svg(w=260, h=14):
    steps = DIVERGING["light"]
    seg = w / len(steps)
    out = [f'<svg viewBox="0 0 {w} {h + 18}" class="cbar" role="img" '
           f'aria-label="colour scale from minus one to plus one">']
    for i, c in enumerate(steps):
        out.append(f'<rect x="{i * seg:.1f}" y="0" width="{seg + 0.6:.1f}" height="{h}" '
                   f'fill="{c}"/>')
    out.append(f'<text x="0" y="{h + 13}" class="cb-t">&#8722;1.0 less surprise</text>'
               f'<text x="{w / 2}" y="{h + 13}" class="cb-t" text-anchor="middle">0</text>'
               f'<text x="{w}" y="{h + 13}" class="cb-t" text-anchor="end">'
               f'+1.0 more surprise</text></svg>')
    return "".join(out)


def bars_svg(rows, width=680, row_h=40, left=210):
    """Paired contrasts as horizontal bars with a 95% interval. One value scale."""
    lo = min(min(r["ci"][0], 0) for r in rows)
    hi = max(max(r["ci"][1], 0) for r in rows)
    span = (hi - lo) or 1.0
    lo, hi = lo - 0.06 * span, hi + 0.06 * span
    plot = width - left - 120   # room for the value label past the whisker
    h = len(rows) * row_h + 44

    def X(v):
        return left + (v - lo) / (hi - lo) * plot

    out = [f'<svg viewBox="0 0 {width} {h}" role="img" class="bars" '
           f'aria-label="paired contrasts with 95 percent intervals">']
    for gv in np.linspace(lo, hi, 5):
        out.append(f'<line x1="{X(gv):.1f}" y1="26" x2="{X(gv):.1f}" y2="{h - 18}" '
                   f'class="grid"/>')
        out.append(f'<text x="{X(gv):.1f}" y="{h - 5}" class="tick" '
                   f'text-anchor="middle">{gv:+.2f}</text>')
    out.append(f'<line x1="{X(0):.1f}" y1="26" x2="{X(0):.1f}" y2="{h - 18}" '
               f'class="zero"/>')
    for i, r in enumerate(rows):
        y = 30 + i * row_h + row_h / 2
        col = SERIES["light"][r["slot"]]
        x0, x1 = X(min(0, r["med"])), X(max(0, r["med"]))
        out.append(f'<rect x="{x0:.1f}" y="{y - 7}" width="{max(x1 - x0, 2):.1f}" '
                   f'height="14" rx="4" fill="{col}"/>')
        out.append(f'<line x1="{X(r["ci"][0]):.1f}" y1="{y}" x2="{X(r["ci"][1]):.1f}" '
                   f'y2="{y}" class="whisk"/>')
        for e in r["ci"]:
            out.append(f'<line x1="{X(e):.1f}" y1="{y - 5}" x2="{X(e):.1f}" '
                       f'y2="{y + 5}" class="whisk"/>')
        out.append(f'<text x="{left - 10}" y="{y + 4}" class="blab" '
                   f'text-anchor="end">{html.escape(r["name"])}</text>')
        out.append(f'<text x="{X(max(r["ci"][1], 0)) + 8:.1f}" y="{y + 4}" '
                   f'class="bval">{r["med"]:+.2f} &#160;({r["times"]})</text>')
    out.append("</svg>")
    return "".join(out)


def pipeline_svg():
    """Where each of the two edits happens. Not a chart -- a diagram of the method."""
    return """
<svg viewBox="0 0 700 210" role="img" class="pipe"
     aria-label="the picture becomes squares, the encoder turns each into a vector,
     the language model reads them; edit A changes a vector, edit B changes pixels">
  <rect x="14" y="58" width="96" height="96" rx="6" class="pbox"/>
  <text x="62" y="112" class="pt" text-anchor="middle">picture</text>
  <text x="62" y="172" class="ps" text-anchor="middle">cut into</text>
  <text x="62" y="186" class="ps" text-anchor="middle">32&#215;32 squares</text>

  <path d="M118 106 H176" class="parr"/>
  <rect x="180" y="58" width="120" height="96" rx="6" class="pbox"/>
  <text x="240" y="100" class="pt" text-anchor="middle">vision</text>
  <text x="240" y="118" class="pt" text-anchor="middle">encoder</text>
  <text x="240" y="172" class="ps" text-anchor="middle">one vector</text>
  <text x="240" y="186" class="ps" text-anchor="middle">per square</text>

  <path d="M308 106 H366" class="parr"/>
  <rect x="370" y="58" width="130" height="96" rx="6" class="pbox"/>
  <text x="435" y="100" class="pt" text-anchor="middle">language</text>
  <text x="435" y="118" class="pt" text-anchor="middle">model</text>
  <text x="435" y="172" class="ps" text-anchor="middle">reads the vectors,</text>
  <text x="435" y="186" class="ps" text-anchor="middle">writes an answer</text>

  <path d="M508 106 H566" class="parr"/>
  <rect x="570" y="58" width="116" height="96" rx="6" class="pbox"/>
  <text x="628" y="100" class="pt" text-anchor="middle">what it</text>
  <text x="628" y="118" class="pt" text-anchor="middle">predicts</text>

  <path d="M62 52 V30 H240 V52" class="pedge b"/>
  <text x="151" y="22" class="pe b" text-anchor="middle">
    B &#8212; paint over the pixels (before the encoder)</text>
  <path d="M240 52 V44" class="pedge a" style="display:none"/>
  <circle cx="330" cy="106" r="15" class="pdot a"/>
  <text x="330" y="111" class="pdotlab" text-anchor="middle">A</text>
  <text x="330" y="40" class="pe a" text-anchor="middle">A &#8212; swap the vector</text>
  <text x="330" y="24" class="pe a" text-anchor="middle">(after the encoder)</text>
  <path d="M330 48 V88" class="pedge a"/>
</svg>"""


# ---------------------------------------------------------------------------
def build(args):
    TM = _load("_tm_probe", "token_mediation_probe.py")
    mdir, adir = Path(args.map), Path(args.arms)

    maps = {}
    for v in ("swap", "swapn", "pix", "row_norm"):
        p = mdir / f"map_{v}_{args.canon}x{args.canon}.npy"
        if p.exists():
            maps[v] = np.load(p)
    if "swapn" not in maps:
        raise SystemExit(f"no maps under {mdir}")
    maps["register"] = maps["swapn"] - maps["pix"]

    mrows = [json.loads(l) for p in sorted(mdir.glob("map_shard*.jsonl"))
             for l in p.read_text().splitlines() if l.strip()]
    arows = TM.read_rows(adir)

    # contrasts, computed by the probe's own functions
    def con(a, b, name, slot):
        x, y = TM._paired(arows, a, b, "kl_logmean")
        s = TM._wilcoxon(x, y)
        return {"name": name, "slot": slot, "med": s["median_diff"], "ci": s["ci"],
                "n": s["n"], "times": f"{math.exp(s['median_diff']):.2f}&#215;",
                "frac": s["frac_gt"]}

    bars = [con("swapn_peak", "swapn_rand", "swap, size held equal", 0),
            con("swap_peak", "swap_rand", "swap, as-is", 1),
            con("pix_peak", "pix_rand", "paint over the pixels", 2)]

    # how much of an outlier the corner is, in each map
    def outlier(m):
        rest = np.delete(m.ravel(), 0)
        return m[0, 0], rest.mean(), rest.std(), (m[0, 0] - rest.mean()) / rest.std()

    g = args.canon
    ring = np.zeros((g, g), bool)
    ring[0], ring[-1], ring[:, 0], ring[:, -1] = True, True, True, True

    def spear(a, b):
        ra, rb = np.argsort(np.argsort(a.ravel())), np.argsort(np.argsort(b.ravel()))
        return float(np.corrcoef(ra, rb)[0, 1])

    n_pic, n_cells = len(mrows), sum(r["n_cells"] for r in mrows)
    rho = spear(maps["swapn"], maps["pix"])
    keep = np.arange(g * g) != 0
    rho_nc = spear(maps["swapn"].ravel()[keep], maps["pix"].ravel()[keep])

    MAPTXT = {
        "swapn": ("Swapping the vector (size held equal)",
                  "How much the model changes when one square's vector is replaced by "
                  "another picture's, pushed the same distance everywhere."),
        "pix": ("Painting over the pixels",
                "How much the model changes when one square's pixels are replaced by "
                "flat colour and the encoder is run again."),
        "swap": ("Swapping the vector (as-is)",
                 "The same swap without holding the distance equal. Part of what you see "
                 "here is just that some vectors are longer than others."),
        "row_norm": ("How long each vector is",
                     "Not an experiment &mdash; the plain length of the encoder's output "
                     "at each square. Shown because it peaks in the same corner."),
        "register": ("The difference: vector minus pixels",
                     "The first map minus the second. Positive means the square's vector "
                     "matters more to the model than the pixels underneath it do."),
    }

    def table(m):
        rows = "".join(
            "<tr>" + "".join(f"<td>{m[r, c]:+.2f}</td>" for c in range(g)) + "</tr>"
            for r in range(g))
        return (f'<details><summary>every number in this map</summary>'
                f'<table class="grid">{rows}</table></details>')

    figs = []
    for key in ("swapn", "pix", "register", "swap", "row_norm"):
        if key not in maps:
            continue
        m = maps[key]
        t, d = MAPTXT[key]
        v00, mu, sd, z = outlier(m)
        figs.append(f"""
<figure class="mapfig">
  <figcaption><b>{t}</b><br><span class="cap">{d}</span></figcaption>
  {heatmap_svg(m, t)}
  <p class="note">Top-left square: <b>{v00:+.2f}</b>. Every other square averages
     {mu:+.2f} (spread {sd:.2f}), so the corner sits <b>{z:+.1f}</b> spreads away.
     Outer frame {np.nanmean(m[ring]):+.2f}, middle {np.nanmean(m[~ring]):+.2f}.</p>
  {table(m)}
</figure>""")

    css = """
:root{--s1:#fcfcfb;--plane:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--mut:#898781;
--grid:#e1e0d9;--base:#c3c2b7;--ring:rgba(11,11,11,.10);--acc:#2a78d6}
@media (prefers-color-scheme:dark){:root{--s1:#1a1a19;--plane:#0d0d0d;--ink:#fff;
--ink2:#c3c2b7;--mut:#898781;--grid:#2c2c2a;--base:#383835;--ring:rgba(255,255,255,.10);
--acc:#3987e5}}
*{box-sizing:border-box}
body{margin:0;background:var(--plane);color:var(--ink);
font:15px/1.65 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:860px;margin:0 auto;padding:40px 22px 80px}
h1{font-size:30px;line-height:1.25;margin:0 0 6px}
h2{font-size:21px;margin:44px 0 10px;padding-top:14px;border-top:1px solid var(--grid)}
h3{font-size:16px;margin:26px 0 6px}
p{margin:10px 0}
.sub{color:var(--ink2);font-size:15px;margin:0 0 4px}
.meta{color:var(--mut);font-size:13px;margin:2px 0 0}
.card{background:var(--s1);border:1px solid var(--ring);border-radius:10px;
padding:16px 18px;margin:16px 0}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;
margin:18px 0}
.tile{background:var(--s1);border:1px solid var(--ring);border-radius:10px;padding:14px}
.tile .v{font-size:27px;font-weight:650;letter-spacing:-.01em}
.tile .k{color:var(--ink2);font-size:13px;margin-top:3px}
dl.gloss{margin:6px 0 0}
dl.gloss dt{font-weight:650;margin-top:11px}
dl.gloss dd{margin:2px 0 0;color:var(--ink2)}
figure{margin:22px 0}
figcaption{margin-bottom:9px}
.cap,.note{color:var(--ink2);font-size:13.5px}
.note{margin:8px 0 0}
svg{max-width:100%;height:auto;display:block}
.hm{width:100%;max-width:390px;background:var(--s1);border-radius:8px}
.hm .cell{stroke:var(--s1);stroke-width:1}
.hm-lab{font:600 9.5px system-ui;pointer-events:none}
.hm-ax,.cb-t,.tick{fill:var(--mut);font:11px system-ui}
.cbar{width:280px;margin:4px 0 14px}
.bars{width:100%}
.grid{stroke:var(--grid);stroke-width:1}
.zero{stroke:var(--base);stroke-width:1.5}
.whisk{stroke:var(--ink2);stroke-width:2}
.blab{fill:var(--ink);font:13px system-ui}
.bval{fill:var(--ink2);font:600 12.5px system-ui}
.pipe{width:100%;background:var(--s1);border-radius:8px;padding:6px 0}
.pbox{fill:none;stroke:var(--base);stroke-width:1.5}
.pt{fill:var(--ink);font:600 13px system-ui}
.ps{fill:var(--mut);font:11px system-ui}
.parr{stroke:var(--base);stroke-width:1.5;fill:none;marker-end:url(#ar)}
.pedge{fill:none;stroke-width:1.8;stroke-dasharray:4 3}
.pedge.a{stroke:#eb6834}.pedge.b{stroke:#2a78d6}
.pe{font:600 11.5px system-ui}.pe.a{fill:#eb6834}.pe.b{fill:#2a78d6}
.pdot{fill:#eb6834}.pdotlab{fill:#fff;font:700 12px system-ui}
details{margin-top:10px}
summary{cursor:pointer;color:var(--acc);font-size:13px}
table.grid{border-collapse:collapse;margin-top:8px;font:11px/1.3 system-ui;
font-variant-numeric:tabular-nums}
table.grid td{border:1px solid var(--grid);padding:2px 4px;text-align:right;
color:var(--ink2)}
table.kv{border-collapse:collapse;width:100%;font-size:13.5px;margin-top:8px}
table.kv th,table.kv td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--grid)}
table.kv th{color:var(--ink2);font-weight:600}
code{background:var(--s1);border:1px solid var(--ring);border-radius:4px;padding:1px 5px;
font:12.5px ui-monospace,SFMono-Regular,Menlo,monospace}
pre{background:var(--s1);border:1px solid var(--ring);border-radius:8px;padding:12px 14px;
overflow-x:auto;font:12.5px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--ink2)}
ul{margin:8px 0;padding-left:20px}li{margin:5px 0}
.warn{border-left:3px solid #eb6834;padding-left:13px;margin:16px 0}
"""

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>The first visual token is a register</title>
<style>{css}</style></head>
<body><main>
<svg width="0" height="0" style="position:absolute"><defs>
<marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6"
 orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="var(--base)"/></marker></defs></svg>

<h1>The first visual token is a register</h1>
<p class="sub">Qwen3-VL 8B looks hardest at the top-left square of every picture &mdash;
but the pixels in that square barely matter. The vector sitting there carries something
that did not come from underneath it.</p>
<p class="meta">Run 2026-09-20/21 &middot; model
<code>coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged</code> &middot;
{n_pic} pictures and {n_cells:,} squares for the maps, {len(arows)} pictures for the
named-square tests &middot; branch <code>probe/token-mediation</code></p>

<h2>Words used on this page</h2>
<div class="card"><dl class="gloss">
<dt>square (or patch)</dt><dd>The picture is cut into a grid of 32&times;32-pixel
squares. One square is the smallest piece the model sees separately.</dd>
<dt>vector</dt><dd>The vision encoder turns each square into a list of numbers. That list
is what the language model actually reads &mdash; it never sees pixels.</dd>
<dt>chain</dt><dd>The text the model writes before its answer.</dd>
<dt>surprise</dt><dd>Our measure of effect. We fix the chain the model already wrote,
change one thing, and measure how much its word-by-word predictions move. If we change
nothing, surprise is exactly zero &mdash; not nearly zero, exactly. That is what makes
small effects readable.</dd>
<dt>spreads away</dt><dd>How unusual a square is compared with the other squares of the
same picture, counted in standard deviations. Bigger means more unusual.</dd>
<dt>donor</dt><dd>A different picture, the same size, whose vectors we borrow.</dd>
<dt>attention</dt><dd>As the model writes, each word it produces draws on the vectors,
some more than others. How much it draws on one vector is that vector's attention. "Looks
hardest at" means "draws most on".</dd>
<dt>register</dt><dd>A slot that holds a summary the encoder wrote, rather than a
description of the square underneath it. The thing this page argues the top-left square
is.</dd>
</dl></div>

<h2>The question</h2>
<p>An earlier experiment found that this model pours attention onto the top-left of every
picture, and that the pull travels with the vector rather than with the position. That
said <i>where</i> the mark lives. It did not say whether the mark has anything to do with
the pixels in that corner.</p>
<p>So we damaged one square in two different places and compared.</p>
{pipeline_svg()}
<p><b>A &mdash; swap the vector.</b> Run the encoder on the real picture, then replace one
square's vector with the vector a donor picture produced for the same slot. The picture is
untouched; the square is still there to be looked at; only what it says has changed.</p>
<p><b>B &mdash; paint over the pixels.</b> Replace that square's 32&times;32 pixels with
flat colour, then run the encoder again. Now the damage happens before the encoder, so
every vector it produces may shift.</p>
<p>If a square's vector really carries what is underneath it, A and B should agree. Where
they disagree, the vector is carrying something else.</p>

<h2>Result 1 &mdash; three named squares</h2>
<p>First we tested only three squares per picture: the one the model looks at hardest, the
top-left one, and a random one. The model looks hardest at the top-left square on
<b>{100 * np.mean([r["peak"] == 0 for r in arows]):.0f}%</b> of pictures, so the first two
are usually the same square.</p>
<p>Each bar is the named square minus a random square, on the same picture. Right of the
line means the named square matters more.</p>
{bars_svg(bars)}
<p class="note">Bars are medians over {bars[0]['n']} pictures; whiskers are 95% intervals.
The multiplier in brackets is how many times more the model moves.</p>
<div class="tiles">
  <div class="tile"><div class="v">{bars[0]['times']}</div>
    <div class="k">more surprise from swapping the top square's vector than a random
    one's, with size held equal</div></div>
  <div class="tile"><div class="v">{bars[2]['med']:+.2f}</div>
    <div class="k">effect of painting over that same square's pixels &mdash; below zero,
    so it matters <i>less</i> than a random square</div></div>
  <div class="tile"><div class="v">{100 * bars[0]['frac']:.0f}%</div>
    <div class="k">of pictures where the top square wins, one picture at a time</div></div>
</div>
<p>So the vector matters and the pixels do not. Note the second row: the size of the
change was part of the story. The top square's vector is about twice as long as a normal
one, so swapping it pushes harder for a trivial reason. Holding the push equal shrinks the
effect from {bars[1]['times']} to {bars[0]['times']} &mdash; but does not remove it.</p>

<h2>Result 2 &mdash; every square</h2>
<p>Then we did the same for <b>every</b> square of every picture: {n_cells:,} squares over
{n_pic} pictures. Pictures come in {len({tuple(r["grid"]) for r in mrows})} different grid
shapes here, so each picture's result is laid onto one shared {g}&times;{g} grid by area,
and each picture is scored against its own average square before pooling. Otherwise the
map would show which pictures are fragile rather than which positions matter.</p>
{colorbar_svg()}
<p class="note">Colour is capped at &plusmn;{CLIP:.1f}; squares past the cap keep their
printed number. Hover any square for its value.</p>
{"".join(figs)}

<h2>What the maps say</h2>
<div class="card">
<p><b>It is one square, not the frame.</b> In the swap map the outer frame is ordinary
({np.nanmean(maps['swapn'][ring]):+.2f}) and so is the middle
({np.nanmean(maps['swapn'][~ring]):+.2f}). Only the top-left corner stands out, at
<b>{outlier(maps['swapn'])[3]:+.1f} spreads</b> above every other square. The next
highest square anywhere is {np.sort(np.delete(maps['swapn'].ravel(), 0))[-1]:+.2f}.</p>
<p><b>The two maps otherwise agree.</b> Rank them against each other and they line up at
<b>{rho:+.2f}</b> ({rho_nc:+.2f} with the corner left out). Squares whose pixels matter are
generally squares whose vectors matter &mdash; which is what you would hope. The top-left
corner is the single exception to an otherwise sensible relationship.</p>
<p><b>The pixels that matter are in the middle.</b> The painting-over map peaks inside the
picture and is weakest around the edge &mdash; frame
{np.nanmean(maps['pix'][ring]):+.2f} against middle
{np.nanmean(maps['pix'][~ring]):+.2f}. That is ordinary: photographs put their content in
the middle. It is also a check that the measurement reads real content.</p>
</div>

<h2>What this does not show</h2>
<div class="warn">
<ul>
<li><b>The effect is small in absolute terms.</b> One square out of ~160 barely moves the
model at all. Everything above is a comparison between squares, not a claim that one
square changes the answer.</li>
<li><b>Length and position are tangled.</b> The vector-length map peaks in the same
corner. We held the push equal and the effect survived, but nothing here moves the
register somewhere else, which is what would separate the two properly.</li>
<li><b>We did not block attention.</b> The published version of this kind of experiment
cuts the attention to a token instead of replacing it. On a square that is soaking up
attention, cutting it also re-spreads everyone else's attention, so a big effect would not
be readable. We replaced content instead. That arm is written and switched off: this model will not
accept the shape of mask it needs.</li>
<li><b>One model, one checkpoint.</b> Nothing here says whether other models do this.</li>
</ul>
</div>

<h2>How it was run</h2>
<pre>python token_mediation_probe.py selftest --model MODEL      # gate: must pass first
python token_mediation_probe.py run  --model MODEL --out OUT  # three named squares
python token_mediation_probe.py map  --model MODEL --out OUT  # every square
python token_mediation_probe.py mapreport --out OUT --canon {g}
python token_mediation_html.py --map OUT --arms ARMS --out docs/token-mediation.html</pre>
<table class="kv">
<tr><th>named-square data</th><td><code>{adir}</code></td></tr>
<tr><th>map data</th><td><code>{mdir}</code> (<code>map_*.npy</code> are the pooled
grids)</td></tr>
<tr><th>pictures</th><td>the 12-type sink-location corpus, 150 per type, the same one the
attention result used</td></tr>
<tr><th>cost</th><td>the map was {n_pic} pictures in about 70 minutes on 4 GPUs</td></tr>
</table>
<p class="note">The selftest is a gate, not a formality: it checks that changing nothing
gives exactly zero surprise, that swapping a square with its own vector changes nothing,
and that the equal-size swap reproduces the plain one. It caught two real bugs during this
work, each of which would have looked like a weak result rather than an error.</p>
</main></body></html>"""

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc)
    print(f"wrote {out}  ({len(doc):,} bytes, {n_pic} pictures, {n_cells:,} squares)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", default="outputs/token_mediation/map")
    ap.add_argument("--arms", default="outputs/token_mediation/norm")
    ap.add_argument("--canon", type=int, default=12)
    ap.add_argument("--out", default="docs/token-mediation.html")
    return build(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
