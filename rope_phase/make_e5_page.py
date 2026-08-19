#!/usr/bin/env python
"""Build a single self-contained HTML page from an E5 run.

The images are the ones the model actually saw -- squashed to a square by
`square_image`, not the originals -- because the whole point is what the model was
looking at when it wrote the text beside it.  They are embedded as base64 so the
page is one file that can be opened or sent anywhere.

    python make_e5_page.py --out-dir outputs/e5 --dataset PATH --html e5.html
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rope_phase_probe as RP      # noqa: E402

BASE, TREAT = "none", "frozen:24"


def terminal_period(toks, window=60):
    """Smallest k such that the last `window` tokens repeat with period k."""
    t = toks[-window:]
    if len(t) < 8:
        return None
    for k in range(1, len(t) // 3 + 1):
        if all(t[i] == t[i - k] for i in range(k, len(t))):
            return k
    return None


def loop_start(toks, k):
    i = len(toks) - 1
    while i - k >= 0 and toks[i] == toks[i - k]:
        i -= 1
    return i + 1


def classify(case, cap):
    """The failure types the run actually produced, in severity order."""
    a = case["arms"][TREAT]
    k = terminal_period(a["tokens"])
    if k == 1:
        return "lock", "single-token lock", k
    if k in (2, 3):
        return "fragment", "connective fragment", k
    if k is not None:
        return "clause", "clause loop", k
    if a["n_tokens"] >= cap:
        return "drift", "near-repetition, no clean period", None
    return "ok", "terminated", None


ORDER = ["lock", "fragment", "clause", "drift", "ok"]
LABEL = {"lock": "Single-token lock", "fragment": "Connective fragment",
         "clause": "Clause loop", "drift": "Near-repetition, no clean period",
         "ok": "Terminated normally"}
BLURB = {
    "lock": "The model latches onto one token and emits it to the cap.",
    "fragment": "A two- or three-token grammatical fragment that never resolves.",
    "clause": "A whole clause cycles.",
    "drift": "Repetitive but drifting just enough to defeat exact periodicity.",
    "ok": "Reached an end-of-sequence token. Degraded rather than broken.",
}


def img_tag(image, side: int, px: int) -> str:
    im = RP.square_image(image, side).resize((px, px), 2)
    buf = io.BytesIO()
    im.convert("RGB").save(buf, format="JPEG", quality=80, optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def esc(s):
    return html.escape(s or "")


def bars(rows, unit=""):
    """A horizontal categorical bar chart. One series, so no legend -- the title
    names it -- and every bar is directly labelled, so nothing rests on colour."""
    top = max((v for _, v in rows), default=1) or 1
    out = []
    for name, v in rows:
        out.append(
            f'<div class="bar-row"><div class="bar-lab">{esc(name)}</div>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{100*v/top:.1f}%"></div></div>'
            f'<div class="bar-val">{v}{unit}</div></div>')
    return "".join(out)


def build(args):
    out_dir = Path(args.out_dir)
    data = json.loads((out_dir / "gen.json").read_text())
    cases, cap = data["cases"], data["max_new_tokens"]
    rows = {r["row_index"]: r for r in RP.load_samples(args.dataset, args.pool, 0)}

    groups = defaultdict(list)
    for c in cases:
        kind, _lab, k = classify(c, cap)
        groups[kind].append((c, k))

    n = len(cases)
    n_cap = sum(c["arms"][TREAT]["n_tokens"] >= cap for c in cases)
    n_loop = sum(terminal_period(c["arms"][TREAT]["tokens"]) is not None for c in cases)
    sub = [c for c in cases if c["single_noun"] and c["reference"]]

    def match(arm, sel):
        import re
        def norm(s):
            return " ".join("".join(ch.lower() if ch.isalnum() or ch.isspace() else " "
                                    for ch in (s or "")).split())
        return sum(norm(c["reference"]) in norm(c["arms"][arm]["text"]) for c in sel)

    arms = data["arms"]
    match_rows = [(a, round(100 * match(a, sub) / max(len(sub), 1))) for a in arms]
    period_counts = Counter()
    for c in cases:
        k = terminal_period(c["arms"][TREAT]["tokens"])
        period_counts[k if k else "none"] += 1
    period_rows = [(f"period {k}" if k != "none" else "no loop", period_counts[k])
                   for k in sorted(period_counts, key=lambda x: (x == "none", x))]

    cards = []
    for kind in ORDER:
        items = groups.get(kind, [])
        if not items:
            continue
        cards.append(f'<h2 class="grp">{LABEL[kind]} '
                     f'<span class="grp-n">{len(items)} of {n}</span></h2>'
                     f'<p class="grp-blurb">{BLURB[kind]}</p>')
        for c, k in items:
            row = rows.get(c["row_index"])
            b64 = img_tag(row["image"], args.image_side, args.px) if row else ""
            a, t = c["arms"][BASE], c["arms"][TREAT]
            badge = ""
            if k:
                st = loop_start(t["tokens"], k)
                reps = (len(t["tokens"]) - st) // k
                badge = (f'<span class="badge">repeating unit x{reps} '
                         f'from token {st}</span>')
            capped = ('<span class="badge crit">&#9888; never terminated</span>'
                      if t["n_tokens"] >= cap else "")
            cards.append(f'''<article class="card">
  <div class="thumb">{'<img alt="the image the model saw" src="data:image/jpeg;base64,'+b64+'">' if b64 else ''}
    <div class="src">{esc(c["dataset"])}</div></div>
  <div class="body">
    <div class="q">{esc(c["question"])}</div>
    <div class="ref">reference answer: <b>{esc(c["reference"])}</b></div>
    <div class="cols">
      <section class="pane p-base"><header><span class="dot d1"></span>none
        <span class="tok">{a["n_tokens"]} tokens</span></header>
        <div class="txt">{esc(a["text"])}</div></section>
      <section class="pane p-treat"><header><span class="dot d2"></span>frozen:24
        <span class="tok">{t["n_tokens"]} tokens</span>{capped}{badge}</header>
        <div class="txt">{esc(t["text"])}</div></section>
    </div>
  </div>
</article>''')

    page = f'''<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>E5 - freezing the cross-modal position during generation</title>
<style>
  .viz-root {{ color-scheme: light;
    --surface-1:#fcfcfb; --plane:#f9f9f7; --ink:#0b0b0b; --ink-2:#52514e;
    --muted:#898781; --grid:#e1e0d9; --rule:rgba(11,11,11,0.10);
    --s1:#2a78d6; --s2:#eb6834; --crit:#d03b3b; }}
  @media (prefers-color-scheme: dark) {{ :root:where(:not([data-theme="light"])) .viz-root {{
    color-scheme: dark;
    --surface-1:#1a1a19; --plane:#0d0d0d; --ink:#fff; --ink-2:#c3c2b7;
    --muted:#898781; --grid:#2c2c2a; --rule:rgba(255,255,255,0.10);
    --s1:#3987e5; --s2:#d95926; --crit:#d03b3b; }} }}
  :root[data-theme="dark"] .viz-root {{ color-scheme: dark;
    --surface-1:#1a1a19; --plane:#0d0d0d; --ink:#fff; --ink-2:#c3c2b7;
    --muted:#898781; --grid:#2c2c2a; --rule:rgba(255,255,255,0.10);
    --s1:#3987e5; --s2:#d95926; --crit:#d03b3b; }}
  * {{ box-sizing:border-box }}
  body {{ margin:0; background:var(--plane); color:var(--ink);
    font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif; }}
  .wrap {{ max-width:1180px; margin:0 auto; padding:40px 22px 80px }}
  h1 {{ font-size:26px; margin:0 0 6px; letter-spacing:-.01em }}
  .sub {{ color:var(--ink-2); margin:0 0 4px }}
  .meta {{ color:var(--muted); font-size:13px; margin:0 0 28px }}
  .tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr));
    gap:12px; margin-bottom:28px }}
  .tile {{ background:var(--surface-1); border:1px solid var(--rule);
    border-radius:10px; padding:16px 18px }}
  .tile .v {{ font-size:32px; font-weight:640; letter-spacing:-.02em; line-height:1.1 }}
  .tile .k {{ color:var(--ink-2); font-size:13px; margin-top:4px }}
  .tile .n {{ color:var(--muted); font-size:12px; margin-top:2px }}
  .tile.bad .v {{ color:var(--crit) }}
  .charts {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:34px }}
  @media (max-width:820px) {{ .charts {{ grid-template-columns:1fr }} }}
  .panel {{ background:var(--surface-1); border:1px solid var(--rule);
    border-radius:10px; padding:16px 18px }}
  .panel h3 {{ margin:0 0 2px; font-size:14px }}
  .panel .cap {{ color:var(--muted); font-size:12px; margin:0 0 14px }}
  .bar-row {{ display:grid; grid-template-columns:96px 1fr 46px; align-items:center;
    gap:10px; margin-bottom:7px }}
  .bar-lab {{ font-size:12px; color:var(--ink-2); text-align:right }}
  .bar-track {{ background:var(--grid); border-radius:4px; height:14px }}
  .bar-fill {{ background:var(--s1); height:14px; border-radius:0 4px 4px 0 }}
  .bar-val {{ font-size:12px; color:var(--ink-2); font-variant-numeric:tabular-nums }}
  details {{ margin-top:12px }} summary {{ font-size:12px; color:var(--muted); cursor:pointer }}
  table {{ border-collapse:collapse; font-size:12px; margin-top:8px; width:100% }}
  th,td {{ text-align:left; padding:3px 8px; border-bottom:1px solid var(--grid) }}
  .grp {{ font-size:17px; margin:38px 0 2px; padding-top:14px;
    border-top:1px solid var(--rule) }}
  .grp-n {{ color:var(--muted); font-weight:400; font-size:13px }}
  .grp-blurb {{ color:var(--ink-2); font-size:13px; margin:0 0 16px }}
  .card {{ display:grid; grid-template-columns:172px 1fr; gap:16px;
    background:var(--surface-1); border:1px solid var(--rule); border-radius:10px;
    padding:14px; margin-bottom:12px }}
  @media (max-width:720px) {{ .card {{ grid-template-columns:1fr }} }}
  .thumb img {{ width:100%; border-radius:7px; display:block }}
  .src {{ color:var(--muted); font-size:11px; margin-top:5px; text-align:center }}
  .q {{ font-weight:600; margin-bottom:2px }}
  .ref {{ color:var(--ink-2); font-size:13px; margin-bottom:11px }}
  .cols {{ display:grid; grid-template-columns:1fr 1fr; gap:11px }}
  @media (max-width:720px) {{ .cols {{ grid-template-columns:1fr }} }}
  .pane {{ border:1px solid var(--rule); border-radius:8px; overflow:hidden }}
  .pane header {{ display:flex; align-items:center; gap:7px; flex-wrap:wrap;
    padding:7px 10px; font-size:12px; font-weight:600;
    border-bottom:1px solid var(--rule); background:var(--plane) }}
  .dot {{ width:9px; height:9px; border-radius:50%; flex:none }}
  .d1 {{ background:var(--s1) }} .d2 {{ background:var(--s2) }}
  .tok {{ font-weight:400; color:var(--muted); font-variant-numeric:tabular-nums }}
  .badge {{ font-weight:500; font-size:11px; color:var(--ink-2);
    border:1px solid var(--rule); border-radius:999px; padding:1px 7px }}
  .badge.crit {{ color:var(--crit); border-color:var(--crit) }}
  .txt {{ padding:10px; font-size:12.5px; line-height:1.5; color:var(--ink-2);
    max-height:210px; overflow:auto; white-space:pre-wrap; word-break:break-word }}
  .toggle {{ position:fixed; top:14px; right:14px; z-index:9;
    background:var(--surface-1); color:var(--ink-2); border:1px solid var(--rule);
    border-radius:999px; padding:6px 13px; font-size:12px; cursor:pointer }}
</style></head>
<body class="viz-root"><div class="wrap">
<button class="toggle" onclick="var r=document.documentElement;
  r.dataset.theme = r.dataset.theme==='dark' ? 'light' : 'dark';">light / dark</button>

<h1>Freezing the cross-modal position, live during generation</h1>
<p class="sub">The same intervention that removes 40&#37; of the positional drift in a
forced-choice logit readout (E4), applied while the model writes.</p>
<p class="meta">Qwen3-VL-8B-Instruct &middot; {n} prompts from set_c (train split) &middot;
greedy decoding, cap {cap} tokens &middot; d0 = 24</p>

<div class="tiles">
  <div class="tile bad"><div class="v">{round(100*n_cap/n)}&#37;</div>
    <div class="k">never terminated</div><div class="n">{n_cap} of {n} hit the {cap}-token cap &middot; baseline 0&#37;</div></div>
  <div class="tile bad"><div class="v">{round(100*n_loop/n)}&#37;</div>
    <div class="k">ended in a repetition loop</div><div class="n">{n_loop} of {n} &middot; baseline 0&#37;</div></div>
  <div class="tile"><div class="v">72&#37; &rarr; {match_rows[arms.index(TREAT)][1]}&#37;</div>
    <div class="k">exact match, {len(sub)} short-answer prompts</div>
    <div class="n">36&#37; among those that terminated</div></div>
  <div class="tile"><div class="v">68&#37;</div>
    <div class="k">changed by the no-op control</div>
    <div class="n">a mathematically exact no-op, at identical accuracy</div></div>
</div>

<div class="charts">
  <div class="panel"><h3>Exact match by arm</h3>
    <p class="cap">{len(sub)} prompts whose reference is 1&ndash;2 words. One measure, so one colour;
      every bar is labelled.</p>
    {bars(match_rows, "&#37;")}
    <details><summary>table view</summary><table><tr><th>arm</th><th>match</th></tr>
      {''.join(f'<tr><td>{esc(a)}</td><td>{v}%</td></tr>' for a, v in match_rows)}</table></details>
  </div>
  <div class="panel"><h3>Period of the terminal loop</h3>
    <p class="cap">Tokens in the repeating unit, frozen:24. Shorter period = tighter loop.</p>
    {bars(period_rows)}
    <details><summary>table view</summary><table><tr><th>period</th><th>cases</th></tr>
      {''.join(f'<tr><td>{esc(a)}</td><td>{v}</td></tr>' for a, v in period_rows)}</table></details>
  </div>
</div>

{''.join(cards)}
</div></body></html>'''
    Path(args.html).write_text(page)
    kb = len(page.encode()) / 1024
    print(f"[e5] wrote {args.html}  ({kb/1024:.1f} MB, {n} cases)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dataset", default=RP.DEFAULT_DATASET)
    ap.add_argument("--html", default="e5.html")
    ap.add_argument("--pool", type=int, default=60)
    ap.add_argument("--image-side", type=int, default=768)
    ap.add_argument("--px", type=int, default=340)
    build(ap.parse_args())


if __name__ == "__main__":
    main()
