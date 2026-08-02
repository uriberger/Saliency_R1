#!/usr/bin/env python
"""Render an overlap-probe run as a single self-contained HTML page.

The markdown report was fine for grepping but useless for actually *looking* at a
completion: the thing you want to see is which sentences became scored observe steps
and how much each one was worth, in place, in the text.

Colours are the data-viz reference palette used unchanged (categorical slots 1-2, the
blue sequential ramp, the blue/red diverging pair). The palette validator is inlined as
a module script and auto-runs on open via the `data-palette` attribute, printing a
console.table report -- this cluster has no node, so that is where the six checks run.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent

# --- reference palette (references/palette.md), used unchanged -------------------
LIGHT = dict(
    surface="#fcfcfb", page="#f9f9f7", ink="#0b0b0b", ink2="#52514e", muted="#898781",
    grid="#e1e0d9", axis="#c3c2b7", border="rgba(11,11,11,0.10)",
    s1="#2a78d6", s2="#eb6834", good="#0ca30c", crit="#d03b3b",
    # blue sequential ramp, steps 100-700
    seq=["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
         "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"],
    divmid="#f0efec",
)
DARK = dict(
    surface="#1a1a19", page="#0d0d0d", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
    grid="#2c2c2a", axis="#383835", border="rgba(255,255,255,0.10)",
    s1="#3987e5", s2="#d95926", good="#0ca30c", crit="#d03b3b",
    seq=["#0d366b", "#104281", "#184f95", "#1c5cab", "#256abf", "#2a78d6",
         "#3987e5", "#5598e7", "#6da7ec", "#86b6ef", "#9ec5f4", "#b7d3f6", "#cde2fb"],
    divmid="#383835",
)
# Only the categorical slots actually carrying identity go to the validator.
PALETTE_LIGHT = f"{LIGHT['s1']},{LIGHT['s2']}"
PALETTE_DARK = f"{DARK['s1']},{DARK['s2']}"


def _css() -> str:
    def block(p, seq_name):
        ramp = "\n".join(f"    --seq-{i}: {c};" for i, c in enumerate(p["seq"]))
        return f"""    --surface: {p['surface']};
    --page: {p['page']};
    --ink: {p['ink']};
    --ink2: {p['ink2']};
    --muted: {p['muted']};
    --grid: {p['grid']};
    --axis: {p['axis']};
    --border: {p['border']};
    --s1: {p['s1']};
    --s2: {p['s2']};
    --good: {p['good']};
    --crit: {p['crit']};
    --divmid: {p['divmid']};
{ramp}"""

    return f"""
:root {{ color-scheme: light; {block(LIGHT, 'l')} }}
@media (prefers-color-scheme: dark) {{
  :root:where(:not([data-theme="light"])) {{ color-scheme: dark; {block(DARK, 'd')} }}
}}
:root[data-theme="dark"] {{ color-scheme: dark; {block(DARK, 'd')} }}

* {{ box-sizing: border-box; }}
body {{
  margin: 0; background: var(--page); color: var(--ink);
  font: 14px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif;
}}
a {{ color: var(--s1); }}
.wrap {{ max-width: 1280px; margin: 0 auto; padding: 24px 20px 80px; }}
h1 {{ font-size: 22px; margin: 0 0 4px; letter-spacing: -0.01em; }}
h2 {{ font-size: 16px; margin: 32px 0 12px; letter-spacing: -0.01em; }}
h3 {{ font-size: 13px; margin: 0 0 10px; color: var(--ink2); font-weight: 600; }}
.sub {{ color: var(--ink2); font-size: 13px; margin: 0 0 18px; }}
.card {{
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 10px; padding: 16px 18px; margin-bottom: 14px;
}}
.grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
@media (max-width: 900px) {{ .grid2 {{ grid-template-columns: 1fr; }} }}

/* --- filter row: one row above everything it scopes ------------------------- */
.filters {{
  display: flex; flex-wrap: wrap; gap: 10px; align-items: center;
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 10px; padding: 10px 12px; margin-bottom: 18px;
  position: sticky; top: 0; z-index: 20;
}}
.filters label {{ font-size: 12px; color: var(--ink2); display: flex; gap: 6px; align-items: center; }}
select, input[type=search], button {{
  font: inherit; font-size: 13px; color: var(--ink); background: var(--surface);
  border: 1px solid var(--axis); border-radius: 7px; padding: 5px 9px; min-height: 30px;
}}
button {{ cursor: pointer; }}
button:hover {{ background: var(--page); }}
.spacer {{ flex: 1; }}

/* --- tables ----------------------------------------------------------------- */
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ text-align: right; padding: 5px 8px; border-bottom: 1px solid var(--grid); }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ color: var(--muted); font-weight: 600; font-size: 11px; text-transform: uppercase;
      letter-spacing: 0.04em; border-bottom: 1px solid var(--axis); }}
td.num {{ font-variant-numeric: tabular-nums; }}
tbody tr:hover {{ background: var(--page); }}
.barcell {{ position: relative; }}
.bar {{
  display: inline-block; height: 9px; background: var(--s1);
  border-radius: 0 4px 4px 0; vertical-align: middle; margin-right: 6px;
}}

/* --- charts ----------------------------------------------------------------- */
svg {{ display: block; overflow: visible; }}
.gridline {{ stroke: var(--grid); stroke-width: 1; }}
.axisline {{ stroke: var(--axis); stroke-width: 1; }}
.tick {{ fill: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }}
.axlabel {{ fill: var(--ink2); font-size: 11px; }}
.legend {{ display: flex; gap: 16px; font-size: 12px; color: var(--ink2); margin-bottom: 8px; }}
.legend i {{ width: 10px; height: 10px; border-radius: 3px; display: inline-block; margin-right: 6px; }}

/* --- completion text -------------------------------------------------------- */
.sent {{ border-radius: 4px; padding: 1px 2px; }}
.sent.scored {{ cursor: help; box-shadow: 0 0 0 2px var(--surface); }}
.sent.skipped {{ text-decoration: underline dotted var(--muted); text-underline-offset: 3px; }}
.sent.plain {{ color: var(--ink2); }}
.think {{ white-space: pre-wrap; }}
.answer {{ margin-top: 8px; padding-top: 8px; border-top: 1px solid var(--grid); }}
.chip {{
  display: inline-block; font-size: 11px; padding: 1px 7px; border-radius: 999px;
  border: 1px solid var(--border); color: var(--ink2); margin-right: 6px;
  font-variant-numeric: tabular-nums; white-space: nowrap;
}}
.chip b {{ color: var(--ink); font-weight: 600; }}
.compl {{ border-top: 1px solid var(--grid); padding-top: 12px; margin-top: 12px; }}
.compl:first-of-type {{ border-top: 0; }}
.advbar {{ display: inline-block; height: 9px; border-radius: 4px; vertical-align: middle; }}
details > summary {{ cursor: pointer; color: var(--ink2); font-size: 12px; margin-top: 8px; }}
.sampleimg {{
  width: 100%; border-radius: 8px; border: 1px solid var(--border); display: block;
}}
.samplehead {{ display: grid; grid-template-columns: 240px 1fr; gap: 16px; align-items: start; }}
@media (max-width: 760px) {{ .samplehead {{ grid-template-columns: 1fr; }} }}
.scale {{ display: flex; align-items: center; gap: 6px; font-size: 11px; color: var(--muted); }}
.scale i {{ width: 18px; height: 10px; display: inline-block; }}
.tip {{
  position: fixed; z-index: 50; pointer-events: none; max-width: 340px;
  background: var(--surface); color: var(--ink); border: 1px solid var(--axis);
  border-radius: 8px; padding: 8px 10px; font-size: 12px; line-height: 1.45;
  box-shadow: 0 6px 24px rgba(0,0,0,0.18); opacity: 0; transition: opacity .1s;
}}
.tip table {{ font-size: 11px; }}
.tip td {{ border: 0; padding: 1px 4px; }}
.hidden {{ display: none !important; }}
.note {{ font-size: 12px; color: var(--muted); margin-top: 6px; }}

/* --- attention-map thumbnails ----------------------------------------------- */
.maps {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px; }}
.mapcell {{ width: 168px; }}
.mapcell canvas {{
  width: 168px; height: auto; display: block; border-radius: 6px;
  border: 1px solid var(--border); background: var(--page);
}}
.mapcell .cap {{ font-size: 11px; color: var(--ink2); margin-top: 3px; line-height: 1.35;
                 font-variant-numeric: tabular-nums; }}
.mapcell .cap b {{ color: var(--ink); }}
.mapcell .txt {{ font-size: 11px; color: var(--muted); overflow: hidden;
                 display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; }}
.repeatrow {{
  border-left: 2px solid var(--s2); padding-left: 10px; margin: 12px 0;
}}
.repeatrow > .rtitle {{ font-size: 12px; color: var(--ink2); margin-bottom: 4px; }}
.maplegend {{ display: flex; flex-wrap: wrap; gap: 14px; align-items: center;
              font-size: 11px; color: var(--muted); margin-top: 8px; }}
.maplegend .sw {{ width: 42px; height: 10px; display: inline-block; border-radius: 3px;
                  vertical-align: middle; margin-right: 5px; }}
.maplegend .bx {{ width: 12px; height: 10px; display: inline-block; vertical-align: middle;
                  margin-right: 5px; }}
"""


def write_html(merged: dict, out_path: Path, validator: Path | None = None) -> Path:
    cfg = merged.get("config", {})
    payload = json.dumps(merged, ensure_ascii=False, default=str).replace("</", "<\\/")

    vjs = ""
    if validator and validator.exists():
        vjs = f'<script type="module">\n{validator.read_text()}\n</script>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Overlap reward probe</title>
<style>{_css()}</style>
</head>
<body data-palette="{PALETTE_LIGHT}" data-mode="light" data-surface="{LIGHT['surface']}">
<div class="wrap">
  <h1>Attention-overlap reward probe</h1>
  <p class="sub" id="runsub"></p>

  <div class="filters">
    <label>Model <select id="fModel"></select></label>
    <label>Heat by <select id="fMetric">
      <option value="score">score (reward metric)</option>
      <option value="mean_in_raw">mean_in</option>
      <option value="auroc_raw">auroc</option>
    </select></label>
    <label>Map scale <select id="fHeat">
      <option value="abs">absolute (shared) — shows dimming</option>
      <option value="own">each map's own peak — shows shape</option>
    </select></label>
    <label>Overlay <select id="fOverlay">
      <option value="strong">strong</option>
      <option value="normal">normal</option>
      <option value="subtle">subtle</option>
    </select></label>
    <label>Boxes <select id="fBoxes">
      <option value="kept">scored boxes only</option>
      <option value="all">all DINO boxes</option>
      <option value="none">none — mask outline only</option>
    </select></label>
    <label>Sort <select id="fSort">
      <option value="row">dataset order</option>
      <option value="dup">duplicate steps</option>
      <option value="steps">observe steps</option>
      <option value="len">completion length</option>
      <option value="overlap">overlap reward</option>
    </select></label>
    <label><input type="checkbox" id="fDup"> duplicates only</label>
    <label><input type="search" id="fQ" placeholder="search question / text" size="18"></label>
    <span class="spacer"></span>
    <button id="tblBtn" aria-pressed="false">Table view</button>
    <button id="themeBtn">Dark</button>
  </div>

  <h2>Summary</h2>
  <div class="card" id="summary"></div>

  <div class="grid2">
    <div class="card"><h3>Repeat premium — same sentence, later occurrences vs first</h3>
      <div id="repeat"></div>
      <p class="note" id="repeatNote"></p></div>
    <div class="card"><h3>Per-step score distribution</h3>
      <div id="dist"></div></div>
  </div>

  <h2>Samples</h2>
  <div id="samples"></div>
</div>

<div class="tip" id="tip" role="tooltip"></div>
<script type="application/json" id="probe-data">{payload}</script>
<script>
const DATA = JSON.parse(document.getElementById('probe-data').textContent);
const CFG = DATA.config || {{}};
const MODELS = Object.keys(DATA.models);
const $ = s => document.querySelector(s);
const el = (t, a = {{}}, ...kids) => {{
  const n = document.createElement(t);
  for (const [k, v] of Object.entries(a)) {{
    if (k === 'class') n.className = v; else if (k === 'html') n.innerHTML = v;
    else if (k.startsWith('on')) n.addEventListener(k.slice(2), v);
    else if (v !== null && v !== undefined) n.setAttribute(k, v);
  }}
  for (const k of kids.flat()) if (k != null) n.append(k.nodeType ? k : String(k));
  return n;
}};
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const fmt = (v, n = 3) => (v === null || v === undefined || Number.isNaN(v)) ? '—' : Number(v).toFixed(n);
const norm = s => s.trim().toLowerCase().replace(/\\s+/g, ' ');

// ---------- state ----------
const S = {{ model: MODELS[0], metric: 'score', sort: 'row', dupOnly: false, q: '', table: false, heat: 'abs', boxes: 'kept', overlay: 'strong' }};

// ---------- derived ----------
function steps(c) {{ return (c.observe_steps || []); }}
function scored(c) {{ return steps(c).filter(s => s.grounded && s.score !== null); }}
function dupFrac(c) {{
  const t = steps(c).map(s => norm(s.text));
  if (!t.length) return 0;
  return 1 - new Set(t).size / t.length;
}}
function modelStats(name) {{
  const cs = DATA.models[name].samples.flatMap(s => s.completions);
  const st = cs.flatMap(scored);
  const mean = a => a.length ? a.reduce((x, y) => x + y, 0) / a.length : NaN;
  const has = k => st.some(s => s[k] !== null && s[k] !== undefined);
  return {{
    completions: cs.length,
    len: mean(cs.map(c => c.n_completion_tokens)),
    truncated: mean(cs.map(c => c.truncated_at_max_tokens ? 1 : 0)),
    format: mean(cs.map(c => c.rewards.think_format_reward ?? NaN)),
    overlap: mean(cs.map(c => c.rewards.think_overlap_reward).filter(v => v !== null)),
    stepsPer: mean(cs.map(c => c.n_observe_steps_total)),
    scoredPer: mean(cs.map(c => c.n_observe_steps_scored)),
    stepScore: mean(st.map(s => s.score)),
    meanIn: has('mean_in_raw') ? mean(st.map(s => s.mean_in_raw).filter(v => v != null)) : NaN,
    auroc: has('auroc_raw') ? mean(st.map(s => s.auroc_raw).filter(v => v != null)) : NaN,
    area: mean(st.map(s => s.box_area_frac).filter(v => v != null)),
    dup: mean(cs.map(dupFrac)),
    acc: mean(cs.map(c => c.rewards.accuracy_reward).filter(v => v !== null)),
    judge: mean(cs.map(c => c.rewards.openai_reward).filter(v => v !== null)),
  }};
}}
// paired first-vs-later occurrence of the same sentence inside one completion
function repeatPairs(name, key) {{
  const out = [];
  for (const s of DATA.models[name].samples) for (const c of s.completions) {{
    const g = new Map();
    for (const st of scored(c)) {{
      if (st[key] === null || st[key] === undefined) continue;
      const k = norm(st.text); if (!g.has(k)) g.set(k, []); g.get(k).push(st[key]);
    }}
    for (const v of g.values()) if (v.length > 1)
      out.push([v[0], v.slice(1).reduce((x, y) => x + y, 0) / (v.length - 1)]);
  }}
  return out;
}}

// ---------- tooltip ----------
const tip = $('#tip');
function showTip(e, html) {{
  tip.innerHTML = html; tip.style.opacity = 1;
  const r = tip.getBoundingClientRect();
  let x = e.clientX + 14, y = e.clientY + 14;
  if (x + r.width > innerWidth - 8) x = e.clientX - r.width - 14;
  if (y + r.height > innerHeight - 8) y = e.clientY - r.height - 14;
  tip.style.left = x + 'px'; tip.style.top = y + 'px';
}}
const hideTip = () => tip.style.opacity = 0;
function bindTip(node, html) {{
  node.addEventListener('mousemove', e => showTip(e, html));
  node.addEventListener('mouseleave', hideTip);
  node.tabIndex = 0;
  node.addEventListener('focus', e => {{
    const b = node.getBoundingClientRect();
    showTip({{ clientX: b.left, clientY: b.bottom }}, html);
  }});
  node.addEventListener('blur', hideTip);
}}

// ---------- summary ----------
const ROWS = [
  ['completions', 'completions', 0, false],
  ['mean length (tok)', 'len', 0, true],
  ['truncated', 'truncated', 3, true],
  ['format valid', 'format', 3, true],
  ['observe steps / completion', 'stepsPer', 2, true],
  ['scored steps / completion', 'scoredPer', 2, true],
  ['duplicate step fraction', 'dup', 3, true],
  ['mean score per step', 'stepScore', 4, true],
  ['mean_in per step', 'meanIn', 4, true],
  ['auroc per step', 'auroc', 4, true],
  ['box area fraction', 'area', 3, true],
  ['overlap reward', 'overlap', 4, true],
  ['accuracy', 'acc', 3, true],
  ['judge', 'judge', 3, true],
];
function renderSummary() {{
  const stats = Object.fromEntries(MODELS.map(m => [m, modelStats(m)]));
  const head = el('tr', {{}}, el('th', {{}}, 'metric'), ...MODELS.map(m => el('th', {{}}, m)));
  const body = ROWS.map(([label, key, nd, bar]) => {{
    const vals = MODELS.map(m => stats[m][key]);
    const max = Math.max(...vals.filter(Number.isFinite).map(Math.abs), 1e-9);
    return el('tr', {{}}, el('td', {{}}, label), ...vals.map(v => {{
      const td = el('td', {{ class: 'num barcell' }});
      if (bar && Number.isFinite(v) && MODELS.length > 1) {{
        td.append(el('span', {{ class: 'bar', style: `width:${{Math.max(2, 46 * Math.abs(v) / max)}}px` }}));
      }}
      td.append(fmt(v, nd));
      return td;
    }}));
  }});
  $('#summary').replaceChildren(el('table', {{}}, el('thead', {{}}, head), el('tbody', {{}}, ...body)));
}}

// ---------- chart: repeat premium (diverging bar around 0% change) ----------
function renderRepeat() {{
  const host = $('#repeat');
  const keys = [['score', 'reward metric'], ['mean_in_raw', 'mean_in'], ['auroc_raw', 'auroc']];
  const rows = [];
  for (const [k, label] of keys) {{
    const p = repeatPairs(S.model, k);
    if (p.length < 5) continue;
    const f = p.reduce((a, b) => a + b[0], 0) / p.length;
    const l = p.reduce((a, b) => a + b[1], 0) / p.length;
    const d = p.map(x => x[1] - x[0]);
    const mu = d.reduce((a, b) => a + b, 0) / d.length;
    const sd = Math.sqrt(d.reduce((a, b) => a + (b - mu) ** 2, 0) / (d.length - 1));
    rows.push({{ label, n: p.length, first: f, later: l, pct: 100 * (l / f - 1),
                up: d.filter(x => x > 0).length / d.length, t: mu / (sd / Math.sqrt(d.length)) }});
  }}
  if (!rows.length) {{ host.replaceChildren(el('p', {{ class: 'note' }}, 'No repeated sentences in this model.')); return; }}
  $('#repeatNote').textContent =
    `Paired within a completion: a sentence's first occurrence vs the mean of its later ones. ` +
    `Positive = repeating a sentence raises its own score.`;

  const W = 520, rowH = 38, H = rows.length * rowH + 34, L = 96, R = 54;
  const span = Math.max(14, ...rows.map(r => Math.abs(r.pct) * 1.35));
  const x = v => L + (W - L - R) * (v + span) / (2 * span);
  const svg = el('svg', {{ viewBox: `0 0 ${{W}} ${{H}}`, width: '100%', height: H, role: 'img',
                          'aria-label': 'Percent change in step score on repeated sentences' }});
  const NS = 'http://www.w3.org/2000/svg';
  const mk = (t, a) => {{ const n = document.createElementNS(NS, t); for (const k in a) n.setAttribute(k, a[k]); return n; }};
  for (const v of [-span, -span / 2, 0, span / 2, span]) {{
    svg.append(mk('line', {{ x1: x(v), x2: x(v), y1: 6, y2: H - 26,
                            class: v === 0 ? 'axisline' : 'gridline' }}));
    const t = mk('text', {{ x: x(v), y: H - 10, class: 'tick', 'text-anchor': 'middle' }});
    t.textContent = (v > 0 ? '+' : '') + v.toFixed(0) + '%'; svg.append(t);
  }}
  rows.forEach((r, i) => {{
    const y = 14 + i * rowH, h = 12;
    const lab = mk('text', {{ x: L - 10, y: y + h - 1, class: 'axlabel', 'text-anchor': 'end' }});
    lab.textContent = r.label; svg.append(lab);
    const x0 = x(0), x1 = x(r.pct);
    const g = mk('g', {{}});
    const rect = mk('rect', {{ x: Math.min(x0, x1), y, width: Math.max(2, Math.abs(x1 - x0)), height: h,
                              rx: 4, fill: r.pct >= 0 ? css('--crit') : css('--s1') }});
    g.append(rect);
    const v = mk('text', {{ x: x1 + (r.pct >= 0 ? 8 : -8), y: y + h - 1, class: 'tick',
                           'text-anchor': r.pct >= 0 ? 'start' : 'end' }});
    v.textContent = (r.pct >= 0 ? '+' : '') + r.pct.toFixed(1) + '%';
    g.append(v);
    svg.append(g);
    bindTip(g, `<b>${{r.label}}</b><table>
      <tr><td>pairs</td><td>${{r.n}}</td></tr>
      <tr><td>first</td><td>${{fmt(r.first, 4)}}</td></tr>
      <tr><td>later</td><td>${{fmt(r.later, 4)}}</td></tr>
      <tr><td>later &gt; first</td><td>${{(100 * r.up).toFixed(1)}}%</td></tr>
      <tr><td>paired t</td><td>${{r.t.toFixed(1)}}</td></tr></table>`);
  }});
  const tbl = el('table', {{ class: 'tv hidden' }},
    el('thead', {{}}, el('tr', {{}}, el('th', {{}}, 'metric'), el('th', {{}}, 'pairs'),
      el('th', {{}}, 'first'), el('th', {{}}, 'later'), el('th', {{}}, 'change'), el('th', {{}}, 'paired t'))),
    el('tbody', {{}}, ...rows.map(r => el('tr', {{}}, el('td', {{}}, r.label),
      el('td', {{ class: 'num' }}, r.n), el('td', {{ class: 'num' }}, fmt(r.first, 4)),
      el('td', {{ class: 'num' }}, fmt(r.later, 4)),
      el('td', {{ class: 'num' }}, (r.pct >= 0 ? '+' : '') + r.pct.toFixed(1) + '%'),
      el('td', {{ class: 'num' }}, r.t.toFixed(1))))));
  host.replaceChildren(svg, tbl);
  applyTableView();
}}

// ---------- chart: per-step score distribution (small multiples, one hue each) ----------
function renderDist() {{
  const host = $('#dist');
  const st = DATA.models[S.model].samples.flatMap(s => s.completions).flatMap(scored);
  const series = [['score', 'reward metric', css('--s1')]];
  if (st.some(s => s.auroc_raw != null)) series.push(['auroc_raw', 'auroc', css('--s2')]);
  const panels = [], tabRows = [];
  for (const [key, label, colour] of series) {{
    const v = st.map(s => s[key]).filter(x => x != null);
    if (!v.length) continue;
    const lo = Math.min(...v), hi = Math.max(...v), NB = 28;
    const bins = new Array(NB).fill(0);
    for (const x of v) bins[Math.min(NB - 1, Math.floor(NB * (x - lo) / (hi - lo || 1)))]++;
    const W = 480, H = 150, L = 34, B = 26, top = 8;
    const maxB = Math.max(...bins);
    const NS = 'http://www.w3.org/2000/svg';
    const mk = (t, a) => {{ const n = document.createElementNS(NS, t); for (const k in a) n.setAttribute(k, a[k]); return n; }};
    const svg = el('svg', {{ viewBox: `0 0 ${{W}} ${{H}}`, width: '100%', height: H, role: 'img',
                            'aria-label': `Distribution of ${{label}} across scored steps` }});
    for (let g = 0; g <= 2; g++) {{
      const y = top + (H - top - B) * g / 2;
      svg.append(mk('line', {{ x1: L, x2: W, y1: y, y2: y, class: g === 2 ? 'axisline' : 'gridline' }}));
      const t = mk('text', {{ x: L - 6, y: y + 4, class: 'tick', 'text-anchor': 'end' }});
      t.textContent = Math.round(maxB * (1 - g / 2)); svg.append(t);
    }}
    const bw = (W - L) / NB;
    bins.forEach((n, i) => {{
      const h = (H - top - B) * n / (maxB || 1);
      const r = mk('rect', {{ x: L + i * bw + 1, y: H - B - h, width: Math.max(1, bw - 2),
                             height: Math.max(n ? 1 : 0, h), rx: Math.min(4, bw / 2), fill: colour }});
      svg.append(r);
      const a = lo + (hi - lo) * i / NB, b = lo + (hi - lo) * (i + 1) / NB;
      bindTip(r, `<b>${{label}}</b><br>${{fmt(a, 3)}} – ${{fmt(b, 3)}}<br>${{n}} steps`);
    }});
    for (const f of [0, 0.5, 1]) {{
      const t = mk('text', {{ x: L + (W - L) * f, y: H - 8, class: 'tick',
                             'text-anchor': f === 0 ? 'start' : f === 1 ? 'end' : 'middle' }});
      t.textContent = fmt(lo + (hi - lo) * f, 2); svg.append(t);
    }}
    const mean = v.reduce((a, b) => a + b, 0) / v.length;
    panels.push(el('div', {{}},
      el('div', {{ class: 'legend' }},
        el('span', {{}}, el('i', {{ style: `background:${{colour}}` }}), `${{label}} — mean ${{fmt(mean, 4)}}, n=${{v.length}}`)),
      svg));
    tabRows.push([label, v.length, fmt(Math.min(...v), 4), fmt(mean, 4), fmt(Math.max(...v), 4)]);
  }}
  const tbl = el('table', {{ class: 'tv hidden' }},
    el('thead', {{}}, el('tr', {{}}, el('th', {{}}, 'metric'), el('th', {{}}, 'steps'),
      el('th', {{}}, 'min'), el('th', {{}}, 'mean'), el('th', {{}}, 'max'))),
    el('tbody', {{}}, ...tabRows.map(r => el('tr', {{}}, el('td', {{}}, r[0]),
      ...r.slice(1).map(x => el('td', {{ class: 'num' }}, x))))));
  host.replaceChildren(...panels, tbl);
  applyTableView();
}}

// ---------- samples ----------
function heatScale() {{
  // Sequential ramp bounds: 0 .. p95 of the selected metric across this model.
  const st = DATA.models[S.model].samples.flatMap(s => s.completions).flatMap(scored)
    .map(s => s[S.metric]).filter(v => v != null).sort((a, b) => a - b);
  if (!st.length) return [0, 1];
  const lo = S.metric === 'auroc_raw' ? 0.5 : 0;
  return [lo, st[Math.floor(0.95 * (st.length - 1))] || 1];
}}
function seqColour(v, lo, hi) {{
  const n = 13, f = Math.max(0, Math.min(1, (v - lo) / (hi - lo || 1)));
  // keep the low end light enough for body text to stay legible on it
  return css('--seq-' + Math.round(f * (n - 6)));
}}

function segmentsFor(c) {{
  // Walk the sentences in order and attach the matching observe step. Duplicate
  // texts are consumed in order so the Nth repeat gets the Nth step's score.
  const byText = new Map();
  for (const s of steps(c)) {{
    const k = norm(s.text); if (!byText.has(k)) byText.set(k, []); byText.get(k).push(s);
  }}
  const used = new Map();
  return (c.all_sentences || []).map(sn => {{
    const k = norm(sn.text), i = used.get(k) || 0;
    const list = byText.get(k);
    let step = null;
    if (list && i < list.length) {{ step = list[i]; used.set(k, i + 1); }}
    return {{ text: sn.text, label: sn.label, step }};
  }});
}}

// ---------- attention-map thumbnails ----------
const OPACITY = {{ subtle: 0.75, normal: 1.0, strong: 1.4 }};
const IMGCACHE = new Map();
function getImage(src) {{
  if (!IMGCACHE.has(src)) {{ const im = new Image(); im.src = src; IMGCACHE.set(src, im); }}
  return IMGCACHE.get(src);
}}
function b64u8(s) {{
  const bin = atob(s), a = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) a[i] = bin.charCodeAt(i);
  return a;
}}
// Absolute reference for the shared scale: p95 of map peaks across this model, so one
// bright step doesn't crush every other map to black.
let ABS_REF = 1;
function computeAbsRef() {{
  const v = DATA.models[S.model].samples.flatMap(s => s.completions).flatMap(steps)
    .map(s => s.map_max).filter(x => x != null).sort((a, b) => a - b);
  ABS_REF = v.length ? (v[Math.floor(0.95 * (v.length - 1))] || 1) : 1;
}}
// Sequential ramp as RGB triples. A single hue at varying alpha was too faint to read
// over a photo, so value drives BOTH the ramp step and the opacity, and a gamma lifts
// the mid range — most patches sit near zero, so a linear map leaves everything but the
// peak invisible, which is exactly the range the flattening lives in.
const HEAT_GAMMA = 0.55;
function seqRamp() {{
  const out = [];
  for (let i = 0; i < 13; i++) {{
    const hex = css('--seq-' + i).replace('#', '');
    out.push([0, 2, 4].map(k => parseInt(hex.slice(k, k + 2), 16)));
  }}
  return out;
}}
function heatRGB() {{
  const hex = css('--s1').replace('#', '');
  return [0, 2, 4].map(i => parseInt(hex.slice(i, i + 2), 16));
}}
function drawStep(cv, st, src) {{
  const im = getImage(src);
  const go = () => {{
    const nw = im.naturalWidth || 1, nh = im.naturalHeight || 1;
    const W = 336, H = Math.max(1, Math.round(W * nh / nw));
    cv.width = W; cv.height = H;
    const ctx = cv.getContext('2d');
    ctx.clearRect(0, 0, W, H);
    ctx.save();
    // Wash the photo out hard: the overlay has to win against arbitrary image content.
    try {{ ctx.filter = 'grayscale(1) contrast(0.45) brightness(1.35)'; }} catch (e) {{}}
    ctx.drawImage(im, 0, 0, W, H);
    ctx.restore();
    if (st.map_q && st.grid) {{
      const [gh, gw] = st.grid, q = b64u8(st.map_q);
      const cw = W / gw, chh = H / gh, RAMP = seqRamp(), K = OPACITY[S.overlay];
      for (let y = 0; y < gh; y++) for (let x = 0; x < gw; x++) {{
        let v = q[y * gw + x] / 255;
        if (S.heat === 'abs') v = v * (st.map_max || 0) / (ABS_REF || 1);
        v = Math.max(0, Math.min(1, v));
        if (v <= 0.015) continue;
        const t = Math.pow(v, HEAT_GAMMA);
        const [r, g, b] = RAMP[Math.min(12, Math.round(t * 12))];
        const a = Math.min(1, (0.30 + 0.70 * t) * K);
        ctx.fillStyle = `rgba(${{r}},${{g}},${{b}},${{a.toFixed(3)}})`;
        ctx.fillRect(x * cw, y * chh, cw + 0.6, chh + 0.6);
      }}
    }}
    // DINO returns 16-19 boxes on a typical step; drawn as 19 solid rectangles on a
    // 168px thumbnail that is an unreadable tangle. So the bold mark is the union mask
    // -- the patches that actually entered the score -- and the individual boxes are
    // thin and optional underneath it.
    const [gh2, gw2] = st.grid || [0, 0];
    const cw2 = gw2 ? W / gw2 : 0, ch2 = gh2 ? H / gh2 : 0;
    if (S.boxes !== 'none') {{
      const kept = new Set((st.boxes_kept || []).map(bb => bb.join(',')));
      for (const bb of (st.boxes_raw || [])) {{
        const isKept = kept.has(bb.join(','));
        if (!isKept && S.boxes !== 'all') continue;
        ctx.lineWidth = 1;
        ctx.strokeStyle = isKept ? css('--s2') : css('--muted');
        ctx.globalAlpha = isKept ? 0.55 : 0.35;
        ctx.strokeRect(bb[0] * W, bb[1] * H, (bb[2] - bb[0]) * W, (bb[3] - bb[1]) * H);
      }}
      ctx.globalAlpha = 1;
    }}
    if (st.mask_q && gh2 && gw2) {{
      // outline of the scored region: every mask cell edge that borders a non-mask cell
      const mk = b64u8(st.mask_q);
      const on = (y, x) => y >= 0 && x >= 0 && y < gh2 && x < gw2 && mk[y * gw2 + x];
      ctx.strokeStyle = css('--s2'); ctx.lineWidth = 2; ctx.beginPath();
      for (let y = 0; y < gh2; y++) for (let x = 0; x < gw2; x++) {{
        if (!on(y, x)) continue;
        const X = x * cw2, Y = y * ch2;
        if (!on(y - 1, x)) {{ ctx.moveTo(X, Y); ctx.lineTo(X + cw2, Y); }}
        if (!on(y + 1, x)) {{ ctx.moveTo(X, Y + ch2); ctx.lineTo(X + cw2, Y + ch2); }}
        if (!on(y, x - 1)) {{ ctx.moveTo(X, Y); ctx.lineTo(X, Y + ch2); }}
        if (!on(y, x + 1)) {{ ctx.moveTo(X + cw2, Y); ctx.lineTo(X + cw2, Y + ch2); }}
      }}
      ctx.stroke();
    }}
  }};
  if (im.complete && im.naturalWidth) go(); else im.addEventListener('load', go, {{ once: true }});
}}
function mapCell(st, src, label) {{
  const cv = el('canvas', {{ role: 'img',
    'aria-label': `attention map for step ${{st.step_index}}: ${{st.text.slice(0, 80)}}` }});
  const cell = el('div', {{ class: 'mapcell' }}, cv,
    el('div', {{ class: 'cap' }}, label ? label + ' · ' : '',
      'score ', el('b', {{}}, st.grounded ? fmt(st.score, 4) : 'skipped'),
      ' · peak ', fmt(st.map_max, 5), ' · flat ', fmt(st.map_mean / st.map_max, 3)),
    el('div', {{ class: 'txt' }}, st.text));
  drawStep(cv, st, src);
  bindTip(cell, `<b>step ${{st.step_index}}</b><table>
    <tr><td>score</td><td>${{st.grounded ? fmt(st.score, 4) : 'skipped'}}</td></tr>
    ${{st.mean_in_raw != null ? `<tr><td>mean_in</td><td>${{fmt(st.mean_in_raw, 4)}}</td></tr>` : ''}}
    ${{st.auroc_raw != null ? `<tr><td>auroc</td><td>${{fmt(st.auroc_raw, 4)}}</td></tr>` : ''}}
    <tr><td>image mass</td><td>${{fmt(st.image_mass, 5)}}</td></tr>
    <tr><td>map peak</td><td>${{fmt(st.map_max, 5)}}</td></tr>
    <tr><td>flatness mean/max</td><td>${{fmt(st.map_mean / st.map_max, 4)}}</td></tr>
    <tr><td>boxes raw/kept</td><td>${{st.n_boxes_raw}}/${{st.n_boxes_kept}}</td></tr>
    <tr><td>box area frac</td><td>${{fmt(st.box_area_frac, 3)}}</td></tr></table>`);
  return cell;
}}
function mapLegend() {{
  const RAMP = seqRamp();
  const grad = 'linear-gradient(90deg,' + RAMP.map((c, i) =>
    `rgba(${{c[0]}},${{c[1]}},${{c[2]}},${{(0.30 + 0.70 * i / 12).toFixed(2)}}) ${{Math.round(100 * i / 12)}}%`).join(',') + ')';
  return el('div', {{ class: 'maplegend' }},
    el('span', {{}}, el('i', {{ class: 'sw', style: `background:${{grad}}` }}),
      S.heat === 'abs' ? `attention, shared scale 0 → ${{fmt(ABS_REF, 5)}}` : "attention, each map's own peak"),
    el('span', {{}}, el('i', {{ class: 'bx', style: `border:2px solid ${{css('--s2')}}` }}),
      'scored region (union mask)'),
    el('span', {{}}, el('i', {{ class: 'bx', style: `border:1px solid ${{css('--s2')}};opacity:.55` }}),
      'DINO box kept'),
    el('span', {{}}, el('i', {{ class: 'bx', style: `border:1px solid ${{css('--muted')}};opacity:.35` }}),
      `dropped by max_box_area=${{CFG.max_box_area}}`),
    el('span', {{}}, 'base image desaturated so the overlay reads'));
}}
function repeatGroups(c) {{
  const g = new Map();
  for (const st of steps(c)) {{
    const k = norm(st.text); if (!g.has(k)) g.set(k, []); g.get(k).push(st);
  }}
  return [...g.values()].filter(v => v.length > 1);
}}

function renderCompletion(c, sample, lo, hi) {{
  const r = c.rewards;
  const dup = dupFrac(c);
  const chips = el('div', {{}},
    el('span', {{ class: 'chip' }}, 'overlap ', el('b', {{}}, fmt(r.think_overlap_reward, 4))),
    el('span', {{ class: 'chip' }}, 'acc ', el('b', {{}}, fmt(r.accuracy_reward, 2))),
    el('span', {{ class: 'chip' }}, 'judge ', el('b', {{}}, fmt(r.openai_reward, 2))),
    el('span', {{ class: 'chip' }}, 'fmt ', el('b', {{}}, fmt(r.think_format_reward, 0))),
    el('span', {{ class: 'chip' }}, 'steps ', el('b', {{}}, `${{c.n_observe_steps_scored}}/${{c.n_observe_steps_total}}`)),
    el('span', {{ class: 'chip' }}, 'dup ', el('b', {{}}, fmt(dup, 2))),
    el('span', {{ class: 'chip' }}, 'tok ', el('b', {{}}, c.n_completion_tokens)),
    c.truncated_at_max_tokens ? el('span', {{ class: 'chip' }}, 'truncated') : null,
  );
  // advantage: diverging around the group mean
  const adv = r.advantage ?? 0;
  const advWrap = el('span', {{ class: 'chip' }}, 'adv ',
    el('span', {{ class: 'advbar', style: `width:${{Math.min(40, Math.abs(adv) * 18)}}px;background:${{adv >= 0 ? css('--crit') : css('--s1')}}` }}),
    ' ', el('b', {{}}, fmt(adv, 2)));
  chips.prepend(advWrap);

  const body = el('div', {{ class: 'think' }});
  for (const seg of segmentsFor(c)) {{
    const s = seg.step;
    if (s && s.grounded && s[S.metric] != null) {{
      const bg = seqColour(s[S.metric], lo, hi);
      const n = el('span', {{ class: 'sent scored', style: `background:${{bg}}` }}, seg.text + ' ');
      bindTip(n, `<b>observe step ${{s.step_index}}</b><table>
        <tr><td>score (reward)</td><td>${{fmt(s.score, 4)}}</td></tr>
        ${{s.mean_in_raw != null ? `<tr><td>mean_in</td><td>${{fmt(s.mean_in_raw, 4)}}</td></tr>` : ''}}
        ${{s.auroc_raw != null ? `<tr><td>auroc</td><td>${{fmt(s.auroc_raw, 4)}}</td></tr>` : ''}}
        <tr><td>tokens</td><td>${{s.n_tokens}}</td></tr>
        <tr><td>boxes raw/kept</td><td>${{s.n_boxes_raw}}/${{s.n_boxes_kept}}</td></tr>
        <tr><td>box area frac</td><td>${{fmt(s.box_area_frac, 3)}}</td></tr>
        <tr><td>image mass</td><td>${{fmt(s.image_mass, 5)}}</td></tr>
        <tr><td>map max</td><td>${{fmt(s.map_max, 5)}}</td></tr>
        <tr><td>flatness mean/max</td><td>${{fmt(s.map_mean / s.map_max, 4)}}</td></tr></table>`);
      body.append(n);
    }} else if (s) {{
      const n = el('span', {{ class: 'sent skipped' }}, seg.text + ' ');
      bindTip(n, `<b>observe step ${{s.step_index}} — not scored</b><br>${{s.note || 'DINO could not ground this step; it is skipped, not scored 0.'}}`);
      body.append(n);
    }} else {{
      const n = el('span', {{ class: 'sent plain' }}, seg.text + ' ');
      bindTip(n, `classified <b>${{seg.label}}</b> — not an observe step, so never scored`);
      body.append(n);
    }}
  }}
  const ans = (c.text.split('</think>')[1] || '').trim();

  const stepTbl = el('table', {{}},
    el('thead', {{}}, el('tr', {{}}, el('th', {{}}, '#'), el('th', {{}}, 'step text'), el('th', {{}}, 'tok'),
      el('th', {{}}, 'boxes'), el('th', {{}}, 'area'), el('th', {{}}, 'mass'),
      el('th', {{}}, 'mean_in'), el('th', {{}}, 'auroc'), el('th', {{}}, 'score'))),
    el('tbody', {{}}, ...steps(c).map(s => el('tr', {{}},
      el('td', {{ class: 'num' }}, s.step_index), el('td', {{}}, s.text),
      el('td', {{ class: 'num' }}, s.n_tokens),
      el('td', {{ class: 'num' }}, `${{s.n_boxes_raw}}/${{s.n_boxes_kept}}`),
      el('td', {{ class: 'num' }}, fmt(s.box_area_frac, 3)),
      el('td', {{ class: 'num' }}, fmt(s.image_mass, 5)),
      el('td', {{ class: 'num' }}, fmt(s.mean_in_raw, 4)),
      el('td', {{ class: 'num' }}, fmt(s.auroc_raw, 4)),
      el('td', {{ class: 'num' }}, s.grounded ? fmt(s.score, 4) : 'skipped')))));

  // Attention maps, drawn only when opened: a 30-sample run has thousands of steps and
  // eagerly rasterising every one would stall the page.
  const hasMaps = steps(c).some(s => s.map_q);
  let mapsBox = null;
  if (hasMaps) {{
    const reps = repeatGroups(c);
    const det = el('details', {{}}, el('summary', {{}},
      `attention maps — ${{steps(c).length}} steps${{reps.length ? `, ${{reps.length}} repeated sentence${{reps.length > 1 ? 's' : ''}}` : ''}}`));
    let built = false;
    det.addEventListener('toggle', () => {{
      if (!det.open || built) return;
      built = true;
      det.append(mapLegend());
      // repeated sentences first, in occurrence order: this is the comparison that
      // shows the peak collapsing while the shape spreads
      for (const grp of reps) {{
        det.append(el('div', {{ class: 'repeatrow' }},
          el('div', {{ class: 'rtitle' }},
            `repeated ${{grp.length}}× — "${{grp[0].text.slice(0, 90)}}"`),
          el('div', {{ class: 'maps' }},
            ...grp.map((st, i) => mapCell(st, sample.image_file, `occurrence ${{i + 1}}`)))));
      }}
      det.append(el('div', {{ class: 'rtitle', style: 'font-size:12px;color:var(--ink2);margin-top:10px' }},
        'all observe steps, in order'));
      det.append(el('div', {{ class: 'maps' }},
        ...steps(c).map(st => mapCell(st, sample.image_file, `#${{st.step_index}}`))));
    }});
    mapsBox = det;
  }}

  return el('div', {{ class: 'compl' }}, chips, body,
    ans ? el('div', {{ class: 'answer' }}, el('span', {{ class: 'chip' }}, 'answer'), ' ', ans) : null,
    mapsBox,
    el('details', {{}}, el('summary', {{}}, `per-step table (${{steps(c).length}} steps)`), stepTbl));
}}

function renderSamples() {{
  const host = $('#samples');
  const [lo, hi] = heatScale();
  let list = DATA.models[S.model].samples.slice();
  const q = S.q.trim().toLowerCase();
  if (q) list = list.filter(s => (s.question + ' ' + s.completions.map(c => c.text).join(' ')).toLowerCase().includes(q));
  if (S.dupOnly) list = list.filter(s => s.completions.some(c => dupFrac(c) > 0));
  const key = {{
    row: s => s.row_index,
    dup: s => -Math.max(...s.completions.map(dupFrac)),
    steps: s => -Math.max(...s.completions.map(c => c.n_observe_steps_total)),
    len: s => -Math.max(...s.completions.map(c => c.n_completion_tokens)),
    overlap: s => -Math.max(...s.completions.map(c => c.rewards.think_overlap_reward ?? -9)),
  }}[S.sort];
  list.sort((a, b) => key(a) - key(b));

  const scaleBar = el('div', {{ class: 'scale' }}, `${{S.metric}} `,
    ...[0, .25, .5, .75, 1].map(f => el('i', {{ style: `background:${{seqColour(lo + f * (hi - lo), lo, hi)}}` }})),
    ` ${{fmt(lo, 2)}} → ${{fmt(hi, 3)}} · dotted underline = grounding failed (skipped) · grey = not an observe step`);

  const cards = list.map(s => el('div', {{ class: 'card' }},
    el('div', {{ class: 'samplehead' }},
      el('img', {{ class: 'sampleimg', src: s.image_file, alt: `image for: ${{s.question}}`, loading: 'lazy' }}),
      el('div', {{}},
        el('h3', {{}}, `${{s.dataset}} · question_id ${{s.question_id}} · row ${{s.row_index}}`),
        el('p', {{ style: 'margin:0 0 6px;font-size:15px' }}, s.question),
        el('p', {{ class: 'note' }}, 'ground truth: ', el('b', {{}}, s.gt_answer)))),
    ...s.completions.slice().sort((a, b) =>
      (b.rewards.think_overlap_reward ?? -9) - (a.rewards.think_overlap_reward ?? -9))
      .map(c => renderCompletion(c, s, lo, hi))));

  host.replaceChildren(scaleBar, ...cards);
  if (!cards.length) host.append(el('p', {{ class: 'note' }}, 'No samples match the current filters.'));
}}

function applyTableView() {{
  document.querySelectorAll('.tv').forEach(t => t.classList.toggle('hidden', !S.table));
  document.querySelectorAll('#repeat svg, #dist svg, #dist .legend').forEach(s => s.classList.toggle('hidden', S.table));
}}
function renderAll() {{ computeAbsRef(); renderSummary(); renderRepeat(); renderDist(); renderSamples(); }}

// ---------- wire up ----------
$('#runsub').textContent =
  `${{CFG.n_samples ?? '?'}} prompts × ${{CFG.num_generations ?? '?'}} completions · temperature ${{CFG.temperature}} · ` +
  `layer ${{CFG.overlap_layer}} heads [${{CFG.overlap_heads}}] token_reduction=${{CFG.token_reduction}} · ` +
  `metric ${{CFG.overlap_metric}} · box_threshold ${{CFG.box_threshold}} max_box_area ${{CFG.max_box_area}} · ` +
  `weights [${{CFG.reward_weights}}] · generation only, no training`;
$('#fModel').replaceChildren(...MODELS.map(m => el('option', {{ value: m }}, m)));
$('#fModel').value = S.model;
$('#fModel').onchange = e => {{ S.model = e.target.value; renderAll(); }};
$('#fMetric').onchange = e => {{ S.metric = e.target.value; renderRepeat(); renderSamples(); }};
$('#fHeat').onchange = e => {{ S.heat = e.target.value; renderSamples(); }};
$('#fBoxes').onchange = e => {{ S.boxes = e.target.value; renderSamples(); }};
$('#fOverlay').onchange = e => {{ S.overlay = e.target.value; renderSamples(); }};
$('#fSort').onchange = e => {{ S.sort = e.target.value; renderSamples(); }};
$('#fDup').onchange = e => {{ S.dupOnly = e.target.checked; renderSamples(); }};
let qt; $('#fQ').oninput = e => {{ clearTimeout(qt); qt = setTimeout(() => {{ S.q = e.target.value; renderSamples(); }}, 180); }};
$('#tblBtn').onclick = e => {{
  S.table = !S.table; e.target.setAttribute('aria-pressed', S.table);
  e.target.textContent = S.table ? 'Chart view' : 'Table view'; applyTableView();
}};
$('#themeBtn').onclick = () => {{
  const dark = document.documentElement.getAttribute('data-theme') === 'dark';
  document.documentElement.setAttribute('data-theme', dark ? 'light' : 'dark');
  $('#themeBtn').textContent = dark ? 'Dark' : 'Light';
  document.body.dataset.mode = dark ? 'light' : 'dark';
  document.body.dataset.palette = dark ? '{PALETTE_LIGHT}' : '{PALETTE_DARK}';
  document.body.dataset.surface = dark ? '{LIGHT['surface']}' : '{DARK['surface']}';
  renderAll();
}};
renderAll();
</script>
{vjs}
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")
    return out_path


def main():
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out_dir", help="probe run directory containing probe_merged.json")
    args = p.parse_args()
    d = Path(args.out_dir)
    merged = json.loads((d / "probe_merged.json").read_text())
    out = write_html(merged, d / "probe_report.html", REPO / "assets/validate_palette.js")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
