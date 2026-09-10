/* Layout check for docs/sink-location.html, with no browser on the box.
 *
 *   node assets/check_sink_location_html.js docs/sink-location.html
 *
 * The palette validator checks colour; it says nothing about whether a bar runs off the
 * canvas or a label lands on top of its neighbour. Without a headless browser the next
 * best thing is to RUN the page's own drawing code against a DOM stub and measure what it
 * emitted: every coordinate finite, every mark inside its own viewBox, every label with
 * room for its text, and every figure actually drawn. That is most of what eyeballing a
 * screenshot would have caught, and unlike a screenshot it fails loudly in CI.
 */
"use strict";
const fs = require("fs");

const file = process.argv[2] || "docs/sink-location.html";
const html = fs.readFileSync(file, "utf8");
const body = html.slice(html.lastIndexOf("<script>") + 8, html.lastIndexOf("</script>"));

const CSS = {   // the light-mode custom properties, as the stylesheet declares them
  "--surface-1": "#fcfcfb", "--plane": "#f9f9f7", "--text-primary": "#0b0b0b",
  "--text-secondary": "#52514e", "--muted": "#898781", "--grid": "#e1e0d9",
  "--axis": "#c3c2b7", "--border": "rgba(11,11,11,0.10)", "--series-1": "#2a78d6",
  "--series-2": "#eb6834", "--neutral-mid": "#f0efec", "--pole-hi": "#0d366b",
  "--pole-lo": "#a32a2a", "--good": "#0ca30c", "--critical": "#d03b3b",
  "--warning": "#fab219",
};

const svgs = [];        // every <svg> made, with its viewBox and children
class Node {
  constructor(tag) {
    this.tag = tag; this.attrs = {}; this.children = []; this.style = {};
    this.classList = {add(){}, remove(){}, toggle(){ return true; }};
    this.dataset = {}; this.textContent = ""; this.innerHTML = "";
  }
  setAttribute(k, v) { this.attrs[k] = v; }
  getAttribute(k) { return this.attrs[k]; }
  appendChild(c) { this.children.push(c); c.parent = this; return c; }
  addEventListener() {}
  querySelectorAll() { return []; }
}
const byId = {};
global.document = {
  documentElement: {getAttribute: () => "light", setAttribute() {}, removeAttribute() {}},
  createElementNS(ns, tag) {
    const n = new Node(tag);
    if (tag === "svg") svgs.push(n);
    return n;
  },
  createElement(tag) { return new Node(tag); },
  getElementById(id) { return byId[id] || (byId[id] = new Node("div")); },
  querySelectorAll() { return []; },
};
global.getComputedStyle = () => ({getPropertyValue: k => CSS[k] || "#808080"});
global.localStorage = {getItem: () => null, setItem() {}};
global.matchMedia = () => ({addEventListener() {}});
global.innerWidth = 1400;

let fail = 0;
const bad = (m) => { console.log("  FAIL  " + m); fail++; };
try { new Function(body)(); } catch (e) { bad("the page's script threw: " + e.message); }

console.log(`drew ${svgs.length} figures from ${file}`);
// eight SVG figures; the ninth (the hypothesis scorecard) is HTML chips, by design
if (svgs.length !== 8) bad(`${svgs.length} SVG figures drew; expected 8`);

const NUMERIC = ["x", "y", "x1", "x2", "y1", "y2", "cx", "cy", "r", "width", "height"];
for (const [i, s] of svgs.entries()) {
  const vb = (s.attrs.viewBox || "0 0 0 0").split(" ").map(Number);
  const [, , W, H] = vb;
  let nodes = 0, texts = 0, minX = Infinity, maxX = -Infinity, maxY = -Infinity;
  const walk = (n) => {
    nodes++;
    for (const k of NUMERIC) {
      if (n.attrs[k] === undefined) continue;
      const v = Number(n.attrs[k]);
      if (!isFinite(v)) bad(`figure ${i + 1}: <${n.tag}> ${k}="${n.attrs[k]}" is not finite`);
    }
    if (n.attrs.d && /NaN|Infinity|undefined/.test(n.attrs.d))
      bad(`figure ${i + 1}: <path> d has a non-finite coordinate`);
    if (n.attrs.d) {
      // only the absolute move-to; the rest of a path is RELATIVE, and treating `h-140`
      // as an x coordinate reported every bar as running off the left edge
      const m0 = String(n.attrs.d).match(/^M([-\d.]+),([-\d.]+)/);
      if (m0) { minX = Math.min(minX, +m0[1]); maxX = Math.max(maxX, +m0[1]);
                maxY = Math.max(maxY, +m0[2]); }
    }
    for (const k of ["x", "x1", "x2", "cx"]) if (n.attrs[k] !== undefined) {
      const v = Number(n.attrs[k]); minX = Math.min(minX, v); maxX = Math.max(maxX, v);
    }
    for (const k of ["y", "y1", "y2", "cy"]) if (n.attrs[k] !== undefined)
      maxY = Math.max(maxY, Number(n.attrs[k]));
    if (n.tag === "text") {
      texts++;
      // 6.6px per char at 11.5-12.5px in the system sans, generous
      const w = String(n.textContent).length * 6.6;
      const anchor = n.attrs["text-anchor"] || "start";
      const x = Number(n.attrs.x);
      const left = anchor === "end" ? x - w : anchor === "middle" ? x - w / 2 : x;
      const right = left + w;
      if (left < -8) bad(`figure ${i + 1}: text "${n.textContent}" runs ${(-left).toFixed(0)}px off the left edge`);
      if (right > W + 8) bad(`figure ${i + 1}: text "${n.textContent}" runs ${(right - W).toFixed(0)}px past the right edge (viewBox ${W})`);
    }
    n.children.forEach(walk);
  };
  s.children.forEach(walk);
  if (maxY > H + 2) bad(`figure ${i + 1}: content reaches y=${maxY.toFixed(0)} in a ${H}-tall viewBox`);
  if (maxX > W + 2) bad(`figure ${i + 1}: content reaches x=${maxX.toFixed(0)} in a ${W}-wide viewBox`);
  if (nodes < 4) bad(`figure ${i + 1}: only ${nodes} marks -- did its data arrive?`);
  console.log(`  figure ${i + 1}: ${nodes} marks, ${texts} labels, ` +
    `x ${minX.toFixed(0)}..${maxX.toFixed(0)} of ${W}, y max ${maxY.toFixed(0)} of ${H}`);
}
console.log(fail ? `\n${fail} LAYOUT PROBLEMS` : "\nlayout OK");
process.exit(fail ? 1 : 0);
