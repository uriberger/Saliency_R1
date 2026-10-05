#!/usr/bin/env python
"""Peak GPU memory against step number and against sequence length, from a job log.

`SR1_MEM_REPORT_EVERY=1 SR1_MEM_REPORT_RANKS=all` makes the trainer print one line per
micro-step per rank:

    [mem] r0 step   17 before the forward     allocated  61.6  reserved  61.7  peak  74.4 \
free  11.8  (of 79.2 GB)  tokens 1141

This turns that into a CSV and two panels. The question it exists to answer is the one the
2026-09-28 run could not: *was it about to fail?* Thirty steps completed and then every
resume died, and the only memory evidence on record was six lines from step 1 -- enough to
say the run STARTED with room, and nothing at all about whether the room was shrinking.

Two things about the panels that are not cosmetic:

  * **FREE is plotted, not just torch's own counters.** The failure was
    `ncclUnhandledCudaError: Cuda failure 2 'out of memory'` inside DDP's allreduce, with
    torch's peak at 68.0 GB of 79.2. NCCL allocates outside the caching allocator, so the
    margin that ran out is `cuda.mem_get_info()`'s free -- and reserved, not allocated, is
    what eats it, because the allocator does not hand pages back until it is told to.
  * **Max over ranks, not the main process.** Peak is per DEVICE and the ranks hold
    completions of very different lengths; a step fits only if the FULLEST card fits.
    `free` is therefore reduced with min and the other two with max.

Usage:
    python omni_mem_series.py <job.log> [--csv out.csv] [--png out.png] [--title T]
"""
from __future__ import annotations

import argparse
import collections
import re

# `where` is printed with {:<22}, so it is space-padded and may itself contain spaces.
MEM_RE = re.compile(
    r"\[mem\] r(?P<rank>\d+) step\s+(?P<step>-?\d+) (?P<where>.{1,22}?)\s{2,}"
    r"allocated\s+(?P<alloc>[\d.]+)\s+reserved\s+(?P<res>[\d.]+)\s+"
    r"peak\s+(?P<peak>[\d.]+)\s+free\s+(?P<free>[\d.]+)\s+\(of (?P<total>[\d.]+) GB\)"
    r"(?:\s+tokens (?P<tokens>\d+))?"
)
# The trainer's own per-step log dict, for completions/mean_length. The tqdm prefix that
# precedes it on the same line carries the step, which the dict itself does not.
STEP_RE = re.compile(r"\|\s*(\d+)/\d+ \[")
MEAN_LEN_RE = re.compile(r"'completions/mean_length': '([\d.e+]+)'")
MAX_LEN_RE = re.compile(r"'completions/max_length': '([\d.e+]+)'")


def parse(path):
    rows, lengths, total = [], {}, None
    # newline="\n" MATTERS. tqdm redraws its bar with a carriage return, and Python's
    # default universal-newline mode treats a bare \r as a line terminator -- which splits
    # the progress bar away from the `{'loss': ...}` dict that follows it on the same
    # physical line. The step number lives in the bar and the lengths live in the dict, so
    # in universal mode they are never seen together and every length comes out NaN.
    # `grep` splits on \n only, which is why the two look joined when you look by hand.
    with open(path, errors="replace", newline="\n") as fh:
        for line in fh:
            # finditer, NOT search: six ranks write `print(flush=True)` to one pipe, and
            # two of them landing in the same write shows up as two reports concatenated
            # on one line. `search` would keep the first and drop the second -- silently,
            # and biased towards keeping whichever rank got there first, which is the
            # opposite of the rank this file is trying to find.
            found = False
            for m in MEM_RE.finditer(line):
                found = True
                d = m.groupdict()
                total = float(d["total"])
                rows.append(dict(
                    rank=int(d["rank"]), step=int(d["step"]), where=d["where"].strip(),
                    alloc=float(d["alloc"]), reserved=float(d["res"]),
                    peak=float(d["peak"]), free=float(d["free"]),
                    tokens=int(d["tokens"]) if d["tokens"] else None,
                ))
            # NO `continue` HERE. The trainer's per-step log dict arrives on the same
            # stream as 7,200 [mem] reports, so it regularly shares a line with one --
            # skipping a line because it matched a memory report is how every
            # `completions/mean_length` in a 446,000-line log came out NaN.
            del found
            ml = MEAN_LEN_RE.search(line)
            if ml:
                st = STEP_RE.search(line)
                if st:
                    xl = MAX_LEN_RE.search(line)
                    lengths[int(st.group(1))] = (float(ml.group(1)),
                                                 float(xl.group(1)) if xl else float("nan"))
    return rows, lengths, total


# `state.global_step` is incremented AFTER the optimizer step, so a [mem] line printed
# during the first step says 0 while tqdm's bar says 1/50. Align on the tqdm numbering,
# which is what the trainer's own logs and wandb use.
def length_for(lengths, mem_step):
    return lengths.get(mem_step + 1, (float("nan"), float("nan")))


def per_step(rows):
    """-> {step: (peak_max, reserved_max, free_min, alloc_max, n_ranks)}."""
    by = collections.defaultdict(list)
    for r in rows:
        by[r["step"]].append(r)
    out = {}
    for step, rs in by.items():
        out[step] = (
            max(r["peak"] for r in rs),
            max(r["reserved"] for r in rs),
            min(r["free"] for r in rs),
            max(r["alloc"] for r in rs),
            len({r["rank"] for r in rs}),
        )
    return out


# The reference palette's categorical slots 1/2/3, in fixed order, validated for CVD
# (worst adjacent pair deltaE 9.2 deutan / 27.6 normal). Every line is also direct-labelled,
# which is what discharges aqua's sub-3:1 contrast warning against the light surface.
C_PEAK, C_RESERVED, C_FREE = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
CRITICAL = "#d03b3b"


def plot(steps, rows, total, png, title, lengths_for=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17.5, 4.6), facecolor=SURFACE)
    for ax in (ax1, ax2, ax3):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#c3c2b7")
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)

    # ---- panel A: the series, over the run -----------------------------------------
    # Only RESERVED and FREE are plotted as series, and that is a correctness point, not
    # a layout one. `max_memory_allocated()` is a high-water mark since the process
    # started, so a "peak per step" line built from it can only ever be flat or rising
    # and says nothing about the step it is drawn at. It goes in as a horizontal
    # reference instead, which is the only honest reading of it.
    xs = sorted(steps)
    series = [
        ("reserved", [steps[s][1] for s in xs], C_RESERVED),
        ("free (driver)", [steps[s][2] for s in xs], C_FREE),
    ]
    if total:
        ax1.axhline(total, color=CRITICAL, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
        # Right-hand end, in the margin the xlim below reserves for the direct labels --
        # at the left it lands on top of the y tick it is closest to.
        ax1.annotate(f" card total {total:.1f} GB", (xs[-1], total), xytext=(2, 4),
                     textcoords="offset points", color=CRITICAL,
                     fontsize=8.5, va="bottom", ha="left", annotation_clip=False)
    hwm = max(steps[s][0] for s in xs)
    ax1.axhline(hwm, color=C_PEAK, linewidth=1.1, linestyle=(0, (2, 3)), zorder=1)
    # LEFT end: the right margin is where the two series put their direct labels, and
    # reserved sits within a few GB of the high-water mark all run.
    ax1.annotate(f"peak allocated {hwm:.1f} (high-water mark) ", (xs[0], hwm), xytext=(0, -12),
                 textcoords="offset points", color=C_PEAK, fontsize=8.5,
                 va="top", ha="left", annotation_clip=False)
    for label, ys, color in series:
        ax1.plot(xs, ys, color=color, linewidth=2, zorder=3)
        ax1.annotate(f" {label}", (xs[-1], ys[-1]), color=color, fontsize=9,
                     va="center", ha="left", annotation_clip=False)
    ax1.set_xlabel("optimizer step", color=MUTED, fontsize=9.5)
    ax1.set_ylabel("GB", color=MUTED, fontsize=9.5)
    ax1.set_title("Per-step memory, worst rank of the six", color=INK,
                  fontsize=11, loc="left", pad=10)
    ax1.set_xlim(xs[0], xs[-1] + 0.22 * (xs[-1] - xs[0] + 1))

    # ---- panel B: the working set against the sequence it is a function of ----------
    # RESERVED at "after the forward", not peak: reserved is instantaneous, so it varies
    # with the micro-step actually being measured. Peak is cumulative and would draw a
    # ceiling rather than a relationship.
    pts = [(r["tokens"], r["reserved"]) for r in rows
           if r["tokens"] and r["where"] == "after the forward"]
    if pts:
        tx, ty = zip(*pts)
        ax2.scatter(tx, ty, s=26, color=C_PEAK, alpha=0.55, linewidths=0, zorder=3)
        if len(set(tx)) > 1:
            import numpy as np
            b, a = np.polyfit(tx, ty, 1)
            xr = np.array([min(tx), max(tx)])
            ax2.plot(xr, a + b * xr, color=INK, linewidth=1.4, zorder=4)
            # Above the LEFT end: the right end is where the cap-length micro-steps pile
            # up, and a label there sits on the densest part of the scatter.
            # Axes coordinates, top-left: the fit runs corner to corner, so any label
            # anchored ON it lands on the line or on the scatter.
            ax2.text(0.03, 0.95, f"{b * 1000:+.2f} GB per 1,000 tokens",
                     transform=ax2.transAxes, color=INK, fontsize=9,
                     va="top", ha="left")
        if total:
            ax2.axhline(total, color=CRITICAL, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    ax2.set_xlabel("tokens forwarded by the micro-step (prompt + trimmed completion)",
                   color=MUTED, fontsize=9.5)
    ax2.set_ylabel("reserved after the forward, GB", color=MUTED, fontsize=9.5)
    ax2.set_title("Working set against sequence length", color=INK,
                  fontsize=11, loc="left", pad=10)

    # ---- panel C: the one the handoff asked for, and why it is the weaker view -------
    # `completions/mean_length` is a mean over the whole 48-rollout generation batch, and
    # what fills a card is the LONGEST rollout on the fullest rank. So this relationship
    # is real but loose, and panel B is the same physics measured on the right variable.
    lengths_for = lengths_for or {}
    ml = [(lengths_for[s_ + 1][0], steps[s_][2]) for s_ in sorted(steps)
          if s_ + 1 in lengths_for]
    if ml:
        mx, my = zip(*ml)
        ax3.scatter(mx, my, s=34, color=C_FREE, alpha=0.75, linewidths=0, zorder=3)
        if len(set(mx)) > 1:
            import numpy as np
            r = float(np.corrcoef(mx, my)[0, 1])
            b, a = np.polyfit(mx, my, 1)
            xr = np.array([min(mx), max(mx)])
            ax3.plot(xr, a + b * xr, color=INK, linewidth=1.4, zorder=4)
            # Put the label in whichever half the points are NOT in. Free memory sits at
            # the top of this panel on a healthy run and at the bottom on a starved one,
            # so a fixed corner collides with the data on one of the two.
            hi = (sum(my) / len(my) - min(my)) > 0.5 * (max(my) - min(my) or 1)
            ax3.text(0.03, 0.05 if hi else 0.95, f"r = {r:+.3f}   (n = {len(ml)} steps)",
                     transform=ax3.transAxes, color=INK, fontsize=9,
                     va="bottom" if hi else "top", ha="left")
        ax3.axhline(0, color=CRITICAL, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
        ax3.annotate(" card full", (max(mx), 0), xytext=(2, 4), textcoords="offset points",
                     color=CRITICAL, fontsize=8.5, va="bottom", ha="right",
                     annotation_clip=False)
    ax3.set_xlabel("completions/mean_length for the step", color=MUTED, fontsize=9.5)
    ax3.set_ylabel("free at its worst, GB", color=MUTED, fontsize=9.5)
    ax3.set_title("Headroom against the batch's mean length", color=INK,
                  fontsize=11, loc="left", pad=10)

    fig.suptitle(title, color=INK, fontsize=12.5, x=0.005, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(png, dpi=160, facecolor=SURFACE)
    print(f"wrote {png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--png", default=None)
    ap.add_argument("--title", default="Omni GRPO — per-rank CUDA memory")
    args = ap.parse_args()

    rows, lengths, total = parse(args.log)
    if not rows:
        raise SystemExit(f"no [mem] lines in {args.log} — was SR1_MEM_REPORT_EVERY set?")
    steps = per_step(rows)
    xs = sorted(steps)
    print(f"{len(rows)} reports, steps {xs[0]}..{xs[-1]}, "
          f"{steps[xs[0]][4]} ranks, card {total} GB")
    print(f"\n{'step':>5} {'peak':>6} {'reserv':>7} {'free':>6} {'meanlen':>8} {'maxlen':>7}")
    for s in xs:
        p, r, f, _a, _n = steps[s]
        ml, xl = length_for(lengths, s)
        print(f"{s:>5} {p:>6.1f} {r:>7.1f} {f:>6.1f} {ml:>8.0f} {xl:>7.0f}")

    if args.csv:
        with open(args.csv, "w") as fh:
            fh.write("step,peak_max,reserved_max,free_min,alloc_max,mean_length,max_length\n")
            for s in xs:
                p, r, f, a, _n = steps[s]
                ml, xl = length_for(lengths, s)
                fh.write(f"{s},{p:.2f},{r:.2f},{f:.2f},{a:.2f},{ml:.1f},{xl:.0f}\n")
        print(f"wrote {args.csv}")
    if args.png:
        plot(steps, rows, total, args.png, args.title, lengths)


if __name__ == "__main__":
    main()
