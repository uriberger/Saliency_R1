#!/usr/bin/env python
"""Three models x three query sets: the patch-set tables and the nine heatmaps.

    python sink_location_xmodel_tables.py --dirs A,B,C --out-dir DIR

WHAT IT PRODUCES

  tables.md / tables.txt   three tables, one per query set, a row per model and a column
                           per patch set: the border ring, its four sides, its four
                           corners, and the centre (everything the ring is not).
  heat_<qset>_<family>.png/.pdf    nine heatmaps, one per (query set, model).
  heat_qwen3_vl_modalgrid.png      the validation panel described below.

THE THREE QUERY SETS. All three come out of ONE forward pass per picture -- the model
writes an answer at full speed, then a single teacher-forced pass over prompt ++ answer is
measured -- so they differ only in which rows of the attention matrix were averaged, never
in the pictures or the weights:

    prompt      the prompt's tokens AFTER the image: the question and the assistant header
    generated   the tokens the model WROTE
    all         both, i.e. every token after the image. Column sums are additive and the
                row counts are added with them, so this is the exact union and not an
                average of two averages (which would silently re-weight a 15-token
                question against a 250-token answer)

EVERY NUMBER IS AN ENRICHMENT: a patch set's share of the picture's attention divided by
its share of the patches. 1.00 is exactly a fair share. Raw percentages are never
comparable here -- the one-patch border is 23% of a 16x16 grid and 16% of a 24x24 one.

THE HEATMAPS ARE THE TABLES. Both are computed from `sink_location.pooled_patch_map`,
which is the all-head table taken per patch rather than per patch set, so summing a
heatmap over any set reproduces that set's table entry exactly. A figure that merely
"illustrates" a table can drift away from it; this one cannot.

QWEN3-VL HAS NO SINGLE GRID -- see `resample_map` for what is done about it.
"""

from __future__ import annotations

import argparse
import importlib.util
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


import sink_location as SL  # noqa: E402

#: (label, the stats field, the pooled-map field)
Q_SETS = (("prompt tokens", "stats", "map_q"),
          ("generated tokens", "stats_gen", "map_gen"),
          ("all tokens after the image", "stats_all", "map_all"))

#: (column header, the stat that carries its share, how to get its area share)
#: `centre` is not a stored statistic: the ring and the interior partition the picture, so
#: the interior's share is 1 - the ring's, exactly, and deriving it beats storing it.
COLUMNS = (
    ("ring", "ring_share", lambda gh, gw: SL.ring_area_frac(gh, gw)),
    ("top", "top_share", lambda gh, gw: gw / (gh * gw)),
    ("bottom", "bottom_share", lambda gh, gw: gw / (gh * gw)),
    ("left", "left_share", lambda gh, gw: gh / (gh * gw)),
    ("right", "right_share", lambda gh, gw: gh / (gh * gw)),
    ("TL", "corner_tl_share", lambda gh, gw: 1.0 / (gh * gw)),
    ("TR", "corner_tr_share", lambda gh, gw: 1.0 / (gh * gw)),
    ("BL", "corner_bl_share", lambda gh, gw: 1.0 / (gh * gw)),
    ("BR", "corner_br_share", lambda gh, gw: 1.0 / (gh * gw)),
    ("centre", None, lambda gh, gw: 1.0 - SL.ring_area_frac(gh, gw)),
)

#: The documented diverging pair: two poles that read as opposite, with a NEUTRAL gray
#: midpoint -- never a hue at the middle, which would make "a fair share" look like a
#: value. Blue is below fair share, red above.
DIVERGING = ("#2a78d6", "#f0efec", "#e34948")
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"

#: The heatmap's colour range, in doublings either side of a fair share. Symmetric, as a
#: diverging scale must be. Anything beyond it is drawn at the end of the ramp AND
#: labelled with its own number, so a clipped cell never silently reads as "the maximum".
CLIP = 2.0


# ---------------------------------------------------------------------------
def resample_map(p, gh, gw, GH, GW):
    """A patch map on a gh x gw grid, onto a GH x GW lattice. Mass-preserving.

    QWEN3-VL'S GRID DEPENDS ON THE PICTURE -- 104 distinct shapes over this corpus -- so
    there is no single grid to draw its heatmap on. Three things were considered:

      resample (this)   every statistic in the tables is already defined by RELATIVE
                        position: "the ring" is the outermost patch of whatever grid the
                        picture got, "the top row" is row 0 of gh. Normalised coordinates
                        are therefore the frame the claim is already made in, and the
                        common lattice only renders it. The resample is an exact
                        area-weighted redistribution -- each source cell's mass is split
                        among the target cells it overlaps, in proportion to the overlap
                        -- so it is mass-preserving and introduces no interpolation.
      modal grid only   exact, but n is small and near-square pictures are a biased
                        subsample of the corpus. Kept as a VALIDATION panel: if the
                        resampled map and the modal-grid map agree, the resampling is not
                        doing any work.
      one panel per common grid shape   honest, and impossible to read against a single
                        panel from each of the other two models.

    What resampling cannot show is an effect that lives at an absolute TOKEN INDEX rather
    than at a relative position -- two pictures' 40th tokens land in different lattice
    cells. The modal-grid panel is the guard against that, and the first/last-patch
    columns of the tables are measured on each picture's own grid and never resampled.
    """
    def overlap(n_src, n_dst):
        es, ed = np.linspace(0, 1, n_src + 1), np.linspace(0, 1, n_dst + 1)
        hi = np.minimum(es[1:, None], ed[None, 1:])
        lo = np.maximum(es[:-1, None], ed[None, :-1])
        return np.clip(hi - lo, 0.0, None)              # [n_src, n_dst]

    P = np.asarray(p, dtype=np.float64).reshape(gh, gw)
    out = overlap(gh, GH).T @ P @ overlap(gw, GW)
    s = out.sum()
    return out * (P.sum() / s) if s > 0 else out


def model_maps(meta, arrays, field, lattice=None, only_grid=None):
    """Mean pooled map per model, on a common lattice. -> ([GH, GW] enrichment, n, grids)"""
    acc, n, grids = None, 0, []
    for m in meta:
        p = arrays.get(m["unit"], {}).get(field)
        if p is None:
            continue
        gh, gw = m["grid"]
        if only_grid is not None and (gh, gw) != tuple(only_grid):
            continue
        p = np.asarray(p, dtype=np.float64)
        if not np.isfinite(p).any() or p.sum() <= 0:
            continue
        p = p / p.sum()
        GH, GW = lattice if lattice else (gh, gw)
        q = resample_map(p, gh, gw, GH, GW) if (gh, gw) != (GH, GW) else p.reshape(gh, gw)
        acc = q if acc is None else acc + q
        n += 1
        grids.append((gh, gw))
    if acc is None:
        return None, 0, []
    mean = acc / n
    return mean * mean.size, n, grids      # -> enrichment: 1.0 is a fair share


def table_rows(meta, arrays, field, min_mass):
    """One row of one table: every column's enrichment, pooled over pictures."""
    vals = {c: [] for c, _s, _a in COLUMNS}
    for m in meta:
        a = arrays.get(m["unit"], {}).get(field)
        if a is None:
            continue
        a = np.asarray(a, dtype=np.float64)
        live = a[..., SL.STAT_INDEX["image_mass"]] >= min_mass
        if not live.any():
            continue
        gh, gw = m["grid"]
        ring = float(np.nanmean(np.where(live, a[..., SL.STAT_INDEX["ring_share"]],
                                         np.nan)))
        for col, stat, area in COLUMNS:
            share = (1.0 - ring) if stat is None else float(
                np.nanmean(np.where(live, a[..., SL.STAT_INDEX[stat]], np.nan)))
            vals[col].append(share / area(gh, gw))
    return ({c: float(np.nanmean(v)) if v else float("nan") for c, v in vals.items()},
            max(len(v) for v in vals.values()))


# ---------------------------------------------------------------------------
def draw(mat, title, subtitle, path, note=""):
    """One heatmap. Diverging, gray at a fair share, symmetric, extremes labelled."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    GH, GW = mat.shape
    cmap = LinearSegmentedColormap.from_list("fair", DIVERGING)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.log2(np.where(mat > 0, mat, np.nan))

    fig, ax = plt.subplots(figsize=(4.6, 4.9), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    im = ax.imshow(np.clip(z, -CLIP, CLIP), cmap=cmap, vmin=-CLIP, vmax=CLIP,
                   interpolation="nearest")
    # a thin surface gap between cells, so adjacent fills never bleed into one another
    ax.set_xticks(np.arange(-0.5, GW, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, GH, 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=0.5)
    ax.tick_params(which="both", length=0)
    step = max(1, GW // 8)
    ax.set_xticks(range(0, GW, step)); ax.set_yticks(range(0, GH, step))
    ax.tick_params(colors=MUTED, labelsize=7)
    for s in ax.spines.values():
        s.set_visible(False)

    # Direct-label every cell the scale could not hold, so a clipped cell is never read
    # as "the maximum" -- and the four corners always, since they are the story.
    hot = np.argwhere(np.abs(z) > CLIP)
    corners = {(0, 0), (0, GW - 1), (GH - 1, 0), (GH - 1, GW - 1)}
    # A label has to fit inside its own cell. On a 24x24 grid the cells are 2/3 the width
    # of a 16x16 one's, so the type scales with them and values past 10 drop the decimal
    # -- otherwise two adjacent clipped cells run their numbers together, which is what a
    # screenshot catches and no colour validator does.
    fs = float(np.clip(5.5 * 16.0 / GW, 3.4, 6.0))
    fmt = lambda v: f"{v:.0f}" if abs(v) >= 10 else f"{v:.1f}"       # noqa: E731
    for r, c in sorted(corners | {tuple(x) for x in hot}):
        v = mat[r, c]
        if not np.isfinite(v):
            continue
        ax.text(c, r, fmt(v), ha="center", va="center", fontsize=fs,
                color="#ffffff" if abs(z[r, c]) > 1.1 else INK,
                fontweight="bold" if abs(z[r, c]) > CLIP else "normal")

    # The title sits above the subtitle, which sits above the plot. `pad` has to clear
    # BOTH or the two overlap -- which is exactly what a screenshot catches and no
    # validator does.
    ax.set_title(title, fontsize=10, color=INK, pad=26, loc="left")
    ax.text(0, 1.018, subtitle, transform=ax.transAxes, fontsize=7.5, color=MUTED,
            va="bottom")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                      ticks=np.log2([0.25, 0.5, 1, 2, 4]))
    cb.ax.set_yticklabels(["0.25x", "0.5x", "fair share", "2x", "4x"], fontsize=7,
                          color=MUTED)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0)
    if note:
        fig.text(0.01, 0.015, note, fontsize=6.5, color=MUTED, va="bottom")
    fig.tight_layout(rect=(0, 0.03 if note else 0, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", required=True, help="comma-separated scan directories")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-mass", type=float, default=0.002)
    ap.add_argument("--lattice", type=int, default=16,
                    help="the common lattice a variable grid is resampled onto")
    args = ap.parse_args()

    P = _load("_xm_probe", "sink_location_probe.py")
    out = Path(args.out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    runs = []
    for d in [x for x in args.dirs.split(",") if x]:
        meta, arrays = P.read_stage(d, "scan")
        if not meta:
            print(f"(skipping {d}: no scan results)")
            continue
        runs.append({"dir": d, "meta": meta, "arrays": arrays,
                     "family": P._family_of(meta)})
    if not runs:
        raise SystemExit("no results")

    lines = ["# Attention inside the picture: three models, three query sets", ""]
    lines += [
        "Every entry is an **enrichment**: that patch set's share of the picture's "
        "attention", "divided by its share of the patches. **1.00 is exactly a fair "
        "share**, so a raw", "percentage never appears -- the one-patch border is 23% of "
        "a 16x16 grid and 16% of a", "24x24 one.  Averaged over **all** layers and heads "
        "that clear an image-mass floor of",
        f"{args.min_mass} (a head that puts no weight on the picture still has a ring "
        "share, and it is", "noise wearing a statistic's name); no head is selected.  "
        "`centre` is everything the", "ring is not.  TL/TR/BL/BR are single patches, so "
        "they are priced against a flat map's", "1/N and run on a different scale from "
        "the block columns beside them.", ""]

    # How long the answers were, per model. The `generated` row is an average over
    # whatever the model wrote, and the picture is the unit of analysis, so a two-token
    # answer weighs as much as a 256-token one. On a model that answers tersely that
    # makes the row noisier -- not biased, but it has to be visible next to the number.
    lines += ["## How much the models wrote", "",
              "The `generated` rows average over the tokens each model actually produced,"
              " capped at 256.",
              "The picture is the unit of analysis throughout, so a two-token answer"
              " weighs as much as a",
              "256-token one; a model that answers tersely therefore has a noisier"
              " `generated` row.", "",
              "| model | median | p10 | p90 | share under 10 tokens |", "|---|---|---|---|---|"]
    print("\n=== how much each model wrote (completion length, capped at 256) ===")
    for r in runs:
        n = np.array([m["n_generated"] for m in r["meta"] if m.get("n_generated")])
        if not n.size:
            continue
        row = (f"| {r['family']} | {np.median(n):.0f} | {np.percentile(n, 10):.0f} | "
               f"{np.percentile(n, 90):.0f} | {np.mean(n < 10):.1%} |")
        lines.append(row)
        print(f"{r['family']:<12} median {np.median(n):>4.0f}  p10 "
              f"{np.percentile(n, 10):>4.0f}  p90 {np.percentile(n, 90):>4.0f}  "
              f"under 10 tokens {np.mean(n < 10):.1%}")
    lines.append("")

    for label, field, mapfield in Q_SETS:
        lines += [f"## Query set: {label}", "",
                  "| model | n | grid | " + " | ".join(c for c, _s, _a in COLUMNS) + " |",
                  "|---|---|---|" + "---|" * len(COLUMNS)]
        print(f"\n=== {label} ===")
        print(f"{'model':<12} {'n':>5} {'grid':>9} " +
              " ".join(f"{c:>8}" for c, _s, _a in COLUMNS))
        for r in runs:
            vals, n = table_rows(r["meta"], r["arrays"], field, args.min_mass)
            if not n:
                continue
            grids = [tuple(m["grid"]) for m in r["meta"]]
            g = (f"{max(set(grids), key=grids.count)[0]}x"
                 f"{max(set(grids), key=grids.count)[1]}"
                 + ("*" if len(set(grids)) > 1 else ""))
            lines.append(f"| {r['family']} | {n} | {g} | " +
                         " | ".join(f"{vals[c]:.2f}" for c, _s, _a in COLUMNS) + " |")
            print(f"{r['family']:<12} {n:>5} {g:>9} " +
                  " ".join(f"{vals[c]:>8.2f}" for c, _s, _a in COLUMNS))
        lines.append("")
        lines.append("\\* the grid varies per picture; the modal shape is shown.")
        lines.append("")

        for r in runs:
            lat = (args.lattice, args.lattice)
            grids = {tuple(m["grid"]) for m in r["meta"]}
            mat, n, _g = model_maps(r["meta"], r["arrays"], mapfield,
                                    lattice=lat if len(grids) > 1 else None)
            if mat is None:
                continue
            slug = f"heat_{field}_{r['family']}"
            note = ("" if len(grids) == 1 else
                    f"{len(grids)} distinct grids, area-resampled onto a "
                    f"{lat[0]}x{lat[1]} lattice in normalised coordinates")
            draw(mat, f"{r['family']} - {label}",
                 f"n={n} pictures, all heads, enrichment over a fair share",
                 str(out / "figures" / slug), note)
            print(f"  wrote figures/{slug}.png")

    # the validation panel: Qwen3-VL on its modal grid alone, no resampling at all
    for r in runs:
        grids = [tuple(m["grid"]) for m in r["meta"]]
        if len(set(grids)) == 1:
            continue
        modal = max(set(grids), key=grids.count)
        mat, n, _g = model_maps(r["meta"], r["arrays"], "map_q", only_grid=modal)
        if mat is None or n < 20:
            continue
        draw(mat, f"{r['family']} - prompt tokens, modal grid only",
             f"n={n} pictures whose grid really is {modal[0]}x{modal[1]}",
             str(out / "figures" / f"heat_modalgrid_{r['family']}"),
             "validation panel: no resampling. If this agrees with the resampled "
             "figure, the resampling is not doing any work.")
        print(f"  wrote figures/heat_modalgrid_{r['family']}.png")

    (out / "tables.md").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out / 'tables.md'} and {len(runs) * len(Q_SETS)} figures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
