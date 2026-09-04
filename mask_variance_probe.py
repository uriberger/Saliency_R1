#!/usr/bin/env python
"""What does each MASK SOURCE leave in the overlap reward, and where does its mask sit?

    python mask_variance_probe.py outputs/overlap_probe/<dir>/probe_merged.json \
        --question-boxes outputs/question_boxes/val_natural_bt0.10.json

CPU only, a few seconds, no GPU and no Grounding-DINO. Everything comes off disk:
`overlap_probe.py --store-maps` already wrote, per observe step, the attention map
quantised to its own peak (`map_q`) with the peak alongside (`map_max`), the DINO union
raster (`mask_q`) and the patch grid.

WHY THIS EXISTS. The reward's mask can come from four places -- once per step (the
incumbent), once per completion (--overlap_chain_boxes), once per row
(--overlap_question_boxes) or from no detector at all (--overlap_rect_frac) -- and the
choice is usually argued from how well the mask resembles DINO's. That is the wrong
question. GRPO subtracts the group mean before anything else, so the ONLY thing a reward
contributes is how it varies BETWEEN the 8 rollouts of one prompt. A mask that is the
same for all 8 cannot contribute through the mask at all: it cancels, and what is left is
a property of the map alone.

So the three columns that decide a mask source are:

    sd_within   the pooled within-group sd the trainer logs as
                rewards/<f>/within_group_std -- the pressure the arm actually applies
    r(flat)     group-centred correlation with `flatness` = mean(m)/max(m), the mask-free
                statistic --maskfree flatness rewards. Near 1 means the arm is a flatness
                run wearing a box's name.
    ring        two numbers about the grid's one-patch border, which is where `mean_in`'s
                denominator lives (76-85% of map peaks sit on it), reported separately
                because they are easy to confuse:
                  ringshr  share of the MASK that is border patches (0 = the mask is
                           entirely interior). This is what mask/ring_frac logs.
                  ringcov  share of the BORDER the mask covers. This is the measure
                           peak-location-results.md quotes for DINO unions.

Every mask is built by importing trl/rewards/overlap_rewards.py, not by reimplementing
it, so a number here is a number about the code a run will execute. Same reason
test_rect_reward_cpu.py compares the shipped rectangle against centre_box_probe.py's.

SCHEMES. `true` is the incumbent. `qbox` and `rect*` are the fixed-mask arms. `chain_*`
are --overlap_chain_boxes. `flat` is the floor: no mask at all.
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import math
import os
import sys
import types

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# the shipped mask code, imported rather than copied
# ---------------------------------------------------------------------------
def _import_overlap_rewards():
    """trl/rewards/overlap_rewards.py without importing the `trl` package.

    The package pulls torch, transformers and the trainer; the reward module itself needs
    only numpy. Same trick test_rect_reward_cpu.py uses.
    """
    pkg = types.ModuleType("trl_mvp"); pkg.__path__ = [os.path.join(ROOT, "trl")]
    sys.modules["trl_mvp"] = pkg
    sub = types.ModuleType("trl_mvp.rewards")
    sub.__path__ = [os.path.join(ROOT, "trl", "rewards")]
    sys.modules["trl_mvp.rewards"] = sub
    for name in ("roll_null", "overlap_rewards"):
        spec = importlib.util.spec_from_file_location(
            f"trl_mvp.rewards.{name}", os.path.join(ROOT, "trl", "rewards", f"{name}.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"trl_mvp.rewards.{name}"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["trl_mvp.rewards.overlap_rewards"]


ORW = _import_overlap_rewards()

F_FIXED = 0.565      # the fraction the centre_box_probe used, and the arms are launched at
MAX_BOX_AREA = 0.5   # the probe runs' per-box cap; the stored masks are already past it


# ---------------------------------------------------------------------------
# stored bytes -> arrays
# ---------------------------------------------------------------------------
def _u8(b64):
    return np.frombuffer(base64.b64decode(b64), dtype=np.uint8)


def decode_mask(b64, gh, gw):
    return _u8(b64).astype(bool).reshape(gh, gw)


def decode_map(b64, gh, gw, mx):
    """The absolute map: the stored byte is the patch's value as a fraction of the peak."""
    return _u8(b64).astype(np.float64).reshape(gh, gw) * (float(mx) / 255.0)


class Step:
    __slots__ = ("gh", "gw", "mask", "smap", "stored", "text")

    def __init__(self, rec):
        self.gh, self.gw = rec["grid"]
        self.mask = decode_mask(rec["mask_q"], self.gh, self.gw)
        self.smap = decode_map(rec["map_q"], self.gh, self.gw, rec.get("map_max") or 0.0)
        self.stored = rec.get("mean_in_raw")
        self.text = rec.get("text") or ""


def load_chains(model_rec):
    """-> [(sample, [[Step, ...] per completion]) ...], completions with >=1 usable step."""
    out = []
    for s in model_rec["samples"]:
        comps = []
        for c in s["completions"]:
            steps = [Step(r) for r in (c.get("observe_steps") or [])
                     if r.get("mask_q") and r.get("grid") and r.get("map_q")]
            comps.append(steps)
        if any(comps):
            out.append((s, comps))
    return out


def qbox_lookup(path):
    """(dataset, question_id) -> raw box list, from a precompute_question_boxes.py file."""
    if not path:
        return {}
    d = json.load(open(path))
    out = {}
    for k, v in d["boxes"].items():
        ds, _split, qid = k.split("|")
        out[(ds, qid)] = v
    return out


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def pooled_sd(groups):
    """Exactly the trainer's rewards/<f>/within_group_std, over the same groups:
    sqrt(sum_g sum_i (r_gi - mean_g)^2 / sum_g (n_g - 1))."""
    ss = dof = 0.0
    for vals in groups:
        v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
        if v.size < 2:
            continue
        ss += float(((v - v.mean()) ** 2).sum())
        dof += v.size - 1
    return math.sqrt(ss / dof) if dof > 0 else float("nan")


def centred(groups):
    """Group-mean-removed values, concatenated -- what the advantage is linear in.

    One entry per completion always, NaN where a scheme had no mask for it, so the
    vectors of two schemes stay aligned and a correlation between them is over the
    completions BOTH could score. Dropping short groups instead would misalign them.
    """
    out = []
    for vals in groups:
        a = np.asarray([np.nan if v is None else v for v in vals], float)
        out.append(a - np.nanmean(a) if np.isfinite(a).sum() >= 2 else np.full(a.shape, np.nan))
    return np.concatenate(out) if out else np.zeros(0)


def pearson(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or x[ok].std() < 1e-15 or y[ok].std() < 1e-15:
        return float("nan")
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def closeness(a, b):
    """(IoU - chance)/(best - chance): 0 = the two mask SIZES explain the overlap, 1 = as
    identical as those sizes permit. step_box_similarity.py's measure."""
    na, nb, n = int(a.sum()), int(b.sum()), a.size
    u = float(np.logical_or(a, b).sum())
    if u <= 0 or n <= 0:
        return float("nan")
    io = float(np.logical_and(a, b).sum()) / u
    inter = na * nb / n
    ch = inter / (na + nb - inter) if (na + nb - inter) > 0 else float("nan")
    bs = min(na, nb) / max(na, nb) if max(na, nb) else float("nan")
    return (io - ch) / (bs - ch) if (bs - ch) > 1e-12 else float("nan")


def ring_mask(gh, gw):
    m = np.zeros((gh, gw), dtype=bool)
    m[0, :] = m[-1, :] = True
    m[:, 0] = m[:, -1] = True
    return m


# ---------------------------------------------------------------------------
# the schemes
# ---------------------------------------------------------------------------
SCHEMES = ["true", "chain_first", "chain_last", "qbox", "rect_centre", "rect_inctr",
           "rect_inhash", "rect_jitter", "flat"]

DESCRIBE = {
    "true": "per-step DINO union (incumbent)",
    "chain_first": "--overlap_chain_boxes first",
    "chain_last": "--overlap_chain_boxes last",
    "qbox": "--overlap_question_boxes",
    "rect_centre": "--overlap_rect_frac, centre",
    "rect_inctr": "... --overlap_rect_placement interior_centre",
    "rect_inhash": "... --overlap_rect_placement interior_hash",
    "rect_jitter": "(rejected) any in-frame offset",
    "flat": "no mask at all = flatness",
}


def _jitter_rect(gh, gw, rng, frac=F_FIXED):
    """The design this probe REJECTS: same rectangle, any in-frame offset. Kept so the
    rejection is a measurement rather than an assertion -- it is the obvious way to give
    each completion its own mask, and the ring column is why it is not used."""
    rows, cols = ORW._rect_dims(gh, gw, frac)
    return ORW._placed_rect_mask(gh, gw, rows, cols,
                                 int(rng.integers(gh - rows + 1)),
                                 int(rng.integers(gw - cols + 1)))


def masks_for(comp, sample_boxes, rng, text, frac):
    """{scheme: mask or None} for one completion. `true` is per step, so it is absent."""
    gh, gw = comp[0].gh, comp[0].gw
    out = {
        "chain_first": comp[0].mask,
        "chain_last": comp[-1].mask,
        "qbox": (ORW._union_mask(sample_boxes, gh, gw) if sample_boxes else None),
        "rect_centre": ORW._centre_rect_mask(gh, gw, frac),
        "rect_inctr": ORW._interior_rect_mask(gh, gw, frac),
        "rect_inhash": ORW._interior_rect_mask(
            gh, gw, frac, index=ORW._blake_u64("overlap-rect", "0", text)),
        "rect_jitter": _jitter_rect(gh, gw, rng, frac),
        "flat": np.ones((gh, gw), dtype=bool),
    }
    return out


def analyse(model_rec, qb, frac, seed=20260904):
    rng = np.random.default_rng(seed)
    groups = {s: [] for s in SCHEMES}
    ring_groups, area_groups = [], []
    ring_shr = {s: [] for s in SCHEMES}
    ring_cov = {s: [] for s in SCHEMES}
    verify, n_comp, n_steps, qb_missing = [], 0, 0, 0

    for s, comps in load_chains(model_rec):
        boxes = qb.get((s.get("dataset"), str(s.get("question_id"))))
        if boxes is None:
            qb_missing += 1
        per = {k: [] for k in SCHEMES}
        rings, areas = [], []
        for ci, comp in enumerate(comps):
            if not comp:
                for k in SCHEMES:
                    per[k].append(None)
                rings.append(None)
                areas.append(None)
                continue
            n_comp += 1
            n_steps += len(comp)
            rg = ring_mask(comp[0].gh, comp[0].gw)
            per["true"].append(float(np.mean([ORW._mean_in(st.smap, st.mask) for st in comp])))
            for k, m in masks_for(comp, boxes, rng, f"completion {ci}", frac).items():
                if m is None or m.shape != comp[0].mask.shape:
                    per[k].append(None)
                    continue
                per[k].append(float(np.mean([ORW._mean_in(st.smap, m) for st in comp])))
                ring_shr[k].append(float(np.logical_and(m, rg).sum()) / max(1, int(m.sum())))
                ring_cov[k].append(float(m[rg].mean()))
            ring_shr["true"].extend(
                float(np.logical_and(st.mask, rg).sum()) / max(1, int(st.mask.sum()))
                for st in comp)
            ring_cov["true"].extend(float(st.mask[rg].mean()) for st in comp)
            # The completion's own attention on the ring: the quantity the arms are
            # supposed to move, and the thing a mask that reaches the ring stops seeing.
            rings.append(float(np.mean([st.smap[rg].sum() / st.smap.sum()
                                        for st in comp if st.smap.sum() > 0])))
            areas.append(float(np.mean([st.mask.mean() for st in comp])))
            for st in comp:
                if st.stored is not None:
                    verify.append(abs(st.stored - ORW._mean_in(st.smap, st.mask)))
        for k in SCHEMES:
            groups[k].append(per[k])
        ring_groups.append(rings)
        area_groups.append(areas)

    base = pooled_sd(groups["true"])
    c_flat, c_ring, c_true = centred(groups["flat"]), centred(ring_groups), centred(groups["true"])
    c_area = centred(area_groups)
    rows = {}
    for k in SCHEMES:
        c = centred(groups[k])
        flat_vals = [v for g in groups[k] for v in g if v is not None]
        sd = pooled_sd(groups[k])
        rows[k] = {
            "sd": sd, "ratio": sd / base if base else float("nan"),
            "level": float(np.mean(flat_vals)) if flat_vals else float("nan"),
            "w": 0.4 * base / sd if sd else float("nan"),
            "r_flat": pearson(c, c_flat), "r_ring": pearson(c, c_ring),
            "r_true": pearson(c, c_true),
            "ringshr": float(np.mean(ring_shr[k])) if ring_shr[k] else float("nan"),
            "ringcov": float(np.mean(ring_cov[k])) if ring_cov[k] else float("nan"),
            "n": len(flat_vals),
        }
    return {
        "rows": rows, "n_comp": n_comp, "n_steps": n_steps, "qb_missing": qb_missing,
        "verify": float(np.max(verify)) if verify else float("nan"),
        "r_true_area": pearson(c_true, c_area),
    }


def step_similarity(model_rec):
    """Are two chains' FIRST steps more alike than two chains' LAST steps?

    The question that decides --overlap_chain_boxes' selector: whichever step is most
    stereotyped across a prompt's rollouts is the one that hands them the most nearly
    identical masks, which is what a per-completion mask exists not to do.
    """
    import itertools
    rng = np.random.default_rng(20260904)
    w, ff, ll, aa = [], [], [], []
    for _s, comps in load_chains(model_rec):
        chains = [c for c in comps if c]
        if len(chains) < 2:
            continue
        for ch in chains:
            for a, b in itertools.combinations(ch, 2):
                if a.mask.shape == b.mask.shape:
                    w.append(closeness(a.mask, b.mask))
        for A, B in itertools.combinations(chains, 2):
            if A[0].mask.shape != B[0].mask.shape:
                continue
            ff.append(closeness(A[0].mask, B[0].mask))
            ll.append(closeness(A[-1].mask, B[-1].mask))
            aa.append(closeness(A[int(rng.integers(len(A)))].mask,
                                B[int(rng.integers(len(B)))].mask))
    f = lambda v: float(np.nanmean(v)) if v else float("nan")   # noqa: E731
    return {"within": f(w), "first": f(ff), "last": f(ll), "any": f(aa),
            "n_pairs": len(ff)}


def ring_geometry(frac=F_FIXED, gh=10, gw=16):
    """Exact ring coverage of each rectangle family on one grid -- no sampling needed."""
    rg = ring_mask(gh, gw)
    rows, cols = ORW._rect_dims(gh, gw, frac)
    nr, nc = gh - rows + 1, gw - cols + 1
    pr = np.array([min(r, gh - rows) - max(0, r - rows + 1) + 1 for r in range(gh)]) / nr
    pc = np.array([min(c, gw - cols) - max(0, c - cols + 1) + 1 for c in range(gw)]) / nc
    prof = np.outer(pr, pc)
    got = ORW.interior_placements(gh, gw, frac)
    return {
        "grid": (gh, gw), "ring_patches": int(rg.sum()),
        "centre_dims": (rows, cols), "centre_area": rows * cols / (gh * gw),
        "centre_ring": float(ORW._centre_rect_mask(gh, gw, frac)[rg].mean()),
        "jitter_ring": float(prof[rg].mean()), "jitter_interior": float(prof[~rg].mean()),
        "jitter_positions": nr * nc,
        "interior_dims": got[:2] if got else None,
        "interior_area": (got[0] * got[1] / (gh * gw)) if got else float("nan"),
        "interior_positions": (got[2] * got[3]) if got else 0,
        "interior_ring": float(ORW._interior_rect_mask(gh, gw, frac)[rg].mean()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("probe", help="an overlap_probe --store-maps probe_merged.json")
    ap.add_argument("--question-boxes", default=None,
                    help="a precompute_question_boxes.py file for the SAME corpus; "
                         "without it the qbox row is empty and every other row is unaffected")
    ap.add_argument("--models", default=None, help="comma-separated subset")
    ap.add_argument("--rect-frac", type=float, default=F_FIXED)
    args = ap.parse_args()

    models = json.load(open(args.probe))["models"]
    want = args.models.split(",") if args.models else list(models)
    qb = qbox_lookup(args.question_boxes)

    g = ring_geometry(args.rect_frac)
    print("RING GEOMETRY -- exact, on the modal 10x16 patch grid")
    print(f"  the border is {g['ring_patches']}/{g['grid'][0] * g['grid'][1]} patches "
          f"({g['ring_patches'] / (g['grid'][0] * g['grid'][1]):.3f} of them) and holds the "
          "peak mean_in divides by")
    print("  'ring covered' below is ringcov: the share of those border patches the mask takes")
    print(f"  centre rectangle   {g['centre_dims'][0]}x{g['centre_dims'][1]} "
          f"({g['centre_area']:.3f} of the grid)   ring covered {g['centre_ring']:.3f}")
    print(f"  interior rectangle {g['interior_dims'][0]}x{g['interior_dims'][1]} "
          f"({g['interior_area']:.3f} of the grid)   ring covered {g['interior_ring']:.3f}"
          f"   {g['interior_positions']} placements")
    print(f"  any in-frame offset (rejected)                       ring covered "
          f"{g['jitter_ring']:.3f} vs {g['jitter_interior']:.3f} interior, "
          f"{g['jitter_positions']} placements")

    for name in want:
        if name not in models:
            print(f"\n-- {name}: not in this probe")
            continue
        r = analyse(models[name], qb, args.rect_frac)
        sim = step_similarity(models[name])
        miss = (f", {r['qb_missing']} rows missing from the qbox cache"
                if r["qb_missing"] else "")
        print(f"\n=== {name}   {r['n_comp']} completions, {r['n_steps']} scored steps{miss}"
              f"   |recomputed - stored| max {r['verify']:.5f}")
        print(f"    within-group r(reward, own union area) = {r['r_true_area']:+.3f}"
              "   <- how much of the incumbent's spread is the completion's own mask SIZE")
        print(f"    mask closeness between chains of one image: first steps {sim['first']:.3f}, "
              f"last {sim['last']:.3f}, any {sim['any']:.3f}, within one chain "
              f"{sim['within']:.3f}  ({sim['n_pairs']} pairs)")
        print(f"    {'scheme':<12} {'sd_within':>10} {'vs true':>8} {'w to match':>11} "
              f"{'level':>8} {'r(flat)':>8} {'r(ring)':>8} {'r(true)':>8} {'ring':>6} "
              f"{'n':>5}  source")
        for k in SCHEMES:
            d = r["rows"][k]
            print(f"    {k:<12} {d['sd']:>10.5f} {d['ratio']:>8.2f} {d['w']:>11.2f} "
                  f"{d['level']:>8.4f} {d['r_flat']:>8.3f} {d['r_ring']:>8.3f} "
                  f"{d['r_true']:>8.3f} {d['ringshr']:>8.3f} {d['ringcov']:>8.3f} "
                  f"{d['n']:>5}  {DESCRIBE[k]}")


if __name__ == "__main__":
    main()
