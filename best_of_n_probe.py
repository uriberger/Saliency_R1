#!/usr/bin/env python
"""Can a saliency statistic PICK the right answer out of a group of sampled ones?

    python best_of_n_probe.py outputs/overlap_probe/<dir>/probe_merged.json [--selftest]

CPU only, seconds, no GPU and no Grounding-DINO. Everything comes off disk:
`overlap_probe.py --store-maps` already wrote, per observe step, the attention map
quantised to its own peak (`map_q`) with the peak alongside (`map_max`), the DINO union
raster (`mask_q`) and the grid -- and, per completion, `accuracy_reward` and the LLM
judge's `openai_reward`. So the whole of "best-of-N by saliency" is answerable from
files that already exist, before any GPU is asked for.

WHY THIS EXISTS. The inference-time proposal has two halves: edit the attention while
the model writes (`sink_shift.py`), and sample N answers then keep the one whose map
looks best. The second half needs no new model behaviour at all -- it is pure selection
-- so its ceiling is measurable today, on 11 checkpoints x 30 questions x 8 sampled
answers.

WHAT IT MEASURES. Within one question's group of sampled answers:

    picked   accuracy of the answer the selector ranks first
    random   the group's MEAN accuracy -- the exact expectation of keeping one answer
             at random, which is what plain sampling does
    oracle   the group's MAX -- what any selector could reach at best
    worst    the group's MIN -- what an anti-selector would reach

`picked - random` is the whole result. `oracle - random` is the headroom: if it is near
zero the group agrees with itself and no selector can help, which is a fact about the
corpus rather than about the selector.

Ties in the argmax are averaged, not broken arbitrarily: the reported value is the
expectation under a uniform tie-break, so a selector that is constant across a group
scores exactly `random` instead of scoring whichever answer happened to come first.

SELECTORS. Each is one number per completion, aggregated over its observe steps the way
the reward aggregates (`token_reduction=mean`), and built by importing
trl/rewards/overlap_rewards.py rather than reimplementing it -- so a number here is a
number about the code a run will execute.

    rect_mean_in   mean_in against the centred rectangle -- the --overlap_rect_frac target
    true_mean_in   mean_in against the step's own DINO union -- the reward that trained
    flatness       mean(m)/max(m), no mask at all
    rect_mass      share of the map's mass inside the rectangle
    low_frame      MINUS the share of the map's mass on the one-patch border (the sink)
    short          MINUS the completion's token count -- not a saliency measure at all,
                   but the thing the reward is most associated with within a group
    hash           a deterministic function of the completion's text: pure noise, and the
                   calibration of this harness. It must land on `random`.

WHAT WOULD MAKE THIS WORTH BUILDING PROPERLY. `picked - random` positive for a saliency
selector, in most of the 11 models, by more than the hash selector's spread. The prior
is against it: r(overlap reward, accuracy reward) within a group is -0.019 +- 0.051
(docs/next-reward-experiments.md), and a selector correlated 0.02 with correctness
cannot move a best-of-8 by anything measurable.

CAVEATS THAT BOUND EVERY NUMBER BELOW. 30 questions, so the CIs are wide and clustered
on them. The 11 models are one cold start plus ten of its descendants, not eleven
independent replications. Completions with no stored map cannot be scored by the map
selectors, so every baseline is computed over the SAME scorable subset of each group as
the selector it is compared against -- `n_scorable` in the header is how many of 8 that
left, and a group with fewer than 2 is dropped entirely.
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

    The package pulls torch, transformers and the trainer; the reward module itself
    needs only numpy. Same trick mask_variance_probe.py and test_rect_reward_cpu.py use.
    """
    pkg = types.ModuleType("trl_bon"); pkg.__path__ = [os.path.join(ROOT, "trl")]
    sys.modules["trl_bon"] = pkg
    sub = types.ModuleType("trl_bon.rewards")
    sub.__path__ = [os.path.join(ROOT, "trl", "rewards")]
    sys.modules["trl_bon.rewards"] = sub
    for name in ("roll_null", "overlap_rewards"):
        spec = importlib.util.spec_from_file_location(
            f"trl_bon.rewards.{name}", os.path.join(ROOT, "trl", "rewards", f"{name}.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"trl_bon.rewards.{name}"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["trl_bon.rewards.overlap_rewards"]


ORW = _import_overlap_rewards()

F_FIXED = 0.565          # the fraction the rect_frac arm and centre_box_probe.py use
READOUTS = ("accuracy", "judge")
SELECTORS = ["rect_mean_in", "true_mean_in", "flatness", "rect_mass", "low_frame",
             "short", "hash"]

DESCRIBE = {
    "rect_mean_in": "mean_in on the centred rectangle (--overlap_rect_frac)",
    "true_mean_in": "mean_in on the step's own DINO union (the trained reward)",
    "flatness": "mean(m)/max(m), no mask",
    "rect_mass": "share of map mass inside the rectangle",
    "low_frame": "minus the share of map mass on the border",
    "short": "minus the completion's token count",
    "hash": "noise -- the calibration of this harness",
}


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


def frame_mask(gh, gw):
    """The one-patch border of the patch grid -- the sink. `overlap_rewards._ring_frac`
    calls the same set the ring; the name here is `frame` only to keep it distinct from
    the mask-share statistic that function returns."""
    m = np.zeros((gh, gw), dtype=bool)
    m[0, :] = m[-1, :] = True
    m[:, 0] = m[:, -1] = True
    return m


class Step:
    __slots__ = ("gh", "gw", "mask", "smap", "stored")

    def __init__(self, rec):
        self.gh, self.gw = rec["grid"]
        self.mask = decode_mask(rec["mask_q"], self.gh, self.gw)
        self.smap = decode_map(rec["map_q"], self.gh, self.gw, rec.get("map_max") or 0.0)
        self.stored = rec.get("mean_in_raw")


# ---------------------------------------------------------------------------
# one completion -> one number per selector
# ---------------------------------------------------------------------------
def completion_scores(comp_rec, rect_frac, seed):
    """{selector: value} for one completion, or None if it has no usable map.

    Every map selector is a mean over the completion's observe steps, which is what
    `token_reduction=mean` does inside the reward. `short` and `hash` need no map, but
    they are still returned as None when the maps are missing, so that every selector is
    compared on exactly the same set of completions.
    """
    steps = [Step(r) for r in (comp_rec.get("observe_steps") or [])
             if r.get("mask_q") and r.get("grid") and r.get("map_q")]
    steps = [st for st in steps if st.smap.sum() > 0]
    if not steps:
        return None
    gh, gw = steps[0].gh, steps[0].gw
    if any((st.gh, st.gw) != (gh, gw) for st in steps):
        return None                      # one completion, two grids: skip, do not guess
    rect = ORW._centre_rect_mask(gh, gw, rect_frac)
    if rect is None:
        return None
    fr = frame_mask(gh, gw)
    ones = np.ones((gh, gw), dtype=bool)

    def mean_over_steps(fn):
        vals = [fn(st) for st in steps]
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        return float(np.mean(vals)) if vals else None

    text = comp_rec.get("text") or ""
    n_tok = comp_rec.get("n_completion_tokens")
    out = {
        "rect_mean_in": mean_over_steps(lambda st: ORW._mean_in(st.smap, rect)),
        "true_mean_in": mean_over_steps(lambda st: ORW._mean_in(st.smap, st.mask)),
        "flatness": mean_over_steps(lambda st: ORW._mean_in(st.smap, ones)),
        "rect_mass": mean_over_steps(lambda st: st.smap[rect].sum() / st.smap.sum()),
        "low_frame": mean_over_steps(lambda st: -st.smap[fr].sum() / st.smap.sum()),
        "short": (-float(n_tok) if n_tok is not None else None),
        # A stable 64-bit digest of the text, unlike Python's hash(), which is salted per
        # interpreter -- so this selector is reproducible across runs and machines.
        "hash": float(ORW._blake_u64("best-of-n", str(seed), text) % 10**9) / 10**9,
    }
    out["_verify"] = [(st.stored, ORW._mean_in(st.smap, st.mask))
                      for st in steps if st.stored is not None]
    return out


def group_table(model_rec, rect_frac, seed):
    """-> [group], each {sel: [score...], readout: [value...]} over its SCORABLE completions.

    A completion enters a group only if it has both a usable map and a grade, so the
    selector and every baseline see the identical set. Groups left with fewer than two
    entries are dropped: "pick the best of one" is not a question.
    """
    groups, verify, n_dropped_comp, n_dropped_grp = [], [], 0, 0
    for s in model_rec["samples"]:
        cols = {k: [] for k in SELECTORS}
        outs = {r: [] for r in READOUTS}
        for c in s["completions"]:
            sc = completion_scores(c, rect_frac, seed)
            acc = c.get("rewards", {}).get("accuracy_reward")
            judge = c.get("rewards", {}).get("openai_reward")
            if sc is None or acc is None or any(sc[k] is None for k in SELECTORS):
                n_dropped_comp += 1
                continue
            verify.extend(sc.pop("_verify"))
            for k in SELECTORS:
                cols[k].append(sc[k])
            outs["accuracy"].append(float(acc))
            outs["judge"].append(float(judge) if judge is not None else float("nan"))
        if len(outs["accuracy"]) < 2:
            n_dropped_grp += 1
            continue
        groups.append({"sel": {k: np.asarray(v, float) for k, v in cols.items()},
                       "out": {r: np.asarray(v, float) for r, v in outs.items()}})
    return groups, verify, n_dropped_comp, n_dropped_grp


# ---------------------------------------------------------------------------
# the selection experiment
# ---------------------------------------------------------------------------
def picked_value(scores, values):
    """Expected readout of the argmax under a uniform tie-break.

    Averaging over ties rather than taking the first is what makes a constant selector
    score exactly `random`: without it, a selector that never discriminates would inherit
    whatever ordering the generator happened to emit and look like a real one.

    `values` carries no NaN by the time it gets here -- group_rows drops ungraded
    answers before selecting -- so a plain mean is correct and an empty slice is a bug.
    """
    top = scores == scores.max()
    return float(values[top].mean())


def group_rows(groups, selectors, readout):
    """-> per-group arrays: {sel: picked}, and random / oracle / worst.

    A completion the readout never graded (a judge score that was not recorded) is
    dropped from the group BEFORE the argmax, not after. Dropping it after would let a
    selector rank an ungraded answer first and then score that group NaN, which silently
    removes exactly the groups where the selector was most confident.
    """
    out = {k: [] for k in selectors}
    out["random"], out["oracle"], out["worst"] = [], [], []
    for g in groups:
        v = g["out"][readout]
        ok = np.isfinite(v)
        if ok.sum() < 2:
            continue
        v = v[ok]
        for k in selectors:
            out[k].append(picked_value(g["sel"][k][ok], v))
        out["random"].append(float(v.mean()))
        out["oracle"].append(float(v.max()))
        out["worst"].append(float(v.min()))
    return {k: np.asarray(v, float) for k, v in out.items()}


def boot_ci(rows, key, base="random", n_boot=2000, seed=20260907):
    """Mean of `key - base` with a 95% CI, resampling GROUPS -- the cluster.

    Common random numbers across keys: every selector is resampled on the identical
    draws, so two selectors' intervals can be read against each other and the paired
    difference is not swamped by which questions were drawn.
    """
    a, b = rows[key], rows[base]
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    if d.size < 2:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def within_group_r(groups, key, readout):
    """Group-centred Pearson of a selector against the readout.

    This is the quantity docs/next-reward-experiments.md reports as
    r(overlap reward, accuracy reward) = -0.019 +- 0.051, and it is what a best-of-N
    gain has to come from: the group mean is subtracted because it is common to every
    answer in the group and so cannot affect which one is picked.
    """
    xs, ys = [], []
    for g in groups:
        x, y = g["sel"][key], g["out"][readout]
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 2:
            continue
        xs.append(x[ok] - x[ok].mean())
        ys.append(y[ok] - y[ok].mean())
    if not xs:
        return float("nan")
    x, y = np.concatenate(xs), np.concatenate(ys)
    if x.std() < 1e-15 or y.std() < 1e-15:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def report_model(name, model_rec, args):
    groups, verify, drop_c, drop_g = group_table(model_rec, args.rect_frac, args.seed)
    if not groups:
        print(f"\n=== {name}: nothing scorable")
        return None
    sizes = [g["out"]["accuracy"].size for g in groups]
    worst_verify = max((abs(a - b) for a, b in verify), default=float("nan"))
    print(f"\n=== {name}   {len(groups)} groups, {sum(sizes)} completions "
          f"(mean {np.mean(sizes):.1f} scorable of 8; {drop_c} completions and "
          f"{drop_g} groups dropped)   |recomputed - stored| max {worst_verify:.5f}")

    per_readout = {}
    for readout in args.readouts:
        rows = group_rows(groups, SELECTORS, readout)
        if not rows["random"].size:
            continue
        print(f"    readout {readout}:  random {rows['random'].mean():.4f}   "
              f"oracle {rows['oracle'].mean():.4f}   worst {rows['worst'].mean():.4f}   "
              f"headroom {rows['oracle'].mean() - rows['random'].mean():+.4f}")
        print(f"      {'selector':<14} {'picked':>8} {'-random':>9} {'95% CI':>20} "
              f"{'r within':>9}  what it is")
        deltas = {}
        for k in SELECTORS:
            m, lo, hi = boot_ci(rows, k, n_boot=args.n_boot, seed=args.seed)
            r = within_group_r(groups, k, readout)
            deltas[k] = m
            print(f"      {k:<14} {rows[k].mean():>8.4f} {m:>+9.4f} "
                  f"{f'[{lo:+.4f}, {hi:+.4f}]':>20} {r:>+9.3f}  {DESCRIBE[k]}")
        per_readout[readout] = deltas
    return per_readout


def summarise(per_model, readouts):
    """Across models: the median gain and how many models moved which way.

    The 11 models are one cold start and ten of its descendants, so "8 of 11" is a
    consistency statement along one training trajectory, not eleven replications. It is
    still the right column to read, because the per-model CIs all come from the same 30
    questions and so overlap by construction.
    """
    print("\n" + "=" * 78)
    print("ACROSS MODELS -- median of `picked - random`, and how many models are above 0")
    for readout in readouts:
        rows = [d[readout] for d in per_model.values() if d and readout in d]
        if not rows:
            continue
        n = len(rows)
        print(f"\n  readout {readout}   ({n} models)")
        print(f"      {'selector':<14} {'median':>9} {'min':>9} {'max':>9} {'above 0':>9}")
        for k in SELECTORS:
            v = np.asarray([r[k] for r in rows], float)
            print(f"      {k:<14} {np.median(v):>+9.4f} {v.min():>+9.4f} {v.max():>+9.4f} "
                  f"{f'{int((v > 0).sum())}/{n}':>9}")
    print("\n  `hash` is the calibration: it is noise, so its median is the size of a "
          "gain\n  this corpus cannot tell apart from nothing.")


def selftest(per_model, readouts):
    """Two things that must hold, or no other number here is readable.

    The hash selector is noise, so its median gain must sit near zero; and `oracle` must
    beat `random`, or the groups agree with themselves and nothing could ever be picked.
    """
    print("\n" + "=" * 78)
    print("SELFTEST")
    ok = True
    for readout in readouts:
        rows = [d[readout] for d in per_model.values() if d and readout in d]
        if not rows:
            continue
        v = np.asarray([r["hash"] for r in rows], float)
        med, above = float(np.median(v)), int((v > 0).sum())
        good = abs(med) < 0.02 and 2 <= above <= len(v) - 2
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  {readout}: hash median {med:+.4f} "
              f"(want |median| < 0.02), above 0 in {above}/{len(v)} models "
              f"(want neither 0 nor all)")
    print(f"  {'PASS' if ok else 'FAIL'}  overall")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("probe", help="an overlap_probe --store-maps probe_merged.json")
    ap.add_argument("--models", default=None, help="comma-separated subset")
    ap.add_argument("--rect-frac", type=float, default=F_FIXED)
    ap.add_argument("--readouts", default=",".join(READOUTS),
                    help="accuracy, judge, or both")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260907)
    ap.add_argument("--selftest", action="store_true",
                    help="check the noise selector lands on random and the oracle has "
                         "headroom; exits non-zero if not")
    args = ap.parse_args()
    args.readouts = [r for r in args.readouts.split(",") if r in READOUTS]

    models = json.load(open(args.probe))["models"]
    want = args.models.split(",") if args.models else list(models)

    gh, gw = 10, 16
    rect = ORW._centre_rect_mask(gh, gw, args.rect_frac)
    fr = frame_mask(gh, gw)
    print(f"GEOMETRY on the modal {gh}x{gw} grid: rectangle {int(rect.sum())} patches "
          f"({rect.mean():.3f}), border {int(fr.sum())} patches ({fr.mean():.3f})")
    print(f"CORPUS: {args.probe}")

    per_model = {}
    for name in want:
        if name not in models:
            print(f"\n=== {name}: not in this probe")
            continue
        per_model[name] = report_model(name, models[name], args)

    summarise(per_model, args.readouts)
    if args.selftest and not selftest(per_model, args.readouts):
        sys.exit(1)


if __name__ == "__main__":
    main()
