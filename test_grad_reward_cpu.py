#!/usr/bin/env python
"""CPU checks for trl/rewards/grad_rewards.py -- the roll-null gradient reward.

No GPU, no model, no Grounding-DINO (the grounding call is stubbed). What this gates is
the arithmetic the whole design rests on:

  * SIZE INVARIANCE. On a flat, uninformative map the score is exactly 0 for a union of
    any area and any shape. This is the property the roll-null exists for: ||g_U|| alone
    grows like sqrt(|U|), and ||g_U|| - ||g_out|| is monotone in |U| even for a flat map,
    so both would pay the policy to name things that ground to bigger boxes.
  * SCALE INVARIANCE. Multiplying the map by any constant leaves the score unchanged --
    which is also why mean-over-tokens and sum-over-tokens in grad_maps give the same
    reward, they differ by exactly 1/|S|.
  * a planted signal is recovered with the right SIGN, in both directions;
  * in-frame offsets never wrap across the border, never include the identity, and hand
    over to toroidal ones (loudly) when a near-full-frame union leaves too few;
  * degenerate steps return None -- absent, not zero -- so they leave the mean over steps
    rather than dragging it down;
  * the completion level: the format gate multiplies, no scorable step masks the row,
    --grad_natural_only masks the row, and duplicate steps are deduped before the mean.

    python test_grad_reward_cpu.py
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent


def _load(dotted, rel):
    spec = importlib.util.spec_from_file_location(dotted, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = mod
    spec.loader.exec_module(mod)
    return mod


# Stub the parent packages so the module's relative import resolves without executing
# trl/__init__.py (which drags in the whole TRL dependency tree for no reason here).
for _name, _path in (("trl", REPO / "trl"), ("trl.rewards", REPO / "trl" / "rewards")):
    _m = types.ModuleType(_name)
    _m.__path__ = [str(_path)]
    sys.modules[_name] = _m

OR = _load("trl.rewards.overlap_rewards", "trl/rewards/overlap_rewards.py")
GR = _load("trl.rewards.grad_rewards", "trl/rewards/grad_rewards.py")

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'ok  ' if cond else 'FAIL'} {name}" + (f"   {detail}" if detail else ""))


def box(gh, gw, r0, r1, c0, c1):
    m = np.zeros((gh, gw), dtype=bool)
    m[r0:r1, c0:c1] = True
    return m


def rng(seed=0):
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
def test_flat_map_is_size_invariant():
    print("\n[null] a flat map scores 0 at every box size -- the point of the roll-null")
    GR.configure(null_offsets=16, logratio_clip=10.0, inframe_rolls=True)
    G = np.full((16, 16), 0.37, dtype=np.float32)
    worst = 0.0
    for side in (1, 2, 4, 6, 8, 10, 12):
        m = box(16, 16, 2, 2 + side, 3, 3 + side)
        r = GR.step_logratio(G, m, rng(side))
        worst = max(worst, abs(r))
    check("|score| == 0 for square unions from 1x1 to 12x12", worst < 1e-12,
          f"worst |r| {worst:.2e}")

    # a ragged, multi-box union -- the real thing is a union of DINO boxes, not a square
    m = box(16, 16, 1, 4, 1, 5) | box(16, 16, 9, 13, 8, 15) | box(16, 16, 6, 7, 2, 3)
    check("and for a ragged multi-box union", abs(GR.step_logratio(G, m, rng(7))) < 1e-12)

    # the two forms this replaces, on the same flat map, for contrast
    g2 = G.astype(np.float64) ** 2
    sizes = [(box(16, 16, 2, 2 + s, 3, 3 + s)) for s in (2, 6, 10)]
    norms = [np.sqrt(g2[m].sum()) for m in sizes]
    diffs = [np.sqrt(g2[m].sum()) - np.sqrt(g2[~m].sum()) for m in sizes]
    check("||g_U|| alone would have risen 5x over the same range",
          norms[-1] / norms[0] > 4.5, f"{norms[0]:.2f} -> {norms[-1]:.2f}")
    check("||g_U|| - ||g_out|| would have risen monotonically too",
          diffs[0] < diffs[1] < diffs[2],
          f"{diffs[0]:.2f} -> {diffs[1]:.2f} -> {diffs[2]:.2f}")


def test_scale_invariance():
    print("\n[null] the score is invariant to any rescale of the map")
    GR.configure(logratio_clip=10.0)
    g = rng(1).random((12, 12)).astype(np.float32) + 0.1
    m = box(12, 12, 2, 6, 3, 9)
    base = GR.step_logratio(g, m, rng(3))
    out = {c: GR.step_logratio((g * c).astype(np.float32), m, rng(3)) for c in (1e-3, 7.0, 1 / 23)}
    worst = max(abs(v - base) for v in out.values())
    # exact in exact arithmetic; the maps are stored float32, so the rescale itself
    # perturbs the input at ~1e-7 relative and that is the floor here, not the metric.
    check("c * G gives the same score for c over six orders of magnitude", worst < 1e-6,
          f"worst |diff| {worst:.2e}")
    check("so mean- and sum-over-tokens in grad_maps are the same reward",
          abs(GR.step_logratio((g / 17).astype(np.float32), m, rng(3)) - base) < 1e-6)
    exact = np.float64
    check("and it is exact when the map is not rounded to float32",
          abs(GR.step_logratio(exact(g) * 1e-3, m, rng(3))
              - GR.step_logratio(exact(g) * 7.0, m, rng(3))) < 1e-12)


def test_planted_signal_has_the_right_sign():
    print("\n[null] a planted signal, both directions")
    GR.configure(logratio_clip=10.0)
    m = box(16, 16, 4, 8, 4, 8)
    hot_in = np.full((16, 16), 0.1, dtype=np.float32)
    hot_in[m] = 2.0
    hot_out = np.full((16, 16), 2.0, dtype=np.float32)
    hot_out[m] = 0.1
    r_in = GR.step_logratio(hot_in, m, rng(5))
    r_out = GR.step_logratio(hot_out, m, rng(5))
    check("mass inside the union scores positive", r_in > 1.0, f"r = {r_in:.3f}")
    check("mass outside it scores negative", r_out < -1.0, f"r = {r_out:.3f}")


def test_offsets():
    print("\n[null] the control placements")
    gh, gw = 16, 16
    m = box(gh, gw, 3, 7, 5, 11)
    offs = GR.inframe_offsets(m)
    check("the identity is excluded", (0, 0) not in offs)
    check("the count matches (gh - h + 1)(gw - w + 1) - 1",
          len(offs) == (gh - 4 + 1) * (gw - 6 + 1) - 1, f"{len(offs)}")
    ok_area = all(np.roll(m, o, axis=(0, 1)).sum() == m.sum() for o in offs)
    ok_inside = True
    for dy, dx in offs:
        rolled = np.roll(m, (dy, dx), axis=(0, 1))
        ys, xs = np.nonzero(rolled)
        # a wrap would split the block; an in-frame translate keeps it contiguous
        ok_inside &= (ys.max() - ys.min() == 3) and (xs.max() - xs.min() == 5)
    check("every in-frame offset preserves the area", ok_area)
    check("and none of them wraps across the border", ok_inside)

    # a ragged union: the constraint is on the bounding box, not on each cell
    ragged = box(gh, gw, 2, 4, 2, 4) | box(gh, gw, 5, 6, 9, 10)
    ok = True
    for dy, dx in GR.inframe_offsets(ragged):
        ys, xs = np.nonzero(ragged)
        ok &= (0 <= ys.min() + dy) and (ys.max() + dy < gh)
        ok &= (0 <= xs.min() + dx) and (xs.max() + dx < gw)
    check("a ragged union's offsets keep every cell inside", ok)


def test_toroidal_fallback():
    print("\n[null] the near-full-frame fallback")
    GR.configure(null_offsets=8, min_inframe=4, inframe_rolls=True, logratio_clip=10.0)
    G = np.full((8, 8), 0.5, dtype=np.float32)
    full = box(8, 8, 0, 8, 0, 7)          # only 2 in-frame offsets exist
    GR.pop_diagnostics()
    r = GR.step_logratio(G, full, rng(2))
    d = GR.pop_diagnostics()
    check("a near-full-frame union falls back to toroidal", d["toroidal_frac"] == 1.0)
    check("it is still scored, not dropped", r is not None and abs(r) < 1e-12)

    small = box(8, 8, 1, 3, 1, 3)
    GR.pop_diagnostics()
    GR.step_logratio(G, small, rng(2))
    check("a small union does not", GR.pop_diagnostics()["toroidal_frac"] == 0.0)


def test_degenerate_steps_are_absent_not_zero():
    print("\n[null] degenerate steps")
    GR.configure(logratio_clip=1.0)
    m = box(10, 10, 2, 5, 2, 5)
    check("an all-zero map returns None",
          GR.step_logratio(np.zeros((10, 10), np.float32), m, rng(0)) is None)
    # a single hot cell, and a 1x1 union on it: every control lands on a zero, so the
    # null has no support and the ratio is undefined. Absent, not +inf and not a huge
    # score -- this is the case that would otherwise blow up a group's advantage.
    spike = np.zeros((10, 10), dtype=np.float32)
    spike[0, 0] = 5.0
    check("a null with no support returns None",
          GR.step_logratio(spike, box(10, 10, 0, 1, 0, 1), rng(0)) is None)
    check("a NaN map returns None",
          GR.step_logratio(np.full((10, 10), np.nan, np.float32), m, rng(0)) is None)


def test_clip():
    print("\n[null] the clip")
    GR.configure(logratio_clip=0.5)
    m = box(16, 16, 4, 8, 4, 8)
    g = np.full((16, 16), 1e-6, dtype=np.float32)
    g[m] = 10.0
    GR.pop_diagnostics()
    r = GR.step_logratio(g, m, rng(0))
    d = GR.pop_diagnostics()
    check("the score is bounded by the clip", abs(r) <= 0.5 + 1e-12, f"r = {r:.3f}")
    check("and the clip event is logged", d["clip_frac"] == 1.0)
    check("the raw value is logged unclipped", d["logratio_raw"] > 0.5,
          f"raw {d['logratio_raw']:.2f}")


# ---------------------------------------------------------------------------
def stub_dino(mapping):
    """Replace grounding with a fixed text -> boxes table (relative coords).

    Keyed on the NORMALISED text, because the real detector does not care about case or a
    trailing period either -- a stub that keys on the raw string would silently drop the
    duplicate variants and make the dedupe test pass for the wrong reason.
    """
    table = {GR._norm_text(k): v for k, v in mapping.items()}

    def _fake(images, texts):
        return [table.get(GR._norm_text(t), []) for t in texts]
    GR._dino_boxes = _fake


def steps(*pairs):
    return [{"map": m, "text": t} for t, m in pairs]


def test_completion_level():
    print("\n[reward] the completion level")
    GR.configure(logratio_clip=10.0, dedupe_steps=True, natural_only=False)
    OR.configure(box_threshold=0.1, max_box_area=0.5, max_union_area=None)
    m = box(16, 16, 4, 8, 4, 8)
    hot = np.full((16, 16), 0.1, dtype=np.float32)
    hot[m] = 3.0
    flat = np.full((16, 16), 0.4, dtype=np.float32)
    # 4/16 .. 8/16 in relative coords is the same block the map is hot on
    stub_dino({"a cat": [[0.25, 0.25, 0.5, 0.5]], "nothing": []})

    out = GR.think_grad_reward(
        saliency_map=[steps(("a cat", hot)), steps(("a cat", flat)), steps(("nothing", hot))],
        valid_list=[True, True, True],
        image=[None, None, None],
    )
    check("a grounded, well-aimed step scores positive", out[0] > 1.0, f"{out[0]:.3f}")
    check("a flat map on the same box scores 0", abs(out[1]) < 1e-12, f"{out[1]:.3g}")
    check("a completion with no groundable step is masked, not zeroed", out[2] is None)

    gated = GR.think_grad_reward(
        saliency_map=[steps(("a cat", hot))], valid_list=[False], image=[None],
    )
    check("the format gate multiplies", gated[0] == 0.0)

    GR.configure(natural_only=True)
    masked = GR.think_grad_reward(
        saliency_map=[steps(("a cat", hot)), steps(("a cat", hot))],
        valid_list=[True, True], image=[None, None], natural=[True, False],
    )
    check("--grad_natural_only masks the non-natural row",
          masked[0] > 1.0 and masked[1] is None)
    GR.configure(natural_only=False)


def test_dedupe():
    print("\n[reward] duplicate steps")
    m = box(16, 16, 4, 8, 4, 8)
    hot = np.full((16, 16), 0.1, dtype=np.float32)
    hot[m] = 3.0
    cold = np.full((16, 16), 0.1, dtype=np.float32)
    cold[box(16, 16, 10, 14, 10, 14)] = 3.0
    stub_dino({"a cat": [[0.25, 0.25, 0.5, 0.5]], "a dog": [[0.25, 0.25, 0.5, 0.5]]})

    # one genuine, hard step (aimed elsewhere) diluted by three copies of an easy one
    hacked = steps(("a cat", hot), ("A cat.", hot), ("a  cat", hot), ("a dog", cold))
    GR.configure(dedupe_steps=True, logratio_clip=10.0)
    GR.pop_diagnostics()
    on = GR.think_grad_reward(saliency_map=[hacked], valid_list=[True], image=[None])
    d_on = GR.pop_diagnostics()
    GR.configure(dedupe_steps=False)
    GR.pop_diagnostics()
    off = GR.think_grad_reward(saliency_map=[hacked], valid_list=[True], image=[None])
    d_off = GR.pop_diagnostics()
    GR.configure(dedupe_steps=True)

    check("duplicates are dropped before the mean", on[0] < off[0],
          f"deduped {on[0]:.3f} vs raw {off[0]:.3f}")
    check("the duplicate fraction is logged either way",
          abs(d_on["dup_frac"] - 0.5) < 1e-9 and abs(d_off["dup_frac"] - 0.5) < 1e-9,
          f"{d_on['dup_frac']:.2f}")
    check("normalisation catches case and whitespace variants",
          GR._norm_text("A  Cat. ") == GR._norm_text("a cat."))


def test_metric_dispatch():
    """The gradient map is scored by all four metrics now, not only the roll-null.

    The point of the check is the DEFAULT: `--overlap_metric` is one flag across three
    maps that had three different historical defaults, so an unset metric must still give
    the gradient map its roll-null. If that regresses, every existing --grad command line
    silently changes meaning, which is the kind of break that shows up as a training
    curve nobody can explain.
    """
    print("\n[metric] all four metrics, and the historical default")
    m = box(16, 16, 4, 10, 4, 10)
    hot = np.full((16, 16), 0.1, dtype=np.float32)
    hot[m] = 3.0
    stub_dino({"a cat": [[0.25, 0.25, 0.625, 0.625]]})
    GR.configure(logratio_clip=10.0, dedupe_steps=True, natural_only=False)

    got = {}
    for met in ("mean_in", "mean_in_v2", "auroc", "logratio"):
        OR.configure(metric=met, box_threshold=0.1, max_box_area=0.5, max_union_area=None,
                     mass_floor_tau=None, null_offsets=16, logratio_clip=10.0)
        GR.pop_diagnostics()
        out = GR.think_grad_reward(saliency_map=[steps(("a cat", hot))],
                                   valid_list=[True], image=[None])
        got[met] = out[0]
        check(f"the gradient map scores under {met}", out[0] is not None,
              f"{out[0]:+.3f}" if out[0] is not None else "None")
    check("and the four metrics are not all the same number",
          len({round(v, 6) for v in got.values()}) >= 3, str({k: round(v, 3) for k, v in got.items()}))

    # the union monitors must survive the switch away from the roll-null
    OR.configure(metric="mean_in_v2")
    GR.pop_diagnostics()
    GR.think_grad_reward(saliency_map=[steps(("a cat", hot))], valid_list=[True], image=[None])
    d = GR.pop_diagnostics()
    check("union_frac and ecc are still logged on a non-roll-null metric",
          np.isfinite(d["union_frac"]) and np.isfinite(d["ecc"]),
          f"union {d['union_frac']:.3f}, ecc {d['ecc']:.3f}")
    OR.configure(metric="logratio")


def test_diagnostics():
    print("\n[reward] the diagnostics the hacks show up in")
    GR.configure(logratio_clip=10.0)
    m = box(16, 16, 6, 10, 6, 10)
    g = np.full((16, 16), 0.5, dtype=np.float32)
    GR.pop_diagnostics()
    GR.step_logratio(g, m, rng(0))
    centre = GR.pop_diagnostics()
    GR.step_logratio(g, box(16, 16, 0, 4, 0, 4), rng(0))
    corner = GR.pop_diagnostics()
    check("ecc is near 0 for a centred union and near 1 for a corner one",
          centre["ecc"] < 0.05 and corner["ecc"] > 0.7,
          f"centre {centre['ecc']:.3f}, corner {corner['ecc']:.3f}")
    check("union_frac is the area fraction", abs(centre["union_frac"] - 16 / 256) < 1e-9)
    check("n_image is the whole map's norm",
          abs(centre["n_image"] - np.linalg.norm(g.astype(np.float64))) < 1e-6)
    # A fixed key set, NaN where nothing was seen: the trainer gathers these across ranks
    # and a rank-dependent set of keys would mean a rank-dependent number of collectives.
    cleared = GR.pop_diagnostics()
    check("pop_diagnostics clears", all(np.isnan(v) for v in cleared.values()))
    check("and always returns every key, whatever this rank saw",
          tuple(cleared) == GR.DIAG_KEYS and set(centre) == set(GR.DIAG_KEYS))


def main():
    test_flat_map_is_size_invariant()
    test_scale_invariance()
    test_planted_signal_has_the_right_sign()
    test_offsets()
    test_toroidal_fallback()
    test_degenerate_steps_are_absent_not_zero()
    test_clip()
    test_completion_level()
    test_dedupe()
    test_metric_dispatch()
    test_diagnostics()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print(f"  FAILED: {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
