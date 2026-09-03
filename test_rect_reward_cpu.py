#!/usr/bin/env python
"""CPU checks for the fixed-rectangle mask (--overlap_rect_frac).

    python test_rect_reward_cpu.py

No GPU, no Grounding-DINO, no model. The whole claim of the flag is that the reward needs
none of those, and this file enforces rather than asserts it: every reward call runs with
`overlap_rewards._dino_boxes` replaced by a bomb that fails the test if it is ever called.

The other claim worth a test is that the rectangle TRAINED ON is the rectangle MEASURED.
centre_box_probe.py is what motivated the flag; test 2 imports it and compares the two
constructions patch for patch over every grid the runs produce, so the training arm cannot
quietly drift from the probe that justified it.
"""
import sys
import os
import importlib.util
import types

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))

# Import trl/rewards/*.py without importing the `trl` package (which pulls torch,
# transformers and the trainer). Same trick test_maskfree_reward_cpu.py uses.
_pkg = types.ModuleType("trl_t"); _pkg.__path__ = [os.path.join(ROOT, "trl")]
sys.modules["trl_t"] = _pkg
_sub = types.ModuleType("trl_t.rewards"); _sub.__path__ = [os.path.join(ROOT, "trl", "rewards")]
sys.modules["trl_t.rewards"] = _sub


def _imp(name):
    spec = importlib.util.spec_from_file_location(
        f"trl_t.rewards.{name}", os.path.join(ROOT, "trl", "rewards", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"trl_t.rewards.{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


ORW = _imp("overlap_rewards")

FAILED = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}  {detail}")
        FAILED.append(name)


_REAL_DINO = ORW._dino_boxes


def _no_dino(*a, **k):
    raise AssertionError("Grounding-DINO was called under --overlap_rect_frac")


ORW._dino_boxes = _no_dino

F = 0.565           # the fraction the probe used, and the one a comparison run wants
GRID = (10, 16)     # the modal patch grid on these runs


def cfg(**kw):
    """Reset to the shipped defaults, then apply kw. configure() ignores None."""
    ORW._CFG.update(box_threshold=0.10, max_box_area=0.5, max_union_area=None,
                    metric="mean_in", mass_floor_tau=None, natural_only=False,
                    rect_frac=None)
    ORW.configure(**kw)


def steps(*maps):
    return [{"map": np.asarray(m, dtype=np.float32), "text": f"step {i}"}
            for i, m in enumerate(maps)]


def reward(saliency_map, **kw):
    n = len(saliency_map)
    return ORW.think_overlap_reward(
        completions=[[{"content": "x"}]] * n, saliency_map=saliency_map,
        image=[None] * n, **kw)


print("\n1. the rectangle's geometry")
cfg(rect_frac=F)
r = ORW._centre_rect_mask(*GRID, F)
check("aspect preserved: 8 x 12 on a 10 x 16 grid",
      (int(r.any(axis=1).sum()), int(r.any(axis=0).sum())) == (8, 12),
      f"got {(int(r.any(axis=1).sum()), int(r.any(axis=0).sum()))}")
check("contiguous block, no holes", r.sum() == 8 * 12, f"got {r.sum()}")
check("centred vertically", list(np.flatnonzero(r.any(axis=1))) == list(range(1, 9)))
check("centred horizontally", list(np.flatnonzero(r.any(axis=0))) == list(range(2, 14)))
# Rounding to whole patches moves the realised area off `frac`; that is expected and the
# reason the probe reports the realised number rather than the requested one.
check("realised area near the requested fraction", abs(r.mean() - F) < 0.05,
      f"realised {r.mean():.3f} for frac {F}")
check("bigger frac -> superset", np.all(ORW._centre_rect_mask(*GRID, 0.9)[r]),
      "a larger rectangle must contain the smaller one")
check("frac=1 is degenerate -> None (never scored as a mask)",
      ORW._centre_rect_mask(*GRID, 1.0) is None)
check("tiny frac still claims >=1 patch", ORW._centre_rect_mask(*GRID, 1e-9).sum() >= 1)
check("cache returns the same object", ORW._centre_rect_mask(*GRID, F) is r)

print("\n2. the trained rectangle IS the probe's rectangle")
_spec = importlib.util.spec_from_file_location(
    "centre_box_probe", os.path.join(ROOT, "centre_box_probe.py"))
if _spec is None or not os.path.exists(os.path.join(ROOT, "centre_box_probe.py")):
    check("centre_box_probe.py present", False, "cannot compare against the probe")
else:
    _probe = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_probe)
    bad = []
    for gh in range(1, 25):
        for gw in range(1, 25):
            for frac in (0.1, 0.3, 0.565, 0.75, 0.99):
                mine = ORW._centre_rect_mask(gh, gw, frac)
                theirs = _probe.centre_rect(gh, gw, frac)
                # The probe keeps the degenerate full-grid rectangle (it scores it as a
                # mask comparison); the reward refuses it, as it refuses a degenerate
                # union. Everywhere else they must agree patch for patch.
                if mine is None:
                    if theirs.all():
                        continue
                    bad.append((gh, gw, frac, "reward refused a non-degenerate rect"))
                elif not np.array_equal(mine, theirs):
                    bad.append((gh, gw, frac, "differs"))
    check("identical to centre_box_probe.centre_rect on 24x24x5 grids/fracs",
          not bad, f"{len(bad)} mismatches, first {bad[:3]}")

print("\n3. no detector is ever constructed")
cfg(rect_frac=F)
m = np.abs(np.random.default_rng(0).random(GRID)).astype(np.float32)
r = reward([steps(m)], valid_list=[True])
check("think_overlap_reward returns a score without calling DINO", r[0] is not None,
      f"got {r}")
check("_dino_boxes is still the bomb (nothing swapped it back)",
      ORW._dino_boxes is _no_dino)
check("rect_active() True with a fraction set", ORW.rect_active() is True)
cfg()
check("rect_active() False by default", ORW.rect_active() is False)
# The default path must still be the DINO one -- the bomb firing here is the proof that
# nothing about this flag leaked into runs that do not pass it.
try:
    reward([steps(m)], valid_list=[True])
    check("default path still calls DINO", False, "the bomb never fired")
except AssertionError:
    check("default path still calls DINO", True)

print("\n4. the score is the metric on the rectangle, and the metric is unchanged")
cfg(rect_frac=F)
rect = ORW._centre_rect_mask(*GRID, F)
rng = np.random.default_rng(1)
bad = 0
for _ in range(50):
    g = (rng.random(GRID) ** 3).astype(np.float32)
    got = reward([steps(g)], valid_list=[True])[0]
    if abs(got - ORW._mean_in(g, rect)) > 1e-12:
        bad += 1
check("mean_in on the rectangle, exactly", bad == 0, f"{bad}/50 disagreed")
for metric in ("mean_in_v2", "auroc"):
    cfg(rect_frac=F, metric=metric)
    g = (rng.random(GRID) ** 2).astype(np.float32)
    got = reward([steps(g)], valid_list=[True])[0]
    want = ORW._step_score(g, rect)
    check(f"--overlap_metric {metric} scores the rectangle through the same path",
          abs(got - want) < 1e-12, f"{got} vs {want}")

print("\n5. per-completion value, the format gate, and the mask cases")
cfg(rect_frac=F)
a = np.zeros(GRID, dtype=np.float32); a[5, 8] = 1.0          # peak inside the rectangle
b = np.zeros(GRID, dtype=np.float32); b[0, 0] = 1.0          # peak outside it
want = (ORW._mean_in(a, rect) + ORW._mean_in(b, rect)) / 2
check("mean over the completion's steps",
      abs(reward([steps(a, b)], valid_list=[True])[0] - want) < 1e-12)
check("a map peaking INSIDE beats one peaking outside",
      ORW._mean_in(a, rect) > ORW._mean_in(b, rect))
check("format gate zeroes it", reward([steps(a)], valid_list=[False])[0] == 0.0)
check("no observe steps -> None (masked, not 0)", reward([[]], valid_list=[True]) == [None])
# The behavioural difference from a DINO run, pinned down so it cannot regress silently:
# there is no such thing as an ungroundable step here, so every step is scored.
r = reward([steps(a, b, a, b)], valid_list=[True])
check("EVERY step is scored -- nothing is ungroundable", r[0] is not None)
# The one exception, and the only grid it applies to at frac 0.565. round(2*sqrt(.565))
# is 2, so on a 2x2 grid the rectangle rounds up to the whole thing and the step takes
# the degenerate-union path. Real patch grids are ~10x16; this is here so the exception
# is a measured edge rather than a surprise.
degen = [(gh, gw) for gh in range(2, 40) for gw in range(2, 40)
         if ORW._centre_rect_mask(gh, gw, F) is None]
check("only a 2x2 grid is degenerate at frac 0.565", degen == [(2, 2)], f"got {degen}")
check("a step on that grid is SKIPPED, not scored 0",
      reward([steps(np.ones((2, 2), dtype=np.float32))], valid_list=[True]) == [None])

print("\n6. --overlap_natural_only still masks, and still costs no grounding")
cfg(rect_frac=F, natural_only=True)
r = reward([steps(a), steps(b)], valid_list=[True, True], natural=[True, False])
check("non-natural row -> None", r[1] is None, f"got {r}")
check("natural row still scored", r[0] is not None, f"got {r}")

print("\n7. the two configurations that would fail silently are refused")
cfg()
for frac in (1.5, 2.0):
    try:
        ORW.configure(rect_frac=frac)
        check(f"frac {frac} > 1 refused", False, "no error raised")
    except ValueError:
        check(f"frac {frac} > 1 refused", True)
    cfg()
# The rectangle is the SAME on every step, so a union cap below its area drops all of
# them and the run trains with no overlap term while looking merely weak in the logs.
try:
    ORW.configure(rect_frac=0.565, max_union_area=0.4)
    check("frac above --max_union_area refused", False, "no error raised")
except ValueError:
    check("frac above --max_union_area refused", True)
cfg()
ORW.configure(rect_frac=0.565, max_union_area=0.6)
check("frac below --max_union_area accepted", ORW.rect_active() is True)
cfg()

print("\n8. random batches: finite, in range, right length")
cfg(rect_frac=F)
rng = np.random.default_rng(7)
bad = 0
for _ in range(300):
    n = int(rng.integers(1, 4))
    batch = [steps(*[(rng.random((int(rng.integers(2, 14)), int(rng.integers(2, 14))))
                      * 10.0 ** int(rng.integers(-7, 2))).astype(np.float32)
                     for _ in range(int(rng.integers(1, 4)))]) for _ in range(n)]
    out = reward(batch, valid_list=[True] * n)
    if len(out) != n:
        bad += 1
        continue
    for comp, v in zip(batch, out):
        if v is None:
            # Only legitimate when EVERY step of the completion had a degenerate
            # rectangle -- i.e. the 2x2 grid above. Anything else is a real failure.
            if all(ORW._centre_rect_mask(*st["map"].shape, F) is None for st in comp):
                continue
            bad += 1
            continue
        # mean_in is a mean of values in [0, 1], so unlike the mask-free rewards it is
        # bounded on both sides and the range check is arithmetic, not measurement.
        if not np.isfinite(v) or not (0.0 <= v <= 1.0):
            bad += 1
check("300 random batches", bad == 0, f"{bad} bad")

ORW._dino_boxes = _REAL_DINO
print("\n" + ("ALL PASS" if not FAILED else f"{len(FAILED)} FAILED: {FAILED}"))
sys.exit(1 if FAILED else 0)
