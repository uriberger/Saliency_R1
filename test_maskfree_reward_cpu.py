#!/usr/bin/env python
"""CPU checks for the two mask-free rewards (--maskfree flatness|mass).

    python test_maskfree_reward_cpu.py

No GPU, no Grounding-DINO, no model. The point of the reward is that it needs none of
those, and this file is where that claim is enforced rather than asserted: every test
runs with `overlap_rewards._dino_boxes` replaced by a bomb that fails the test if it is
ever called.
"""
import sys
import os
import importlib.util
import types

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))

# Import trl/rewards/*.py without importing the `trl` package (which pulls torch,
# transformers and the trainer). Same trick overlap_metric_spread.py uses.
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
MF = _imp("maskfree_rewards")

FAILED = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}  {detail}")
        FAILED.append(name)


ANCHOR = 18.0


def _no_dino(*a, **k):
    raise AssertionError("Grounding-DINO was called by a --maskfree reward")


ORW._dino_boxes = _no_dino


def steps(*maps):
    return [{"map": np.asarray(m, dtype=np.float32), "text": f"step {i}"}
            for i, m in enumerate(maps)]


def reward(kind, saliency_map, **kw):
    MF._DIAG.clear()
    MF.configure(kind=kind, parity=False, mass_anchor=ANCHOR)
    return MF.think_maskfree_reward(
        completions=[[{"content": "x"}]] * len(saliency_map),
        saliency_map=saliency_map, image=[None] * len(saliency_map), **kw)


print("\n1. the two values, against hand arithmetic")
m = np.array([[1.0, 2.0], [3.0, 4.0]])
check("flatness = mean/max", abs(MF.flatness(m) - (2.5 / 4.0)) < 1e-12,
      f"got {MF.flatness(m)}")
check("mass = log(sum)+anchor", abs(MF.mass(m, ANCHOR) - (np.log(10.0) + ANCHOR)) < 1e-12,
      f"got {MF.mass(m, ANCHOR)}")
check("flatness of a constant map is 1", abs(MF.flatness(np.full((3, 3), 7.0)) - 1.0) < 1e-12)
spike = np.zeros((4, 4)); spike[1, 1] = 1.0
check("flatness of a single spike is 1/n", abs(MF.flatness(spike) - 1.0 / 16) < 1e-12)

print("\n2. scale invariance -- flatness is shape only, mass is scale only")
for c in (0.5, 3.0, 1e4):
    check(f"flatness invariant under m -> {c}m",
          abs(MF.flatness(m) - MF.flatness(m * c)) < 1e-9)
    check(f"mass shifts by log({c}) under m -> {c}m",
          abs((MF.mass(m * c, ANCHOR) - MF.mass(m, ANCHOR)) - np.log(c)) < 1e-9)

print("\n3. flatness IS mean_in with the union replaced by the whole image")
rng = np.random.default_rng(0)
for _ in range(50):
    g = rng.random((5, 7)) ** 3
    full = np.ones((5, 7), dtype=bool)
    check_val = ORW._mean_in(g, full)
    if abs(check_val - MF.flatness(g)) > 1e-12:
        check("flatness == _mean_in(map, all-True mask)", False,
              f"{MF.flatness(g)} vs {check_val}")
        break
else:
    check("flatness == _mean_in(map, all-True mask)", True)

print("\n4. degenerate maps are UNSCORED (None), never 0")
check("flatness None on all-zero", MF.flatness(np.zeros((3, 3))) is None)
check("mass None on all-zero", MF.mass(np.zeros((3, 3))) is None)
check("flatness None on empty", MF.flatness(np.zeros((0, 0))) is None)
r = reward("flatness", [steps(np.zeros((3, 3)))])
check("completion with only degenerate steps -> None", r == [None], f"got {r}")
r = reward("mass", [[]])
check("completion with no observe steps -> None", r == [None], f"got {r}")

print("\n5. per-completion value is the MEAN over the completion's scored steps")
a, b = np.array([[1.0, 1.0], [1.0, 1.0]]), np.array([[0.0, 0.0], [0.0, 4.0]])
r = reward("flatness", [steps(a, b)])
check("mean over steps", abs(r[0] - (1.0 + 0.25) / 2) < 1e-12, f"got {r}")
# A degenerate step is skipped, not averaged in as 0 -- the same rule the overlap reward
# follows, and the reason commit 8489767 exists.
r = reward("flatness", [steps(a, np.zeros((2, 2)))])
check("degenerate step skipped, not scored 0", abs(r[0] - 1.0) < 1e-12, f"got {r}")

print("\n6. the format gate is multiplicative, as in think_overlap_reward")
r = reward("flatness", [steps(a)], valid_list=[False])
check("format-invalid -> 0", r == [0.0], f"got {r}")
r = reward("flatness", [steps(a)], valid_list=[True])
check("format-valid -> the value", abs(r[0] - 1.0) < 1e-12, f"got {r}")

print("\n7. --overlap_natural_only masks non-natural rows to None")
ORW._CFG["natural_only"] = True
try:
    r = reward("flatness", [steps(a), steps(a)], natural=[True, False])
    check("non-natural -> None, natural -> scored", r[1] is None and r[0] is not None,
          f"got {r}")
    try:
        reward("flatness", [steps(a)])
        check("missing 'natural' column raises", False)
    except KeyError:
        check("missing 'natural' column raises", True)
finally:
    ORW._CFG["natural_only"] = False

print("\n8. ordering: the reward moves the way the hypothesis says it should")
flat = np.full((6, 6), 0.02)
peaky = np.zeros((6, 6)); peaky[2, 3] = 0.72
check("flatter map scores higher on flatness",
      reward("flatness", [steps(flat)])[0] > reward("flatness", [steps(peaky)])[0])
# 1e-6, not 1e-9: `steps()` casts to float32 like the trainer does, and float32(0.02)*36
# differs from float32(0.72) in the last bits. Tightening this would be testing numpy.
check("equal-mass maps tie on mass",
      abs(reward("mass", [steps(flat)])[0] - reward("mass", [steps(peaky)])[0]) < 1e-6,
      "0.02*36 == 0.72, so these two carry identical image mass")
heavy = flat * 3.0
check("more image mass scores higher on mass",
      reward("mass", [steps(heavy)])[0] > reward("mass", [steps(flat)])[0])
check("more image mass does NOT move flatness",
      abs(reward("flatness", [steps(heavy)])[0] - reward("flatness", [steps(flat)])[0]) < 1e-9)

print("\n9. mass stays POSITIVE over the measured range (the nansum trap)")
# Under a `.nansum(dim=1)` fold an unscored reward reads as 0. If scored values were
# negative, 0 would be the best possible score and "produce no gradeable observe step"
# the best move. See maskfree_rewards.__doc__.
#
# The bounds are the MEASURED image-mass range over 13,648 observe steps and all ten
# models in the val_natural probe (min 1.40e-04, max 8.49e-02), plus three orders of
# magnitude of margin below it. This is the check that caught the first draft's anchor of
# 8.0, which needed 8.87 and went negative on the lowest ~1% of real steps.
# The lowest bound is 2e-08, just above exp(-18) = 1.52e-08, which is where the anchor
# stops covering. That is ~1e4 below the observed minimum, and stating it exactly is the
# point: the positivity guarantee is a measured margin, not an identity.
for total in (2e-8, 1e-7, 1e-6, 1.4e-4, 4.2e-4, 5.7e-3, 8.5e-2, 0.5):
    g = np.full((10, 10), total / 100.0)
    check(f"mass(sum={total:g}) > 0", MF.mass(g, ANCHOR) > 0, f"got {MF.mass(g, ANCHOR)}")
check("the default anchor is the one being tested", MF._CFG["mass_anchor"] == ANCHOR,
      f"module default {MF._CFG['mass_anchor']} != {ANCHOR}")

print("\n10. diagnostics: BOTH statistics recorded under EITHER kind, fixed key set")
for kind in MF.KINDS:
    reward(kind, [steps(a, b)])
    d = MF.pop_diagnostics()
    check(f"[{kind}] key set is exactly DIAG_KEYS", tuple(d) == MF.DIAG_KEYS, f"got {tuple(d)}")
    check(f"[{kind}] flatness recorded", np.isfinite(d["flatness"]))
    check(f"[{kind}] mass recorded", np.isfinite(d["mass"]))
    check(f"[{kind}] pop clears the buffer", all(
        not np.isfinite(v) for v in MF.pop_diagnostics().values()))

print("\n11. configure() rejects an unknown kind, and think_ refuses to run without one")
try:
    MF.configure(kind="grounding"); check("bad --maskfree raises", False)
except ValueError:
    check("bad --maskfree raises", True)
MF._CFG["kind"] = None
check("is_active() False when off", MF.is_active() is False)
check("needs_dino() False when off", MF.needs_dino() is False)
try:
    MF.think_maskfree_reward(saliency_map=[steps(a)], image=[None])
    check("reward refuses to run with kind=None", False)
except ValueError:
    check("reward refuses to run with kind=None", True)
MF.configure(kind="flatness")
check("needs_dino() False with parity off", MF.needs_dino() is False)
MF.configure(parity=True)
check("needs_dino() True with parity on", MF.needs_dino() is True)
MF.configure(parity=False)

print("\n12. parity mode uses DINO only as a GATE, never as a value")
# The bomb above is still installed for every other test; here it is replaced by a stub
# that grounds only the FIRST step, so the gate is observable.
calls = {"n": 0}


def _stub_boxes(images, texts):
    calls["n"] += 1
    return [[[0.1, 0.1, 0.4, 0.4]] if i == 0 else [] for i in range(len(texts))]


ORW._dino_boxes = _stub_boxes
try:
    MF._DIAG.clear()
    MF.configure(kind="flatness", parity=True)
    sm = [steps(a, b)]
    r_parity = MF.think_maskfree_reward(completions=[[{"content": "x"}]],
                                        saliency_map=sm, image=[None])
    check("parity called DINO exactly once (batched)", calls["n"] == 1, f"got {calls['n']}")
    # Only step 0 grounded, so the value must be step 0's flatness alone -- and it must be
    # the MASK-FREE value (1.0 over the whole map), not mean_in inside the box.
    check("parity keeps the mask-free VALUE, only the gate changes",
          abs(r_parity[0] - 1.0) < 1e-12, f"got {r_parity}")
    MF.configure(parity=False)
    r_free = MF.think_maskfree_reward(completions=[[{"content": "x"}]],
                                      saliency_map=sm, image=[None])
    check("without parity both steps count", abs(r_free[0] - 0.625) < 1e-12, f"got {r_free}")
finally:
    ORW._dino_boxes = _no_dino
    MF.configure(kind=None, parity=False)
    MF._CFG["kind"] = None

print("\n13. randomised: never NaN, monotone, shapes line up")
MF._CFG["kind"] = None
for kind in MF.KINDS:
    bad = 0
    for t in range(300):
        rg = np.random.default_rng(t)
        n = int(rg.integers(1, 5))
        sm = []
        for _ in range(n):
            k = int(rg.integers(0, 4))
            ms = []
            for _ in range(k):
                gh, gw = int(rg.integers(1, 8)), int(rg.integers(1, 8))
                mm = rg.random((gh, gw)) * rg.choice([0.0, 1e-4, 1.0])
                ms.append(mm)
            sm.append(steps(*ms) if ms else [])
        out = reward(kind, sm)
        if len(out) != n:
            bad += 1; continue
        for v in out:
            if v is None:
                continue
            # Finiteness is the invariant that holds for ANY input. Non-negativity is
            # NOT: `mass` is a log, so a map dimmer than anything real (these synthetic
            # ones go to 1e-7 on a 1x1 grid) can still go under the anchor. That case is
            # bounded by measurement in test 9, not by arithmetic, and pretending
            # otherwise here would hide it.
            if not np.isfinite(v):
                bad += 1
            if kind == "flatness" and not (0.0 <= v <= 1.0):
                bad += 1
    check(f"[{kind}] 300 random batches: finite, in range, right length", bad == 0,
          f"{bad} bad")

print("\n" + ("ALL PASS" if not FAILED else f"{len(FAILED)} FAILED: {FAILED}"))
sys.exit(1 if FAILED else 0)
