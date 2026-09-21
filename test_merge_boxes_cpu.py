#!/usr/bin/env python
"""CPU checks for --overlap_merge_boxes: every step against the whole chain's union.

    python test_merge_boxes_cpu.py

No GPU and no detector: `_dino_boxes` is replaced by a recording stub, which is also how
the central claims are tested. This arm's claims are different from every other mask
source's, because it is the only one that does NOT change the grounding:

  * the detector calls are the INCUMBENT's -- one per observe step, on that step's own
    sentence, the same count. An implementation that quietly grounded once per completion
    would be a cheaper arm answering a different question, and it would pass every test
    about the mask. The stub records the call boundary so it cannot.
  * the mask is the union of ALL of the completion's boxes, applied to every one of its
    steps. Checked against a hand-built union rather than against itself.
  * merging only GROWS the mask, so the failure mode is saturation rather than an empty
    union: at full coverage `_union_mask` refuses, and under this flag that costs the
    WHOLE completion. `mask/merged_cover` must be recorded for those completions too --
    a coverage mean taken over the survivors alone is exactly the number that would hide
    this -- and `mask/merged_unscored_frac` must count them.
  * a step that grounds nothing is no longer skipped: its neighbours' boxes give it a
    mask. That widens the scored set against the per-step reference, which is a real
    behavioural difference and is pinned in both directions.
"""
import importlib.util
import os
import sys
import types

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))

# Import trl/rewards/*.py without importing the `trl` package (which pulls torch,
# transformers and the trainer). Same trick test_chain_boxes_cpu.py uses.
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


GRID = (10, 16)

# One box per sentence, deterministic and distinct, so "which sentences went into the
# union" is readable straight off the mask. "s<k>" claims column band k; the two halves
# exist so a merged union can be driven to full coverage without any single box tripping
# the 0.5 per-box cap, which is the saturation case this arm has to be watched for.
CALLS = []


def fake_dino(images, texts):
    CALLS.append(list(zip(images, texts)))
    out = []
    for t in texts:
        if t == "ungroundable":
            out.append([])                      # DINO found nothing for this sentence
        elif t == "top":
            out.append([[0.0, 0.0, 1.0, 0.5]])  # area exactly at the cap, so it survives
        elif t == "bottom":
            out.append([[0.0, 0.5, 1.0, 1.0]])
        else:
            k = int(t.split("s")[-1]) % 4
            out.append([[0.25 * k, 0.1, 0.25 * k + 0.2, 0.9]])
    return out


ORW._dino_boxes = fake_dino


def cfg(**kw):
    """Reset to the shipped defaults, then apply kw. configure() ignores None."""
    ORW._CFG.update(box_threshold=0.10, max_box_area=0.5, max_union_area=None,
                    metric="mean_in", mass_floor_tau=None, natural_only=False,
                    rect_frac=None, rect_placement="centre", rect_seed=0,
                    chain_boxes=None, question_boxes=None, merge_boxes=False)
    ORW.configure(**kw)


def steps(*names, shape=GRID):
    rng = np.random.default_rng(abs(hash(names)) % 2**32)
    return [{"map": (rng.random(shape) ** 2).astype(np.float32), "text": n} for n in names]


def reward(batch, **kw):
    n = len(batch)
    kw.setdefault("valid_list", [True] * n)
    return ORW.think_overlap_reward(
        completions=[[{"content": f"c{i}"}] for i in range(n)],
        saliency_map=batch, image=[f"img{i}" for i in range(n)], **kw)


def hand_union(*names, shape=GRID):
    """The union a caller would build by hand from these sentences."""
    boxes = [b for lst in fake_dino(["x"] * len(names), list(names)) for b in lst]
    return ORW._union_mask(boxes, *shape)


print("\n1. the grounding is the INCUMBENT's -- this arm does not buy calls")
cfg(merge_boxes=True)
CALLS.clear()
reward([steps("s0", "s1", "s2"), steps("s3"), steps("s0", "s1")])
flat = [t for call in CALLS for _i, t in call]
check("3 completions, 6 steps -> 6 groundings, one per STEP",
      flat == ["s0", "s1", "s2", "s3", "s0", "s1"], f"got {flat}")
check("still a single batched call for the whole batch", len(CALLS) == 1, f"got {len(CALLS)}")
imgs = [i for call in CALLS for i, _t in call]
check("each step grounded with its own completion's image",
      imgs == ["img0"] * 3 + ["img1"] + ["img2"] * 2, f"got {imgs}")
# The distinguishing check against --overlap_chain_boxes, which IS a cost arm.
cfg(chain_boxes="last")
CALLS.clear()
reward([steps("s0", "s1", "s2")])
n_chain = len([t for call in CALLS for _i, t in call])
cfg(merge_boxes=True)
CALLS.clear()
reward([steps("s0", "s1", "s2")])
n_merge = len([t for call in CALLS for _i, t in call])
check("--merge-boxes costs 3 calls where --chain-boxes costs 1",
      (n_merge, n_chain) == (3, 1), f"merge {n_merge}, chain {n_chain}")

print("\n2. the mask is the union of ALL the steps' boxes")
cfg(merge_boxes=True)
batch = [steps("s0", "s1", "s2")]
got = reward(batch)[0]
mask = hand_union("s0", "s1", "s2")
want = float(np.mean([ORW._mean_in(st["map"], mask) for st in batch[0]]))
check("every step scored against the merged union", abs(got - want) < 1e-12, f"{got} vs {want}")
# ... and it really is WIDER than any one step's own union, or the arm is a no-op here.
own = [ORW._union_mask(fake_dino(["x"], [t])[0], *GRID) for t in ("s0", "s1", "s2")]
check("the merged union covers strictly more than each step's own",
      all(int(mask.sum()) > int(m.sum()) for m in own),
      f"merged {int(mask.sum())} vs {[int(m.sum()) for m in own]}")
# One completion's boxes must not leak into another's.
cfg(merge_boxes=True)
two = [steps("s0"), steps("s2")]
out = reward(two)
for i, name in enumerate(("s0", "s2")):
    m = hand_union(name)
    w = float(np.mean([ORW._mean_in(st["map"], m) for st in two[i]]))
    check(f"completion {i}'s union is its own boxes only", abs(out[i] - w) < 1e-12,
          f"{out[i]} vs {w}")

print("\n3. the incumbent path is untouched, and agrees where it must")
cfg()
check("merge_boxes_active() False by default", ORW.merge_boxes_active() is False)
cfg(merge_boxes=True)
check("merge_boxes_active() True with the flag", ORW.merge_boxes_active() is True)
check("mask_diag_active() True under the flag", ORW.mask_diag_active() is True)
# A chain whose steps all carry the SAME sentence has one union either way, so the two
# paths must agree to the bit. If they do not, the arm differs from its reference in the
# scoring path as well as in the target and no comparison between them is clean.
bad = 0
for i in range(50):
    batch = [steps(*(["s1"] * int(np.random.default_rng(i).integers(1, 5))))]
    cfg(merge_boxes=True)
    a = reward(batch)[0]
    cfg()
    b = reward(batch)[0]
    if a is None or b is None or abs(a - b) > 1e-12:
        bad += 1
check("50 uniform-sentence chains score identically both ways", bad == 0, f"{bad} differ")
# And on a MIXED chain they must NOT agree -- otherwise the flag is doing nothing.
cfg(merge_boxes=True)
mixed = [steps("s0", "s2")]
a = reward(mixed)[0]
cfg()
b = reward(mixed)[0]
check("a mixed chain scores differently from the per-step path", abs(a - b) > 1e-9,
      f"{a} vs {b}")

print("\n4. an ungroundable step is carried by its neighbours")
cfg(merge_boxes=True)
ORW.pop_mask_diagnostics()
batch = [steps("s0", "ungroundable", "s2")]
got = reward(batch)[0]
mask = hand_union("s0", "s2")
want = float(np.mean([ORW._mean_in(st["map"], mask) for st in batch[0]]))
check("the ungroundable step is SCORED, on the other steps' union",
      got is not None and abs(got - want) < 1e-12, f"{got} vs {want}")
# The per-step reference skips it instead, so the two arms score different step SETS --
# the behavioural difference this flag introduces, measured rather than assumed.
cfg()
per_step = reward(batch)[0]
check("the per-step path would have skipped it (different value)",
      abs(per_step - want) > 1e-9, f"{per_step} vs {want}")
# A completion where NOTHING grounds is still lost, exactly as on the per-step path.
cfg(merge_boxes=True)
ORW.pop_mask_diagnostics()
out = reward([steps("ungroundable", "ungroundable"), steps("s1")])
check("no step grounded -> None (masked, not 0)", out[0] is None, f"got {out}")
check("its neighbour is unaffected", out[1] is not None, f"got {out}")
d = ORW.pop_mask_diagnostics()
check("merged_unscored_frac counts it", abs(d["merged_unscored_frac"] - 0.5) < 1e-9,
      f"got {d['merged_unscored_frac']}")

print("\n5. saturation: the failure mode merging actually has")
cfg(merge_boxes=True)
ORW.pop_mask_diagnostics()
# top + bottom tile the grid. Neither box trips the 0.5 per-box cap on its own, which is
# the point: --max_box_area does not bound a union, and merging is the fastest way there.
out = reward([steps("top", "bottom")])
check("a merged union covering the whole grid drops the completion", out[0] is None,
      f"got {out}")
d = ORW.pop_mask_diagnostics()
check("merged_cover records it as 1.0 even though it was dropped",
      abs(d["merged_cover"] - 1.0) < 1e-9, f"got {d['merged_cover']}")
check("merged_unscored_frac is 1.0", abs(d["merged_unscored_frac"] - 1.0) < 1e-9,
      f"got {d['merged_unscored_frac']}")
# Each half ALONE is scoreable, so the loss is the merge's doing and not the boxes'.
cfg()
out = reward([steps("top", "bottom")])
check("the per-step path scores the same chain fine", out[0] is not None, f"got {out}")
# Coverage is a mean over completions, and it must not be silently restricted to the
# survivors: one saturated and one small completion read as the mean of the two.
cfg(merge_boxes=True)
ORW.pop_mask_diagnostics()
reward([steps("top", "bottom"), steps("s0")])
d = ORW.pop_mask_diagnostics()
small = float(hand_union("s0").mean())
check("merged_cover averages the dropped completion in",
      abs(d["merged_cover"] - (1.0 + small) / 2) < 1e-9,
      f"got {d['merged_cover']}, want {(1.0 + small) / 2}")

print("\n6. --max_union_area applies per completion")
cfg(merge_boxes=True, max_union_area=0.05)
check("a merged union above the cap drops the whole completion",
      reward([steps("s0", "s2")])[0] is None)
cfg(merge_boxes=True, max_union_area=0.9)
check("below the cap it is scored", reward([steps("s0", "s2")])[0] is not None)
# The cap is the bound this arm is meant to be run with, so it must bite on the MERGED
# union and not on the per-step ones: s0 alone sits under a cap that s0+s2 exceeds.
cfg(merge_boxes=True, max_union_area=0.2)
check("a cap the merged union trips but a single step does not",
      reward([steps("s0", "s2")])[0] is None and reward([steps("s0")])[0] is not None)

print("\n7. the gates that still apply")
cfg(merge_boxes=True)
check("format gate zeroes it", reward([steps("s0")], valid_list=[False])[0] == 0.0)
check("no observe steps -> None", reward([[]])[0] is None)
CALLS.clear()
cfg(merge_boxes=True, natural_only=True)
out = reward([steps("s0"), steps("s1")], natural=[True, False])
check("--overlap_natural_only masks the non-natural row", out[1] is None, f"got {out}")
flat = [t for call in CALLS for _i, t in call]
check("... and costs it no grounding call", flat == ["s0"], f"got {flat}")

print("\n8. configurations that would fail silently are refused")
for other, kw in (("--overlap_rect_frac", {"rect_frac": 0.565}),
                  ("--overlap_question_boxes", {"question_boxes": "/nonexistent.json"}),
                  ("--overlap_chain_boxes", {"chain_boxes": "last"})):
    cfg()
    try:
        ORW.configure(merge_boxes=True, **kw)
        check(f"merge_boxes + {other} refused", False, "no error raised")
    except ValueError:
        check(f"merge_boxes + {other} refused", True)
    cfg()
    # ... and in the other configure order, since a launcher may set either first.
    try:
        ORW.configure(**kw)
        ORW.configure(merge_boxes=True)
        check(f"{other} then merge_boxes refused", False, "no error raised")
    except ValueError:
        check(f"{other} then merge_boxes refused", True)
    cfg()

print("\n9. _raster_union: the split _union_mask was refactored through")
cfg()
rng = np.random.default_rng(11)
bad_eq = bad_none = 0
for _ in range(300):
    gh, gw = int(rng.integers(2, 14)), int(rng.integers(2, 14))
    boxes = [[float(a), float(b), float(a + c), float(b + d)]
             for a, b, c, d in rng.random((int(rng.integers(0, 4)), 4)) * [0.6, 0.6, 0.5, 0.5]]
    raw = ORW._raster_union(boxes, gh, gw)
    m = ORW._union_mask(boxes, gh, gw)
    n = int(raw.sum())
    # _union_mask is _raster_union plus two refusals, and nothing else.
    if m is None:
        if not (n == 0 or n == gh * gw):
            bad_none += 1
    elif not np.array_equal(m, raw):
        bad_eq += 1
check("_union_mask is _raster_union where it is defined", bad_eq == 0, f"{bad_eq} differ")
check("... and None exactly on the degenerate unions", bad_none == 0, f"{bad_none} wrong")
# The whole-grid box needs the per-box cap off to reach the rasteriser at all -- which is
# the next pair of checks, and the reason this one comes after them.
cfg(max_box_area=0.1)
check("_raster_union applies --max_box_area", not ORW._raster_union([[0, 0, 1, 1]], 4, 4).any())
cfg(max_box_area=0)
check("... and honours 0 = disabled", ORW._raster_union([[0, 0, 1, 1]], 4, 4).all())
check("_raster_union keeps an all-covered mask instead of refusing it",
      bool(ORW._raster_union([[0.0, 0.0, 1.0, 1.0]], 4, 4).all()))
check("... and _union_mask still refuses it",
      ORW._union_mask([[0.0, 0.0, 1.0, 1.0]], 4, 4) is None)

print("\n10. random batches")
cfg(merge_boxes=True)
rng = np.random.default_rng(5)
names = ("s0", "s1", "s2", "s3", "ungroundable", "top", "bottom")
bad = 0
for _ in range(200):
    n = int(rng.integers(1, 4))
    batch = [steps(*[names[int(rng.integers(0, len(names)))]
                     for _ in range(int(rng.integers(1, 5)))],
                   shape=(int(rng.integers(2, 14)), int(rng.integers(2, 14))))
             for _ in range(n)]
    out = reward(batch)
    if len(out) != n:
        bad += 1
        continue
    for v in out:
        if v is not None and (not np.isfinite(v) or not (0.0 <= v <= 1.0)):
            bad += 1
check("200 random batches", bad == 0, f"{bad} bad")

print("\n" + ("ALL PASS" if not FAILED else f"{len(FAILED)} FAILED: {FAILED}"))
sys.exit(1 if FAILED else 0)
