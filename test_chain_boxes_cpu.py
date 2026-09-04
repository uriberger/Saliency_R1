#!/usr/bin/env python
"""CPU checks for --overlap_chain_boxes: one Grounding-DINO call per COMPLETION.

    python test_chain_boxes_cpu.py

No GPU and no detector: `_dino_boxes` is replaced by a recording stub, which is also how
the central claim is tested. The flag's whole justification is a cost claim and a
one-variable claim, and both are only checkable at the call boundary:

  * ONE call per completion, on the CHOSEN step's sentence. Not one per step, not one per
    prompt. The stub records every (image, text) it is asked for, so an implementation
    that quietly kept per-step grounding fails here rather than in a 30-hour run.
  * given the same boxes, this path and the per-step path score IDENTICALLY. That is what
    makes the arm differ from its reference in which sentence was grounded and in nothing
    else; if the two scoring paths disagree, every comparison between them is confounded.
  * a completion whose chosen step grounds nothing is unscored AS A WHOLE -- masked, not
    zeroed, and with no fallback to a second sentence. The rate is logged; the test pins
    both the behaviour and the number.
"""
import importlib.util
import os
import sys
import types

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))

# Import trl/rewards/*.py without importing the `trl` package (which pulls torch,
# transformers and the trainer). Same trick test_rect_reward_cpu.py uses.
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

# One box per sentence, deterministic and distinct, so "which sentence was grounded" is
# readable straight off the mask. Sentence "s<k>" claims column band k.
CALLS = []


def fake_dino(images, texts):
    CALLS.append(list(zip(images, texts)))
    out = []
    for t in texts:
        if t == "ungroundable":
            out.append([])                      # DINO found nothing for this sentence
            continue
        k = int(t.split("s")[-1]) % 4
        out.append([[0.25 * k, 0.1, 0.25 * k + 0.2, 0.9]])
    return out


ORW._dino_boxes = fake_dino


def cfg(**kw):
    """Reset to the shipped defaults, then apply kw. configure() ignores None."""
    ORW._CFG.update(box_threshold=0.10, max_box_area=0.5, max_union_area=None,
                    metric="mean_in", mass_floor_tau=None, natural_only=False,
                    rect_frac=None, rect_placement="centre", rect_seed=0,
                    chain_boxes=None, question_boxes=None)
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


print("\n1. one call per COMPLETION, on the chosen step")
for sel, want_text in (("last", "s2"), ("first", "s0")):
    cfg(chain_boxes=sel)
    CALLS.clear()
    batch = [steps("s0", "s1", "s2")]
    reward(batch)
    check(f"{sel}: exactly one batched call", len(CALLS) == 1, f"got {len(CALLS)}")
    check(f"{sel}: exactly one sentence grounded for a 3-step chain",
          len(CALLS[0]) == 1, f"got {[t for _i, t in CALLS[0]]}")
    check(f"{sel}: it is the {sel} step", CALLS[0][0][1] == want_text,
          f"got {CALLS[0][0][1]}")
    check(f"{sel}: with that completion's own image", CALLS[0][0][0] == "img0")

cfg(chain_boxes="last")
CALLS.clear()
reward([steps("s0", "s1", "s2"), steps("s3"), steps("s0", "s1")])
flat = [t for call in CALLS for _i, t in call]
check("3 completions, 6 steps -> 3 groundings", len(flat) == 3, f"got {flat}")
check("... one per completion, each its own last step", flat == ["s2", "s3", "s1"],
      f"got {flat}")

print("\n2. the incumbent path is untouched")
cfg()
CALLS.clear()
reward([steps("s0", "s1", "s2")])
flat = [t for call in CALLS for _i, t in call]
check("without the flag it is still ONE call per STEP", flat == ["s0", "s1", "s2"],
      f"got {flat}")
check("chain_boxes_active() False by default", ORW.chain_boxes_active() is False)
cfg(chain_boxes="last")
check("chain_boxes_active() True with the flag", ORW.chain_boxes_active() is True)

print("\n3. same boxes -> same score as the per-step path")
# The one-variable claim. A chain whose steps all carry the SAME sentence gets the same
# boxes either way, so the two paths must agree to the bit; if they do not, the arm
# differs from its reference in the scoring path too and no comparison is clean.
bad = 0
for i in range(50):
    batch = [steps(*(["s1"] * int(np.random.default_rng(i).integers(1, 5))))]
    cfg(chain_boxes="last")
    a = reward(batch)[0]
    cfg()
    b = reward(batch)[0]
    if a is None or b is None or abs(a - b) > 1e-12:
        bad += 1
check("50 uniform-sentence chains score identically both ways", bad == 0, f"{bad} differ")

# And the mask really is the chosen step's, applied to every step: score by hand.
cfg(chain_boxes="last")
batch = [steps("s0", "s1", "s2")]
got = reward(batch)[0]
mask = ORW._union_mask(fake_dino(["img0"], ["s2"])[0], *GRID)
want = float(np.mean([ORW._mean_in(st["map"], mask) for st in batch[0]]))
check("every step scored against the LAST step's union", abs(got - want) < 1e-12,
      f"{got} vs {want}")

print("\n4. an ungroundable chosen step costs the whole completion")
cfg(chain_boxes="last")
ORW.pop_mask_diagnostics()
out = reward([steps("s0", "s1", "ungroundable"), steps("s0", "s1")])
check("that completion is None (masked, not 0)", out[0] is None, f"got {out}")
check("its neighbour is unaffected", out[1] is not None, f"got {out}")
d = ORW.pop_mask_diagnostics()
check("chain_ungrounded_frac logs the rate", abs(d["chain_ungrounded_frac"] - 0.5) < 1e-9,
      f"got {d['chain_ungrounded_frac']}")
check("mask_diag_active() True under the flag", ORW.mask_diag_active() is True)
# No fallback: the point of the flag is one call, and a second sentence would be a second
# call. Pinned so a well-meaning fix cannot land silently.
CALLS.clear()
reward([steps("s0", "s1", "ungroundable")])
check("no second call is made to rescue it", len(CALLS) == 1 and len(CALLS[0]) == 1,
      f"got {CALLS}")
# Under the per-step path the same chain keeps its other two steps -- the behavioural
# difference this flag introduces, measured rather than assumed.
cfg()
out = reward([steps("s0", "s1", "ungroundable")])
check("per-step grounding would have kept the completion", out[0] is not None, f"got {out}")

print("\n5. --max_union_area applies per completion now")
cfg(chain_boxes="last", max_union_area=0.05)
out = reward([steps("s0", "s1", "s2")])
check("a union above the cap drops the whole completion", out[0] is None, f"got {out}")
cfg(chain_boxes="last", max_union_area=0.9)
check("below the cap it is scored", reward([steps("s0", "s1", "s2")])[0] is not None)

print("\n6. the gates that still apply")
cfg(chain_boxes="last")
check("format gate zeroes it", reward([steps("s0")], valid_list=[False])[0] == 0.0)
check("no observe steps -> None", reward([[]])[0] is None)
CALLS.clear()
cfg(chain_boxes="last", natural_only=True)
out = reward([steps("s0"), steps("s1")], natural=[True, False])
check("--overlap_natural_only masks the non-natural row", out[1] is None, f"got {out}")
flat = [t for call in CALLS for _i, t in call]
check("... and costs it no grounding call", flat == ["s0"], f"got {flat}")

print("\n7. configurations that would fail silently are refused")
cfg()
for bad_sel in ("middle", "LAST", "random"):
    try:
        ORW.configure(chain_boxes=bad_sel)
        check(f"selector {bad_sel!r} refused", False, "no error raised")
    except ValueError:
        check(f"selector {bad_sel!r} refused", True)
    cfg()
# An empty string is the launcher's "flag not given", not a bad selector: it must leave
# the incumbent per-step path alone rather than raise at every default invocation.
ORW.configure(chain_boxes="")
check("an empty selector is the flag being OFF, not an error",
      ORW.chain_boxes_active() is False)
cfg()
for other, kw in (("--overlap_rect_frac", {"rect_frac": 0.565}),
                  ("--overlap_question_boxes", {"question_boxes": "/nonexistent.json"})):
    cfg()
    try:
        ORW.configure(chain_boxes="last", **kw)
        check(f"chain_boxes + {other} refused", False, "no error raised")
    except ValueError:
        check(f"chain_boxes + {other} refused", True)
    cfg()
    # ... and in the other configure order, since a launcher may set either first.
    try:
        ORW.configure(**kw)
        ORW.configure(chain_boxes="last")
        check(f"{other} then chain_boxes refused", False, "no error raised")
    except ValueError:
        check(f"{other} then chain_boxes refused", True)
    cfg()

print("\n8. random batches")
cfg(chain_boxes="last")
rng = np.random.default_rng(5)
bad = 0
for _ in range(200):
    n = int(rng.integers(1, 4))
    batch = [steps(*[f"s{int(rng.integers(0, 4))}" for _ in range(int(rng.integers(1, 5)))],
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
