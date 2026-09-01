#!/usr/bin/env python
"""CPU checks for the length guard (--length-guard REF_TOKENS) and its off-by-default rule.

    python test_length_guard_reward_cpu.py

No GPU, no Grounding-DINO, no model, no attention map. The guard reads `completion_ids`
and nothing else, and this file enforces that rather than asserting it: the reward is
called with `overlap_rewards._dino_boxes` replaced by a bomb and with `saliency_map=None`,
so touching either would fail the test.

Two of the sections below are not about arithmetic:

  section 6 pins MEASURED completion lengths -- every one is a real number from a run on
  disk, and the band was calibrated against their distribution, not the other way round.
  It pins three things at once: that no healthy length is ever touched, that the three
  recorded length COLLAPSES are, and that the documented BLIND SPOT (set_c inflated only
  to 1.15x, inside the band) stays documented. A future widening of the band that quietly
  claims to close that blind spot fails here instead of 3,000 steps into a run.

  section 8 pins OFF-BY-DEFAULT structurally. trl_repo/ is shared and re-patched under
  jobs that are already queued, so a run submitted before this change and started after it
  must get a byte-identical reward_funcs and --reward_weights. That is a property of where
  the append sits in grpo_vlm_qwen3.py and of how the launcher emits its flags, so it is
  checked there, in the source, not by hoping.
"""
import ast
import importlib.util
import os
import re
import sys
import types

import math

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
LG = _imp("length_guard_rewards")

FAILED = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}  {detail}")
        FAILED.append(name)


def _no_dino(*a, **k):
    raise AssertionError("Grounding-DINO was called by the length guard")


ORW._dino_boxes = _no_dino

# The calibrated defaults, restated here rather than read from _CFG: a test that reads the
# value it is checking cannot catch a change to it.
REF, LO, HI, KNEE, K = 221.0, 0.30, 3.0, 1.0, 0.20


def p(n):
    return LG.penalty(n, REF, LO, HI, KNEE)


def reward(lengths, **kw):
    """Call the reward the way the trainer does, with no map and no image."""
    LG._DIAG.clear()
    LG._CFG.update(l_ref=REF, band_lo=LO, band_hi=HI, knee=KNEE)
    return LG.length_guard_reward(
        completions=[[{"content": "x"}]] * len(lengths),
        completion_ids=[[0] * n for n in lengths],
        saliency_map=None, image=None, valid_list=None, **kw)


print("\n1. the free band: exactly zero inside, at both edges, and at the reference")
check("reference length scores 0", p(REF) == 0.0, f"got {p(REF)}")
check("upper edge (band_hi x ref) is inside", p(REF * HI) == 0.0)
check("lower edge (band_lo x ref) is inside", p(REF * LO) == 0.0)
check("just past the upper edge is negative", p(REF * HI + 2.0) < 0.0)
check("just past the lower edge is negative", p(REF * LO - 2.0) < 0.0)
check("zero-length completion is scored, not skipped", p(0) < 0.0, f"got {p(0)}")
check("a 1-token completion is scored", p(1) < 0.0)

print("\n2. the piecewise value, against hand arithmetic (d = ln(n/l_ref))")
n = REF * HI * math.e ** 0.5                    # half a log unit past the upper edge
check("long side: -e^2 with e = ln(n/ref) - ln(band_hi)", abs(p(n) - -(0.5 ** 2)) < 1e-9,
      f"got {p(n)}")
n = REF * LO / math.e ** 0.5                    # half a log unit below the lower edge
check("short side: -e^2 with e = ln(band_lo) - ln(n/ref)", abs(p(n) - -(0.5 ** 2)) < 1e-9,
      f"got {p(n)}")
check("LOG-SYMMETRIC: equal log distance either side costs the same",
      abs(p(REF * HI * math.e) - p(REF * LO / math.e)) < 1e-9,
      f"{p(REF * HI * math.e)} vs {p(REF * LO / math.e)}")
e_far = 2.5
check("past the knee it is linear",
      abs(p(REF * HI * math.e ** e_far) - -(KNEE ** 2 + 2 * KNEE * (e_far - KNEE))) < 1e-8,
      f"got {p(REF * HI * math.e ** e_far)}")

print("\n3. the knee is C1 -- value AND slope agree, so there is no kink to sit in")
h = 1e-6
n_knee = REF * HI * math.e ** KNEE
lo, hi = p(n_knee - h * REF), p(n_knee + h * REF)
check("value is continuous at the knee", abs(p(n_knee) - -(KNEE ** 2)) < 1e-9,
      f"got {p(n_knee)}")
slope_lo = (p(n_knee) - lo) / (h * REF)
slope_hi = (hi - p(n_knee)) / (h * REF)
check("slope is continuous at the knee", abs(slope_lo - slope_hi) < 1e-4,
      f"{slope_lo:.8f} vs {slope_hi:.8f}")

print("\n4. monotone, and never positive")
longs = [p(n) for n in range(int(REF * HI), 6 * int(REF), 7)]
shorts = [p(n) for n in range(int(REF * LO), 0, -1)]
check("non-increasing as completions get longer", all(b <= a + 1e-12 for a, b in zip(longs, longs[1:])))
check("non-increasing as completions get shorter", all(b <= a + 1e-12 for a, b in zip(shorts, shorts[1:])))
check("never positive", all(v <= 0.0 for v in longs + shorts))
check("the best attainable score is exactly 0", max(p(n) for n in range(1, 6 * int(REF))) == 0.0)
# The slope grows with the excess: this is what --placebo length lacked (it was linear,
# so once a group converged on one length its within-group spread vanished and the term
# switched itself off -- frac_reward_zero_std 0.000 -> 0.64).
d70 = abs(p(71) - p(69))
d30 = abs(p(31) - p(29))
check("penalty spread grows with the excess (vs placebo length's constant slope)",
      d30 > 5 * d70, f"d@70={d70:.3e} d@30={d30:.3e}")

print("\n5. the reward: every completion scored, never None, no map touched")
lens = [50, 150, 217, 260, 400, 900]
r = reward(lens)
check("one score per completion", len(r) == len(lens))
check("never None", all(v is not None for v in r))
check("all floats", all(isinstance(v, float) for v in r))
check("matches penalty() elementwise", all(abs(a - p(n)) < 1e-12 for a, n in zip(r, lens)))
check("deterministic across calls", reward(lens) == r)
check("empty batch is harmless", reward([]) == [])
try:
    LG._CFG.update(l_ref=REF)
    LG.length_guard_reward(completions=[[{"content": "x"}]], completion_ids=None)
    check("missing completion_ids raises", False, "no exception")
except KeyError as e:
    check("missing completion_ids raises KeyError naming the kwarg", "completion_ids" in str(e))
_saved = LG._CFG["l_ref"]
LG._CFG["l_ref"] = None
try:
    LG.length_guard_reward(completions=[[{"content": "x"}]], completion_ids=[[0]])
    check("unconfigured guard raises", False, "no exception")
except ValueError:
    check("unconfigured guard raises ValueError", True)
LG._CFG["l_ref"] = _saved

print("\n5b. configure() rejects settings that would silently disable half the term")
for kwargs, why in [
    (dict(l_ref=0), "l_ref must be positive"),
    (dict(l_ref=-5), "l_ref must be positive"),
    (dict(l_ref=REF, band_lo=3.0, band_hi=0.30), "band_lo >= band_hi is an empty band"),
    (dict(l_ref=REF, band_lo=1.0, band_hi=1.0), "a zero-width band penalises everything"),
    (dict(l_ref=REF, band_lo=1.5), "band_lo > 1 excludes the reference length itself"),
    (dict(l_ref=REF, band_hi=0.8), "band_hi < 1 excludes the reference length itself"),
    (dict(l_ref=REF, band_lo=0.0), "band edges are multiples and must be > 0"),
    (dict(l_ref=REF, knee=0.0), "knee must be > 0"),
]:
    saved = dict(LG._CFG)
    try:
        LG.configure(**kwargs)
        check(f"configure rejects: {why}", False, "no exception")
    except ValueError:
        check(f"configure rejects: {why}", True)
    finally:
        LG._CFG.clear(); LG._CFG.update(saved)
check("is_active() is False with no reference length",
      (LG._CFG.update(l_ref=None), LG.is_active())[1] is False)
LG._CFG.update(l_ref=REF)

print("\n6. regression fixture: the ten MEASURED lengths this was calibrated against")
# n, k*penalty rounded to 3dp, and what the row is. Sources: overlap-reward-hack-set-a.md
# (set_a), the set_c/set_d trainer_state.json curves, and the direction-controls table.
# The four "silent" rows are the guarantee that the guard does not touch a healthy run;
# the set_c MEAN row is the documented BLIND SPOT and is pinned so that a future widening
# of the band cannot quietly claim to have closed it.
FIXTURE = [
    # HEALTHY -- every one of these must be exactly 0, or the guard is taxing the
    # behaviour every good run on record exhibits.
    (146, 0.0000, "8k mean_in final, healthy       (0.66x) -> silent"),
    (176, 0.0000, "set_d final, healthy            (0.80x) -> silent"),
    (178, 0.0000, "set_c trough @1200, healthy     (0.81x) -> silent"),
    (221, 0.0000, "the reference length itself     (1.00x) -> silent"),
    (103, 0.0000, "cold-start p1                   (0.47x) -> silent"),
    # THE DOCUMENTED BLIND SPOT -- set_c inflated only to 1.15x (mean) and 1.81x (tail),
    # both inside the band. Pinned so a future widening cannot quietly claim to close it.
    (254, 0.0000, "set_c MEAN @3600                (1.15x) -> silent, BLIND SPOT"),
    (400, 0.0000, "set_c long tail @3600           (1.81x) -> silent, BLIND SPOT"),
    (356, 0.0000, "set_a hack mean @1900           (1.61x) -> silent, BLIND SPOT"),
    # LONG SIDE -- weak by design: truncation already costs accuracy AND format (~2.0).
    (900, -0.0186, "near the 1024 cap               (4.07x) -> soft ramp"),
    (1024, -0.0376, "truncated (already scores 0/0)  (4.63x) -> soft ramp"),
    # SHORT SIDE -- the side that does the work; nothing else penalises these at all.
    (76, 0.0000, "auroc cp2500's LONGEST           (0.34x) -> silent"),
    (49, -0.0184, "auroc collapse mean              (0.22x) -> bites"),
    (31, -0.1159, "maskfree mass collapse           (0.14x) -> bites hard"),
    (13, -0.4521, "placebo length collapse          (0.06x) -> bites hard"),
]
for n, want, label in FIXTURE:
    got = K * p(n)
    check(f"n={n:>4}  {label}", abs(got - want) < 5e-4, f"got {got:+.4f}, want {want:+.3f}")
# The asymmetry is the design, so assert its direction rather than leaving it to prose:
# a collapse must cost more than an equally-extreme inflation, because inflation already
# has a hard guard (truncation loses accuracy AND format) and collapse has none.
check("equal log distance either side costs EXACTLY the same",
      abs(p(REF * LO / math.e) - p(REF * HI * math.e)) < 1e-12,
      f"{p(REF * LO / math.e)} vs {p(REF * HI * math.e)}")
# The asymmetry is not in the shape -- it is in how far each side can REACH. The long side
# is capped by max_completion_length (1024 = 4.63x l_ref = 0.43 log units past the band);
# the short side is unbounded and the recorded collapses went to 0.06x = 1.61 log units.
check("the long side is capped by max_completion_length, at under half a log unit out",
      math.log(1024 / REF) - math.log(HI) < 0.5)
check("the short side reaches 3x further out (13 tokens)",
      math.log(LO) - math.log(13 / REF) > 3 * (math.log(1024 / REF) - math.log(HI)))
check("so a 13-token collapse is expensive (>= 0.25 reward units) where 1024 is not",
      abs(K * p(13)) >= 0.25 and abs(K * p(1024)) < 0.05,
      f"13 -> {K*p(13):.4f}, 1024 -> {K*p(1024):.4f}")

print("\n7. diagnostics")
check("DIAG_KEYS is fixed and non-empty", len(LG.DIAG_KEYS) > 0)
# band at l_ref=221 is [66.3, 663]: 40 is short, 900 is long, 150/221 are inside.
r = reward([40, 150, 221, 900])
d = LG.pop_diagnostics()
check("pop returns exactly DIAG_KEYS", set(d) == set(LG.DIAG_KEYS), f"got {sorted(d)}")
check("frac_long counts the one long one", abs(d["frac_long"] - 0.25) < 1e-9, f"got {d['frac_long']}")
check("frac_short counts the one short one", abs(d["frac_short"] - 0.25) < 1e-9, f"got {d['frac_short']}")
check("frac_penalized = long + short", abs(d["frac_penalized"] - 0.5) < 1e-9, f"got {d['frac_penalized']}")
check("mean_len is the batch mean", abs(d["mean_len"] - np.mean([40, 150, 221, 900])) < 1e-9)
check("mean_logratio is 0 for a batch all at the reference",
      (reward([221] * 4), abs(LG.pop_diagnostics()["mean_logratio"]) < 1e-9)[1])
check("pop clears", all(np.isnan(v) for v in LG.pop_diagnostics().values()))
reward([217] * 8)
d = LG.pop_diagnostics()
check("a wholly in-band batch reports frac_penalized 0.0", d["frac_penalized"] == 0.0)
check("... and mean_penalty 0.0", d["mean_penalty"] == 0.0)

print("\n8. OFF BY DEFAULT, checked structurally in the sources that decide it")
# (a) a fresh import is inert, so the trainer's drain and grpo_vlm_qwen3's install both
#     see False without anything having to set them.
src_cfg = re.search(r'^_CFG = \{(.*?)^\}', open(os.path.join(ROOT, "trl", "rewards",
                    "length_guard_rewards.py")).read(), re.S | re.M).group(1)
check("module default l_ref is None (guard inert on import)",
      re.search(r'"l_ref":\s*None', src_cfg) is not None)
check("module default band is the measured 0.30x..3.0x",
      re.search(r'"band_lo":\s*0\.30', src_cfg) and re.search(r'"band_hi":\s*3\.0', src_cfg))

# (b) the ONLY place `length_guard_reward` is named in grpo_vlm_qwen3.py must be inside
#     `if script_args.length_guard_ref is not None:`. This is the check that matters: it
#     is what makes reward_funcs and --reward_weights byte-identical for a run that does
#     not ask for the guard, including one already queued when trl_repo/ is re-patched.
vlm_src = open(os.path.join(ROOT, "trl", "grpo_vlm_qwen3.py")).read()
tree = ast.parse(vlm_src)
guard_ranges = [
    (nd.body[0].lineno, max(getattr(x, "end_lineno", x.lineno) for x in nd.body))
    for nd in ast.walk(tree)
    if isinstance(nd, ast.If) and "length_guard_ref" in ast.dump(nd.test)
]
check("grpo_vlm_qwen3 has a `length_guard_ref is not None` guard", len(guard_ranges) == 1,
      f"found {len(guard_ranges)}")
named = [nd.lineno for nd in ast.walk(tree)
         if isinstance(nd, ast.Name) and nd.id == "length_guard_reward"]
check("length_guard_reward is referenced at all", len(named) > 0)
check("every reference to it sits inside that guard",
      all(any(a <= ln <= b for a, b in guard_ranges) for ln in named),
      f"stray at lines {[ln for ln in named if not any(a <= ln <= b for a, b in guard_ranges)]}")

# (c) the launcher must not emit either regulator's flag when it is off, or a run that
#     names neither would stop reproducing the command line of every run on record.
sh = open(os.path.join(ROOT, "launch_grpo_qwen3_overlap_colocated_job.sh")).read()
check("launcher default BETA=0", re.search(r'^BETA=0\s*$', sh, re.M) is not None)
check("launcher default LENGTH_GUARD_REF empty",
      re.search(r'^LENGTH_GUARD_REF=""', sh, re.M) is not None)
check("launcher band defaults match the module's",
      re.search(r'^LENGTH_GUARD_BAND_LO=0\.30', sh, re.M) and
      re.search(r'^LENGTH_GUARD_BAND_HI=3\.0', sh, re.M))
check("launcher k default matches the dataclass's 0.20",
      re.search(r'^LENGTH_GUARD_WEIGHT=0\.20', sh, re.M) is not None)
check("BETA_FLAG starts empty and is set only when beta != 0",
      re.search(r'BETA_FLAG=""\s*\n\[\[ "\$BETA" != "0" .*\]\] && BETA_FLAG="--beta \$BETA"', sh) is not None)
check("LENGTH_GUARD_FLAG starts empty and is set only inside the -n test",
      re.search(r'LENGTH_GUARD_FLAG=""\s*\nif \[\[ -n "\$LENGTH_GUARD_REF" \]\]; then', sh) is not None)
check("--length_guard_ref is emitted from exactly one place",
      sh.count("--length_guard_ref ") == 1, f"found {sh.count('--length_guard_ref ')}")

# (d) patch_trl_qwen3.sh must carry the module. The trainer imports it OUTSIDE every
#     reward_variant branch, so a missing cp line is an ImportError on the first metrics
#     log of ANY run -- the 069bd32 failure, which surfaced only on the second cluster.
patch = open(os.path.join(ROOT, "patch_trl_qwen3.sh")).read()
check("patch_trl_qwen3.sh copies length_guard_rewards.py",
      "length_guard_rewards.py" in patch)

print("\n" + "=" * 70)
if FAILED:
    print(f"FAILED {len(FAILED)}: {', '.join(FAILED)}")
    sys.exit(1)
print("all length-guard CPU checks passed")
