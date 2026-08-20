"""CPU tests for the three placebo rewards (--placebo roll|random|length).

The one that matters is T2/T3: a placebo must return UNSCORED on exactly the completions
the real overlap reward would leave unscored. If `random` or `length` scored a completion
mean_in skips, a placebo run would differ from its reference in two ways at once -- the
direction of the auxiliary signal AND which rollouts receive one -- and the experiment in
docs/next-reward-experiments.md would answer neither question. No 8B model, no GPU, no
Grounding-DINO (mocked).

    python test_placebo_reward_cpu.py                    # tests trl_repo/ (what runs)
    OVERLAP_TEST_TREE=local python test_placebo_reward_cpu.py   # tests this checkout
"""
import importlib.util
import os
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent

# Which copy of the sources to test -- same switch and same reasoning as
# test_overlap_reward_cpu.py: from a git worktree, copying into the shared trl_repo/
# would hit every other session and every running job, so `local` is the only option there.
_LOCAL = os.environ.get("OVERLAP_TEST_TREE") == "local"
_REWARDS_DIR = Path("trl/rewards") if _LOCAL else Path("trl_repo/trl/rewards")
_PKG = "trl_local" if _LOCAL else "trl_running"


def _load_tree(pkg):
    """Stub `<pkg>` / `<pkg>.rewards` at the tree under test and load both modules there.

    placebo_rewards imports its siblings relatively (`from . import overlap_rewards`), so
    they must share one package whose __path__ points at exactly one tree -- otherwise a
    checkout's sources could be spliced into a trl_repo/ run.
    """
    d = ROOT / _REWARDS_DIR
    for name, path in ((pkg, d.parent), (f"{pkg}.rewards", d)):
        m = types.ModuleType(name)
        m.__path__ = [str(path)]
        sys.modules[name] = m

    def _load(mod_name):
        spec = importlib.util.spec_from_file_location(
            f"{pkg}.rewards.{mod_name}", d / f"{mod_name}.py"
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"{pkg}.rewards.{mod_name}"] = mod
        spec.loader.exec_module(mod)
        setattr(sys.modules[f"{pkg}.rewards"], mod_name, mod)
        return mod

    orw = _load("overlap_rewards")
    plc = _load("placebo_rewards")
    return orw, plc


orw, plc = _load_tree(_PKG)
print(f"[tree] testing {'this checkout' if _LOCAL else 'trl_repo/ (the running copy)'}: {_REWARDS_DIR}")


class _Img:
    size = (64, 64)


def _map(peak_rc, peak=3.0, gh=6, gw=6, floor=0.05):
    m = np.full((gh, gw), floor, dtype=np.float32)
    m[peak_rc] = peak
    return m


# ---------------------------------------------------------------------------
# Test 1: the three placebo values, in isolation
# ---------------------------------------------------------------------------
# length: linear, monotone decreasing, and -n/1000 up to the constant `anchor`. The
# anchor keeps every scored completion NON-NEGATIVE, so that under the pre-8489767
# `nansum` fold that trl_repo still runs -- where an unscored reward reads as 0 -- an
# unscored completion looks like the LONGEST possible one rather than the shortest.
assert plc.length_score(0, anchor=0.0) == 0.0
assert plc.length_score(250, anchor=0.0) == -0.25
assert plc.length_score(300) < plc.length_score(299), "length must be DECREASING in tokens"
# affine: the anchor shifts every score by the same amount, so within-group spread (and
# hence the calibrated weight, and hence the GRPO advantage) is untouched.
_diffs = {plc.length_score(n, anchor=a) - plc.length_score(n)
          for n in (10, 300, 900) for a in (0.0, 1024.0)}
assert len(_diffs) == 2, _diffs
assert plc.length_score(1024, anchor=1024.0) == 0.0        # the cap is the floor
assert plc.length_score(0, anchor=1024.0) == 1.024
assert plc.length_score(200, anchor=1024.0) > 0, "a scored completion must beat unscored-as-0"
print("[T1a] length = (anchor - n_tokens)/1000: decreasing, affine in the anchor, "
      "non-negative up to the completion cap")

# random: in [0,1), deterministic, different for different text, uniform-ish.
u = [plc.uniform01(f"completion number {i}") for i in range(20000)]
assert all(0.0 <= x < 1.0 for x in u)
assert plc.uniform01("abc") == plc.uniform01("abc"), "must be deterministic"
assert plc.uniform01("abc") != plc.uniform01("abd")
assert plc.uniform01("abc", seed=0) != plc.uniform01("abc", seed=1), "seed must move it"
assert abs(float(np.mean(u)) - 0.5) < 0.01, float(np.mean(u))
assert abs(float(np.std(u)) - (1 / 12) ** 0.5) < 0.01, float(np.std(u))
print(f"[T1b] random: deterministic U(0,1), mean {np.mean(u):.4f} sd {np.std(u):.4f} "
      f"(target 0.5000 / {(1/12)**0.5:.4f})")

# roll: same area, actually moved, deterministic in (seed, prompt, step text).
mask = np.zeros((8, 8), dtype=bool)
mask[2:5, 1:4] = True
r1, i1 = plc.roll_mask(mask, "why is the sky blue?", "the red car on the left")
r2, _ = plc.roll_mask(mask, "why is the sky blue?", "the red car on the left")
r3, _ = plc.roll_mask(mask, "why is the sky blue?", "a stop sign")
r4, _ = plc.roll_mask(mask, "a different prompt", "the red car on the left")
assert r1.sum() == mask.sum(), (r1.sum(), mask.sum())          # area preserved
assert not np.array_equal(r1, mask), "the roll must not be the identity"
assert np.array_equal(r1, r2), "same (seed, prompt, step) -> same wrong place, every epoch"
assert not np.array_equal(r1, r3), "a different step must get its own offset"
assert not np.array_equal(r1, r4), "a different prompt must get its own offset"
assert i1["toroidal"] is False, i1                              # in-frame here
# in-frame really means in-frame: the rolled mask keeps its bounding-box SHAPE
_rows, _cols = np.nonzero(r1)
assert (_rows.max() - _rows.min(), _cols.max() - _cols.min()) == (2, 2), i1
print(f"[T1c] roll: area preserved ({int(mask.sum())} patches), in-frame, "
      f"deterministic per (prompt, step), offset {i1['offset']}")

# A union whose bounding box already spans the grid cannot move in-frame -> the sampler
# falls back to a toroidal wrap and SAYS SO, because that changes the control's shape.
_wide = np.zeros((8, 8), dtype=bool)
_wide[0, :] = True
_wide[7, :] = True
_rw, _iw = plc.roll_mask(_wide, "q", "s")
assert _rw is not None and _rw.sum() == _wide.sum()
assert _iw["toroidal"] is True, _iw
print("[T1d] a grid-spanning union falls back to a toroidal wrap and reports it")

# ---------------------------------------------------------------------------
# Test 2: THE CONTRACT -- unscored on exactly the completions mean_in leaves unscored
# ---------------------------------------------------------------------------
orw.configure(metric="mean_in", mass_floor_tau=None, max_box_area=0.5, natural_only=False)
orw._CFG["max_union_area"] = None

# Five completions covering every way the real reward can decline to score one.
GROUNDABLE = "the red car"
UNGROUNDABLE = "nothing here"
HUGE = "the whole background"

sal = [
    [{"map": _map((1, 1)), "text": GROUNDABLE}, {"map": _map((2, 2)), "text": GROUNDABLE}],
    [],                                                            # no observe steps
    [{"map": _map((1, 1)), "text": UNGROUNDABLE}],                 # DINO grounds nothing
    [{"map": _map((1, 1)), "text": UNGROUNDABLE},
     {"map": _map((3, 3)), "text": GROUNDABLE}],                   # one of two grounds
    [{"map": _map((1, 1)), "text": HUGE}],                         # union too large
]
_QUADS = [[0.0, 0.0, 0.5, 0.5], [0.5, 0.0, 1.0, 0.5], [0.0, 0.5, 0.5, 1.0]]  # union 0.75


def _fake_dino(images, texts):
    out = []
    for t in texts:
        if t == UNGROUNDABLE:
            out.append([])
        elif t == HUGE:
            out.append(list(_QUADS))
        else:
            out.append([[0.15, 0.15, 0.45, 0.45]])
    return out


orw._dino_boxes = _fake_dino

_N = len(sal)
_kw = dict(
    completions=[[{"role": "assistant", "content": f"completion {i}"}] for i in range(_N)],
    saliency_map=sal,
    valid_list=[True] * _N,
    image=[_Img()] * _N,
    prompts=[f"question {i}" for i in range(_N)],
    completion_ids=[[7] * (50 + 30 * i) for i in range(_N)],
)


def _unscored(rewards):
    return tuple(r is None for r in rewards)


for _cap in (None, 0.5):                       # --max_union_area off, then biting
    orw.configure(max_union_area=_cap) if _cap else orw._CFG.update(max_union_area=None)
    ref = orw.think_overlap_reward(**_kw)
    for kind in plc.KINDS:
        plc.configure(kind=kind, seed=0)
        got = plc.think_placebo_reward(**_kw)
        assert len(got) == len(ref) == _N
        assert _unscored(got) == _unscored(ref), (
            f"--placebo {kind} with max_union_area={_cap}: unscored set "
            f"{_unscored(got)} != mean_in's {_unscored(ref)}"
        )
    print(f"[T2] max_union_area={_cap or 'off'}: all three placebos unscored exactly where "
          f"mean_in is {_unscored(ref)}")

orw._CFG["max_union_area"] = None

# Same, with --overlap_natural_only masking a row: the placebo reads the SAME switch, so
# a masked row is masked in both and never costs a Grounding-DINO call.
orw.configure(natural_only=True)
_nat = [True, True, True, False, True]
_seen = []


def _counting_dino(images, texts):
    _seen.append(len(texts))
    return _fake_dino(images, texts)


orw._dino_boxes = _counting_dino
ref = orw.think_overlap_reward(natural=_nat, **_kw)
_n_ref = _seen[-1]
for kind in plc.KINDS:
    plc.configure(kind=kind)
    got = plc.think_placebo_reward(natural=_nat, **_kw)
    assert _unscored(got) == _unscored(ref), (kind, _unscored(got), _unscored(ref))
    assert _seen[-1] == _n_ref, (kind, _seen[-1], _n_ref)
print(f"[T2b] --overlap_natural_only: same masked set {_unscored(ref)}, "
      f"same {_n_ref} grounding calls")
orw.configure(natural_only=False)
orw._dino_boxes = _fake_dino

# ---------------------------------------------------------------------------
# Test 3: the same contract under 300 randomised batches, all four metrics
# ---------------------------------------------------------------------------
# The fixture above covers the cases someone thought of. This covers the ones nobody did:
# random step counts, random grounding outcomes, random box sizes, random formats.
rng = np.random.default_rng(20260820)
_boxes_for = {}


def _random_dino(images, texts):
    return [_boxes_for[t] for t in texts]


orw._dino_boxes = _random_dino
_checked = 0
for _metric in ("mean_in", "mean_in_v2", "auroc"):
    orw.configure(metric=_metric)
    for trial in range(100):
        n = int(rng.integers(1, 7))
        smap, keys = [], []
        for c in range(n):
            steps = []
            for si in range(int(rng.integers(0, 4))):
                key = f"t{trial}-c{c}-s{si}"
                roll = rng.random()
                if roll < 0.3:
                    _boxes_for[key] = []                              # ungroundable
                elif roll < 0.45:
                    _boxes_for[key] = [[0.0, 0.0, 1.0, 1.0]]          # degenerate/capped
                else:
                    x0, y0 = rng.random(2) * 0.5
                    _boxes_for[key] = [[x0, y0, x0 + 0.3, y0 + 0.3]]
                peak = (int(rng.integers(0, 6)), int(rng.integers(0, 6)))
                # An all-zero map is the one thing mean_in_v2 rejects on map content:
                # keep some, so the gate is exercised rather than assumed away.
                m = np.zeros((6, 6), np.float32) if rng.random() < 0.1 else _map(peak)
                steps.append({"map": m, "text": key})
                keys.append(key)
            smap.append(steps)
        kw = dict(
            completions=[[{"role": "assistant", "content": f"c{trial}-{i}"}] for i in range(n)],
            saliency_map=smap,
            valid_list=[bool(rng.random() > 0.15) for _ in range(n)],
            image=[_Img()] * n,
            prompts=[f"q{trial}-{i}" for i in range(n)],
            completion_ids=[[7] * int(rng.integers(5, 900)) for _ in range(n)],
        )
        ref = orw.think_overlap_reward(**kw)
        for kind in plc.KINDS:
            plc.configure(kind=kind)
            got = plc.think_placebo_reward(**kw)
            assert _unscored(got) == _unscored(ref), (
                _metric, kind, trial, _unscored(got), _unscored(ref)
            )
        _checked += 1
print(f"[T3] {_checked} randomised batches x 3 metrics x 3 placebos: unscored sets identical")
orw.configure(metric="mean_in")
orw._dino_boxes = _fake_dino

# ---------------------------------------------------------------------------
# Test 4: the placebo VALUES are what they claim, on a scored batch
# ---------------------------------------------------------------------------
plc.configure(kind="length", length_anchor=1024.0)
r_len = plc.think_placebo_reward(**_kw)
# completions 0 and 3 are the scored ones; ids are 50 + 30*i tokens long
assert abs(r_len[0] - 0.974) < 1e-12 and abs(r_len[3] - 0.884) < 1e-12, r_len
assert r_len[3] < r_len[0], "the longer completion must score lower"
assert all(v > 0 for v in r_len if v is not None), "must stay above the unscored-as-0 read"
print(f"[T4a] length: {[None if v is None else round(v, 3) for v in r_len]} "
      f"((1024 - n_tokens)/1000, per completion)")

plc.configure(kind="random")
r_rand = plc.think_placebo_reward(**_kw)
assert r_rand[0] != r_rand[3], "per COMPLETION -- a per-prompt value would give zero advantage"
assert r_rand == plc.think_placebo_reward(**_kw), "must be deterministic"
assert all(0.0 <= v < 1.0 for v in r_rand if v is not None)
print(f"[T4b] random: {[None if v is None else round(v, 3) for v in r_rand]} "
      f"(differs within the group, stable across calls)")

# roll needs its own fixture: a map whose mass sits exactly on the box, so a moved union
# is strictly worse. (The shared fixture above is a single peak on a flat floor, and any
# equal-area mask that still covers the peak scores identically -- which is a fine
# property for the parity tests and useless for this one.)
_grounded_map = np.full((6, 6), 0.001, dtype=np.float32)
_grounded_map[0:3, 0:3] = 1.0          # exactly the cells [0.15,0.15,0.45,0.45] rasterises to
_kw_roll = dict(
    completions=[[{"role": "assistant", "content": "grounded"}]],
    saliency_map=[[{"map": _grounded_map, "text": GROUNDABLE}]],
    valid_list=[True], image=[_Img()], prompts=["where is the car?"],
    completion_ids=[[7] * 120],
)
plc.configure(kind="roll")
plc.pop_diagnostics()                  # drain what the randomised batches above recorded
r_roll = plc.think_placebo_reward(**_kw_roll)
r_real = orw.think_overlap_reward(**_kw_roll)
assert r_roll == plc.think_placebo_reward(**_kw_roll), "the roll must be fixed, not redrawn"
assert abs(r_real[0] - 1.0) < 1e-6, r_real          # perfectly grounded on the true union
assert r_roll[0] < r_real[0], (r_roll[0], r_real[0])
_diag = plc.pop_diagnostics()
assert _diag["roll_dist"] > 0 and not np.isnan(_diag["union_frac"]), _diag
print(f"[T4c] roll: a perfectly grounded map scores {r_real[0]:.4f} on its own union and "
      f"{r_roll[0]:.4f} on the moved one; diagnostics "
      f"{({k: round(v, 3) for k, v in _diag.items()})}")

# random/length record no roll diagnostics, but pop_diagnostics still returns the FIXED
# key set -- the trainer gathers it, so a rank-dependent key set would hang.
plc.configure(kind="length")
plc.think_placebo_reward(**_kw)
_d = plc.pop_diagnostics()
assert tuple(_d.keys()) == plc.DIAG_KEYS and all(np.isnan(v) for v in _d.values()), _d
print(f"[T4d] pop_diagnostics returns the fixed key set {plc.DIAG_KEYS}, all NaN for length")

# ---------------------------------------------------------------------------
# Test 5: refusals
# ---------------------------------------------------------------------------
try:
    plc.configure(kind="brevity")
except ValueError as e:
    assert "roll|random|length" in str(e)
    print("[T5a] an unknown --placebo is refused")
else:
    raise AssertionError("expected ValueError for an unknown placebo kind")

plc._CFG["kind"] = None
orw.configure(metric="logratio")
try:
    plc.configure(kind="length")
except ValueError as e:
    assert "logratio" in str(e)
    print("[T5b] --placebo with --overlap_metric logratio is refused (the roll-null is "
          "already a rolled control, and its scorer would consume its own draw)")
else:
    raise AssertionError("expected ValueError for --placebo with logratio")
orw.configure(metric="mean_in")

plc._CFG["kind"] = None
try:
    plc.think_placebo_reward(**_kw)
except ValueError as e:
    assert "expected one of" in str(e)
    print("[T5c] calling the reward with no --placebo set is refused, not silently scored")
else:
    raise AssertionError("expected ValueError when no placebo kind is configured")

plc.configure(kind="length")
try:
    plc.think_placebo_reward(**{**_kw, "completion_ids": None})
except KeyError as e:
    assert "completion_ids" in str(e)
    print("[T5d] --placebo length without completion token ids raises instead of guessing")
else:
    raise AssertionError("expected KeyError when completion_ids is missing")

# ---------------------------------------------------------------------------
# Test 6: the real reward is untouched by any of this
# ---------------------------------------------------------------------------
_fresh_orw, _ = _load_tree(f"{_PKG}_fresh")
assert _fresh_orw._CFG["metric"] == "mean_in"
assert _fresh_orw._CFG["mass_floor_tau"] is None
assert _fresh_orw._CFG["natural_only"] is False
_, _fresh_plc = _load_tree(f"{_PKG}_fresh2")
assert _fresh_plc._CFG["kind"] is None, _fresh_plc._CFG
assert _fresh_plc.is_active() is False
print("[T6] defaults unchanged: overlap reward is still mean_in, placebo is off")

print("\nAll placebo CPU tests passed.")
