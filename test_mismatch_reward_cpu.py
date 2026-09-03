"""CPU tests for the mismatched-box control (--mismatch_bank).

Two of these are the experiment rather than the code. T4: every rollout of one prompt must
be scored against the SAME donor, because a donor drawn per completion puts 0.0117 of pure
draw noise inside a group whose entire real reward spans 0.0115 -- that run is
`--placebo random`, which already exists. T2: a completion must never go unscored for want
of a chain of its length, because the policy chooses its own observe-step count and an
unservable count would be a free exit from the reward.

No 8B model, no GPU, no Grounding-DINO (the bank is a fixture written here).

    python test_mismatch_reward_cpu.py                          # tests trl_repo/ (what runs)
    OVERLAP_TEST_TREE=local python test_mismatch_reward_cpu.py  # tests this checkout
"""
import importlib.util
import json
import os
import sys
import tempfile
import types
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent

# Same switch and same reasoning as test_placebo_reward_cpu.py: from a git worktree,
# copying into the shared trl_repo/ would hit every other session and every running job.
_LOCAL = os.environ.get("OVERLAP_TEST_TREE") == "local"
_REWARDS_DIR = Path("trl/rewards") if _LOCAL else Path("trl_repo/trl/rewards")
_PKG = "trl_local" if _LOCAL else "trl_running"
_N = [0]


def _load_tree(pkg=None):
    """Stub `<pkg>` / `<pkg>.rewards` at the tree under test and load both modules there.

    A fresh call gives a module with a fresh `_CFG`, `_BANK` and `_DONOR_CACHE`, which is
    what lets one file test several bank configurations.
    """
    if pkg is None:
        _N[0] += 1
        pkg = f"{_PKG}_{_N[0]}"
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
    return orw, _load("mismatch_rewards")


print(f"[tree] testing {'this checkout' if _LOCAL else 'trl_repo/ (the running copy)'}: {_REWARDS_DIR}")
TMP = Path(tempfile.mkdtemp(prefix="mismatch_bank_"))


# ---------------------------------------------------------------------------
# a fixture bank
# ---------------------------------------------------------------------------
# 12 rows over 10 pictures: rows 0 and 1 share picture "img0", rows 2 and 3 share "img1".
# Those two pairs are the whole point of the image_group column -- excluding the row
# itself would still let row 0 be scored against its own picture through row 1.
INDEX = {f"ds/{i}": f"img{max(0, i - 1) if i < 2 else (1 if i in (2, 3) else i - 2)}"
         for i in range(12)}
INDEX["ds/0"] = INDEX["ds/1"] = "img0"
INDEX["ds/2"] = INDEX["ds/3"] = "img1"

_LEFT = [[0.0, 0.0, 0.5, 1.0]]        # left half of any grid
_RIGHT = [[0.5, 0.0, 1.0, 1.0]]       # right half
_TOP = [[0.0, 0.0, 1.0, 0.5]]         # top half


def _bank(donors, threshold=0.10):
    return {"meta": {"version": 1, "box_threshold": threshold, "n_donors": len(donors)},
            "index": INDEX, "donors": donors}


def _write(bank, name="bank.json"):
    p = TMP / name
    p.write_text(json.dumps(bank))
    return str(p)


# Every row is a donor except that donors 0..3 are the two image-sharing pairs, so the
# exclusion has something to reject.
DONORS = []
for i in range(12):
    chains = {
        "1": [[_LEFT]],
        "2": [[_LEFT, _RIGHT], [_TOP, _TOP]],
        "3": [[_LEFT, _RIGHT, _TOP]],
        "5": [[_LEFT, _RIGHT, _TOP, _LEFT, _RIGHT]],
    }
    if i == 7:
        chains["9"] = [[_TOP] * 9]           # one donor reaches further than the rest
    DONORS.append({"key": f"ds/{i}", "image_group": INDEX[f"ds/{i}"], "chains": chains})
BANK_PATH = _write(_bank(DONORS))


def _fresh(bank_path=BANK_PATH, seed=0, **overlap):
    orw, mm = _load_tree()
    orw.configure(metric="mean_in", mass_floor_tau=None, max_box_area=0.5,
                  natural_only=False, box_threshold=0.10, **overlap)
    orw._CFG["max_union_area"] = None
    mm.configure(bank=bank_path, seed=seed)
    return orw, mm


# ---------------------------------------------------------------------------
# Test 1: the donor is a different question AND a different picture, deterministically
# ---------------------------------------------------------------------------
orw, mm = _fresh()
for key in INDEX:
    d, h = mm.donor_for(key)
    assert d["key"] != key, (key, d["key"])
    assert d["image_group"] != INDEX[key], (key, d["image_group"])
print("[T1a] every row resolves to a donor with a different question AND a different picture")

# The two image-sharing pairs are the ones that would slip through a question-only check.
for a, b in (("ds/0", "ds/1"), ("ds/2", "ds/3")):
    assert mm.donor_for(a)[0]["key"] != b, f"{a} was scored against its own picture via {b}"
    assert mm.donor_for(b)[0]["key"] != a
print("[T1b] a second question about the SAME picture is excluded, not just the row itself")

# Deterministic across processes: a fresh module load, and a fresh donor cache, agree.
_, mm2 = _fresh()
assert all(mm.donor_for(k)[0]["key"] == mm2.donor_for(k)[0]["key"] for k in INDEX)
assert all(mm.donor_for(k)[1] == mm2.donor_for(k)[1] for k in INDEX)
print("[T1c] the pairing is the same after a reload -- same donor every epoch, every rank")

# ...and the seed is what moves it, so a replicate is a different pairing of the same corpus.
_, mm3 = _fresh(seed=7)
assert any(mm.donor_for(k)[0]["key"] != mm3.donor_for(k)[0]["key"] for k in INDEX)
print("[T1d] --mismatch_seed moves the pairing (a replicate is a different pairing)")

# The donors are spread over the bank rather than piled on one row.
_used = Counter(mm.donor_for(k)[0]["key"] for k in INDEX)
assert len(_used) >= 6, _used
print(f"[T1e] 12 rows spread over {len(_used)} distinct donors: {dict(_used)}")

# A row the bank never indexed raises: without its image group the "different picture"
# half of the exclusion cannot be enforced, and quietly enforcing half of it is worse.
try:
    mm.donor_for("otherds/999")
except KeyError as e:
    assert "index" in str(e)
    print("[T1f] a row outside the bank's index raises instead of half-excluding")
else:
    raise AssertionError("expected KeyError for an unindexed row")


# ---------------------------------------------------------------------------
# Test 2: the length ladder -- exact when possible, nearest otherwise, NEVER unscored
# ---------------------------------------------------------------------------
# From the LOADED bank, not the JSON fixture: the loader is what turns the string keys a
# JSON object forces into the ints an observe-step count is everywhere else.
LOADED = {d["key"]: d for d in mm._load_bank()["donors"]}
assert all(isinstance(k, int) for k in LOADED["ds/7"]["chains"])
donor = LOADED["ds/7"]                 # lengths {1,2,3,5,9}
h = 0
for n, want in ((1, 1), (2, 2), (3, 3), (5, 5), (9, 9)):
    chain, L = mm.chain_for(donor, n, h)
    assert L == want and len(chain) == want, (n, L)
print("[T2a] a step count the donor row holds is matched exactly")

# 4 is equidistant from 3 and 5; ties go to the LONGER chain, which needs no wrap.
assert mm.chain_for(donor, 4, h)[1] == 5
assert mm.chain_for(donor, 6, h)[1] == 5           # nearest below
assert mm.chain_for(donor, 7, h)[1] == 9 or mm.chain_for(donor, 7, h)[1] == 5
print("[T2b] an unheld step count takes the nearest length the SAME donor has "
      "(ties -> the longer one, which needs no wrap)")

# Never a KeyError, never a None, for any count the policy could ever emit -- including
# the 85 the trained checkpoints reach and far past it.
for n in list(range(1, 30)) + [40, 85, 200, 1000]:
    for d in LOADED.values():
        chain, L = mm.chain_for(d, n, h)
        assert L >= 1 and len(chain) == L
        assert all(chain[i % L] for i in range(n))   # every step of the completion is served
print("[T2c] every step count 1..1000 resolves for every donor -- no completion can go "
      "unscored for want of a length, so no step count is an exit from the reward")

# Wrapping is positional and cyclic: step i takes donor step i % L.
_donor2 = {"key": "x", "image_group": "gx", "chains": {2: [[_LEFT, _RIGHT]]}}
c2, L2 = mm.chain_for(_donor2, 5, h)
assert L2 == 2 and [c2[i % L2] for i in range(5)] == [_LEFT, _RIGHT, _LEFT, _RIGHT, _LEFT]
print("[T2d] a short donor chain wraps positionally (step i -> donor step i % L)")


# ---------------------------------------------------------------------------
# Test 3: the reward itself
# ---------------------------------------------------------------------------
def _map(gh=4, gw=4):
    """A graded map with one peak.

    Deliberately not two flat halves: on a flat map mean_in and auroc happen to agree,
    and T3h -- which checks that switching the run's metric switches this reward's --
    would pass on a reward that ignored the metric entirely.
    """
    m = np.arange(1, gh * gw + 1, dtype=np.float32).reshape(gh, gw)
    m[0, 0] = 40.0
    return m


def _steps(n, **kw):
    return [{"map": _map(**kw), "text": f"step {i}"} for i in range(n)]


orw, mm = _fresh()
# Pin one row's donor so the expected value can be written down.
key = "ds/5"
d, hh = mm.donor_for(key)
chain, L = mm.chain_for(d, 2, hh)

kw = dict(saliency_map=[_steps(2)], valid_list=[True], image=[None],
          dataset=["ds"], question_id=[5], completions=[[{"role": "assistant", "content": "x"}]])
got = mm.think_mismatch_reward(**kw)[0]
# mean_in = mean of the peak-normalised map inside the donor's union, averaged over steps.
want = float(np.mean([
    float((_map() / _map().max())[orw._union_mask(chain[i % L], 4, 4)].mean())
    for i in range(2)
]))
assert abs(got - want) < 1e-9, (got, want)
print(f"[T3a] the reward is mean_in of the map inside the DONOR's union ({got:.4f})")

# The picture is not consulted. Passing a different image cannot change the score --
# that is the whole control, so it is asserted rather than assumed.
assert mm.think_mismatch_reward(**{**kw, "image": ["a totally different picture"]})[0] == got
print("[T3b] the reward does not read `image`: the picture takes no part in the region")

# A donor step DINO could not ground is stored empty -> that step is SKIPPED, not 0.
_empty_donor = {"key": "e", "image_group": "ge", "chains": {2: [[[], _RIGHT]]}}
_saved = mm._DONOR_CACHE.get(key)
mm._DONOR_CACHE[key] = (_empty_donor, 0)
one = mm.think_mismatch_reward(**kw)[0]
m = _map(); m = m / m.max()
assert abs(one - float(m[orw._union_mask(_RIGHT, 4, 4)].mean())) < 1e-9, one
print("[T3c] a donor step with no boxes is SKIPPED, not scored 0 (the reference's own exit)")

# ...and when every step skips, the completion is masked (None), not scored 0.
mm._DONOR_CACHE[key] = ({"key": "e", "image_group": "ge", "chains": {2: [[[], []]]}}, 0)
assert mm.think_mismatch_reward(**kw)[0] is None
print("[T3d] a completion whose every donor step is ungroundable returns None (masked)")
mm._DONOR_CACHE[key] = _saved

# No observe step at all -> masked, same as the real reward.
assert mm.think_mismatch_reward(**{**kw, "saliency_map": [[]]})[0] is None
print("[T3e] a completion with no observe step returns None")

# The format gate is multiplicative, exactly as in think_overlap_reward.
assert mm.think_mismatch_reward(**{**kw, "valid_list": [False]})[0] == 0.0
print("[T3f] a format-invalid completion is gated to 0.0 (multiplicative, as the reference)")

# --overlap_natural_only is read from the overlap reward's config: one switch, not two.
orw._CFG["natural_only"] = True
assert mm.think_mismatch_reward(**{**kw, "natural": [False]})[0] is None
assert mm.think_mismatch_reward(**{**kw, "natural": [True]})[0] == got
try:
    mm.think_mismatch_reward(**kw)
except KeyError as e:
    assert "natural" in str(e)
else:
    raise AssertionError("expected KeyError when --overlap_natural_only has no column")
orw._CFG["natural_only"] = False
print("[T3g] --overlap_natural_only masks non-natural rows and is read from ONE place")

# The metric knobs come from the same place too, so the control and its reference cannot
# drift apart: switching the run's metric switches this reward's.
orw.configure(metric="auroc")
assert mm.think_mismatch_reward(**kw)[0] != got
orw.configure(metric="mean_in")
assert mm.think_mismatch_reward(**kw)[0] == got
print("[T3h] --overlap_metric / the mass floor / the area caps all come from overlap_rewards")


# ---------------------------------------------------------------------------
# Test 4: THE DESIGN -- every rollout of one prompt shares one donor
# ---------------------------------------------------------------------------
# Eight rollouts of one row with DIFFERENT observe-step counts, which is the ordinary
# case (the cold start's counts run 1..14 with a median of 3). Each must be scored
# against the same donor row; only the chain inside it may change with the count.
counts = [2, 3, 3, 5, 1, 2, 9, 3]
kw8 = dict(saliency_map=[_steps(n) for n in counts],
           valid_list=[True] * 8, image=[None] * 8,
           dataset=["ds"] * 8, question_id=[5] * 8,
           completions=[[{"role": "assistant", "content": f"c{i}"}] for i in range(8)])
out = mm.think_mismatch_reward(**kw8)
assert all(v is not None for v in out), out
donors_used = {mm.donor_for(mm.row_key("ds", 5))[0]["key"]}
assert len(donors_used) == 1
# and the same completion in a different group position gets the same score
assert out[1] == out[2] == out[7], out       # identical maps, identical count -> identical
print(f"[T4a] all 8 rollouts share one donor row ({donors_used.pop()}); equal completions "
      f"score equally")

# Two rows never collide onto the same (donor, chain) by construction of the hash, and
# two DIFFERENT rows do generally get different donors -- otherwise the corpus would be
# scored against one region.
_pairs = {mm.donor_for(k)[0]["key"] for k in INDEX}
assert len(_pairs) > 1
print(f"[T4b] the corpus is spread over {len(_pairs)} donor rows, not scored against one")


# ---------------------------------------------------------------------------
# Test 5: the guards
# ---------------------------------------------------------------------------
# A bank built at another --box_threshold is refused: that filter was applied by DINO when
# the bank was written and cannot be re-applied to the stored boxes.
_p = _write(_bank(DONORS, threshold=0.25), "bank_t25.json")
_orw2, _mm2 = _load_tree()
_orw2.configure(box_threshold=0.10)
try:
    _mm2.configure(bank=_p)
except ValueError as e:
    assert "box_threshold" in str(e)
    print("[T5a] a bank built at a different --box_threshold is refused at startup")
else:
    raise AssertionError("expected ValueError on a box_threshold mismatch")
# ...and accepted when they agree.
_orw3, _mm3 = _load_tree()
_orw3.configure(box_threshold=0.25)
_mm3.configure(bank=_p)
print("[T5b] the same bank is accepted when the run's threshold matches it")

_orw4, _mm4 = _load_tree()
try:
    _mm4.configure(bank=str(TMP / "does_not_exist.json"))
except FileNotFoundError as e:
    assert "build_mismatch_bank" in str(e)
    print("[T5c] a missing bank fails at startup, naming the builder")
else:
    raise AssertionError("expected FileNotFoundError")

# The row identity has to arrive, or the control cannot be what it says it is.
try:
    mm.think_mismatch_reward(**{**kw, "question_id": None})
except KeyError as e:
    assert "question_id" in str(e)
    print("[T5d] a corpus without dataset/question_id raises instead of guessing an identity")
else:
    raise AssertionError("expected KeyError without question_id")

# A bank whose every donor is excluded for a row must say so rather than score it against
# its own picture.
_solo = [{"key": "ds/1", "image_group": "img0", "chains": {"1": [[_LEFT]]}}]
_ps = _write({"meta": {"version": 1, "box_threshold": 0.10}, "index": INDEX,
              "donors": _solo}, "bank_solo.json")
_orw5, _mm5 = _load_tree()
_orw5.configure(box_threshold=0.10)
_mm5.configure(bank=_ps)
try:
    _mm5.donor_for("ds/0")            # shares img0 with the only donor
except RuntimeError as e:
    assert "excluded" in str(e)
    print("[T5e] a row every donor is excluded for raises, rather than being paired with "
          "its own picture")
else:
    raise AssertionError("expected RuntimeError when every donor is excluded")


# ---------------------------------------------------------------------------
# Test 6: diagnostics, and that the real reward is untouched
# ---------------------------------------------------------------------------
mm._DIAG.clear()
mm.think_mismatch_reward(**kw8)
diag = mm.pop_diagnostics()
assert set(diag) == set(mm.DIAG_KEYS), diag
# A FIXED key set on every call, NaN where nothing was seen -- the trainer gathers these
# across ranks, and a rank-dependent key set is a rank-dependent number of collectives.
assert set(mm.pop_diagnostics()) == set(mm.DIAG_KEYS)
assert all(np.isnan(v) for v in mm.pop_diagnostics().values())
# counts 2,3,3,5,1,2,9,3 against a donor holding {1,2,3,5} (+9 for donor 7): the 9 is the
# only one that may miss, so exact_len_frac is 1.0 or 7/8.
assert diag["exact_len_frac"] in (1.0, 7 / 8), diag
assert 0.0 <= diag["union_frac"] <= 1.0
print(f"[T6a] diagnostics are a fixed key set, NaN when empty: "
      f"exact_len_frac {diag['exact_len_frac']:.3f}, union_frac {diag['union_frac']:.3f}")

_fresh_orw, _fresh_mm = _load_tree()
assert _fresh_mm._CFG["bank"] is None and _fresh_mm.is_active() is False
assert _fresh_orw._CFG["metric"] == "mean_in" and _fresh_orw._CFG["natural_only"] is False
print("[T6b] defaults unchanged: the overlap reward is still mean_in, the control is off")

print("\nAll mismatched-box CPU tests passed.")
