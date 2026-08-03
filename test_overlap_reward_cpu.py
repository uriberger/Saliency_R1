"""CPU sanity tests for the attention-overlap reward port. No 8B model / no GPU."""
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent

# Which copy of the sources to test. Default: trl_repo/, the tree that actually runs
# (what patch_trl_qwen3.sh installs there). OVERLAP_TEST_TREE=local tests this checkout's
# tracked sources instead -- the only option from a git worktree, where copying into the
# shared trl_repo/ would hit every other session and every running job.
_LOCAL = os.environ.get("OVERLAP_TEST_TREE") == "local"
REWARDS_SRC = "trl/rewards/overlap_rewards.py" if _LOCAL else "trl_repo/trl/rewards/overlap_rewards.py"
STEPS_SRC = "trl/overlap_steps.py" if _LOCAL else "trl_repo/trl/trainer/overlap_steps.py"


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

orw = _load("overlap_rewards", REWARDS_SRC)
ost = _load("overlap_steps", STEPS_SRC)
print(f"[tree] testing {'this checkout' if _LOCAL else 'trl_repo/ (the running copy)'}")

# ---------------------------------------------------------------------------
# Test 1: mean_in metric matches offline _score_saliency_flat (max-norm then mean-in)
# ---------------------------------------------------------------------------
gh, gw = 4, 4
m = np.zeros((gh, gw), dtype=np.float32)
m[1, 1] = 2.0    # inside box
m[1, 2] = 1.0    # inside box
m[3, 3] = 4.0    # outside box (this is the max)
mask = np.zeros((gh, gw), dtype=bool)
mask[1, 1] = mask[1, 2] = True
# offline reference: normalize by global max (4.0), mean inside = (2/4 + 1/4)/2 = 0.375
expected = ((2.0 / 4.0) + (1.0 / 4.0)) / 2
got = orw._mean_in(m, mask)
assert abs(got - expected) < 1e-6, (got, expected)
print(f"[T1] mean_in max-norm metric OK: {got:.4f} == {expected:.4f}")

# ---------------------------------------------------------------------------
# Test 2: union mask + area filter
# ---------------------------------------------------------------------------
orw.configure(box_threshold=0.10, max_box_area=0.5)
# one small box (area 0.25) kept, one full-frame box (area 1.0) dropped
boxes = [[0.25, 0.25, 0.75, 0.75], [0.0, 0.0, 1.0, 1.0]]
um = orw._union_mask(boxes, 8, 8)
assert um is not None and 0 < um.sum() < 64, um.sum()
# a box list of only the full-frame box -> dropped -> degenerate -> None
assert orw._union_mask([[0.0, 0.0, 1.0, 1.0]], 8, 8) is None
print(f"[T2] union mask + area filter OK: n_in={int(um.sum())}")

# ---------------------------------------------------------------------------
# Test 3: think_overlap_reward end-to-end with mocked DINO
# ---------------------------------------------------------------------------
class _Img:
    size = (64, 64)

# completion 0: 2 observe steps, both groundable; completion 1: no steps; completion 2: step ungroundable
map_a = np.zeros((4, 4), np.float32); map_a[1, 1] = 3.0; map_a[0, 0] = 1.0
map_b = np.zeros((4, 4), np.float32); map_b[2, 2] = 2.0; map_b[3, 3] = 5.0
sal = [
    [{"map": map_a, "text": "the red car"}, {"map": map_b, "text": "a stop sign"}],
    [],
    [{"map": map_a, "text": "nothing here"}],
]

def _fake_dino(images, texts):
    out = []
    for t in texts:
        if t == "nothing here":
            out.append([])                       # ungroundable -> skip
        else:
            out.append([[0.25, 0.25, 0.5, 0.5]])  # covers grid cell (1,1)
    return out

orw._dino_boxes = _fake_dino
rewards = orw.think_overlap_reward(
    completions=[None, None, None],
    saliency_map=sal,
    valid_list=[True, True, True],
    image=[_Img(), _Img(), _Img()],
)
# comp0: step a mean_in over cell(1,1): map/max(=3) inside {(1,1)} = 3/3 =1.0; step b: cell(1,1)=0 -> /max(5)=0 -> 0.0; mean=0.5
# comp1: no steps -> None ; comp2: ungroundable -> None
assert abs(rewards[0] - 0.5) < 1e-6, rewards
assert rewards[1] is None and rewards[2] is None, rewards
print(f"[T3] think_overlap_reward OK: {rewards}")

# format gate: invalid format -> 0
rewards2 = orw.think_overlap_reward(
    completions=[None], saliency_map=[sal[0]], valid_list=[False], image=[_Img()],
)
assert rewards2[0] == 0.0, rewards2
print(f"[T3b] format gate OK: {rewards2}")

# ---------------------------------------------------------------------------
# Test 4: sentence splitter spans reconstruct
# ---------------------------------------------------------------------------
txt = "Looking at the image. I see a red car on the left. Therefore the answer is A."
spans = ost.split_sentences_with_spans(txt, base_offset=0)
for s, cs, ce in spans:
    assert txt[cs:ce] == s, (txt[cs:ce], s)
print(f"[T4] sentence spans OK: {len(spans)} sentences, offsets reconstruct")

# ---------------------------------------------------------------------------
# Test 5: the auroc metric (--overlap_metric auroc)
# ---------------------------------------------------------------------------
orw.configure(metric="mean_in", mass_floor_tau=None)   # reset to defaults

# 5a. perfectly separated: every in-box patch above every out-box patch -> 1.0
m5 = np.array([[9.0, 8.0], [1.0, 2.0]], dtype=np.float32)
mk5 = np.array([[True, True], [False, False]])
assert abs(orw._auroc(m5, mk5) - 1.0) < 1e-9, orw._auroc(m5, mk5)
# reversed -> 0.0
assert abs(orw._auroc(m5, ~mk5) - 0.0) < 1e-9
# a completely flat map is all ties -> average ranks give exactly chance
assert abs(orw._auroc(np.ones((2, 2), np.float32), mk5) - 0.5) < 1e-9
print("[T5a] auroc endpoints + tie handling OK (1.0 / 0.0 / 0.5)")

# 5b. THE point of the metric: invariant to any monotone reshaping. This is the
#     exact transform the wov0.4 run exploited (mean_in moves 32x under it).
rng = np.random.default_rng(0)
m5b = rng.random((8, 8)).astype(np.float32) * 1e-2
mk5b = np.zeros((8, 8), dtype=bool)
mk5b[2:6, 2:6] = True
a_base = orw._auroc(m5b, mk5b)
for gamma in (0.25, 0.5, 2.0):
    assert abs(orw._auroc(m5b ** gamma, mk5b) - a_base) < 1e-9, gamma
assert abs(orw._auroc(m5b + 0.5 * m5b.mean(), mk5b) - a_base) < 1e-9   # uniform floor
assert abs(orw._auroc(0.5 * m5b, mk5b) - a_base) < 1e-9               # rescale
# ... whereas mean_in moves a long way under the same flattening
mi_base, mi_flat = orw._mean_in(m5b, mk5b), orw._mean_in(m5b ** 0.25, mk5b)
assert mi_flat - mi_base > 0.1, (mi_base, mi_flat)
print(f"[T5b] auroc invariant to flatten/sharpen/floor/rescale (all {a_base:.4f}); "
      f"mean_in moves {mi_base:.3f} -> {mi_flat:.3f} under the same flattening")

# 5c. metric flag actually switches what think_overlap_reward returns
orw.configure(metric="auroc")
r_auroc = orw.think_overlap_reward(
    completions=[None], saliency_map=[sal[0]], valid_list=[True], image=[_Img()],
)
orw.configure(metric="mean_in")
r_mean = orw.think_overlap_reward(
    completions=[None], saliency_map=[sal[0]], valid_list=[True], image=[_Img()],
)
assert abs(r_mean[0] - 0.5) < 1e-6, r_mean          # unchanged from T3
assert r_auroc[0] != r_mean[0], (r_auroc, r_mean)
print(f"[T5c] --overlap_metric switches the reward: auroc={r_auroc[0]:.4f} mean_in={r_mean[0]:.4f}")

# ---------------------------------------------------------------------------
# Test 5d-f: the mean_in_v2 metric (--overlap_metric mean_in_v2), mean-in / mean-all
# ---------------------------------------------------------------------------
orw.configure(metric="mean_in", mass_floor_tau=None)

# 5d. definition + endpoints. m: inside {(1,1),(1,2)} = 2 and 1, outside 4 at (3,3).
#     mean over the whole 4x4 map = 7/16; mean inside = 1.5 -> 1.5 / (7/16)
v2 = orw._mean_in_v2(m, mask)
assert abs(v2 - 1.5 / (7.0 / 16.0)) < 1e-6, v2
# a flat map has no preference for the box -> exactly chance (1.0), whatever the mask
assert abs(orw._mean_in_v2(np.ones((4, 4), np.float32), mask) - 1.0) < 1e-9
# all the mass inside the box -> the ceiling, n_patches / n_in
only_in = np.zeros((4, 4), np.float32); only_in[mask] = 3.0
assert abs(orw._mean_in_v2(only_in, mask) - 16.0 / 2) < 1e-6, orw._mean_in_v2(only_in, mask)
# an all-zero map has no defined ratio -> None (the step is skipped, not scored 0)
assert orw._mean_in_v2(np.zeros((4, 4), np.float32), mask) is None
print(f"[T5d] mean_in_v2 OK: {v2:.4f}; flat -> 1.0 (chance), all-in -> {16.0 / 2:.1f} (ceiling)")

# 5e. rescale-invariant like auroc, and flattening moves it TOWARD chance rather than
#     up -- the opposite direction to mean_in, which is the point of the metric.
b2 = orw._mean_in_v2(m5b, mk5b)
assert abs(orw._mean_in_v2(7.3 * m5b, mk5b) - b2) < 1e-9          # rescale cancels
flat2 = orw._mean_in_v2(m5b + 0.5 * m5b.mean(), mk5b)             # uniform floor
assert abs(flat2 - 1.0) < abs(b2 - 1.0), (b2, flat2)
print(f"[T5e] mean_in_v2 rescale-invariant; uniform floor moves it {b2:.4f} -> {flat2:.4f} "
      f"toward chance (mean_in moved {mi_base:.3f} -> {mi_flat:.3f} away from 0)")

# 5f. the flag switches think_overlap_reward. comp0: step a = 3.0 / (4/16) = 12,
#     step b = 0 (cell (1,1) is empty) -> mean 6.0
orw.configure(metric="mean_in_v2")
r_v2 = orw.think_overlap_reward(
    completions=[None], saliency_map=[sal[0]], valid_list=[True], image=[_Img()],
)
assert abs(r_v2[0] - 6.0) < 1e-6, r_v2
orw.configure(metric="mean_in")
print(f"[T5f] --overlap_metric mean_in_v2 switches the reward: {r_v2[0]:.4f} (mean_in {r_mean[0]:.4f})")

# ---------------------------------------------------------------------------
# Test 6: the image-mass floor (--mass_floor_tau), and that it is OFF by default
# ---------------------------------------------------------------------------
orw.configure(metric="auroc", mass_floor_tau=None)
assert orw._mass_gate(m5b) == 1.0, "floor must be a no-op when tau is unset"
# tau above the map's mass -> proportional penalty; tau below -> no penalty
total = float(m5b.sum())
orw._CFG["mass_floor_tau"] = total * 2
assert abs(orw._mass_gate(m5b) - 0.5) < 1e-9, orw._mass_gate(m5b)
orw._CFG["mass_floor_tau"] = total * 0.5
assert orw._mass_gate(m5b) == 1.0
# withdrawing attention from the image must COST reward once the floor bites
orw._CFG["mass_floor_tau"] = total
assert abs(orw._mass_gate(0.5 * m5b) - 0.5) < 1e-9
print("[T6] mass floor OK: off by default, min(1, mass/tau) once set, penalises withdrawal")

# ---------------------------------------------------------------------------
# Test 7: DEFAULTS ARE UNCHANGED -- a fresh import must reproduce the incumbent
# ---------------------------------------------------------------------------
orw_fresh = _load("overlap_rewards_fresh", REWARDS_SRC)
assert orw_fresh._CFG["metric"] == "mean_in", orw_fresh._CFG
assert orw_fresh._CFG["mass_floor_tau"] is None, orw_fresh._CFG
assert orw_fresh._CFG["natural_only"] is False, orw_fresh._CFG
assert orw_fresh._step_score(m5b, mk5b) == orw_fresh._mean_in(m5b, mk5b)
print("[T7] defaults unchanged: metric=mean_in, no mass floor, natural_only off, _step_score == _mean_in")

# ---------------------------------------------------------------------------
# Test 8: natural-images-only gating (--overlap_natural_only)
# ---------------------------------------------------------------------------
orw.configure(metric="mean_in", mass_floor_tau=None)
_two = dict(completions=[None, None], saliency_map=[sal[0], sal[0]],
            valid_list=[True, True], image=[_Img(), _Img()])

# 8a. off (the default): the `natural` column is ignored, both rows are scored
r_off = orw.think_overlap_reward(natural=[True, False], **_two)
assert abs(r_off[0] - 0.5) < 1e-6 and abs(r_off[1] - 0.5) < 1e-6, r_off
print(f"[T8a] switch off -> natural column ignored, both rows scored: {r_off}")

# 8b. on: the non-natural row is MASKED (None, not 0.0), the natural row is untouched,
#     and the masked row must not cost a Grounding-DINO call
_grounded = []
def _counting_dino(images, texts):
    _grounded.extend(texts)
    return _fake_dino(images, texts)
orw._dino_boxes = _counting_dino
orw.configure(natural_only=True)
r_on = orw.think_overlap_reward(natural=[True, False], **_two)
assert abs(r_on[0] - 0.5) < 1e-6, r_on
assert r_on[1] is None, r_on          # masked (NaN -> neutral), NOT scored 0.0
assert len(_grounded) == 2, _grounded  # only completion 0's two observe steps
print(f"[T8b] switch on -> non-natural row masked and never grounded: {r_on}, dino calls={len(_grounded)}")

# 8c. no `natural` column while the switch is on must fail loudly. Silently masking
#     every row would turn the run into --reward_variant none without saying so.
try:
    orw.think_overlap_reward(natural=None, **_two)
except KeyError as e:
    assert "natural" in str(e)
    print("[T8c] missing natural column raises instead of silently masking everything")
else:
    raise AssertionError("expected KeyError when natural_only=True and natural is None")

orw.configure(natural_only=False)
orw._dino_boxes = _fake_dino

print("\nAll CPU logic tests passed.")
