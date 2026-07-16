"""CPU sanity tests for the attention-overlap reward port. No 8B model / no GPU."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

orw = _load("overlap_rewards", "trl_repo/trl/rewards/overlap_rewards.py")
ost = _load("overlap_steps", "trl_repo/trl/trainer/overlap_steps.py")

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

print("\nAll CPU logic tests passed.")
