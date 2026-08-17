#!/usr/bin/env python
"""CPU test for rope_phase_e1: are the arms actually the interventions they claim?

The whole experiment rests on each arm perturbing exactly one family of relative
offsets and leaving the rest untouched.  That is a property of integers and can be
checked exhaustively without a GPU -- and it is where an off-by-one would silently
invalidate every number the GPU run produces.

Rather than testing the implementation against itself, these tests enumerate the
pairwise offsets that actually reach the attention logit -- tail-to-tail,
image-to-image, tail-to-image -- and assert which ones each arm is allowed to move.

    python test_rope_phase_e1_cpu.py        # or: pytest test_rope_phase_e1_cpu.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
_spec = importlib.util.spec_from_file_location("_e1", HERE / "rope_phase_e1.py")
E1 = importlib.util.module_from_spec(_spec)
sys.modules["_e1"] = E1
_spec.loader.exec_module(E1)

# A miniature but faithful layout: 3 system tokens, a 2x2 image anchored at s=3,
# then the tail.  Text carries (p, p, p); a patch at (r, c) carries (s, s+r, s+c);
# the tail resumes at s + max(gh, gw), exactly as get_rope_index builds it.
SYS, S, GH, GW = 3, 3, 2, 2
TAIL_START = SYS + GH * GW
IMG = list(range(SYS, TAIL_START))
TAIL = list(range(TAIL_START, TAIL_START + 5))


def base_positions():
    t, h, w = [], [], []
    for p in range(SYS):                      # system prompt
        t.append(p); h.append(p); w.append(p)
    for r in range(GH):                       # image patches, raster order
        for c in range(GW):
            t.append(S); h.append(S + r); w.append(S + c)
    for i in range(len(TAIL)):                # tail resumes past the image extent
        p = S + max(GH, GW) + i
        t.append(p); h.append(p); w.append(p)
    return torch.tensor([[t], [h], [w]], dtype=torch.long)   # [3, 1, S]


def offsets(pos, qs, ks):
    """Every (query, key) position difference per axis -- what RoPE actually sees."""
    return np.array([[[int(pos[a, 0, q] - pos[a, 0, k]) for k in ks] for q in qs]
                     for a in range(3)])


def moved(arm, delta, qs, ks):
    """Per-axis change in the offsets between query set qs and key set ks."""
    b = base_positions()
    p = E1.build_position_ids(b, TAIL_START, arm, delta)
    return offsets(p, qs, ks) - offsets(b, qs, ks)


def test_null_moves_no_offset_at_all():
    """RoPE depends only on differences, so a global shift must be a no-op."""
    for group in ((TAIL, IMG), (TAIL, TAIL), (IMG, IMG), (TAIL, range(SYS))):
        d = moved("null", 137, *group)
        assert np.all(d == 0), (group, d)


def test_hw_moves_only_the_cross_modal_spatial_offsets():
    """The claim of the experiment: `hw` is surgical."""
    d = moved("hw", 5, TAIL, IMG)
    assert np.all(d[0] == 0), "t offsets must not move"
    assert np.all(d[1] == 5) and np.all(d[2] == 5), d[1:]
    # everything else is untouched -- this is what makes it not-visual-fading
    for group in ((TAIL, TAIL), (IMG, IMG)):
        assert np.all(moved("hw", 5, *group) == 0), group


def test_t_moves_only_mass_never_shape():
    """Every patch shares the image's t index, so `t` cannot reshape the profile."""
    d = moved("t", 7, TAIL, IMG)
    assert np.all(d[1] == 0) and np.all(d[2] == 0), "h/w offsets must not move"
    assert np.all(d[0] == 7)
    # the t offset a tail token sees is IDENTICAL for every patch, before and after,
    # which is why this arm can only scale image-vs-text mass
    b = base_positions()
    for pos in (b, E1.build_position_ids(b, TAIL_START, "t", 7)):
        o = offsets(pos, TAIL, IMG)[0]
        assert (o == o[:, :1]).all(), "t offset varies across patches -- premise broken"
    assert np.all(moved("t", 7, TAIL, TAIL) == 0)


def test_full_is_t_plus_hw():
    a = moved("full", 9, TAIL, IMG)
    assert np.array_equal(a, moved("t", 9, TAIL, IMG) + moved("hw", 9, TAIL, IMG))


def test_fix_removes_the_p_dependence():
    """The positive control: the cross-modal h/w offset stops depending on the text index."""
    b = base_positions()
    for delta in (0, 3, 40):
        p = E1.build_position_ids(b, TAIL_START, "fix", delta)
        o = offsets(p, TAIL, IMG)
        for axis in (1, 2):
            # every tail token sees the SAME spatial offset to a given patch
            assert (o[axis] == o[axis][:1]).all(), (axis, o[axis])
        # and it does not depend on delta either, which is why the curve must be flat
        o0 = offsets(E1.build_position_ids(b, TAIL_START, "fix", 0), TAIL, IMG)
        assert np.array_equal(o[1:], o0[1:]), "fix must be delta-invariant on h/w"
    # the baseline genuinely does depend on p, or there would be nothing to fix
    ob = offsets(b, TAIL, IMG)
    assert not (ob[1] == ob[1][:1]).all()


def test_nothing_before_the_image_ever_moves():
    b = base_positions()
    for arm in ("full", "t", "hw", "fix"):
        p = E1.build_position_ids(b, TAIL_START, arm, 11)
        assert torch.equal(p[:, :, :TAIL_START], b[:, :, :TAIL_START]), arm


def test_parse_deltas():
    assert E1.parse_deltas("0-4,10,32") == [0, 1, 2, 3, 4, 10, 32]
    assert E1.parse_deltas("0-16,20")[:3] == [0, 1, 2]
    assert len(E1.parse_deltas("0-16,20,24,32,48,64,128,256")) == 24
    assert E1.parse_deltas("3,3,1") == [1, 3], "duplicates collapse, order normalised"


def test_centroid_is_in_patch_units():
    gh = gw = 4
    m = np.zeros((1, gh * gw)); m[0, 0] = 1.0                 # top-left patch
    r, c = E1.centroid(m, gh, gw)
    assert np.allclose(r, 0) and np.allclose(c, 0)
    m = np.zeros((1, gh * gw)); m[0, gh * gw - 1] = 1.0       # bottom-right
    r, c = E1.centroid(m, gh, gw)
    assert np.allclose(r, gh - 1) and np.allclose(c, gw - 1)
    m = np.ones((1, gh * gw))                                 # uniform -> centre
    r, c = E1.centroid(m, gh, gw)
    assert np.allclose(r, (gh - 1) / 2) and np.allclose(c, (gw - 1) / 2)


def test_unknown_arm_is_refused():
    try:
        E1.build_position_ids(base_positions(), TAIL_START, "sideways", 1)
    except ValueError:
        return
    raise AssertionError("an unknown arm must not silently pass through unchanged")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fails = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            fails += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - fails}/{len(tests)} passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
