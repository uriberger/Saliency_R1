#!/usr/bin/env python
"""CPU tests for rope_phase_e5: the decode-time freezer and the readouts.

The thing that could silently ruin this run is the rule that picks which rows to
freeze.  E4 knew `tail_start` because it saw the whole sequence at once; a decoder
sees one token and a cache, so E5 decides from the ids themselves -- a token is in
the tail exactly when its t index exceeds the image anchor.  If that rule were
wrong it would either freeze the image against itself or freeze nothing, and the
run would look fine either way.  So it is tested on a layout with text before the
image, the image, and text after it, in both the prefill and the decode shape.

    python test_rope_phase_e5_cpu.py      # or: pytest test_rope_phase_e5_cpu.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
_spec = importlib.util.spec_from_file_location("_e5", HERE / "rope_phase_e5.py")
E5 = importlib.util.module_from_spec(_spec)
sys.modules["_e5"] = E5
_spec.loader.exec_module(E5)

SYS, S, GH, GW = 3, 3, 2, 2          # 3 text tokens, a 2x2 image anchored at 3
TAIL_START = SYS + GH * GW


def positions(n_tail: int):
    t, h, w = [], [], []
    for p in range(SYS):
        t.append(p); h.append(p); w.append(p)
    for r in range(GH):
        for c in range(GW):
            t.append(S); h.append(S + r); w.append(S + c)
    for i in range(n_tail):
        p = S + max(GH, GW) + i
        t.append(p); h.append(p); w.append(p)
    return torch.tensor([[t], [h], [w]], dtype=torch.long)


def freeze(pos, anchor, d0):
    """The rule under test, lifted verbatim from RotaryFreezer.forward."""
    frozen = pos.clone()
    tail = pos[0] > anchor
    frozen[1][tail] = anchor + d0
    frozen[2][tail] = anchor + d0
    return frozen


# ---------------------------------------------------------------------------
def test_the_tail_rule_selects_exactly_the_post_image_tokens():
    pos = positions(5)
    tail = (pos[0] > S)[0]
    assert not tail[:SYS].any(), "text before the image must not be frozen"
    assert not tail[SYS:TAIL_START].any(), "the image must not be frozen against itself"
    assert tail[TAIL_START:].all(), "every post-image token must be frozen"
    assert int(tail.sum()) == 5


def test_frozen_rows_match_e4_on_the_same_layout():
    """E5 decides from the ids, E4 from tail_start; they must agree exactly."""
    import importlib.util as iu
    spec = iu.spec_from_file_location("_e4t", HERE / "rope_phase_e4.py")
    E4 = iu.module_from_spec(spec); sys.modules["_e4t"] = E4; spec.loader.exec_module(E4)
    for n_tail in (1, 5, 40):
        for d0 in (0, 2, 24):
            pos = positions(n_tail)
            assert torch.equal(freeze(pos, S, d0),
                               E4.frozen_position_ids(pos, TAIL_START, S, d0)), (n_tail, d0)


def test_decode_shape_is_frozen_too():
    """One token at a time is the case E4's patch refuses; it must work here."""
    full = positions(6)
    for i in range(6):
        step = full[:, :, TAIL_START + i: TAIL_START + i + 1]
        fz = freeze(step, S, 24)
        assert (fz[1] == S + 24).all() and (fz[2] == S + 24).all()
        assert torch.equal(fz[0], step[0]), "t must keep its real, advancing value"
    # and a single PRE-image token in decode shape must be left alone
    pre = full[:, :, 1:2]
    assert torch.equal(freeze(pre, S, 24), pre)


def test_freezing_is_flat_in_the_tail_and_still_varies_over_patches():
    pos = positions(9)
    fz = freeze(pos, S, 24)
    img = slice(SYS, TAIL_START)
    tail = slice(TAIL_START, None)
    for axis in (1, 2):
        v = fz[axis, 0, tail]
        assert len(set(v.tolist())) == 1, "every tail token must see the same phase"
        p = fz[axis, 0, img]
        assert len(set(p.tolist())) > 1, "the image's own spatial code must survive"


def test_first_divergence():
    assert E5.first_divergence([1, 2, 3], [1, 2, 3]) == -1
    assert E5.first_divergence([1, 2, 3], [1, 9, 3]) == 1
    assert E5.first_divergence([1, 2], [1, 2, 3]) == 2, "a prefix diverges at its end"
    assert E5.first_divergence([], []) == -1
    assert E5.first_divergence([5], [6]) == 0


def test_match_is_normalised_but_not_generous():
    assert E5.matches("The answer is a Giraffe.", "giraffe")
    assert E5.matches("it's a fire-hydrant", "fire hydrant"), "punctuation normalised"
    assert not E5.matches("there is a horse", "giraffe")
    assert not E5.matches("", "giraffe")


def test_parse_arms():
    got = E5.parse_arms("none,null:1000,frozen:24,frozen:0")
    assert got == [("none", "none", 0), ("null:1000", "null", 1000),
                   ("frozen:24", "frozen", 24), ("frozen:0", "frozen", 0)]
    assert E5.parse_arms("null")[0][2] == 1000, "null defaults to a large offset"
    try:
        E5.parse_arms("sideways")
    except SystemExit:
        return
    raise AssertionError("an unknown arm must not pass through silently")


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
