#!/usr/bin/env python
"""CPU test for rope_phase_e2: the stimulus and the crossing-point estimator.

The stimulus IS the experiment here -- if the square is not where the offset says
it is, every number downstream is wrong in a way no GPU run would reveal.  And
the crossing point is the whole readout, so its interpolation and its failure
case both get pinned down.

    python test_rope_phase_e2_cpu.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
_spec = importlib.util.spec_from_file_location("_e2", HERE / "rope_phase_e2.py")
E2 = importlib.util.module_from_spec(_spec)
sys.modules["_e2"] = E2
_spec.loader.exec_module(E2)

SIDE, PATCH, SQ = 768, 32, 64


def _px(img, x, y):
    return img.convert("RGB").getpixel((int(x), int(y)))


def test_square_sits_where_the_offset_says():
    """Offset is in patches and positive is DOWN, matching the row axis."""
    for off in (-4, -1, 0, 1, 4):
        img = E2.make_image(SIDE, off, "red", PATCH, SQ)
        cy = SIDE / 2 + off * PATCH
        assert _px(img, SIDE / 2, cy) == E2.COLOURS["red"], (off, "centre not coloured")
        # and the canvas is grey a comfortable distance away on the other side
        away = SIDE / 2 - off * PATCH - np.sign(off or 1) * 4 * PATCH
        if 0 < away < SIDE:
            assert _px(img, SIDE / 2, away) == (128, 128, 128), (off, "background not grey")


def test_offset_moves_by_exactly_one_patch():
    """A one-patch offset must move the square 32 px, not 32 of something else."""
    a = E2.make_image(SIDE, 0, "blue", PATCH, SQ)
    b = E2.make_image(SIDE, 1, "blue", PATCH, SQ)
    col = lambda im: [y for y in range(SIDE) if _px(im, SIDE / 2, y) == E2.COLOURS["blue"]]
    ca, cb = np.mean(col(a)), np.mean(col(b))
    assert abs((cb - ca) - PATCH) < 1.0, (ca, cb, "one patch should be one patch")


def test_square_is_horizontally_centred():
    """It must not carry a horizontal cue, or the column axis leaks into the answer."""
    img = E2.make_image(SIDE, 0, "green", PATCH, SQ)
    xs = [x for x in range(SIDE) if _px(img, x, SIDE / 2) == E2.COLOURS["green"]]
    assert abs(np.mean(xs) - SIDE / 2) < 1.0, np.mean(xs)


def test_crossing_interpolates():
    off = np.array([-2.0, -1.0, 1.0, 2.0])
    assert abs(E2.crossing(off, np.array([4.0, 2.0, -2.0, -4.0])) - 0.0) < 1e-9
    assert abs(E2.crossing(off, np.array([5.0, 1.0, -3.0, -7.0])) - (-0.5)) < 1e-9
    # unsorted input must not change the answer
    assert abs(E2.crossing(off[::-1], np.array([-4.0, -2.0, 2.0, 4.0])) - 0.0) < 1e-9


def test_crossing_reports_saturation_rather_than_guessing():
    """A model that always says 'top' has no midline; that must not read as 0."""
    off = np.array([-2.0, -1.0, 1.0, 2.0])
    assert np.isnan(E2.crossing(off, np.array([9.0, 8.0, 7.0, 6.0])))
    assert np.isnan(E2.crossing(off, np.array([-9.0, -8.0, -7.0, -6.0])))


def test_a_planted_midline_shift_is_recovered():
    """If the perceived midline really moves by d, the estimator should say d."""
    off = np.linspace(-4, 4, 17)
    for d in (-1.5, -0.5, 0.0, 0.5, 1.5):
        assert abs(E2.crossing(off, -(off - d)) - d) < 1e-9, d


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
