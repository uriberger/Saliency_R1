#!/usr/bin/env python
"""CPU test for rope_phase_e1: are the arms actually the interventions they claim?

The whole experiment rests on each arm perturbing exactly one family of relative
offsets and leaving the rest untouched.  That is a property of integers and can be
checked exhaustively without a GPU -- and it is where an off-by-one would silently
invalidate every number the GPU run produces.

Rather than testing the implementation against itself, these tests enumerate the
pairwise offsets that actually reach the attention logit -- tail-to-tail,
image-to-image, tail-to-image -- and assert which ones each arm is allowed to move.

The second half tests the phase-bucketed analysis: plant a pattern that translates
by a known amount, impose a gap, and check the gap is read back out -- including
that it wraps at one period, and that half a period is reported as the genuinely
ambiguous quantity it is.

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


def test_unknown_arm_is_refused():
    try:
        E1.build_position_ids(base_positions(), TAIL_START, "sideways", 1)
    except ValueError:
        return
    raise AssertionError("an unknown arm must not silently pass through unchanged")


# ---------------------------------------------------------------------------
# the phase-bucketed analysis: does an imposed gap come back as itself?
# ---------------------------------------------------------------------------
def test_sawtooth_prediction():
    """A gap of one period is indistinguishable from no gap; half a period aliases."""
    f = E1.sawtooth
    assert [f(g, 8.0, -4, 4) for g in range(9)] == [0, 1, 2, 3, 4, -3, -2, -1, 0]
    assert f(16, 8.0, -4, 4) == 0 and f(9, 8.0, -4, 4) == 1
    assert f(8, 7.996, -4, 4) == 0, "the true period is a shade under 8; still wraps"
    assert f(10, 10.17, -4, 4) == 0, "the column clock wraps on its own period"


def _recover(cur, base, shifts, axis=2):
    sc = [E1.RP._corr(*E1.RP._crop_pair(cur, base, int(s), axis)).mean() for s in shifts]
    return shifts[int(np.argmax(sc))]


def test_imposed_gap_is_recovered_and_wraps():
    """Plant a translating pattern, impose a gap, and read the gap back out.

    This is the whole logic of the rebuilt E1: a bucket at gap N should be that
    bucket at gap 0 translated by N, so the fitted shift should return N -- and
    wrap once N reaches a full period.
    """
    gh = gw = 24
    period = 8.0
    rng = np.random.default_rng(0)
    rows = np.arange(gh)[None, None, :, None]
    phase = rng.uniform(0, 2 * np.pi, size=(3, 8, 1, 1))     # heads x buckets

    def pattern(gap):
        clean = np.cos(2 * np.pi * (rows - gap) / period + phase)
        return clean + 0.05 * rng.normal(size=(3, 8, gh, gw))

    base = pattern(0)
    shifts = list(range(-4, 5))
    for gap in (0, 1, 2, 3, 5, 6, 7, 8, 9, 11, 16):
        got = _recover(pattern(gap), base, shifts)
        assert got == E1.sawtooth(gap, period, -4, 4), (gap, got)

    # half a period is genuinely ambiguous: +4 and -4 are the same pattern, so the
    # measurement cannot tell them apart and the report must not pretend otherwise
    assert abs(_recover(pattern(4), base, shifts)) == 4


def test_a_pattern_that_does_not_move_recovers_zero():
    """The `t` and `fix` arms must land here: no translation, no recovered shift."""
    gh = gw = 24
    rng = np.random.default_rng(1)
    rows = np.arange(gh)[None, None, :, None]
    phase = rng.uniform(0, 2 * np.pi, size=(3, 8, 1, 1))
    still = np.cos(2 * np.pi * rows / 8.0 + phase)
    shifts = list(range(-4, 5))
    for _ in range(3):
        cur = still + 0.05 * rng.normal(size=still.shape)
        base = still + 0.05 * rng.normal(size=still.shape)
        assert _recover(cur, base, shifts) == 0


def test_content_must_be_removed_or_the_shift_test_is_blind():
    """Why the bucket mean is subtracted: a shared static term hides the translation.

    A head's real, content-driven preference is the same in every bucket and at
    every gap.  Left in, it sits on both sides of the comparison and dominates the
    correlation, so the fitted shift stays at zero however far the positional part
    has actually moved.  This test plants exactly that situation.
    """
    gh = gw = 24
    rng = np.random.default_rng(3)
    rows = np.arange(gh)[None, None, :, None]
    phase = rng.uniform(0, 2 * np.pi, size=(3, 8, 1, 1))
    content = rng.normal(size=(3, 1, gh, gw)) * 3.0          # same in every bucket

    def profile(gap):
        return content + np.cos(2 * np.pi * (rows - gap) / 8.0 + phase)

    shifts = list(range(-4, 5))
    raw_base, raw_cur = profile(0), profile(3)
    assert _recover(raw_cur, raw_base, shifts) == 0, "content should swamp it"

    dec = lambda x: x - x.mean(axis=1, keepdims=True)         # the subtraction
    assert _recover(dec(raw_cur), dec(raw_base), shifts) == 3, "and removing it should fix it"


def test_flip_rate_slice_is_aligned():
    """An off-by-one here would compare each position with its neighbour.

    The logits row at prompt_len-1 predicts the FIRST completion token, and the
    last row predicts one past the end and must be dropped -- the same slice the
    NLL uses.  Get it wrong and every position looks flipped.
    """
    V, prompt_len, n_comp = 7, 4, 5
    logits = torch.full((1, prompt_len + n_comp, V), -10.0)
    want = torch.tensor([1, 2, 3, 4, 5])
    for i, tok in enumerate(want):          # row prompt_len-1+i predicts completion i
        logits[0, prompt_len - 1 + i, tok] = 10.0
    got = E1.greedy_tokens(logits, prompt_len)
    assert got.shape == want.shape, (got.shape, want.shape)
    assert torch.equal(got, want), (got, want)
    assert E1.flip_rate(logits, want, prompt_len) == 0.0

    moved = logits.clone()
    moved[0, prompt_len] = -10.0
    moved[0, prompt_len, 6] = 10.0          # change exactly one position
    assert abs(E1.flip_rate(moved, want, prompt_len) - 1 / n_comp) < 1e-9


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
