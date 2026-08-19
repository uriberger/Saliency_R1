#!/usr/bin/env python
"""CPU tests for rope_phase_e4: is the intervention the one it claims to be?

Two things could silently invalidate the GPU run and both are decidable without a
GPU.  First, the frozen query must change the cross-modal spatial offsets and
NOTHING else -- in particular it must leave tail-to-tail attention alone, which is
the whole difference between this and E1's `fix` arm.  That is a property of
integers, so it is checked by enumerating the pairwise offsets that actually reach
the attention logit rather than by testing the implementation against itself.

Second, the splice has to land on the image columns.  A miniature attention with a
planted, position-dependent pattern makes that visible: freeze the query and the
pattern must stop depending on where the query sits, while text-to-text scores
come back bit-identical.

    python test_rope_phase_e4_cpu.py       # or: pytest test_rope_phase_e4_cpu.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
_spec = importlib.util.spec_from_file_location("_e4", HERE / "rope_phase_e4.py")
E4 = importlib.util.module_from_spec(_spec)
sys.modules["_e4"] = E4
_spec.loader.exec_module(E4)

# The same miniature layout E1's tests use: 3 system tokens, a 2x2 image anchored
# at s = 3, then the tail.  Text carries (p, p, p); a patch at (r, c) carries
# (s, s+r, s+c); the tail resumes at s + max(gh, gw).
SYS, S, GH, GW = 3, 3, 2, 2
TAIL_START = SYS + GH * GW
IMG = list(range(SYS, TAIL_START))
TAIL = list(range(TAIL_START, TAIL_START + 5))


def base_positions():
    t, h, w = [], [], []
    for p in range(SYS):
        t.append(p); h.append(p); w.append(p)
    for r in range(GH):
        for c in range(GW):
            t.append(S); h.append(S + r); w.append(S + c)
    for i in range(len(TAIL)):
        p = S + max(GH, GW) + i
        t.append(p); h.append(p); w.append(p)
    return torch.tensor([[t], [h], [w]], dtype=torch.long)


def offsets(pos, qs, ks):
    return np.array([[[int(pos[a, 0, q] - pos[a, 0, k]) for k in ks] for q in qs]
                     for a in range(3)])


# ---------------------------------------------------------------------------
# what the intervention is allowed to move
# ---------------------------------------------------------------------------
def test_frozen_query_moves_only_the_cross_modal_spatial_offsets():
    """The claim of E4, and the whole difference from E1's `fix` arm.

    The frozen ids are used ONLY when scoring image columns, so the offsets that
    matter are tail->image under the frozen ids and everything else under the real
    ones.  Tail-to-tail must be untouched -- `fix` flattened it and cost 168% NLL.
    """
    base = base_positions()
    fro = E4.frozen_position_ids(base, TAIL_START, S, 24)

    d = offsets(fro, TAIL, IMG) - offsets(base, TAIL, IMG)
    assert np.all(d[0] == 0), "the t offset must not move: mass is left alone"
    assert np.all(d[1] != 0) and np.all(d[2] != 0), "h/w to the image must move"
    # every tail token now sees the SAME spatial offset to a given patch
    o = offsets(fro, TAIL, IMG)
    for axis in (1, 2):
        assert (o[axis] == o[axis][:1]).all(), (axis, "the drift is not removed")

    # ... and text-to-text scoring keeps using the REAL ids, so the only way this
    # could leak into tail<->tail or tail<->system is if the frozen ids were used
    # somewhere they should not be.  Pin down exactly which entries differ: the h
    # and w rows of the tail, and nothing else anywhere.
    diff = (fro != base)
    assert not diff[0].any(), "the t axis must be identical everywhere"
    assert not diff[:, :, :TAIL_START].any(), "nothing before the tail may change"
    assert diff[1:, :, TAIL_START:].all(), "every tail h/w entry should be pinned"
    assert torch.equal(base, base_positions()), "frozen_position_ids must not mutate"


def test_frozen_is_flat_in_the_gap():
    """Whatever the answer length, the image sees the same query position."""
    base = base_positions()
    ref = None
    for gap in (0, 7, 8, 512):
        pos = base.clone()
        pos[:, :, TAIL_START:] += gap
        fro = E4.frozen_position_ids(pos, TAIL_START, S, 24)
        o = offsets(fro, TAIL, IMG)
        # Constant down the QUERY axis -- every tail token sees the same thing.  It
        # must still vary along the KEY axis, since that variation IS the spatial
        # code: patch row r is seen at d0 - r.
        for axis in (1, 2):
            assert (o[axis] == o[axis][:1]).all(), (gap, axis, "the drift survives")
            assert o[axis][0].min() != o[axis][0].max(), \
                (gap, axis, "the spatial code was flattened, not just frozen")
        if ref is None:
            ref = (o[1].copy(), o[2].copy())
        assert np.array_equal(o[1], ref[0]) and np.array_equal(o[2], ref[1]), gap
        # the t offset, deliberately, still grows with the gap
        assert o[0].min() == gap + max(GH, GW)


def test_natural_d0_is_a_no_op_for_the_first_tail_token():
    """d0 = max(gh, gw) must leave the first post-image token exactly as it was."""
    base = base_positions()
    fro = E4.frozen_position_ids(base, TAIL_START, S, max(GH, GW))
    first = [TAIL_START]
    assert np.array_equal(offsets(fro, first, IMG), offsets(base, first, IMG))
    later = TAIL[1:]
    assert not np.array_equal(offsets(fro, later, IMG), offsets(base, later, IMG))


# ---------------------------------------------------------------------------
# does the splice land on the image columns?
# ---------------------------------------------------------------------------
class _Tiny(torch.nn.Module):
    """The smallest thing with E4's attention contract: one head, no GQA, no norms."""

    def __init__(self, dim=8, seq=12):
        super().__init__()
        self.head_dim, self.num_key_value_groups, self.scaling = dim, 1, 1.0
        self.layer_type = None
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            lin = torch.nn.Linear(dim, dim, bias=False)
            lin.weight.data = torch.eye(dim)
            setattr(self, name, lin)
        self.q_norm = self.k_norm = torch.nn.Identity()


def _rope(q, k, cos, sin):
    """A stand-in for apply_rotary_pos_emb: a phase, not a rotation, but it is the
    same contract -- q and k are modulated by the position-derived cos/sin."""
    return q * cos.unsqueeze(1), k * cos.unsqueeze(1)


def _run_tiny(mod, ctl, h, cos, sin, mask=None):
    return E4._frozen_forward(mod, h, (cos, sin), mask)[0]


def test_the_splice_hits_the_image_columns_and_only_those():
    torch.manual_seed(0)
    dim, seq = 8, 12
    img = torch.arange(3, 8)
    mod = _Tiny(dim, seq)

    class Ctl:
        pass
    ctl = Ctl()
    ctl.apply_rope, ctl.repeat_kv = _rope, lambda x, n: x
    mod._rp_ctl = ctl

    h = torch.randn(1, seq, dim)
    cos = torch.rand(1, seq, dim) + 0.5
    sin = torch.zeros(1, seq, dim)
    cos_f = cos.clone()
    cos_f[:, 8:, :] = 0.25            # a different "position" for the tail only

    ctl.enabled = False
    plain = _run_tiny(mod, ctl, h, cos, sin)
    ctl.enabled, ctl.img_cols, ctl.cos_f, ctl.sin_f = True, img, cos_f, sin
    spliced = _run_tiny(mod, ctl, h, cos, sin)
    assert not torch.allclose(plain, spliced), "the splice did nothing at all"

    # with an identical frozen phase it must be a no-op to arithmetic
    ctl.cos_f = cos.clone()
    same = _run_tiny(mod, ctl, h, cos, sin)
    assert torch.allclose(plain, same, atol=1e-6), (plain - same).abs().max()

    # and rows before the tail must be untouched even when the splice is active,
    # because the frozen phase equals the real one there
    ctl.cos_f = cos_f
    again = _run_tiny(mod, ctl, h, cos, sin)
    assert torch.allclose(plain[:, :8], again[:, :8], atol=1e-6), \
        "rows whose frozen phase equals their real one must not move"
    assert not torch.allclose(plain[:, 8:], again[:, 8:]), "tail rows must move"

    # With no image columns there is nothing to splice, so however different the
    # frozen phase is the result must be the untouched one.  That is what pins the
    # edit to the listed columns rather than to the whole score matrix.
    ctl.img_cols = torch.empty(0, dtype=torch.long)
    none_spliced = _run_tiny(mod, ctl, h, cos, sin)
    assert torch.allclose(plain, none_spliced, atol=1e-6), \
        "the splice touched columns that were not in img_cols"


def test_causal_mask_is_applied_when_none_is_passed():
    """If the decoder hands attention no mask, the fallback must still be causal."""
    dim, seq = 8, 6
    mod = _Tiny(dim, seq)

    class Ctl:
        pass
    ctl = Ctl()
    ctl.apply_rope, ctl.repeat_kv, ctl.enabled = _rope, (lambda x, n: x), False
    mod._rp_ctl = ctl
    h = torch.zeros(1, seq, dim)
    h[0, 3] = 5.0                     # a spike only a non-causal model could see early
    cos, sin = torch.ones(1, seq, dim), torch.zeros(1, seq, dim)
    out = _run_tiny(mod, ctl, h, cos, sin)
    assert out[0, 0].abs().max() < 1e-6, "position 0 attended to a later token"


def test_resolve_mask_accepts_the_shapes_the_decoder_may_send():
    dt = torch.float32
    m = E4._resolve_mask(None, _Tiny(), 4, 4, dt, torch.device("cpu"))
    assert m.shape == (1, 1, 4, 4) and m[0, 0, 0, 1] < -1e30 and m[0, 0, 1, 0] == 0
    b = torch.ones(1, 1, 4, 4, dtype=torch.bool)
    b[0, 0, 0, 2] = False
    mb = E4._resolve_mask(b, _Tiny(), 4, 4, dt, torch.device("cpu"))
    assert mb[0, 0, 0, 2] < -1e30 and mb[0, 0, 0, 0] == 0
    mod = _Tiny()
    mod.layer_type = "full_attention"
    md = E4._resolve_mask({"full_attention": b}, mod, 4, 4, dt, torch.device("cpu"))
    assert torch.equal(md, mb)


# ---------------------------------------------------------------------------
# the stimulus and the readout
# ---------------------------------------------------------------------------
def test_real_stimulus_translates_and_never_crops():
    from PIL import Image

    side, patch, target_h = 768, 32, 512
    img = Image.new("RGB", (640, 400), (10, 200, 30))
    bbox = [0.35, 0.4, 0.65, 0.6]
    counts, centres = [], []
    for off in (-1.5, -0.5, 0.0, 1.0, 1.5):
        canv = E4.real_stimulus(img, bbox, side, patch, target_h, off)
        rows = [y for y in range(side)
                if canv.getpixel((side // 2, y)) != (128, 128, 128)]
        counts.append(len(rows))
        centres.append((np.mean(rows) - side / 2) / patch)
        assert abs(centres[-1] - off) < 0.06, (off, centres[-1])
    assert len(set(counts)) == 1, (counts, "content changed across offsets")


def test_slack_window_is_the_condition_for_not_cropping():
    side, off_px = 768, 48.0
    for h in (300.0, 512.0, 600.0):
        lo, hi = E4.slack_window(h, side, off_px)
        for cy in (lo, hi, (lo + hi) / 2):
            if lo > hi:
                continue
            for sgn in (-1, 1):
                top = side / 2 + sgn * off_px - cy
                assert -1e-9 <= top <= side - h + 1e-9, (h, cy, sgn, top)


def test_fit_zero_recovers_a_planted_midline():
    offs = np.array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
    for true in (-0.8, -0.2, 0.0, 0.4):
        z, slope = E4.fit_zero(offs, -3.3 * (offs - true))
        assert abs(z - true) < 1e-9, (true, z)
        assert abs(slope + 3.3) < 1e-9
    # a flat curve has no midline to report, and the slope says so
    z, slope = E4.fit_zero(offs, np.zeros_like(offs))
    assert np.isnan(z) and abs(slope) < 1e-9


def _write_partial(tmp, meta, ev, done):
    np.savez(tmp, ev=ev, __meta__=np.frombuffer(
        json.dumps({**meta, "done": done}).encode(), dtype=np.uint8))


def test_resume_keeps_finished_stimuli_and_redoes_nothing_else():
    import tempfile

    shape = (5, 3, 2, 2)
    meta = {"family": "real", "offsets": [-1.0, 0.0, 1.0], "d0s": [24],
            "gaps": [0, 512], "arms": ["none", "frozen:24"], "image_side": 768,
            "square_px": 64, "target_h": 512, "base_model": "m", "dataset": "d",
            "gh": 24, "gw": 24, "rows": [{"row_index": i} for i in range(5)]}
    ev = np.full(shape, np.nan)
    ev[:3] = 1.0
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "scan_real.npz"
        _write_partial(p, meta, ev, 3)
        got, done = E4._load_partial(p, meta, shape)
        assert done == 3 and np.all(got[:3] == 1.0) and np.isnan(got[3]).all()

        # a counter left ahead of the data by a kill mid-write must not be believed
        ev2 = np.full(shape, np.nan)
        ev2[:2] = 1.0
        _write_partial(p, meta, ev2, 4)
        _, done2 = E4._load_partial(p, meta, shape)
        assert done2 == 2, (done2, "trusted a count the data does not back up")

        # a different sweep must be refused, not silently stitched together
        for key, bad in (("d0s", [25]), ("gaps", [0, 256]), ("dataset", "other"),
                         ("offsets", [-2.0, 0.0, 2.0]),
                         ("rows", [{"row_index": i} for i in range(5, 10)])):
            try:
                E4._load_partial(p, {**meta, key: bad}, shape)
            except SystemExit:
                continue
            raise AssertionError(f"a sweep differing on {key} was accepted")

        # and a fresh directory simply starts at zero
        assert E4._load_partial(Path(td) / "absent.npz", meta, shape) == (None, 0)


def test_arm_labels_and_d0_parsing():
    assert E4.parse_d0("0,12,24-27,64") == [0, 12, 24, 25, 26, 27, 64]
    assert E4.arm_labels([24, 25])[0] == "none"
    assert E4.arm_labels([24, 25])[1:] == ["frozen:24", "frozen:25"]


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
