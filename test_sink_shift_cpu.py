#!/usr/bin/env python
"""CPU checks for the inference-time attention edit. No GPU, no model download.

    python test_sink_shift_cpu.py

What each group is for:

  geometry      the patch sets are what the docstring says they are, and `rect` agrees
                patch for patch with the rectangle the REWARD scores. The two are written
                separately -- the reward's in numpy, this one in torch -- so nothing but
                a test keeps them from drifting, and a drifted rectangle would make the
                intervention and the training arm silently different experiments.
  algebra       the edit conserves mass exactly, alpha=0 is the identity, alpha=1 empties
                the source, and nothing ever goes negative. These are the invariants the
                whole design rests on: if the row stops summing to 1 the model is being
                fed something that is not a probability distribution and every downstream
                number is meaningless.
  attention     the registered attention function reproduces stock SDPA at alpha=0, and
                at alpha>0 moves exactly the mass it claims to. Rows before the first
                edited position, unselected heads and unselected layers are untouched.
  locate        the image columns and the per-picture grids come out right, including two
                pictures of different shapes in one prompt, and a wrong grid is refused
                rather than guessed at.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import sink_shift as SS  # noqa: E402

FAILURES = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def _import_overlap_rewards():
    """The shipped reward module, without importing the `trl` package (which needs torch,
    transformers and the trainer). Same trick mask_variance_probe.py uses."""
    pkg = types.ModuleType("trl_ss"); pkg.__path__ = [os.path.join(ROOT, "trl")]
    sys.modules["trl_ss"] = pkg
    sub = types.ModuleType("trl_ss.rewards")
    sub.__path__ = [os.path.join(ROOT, "trl", "rewards")]
    sys.modules["trl_ss.rewards"] = sub
    for name in ("roll_null", "overlap_rewards"):
        spec = importlib.util.spec_from_file_location(
            f"trl_ss.rewards.{name}", os.path.join(ROOT, "trl", "rewards", f"{name}.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"trl_ss.rewards.{name}"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["trl_ss.rewards.overlap_rewards"]


# ---------------------------------------------------------------------------
def test_geometry():
    print("\ngeometry")
    gh, gw = 10, 16
    fr, rc = SS.frame_set(gh, gw), SS.rect_set(gh, gw)
    r2, co = SS.ring2_set(gh, gw), SS.core_set(gh, gw)
    check("frame is 48 of 160", int(fr.sum()) == 48, f"got {int(fr.sum())}")
    check("rect is 96 of 160", int(rc.sum()) == 96, f"got {int(rc.sum())}")
    check("ring2 is 40 of 160", int(r2.sum()) == 40, f"got {int(r2.sum())}")
    check("core is 72 of 160", int(co.sum()) == 72, f"got {int(co.sum())}")
    check("rect touches no frame patch", int((rc & fr).sum()) == 0)
    check("ring2 touches no frame patch", int((r2 & fr).sum()) == 0)
    check("core is inside rect", bool((co & ~rc).sum() == 0))
    # The 16 patches the user's design decision excluded: outside the rectangle but not
    # on the frame. They must be in neither the source nor the destination of `centre`.
    mid = (~rc) & (~fr)
    rows = sorted(set(torch.nonzero(mid)[:, 0].tolist()))
    cols = sorted(set(torch.nonzero(mid)[:, 1].tolist()))
    check("16 patches are in neither frame nor rect", int(mid.sum()) == 16,
          f"got {int(mid.sum())} at rows {rows} cols {cols}")
    check("...and they are the two side strips", rows == list(range(1, 9)) and cols == [1, 14])

    ORW = _import_overlap_rewards()
    bad = []
    for gh in range(4, 22):
        for gw in range(4, 26):
            mine = SS.rect_set(gh, gw, SS.RECT_FRAC).numpy()
            theirs = ORW._centre_rect_mask(gh, gw, SS.RECT_FRAC)
            if theirs is None:
                continue
            if not np.array_equal(mine, theirs):
                bad.append((gh, gw))
    check("rect matches overlap_rewards._centre_rect_mask on every grid",
          not bad, f"{len(bad)} mismatched: {bad[:5]}")


# ---------------------------------------------------------------------------
def test_algebra():
    print("\nalgebra")
    g = torch.Generator().manual_seed(7)
    n = 160
    w = torch.rand(2, 3, 5, n, generator=g).double()
    src = SS.frame_set(10, 16).reshape(-1)
    dst = SS.rect_set(10, 16).reshape(-1)
    before = w.sum(-1)

    w0, moved0 = SS.move_fraction(w, src, dst, 0.0)
    check("alpha=0 is the identity", torch.equal(w0, w))
    check("alpha=0 moves nothing", float(moved0.abs().max()) == 0.0)

    for alpha in (0.25, 0.5, 1.0):
        out, moved = SS.move_fraction(w, src, dst, alpha)
        check(f"alpha={alpha} conserves the slice total",
              torch.allclose(out.sum(-1), before, atol=1e-12),
              f"max drift {float((out.sum(-1) - before).abs().max()):.2e}")
        check(f"alpha={alpha} never goes negative", bool((out >= 0).all()))
        want = (1 - alpha) * w[..., src].sum(-1)
        check(f"alpha={alpha} leaves (1-alpha) of the source",
              torch.allclose(out[..., src].sum(-1), want, atol=1e-12))
        check(f"alpha={alpha} reports the mass it moved",
              torch.allclose(moved.squeeze(-1), alpha * w[..., src].sum(-1), atol=1e-12))
        untouched = (~src) & (~dst)
        check(f"alpha={alpha} leaves the in-between patches alone",
              torch.equal(out[..., untouched], w[..., untouched]))

    everything = torch.ones(n, dtype=torch.bool)
    out, _ = SS.move_fraction(w, everything, everything, 0.4, target="uniform")
    ref = 0.6 * w + 0.4 * w.sum(-1, keepdim=True) / n
    check("flat is a blend toward even", torch.allclose(out, ref, atol=1e-12))

    zero = w.clone()
    zero[..., dst] = 0.0
    out, _ = SS.move_fraction(zero, src, dst, 1.0)
    got = out[..., dst]
    check("a destination with no shape gets an even split",
          torch.allclose(got, got[..., :1].expand_as(got), atol=1e-12))
    check("...and the mass still arrives",
          torch.allclose(out.sum(-1), zero.sum(-1), atol=1e-12))

    want = torch.full((2, 3, 5, 1), 0.05, dtype=torch.float64)
    out, took = SS.move_mass(w, src, dst, want)
    check("move_mass moves exactly what was asked",
          torch.allclose(took, want, atol=1e-12))
    check("move_mass conserves the total",
          torch.allclose(out.sum(-1), before, atol=1e-12))
    huge = torch.full((2, 3, 5, 1), 1e6, dtype=torch.float64)
    out, took = SS.move_mass(w, src, dst, huge)
    check("move_mass cannot take more than is there",
          torch.allclose(took.squeeze(-1), w[..., src].sum(-1), atol=1e-12))
    check("...and still nothing is negative", bool((out >= -1e-15).all()))

    allowed = torch.zeros(n, dtype=torch.bool)
    allowed[:20] = True
    t = SS.build_target(w, dst, allowed)
    check("build_target sums to 1", torch.allclose(t.sum(-1), torch.ones_like(t.sum(-1)),
                                                   atol=1e-12))
    check("build_target puts nothing where it is not allowed",
          float(t[..., ~(dst & allowed)].abs().max()) == 0.0)


# ---------------------------------------------------------------------------
class FakeAttn(torch.nn.Module):
    """Just enough of Qwen3VLTextAttention for the registered function to run."""

    def __init__(self, layer_idx, heads=4, kv_heads=2, head_dim=8):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_key_value_groups = heads // kv_heads
        self.head_dim = head_dim
        self.is_causal = True


def _state(arm="centre", alpha=0.5, layers=None, heads=None, rows="after_image",
           gh=10, gw=16, n_prompt=None, prefix=3):
    """A SinkShift wired to a synthetic prompt: `prefix` text tokens, then the picture,
    then the rest. No model is installed; only the attention function is exercised."""
    st = SS.SinkShift(torch.nn.Module(), arm=arm, alpha=alpha, layers=layers,
                      heads=heads, rows=rows)
    n_img = gh * gw
    n_prompt = n_prompt or (prefix + n_img + 4)
    ids = torch.zeros(1, n_prompt, dtype=torch.long)
    ids[0, prefix:prefix + n_img] = SS.IMAGE_TOKEN_ID
    st._locate_images(ids, torch.tensor([[1, gh * 2, gw * 2]]))
    return st, n_prompt


def _qkv(b, h, kv_h, q_len, kv_len, d, seed):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(b, h, q_len, d, generator=g)
    k = torch.randn(b, kv_h, kv_len, d, generator=g)
    v = torch.randn(b, kv_h, kv_len, d, generator=g)
    return q, k, v


def test_attention():
    print("\nattention")
    st, n = _state(alpha=0.0)
    fn = SS._make_attention(st)
    mod = FakeAttn(layer_idx=22)
    q, k, v = _qkv(1, 4, 2, n, n, 8, 11)

    ref, _ = SS._sdpa(mod, q, k, v, None, 0.0, None, True)
    got, _ = fn(mod, q, k, v, None, scaling=None)
    check("alpha=0 reproduces stock SDPA",
          torch.allclose(ref, got, atol=1e-5),
          f"max |diff| {float((ref - got).abs().max()):.2e}")

    # alpha=0 short-circuits to SDPA, so force the explicit path too: an arm that moves
    # nothing because its source is empty must ALSO reproduce SDPA, which is what proves
    # the hand-written softmax and the fused kernel agree rather than the branch hiding it.
    st2, _ = _state(alpha=1.0)
    st2.src_cols = torch.zeros_like(st2.src_cols)
    got2, _ = SS._make_attention(st2)(mod, q, k, v, None, scaling=None)
    check("the explicit softmax matches SDPA with an empty source",
          torch.allclose(ref, got2, atol=1e-5),
          f"max |diff| {float((ref - got2).abs().max()):.2e}")

    for alpha in (0.25, 0.5, 1.0):
        st3, n3 = _state(alpha=alpha)
        f3 = SS._make_attention(st3)
        q3, k3, v3 = _qkv(1, 4, 2, n3, n3, 8, 12)
        f3(mod, q3, k3, v3, None, scaling=None)
        d = st3.diagnostics()
        want = (1 - alpha) * d["frame_share_before"]
        check(f"alpha={alpha}: frame share falls to (1-alpha) of itself",
              abs(d["frame_share_after"] - want) < 1e-6,
              f"{d['frame_share_before']:.4f} -> {d['frame_share_after']:.4f}, "
              f"want {want:.4f}")
        check(f"alpha={alpha}: the rectangle gains what the frame lost",
              abs((d["rect_share_after"] - d["rect_share_before"])
                  - alpha * d["frame_share_before"]) < 1e-6)

    st4, n4 = _state(alpha=1.0)
    f4 = SS._make_attention(st4)
    q4, k4, v4 = _qkv(1, 4, 2, n4, n4, 8, 13)
    edited, _ = f4(mod, q4, k4, v4, None, scaling=None)
    plain, _ = SS._sdpa(mod, q4, k4, v4, None, 0.0, None, True)
    first = st4.first_row
    check("rows before the image are untouched",
          torch.allclose(edited[:, :first], plain[:, :first], atol=1e-6),
          f"first edited row is {first}")
    check("rows after the image did change",
          not torch.allclose(edited[:, first:], plain[:, first:], atol=1e-6))

    st5, n5 = _state(alpha=1.0, heads=[1])
    f5 = SS._make_attention(st5)
    q5, k5, v5 = _qkv(1, 4, 2, n5, n5, 8, 14)
    e5, _ = f5(mod, q5, k5, v5, None, scaling=None)
    p5, _ = SS._sdpa(mod, q5, k5, v5, None, 0.0, None, True)
    check("an unselected head is untouched",
          torch.allclose(e5[:, :, 0], p5[:, :, 0], atol=1e-6))
    check("the selected head is not",
          not torch.allclose(e5[:, :, 1], p5[:, :, 1], atol=1e-6))

    st6, n6 = _state(alpha=1.0, layers=[22])
    f6 = SS._make_attention(st6)
    q6, k6, v6 = _qkv(1, 4, 2, n6, n6, 8, 15)
    other = FakeAttn(layer_idx=21)
    e6, _ = f6(other, q6, k6, v6, None, scaling=None)
    p6, _ = SS._sdpa(other, q6, k6, v6, None, 0.0, None, True)
    check("an unselected layer is untouched", torch.allclose(e6, p6, atol=1e-6))
    check("...and records nothing", st6.diagnostics()["rows_edited"] == 0)

    # One decode step: the query is a single row and the keys carry the whole prompt.
    st7, n7 = _state(alpha=1.0)
    f7 = SS._make_attention(st7)
    g = torch.Generator().manual_seed(16)
    q7 = torch.randn(1, 4, 1, 8, generator=g)
    k7 = torch.randn(1, 2, n7 + 5, 8, generator=g)
    v7 = torch.randn(1, 2, n7 + 5, 8, generator=g)
    f7(mod, q7, k7, v7, None, scaling=None)
    d7 = st7.diagnostics()
    check("a decode step is edited too", d7["rows_edited"] == 1)
    check("...and empties the frame", d7["frame_share_after"] < 1e-9,
          f"{d7['frame_share_before']:.4f} -> {d7['frame_share_after']:.4f}")

    # `reverse` must be the mirror image: the frame GAINS what the rectangle loses.
    st8, n8 = _state(arm="reverse", alpha=0.5)
    f8 = SS._make_attention(st8)
    q8, k8, v8 = _qkv(1, 4, 2, n8, n8, 8, 17)
    f8(mod, q8, k8, v8, None, scaling=None)
    d8 = st8.diagnostics()
    check("reverse halves the rectangle",
          abs(d8["rect_share_after"] - 0.5 * d8["rect_share_before"]) < 1e-6)
    check("reverse hands it to the frame",
          abs((d8["frame_share_after"] - d8["frame_share_before"])
              - 0.5 * d8["rect_share_before"]) < 1e-6)

    # `text` must leave the picture exactly as it found it.
    st9, n9 = _state(arm="text", alpha=1.0)
    f9 = SS._make_attention(st9)
    q9, k9, v9 = _qkv(1, 4, 2, n9, n9, 8, 18)
    f9(mod, q9, k9, v9, None, scaling=None)
    d9 = st9.diagnostics()
    check("text leaves the frame share alone",
          abs(d9["frame_share_after"] - d9["frame_share_before"]) < 1e-9)
    check("text still moves a comparable mass", d9["row_mass_moved"] > 0)


# ---------------------------------------------------------------------------
def test_locate():
    print("\nlocate")
    st, n = _state(gh=10, gw=16, prefix=3)
    check("finds every image token", int(st.img_cols.numel()) == 160)
    check("finds them in the right place",
          int(st.img_cols.min()) == 3 and int(st.img_cols.max()) == 162)
    check("first edited row is just after the picture", st.first_row == 163)
    check("source is the frame", int(st.src_cols.sum()) == 48)
    check("destination is the rectangle", int(st.dst_cols.sum()) == 96)

    st_g = SS.SinkShift(torch.nn.Module(), arm="centre", alpha=0.5, rows="generated")
    ids = torch.zeros(1, 200, dtype=torch.long)
    ids[0, 3:163] = SS.IMAGE_TOKEN_ID
    st_g._locate_images(ids, torch.tensor([[1, 20, 32]]))
    check("rows=generated starts after the whole prompt", st_g.first_row == 200)

    # Two pictures of different shapes in one prompt.
    st2 = SS.SinkShift(torch.nn.Module(), arm="centre", alpha=0.5)
    ids = torch.zeros(1, 300, dtype=torch.long)
    ids[0, 5:5 + 160] = SS.IMAGE_TOKEN_ID          # 10x16
    ids[0, 200:200 + 96] = SS.IMAGE_TOKEN_ID       # 8x12
    st2._locate_images(ids, torch.tensor([[1, 20, 32], [1, 16, 24]]))
    check("two pictures give two grids", st2.grids == [(1, 10, 16), (1, 8, 12)])
    check("...and each gets its own border",
          int(st2.src_cols.sum()) == 48 + (2 * 8 + 2 * 12 - 4),
          f"got {int(st2.src_cols.sum())}")
    check("...and the columns are both runs",
          int(st2.img_cols.numel()) == 256)

    st3 = SS.SinkShift(torch.nn.Module(), arm="centre", alpha=0.5)
    ids = torch.zeros(1, 200, dtype=torch.long)
    ids[0, 5:5 + 160] = SS.IMAGE_TOKEN_ID
    try:
        st3._locate_images(ids, torch.tensor([[1, 12, 12]]))
        check("a grid that does not match the token count is refused", False)
    except RuntimeError:
        check("a grid that does not match the token count is refused", True)

    st4 = SS.SinkShift(torch.nn.Module(), arm="centre", alpha=0.5)
    try:
        st4._pre_hook(None, (), {"input_ids": torch.zeros(2, 10, dtype=torch.long)})
        check("a batch bigger than one is refused", False)
    except RuntimeError:
        check("a batch bigger than one is refused", True)

    st5, _ = _state()
    a = SS._cached_set(st5, "frame")
    st5._locate_images(torch.cat([torch.zeros(1, 3, dtype=torch.long),
                                  torch.full((1, 96), SS.IMAGE_TOKEN_ID),
                                  torch.zeros(1, 4, dtype=torch.long)], dim=1),
                       torch.tensor([[1, 16, 24]]))
    b = SS._cached_set(st5, "frame")
    check("the cached sets are dropped when the picture changes",
          a.numel() == 160 and b.numel() == 96)

    for arm in SS.ARMS:
        SS.SinkShift(torch.nn.Module(), arm=arm, alpha=0.5)
    check("every arm constructs", True)
    for bad in ({"arm": "nope"}, {"alpha": 1.5}, {"rows": "nope"}, {"target": "nope"}):
        try:
            SS.SinkShift(torch.nn.Module(), **bad)
            check(f"{bad} is refused", False)
        except ValueError:
            check(f"{bad} is refused", True)


def main():
    print("sink_shift CPU checks")
    test_geometry()
    test_algebra()
    test_attention()
    test_locate()
    print(f"\n{'ALL PASS' if not FAILURES else f'{len(FAILURES)} FAILED: {FAILURES}'}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
