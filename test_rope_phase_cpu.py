#!/usr/bin/env python
"""CPU test for rope_phase_probe: does the lock-in actually recover a known march?

Runs on the login node in a few seconds.  It plants a marching overlay of known
amplitude and rate underneath synthetic "content", pushes it through the SAME
accumulation and shift-test code the GPU scan uses, and checks that the analysis
returns the planted rate -- and, just as important, that it returns nothing when
there is nothing planted.

    python test_rope_phase_cpu.py          # or: pytest test_rope_phase_cpu.py
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("_rope_phase", REPO / "rope_phase_probe.py")
RP = importlib.util.module_from_spec(_spec)
sys.modules["_rope_phase"] = RP
_spec.loader.exec_module(RP)


# ---------------------------------------------------------------------------
# 1. the frequency table -- these are the numbers the whole prediction rests on
# ---------------------------------------------------------------------------
def qwen3_vl_text_config():
    return SimpleNamespace(
        head_dim=128, hidden_size=4096, num_attention_heads=32, num_hidden_layers=36,
        rope_theta=5_000_000.0,
        rope_scaling={"mrope_section": [24, 20, 20], "mrope_interleaved": True,
                      "rope_type": "default"},
    )


def qwen25_vl_text_config():
    return SimpleNamespace(
        head_dim=128, hidden_size=3584, num_attention_heads=28, num_hidden_layers=28,
        rope_theta=1_000_000.0,
        rope_scaling={"mrope_section": [16, 24, 24], "rope_type": "default"},
    )


def test_interleaved_channel_map():
    ch = RP.axis_channels(qwen3_vl_text_config())
    h_idx, h_freq = ch["h"]
    w_idx, w_freq = ch["w"]
    t_idx, _ = ch["t"]
    assert h_idx[:4] == [1, 4, 7, 10], h_idx[:4]
    assert w_idx[:4] == [2, 5, 8, 11], w_idx[:4]
    assert len(h_idx) == 20 and len(w_idx) == 20 and len(t_idx) == 24
    assert len(h_idx) + len(w_idx) + len(t_idx) == 64
    assert math.isclose(max(h_freq), 0.7858, abs_tol=1e-3), max(h_freq)
    assert math.isclose(max(w_freq), 0.6175, abs_tol=1e-3), max(w_freq)


def test_chunked_channel_map():
    """Qwen2.5-VL's W axis is numerically dead; the probe must report that, not hide it."""
    ch = RP.axis_channels(qwen25_vl_text_config())
    assert ch["h"][0][0] == 16 and ch["w"][0][0] == 40
    assert math.isclose(max(ch["h"][1]), 0.03162, rel_tol=1e-3)
    assert math.isclose(max(ch["w"][1]), 1.78e-4, rel_tol=1e-2)
    # across a 32-column image the entire horizontal phase sweep is milliradians
    assert 31 * max(ch["w"][1]) < 0.01


def test_predicted_shifts():
    tc = qwen3_vl_text_config()
    bs = {b.name: b for b in RP.default_binnings(tc, decoy=0.31)}
    assert math.isclose(bs["h8"].predicted_shift, 1.0, abs_tol=0.01)
    assert math.isclose(bs["h4"].predicted_shift, 2.0, abs_tol=0.02)
    assert math.isclose(bs["w10"].predicted_shift, 1.018, abs_tol=0.01)
    assert bs["perm8"].predicted_shift == 0.0


# ---------------------------------------------------------------------------
# 2. plant a march, recover it
# ---------------------------------------------------------------------------
def synth(amp: float, seed: int = 0, gh: int = 24, gw: int = 24, n_heads: int = 4,
          n_layers: int = 2, n_cases: int = 40, per_case: int = 200,
          content: float = 1.0, noise: float = 0.5):
    """Build the accumulators the GPU scan would have built, for a known overlay.

    Content is drawn once per case and shared by that case's tokens -- the real
    situation, where the model looks at the same picture for a whole completion.
    That is what the bucket residual has to cancel.
    """
    tc = qwen3_vl_text_config()
    th, tw = RP.fastest_theta(tc, "h"), RP.fastest_theta(tc, "w")
    binnings = RP.default_binnings(tc, decoy=0.31)
    P = gh * gw
    r = np.repeat(np.arange(gh), gw).astype(np.float64)
    c = np.tile(np.arange(gw), gh).astype(np.float64)

    rng = np.random.default_rng(seed)
    acc = RP.BinAccumulator(binnings, n_layers, n_heads, P, "cpu")
    phase_h = rng.uniform(0, 2 * np.pi, size=(n_layers, n_heads))
    phase_w = rng.uniform(0, 2 * np.pi, size=(n_layers, n_heads))

    for case in range(n_cases):
        d = np.arange(per_case, dtype=np.float64) + rng.integers(60, 400)
        dt = torch.as_tensor(d)
        bins = RP.make_bins(dt, binnings, rng)
        base = rng.normal(0, content, size=(n_layers, n_heads, P))
        for li in range(n_layers):
            x = np.broadcast_to(base[li][:, None, :], (n_heads, per_case, P)).copy()
            x += rng.normal(0, noise, size=x.shape)
            if amp:
                for hi in range(n_heads):
                    x[hi] += amp * np.cos(th * (d[:, None] - r[None, :]) + phase_h[li, hi])
                    x[hi] += amp * np.cos(tw * (d[:, None] - c[None, :]) + phase_w[li, hi])
            xt = torch.as_tensor(x, dtype=torch.float32)
            xt -= xt.mean(-1, keepdim=True)          # the probe's mean-centring
            acc.add_layer(li, xt, bins)
        acc.add_case(bins, dt)
    return acc, binnings, gh, gw


def run_analysis(acc, binnings, gh, gw, max_shift=4):
    shifts = list(range(-max_shift, max_shift + 1))
    out = {}
    for b in binnings:
        sums = acc.sum[b.name].numpy()
        counts = acc.count[b.name].numpy()
        resid = RP.bucket_residuals(sums, counts, gh, gw)
        sc = RP.shift_scores(resid, b.axis, shifts)
        out[b.name] = RP.summarize(resid, sc, shifts, b.predicted_shift)
    return out


def test_planted_march_is_recovered():
    acc, binnings, gh, gw = synth(amp=0.15, seed=1)
    res = run_analysis(acc, binnings, gh, gw)

    # h8: one row per bucket, on the row axis
    assert res["h8"]["predicted_ints"] == [1]
    assert res["h8"]["median_argmax"] == 1.0, res["h8"]["hist"]
    assert res["h8"]["frac_at_predicted"] >= 0.99, res["h8"]["hist"]

    # h4: same channel, half the buckets, twice the rate -- the second free prediction
    assert res["h4"]["predicted_ints"] == [2]
    assert res["h4"]["median_argmax"] == 2.0, res["h4"]["hist"]
    assert res["h4"]["frac_at_predicted"] >= 0.99, res["h4"]["hist"]

    # w10: the other axis, its own frequency, sliding columns not rows
    assert res["w10"]["predicted_ints"] == [1]
    assert res["w10"]["median_argmax"] == 1.0, res["w10"]["hist"]

    # decoy8 is not a null: bucketing by ANY phase separates d, so the same law
    # predicts its own (non-integer) rate, and the bracketing pair must catch it
    # It is the weakest of the checks -- a decoy bucket spans ~2.5 periods of the
    # planted stripe, so most of the signal averages away inside it -- so require
    # only that it beats chance, not that it is clean.
    assert res["decoy8"]["predicted_ints"] == [2, 3], res["decoy8"]["predicted_ints"]
    assert res["decoy8"]["frac_at_predicted"] >= 2 * res["decoy8"]["frac_expected_by_chance"], \
        res["decoy8"]["hist"]

    # perm8 is the matched floor: no consistent shift, and far less residual power
    assert res["perm8"]["frac_at_predicted"] <= 0.4, res["perm8"]["hist"]
    assert res["h8"]["resid_power"] > 5 * res["perm8"]["resid_power"], (
        res["h8"]["resid_power"], res["perm8"]["resid_power"])


def test_no_march_when_nothing_is_planted():
    """The control that matters: content alone must not manufacture a shift.

    Also pins the floor.  perm8 relabels each case's buckets rather than drawing
    them iid, so with no overlay planted its residual power must land on top of
    the real binning's -- an iid draw instead lands ~5x high, because consecutive
    d values fill the phase buckets evenly and cancel a case's content better than
    a random assignment does.  That difference would masquerade as signal.
    """
    acc, binnings, gh, gw = synth(amp=0.0, seed=2)
    res = run_analysis(acc, binnings, gh, gw)
    for name in ("h8", "h4", "w10"):
        assert res[name]["frac_at_predicted"] <= 0.4, (name, res[name]["hist"])
    ratio = res["h8"]["resid_power"] / res["perm8"]["resid_power"]
    assert 0.7 < ratio < 1.4, ratio


def test_phase_bins_do_not_slip():
    """d % 8 slips against the true 7.996 period; phase bucketing must not."""
    th = RP.fastest_theta(qwen3_vl_text_config(), "h")
    d = np.arange(0, 4000)
    b_phase = RP.phase_bins(d, th, 8)
    b_mod = (d % 8).astype(np.int64)
    early = (b_phase[:100] == b_mod[:100]).mean()
    late = (b_phase[-100:] == b_mod[-100:]).mean()
    assert early > 0.8, early
    assert late < 0.4, late          # by d ~ 4000 the naive modulus is half a period out


def test_crop_pair_is_not_circular():
    a = np.arange(12, dtype=float).reshape(1, 1, 1, 4, 3)
    va, vb = RP._crop_pair(a, a, 1, 3)
    assert va.shape[3] == 3 and vb.shape[3] == 3
    assert np.allclose(va[0, 0, 0, :, 0], [3, 6, 9])
    assert np.allclose(vb[0, 0, 0, :, 0], [0, 3, 6])


# ---------------------------------------------------------------------------
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
