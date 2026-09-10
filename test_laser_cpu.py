#!/usr/bin/env python
"""CPU checks for the LASER go/no-go. No GPU, no model, no dataset.

    python test_laser_cpu.py

What each group is for:

  transcription   the two rewards against a TORCH rewrite of upstream's own lines --
                  `unfold`, `.max().detach()`, `torch.clamp`, `torch.std` -- rather than
                  against this module's reading of them. `laser.py` is a transcription and
                  the only way a transcription can be checked is against the original.
  window          the window arithmetic: how many windows a length gives, that the stride
                  is `window_size // 2`, and that `t <= window_size` short-circuits to a
                  reward of exactly 0.0. The call site uses 10, not the signature's 20,
                  and getting that wrong halves the reported share of zero-scoring
                  rollouts.
  sink            the mean + 2*sd rule, including the ddof=1 trap: torch's `.std()` is
                  unbiased and numpy's is not, and taking numpy's default moves the
                  threshold enough to change the set at the margin.
  slice           `response_query_slice` reproduces upstream's `desc_start:desc_end`,
                  which skips the first response token and drops the last. Neither looks
                  deliberate; both are inherited on purpose.
  reward          the combined reward gates MULTIPLICATIVELY on accuracy and format, so
                  an incorrect rollout carries no attention term at all. That is what
                  makes the within-group spread among correct rollouts, and not the spread
                  over all eight, the quantity the go/no-go turns on.
  geometry        `ring_agreement` against hand-built masks, and against
                  `sink_location.ring_set` -- the border the cross-check is against.
  stats           `within_group_sd`, `column_cv` and `decay_spearman` on inputs whose
                  answers are known by hand, including the degenerate cases that a
                  collected run will certainly contain (a constant alpha, a one-row block,
                  an empty sink set).
  probe           the storage round-trips ragged and optional fields, resume survives a
                  torn last line, and the pre-registered thresholds in the harness are the
                  ones the design document fixed.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import laser as LZ  # noqa: E402
import sink_location as SL  # noqa: E402

FAILURES = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# upstream, rewritten in torch. The reference the transcription is checked against.
# ---------------------------------------------------------------------------
def upstream_stability(attention, window_size=10, sensitivity=5.0, scale=0.1,
                       penalty=0.015):
    """`DataParallelPPOActor.compute_windowed_global_stability`, verbatim."""
    t = attention.shape[0]
    if t <= window_size:
        return torch.tensor(0.0, device=attention.device)
    windows = attention.unfold(dimension=0, size=window_size, step=window_size // 2)
    window_means = windows.mean(dim=-1)
    target_level = window_means.max().detach()
    ratios = window_means / (target_level + 1e-12)
    per_window_rewards = torch.exp(-sensitivity * (1.0 - ratios)) - penalty
    return per_window_rewards.sum() * scale


def upstream_stability_early(attention, window_size=10, sensitivity=5.0, scale=0.1,
                             penalty=0.015, decay_rate=1.5, weight_mode="exp"):
    """`...compute_windowed_global_stability_early_weighted`, verbatim."""
    t = attention.shape[0]
    if t <= window_size:
        return torch.tensor(0.0, device=attention.device)
    windows = attention.unfold(dimension=0, size=window_size, step=window_size // 2)
    window_means = windows.mean(dim=-1)
    num_windows = window_means.shape[0]
    target_level = window_means.max().detach()
    ratios = window_means / (target_level + 1e-12)
    per_window_rewards = torch.exp(-sensitivity * (1.0 - ratios)) - penalty
    idx = torch.arange(num_windows, device=attention.device, dtype=per_window_rewards.dtype)
    denom = float(max(num_windows - 1, 1))
    if weight_mode == "exp":
        raw_weights = torch.exp(-decay_rate * (idx / denom))
    elif weight_mode == "linear":
        raw_weights = torch.clamp(1.0 - decay_rate * (idx / denom), min=1e-6)
    else:
        raw_weights = (idx + 1.0).pow(-decay_rate)
    weights = raw_weights * (num_windows / (raw_weights.sum() + 1e-12))
    return (per_window_rewards * weights).sum() * scale


def upstream_suppression(attention, sink_indices, tau=0.9, beta=0.5, scale=1.0):
    """`...compute_sink_suppression_reward`, verbatim."""
    _num_output_tokens, num_visual_tokens = attention.shape
    sink_mask = torch.zeros(num_visual_tokens, dtype=torch.bool, device=attention.device)
    if sink_indices.numel() > 0:
        sink_indices = sink_indices.clamp(0, num_visual_tokens - 1)
        sink_mask[sink_indices] = True
    if sink_mask.sum() == 0:
        return torch.tensor(0.0, device=attention.device)
    sink_attention = attention[:, sink_mask].mean(dim=-1)
    total_attention = attention.mean(dim=-1)
    sink_ratio = sink_attention / (total_attention + 1e-10)
    excess = torch.clamp(sink_ratio - tau, min=0.0)
    return torch.exp(-beta * excess).mean() * scale


def upstream_sinks(a_bos):
    """The sink rule, at both call sites: `attn > attn.mean() + 2 * attn.std()`."""
    a = torch.as_tensor(a_bos, dtype=torch.float64)
    return a > a.mean() + 2 * a.std()


# ---------------------------------------------------------------------------
def test_transcription():
    print("\ntranscription -- against upstream's own torch lines")
    rng = np.random.default_rng(11)
    worst_v = worst_e = worst_s = 0.0
    for trial in range(60):
        t = int(rng.integers(11, 400))
        # Three regimes on purpose: a flat trajectory (every ratio 1, every window at the
        # peak), a decaying one (what Finding 1 claims), and noise.
        base = rng.random(t) * 1e-3
        alpha = {0: np.full(t, 4e-4), 1: np.linspace(1e-3, 1e-4, t), 2: base}[trial % 3]
        a = torch.tensor(alpha, dtype=torch.float64)
        worst_v = max(worst_v, abs(LZ.visual_grounding_reward(alpha)
                                   - float(upstream_stability(a))))
        worst_e = max(worst_e, abs(LZ.visual_grounding_reward(alpha, early_weighted=True)
                                   - float(upstream_stability_early(a))))
    check("R_vis reproduces upstream over 60 trajectories", worst_v < 1e-12,
          f"max |delta| {worst_v:.2e}")
    check("R_vis early-weighted reproduces upstream", worst_e < 1e-12,
          f"max |delta| {worst_e:.2e}")

    for _ in range(40):
        T, m = int(rng.integers(2, 90)), int(rng.integers(20, 300))
        A = rng.random((T, m)) * 1e-3
        a_bos = rng.random(m) ** 6
        s = LZ.sink_mask(a_bos)
        idx = torch.tensor(np.flatnonzero(s), dtype=torch.long)
        worst_s = max(worst_s, abs(
            LZ.sink_suppression_reward(A, s)
            - float(upstream_suppression(torch.tensor(A, dtype=torch.float64), idx))))
    check("R_supp reproduces upstream over 40 blocks", worst_s < 1e-12,
          f"max |delta| {worst_s:.2e}")

    # An empty sink set is upstream's own short-circuit and is not rare: `mean + 2*sd`
    # selects nothing on a near-uniform column vector.
    flat = np.full(64, 0.015625)
    check("an empty sink set gives R_supp exactly 0.0",
          LZ.sink_mask(flat).sum() == 0
          and LZ.sink_suppression_reward(rng.random((5, 64)), LZ.sink_mask(flat)) == 0.0)

    check("the constants are the CALL SITE's, not the signature's",
          (LZ.WINDOW_SIZE, LZ.STABILITY_PENALTY, LZ.TAU, LZ.SUPP_BETA) == (10, 0.015, 0.9, 0.5),
          f"window {LZ.WINDOW_SIZE}, penalty {LZ.STABILITY_PENALTY}, tau {LZ.TAU}")
    check("the omegas are openr1_verl's",
          (LZ.OMEGA_VIS, LZ.OMEGA_SUPP, LZ.FORMAT_WEIGHT) == (0.05, 0.1, 0.3))


def test_window():
    print("\nwindow -- the arithmetic the short-circuit turns on")
    for t in (1, 5, 10):
        check(f"t={t} <= window short-circuits to None", LZ.windows_of(np.zeros(t)) is None)
    check("t = window + 1 gives exactly one window",
          LZ.windows_of(np.zeros(11)) is not None and LZ.windows_of(np.zeros(11)).size == 1)
    for t in (11, 12, 15, 20, 21, 100, 511):
        got = LZ.windows_of(np.zeros(t))
        want = torch.zeros(t).unfold(0, LZ.WINDOW_SIZE, LZ.WINDOW_SIZE // 2).shape[0]
        if got is None or got.size != want:
            check(f"window count at t={t}", False, f"{None if got is None else got.size} vs {want}")
            break
    else:
        check("the window count matches torch's unfold at every length tried", True,
              "t in 11..511")
    a = np.arange(30, dtype=float)
    w = LZ.windows_of(a)
    check("the stride is window_size // 2",
          np.allclose(w[:2], [np.arange(0, 10).mean(), np.arange(5, 15).mean()]),
          f"{w[0]:.1f}, {w[1]:.1f}")
    # The tolerances below are 1e-6 and not 1e-12, and the reason is upstream's own
    # `target_level + 1e-12`. At a realistic alpha -- a per-patch mean around 5e-4 -- that
    # epsilon is a relative 2e-9, so a flat trajectory's ratio is 1 - 2e-9 rather than 1
    # and R_vis picks up a level dependence of order 1e-8 reward units. It is real, it is
    # inherited deliberately, and it is six orders of magnitude below the 0.010 the
    # go/no-go thresholds are set at. Asserting equality to 1e-12 would only be asserting
    # that we had removed it.
    flat = np.full(200, 3e-4)
    n_w = LZ.windows_of(flat).size
    want = n_w * (np.exp(0.0) - LZ.STABILITY_PENALTY) * LZ.STABILITY_SCALE
    check("a perfectly flat trajectory scores n_windows * (1 - penalty) * scale",
          abs(LZ.visual_grounding_reward(flat) - want) < 1e-6,
          f"{LZ.visual_grounding_reward(flat):.4f} over {n_w} windows, "
          f"{abs(LZ.visual_grounding_reward(flat) - want):.1e} off the ideal")
    d_level = abs(LZ.visual_grounding_reward(flat)
                  - LZ.visual_grounding_reward(flat * 1000.0))
    check("R_vis is blind to the LEVEL of a flat trajectory", d_level < 1e-6,
          f"a 1000x brighter map moves it by {d_level:.1e} -- design doc §1, and the "
          f"residue is upstream's 1e-12 epsilon, not magnitude sensitivity")
    check("early weighting keeps the magnitude comparable",
          abs(LZ.early_weights(7).sum() - 7) < 1e-9
          and abs(LZ.visual_grounding_reward(flat, early_weighted=True)
                  - LZ.visual_grounding_reward(flat)) < 1e-9,
          "weights sum to n_windows, so a flat trajectory is unmoved")
    decaying = np.linspace(1.0, 0.1, 200)
    check("early weighting punishes late decay less than uniform weighting does",
          LZ.visual_grounding_reward(decaying, early_weighted=True)
          > LZ.visual_grounding_reward(decaying),
          f"{LZ.visual_grounding_reward(decaying, early_weighted=True):.4f} vs "
          f"{LZ.visual_grounding_reward(decaying):.4f}")


def test_sink():
    print("\nsink -- mean + 2*sd, and the ddof trap")
    rng = np.random.default_rng(3)
    worst = 0
    for _ in range(50):
        v = rng.random(int(rng.integers(8, 400))) ** 5
        worst = max(worst, int((LZ.sink_mask(v) != upstream_sinks(v).numpy()).sum()))
    check("sink_mask agrees with the torch rule patch for patch", worst == 0,
          "50 vectors, no disagreement")
    # Not asserted as "always different" -- on most vectors the two agree, which is
    # exactly why the trap is worth a test. Search for a vector where they do not, so the
    # detail line shows a real disagreement rather than asserting one in the abstract.
    diff = None
    for _ in range(4000):
        v = rng.random(24) ** 3
        n1 = int((v > v.mean() + 2 * v.std(ddof=1)).sum())
        n0 = int((v > v.mean() + 2 * v.std(ddof=0)).sum())
        if n1 != n0:
            diff = (n1, n0)
            break
    check("ddof=1 vs ddof=0 can select different patches", diff is not None,
          f"found a vector where ddof=1 selects {diff[0]} and ddof=0 selects {diff[1]}"
          if diff else "no disagreement found in 4000 draws")
    check("a single-element vector selects nothing rather than raising",
          LZ.sink_mask(np.array([1.0])).sum() == 0)
    # A delta column: one patch far above the rest is what the rule is FOR.
    d = np.full(100, 1e-4)
    d[7] = 1.0
    check("a lone spike is selected", LZ.sink_mask(d).sum() == 1 and LZ.sink_mask(d)[7])
    # ... and a broad plateau is not, which is why an empty S is a real outcome.
    p = np.full(100, 1e-4)
    p[:40] = 2e-4
    check("a broad plateau selects nothing", LZ.sink_mask(p).sum() == 0,
          f"{int(LZ.sink_mask(p).sum())} selected")


def test_slice():
    print("\nslice -- upstream's desc_start:desc_end, off-by-ones included")
    for n in (2, 3, 10, 137, 512):
        s = LZ.response_query_slice(n)
        # desc_start = prompt+1 and desc_end = prompt+n-1, both absolute.
        check(f"n_valid={n} gives response indices 1..{n - 2}",
              (s.start, s.stop) == (1, max(1, n - 1)), f"{s.start}:{s.stop}")
    A = np.arange(20 * 4, dtype=float).reshape(20, 4)
    check("the first response row is skipped and the last dropped",
          np.array_equal(A[LZ.response_query_slice(20)], A[1:19]),
          "18 of 20 rows survive")
    # desc_end == desc_start at n_valid = 2, and upstream's hooked path skips the sample
    # outright (`if desc_end <= desc_start: continue`). Zero rows is the faithful answer,
    # and it flows through to R_vis == 0 via the short-circuit rather than to a crash.
    check("a 2-token response leaves NO query rows, as upstream's guard implies",
          A[LZ.response_query_slice(2)].shape[0] == 0)
    empty = A[LZ.response_query_slice(2)]
    check("an empty query block scores zero rather than raising",
          LZ.visual_grounding_reward(LZ.alpha_per_step(empty)) == 0.0
          and LZ.sink_suppression_reward(empty, np.ones(4, dtype=bool)) == 0.0)


def test_reward():
    print("\nreward -- the gate is multiplicative on accuracy AND format")
    check("an incorrect rollout carries no attention term",
          LZ.total_reward(0.0, 1.0, 5.0, 1.0) == LZ.total_reward(0.0, 1.0, 0.0, 0.0),
          f"{LZ.total_reward(0.0, 1.0, 5.0, 1.0):.4f}")
    check("a malformed rollout carries no attention term",
          LZ.total_reward(1.0, 0.0, 5.0, 1.0) == 1.0)
    got = LZ.total_reward(1.0, 1.0, 2.0, 0.8)
    want = 1.0 + 0.3 * 1.0 + 0.05 * 2.0 * 1.0 + 0.1 * 0.8 * 1.0
    check("the combined reward is openr1_verl's", abs(got - want) < 1e-12, f"{got:.4f}")
    check("tau = 0.9 penalises a map that gives the sinks their FAIR share",
          LZ.sink_suppression_reward(np.ones((5, 20)), np.array([True] * 4 + [False] * 16))
          < 1.0,
          f"{LZ.sink_suppression_reward(np.ones((5, 20)), np.array([True] * 4 + [False] * 16)):.4f}"
          "  (a uniform map scores exp(-0.5*0.1) = 0.9512, not 1.0)")


def test_geometry():
    print("\ngeometry -- S against the border the cross-check is against")
    gh, gw = 10, 16
    ring = SL.ring_set(gh, gw).reshape(-1)
    r = LZ.ring_agreement(ring, gh, gw)
    check("S == the ring gives Jaccard 1.0 and enrichment 1/area",
          abs(r["jaccard"] - 1.0) < 1e-12
          and abs(r["enrichment"] - 1.0 / ring.mean()) < 1e-9,
          f"jaccard {r['jaccard']:.3f}, enrichment {r['enrichment']:.3f}, "
          f"area {r['ring_area']:.3f}")
    interior = ~ring
    r2 = LZ.ring_agreement(interior, gh, gw)
    check("S disjoint from the ring gives Jaccard 0 and enrichment 0",
          r2["jaccard"] == 0.0 and r2["enrichment"] == 0.0)
    half = ring.copy()
    half[np.flatnonzero(ring)[: int(ring.sum()) // 2]] = False
    r3 = LZ.ring_agreement(half, gh, gw)
    check("a subset of the ring is fully on the ring but has a small Jaccard",
          abs(r3["sink_on_ring"] - 1.0) < 1e-12 and r3["jaccard"] < 0.6,
          f"on_ring {r3['sink_on_ring']:.3f}, jaccard {r3['jaccard']:.3f}")
    check("the ring's area is (2gh+2gw-4)/(gh*gw)",
          abs(r["ring_area"] - SL.ring_area_frac(gh, gw)) < 1e-12,
          f"{r['ring_area']:.4f}")
    try:
        LZ.ring_agreement(np.zeros(7, dtype=bool), gh, gw)
        check("a sink mask that does not match the grid is refused", False)
    except ValueError:
        check("a sink mask that does not match the grid is refused", True)


def test_stats():
    print("\nstats -- the helpers the verdict is read off")
    vals = [1.0, 3.0, 10.0, 10.0, 5.0]
    groups = ["a", "a", "b", "b", "c"]
    sd, n = LZ.within_group_sd(vals, groups)
    # group a: sd([1,3], ddof=1) = sqrt(2) = 1.4142; group b: 0; group c: one element,
    # excluded. GRPO's advantage divides by a group's own spread, so a singleton group
    # contributes no gradient and must not contribute to the mean either.
    check("within_group_sd averages per group and drops singletons",
          abs(sd - (np.sqrt(2.0) + 0.0) / 2) < 1e-12 and n == 2, f"{sd:.4f} over {n} groups")
    sd2, n2 = LZ.within_group_sd([1.0, float("nan"), 2.0], ["a", "a", "a"])
    check("a NaN is dropped rather than propagated",
          np.isfinite(sd2) and n2 == 1, f"{sd2:.4f}")
    sd3, n3 = LZ.within_group_sd([], [])
    check("no groups gives NaN and a count of zero", not np.isfinite(sd3) and n3 == 0)

    A = np.tile(np.array([[1.0, 2.0, 3.0]]), (12, 1))
    check("a perfectly query-invariant column has CV 0",
          abs(LZ.column_cv(A, np.array([True, True, True]))) < 1e-12)
    rng = np.random.default_rng(5)
    B = rng.random((200, 3))
    cv = LZ.column_cv(B, np.array([True, False, False]))
    # uniform(0,1): sd/mean = (1/sqrt(12)) / 0.5 = 0.577
    check("a uniform column has CV near 0.577", abs(cv - 0.577) < 0.08, f"{cv:.3f}")
    check("an empty sink set gives NaN, not 0", not np.isfinite(LZ.column_cv(B, np.zeros(3, bool))))
    check("a one-row block gives NaN", not np.isfinite(LZ.column_cv(B[:1], np.array([True, False, False]))))

    check("decay_spearman is -1 on a strictly falling trajectory",
          abs(LZ.decay_spearman(np.linspace(1, 0, 50)) + 1.0) < 1e-9)
    check("decay_spearman is +1 on a strictly rising one",
          abs(LZ.decay_spearman(np.linspace(0, 1, 50)) - 1.0) < 1e-9)
    check("decay_spearman is NaN on a constant trajectory",
          not np.isfinite(LZ.decay_spearman(np.full(50, 0.3))))
    check("decay_spearman is NaN on a trajectory too short to rank",
          not np.isfinite(LZ.decay_spearman(np.array([1.0, 2.0]))))


def test_alpha():
    print("\nalpha -- the per-step quantity R_vis is computed on")
    A = np.zeros((4, 10))
    A[:, 0] = 1.0                       # one patch takes everything
    A[:, 1:] = 0.01
    s = np.zeros(10, dtype=bool)
    s[0] = True
    with_sink = LZ.alpha_per_step(A, None)
    without = LZ.alpha_per_step(A, s)
    check("dropping a sink RAISES alpha only if the sink was below average",
          without[0] < with_sink[0],
          f"{with_sink[0]:.4f} -> {without[0]:.4f}: the dominant patch left the numerator")
    check("alpha is a MEAN, so it scales with 1/n_patches not with n_patches",
          abs(LZ.alpha_per_step(np.full((3, 100), 0.01))[0] - 0.01) < 1e-12)
    check("an all-sink set gives alpha 0 rather than a division by zero",
          np.all(LZ.alpha_per_step(A, np.ones(10, dtype=bool)) == 0.0))
    r = LZ.sink_ratio_per_step(A, s)
    # sink mean 1.0, overall mean (1 + 9*0.01)/10 = 0.109 -> ratio 9.174. The tolerance is
    # 1e-6 because upstream divides by `total + 1e-10`, which at this scale is a relative
    # 1e-9 -- the same inherited-epsilon story as the window test above.
    check("sink_ratio_per_step is an enrichment over ALL visual tokens",
          abs(r[0] - 1.0 / ((1.0 + 9 * 0.01) / 10)) < 1e-6, f"{r[0]:.3f}")


def test_probe():
    print("\nprobe -- storage, resume, and the pre-registered thresholds")
    spec = importlib.util.spec_from_file_location("_t_laser_probe",
                                                  os.path.join(ROOT, "laser_probe.py"))
    LP = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(LP)
    except Exception as exc:                 # a missing GPU dep is not this test's problem
        check("laser_probe imports", False, f"{type(exc).__name__}: {exc}")
        return
    check("laser_probe imports", True)

    with tempfile.TemporaryDirectory() as d:
        sink = LP.Sink(d, "collect", 0, flush_every=2)
        rng = np.random.default_rng(1)
        want = {}
        for i in range(5):
            # ragged on purpose: response lengths differ, and only some units carry
            # `sinks`, which is the shape of bug that silently dropped a field once before
            T = int(rng.integers(3, 30))
            arrays = {"A0": rng.random((T, 12)).astype(np.float16)}
            if i % 2 == 0:
                arrays["sinks0"] = (rng.random(12) > 0.7).astype(np.int32)
            want[f"k{i}"] = arrays
            sink.write(f"k{i}", {"key": f"k{i}", "set": "s", "rollouts": []}, arrays)
        sink.close()
        meta, arrays = LP.read_stage(d, "collect")
        check("every unit round-trips", len(meta) == 5 and len(arrays) == 5,
              f"{len(meta)} rows, {len(arrays)} array sets")
        ok = all(np.array_equal(arrays[k][f], v[f]) for k, v in want.items() for f in v)
        check("ragged and optional fields round-trip exactly", ok)
        check("an optional field is not dropped from parts it shares with units lacking it",
              all("sinks0" in arrays[f"k{i}"] for i in (0, 2, 4))
              and all("sinks0" not in arrays[f"k{i}"] for i in (1, 3)))

        with open(os.path.join(d, "collect_shard0.jsonl"), "a") as fh:
            fh.write('{"key": "torn"')      # a killed job's last line
        sink2 = LP.Sink(d, "collect", 0)
        check("resume ignores a torn last line",
              sink2.done_prompts() == {f"k{i}" for i in range(5)},
              f"{sorted(sink2.done_prompts())}")
        sink2.close()

    rr = [{"group": "g1", "acc": 1.0, "format": 1.0, "r_vis": 1.0, "set": "s",
           "rollouts": None}]
    check("flatten pairs every rollout with its prompt",
          [r["group"] for r in LP.flatten([{"key": "g1", "set": "s", "rollouts": rr}])]
          == ["g1"])
    check("the harness thresholds are the design document's",
          LP.THRESHOLDS["T7_cv"] == 0.5 and LP.THRESHOLDS["T8_jaccard"] == 0.5
          and LP.THRESHOLDS["T5_sink_ratio"] == 1.5,
          "T7 is sink-location's own pre-registered 0.5")
    check("the group size is the trainer's",
          (LP.TRAINER_ROLLOUTS, LP.TRAINER_TEMPERATURE, LP.TRAINER_MAX_COMPLETION)
          == (8, 1.0, 512), "run_grpo.sh")
    check("boot_mean drops NaNs rather than zeroing them",
          LP.boot_mean([1.0, float("nan"), 1.0, 1.0])[0] == 1.0)
    check("boot_mean refuses to invent a CI from fewer than 3 values",
          not np.isfinite(LP.boot_mean([1.0, 2.0])[0]))
    check("pearson is NaN when a side is constant",
          not np.isfinite(LP.pearson([1, 1, 1, 1], [1, 2, 3, 4])))


def main():
    print("laser CPU checks")
    test_transcription()
    test_window()
    test_sink()
    test_slice()
    test_reward()
    test_geometry()
    test_stats()
    test_alpha()
    test_probe()
    print(f"\n{'ALL PASS' if not FAILURES else f'{len(FAILURES)} FAILED: {FAILURES}'}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
