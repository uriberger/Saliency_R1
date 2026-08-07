#!/usr/bin/env python
"""CPU checks for the flow intervention: the edit algebra, the carried scalars, the
deepstack re-seed, the grid and the report pairing. No GPU, no model.

The edit's whole claim is that it preserves row sums and each head's mass on the
eligible keys while moving only the split between union-carriers and other
image-carriers. Every one of those is checked against a naive reference here, because
none of them is visible in a GPU run's output.
"""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

spec = importlib.util.spec_from_file_location(
    "fi", Path(__file__).resolve().parent / "flow_intervene_probe.py")
FI = importlib.util.module_from_spec(spec)
sys.modules["fi"] = FI
spec.loader.exec_module(FI)

PASS, FAIL = [], []


def check(name, cond, extra=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{('   ' + extra) if extra else ''}")


def rand_attn(H, P, seed=0):
    """Causal row-stochastic attention, the shape a real softmax produces."""
    g = torch.Generator().manual_seed(seed)
    a = torch.rand(H, P, P, generator=g)
    a = a.masked_fill(torch.triu(torch.ones(P, P, dtype=torch.bool), 1)[None], 0.0)
    return a / a.sum(-1, keepdim=True)


# ---------------------------------------------------------------------------
def test_targets_normalise_per_row():
    print("\n[targets] T sums to 1 over q <= p, and never over q > p")
    P = 10
    u = torch.tensor([0., 0., 2., 0., 1., 0., 0., 3., 0., 0.])
    m = torch.tensor([0., 0., 1., 1., 1., 0., 0., 1., 0., 1.])
    rows = torch.tensor([4, 8])
    T, keep = FI.build_targets(u, m, rows)
    check("every kept row sums to 1", torch.allclose(T[keep].sum(-1),
                                                     torch.ones(int(keep.sum()))),
          str(T.sum(-1).tolist()))
    check("no mass beyond q = p", float(T[0, 5:].abs().sum()) == 0.0)
    check("row 4 splits 2:1 between q=2 and q=4",
          torch.allclose(T[0, [2, 4]], torch.tensor([2 / 3, 1 / 3])))
    check("row 8 also sees q=7, so 2:1:3 over q=2,4,7",
          torch.allclose(T[1, [2, 4, 7]], torch.tensor([2 / 6, 1 / 6, 3 / 6])))
    check("ineligible positions get zero (q=9 has m>0 but u=0; q=3 has u=0)",
          float(T[1, 3]) == 0.0 and float(T[1, 9]) == 0.0)

    # a position with union mass but m == 0 cannot exist physically, but if it did the
    # eligibility test -- not the union test -- must be what excludes it.
    u2 = torch.tensor([5., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    T2, keep2 = FI.build_targets(u2, m, rows)
    check("u>0 at an ineligible key is ignored, and the row is dropped",
          (not bool(keep2[0])) and float(T2[0].abs().sum()) == 0.0)


def test_edit_preserves_row_sum_and_mass():
    print("\n[edit] row sums and each head's eligible mass are exactly preserved")
    H, P = 3, 12
    a = rand_attn(H, P, seed=1)
    m = (torch.rand(P, generator=torch.Generator().manual_seed(2)) > 0.4).float()
    u = m * torch.rand(P, generator=torch.Generator().manual_seed(3))
    rows = torch.tensor([7, 9, 11])
    T, keep = FI.build_targets(u, m, rows)
    for alpha in (0.0, 0.25, 1.0):
        a2, mass = FI.edit_attention(a, rows, T, keep, m, alpha)
        check(f"alpha={alpha}: every row still sums to 1",
              torch.allclose(a2.sum(-1), torch.ones(H, P), atol=1e-5),
              f"max dev {float((a2.sum(-1) - 1).abs().max()):.2e}")
        sel = ((m > 0)[None, :] & (torch.arange(P)[None, :] <= rows[:, None]))
        after = (a2[:, rows, :] * sel[None]).sum(-1)
        check(f"alpha={alpha}: eligible mass unchanged per head",
              torch.allclose(after, mass, atol=1e-5),
              f"max dev {float((after - mass).abs().max()):.2e}")
        check(f"alpha={alpha}: untouched rows are bit-identical",
              torch.equal(a2[:, [0, 1, 2, 3], :], a[:, [0, 1, 2, 3], :]))
        check(f"alpha={alpha}: nothing lands above the diagonal",
              float(a2.masked_select(
                  torch.triu(torch.ones(P, P, dtype=torch.bool), 1)[None]).abs().max())
              == 0.0)


def test_edit_endpoints():
    print("\n[edit] alpha=0 is identity; alpha=1 is proportional to union content")
    H, P = 2, 9
    a = rand_attn(H, P, seed=4)
    m = torch.ones(P)
    u = torch.tensor([1., 0., 0., 3., 0., 0., 0., 0., 0.])
    rows = torch.tensor([6])
    T, keep = FI.build_targets(u, m, rows)
    a0, _ = FI.edit_attention(a, rows, T, keep, m, 0.0)
    check("alpha=0 leaves the tensor untouched", torch.allclose(a0, a, atol=0))
    a1, mass = FI.edit_attention(a, rows, T, keep, m, 1.0)
    got = a1[:, 6, :]
    want = mass[:, 0:1] * T[0][None, :]
    check("alpha=1 puts all eligible mass in proportion to u",
          torch.allclose(got, want, atol=1e-6))
    check("...which is a 1:3 split between q=0 and q=3",
          torch.allclose(got[:, 3] / got[:, 0], torch.full((H,), 3.0), atol=1e-4))
    check("keys with no union content get exactly nothing at alpha=1",
          float(got[:, [1, 2, 4, 5]].abs().max()) == 0.0)


def test_ineligible_keys_are_never_touched():
    print("\n[edit] text keys that never saw the image keep their weight exactly")
    H, P = 2, 10
    a = rand_attn(H, P, seed=5)
    m = torch.zeros(P)
    m[[1, 2, 3]] = 1.0                       # only these carry image content
    u = torch.zeros(P)
    u[2] = 1.0
    rows = torch.tensor([8])
    T, keep = FI.build_targets(u, m, rows)
    a1, _ = FI.edit_attention(a, rows, T, keep, m, 1.0)
    ineligible = [0, 4, 5, 6, 7, 8, 9]
    check("every ineligible column is bit-identical",
          torch.equal(a1[:, 8, ineligible], a[:, 8, ineligible]))
    check("the eligible mass all moved onto the one union carrier",
          torch.allclose(a1[:, 8, 2], a[:, 8, [1, 2, 3]].sum(-1), atol=1e-6))


def test_edit_rows_matches_full_edit_and_survives_bf16():
    print("\n[edit] the rows-only path equals the full-copy path, in fp32 and bf16")
    H, P = 4, 16
    a = rand_attn(H, P, seed=7)
    m = (torch.rand(P, generator=torch.Generator().manual_seed(8)) > 0.3).float()
    u = m * torch.rand(P, generator=torch.Generator().manual_seed(9))
    rows = torch.tensor([10, 11, 15])
    T, keep = FI.build_targets(u, m, rows)
    new, mass = FI.edit_rows(a, rows, T, keep, m, 0.7)
    full, mass2 = FI.edit_attention(a, rows, T, keep, m, 0.7)
    check("the two paths give identical rows", torch.allclose(full[:, rows, :], new,
                                                              atol=1e-6))
    check("and identical mass", torch.allclose(mass, mass2, atol=0))

    # The hook writes float32 rows back into a bf16 tensor. Row sums must survive that.
    ab = a.to(torch.bfloat16)
    nb, _ = FI.edit_rows(ab, rows, T, keep, m, 1.0)
    ab = ab.clone()
    ab[:, rows, :] = nb.to(ab.dtype)
    dev = float((ab.float().sum(-1) - 1).abs().max())
    check("bf16 write-back keeps row sums to bf16 precision", dev < 5e-2,
          f"max dev {dev:.2e}")
    check("the edit is computed in fp32 even from a bf16 input",
          nb.dtype == torch.float32, str(nb.dtype))

    # Disjoint step spans: editing one step's rows must not disturb another's.
    r2 = torch.tensor([4, 5])
    T2, k2 = FI.build_targets(u, m, r2)
    n2, _ = FI.edit_rows(a, r2, T2, k2, m, 1.0)
    both = a.clone()
    both[:, rows, :] = new.to(both.dtype)
    both[:, r2, :] = n2.to(both.dtype)
    check("two disjoint step edits compose without interfering",
          torch.allclose(both[:, rows, :], new, atol=1e-6)
          and torch.allclose(both[:, r2, :], n2, atol=1e-6))


def test_propagate_matches_naive():
    print("\n[propagate] the vectorised recursion equals a per-column loop")
    P, K = 7, 3
    g = torch.Generator().manual_seed(6)
    w = torch.rand(P, P, generator=g)
    w = w.masked_fill(torch.triu(torch.ones(P, P, dtype=torch.bool), 1), 0.0)
    w = w / w.sum(-1, keepdim=True)
    X = torch.rand(P, K, generator=g)
    got = FI.propagate(X, w, 0.5)
    want = torch.stack([0.5 * (w @ X[:, k]) + 0.5 * X[:, k] for k in range(K)], dim=1)
    check("columns propagate independently and identically",
          torch.allclose(got, want, atol=1e-6))
    check("a row-stochastic w cannot create mass",
          float(got.sum(0).max()) <= float(X.sum(0).max()) + 1e-5)


def test_reseed_only_at_deepstack_layers():
    print("\n[deepstack] the initial condition is added back at layers 0,1,2 only")
    X = torch.ones(4, 2)
    X0 = torch.full((4, 2), 3.0)
    check("layer 1 is a re-seed layer", torch.allclose(
        FI.reseed(X, X0, 1, {0, 1, 2}), X + X0))
    check("layer 5 is not", torch.allclose(FI.reseed(X, X0, 5, {0, 1, 2}), X))
    check("an empty set never re-seeds", torch.allclose(FI.reseed(X, X0, 0, set()), X))

    class Cfg:
        class vision_config:
            deepstack_visual_indexes = [8, 16, 24]

    class M:
        config = Cfg

    got = FI.deepstack_layers_of(M())
    check("LM layers are range(len(indexes)), NOT the vision indexes themselves",
          got == {0, 1, 2}, f"got {sorted(got)}")


def test_init_columns_layout():
    print("\n[columns] column 0 is m, then (box, roll) per step")
    img = torch.tensor([2, 3, 4, 5])
    b0 = torch.tensor([1., 0., 0., 0.])
    b1 = torch.tensor([0., 0., 1., 1.])
    r0 = torch.tensor([0., 1., 0., 0.])
    r1 = torch.tensor([1., 1., 0., 0.])
    X0 = FI.init_columns(8, img, [b0, b1], [r0, r1], torch.device("cpu"))
    check("shape is [P, 1 + 2*steps]", tuple(X0.shape) == (8, 5), str(tuple(X0.shape)))
    check("m is 1 at every image token and 0 elsewhere",
          torch.equal(X0[:, 0], torch.tensor([0., 0., 1., 1., 1., 1., 0., 0.])))
    check("step 0's box lands in column 1", torch.equal(X0[img, 1], b0))
    check("step 0's roll lands in column 2", torch.equal(X0[img, 2], r0))
    check("step 1's box lands in column 3", torch.equal(X0[img, 3], b1))
    check("no union column exceeds m anywhere",
          bool(torch.all(X0[:, 1:] <= X0[:, :1] + 1e-9)))


def test_rolled_mask_matches_area():
    print("\n[control] the rolled union has the same area and a different place")
    gh, gw = 6, 8
    m = torch.zeros(gh, gw)
    m[1:3, 2:5] = 1.0
    flat = m.reshape(-1)
    moved = 0
    for s in range(20):
        r = FI.rolled_mask(flat, gh, gw, np.random.default_rng(s))
        check_area = float(r.sum()) == float(flat.sum())
        if not check_area:
            check("area preserved under every roll", False, f"seed {s}")
            return
        moved += int(not torch.equal(r, flat))
    check("area preserved under every roll", True, f"{float(flat.sum()):.0f} patches")
    check("and the roll actually moves it most of the time", moved >= 15,
          f"{moved}/20 offsets differ")


def test_grid_has_one_baseline_per_cutoff():
    print("\n[grid] alpha=0 is a per-cutoff baseline, not a per-condition one")

    class A:
        cutoffs, alphas, conditions = "8,35", "0.5,1.0", "box,roll"

    g = FI.build_grid(A())
    base = [x for x in g if x[1] == "base"]
    check("one baseline per cutoff", len(base) == 2 and {b[0] for b in base} == {8, 35},
          str(base))
    check("no baseline carries a nonzero alpha", all(b[2] == 0.0 for b in base))
    check("2 cutoffs x 2 conditions x 2 alphas + 2 baselines = 10", len(g) == 10,
          str(len(g)))
    check("a bad condition is rejected, not silently dropped",
          _raises(lambda: FI.build_grid(type("B", (), {
              "cutoffs": "8", "alphas": "1.0", "conditions": "box,shape"})())))


def _raises(fn):
    try:
        fn()
    except SystemExit:
        return True
    except Exception:
        return False
    return False


def test_case_image_unwraps_the_record():
    print("\n[images] load_case_images returns records, not bare images")

    class FakePIL:
        pass

    pil = FakePIL()
    imgs = {7: {"row_index": 7, "dataset": "d", "question": "q", "gt_answer": "a",
                "image": pil}}
    check("the record's image is unwrapped", FI.case_image(imgs, 7) is pil)
    check("a missing row gives None, not a KeyError", FI.case_image(imgs, 8) is None)
    check("a numpy/str row index still resolves", FI.case_image(imgs, "7") is pil)
    check("a bare image is passed through unchanged",
          FI.case_image({3: pil}, 3) is pil)
    check("what reaches the processor is never a dict",
          not isinstance(FI.case_image(imgs, 7), dict))


def test_manipulation_check():
    print("\n[check] ushare/rshare read the step's own rows of the right columns")
    X = torch.zeros(6, 5)
    X[:, 0] = 2.0                                   # m = 2 everywhere
    X[4:6, 1] = 1.0                                 # step 0 box  -> ushare 0.5
    X[4:6, 2] = 0.5                                 # step 0 roll -> rshare 0.25
    steps = [{"rows": torch.tensor([4, 5])}]
    got = FI.manipulation_check(X, steps)
    check("ushare is u/m over the step's rows", abs(got["ushare"] - 0.5) < 1e-6,
          f"{got['ushare']:.4f}")
    check("rshare uses the roll column", abs(got["rshare"] - 0.25) < 1e-6,
          f"{got['rshare']:.4f}")


def test_report_pairs_and_recovers_a_planted_effect():
    print("\n[report] box-roll is paired per case, and a planted gap is recovered")
    rng = np.random.default_rng(11)
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        (out / "results").mkdir()
        with (out / "results" / "shard00.jsonl").open("w") as fh:
            for ri in range(120):
                lvl = rng.normal() * 5.0            # per-case level the pairing removes
                for cut in (8, 35):
                    fh.write(json.dumps({"row_index": ri, "cutoff": cut, "kind": "base",
                                         "alpha": 0.0, "logp_gold": lvl,
                                         "ushare": 0.30, "rshare": 0.30,
                                         "n_gold": 3, "top1_id": 1,
                                         "first_correct": 1}) + "\n")
                    gap = 0.40 if cut == 35 else 0.0
                    fh.write(json.dumps({"row_index": ri, "cutoff": cut, "kind": "box",
                                         "alpha": 1.0, "logp_gold": lvl + gap,
                                         "ushare": 0.55, "rshare": 0.30,
                                         "n_gold": 3, "top1_id": 1,
                                         "first_correct": 1}) + "\n")
                    fh.write(json.dumps({"row_index": ri, "cutoff": cut, "kind": "roll",
                                         "alpha": 1.0, "logp_gold": lvl,
                                         "ushare": 0.30, "rshare": 0.52,
                                         "n_gold": 3, "top1_id": 1,
                                         "first_correct": 1}) + "\n")

        class A:
            out_dir = str(out)

        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            FI.report(A())
        txt = buf.getvalue()
        check("both cutoffs are reported", "   8   1.00" in txt and "  35   1.00" in txt)
        line35 = [l for l in txt.splitlines() if l.strip().startswith("35 ")][0]
        line8 = [l for l in txt.splitlines() if l.strip().startswith("8 ")][0]
        check("the planted +0.40 gap is recovered at cutoff 35", "+0.40000" in line35,
              line35.strip())
        check("the no-gap cutoff reads ~0", "+0.00000" in line8, line8.strip())
        import re
        lo, hi = (float(x) for x in
                  re.search(r"\[\s*([-+][\d.]+),\s*([-+][\d.]+)\]", line35).groups())
        check("per-case level noise is removed by the pairing, so the CI excludes 0",
              lo > 0 and hi > 0, f"CI [{lo:+.5f}, {hi:+.5f}]")
        lo8, hi8 = (float(x) for x in
                    re.search(r"\[\s*([-+][\d.]+),\s*([-+][\d.]+)\]", line8).groups())
        check("and the no-gap cutoff's CI contains 0", lo8 <= 0 <= hi8,
              f"CI [{lo8:+.5f}, {hi8:+.5f}]")
        check("the manipulation columns show box moving u and roll moving r",
              "+0.2500" in line35 and "+0.2200" in line35, line35.strip())


def test_report_survives_a_torn_line():
    print("\n[report] a half-written record from a full filesystem is skipped")
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        (out / "results").mkdir()
        with (out / "results" / "shard00.jsonl").open("w") as fh:
            for ri in range(20):
                for kind, al, lp in (("base", 0.0, 0.0), ("box", 1.0, 0.5),
                                     ("roll", 1.0, 0.1)):
                    fh.write(json.dumps({"row_index": ri, "cutoff": 35, "kind": kind,
                                         "alpha": al, "logp_gold": lp, "ushare": 0.3,
                                         "rshare": 0.3}) + "\n")
            fh.write('{"row_index": 99, "cutoff": 35, "kind": "box", "alph')

        class A:
            out_dir = str(out)

        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            FI.report(A())
        check("the torn line does not crash the report",
              "20 cases" in buf.getvalue(), buf.getvalue().splitlines()[0])


def main():
    test_targets_normalise_per_row()
    test_edit_preserves_row_sum_and_mass()
    test_edit_endpoints()
    test_ineligible_keys_are_never_touched()
    test_edit_rows_matches_full_edit_and_survives_bf16()
    test_propagate_matches_naive()
    test_reseed_only_at_deepstack_layers()
    test_init_columns_layout()
    test_rolled_mask_matches_area()
    test_grid_has_one_baseline_per_cutoff()
    test_case_image_unwraps_the_record()
    test_manipulation_check()
    test_report_pairs_and_recovers_a_planted_effect()
    test_report_survives_a_torn_line()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print(f"  FAILED: {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
