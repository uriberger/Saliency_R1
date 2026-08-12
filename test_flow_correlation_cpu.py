#!/usr/bin/env python
"""CPU checks for flow_correlation_probe: the rollout algebra, the wnorm edge weights,
the increment, and the report.

Nothing here touches a GPU or loads a model -- that is what a real shard is for. What
it does cover is every place a silent sign or index error would corrupt all downstream
numbers while raising nothing:

  * the rollout preserves the "fraction traceable to the image" invariant (mass in
    [0, 1] at every layer, exactly 1 at an image position before any mixing);
  * a fixed point: with attention that is pure self-attention the map never changes,
    whatever alpha, so the residual term is wired the right way round;
  * `edge_weights_wnorm`'s Gram expansion equals the naive || sum_h a W_O^h v^h ||
    it replaces -- the expansion is the only nontrivial algebra in the file;
  * the increment is the step's mean map minus the pre-step map, at the final layer;
  * `report` recovers a planted correlation and holds it on the even-row half;
  * the union-decile table reads a union-dependent column as falling and a flat one as
    flat, and `--max-union` reaches the Bonferroni threshold rather than the tables
    alone -- dropping steps drops completions, and completions are the effective n.

    python test_flow_correlation_cpu.py
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


FC = _load("_t_flow_corr", "flow_correlation_probe.py")

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'ok  ' if cond else 'FAIL'} {name}" + (f"   {detail}" if detail else ""))


def row_stochastic(p, gen):
    """A causal, row-stochastic attention matrix, as a real softmax row would be."""
    x = torch.rand(p, p, generator=gen)
    x = x.tril()
    return x / x.sum(-1, keepdim=True)


# ---------------------------------------------------------------------------
def test_mass_invariant():
    print("\n[rollout] mass invariant")
    gen = torch.Generator().manual_seed(0)
    p, m = 40, 7
    img = torch.arange(5, 5 + m)
    sal = FC.init_sal(p, img)
    check("image positions start at mass 1",
          torch.allclose(sal[img].sum(-1), torch.ones(m)))
    check("every other position starts at 0",
          float(sal.sum()) == float(m))
    check("image position j is one-hot at j",
          bool((sal[img].argmax(-1) == torch.arange(m)).all()))

    worst = 0.0
    for _ in range(12):
        w = row_stochastic(p, gen)
        sal = FC.rollout_update(sal, w, 0.5)
        worst = max(worst, float(sal.sum(-1).max()))
        if float(sal.min()) < 0:
            break
    check("mass never exceeds 1 over 12 layers", worst <= 1.0 + 1e-5, f"max {worst:.6f}")
    check("saliency stays non-negative", float(sal.min()) >= 0.0)
    check("pre-image positions stay at exactly 0",
          float(sal[:5].abs().max()) == 0.0)


def test_residual_fixed_point():
    print("\n[rollout] self-attention is a fixed point")
    p, m = 12, 4
    img = torch.arange(2, 2 + m)
    sal0 = FC.init_sal(p, img)
    eye = torch.eye(p)
    for alpha in (0.0, 0.5, 1.0):
        sal = sal0.clone()
        for _ in range(5):
            sal = FC.rollout_update(sal, eye, alpha)
        check(f"identity attention leaves the map unchanged (alpha={alpha})",
              torch.allclose(sal, sal0))
    # alpha=1 with no residual must equal a plain matrix product
    gen = torch.Generator().manual_seed(1)
    w1, w2 = row_stochastic(p, gen), row_stochastic(p, gen)
    got = FC.rollout_update(FC.rollout_update(sal0, w1, 1.0), w2, 1.0)
    check("alpha=1 composes as W2 @ W1 @ sal0", torch.allclose(got, w2 @ w1 @ sal0, atol=1e-6))


class FakeAttn(torch.nn.Module):
    """Just the pieces edge_weights_wnorm reads off a Qwen3VLTextAttention."""

    def __init__(self, d_model, n_heads, n_kv, head_dim, gen):
        super().__init__()
        self.head_dim = head_dim
        self.num_key_value_groups = n_heads // n_kv
        self.v_proj = torch.nn.Linear(d_model, n_kv * head_dim, bias=False)
        self.o_proj = torch.nn.Linear(n_heads * head_dim, d_model, bias=False)
        with torch.no_grad():
            self.v_proj.weight.normal_(0, 0.3, generator=gen)
            self.o_proj.weight.normal_(0, 0.3, generator=gen)


def test_wnorm_matches_naive():
    print("\n[wnorm] Gram expansion == the naive norm")
    gen = torch.Generator().manual_seed(2)
    d_model, n_heads, n_kv, head_dim, p = 32, 6, 3, 8, 11
    mod = FakeAttn(d_model, n_heads, n_kv, head_dim, gen)
    hs = torch.randn(1, p, d_model, generator=gen)
    a = torch.rand(n_heads, p, p, generator=gen).tril()
    a = a / a.sum(-1, keepdim=True).clamp_min(1e-9)

    for chunk in (4, 256):
        got = FC.edge_weights_wnorm(mod, hs, a, chunk=chunk)

        # naive: build u[h,k] = W_O^h v^h_k explicitly and take the norm per (n,k)
        v = mod.v_proj(hs).view(1, p, n_kv, head_dim).transpose(1, 2)
        v = v.repeat_interleave(mod.num_key_value_groups, dim=1)[0]      # [H, P, dh]
        wo = mod.o_proj.weight.view(d_model, n_heads, head_dim)
        u = torch.stack([v[h] @ wo[:, h, :].T for h in range(n_heads)])  # [H, P, d]
        want = torch.zeros(p, p)
        for n in range(p):
            for k in range(p):
                want[n, k] = torch.linalg.norm((a[:, n, k, None] * u[:, k, :]).sum(0))
        check(f"matches the explicit per-pair norm (chunk={chunk})",
              torch.allclose(got, want, atol=1e-4),
              f"max diff {float((got - want).abs().max()):.2e}")

    check("non-negative", float(got.min()) >= 0.0)
    check("zero where attention is masked", float(got[0, 1:].abs().max()) < 1e-6)


def test_repeat_v_agrees():
    print("\n[wnorm] GQA expansion matches intervene_probe.repeat_v")
    gen = torch.Generator().manual_seed(3)
    v = torch.randn(1, 3, 5, 8, generator=gen)
    check("repeat_v == repeat_interleave on the head axis",
          torch.equal(FC.IV.repeat_v(v, 2), v.repeat_interleave(2, dim=1)))


def test_increment_definition():
    print("\n[increment] step mean minus the pre-step map, at every layer")
    n_layers, n_need, m = 3, 6, 4
    snaps = torch.arange(n_layers * n_need * m, dtype=torch.float32).reshape(
        n_layers, n_need, m)
    # positions 10,11,12,13,14,15 ; one step spanning 12..14 so a-1 == 11
    need = [10, 11, 12, 13, 14, 15]
    pos = {p: i for i, p in enumerate(need)}
    a, b = 12, 15
    mean = snaps[:, [pos[p] for p in range(a, b)]].mean(dim=1)
    inc = mean - snaps[:, pos[a - 1]]
    check("there is one increment per layer", tuple(inc.shape) == (n_layers, m),
          str(tuple(inc.shape)))
    for li in range(n_layers):
        want = snaps[li, [pos[12], pos[13], pos[14]]].mean(0) - snaps[li, pos[11]]
        check(f"inc at layer {li} subtracts that layer's pre-step map",
              torch.allclose(inc[li], want))
    check("the base column is the step's mean over its own tokens",
          torch.allclose(mean[-1], snaps[-1, [pos[12], pos[13], pos[14]]].mean(0)))


def test_partial_corr():
    print("\n[partial] a covariate-driven correlation is removed, a real one is not")
    rng = np.random.default_rng(7)
    n = 600
    z = rng.normal(size=n)                       # the confounder: e.g. union area
    noise = rng.normal(size=n)
    y = 1.2 * z + rng.normal(size=n)             # correctness driven only by z
    x_conf = 1.5 * z + 0.3 * rng.normal(size=n)  # a column that only tracks z
    x_real = 0.2 * noise + 0.9 * rng.normal(size=n) + 0.8 * (y - 1.2 * z)

    r_conf = np.corrcoef(x_conf, y)[0, 1]
    p_conf, n_conf = FC.partial_corr(x_conf, y, z[:, None])
    check("the confounded column correlates raw", abs(r_conf) > 0.4, f"r={r_conf:+.3f}")
    check("...and does not once z is held fixed", abs(p_conf) < 0.12,
          f"partial={p_conf:+.3f}")
    check("n is the finite-row count", n_conf == n)

    p_real, _ = FC.partial_corr(x_real, y, z[:, None])
    check("a genuine association survives partialling", p_real > 0.4,
          f"partial={p_real:+.3f}")

    xn = x_real.copy()
    xn[:590] = np.nan
    check("too few finite rows -> NaN", np.isnan(FC.partial_corr(xn, y, z[:, None])[0]))
    check("a constant covariate is dropped, not fatal",
          np.isfinite(FC.partial_corr(x_real, y, np.column_stack([z, np.ones(n)]))[0]))


def test_sample_columns():
    print("\n[report] the default column spread keeps the ends")
    got = FC.sample_columns(list(range(36)), want=9)
    check("first and last are always shown", got[0] == 0 and got[-1] == 35, str(got))
    check("it thins the middle", 9 <= len(got) <= 11, str(len(got)))
    check("a short list is returned whole", FC.sample_columns([0, 1, 2]) == [0, 1, 2])


def test_grad_maps_returns_numpy():
    """The leaves being differentiated require grad, and gradient-times-input
    multiplies by them. Without detaching, that product is a graph node and the final
    .numpy() raises -- which it did, on the first GPU run, after the whole model had
    loaded. A toy graph reproduces it in a millisecond."""
    print("\n[grad] gradient-times-input leaves the graph behind")
    gen = torch.Generator().manual_seed(4)
    m, d, vocab, s, prompt_len = 5, 8, 20, 7, 3

    class Leaves:
        pass

    leaves = Leaves()
    leaves.embeds = torch.randn(m, d, generator=gen).requires_grad_(True)
    leaves.deep = [torch.randn(m, d, generator=gen).requires_grad_(True) for _ in range(2)]
    w = torch.randn(d, vocab, generator=gen)
    logits = ((leaves.embeds.sum(0) + sum(t.sum(0) for t in leaves.deep)) @ w
              ).expand(s, vocab)[None]
    ids = torch.randint(0, vocab, (1, s + prompt_len), generator=gen)
    spans = [(prompt_len, prompt_len + 2), (prompt_len + 2, prompt_len + 5)]

    maps = FC.grad_maps(None, leaves, logits, ids, prompt_len, spans)
    check("returns [4, 1, n_steps, M]", maps.shape == (4, 1, len(spans), m), str(maps.shape))
    check("is finite", bool(np.isfinite(maps).all()))
    check("gnorm and gnorm_ds are non-negative", float(maps[:2].min()) >= 0.0)
    check("deepstack columns differ from the input-only ones",
          float(np.abs(maps[1] - maps[0]).max()) > 0)
    check("every column is defined for every step",
          bool((np.abs(maps).sum(-1) > 0).all()))


def test_glimpse_adapter():
    """The boundary between this probe and map 6, which is where the grad probe broke.

    Its first GPU run died on sample 1 because the adapter and the map disagreed about
    a type; the second died on sample 4 of 4. Both are the same failure mode -- a
    convention mismatch that only surfaces after an 8B model has loaded -- and both are
    a millisecond of arithmetic to rule out. `spans` here are ABSOLUTE; glimpse_map
    wants them chain-relative, and it must end up scoring exactly the tokens the grad
    column scores or the two columns of this screen are not comparable.
    """
    print("\n[glimpse] the span adapter, and the map -> [K,1,S,P] reshape")
    SV = FC._glimpse_module()
    check("map 6 is reachable without re-executing this probe",
          SV.glimpse_map.__name__ == "glimpse_map"
          and sys.modules.get("_sv_flow") is not None)

    prompt_len = 11
    # ids[0, p] == p, so a token's VALUE is its absolute position and the two
    # conventions can be compared directly rather than through a second index.
    ids = torch.arange(prompt_len + 29)[None]
    spans = [(prompt_len + 1, prompt_len + 4), (prompt_len + 6, prompt_len + 9)]
    gsteps = [("", a - prompt_len, b - prompt_len) for a, b in spans]   # scan_case

    # glimpse_map's own two lines, over what the adapter hands it.
    rows = [prompt_len + a - 1 + i for _t, a, b in gsteps for i in range(b - a)]
    targets = torch.cat([ids[0, prompt_len + a: prompt_len + b] for _t, a, b in gsteps])
    # and what the grad column scores from the same absolute spans (FC.grad_maps).
    grad_tokens = torch.cat([ids[0, a:b] for a, b in spans])

    check("glimpse scores the same tokens as grad, so the columns are comparable",
          torch.equal(targets, grad_tokens),
          f"{targets.tolist()} vs {grad_tokens.tolist()}")
    check("every row is the one that PREDICTS its token, p-1",
          rows == [p - 1 for p in grad_tokens.tolist()], str(rows))
    check("one row per scored token", len(rows) == sum(b - a for a, b in spans))

    # The negative control: this is the bug, and it must not pass silently.
    unshifted = [("", a, b) for a, b in spans]
    bad = torch.cat([ids[0, prompt_len + a: prompt_len + b] for _t, a, b in unshifted])
    check("forgetting the shift would score different tokens",
          not torch.equal(bad, grad_tokens))

    # [S, gh, gw] -> [K, 1, S, P]. The union mask is flattened with the same row-major
    # (gh, gw) reshape a few lines away in scan; a swapped pair would score every step
    # against a transposed union and still produce numbers that look like numbers.
    gh, gw, S = 3, 5, len(spans)
    g = np.arange(S * gh * gw, dtype=np.float32).reshape(S, gh, gw)
    maps = np.ascontiguousarray(g.reshape(1, 1, S, gh * gw), dtype=np.float32)
    check("reshapes to [1, 1, n_steps, gh*gw]", maps.shape == (1, 1, S, gh * gw))
    check("flattening is row-major over (gh, gw), as the mask's is",
          all(np.array_equal(maps[0, 0, si], g[si].reshape(-1)) for si in range(S)))
    check("a transposed grid would not agree, so the check has teeth",
          not np.array_equal(maps[0, 0, 0], g[0].T.reshape(-1)))


def test_report_recovers_planted_effect():
    print("\n[report] recovers a planted correlation, held out")
    rng = np.random.default_rng(0)
    n_comp, k = 400, 4
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "rollout_mean"
        (out / "scan").mkdir(parents=True)
        rows, cor, v2, au, step, uni, mass = [], [], [], [], [], [], []
        for c in range(n_comp):
            y = float(rng.integers(0, 2))
            for s in range(rng.integers(1, 4)):
                rows.append(c)
                cor.append(y)
                step.append(s)
                uni.append(float(rng.uniform(0.2, 0.8)))
                base = rng.normal(size=k)
                base[2] += 1.4 * y      # column 2 (the last non-inc one) is the PRIMARY
                v2.append(base)
                au.append(rng.normal(size=k))
                mass.append(rng.normal(size=k))
        np.savez_compressed(
            out / "scan" / "shard00.npz",
            v2=np.array(v2, dtype=np.float32), auroc=np.array(au, dtype=np.float32),
            row=np.array(rows), step=np.array(step),
            correct=np.array(cor, dtype=np.float32), union=np.array(uni, dtype=np.float32),
            mass=np.array(mass, dtype=np.float32),
            names=np.array(["L0", "L1", "L2", "inc2"]), map=np.array("rollout_mean"),
            alpha=np.array(0.5))

        class A:
            out_dir = str(out)
            all_columns = True
            max_union = 0.0
        FC.report(A())
        d = np.load(out / "corr.npz")
        r_all, _r_sel, r_out, r_par, lvl, lse = d["mean_in_v2_completion"]
        names = [str(s) for s in d["names"]]
        check("the planted column is the strongest", int(np.nanargmax(np.abs(r_all))) == 2,
              f"argmax={names[int(np.nanargmax(np.abs(r_all)))]}")
        check("the planted effect survives on the held-out half", r_out[2] > 0.3,
              f"held out r={r_out[2]:+.3f}")
        check("...and survives partialling, since nothing confounds it", r_par[2] > 0.3,
              f"partial r={r_par[2]:+.3f}")
        check("a null column does not", abs(r_out[0]) < 0.2, f"r={r_out[0]:+.3f}")
        check("names round-trip", names == ["L0", "L1", "L2", "inc2"])
        rs = d["auroc_step"][0]
        check("the auroc columns stay null", float(np.nanmax(np.abs(rs))) < 0.2)
        check("a threshold was recorded", float(d["threshold"]) > 0)
        # The level is reported independently of the correlation: column 2 carries the
        # planted effect AND is shifted up by 1.4*y, so its mean must exceed column 0's.
        check("the level row is the column's own mean", lvl[2] > lvl[0] + 0.5,
              f"levels {lvl[0]:+.3f} vs {lvl[2]:+.3f}")
        check("the level standard error is positive", bool(np.all(lse > 0)))


def test_union_cap_and_decile_table():
    print("\n[union] the decile table reads the curve, --max-union restricts the rest")
    HC = FC.HC
    rng = np.random.default_rng(3)

    # apply_union_cap: the contract is "drop steps, keep every array aligned".
    uni = np.where(np.arange(40) % 2 == 0, 0.3, 0.9)     # evens under a 0.5 cap
    a = np.arange(40)
    b = np.arange(80).reshape(40, 2)
    (ca, cb, cn), keep = HC.apply_union_cap(0.5, uni, (a, b, None))
    check("the cap keeps exactly the steps at or below it",
          list(np.flatnonzero(keep)) == list(range(0, 40, 2)),
          f"kept {int(keep.sum())} of 40")
    check("...and applies to every array, 1-D and 2-D alike",
          list(ca) == list(range(0, 40, 2)) and cb.shape == (20, 2)
          and list(cb[:, 0]) == list(range(0, 80, 4)))
    check("a None array (an absent `mass`) survives the cap", cn is None)
    (na,), nkeep = HC.apply_union_cap(0.0, uni, (a,))
    check("cap 0 is a no-op, which is the default and every published number",
          bool(nkeep.all()) and list(na) == list(a))
    try:
        HC.apply_union_cap(0.05, uni, (a,))
        check("a cap that leaves too little to analyse raises", False)
    except SystemExit:
        check("a cap that leaves too little to analyse raises", True)

    # The decile table: a column built to fall with union size must read that way, and
    # one built independent of it must read flat. This is the whole point of the table.
    n = 2000
    u = rng.uniform(0.05, 0.95, n)
    cols = {"falls": 1.0 - u + rng.normal(0, 0.02, n), "flat": rng.normal(0.5, 0.3, n)}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        HC.union_decile_table(u, cols, null=0.5)
    def cells(label):
        ln = next(l for l in buf.getvalue().splitlines() if l.strip().startswith(label))
        return [float(x) for x in ln.strip()[len(label):].split()]

    falls = cells("falls")
    check("the decile cells of a union-dependent column decrease monotonically",
          all(falls[i] > falls[i + 1] for i in range(9)),
          f"{falls[0]:.2f}->{falls[9]:.2f}")
    check("r(union) is strongly negative for it", falls[10] < -0.9, f"r={falls[10]:+.3f}")
    check("r(union) is ~0 for a column independent of union size",
          abs(cells("flat")[10]) < 0.1, f"r={cells('flat')[10]:+.3f}")
    check("the deciles are equal-count", len(set(cells("n steps"))) <= 2,
          f"{cells('n steps')}")
    check("the bin means increase across the deciles",
          cells("mean union") == sorted(cells("mean union")))

    # End to end: the cap has to reach the Bonferroni threshold, not just the tables.
    # Steps are dropped, so the completion count -- the effective n -- falls with it.
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "rollout_mean"
        (out / "scan").mkdir(parents=True)
        rows_, cor, v2, au, step, uu, mass = [], [], [], [], [], [], []
        for c in range(400):
            y = float(rng.integers(0, 2))
            for s in range(2):
                rows_.append(c); cor.append(y); step.append(s)
                # one completion in four is entirely above the cap, so it disappears
                uu.append(0.9 if c % 4 == 0 else 0.3)
                v2.append(rng.normal(size=3)); au.append(rng.normal(size=3))
                mass.append(rng.normal(size=3))
        np.savez_compressed(
            out / "scan" / "shard00.npz",
            v2=np.array(v2, dtype=np.float32), auroc=np.array(au, dtype=np.float32),
            row=np.array(rows_), step=np.array(step),
            correct=np.array(cor, dtype=np.float32), union=np.array(uu, dtype=np.float32),
            mass=np.array(mass, dtype=np.float32),
            names=np.array(["L0", "L1", "inc1"]), map=np.array("rollout_mean"),
            alpha=np.array(0.5))

        def run(cap):
            class A:
                out_dir = str(out)
                all_columns = True
                max_union = cap
            with contextlib.redirect_stdout(io.StringIO()):
                FC.report(A())
            # read eagerly: np.load is lazy and the next run overwrites this file
            with np.load(out / "corr.npz") as z:
                return float(z["max_union"]), float(z["threshold"])

        full, capped = run(0.0), run(0.5)
        check("the cap is recorded in corr.npz, so a saved report is unambiguous",
              full[0] == 0.0 and capped[0] == 0.5)
        check("the cap raises the Bonferroni threshold, the effective n having fallen",
              capped[1] > full[1], f"{full[1]:.4f} -> {capped[1]:.4f}")


def main():
    test_mass_invariant()
    test_residual_fixed_point()
    test_wnorm_matches_naive()
    test_repeat_v_agrees()
    test_increment_definition()
    test_partial_corr()
    test_sample_columns()
    test_grad_maps_returns_numpy()
    test_glimpse_adapter()
    test_report_recovers_planted_effect()
    test_union_cap_and_decile_table()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print(f"  FAILED: {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
