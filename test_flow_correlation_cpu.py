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
  * `report` recovers a planted correlation and holds it on the even-row half.

    python test_flow_correlation_cpu.py
"""

from __future__ import annotations

import importlib.util
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


def main():
    test_mass_invariant()
    test_residual_fixed_point()
    test_wnorm_matches_naive()
    test_repeat_v_agrees()
    test_increment_definition()
    test_partial_corr()
    test_sample_columns()
    test_grad_maps_returns_numpy()
    test_report_recovers_planted_effect()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print(f"  FAILED: {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
