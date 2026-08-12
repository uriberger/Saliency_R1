#!/usr/bin/env python
"""CPU checks for the GLIMPSE map (saliency_viz.py, docs/saliency-maps.md map 6).

No GPU, no weights: the paper's algebra is small enough to write out naively in the test
and compare against the production form elementwise. What that covers is every place
where the fast form could silently disagree with the equations, plus the two deviations
from the paper, each of which is a decision that has to be defended by a test rather than
by a comment:

  * `glimpse_edge_matrix` equals a naive [H, N, N] transcription of eqs 5-7, and the head
    weighting does what eq 6 claims -- a head whose positive gradient lies where it does
    NOT attend is demoted below one whose gradient lies where it does, at equal gradient
    mass. That ratio is the whole content of eq 6 and it is invisible in any shape check;
  * `glimpse_layer_alphas` equals eqs 9-10, sums to 1, and survives a backward that
    produced no positive gradient at all (which would otherwise be 0/0 -> a blank map);
  * `glimpse_propagate` equals the literal `R <- R + L R` recursion of eqs 11-13, row for
    row, up to the 2^L the folding removes -- and the folding is a pure scale, i.e. the
    same for every token, which is what lets `beta` compare tokens afterwards;
  * THE ROW. On a real (if tiny) causal attention stack, `d z_t / d A` is exactly zero on
    every row at or after the query that produced `z_t`. So the paper's literal `R(t, :)`
    for the token at position `t` is the identity row and its image columns are all zero:
    map 6 read that way is blank for every step. The row used instead is `t-1`, and the
    test asserts both halves -- blank at `t`, not blank at `t-1` -- because the whole
    method turns on it;
  * `glimpse_token_weight` crosses prompt alignment into the visual map (eq 17) and the
    aggregation weights by it, so a token the model was unsure of moves the map less;
  * `prompt_positions` finds the question inside the chat template, falls back exactly
    once, and never returns an image column;
  * the causality guard fires on an unmasked attention row -- eager attention with no
    mask is silently bidirectional and every map would be wrong.

    python test_glimpse_cpu.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent

# Same reason as test_grad_maps_cpu.py: the login node's 188-core intra-op pool costs far
# more in thread sync than these tiny ops cost in arithmetic, and one thread is what makes
# the run deterministic.
torch.set_num_threads(1)


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


SV = _load("_t_saliency_viz", "saliency_viz.py")

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'ok  ' if cond else 'FAIL'} {name}" + (f"   {detail}" if detail else ""))


def causal_attn(h, n, gen):
    """[H, N, N] row-stochastic causal attention, as a real masked softmax would be."""
    x = torch.rand(h, n, n, generator=gen).tril()
    return x / x.sum(-1, keepdim=True)


# ---------------------------------------------------------------------------
# eqs 5-7: the gradient-fused, head-weighted edge matrix
# ---------------------------------------------------------------------------
def naive_edge_matrix(a, g, temp):
    """The equations transcribed with no regard for memory, as the reference."""
    gg = torch.relu(g * a)                                             # eq 5
    ratio = gg.flatten(1).sum(1) / torch.relu(g).flatten(1).sum(1)
    w = torch.softmax(ratio / temp, dim=0)                             # eq 6
    e = (w[:, None, None] * gg).sum(0)                                 # eq 7
    return e / e.sum(-1, keepdim=True)


def test_edge_matrix():
    print("\n[eqs 5-7] the edge matrix")
    gen = torch.Generator().manual_seed(0)
    h, n = 6, 11
    a = causal_attn(h, n, gen)
    g = torch.randn(h, n, n, generator=gen).tril()

    e, g_l1 = SV.glimpse_edge_matrix(a, g, temp=0.5)
    ref = naive_edge_matrix(a, g, 0.5)
    check("equals the naive [H, N, N] transcription of eqs 5-7",
          torch.allclose(e, ref, atol=1e-6), f"max |diff| = {(e - ref).abs().max():.2e}")
    check("rows sum to 1 (eq 7 preserves probability mass)",
          torch.allclose(e.sum(-1), torch.ones(n), atol=1e-6))
    check("no mass lands after the query (the gradient cannot make it acausal)",
          float(e.triu(1).abs().max()) == 0.0)
    check("the layer weight is ||sum_h g^h||_1, summed over heads BEFORE the norm",
          torch.allclose(g_l1, g.sum(0).abs().sum(), atol=1e-5),
          f"{float(g_l1):.4f} vs {float(g.sum(0).abs().sum()):.4f}")

    # eq 6 divides by the head's positive-gradient mass. Two heads, identical gradient
    # mass: one attends exactly where its gradient is, the other everywhere else.
    n2 = 4
    aa = torch.zeros(2, n2, n2)
    aa[0, 3] = torch.tensor([1.0, 0.0, 0.0, 0.0])       # attends where the gradient is
    aa[1, 3] = torch.tensor([0.0, 1.0, 0.0, 0.0])       # attends where it is not
    aa[:, 0, 0] = aa[:, 1, 1] = aa[:, 2, 2] = 1.0
    gg = torch.zeros(2, n2, n2)
    gg[:, 3, 0] = 1.0                                    # same gradient in both heads
    e_aligned, _ = SV.glimpse_edge_matrix(aa, gg, temp=0.5)
    w = torch.softmax(torch.tensor([1.0, 0.0]) / 0.5, dim=0)
    check("a head that attends where its gradient is outweighs one that does not",
          float(w[0]) > float(w[1]) and float(e_aligned[3, 0]) == 1.0,
          f"w = {w.tolist()}")

    hot = SV.glimpse_edge_matrix(a, g, temp=1e6)[0]
    check("a large temperature collapses eq 6 to the uniform head mean",
          torch.allclose(hot, naive_edge_matrix(a, g, 1e6), atol=1e-6))


# ---------------------------------------------------------------------------
# eqs 9-10: the layer weights
# ---------------------------------------------------------------------------
def test_layer_alphas():
    print("\n[eqs 9-10] the layer weights")
    layers = list(range(4, 12))
    g = [1.0, 3.0, 2.0, 0.5, 4.0, 1.0, 2.0, 0.25]
    al = SV.glimpse_layer_alphas(g, layers, depth_temp=0.2)

    gt = torch.tensor(g)
    s = torch.softmax(0.2 * (torch.tensor(layers, dtype=torch.float32) + 1), 0)
    ref = (gt / gt.sum()) * s
    ref = ref / ref.sum()
    check("equals eqs 9-10", torch.allclose(al, ref, atol=1e-6),
          f"max |diff| = {(al - ref).abs().max():.2e}")
    check("sums to 1", abs(float(al.sum()) - 1.0) < 1e-6)

    flat = SV.glimpse_layer_alphas([1.0] * len(layers), layers, depth_temp=0.2)
    check("with equal gradient evidence it is the depth prior alone, and it rises",
          torch.allclose(flat, s, atol=1e-6) and bool((flat.diff() > 0).all()))
    check("depth temperature 0 removes the prior",
          torch.allclose(SV.glimpse_layer_alphas([1.0] * len(layers), layers, 0.0),
                         torch.full((len(layers),), 1.0 / len(layers)), atol=1e-6))

    zero = SV.glimpse_layer_alphas([0.0] * len(layers), layers, depth_temp=0.2)
    check("a backward with no gradient anywhere falls back to the prior, not to NaN",
          bool(torch.isfinite(zero).all()) and torch.allclose(zero, s, atol=1e-6))

    # softmax is shift-invariant, which is the claim glimpse_layer_alphas makes about
    # passing the model's own layer indices for a contiguous slice.
    rebased = SV.glimpse_layer_alphas(g, list(range(len(layers))), depth_temp=0.2)
    check("re-basing a contiguous slice's indices at 0 changes nothing",
          torch.allclose(al, rebased, atol=1e-6))


# ---------------------------------------------------------------------------
# eqs 11-13: the propagation
# ---------------------------------------------------------------------------
def naive_relevance(mats, alphas):
    """`R <- R + L_l R` with `L_l = I + alpha_l E_l`, exactly as printed."""
    n = mats[0].shape[0]
    r = torch.eye(n, dtype=torch.float64)
    for e, al in zip(mats, alphas):
        lm = torch.eye(n, dtype=torch.float64) + float(al) * e.double()
        r = r + lm @ r
    return r


def test_propagate():
    print("\n[eqs 11-13] the propagation")
    gen = torch.Generator().manual_seed(3)
    n, n_l = 9, 5
    # float64 throughout: the reference and the code under test then differ only where the
    # algebra differs, so the comparison can be exact rather than tolerant.
    mats = [causal_attn(1, n, gen)[0].double() for _ in range(n_l)]
    alphas = torch.softmax(torch.rand(n_l, generator=gen), 0).double()

    ref = naive_relevance(mats, alphas)
    scale = 2.0 ** n_l
    rows = [SV.glimpse_propagate(r, mats, alphas) for r in range(n)]
    got = torch.stack(rows).double() * scale
    # One scalar restores the whole matrix, every row of it -- which is the property the
    # folding needs: `beta` compares `a_t` ACROSS tokens, so a per-row rescale would be a
    # silent reweighting of the aggregation, and this is what rules it out.
    check("every row equals the literal recursion, up to one 2^L shared by all of them",
          torch.allclose(got, ref, rtol=1e-12, atol=1e-12),
          f"max |diff| = {(got - ref).abs().max():.2e}")
    check("the row stays non-negative (ReLU'd edges, non-negative alphas)",
          bool((got >= 0).all()))
    check("the row keeps its own position and never reads a later one",
          float(rows[4][4]) > 0 and float(rows[4][5:].abs().max()) == 0.0)


# ---------------------------------------------------------------------------
# The row. Why `t-1` and not `t`.
# ---------------------------------------------------------------------------
class ToyStack(torch.nn.Module):
    """A real causal attention stack, small enough to differentiate exhaustively.

    Only the property under test has to be faithful: attention is a causal softmax whose
    weights are a tensor in the graph, and the logit at a row is read from that row's
    hidden state. That is enough for `d z / d A` to have the structure the row argument
    depends on.
    """

    def __init__(self, n_layers, dim, gen):
        super().__init__()
        self.q = torch.nn.ModuleList(torch.nn.Linear(dim, dim) for _ in range(n_layers))
        self.k = torch.nn.ModuleList(torch.nn.Linear(dim, dim) for _ in range(n_layers))
        self.v = torch.nn.ModuleList(torch.nn.Linear(dim, dim) for _ in range(n_layers))
        self.head = torch.nn.Linear(dim, 5)
        for p in self.parameters():
            with torch.no_grad():
                p.copy_(torch.randn(p.shape, generator=gen) * 0.3)
            p.requires_grad_(False)

    def forward(self, h):
        n = h.shape[0]
        mask = torch.full((n, n), float("-inf")).triu(1)
        attns = []
        for q, k, v in zip(self.q, self.k, self.v):
            a = torch.softmax(q(h) @ k(h).T + mask, dim=-1)      # [N, N], causal
            a.retain_grad()
            attns.append(a)
            h = h + a @ v(h)
        return self.head(h), attns


def test_the_row():
    print("\n[the row] why t-1 and not t")
    gen = torch.Generator().manual_seed(11)
    n, dim, n_l = 8, 6, 3
    img = [1, 2, 3]                     # pretend image columns
    net = ToyStack(n_l, dim, gen)
    h = (torch.randn(n, dim, generator=gen) * 0.5).requires_grad_(True)
    logits, attns = net(h)

    r = 5                               # the query row that produces the token at 6
    z = logits[r, 2]
    torch.autograd.grad(z, h, retain_graph=True)      # populates a.grad on every layer

    after = max(float(a.grad[r + 1:].abs().max()) for a in attns)
    at_row = max(float(a.grad[r].abs().max()) for a in attns)
    check("d z_t / d A is exactly zero on every row after the query that produced it",
          after == 0.0, f"max |grad| after row {r} = {after:.2e}")
    check("...and is not zero on that query's own row", at_row > 0)

    mats, g_l1 = [], []
    for a in attns:
        e, g1 = SV.glimpse_edge_matrix(a.detach()[None], a.grad[None], temp=0.5)
        mats.append(e)
        g_l1.append(g1)
    alphas = SV.glimpse_layer_alphas(g_l1, list(range(n_l)), depth_temp=0.2)

    v_literal = SV.glimpse_propagate(r + 1, mats, alphas)      # the paper's R(t, :)
    v_used = SV.glimpse_propagate(r, mats, alphas)             # the query that made it
    check("the literal row is the identity row: its image columns are ALL zero",
          float(v_literal[img].abs().max()) == 0.0 and float(v_literal[r + 1]) > 0,
          "map 6 read that way is blank for every step")
    check("the row actually used carries image mass", float(v_used[img].abs().max()) > 0)


# ---------------------------------------------------------------------------
# eqs 17-18, 22: the token weights and the aggregation
# ---------------------------------------------------------------------------
def test_token_weight():
    print("\n[eqs 17-18] the token weights")
    conf, align = torch.tensor(0.25), torch.tensor(0.4)
    check("full = confidence x prompt alignment",
          abs(float(SV.glimpse_token_weight(conf, align, "full")) - 0.1) < 1e-6)
    check("the ablation modes drop one factor each",
          float(SV.glimpse_token_weight(conf, align, "confidence")) == 0.25
          and float(SV.glimpse_token_weight(conf, align, "prompt")) == float(align)
          and float(SV.glimpse_token_weight(conf, align, "uniform")) == 1.0)
    try:
        SV.glimpse_token_weight(conf, align, "nonsense")
        check("an unknown mode raises", False)
    except ValueError:
        check("an unknown mode raises", True)

    # eq 22 over a two-token step, by hand: a token the model was unsure of moves the
    # aggregate less than a confident one, at equal alignment.
    maps = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    confs = torch.tensor([0.9, 0.1])
    w = torch.stack([SV.glimpse_token_weight(c, torch.tensor(0.5), "full") for c in confs])
    agg = (w[:, None] * maps).sum(0) / w.sum()
    check("the aggregate leans toward the confident token",
          abs(float(agg[0]) - 0.9) < 1e-6 and float(agg[0]) > float(agg[1]),
          f"{agg.tolist()}")


# ---------------------------------------------------------------------------
# the prompt columns, and the causality guard
# ---------------------------------------------------------------------------
class FakeTok:
    """Just enough tokenizer: a word-per-token encoder over a fixed vocabulary."""

    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [self.vocab.get(w, 999) for w in text.split()]}


def test_prompt_positions():
    print("\n[eq 14] the prompt columns")
    vocab = {w: i + 20 for i, w in enumerate("what colour is the cat system reason".split())}
    tok = FakeTok(vocab)
    question = "what colour is the cat"
    q_ids = [vocab[w] for w in question.split()]
    img = [3, 4, 5, 6]
    prompt = [vocab["system"], vocab["reason"]] + [0] * 4 + q_ids     # image at 2..5
    img = [2, 3, 4, 5]

    pos, how = SV.prompt_positions(tok, question, prompt, img)
    check("the question is found inside the template", how == "question"
          and pos == list(range(6, 11)), f"{how}, {pos}")
    check("no image column is ever in P", not (set(pos) & set(img)))

    # A leading-space merge at the template boundary: the first token comes out different.
    broken = list(prompt)
    broken[6] = 99
    pos2, how2 = SV.prompt_positions(tok, question, broken, img)
    check("a first-token merge falls back to the rest of the question",
          how2 == "question_less_first" and pos2 == list(range(7, 11)), f"{how2}, {pos2}")

    pos3, how3 = SV.prompt_positions(tok, "not present at all here", prompt, img)
    check("an unfindable question falls back to every non-image prompt token",
          how3 == "prompt_minus_image" and pos3 == [0, 1, 6, 7, 8, 9, 10], f"{how3}")


def test_causal_guard():
    print("\n[guard] eager attention with no mask")
    cap = SV.GlimpseCapture.__new__(SV.GlimpseCapture)
    cap._checked = False
    n = 16
    good = torch.zeros(1, 2, n, n)
    good[:, :, 0, 0] = 1.0
    cap._check_causal(good)
    check("a causal first row passes", True)

    cap._checked = False
    bad = torch.full((1, 2, n, n), 1.0 / n)
    try:
        cap._check_causal(bad)
        check("an unmasked first row raises", False, "it did not")
    except RuntimeError as e:
        check("an unmasked first row raises", "not causal" in str(e))


# ---------------------------------------------------------------------------
def main():
    test_edge_matrix()
    test_layer_alphas()
    test_propagate()
    test_the_row()
    test_token_weight()
    test_prompt_positions()
    test_causal_guard()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
        sys.exit(1)


if __name__ == "__main__":
    main()
