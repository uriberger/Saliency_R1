#!/usr/bin/env python
"""CPU checks for trl/grad_maps.py -- the pixel-gradient map the gradient reward scores.

No GPU, no weights: a toy VLM that honours the Qwen2-VL processor's pixel layout stands
in for the 8B model, which is enough to gate every place a silent index, sign or
chain-rule error would corrupt the map while raising nothing:

  * the vmapped VJP (`is_grads_batched`) equals the naive retain_graph loop, elementwise
    -- the batched path is the default and it is the one nobody would notice going wrong;
  * `pixel_regroup` equals the norm of the true gradient restricted to a token's pixels,
    measured by central differences on the ACTUAL pixels. This is the test that catches
    the temporal axis: the processor stores `tps` copies of each pixel, so the gradients
    of the copies must be SUMMED before the norm, and a version that norms first passes
    every shape check and fails this one;
  * the whole `step_grad_maps` bookkeeping -- the tensor `logits_to_keep`, the `a-1` row
    that predicts the token at `a`, the target ids, the per-step mean -- against a naive
    reference that takes a full forward and indexes rows by hand;
  * the centering identity, d(z_t - mean_v z_v) == d z_t - mean_v d z_v;
  * the reason `clogit` is the default at all: as the model grows confident the `logprob`
    map collapses toward zero and the `clogit` map does not;
  * `frozen_params` restores `requires_grad` exactly, per parameter -- under PEFT only
    the adapter was trainable, and a blanket restore would silently start training the
    base model.

The pixel LAYOUT itself (which rows belong to which token) is a property of the real
processor and is gated separately by `python saliency_viz.py --stage selftest`, which
also needs no GPU. This file assumes the layout and tests the arithmetic on top of it.

    python test_grad_maps_cpu.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent

# The login node has 188 cores and torch defaults its intra-op pool to all of them. Every
# op here is tiny, so the thread-sync overhead dominates completely: the same run takes
# ~10 min at the default and a few seconds at 1. This is a correctness test, not a
# benchmark -- one thread is also what makes it deterministic.
torch.set_num_threads(1)


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


GM = _load("_t_grad_maps", "trl/grad_maps.py")

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'ok  ' if cond else 'FAIL'} {name}" + (f"   {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# A toy VLM with the processor's pixel layout
# ---------------------------------------------------------------------------
C, TPS, PS = 3, 2, 2          # channels, temporal copies, patch side
GH2, GW2 = 4, 6               # pre-merge patch grid -> 2x3 language-model tokens
GH, GW = GH2 // 2, GW2 // 2
N_PATCH = GH2 * GW2
D_PIX = C * TPS * PS * PS
M = GH * GW                   # image tokens
VOCAB, DIM = 29, 16
IMAGE_TOKEN_ID = 7


def pack_pixels(img: torch.Tensor) -> torch.Tensor:
    """[n_patch, C, ps*ps] unique pixels -> [n_patch, C*T*ps*ps] as the processor packs.

    Row layout is [C, T, ps, ps] and the T axis holds copies of the same still frame,
    which is exactly the duplication `pixel_regroup` has to undo.
    """
    return img[:, :, None, :].expand(N_PATCH, C, TPS, PS * PS).reshape(N_PATCH, D_PIX)


class ToyVLM(nn.Module):
    """Small, real autograd graph: pixels -> patch embeds -> 2x2 merge -> causal LM.

    Mirrors only what `step_grad_maps` touches: the `pixel_values` / `image_grid_thw`
    entry points, the image-token positions in the sequence, and `logits_to_keep` as
    either an int or a tensor of absolute positions.
    """

    def __init__(self, gen, scale: float = 1.0):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, DIM)
        self.vis = nn.Linear(D_PIX, DIM)
        self.merge = nn.Linear(4 * DIM, DIM)
        self.wq = nn.Linear(DIM, DIM)
        self.wk = nn.Linear(DIM, DIM)
        self.wv = nn.Linear(DIM, DIM)
        self.mlp = nn.Sequential(nn.Linear(DIM, 2 * DIM), nn.GELU(), nn.Linear(2 * DIM, DIM))
        self.lm_head = nn.Linear(DIM, VOCAB, bias=False)
        self.scale = scale
        for p in self.parameters():
            with torch.no_grad():
                p.copy_(torch.randn(p.shape, generator=gen, dtype=p.dtype) * 0.5)

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None,
                image_grid_thw=None, use_cache=False, logits_to_keep=0, **kw):
        h = self.embed(input_ids)                                   # [1, L, DIM]
        if pixel_values is not None:
            per_patch = self.vis(pixel_values)                      # [n_patch, DIM]
            # a language-model token owns 4 consecutive patch rows (the merge block)
            tok = self.merge(per_patch.reshape(M, 4 * DIM))         # [M, DIM]
            pos = (input_ids[0] == IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0]
            # out-of-place: `is_grads_batched` vmaps the backward, and in-place index
            # assignment is the kind of op vmap refuses.
            h = h + torch.zeros_like(h).index_add(1, pos, tok[None].to(h.dtype))
        q, k, v = self.wq(h), self.wk(h), self.wv(h)
        att = (q @ k.transpose(1, 2)) / DIM ** 0.5
        causal = torch.triu(torch.ones(h.shape[1], h.shape[1], dtype=torch.bool), 1)
        att = att.masked_fill(causal, float("-inf")).softmax(-1)
        h = h + att @ v
        h = h + self.mlp(h)
        idx = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        if isinstance(logits_to_keep, int) and logits_to_keep == 0:
            idx = slice(None)
        logits = self.lm_head(h[:, idx, :]) * self.scale

        class _Out:
            pass
        out = _Out()
        out.logits = logits
        return out


def toy_case(gen, scale: float = 1.0):
    """-> (model, forward_inputs, spans, img) with the image tokens inside the prompt."""
    model = ToyVLM(gen, scale=scale).double()
    img = torch.rand(N_PATCH, C, PS * PS, generator=gen, dtype=torch.float64)
    prompt = torch.randint(8, VOCAB, (5,), generator=gen)
    ids = torch.cat([prompt[:2], torch.full((M,), IMAGE_TOKEN_ID), prompt[2:],
                     torch.randint(8, VOCAB, (11,), generator=gen)])[None]
    fwd = {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "pixel_values": pack_pixels(img),
        "image_grid_thw": torch.tensor([[1, GH2, GW2]]),
    }
    prompt_len = 2 + M + 3
    spans = [(prompt_len + 1, prompt_len + 4), (prompt_len + 5, prompt_len + 11)]
    return model, fwd, spans, img


GRID = [1, GH2, GW2]


def run(model, fwd, spans, **kw):
    return GM.step_grad_maps(model, fwd, spans, GRID, PS, TPS, **kw)


# ---------------------------------------------------------------------------
def test_batched_vjp_matches_loop():
    print("\n[vjp] the vmapped backward equals the loop")
    gen = torch.Generator().manual_seed(0)
    model, fwd, spans, _ = toy_case(gen)
    a = run(model, fwd, spans, batched=True)
    b = run(model, fwd, spans, batched=False)
    check("shape is [n_steps, gh, gw]", a.shape == (len(spans), GH, GW), str(a.shape))
    check("the two paths agree elementwise",
          np.allclose(a, b, rtol=1e-10, atol=1e-12),
          f"max |diff| {np.abs(a - b).max():.2e}")
    check("the map is not identically zero", float(np.abs(a).max()) > 0)
    check("different steps give different maps",
          not np.allclose(a[0], a[1], rtol=1e-6))


def test_pixel_regroup_matches_finite_differences():
    print("\n[regroup] the map is the true gradient norm over a token's pixels")
    gen = torch.Generator().manual_seed(1)
    model, fwd, spans, img = toy_case(gen)
    span = spans[0]

    def scalar(image: torch.Tensor) -> float:
        f = dict(fwd)
        f["pixel_values"] = pack_pixels(image)
        out = model(**f, logits_to_keep=torch.arange(span[0] - 1, span[1] - 1))
        z = out.logits[0]
        picked = z.gather(-1, fwd["input_ids"][0, span[0]:span[1]][:, None]).squeeze(-1)
        return float(((picked - z.mean(-1))).mean())

    # central differences w.r.t. every unique pixel; the packing duplicates each one
    # across the temporal axis, so this perturbs both copies at once -- which is what
    # makes it a test of the sum-over-T rule rather than of the reshape alone.
    eps = 1e-6
    fd = torch.zeros(N_PATCH, C, PS * PS, dtype=torch.float64)
    for r in range(N_PATCH):
        for c in range(C):
            for k in range(PS * PS):
                up, dn = img.clone(), img.clone()
                up[r, c, k] += eps
                dn[r, c, k] -= eps
                fd[r, c, k] = (scalar(up) - scalar(dn)) / (2 * eps)

    # a language-model token owns 4 consecutive patch rows; its gradient norm is the
    # norm of the concatenation over those rows.
    ref = fd.reshape(GH, GW, 4, C * PS * PS).pow(2).sum(dim=(2, 3)).sqrt().numpy()
    got = run(model, fwd, [span])[0]
    rel = np.abs(got - ref).max() / max(float(np.abs(ref).max()), 1e-30)
    check("map == || d F_S / d(token's pixels) ||, by central differences",
          rel < 1e-6, f"max relative error {rel:.2e}")

    # The failure mode the temporal axis actually invites: taking the norm over the
    # packed buffer without first summing the duplicated frames. It has the right shape,
    # the right units and the wrong value, so only a numeric check catches it.
    leaf = fwd["pixel_values"].detach().clone().requires_grad_(True)
    f2 = dict(fwd)
    f2["pixel_values"] = leaf
    z = model(**f2, logits_to_keep=torch.arange(span[0] - 1, span[1] - 1)).logits[0]
    picked = z.gather(-1, fwd["input_ids"][0, span[0]:span[1]][:, None]).squeeze(-1)
    (g_raw,) = torch.autograd.grad((picked - z.mean(-1)).mean(), leaf)
    gr = g_raw.reshape(N_PATCH, C, TPS, PS * PS)
    no_sum = gr.pow(2).sum(dim=(1, 2, 3)).reshape(GH, GW, 4).sum(dim=2).sqrt().numpy()
    check("the two temporal copies carry different gradients, so the rule is testable",
          not torch.allclose(gr[:, :, 0], gr[:, :, 1], rtol=1e-3))
    check("norming without summing the temporal copies gives a different map",
          not np.allclose(got, no_sum, rtol=1e-3),
          f"differs by {np.abs(got - no_sum).max():.2e}")


def test_matches_naive_reference():
    print("\n[bookkeeping] rows, targets and the per-step mean")
    gen = torch.Generator().manual_seed(2)
    model, fwd, spans, _ = toy_case(gen)

    # Naive reference: one full forward (no logits_to_keep), rows indexed by hand.
    leaf = fwd["pixel_values"].detach().clone().requires_grad_(True)
    f2 = dict(fwd)
    f2["pixel_values"] = leaf
    out = model(**f2)
    z = out.logits[0].double()
    ref = []
    for a, b in spans:
        acc = 0.0
        for n in range(a, b):
            row = z[n - 1]                      # the row that predicts the token at n
            tgt = int(fwd["input_ids"][0, n])
            acc = acc + (row[tgt] - row.mean())
        (g,) = torch.autograd.grad(acc / (b - a), leaf, retain_graph=True)
        ref.append(GM.pixel_regroup(g, GRID, PS, TPS).numpy())
    ref = np.stack(ref)

    got = run(model, fwd, spans)
    # `step_grad_maps` returns float32 -- that is the dtype the reward consumes, and the
    # graph itself runs in the model's dtype -- so a float64 reference cannot be matched
    # tighter than float32 rounding, whatever the toy is run in.
    check("step_grad_maps == the hand-indexed reference",
          np.allclose(got, ref, rtol=1e-6, atol=1e-12),
          f"max relative error {np.abs(got - ref).max() / np.abs(ref).max():.2e}")

    # an off-by-one in the row index is the error this is really guarding, so make sure
    # the reference would have caught it
    shifted = []
    for a, b in spans:
        acc = sum(z[n][int(fwd["input_ids"][0, n])] - z[n].mean() for n in range(a, b))
        (g,) = torch.autograd.grad(acc / (b - a), leaf, retain_graph=True)
        shifted.append(GM.pixel_regroup(g, GRID, PS, TPS).numpy())
    check("using row n instead of n-1 gives a different map",
          not np.allclose(got, np.stack(shifted), rtol=1e-3))


def test_centering_identity():
    print("\n[target] centering removes the common-mode channel")
    gen = torch.Generator().manual_seed(3)
    model, fwd, spans, _ = toy_case(gen)
    span = spans[0]
    leaf = fwd["pixel_values"].detach().clone().requires_grad_(True)
    f2 = dict(fwd)
    f2["pixel_values"] = leaf
    z = model(**f2, logits_to_keep=torch.arange(span[0] - 1, span[1] - 1)).logits[0].double()
    tgt = fwd["input_ids"][0, span[0]:span[1]]

    picked = z.gather(-1, tgt[:, None]).squeeze(-1).mean()
    meanv = z.mean(-1).mean()
    (g_pick,) = torch.autograd.grad(picked, leaf, retain_graph=True)
    (g_mean,) = torch.autograd.grad(meanv, leaf, retain_graph=True)
    (g_cent,) = torch.autograd.grad(picked - meanv, leaf, retain_graph=True)
    check("d(z_t - mean_v z_v) == d z_t - mean_v d z_v",
          torch.allclose(g_cent, g_pick - g_mean, rtol=1e-10, atol=1e-12))
    check("the common-mode part is not negligible, so centering does something",
          float(g_mean.norm() / g_pick.norm()) > 1e-3,
          f"||mean-logit grad|| / ||logit grad|| = {float(g_mean.norm() / g_pick.norm()):.3f}")


def test_logprob_saturates_and_clogit_does_not():
    print("\n[target] the saturation that ruled out logprob")

    def confident_case(scale):
        """A case teacher-forced on the model's OWN greedy tokens.

        This is the setting that matters: the reward differentiates the chain the policy
        actually produced, so `P(t_n)` is high by construction. Scoring a random token id
        instead would measure the opposite -- sharpening the softmax drives an arbitrary
        token's probability toward zero, not toward one.
        """
        gen = torch.Generator().manual_seed(4)
        model, fwd, spans, _ = toy_case(gen, scale=scale)
        span = spans[0]
        rows = torch.arange(span[0] - 1, span[1] - 1)
        for _ in range(4):                       # a few greedy passes to reach a fixpoint
            z = model(**fwd, logits_to_keep=rows).logits[0]
            ids = fwd["input_ids"].clone()
            ids[0, span[0]:span[1]] = z.argmax(-1)
            if torch.equal(ids, fwd["input_ids"]):
                break
            fwd["input_ids"] = ids
        z = model(**fwd, logits_to_keep=rows).logits[0]
        p = z.softmax(-1).gather(-1, fwd["input_ids"][0, span[0]:span[1]][:, None])
        # 1 - P, not P: teacher-forced on its own greedy tokens the toy is already at
        # P = 1.000 to three decimals at both scales, so the probability itself cannot
        # show the sharpening and the gap is the only thing that moves.
        return model, fwd, span, float((1.0 - p).max())

    out = {}
    for scale in (1.0, 8.0):     # scaling the logits sharpens the softmax
        model, fwd, span, gap = confident_case(scale)
        out[(scale, "gap")] = gap
        for tgt in ("logprob", "clogit"):
            out[(scale, tgt)] = float(np.linalg.norm(run(model, fwd, [span], target=tgt)[0]))

    check("the model is confident in its own tokens, and sharpening makes it more so",
          out[(1.0, "gap")] < 0.1 and out[(8.0, "gap")] < out[(1.0, "gap")],
          f"worst 1 - P(t_n) {out[(1.0, 'gap')]:.2e} -> {out[(8.0, 'gap')]:.2e}")
    lp = out[(8.0, "logprob")] / out[(1.0, "logprob")]
    cl = out[(8.0, "clogit")] / out[(1.0, "clogit")]
    check("the clogit map scales with the logits, i.e. it does not saturate",
          cl > 4.0, f"||map|| ratio {cl:.2f} (the logits were scaled 8x)")
    check("the logprob map is suppressed by confidence instead",
          lp / cl < 0.5, f"logprob ratio {lp:.3f} against clogit's {cl:.2f}")


def test_frozen_params_restores():
    print("\n[hygiene] frozen_params")
    gen = torch.Generator().manual_seed(5)
    model, fwd, spans, _ = toy_case(gen)
    names = [n for n, _ in model.named_parameters()]
    for i, (_, p) in enumerate(model.named_parameters()):
        p.requires_grad_(i % 2 == 0)        # a PEFT-like mix of frozen and trainable
    before = [p.requires_grad for p in model.parameters()]

    with GM.frozen_params(model):
        inside = [p.requires_grad for p in model.parameters()]
        maps = run(model, fwd, spans)
    after = [p.requires_grad for p in model.parameters()]

    check("every parameter is frozen inside the context", not any(inside))
    check("the flags are restored exactly, per parameter", before == after,
          f"{sum(before)} of {len(names)} trainable before and after")
    check("no parameter received a gradient",
          all(p.grad is None for p in model.parameters()))
    check("the map is still computed while frozen", float(np.abs(maps).max()) > 0)


def main():
    torch.manual_seed(0)
    test_batched_vjp_matches_loop()
    test_pixel_regroup_matches_finite_differences()
    test_matches_naive_reference()
    test_centering_identity()
    test_logprob_saturates_and_clogit_does_not()
    test_frozen_params_restores()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print(f"  FAILED: {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
