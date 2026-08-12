#!/usr/bin/env python3
"""Side-by-side pictures of five saliency maps, on the model's own observe steps.

Every number in docs/saliency-maps.md is a scalar against a Grounding-DINO union.
This script draws the maps themselves instead: one directory per sample holding the
question, the generated chain, the image the model actually saw, and that image with
each map laid over it as a heatmap. No DINO, no scoring -- looking, not measuring.

Five maps, all reduced over the same tokens (the step's own span), so the only
difference between the five pictures is how a patch's saliency is defined:

  direct         mean_{p in S} mean_{h in {28,31}} A^{22,h}_{p, I_j}
                 docs/saliency-maps.md map 1 -- what the GRPO overlap reward paid for.

  rollout_mean   docs map 2. Abnar-Zuidema rollout, heads merged by the mean, read at
                 the last layer, averaged over the step's tokens.

  rollout_wnorm  docs map 3. Same recursion, edge weight ||sum_h A^h_{p,q} W_O^h v^h_q||
                 -- the value-norm correction, so attention sinks stop dominating.

  grad           docs map 5, moved from the image embeddings down to the PIXELS, which
                 is what was asked for:

                     g_j = mean_{n in S} || d f_n / d (pixels of patch j) ||_2

                 f_n is the log-prob (or the raw logit, --grad-target) of the token the
                 model actually generated at n, teacher-forced on its own chain. Taking
                 the norm per token and averaging the norms -- rather than differentiating
                 the summed logit once -- keeps the "average over the step's tokens"
                 identical in meaning to the three attention maps, at the price of one
                 backward per token. Differentiating w.r.t. pixels rather than embeddings
                 means the vision tower is inside the graph, so the deepstack taps are
                 counted automatically and there is no `_ds` variant to choose.

  glimpse        docs map 6 -- GLIMPSE (arXiv 2506.18985v1), gradient-weighted attention
                 propagated with adaptive layer weights, aggregated over the step's
                 tokens by a confidence x prompt-alignment weight rather than a plain
                 mean. It is the one map here that is neither pure attention (1-3) nor
                 pure gradient (4): the gradient only says which heads and which layers
                 to believe, and the attention still says where to look.

The GLIMPSE deviations from the paper are in `glimpse_map` and in docs map 6; the one
that is not cosmetic is the relevance ROW. Taken literally, `R(t, :)` for the token at
position `t` is identically zero -- causality means position `t`'s attention row cannot
affect the logit that generated the token sitting there -- so the row read here is the
query that produced the token, `t-1`, which is what the method has to mean.

The pixel map has to be regrouped from the processor's patch layout to the language
model's token grid. `Qwen2VLImageProcessor` flattens to
[gh, gw, merge, merge, channel, temporal, patch, patch], so language-model token
(i, j) owns pixel rows 4*(i*gw + j) .. +4, and within a row the temporal axis is a
duplicate of the same pixels -- its two gradient halves are SUMMED (chain rule through
the duplication) before the norm is taken. `--stage selftest` proves that layout on the
real processor with a synthetic image; it needs no GPU and no weights.

Stages
------
  scan     GPU, shardable. Per sample: greedy chain -> observe-step segmentation
           (the FLAN-T5 classifier the reward uses) -> the five maps. Writes
           maps.npz + question.txt + generation.txt + original.png per sample.
  render   CPU, single process. Turns maps.npz into the overlays and an index.html.
           Re-runnable with different --norm/--cmap/--overlay-alpha; no GPU needed.
  selftest CPU. Gates the pixel->token regrouping.

  bash launch_saliency_viz.sh --gpus 8 --out-dir outputs/saliency_viz/run1

Output layout
-------------
  <out-dir>/samples/sample_000_row001234/
      question.txt  generation.txt  meta.json  maps.npz  original.png
      sal_direct.png  sal_rollout_mean.png  sal_rollout_wnorm.png  sal_grad.png
      sal_glimpse.png  contact_sheet.png
      steps/step00/  step.txt + the same six images for that step alone
  <out-dir>/index.html

The six images at sample level are the maps averaged over every observe step of that
sample; steps/stepNN/ keeps them separated, which is the thing the maps are actually
defined on.
"""

from __future__ import annotations

import argparse
import contextlib
import html
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent


def repo_path(rel: str) -> Path:
    """Resolve a repo-relative path, falling back to the central tree (see overlap_probe)."""
    p = REPO / rel
    if p.exists():
        return p
    if REPO.parent.name == ".worktrees":
        alt = REPO.parent.parent / rel
        if alt.exists():
            return alt
    return p


def _load_module(name: str, relpath: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, REPO / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# flow_correlation_probe already imports the other three; going through it loads each
# of them exactly once instead of a second copy under a different module name.
FC = _load_module("_sv_flow", "flow_correlation_probe.py")
PROBE, IV = FC.PROBE, FC.IV
# The pixel->token regrouping now lives with the training-time gradient map, so the
# picture drawn here and the map the reward scores cannot drift apart. `--stage selftest`
# still gates it against the real processor.
GM = _load_module("_sv_grad_maps", "trl/grad_maps.py")
pixel_regroup = GM.pixel_regroup
OSTEPS = PROBE.OSTEPS
IMAGE_TOKEN_ID = PROBE.IMAGE_TOKEN_ID

METHODS = ("direct", "rollout_mean", "rollout_wnorm", "grad", "glimpse")
TITLES = {
    "direct": "direct  L{layer} heads {heads}",
    "rollout_mean": "rollout_mean  (L{rl})",
    "rollout_wnorm": "rollout_wnorm  (L{rl})",
    "grad": "grad  d {target} / d pixels",
    "glimpse": "glimpse  L{gfirst}+  ld={gdepth}",
}


# ---------------------------------------------------------------------------
# chain + observe-step segmentation
# ---------------------------------------------------------------------------
def segment_case(tok, clf, question: str, comp_ids: list[int]):
    """-> (text, steps, format_ok) or (text, None, reason).

    `steps` are (step_text, tok_a, tok_b) half-open spans into `comp_ids`, exactly the
    space intervene_probe.build_case works in.
    """
    text = tok.decode(comp_ids, skip_special_tokens=False,
                      clean_up_tokenization_spaces=False)
    enc = tok(text, add_special_tokens=False)
    # Everything downstream indexes `comp_ids` with spans found in `enc`'s space. That
    # is only sound if the decode/encode round trip is the identity, which it is for
    # this tokeniser but is an assumption, not a guarantee -- so check it rather than
    # silently mislabelling a sample's tokens.
    if list(enc["input_ids"]) != list(comp_ids):
        return text, None, "retokenise_mismatch"

    ms = re.search(r"<think>\s*(\S\S*)", text, re.DOTALL | re.MULTILINE)
    me = re.search(r"(\S)\s*</think>", text, re.DOTALL | re.MULTILINE)
    format_ok = bool(ms and me and me.start(1) > ms.start(1))
    if format_ok:
        ts_char, te_char = ms.start(1), me.start(1)
    else:
        # A malformed completion is still worth drawing -- the interesting failures are
        # often the malformed ones -- so fall back to "the whole completion is the
        # chain" and record that this happened.
        stripped = text.rstrip()
        ts_char = len(text) - len(text.lstrip())
        te_char = max(len(stripped) - 1, ts_char)
    ts = enc.char_to_token(0, ts_char)
    te = enc.char_to_token(0, te_char)
    if ts is None or te is None or te <= ts:
        return text, None, "bad_think_tokens"

    steps = OSTEPS.segment_observe_steps(text, ts_char, te_char, enc, 0, ts, te,
                                         question, clf)
    steps = [(s, a, b) for (s, a, b) in steps if 0 <= a < b <= len(comp_ids)]
    if not steps:
        return text, None, "no_observe_steps"
    return text, (steps, format_ok), None


# ---------------------------------------------------------------------------
# map 1: the direct map, layer 22 heads 28/31
# ---------------------------------------------------------------------------
def direct_map(model, inputs, prompt_len, comp_ids, steps, gh, gw, layer, heads, device):
    attn_mod = PROBE.find_attn_module(model, layer)
    if attn_mod is None:
        raise SystemExit(f"no Qwen3VLTextAttention with layer_idx={layer}")
    per_tok = PROBE.capture_layer_attention(model, attn_mod, inputs, prompt_len,
                                            comp_ids, list(heads), device)
    if per_tok.shape[-1] != gh * gw:
        raise RuntimeError(f"direct map has {per_tok.shape[-1]} patches, grid is {gh}x{gw}")
    out = np.zeros((len(steps), gh, gw), dtype=np.float32)
    for si, (_t, a, b) in enumerate(steps):
        # mean over the step's tokens, then over the two rewarded heads -- the reward's
        # own reduction (token_reduction="mean", then a head mean).
        out[si] = per_tok[:, a:b, :].mean(axis=1).mean(axis=0).reshape(gh, gw)
    return np.maximum(out, 0.0)


# ---------------------------------------------------------------------------
# maps 2/3: the rollout, heads merged by the mean or by value norm
# ---------------------------------------------------------------------------
def rollout_map(model, inputs, ids, prompt_len, steps, gh, gw, weighting, args, device):
    img_cols = (inputs["input_ids"][0] == IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0]
    if img_cols.numel() != gh * gw:
        raise RuntimeError(f"{img_cols.numel()} image tokens, grid is {gh}x{gw}")
    need = sorted({prompt_len + p for _t, a, b in steps for p in range(a, b)})
    pos = {p: i for i, p in enumerate(need)}
    rows = torch.tensor(need, device=device)
    seq = ids.shape[1]

    engine = FC.RolloutFlow(model, weighting, args.alpha, args.chunk)
    try:
        engine.arm(seq, img_cols, rows,
                   IV.causal_mask(seq, next(model.parameters()).dtype, device))
        with torch.no_grad():
            model(**FC.build_forward(inputs, ids, prompt_len), use_cache=False)
        snaps = torch.stack(engine.snaps).cpu().numpy()      # [n_layers, n_rows, M]
    finally:
        engine.disarm()
        engine.close()

    li = args.rollout_layer if args.rollout_layer >= 0 else snaps.shape[0] - 1
    if not 0 <= li < snaps.shape[0]:
        raise SystemExit(f"--rollout-layer {args.rollout_layer} outside 0..{snaps.shape[0]-1}")
    sal = snaps[li]
    out = np.zeros((len(steps), gh, gw), dtype=np.float32)
    for si, (_t, a, b) in enumerate(steps):
        sel = [pos[prompt_len + p] for p in range(a, b)]
        out[si] = sal[sel].mean(axis=0).reshape(gh, gw)
    return out


# ---------------------------------------------------------------------------
# map 4: the gradient w.r.t. the pixels
#
# `pixel_regroup` moved to trl/grad_maps.py (imported above as GM and re-exported), so
# the map drawn here and the one the GRPO gradient reward scores are the same function.
# ---------------------------------------------------------------------------
def grad_map(model, processor, inputs, ids, prompt_len, steps, gh, gw, args, device):
    ip = processor.image_processor
    ps = int(getattr(ip, "patch_size", 16))
    tps = int(getattr(ip, "temporal_patch_size", 2))
    grid = inputs["image_grid_thw"][0].tolist()

    pv = inputs["pixel_values"].detach().float().requires_grad_(True)
    fwd = FC.build_forward(inputs, ids, prompt_len)
    # The leaf is fp32 and the cast into the model's dtype is part of the graph, so the
    # gradient that arrives back at `pv` is fp32 even though the tower runs in bf16.
    fwd["pixel_values"] = pv.to(inputs["pixel_values"].dtype)

    n_chain = ids.shape[1] - prompt_len
    out = np.zeros((len(steps), gh, gw), dtype=np.float32)
    with torch.enable_grad():
        res = model(**fwd, use_cache=False, logits_to_keep=n_chain + 1)
        logits = res.logits[0]
        # logits_to_keep trimmed the leading rows: absolute position p-1 predicts the
        # token at p and now sits at index p - prompt_len.
        for si, (_t, a, b) in enumerate(steps):
            acc = torch.zeros(gh, gw, dtype=torch.float32, device=device)
            for p in range(prompt_len + a, prompt_len + b):
                row = logits[p - prompt_len].float()
                tgt = int(ids[0, p])
                f = row[tgt] if args.grad_target == "logit" else row.log_softmax(-1)[tgt]
                (g,) = torch.autograd.grad(f, pv, retain_graph=True)
                acc += pixel_regroup(g, grid, ps, tps)
                del g
            out[si] = (acc / max(b - a, 1)).cpu().numpy()
    del res, logits
    return out


# ---------------------------------------------------------------------------
# map 5: GLIMPSE -- gradient-weighted attention, adaptively propagated
#
# arXiv 2506.18985v1, sections 3.2-3.5. The equation numbers below are the paper's.
# The algebra is split into three pure functions so test_glimpse_cpu.py can check it
# against a naive [N, N] reference with no model and no GPU.
# ---------------------------------------------------------------------------
def glimpse_edge_matrix(a, g, temp: float, eps: float = 1e-12):
    """[H, N, N] attention + its gradient -> (E, ||sum_h g^h||_1), eqs 5-8.

        G^h = ReLU(g^h * A^h)                                            (5)
        w^h = softmax_h( (1/lambda) * sum_ij G^h / sum_ij ReLU(g^h) )    (6)
        E   = sum_h w^h G^h, row-normalised                              (7)

    Eq 6 divides by the head's total positive gradient mass, so `w` ranks heads by how
    much of the gradient they actually attend *along* rather than by gradient magnitude
    -- a head whose positive gradient sits where it does not attend is demoted.

    The head loop keeps the peak at one [N, N] fp32 temporary. The vectorised form would
    need [H, N, N] in fp32 on top of the [H, N, N] the graph already holds, which at
    H=32 and a 1700-token sequence is another 370 MB per layer, times 36 layers, per
    backward.
    """
    h, n, _ = a.shape
    # `_at_least_fp32` rather than `.float()`: bf16 attention has to be promoted, but a
    # float64 reference must not be silently demoted to meet the code halfway.
    up = GM._at_least_fp32
    dt = torch.promote_types(up(a[:1, :1, :1]).dtype, up(g[:1, :1, :1]).dtype)
    num = torch.zeros(h, dtype=dt, device=a.device)
    den = torch.zeros(h, dtype=dt, device=a.device)
    gsum = torch.zeros(n, n, dtype=dt, device=a.device)
    for i in range(h):
        gi = up(g[i]).to(dt)
        num[i] = torch.relu(gi * up(a[i]).to(dt)).sum()
        den[i] = torch.relu(gi).sum()
        gsum += gi
    w = torch.softmax((num / den.clamp_min(eps)) / temp, dim=0)

    e = torch.zeros(n, n, dtype=dt, device=a.device)
    for i in range(h):
        e += w[i] * torch.relu(up(g[i]).to(dt) * up(a[i]).to(dt))
    e = e / e.sum(-1, keepdim=True).clamp_min(eps)
    # eq 8 sums the heads BEFORE the norm, so a layer whose heads pull against each other
    # is scored as the small net force it is, not as the large forces it is made of.
    return e, gsum.abs().sum()


def glimpse_layer_alphas(g_l1, layer_ids, depth_temp: float, eps: float = 1e-12):
    """Per-layer propagation weights, eqs 9-10: gradient evidence x an exponential
    depth prior, `alpha_l = g_l s_l / sum_k g_k s_k` with `s_l = softmax(lambda_d(l+1))`.

    `layer_ids` are the model's own indices. `softmax` is shift-invariant and the
    propagated set is always a contiguous slice off the top of the stack, so re-basing
    them at 0 would give the same `s`; they are passed in to keep this readable next to
    eq 9, not because the offset changes anything.

    The paper's lambda_d = 0.2 was tuned on a 64-layer backbone, where it makes the
    prior fall by e every 5 layers -- 7.8% of the depth. On this 36-layer model the same
    number spans 14% of the depth, so `--glimpse-depth-temp 0.36` is the setting that
    matches the paper's *shape*, and 0.2 the setting that matches its *text*. The
    ablation calls this the single most important component (removing it takes their NSS
    from 1.014 to -0.210), which is exactly why the mismatch is worth naming.
    """
    g = GM._at_least_fp32(torch.stack([torch.as_tensor(v) for v in g_l1]))
    ell = torch.as_tensor(list(layer_ids), dtype=g.dtype, device=g.device) + 1.0
    s = torch.softmax(depth_temp * ell, dim=0)
    tot = g.sum()
    # A backward that produced no positive gradient anywhere would otherwise make every
    # alpha NaN and silently blank the map; fall back to the depth prior alone.
    g = torch.where(tot > 0, g / tot.clamp_min(eps), torch.full_like(g, 1.0 / g.numel()))
    a = g * s
    return a / a.sum().clamp_min(eps)


def glimpse_propagate(row: int, mats, alphas):
    """Row `row` of `R = prod_{l=L..first} (2I + alpha_l E_l)`, eqs 11-13, scaled by 2^-L.

    Eq 13 is `R <- R + L_l R` with `L_l = I + alpha_l E_l` (eq 12), i.e. `R <- (2I +
    alpha_l E_l) R`: the identity path DOUBLES at every layer, so over 36 layers the row
    would grow by 2^36 before anything is read off it. Every quantity the method takes
    from R is a ratio -- `beta` normalises over the tokens, the map is normalised for
    display -- and the factor is identical for every token, so it is folded in per layer
    as `v + (alpha/2) v E` instead.

    R is only ever read one row at a time, and `v (2I + alpha E)` is linear in `v`, so the
    row is carried through the product as a vector: O(L*N^2) rather than the O(L*N^3) of
    the matrix form. The product applies the LAST layer's factor to the row first, hence
    the reversed loop.
    """
    n = mats[0].shape[0]
    v = torch.zeros(n, dtype=mats[0].dtype, device=mats[0].device)
    v[row] = 1.0
    for i in range(len(mats) - 1, -1, -1):
        v = v + (0.5 * alphas[i]) * (v @ mats[i])
    return v


def glimpse_token_weight(conf, align, mode: str):
    """The token's weight in the aggregation, eq 18 -- `beta_t ~ p_t * a_t`.

    `a_t` is the token's alignment to the PROMPT (eq 14) even though the map being built
    is the visual one: eq 17 crosses them on purpose, so a token earns its say in *where
    the model looked* by being about the question, not by being visually grounded, which
    would be circular. `mode` reproduces the paper's token-saliency ablation.
    """
    if mode == "full":
        return conf * align
    if mode == "confidence":
        return conf
    if mode == "prompt":
        return align
    if mode == "uniform":
        return torch.ones_like(align)
    raise ValueError(f"token weight {mode!r} not in full|confidence|prompt|uniform")


def _find_subseq(hay: list[int], needle: list[int]) -> int:
    """Last start index of `needle` in `hay`, or -1."""
    if not needle or len(needle) > len(hay):
        return -1
    for s in range(len(hay) - len(needle), -1, -1):
        if hay[s:s + len(needle)] == needle:
            return s
    return -1


def prompt_positions(tok, question: str, prompt_ids: list[int], img_positions):
    """-> (positions, how). The prompt columns `P` of eq 14: the question's own tokens.

    The paper's `P` is the user prompt. Ours is a chat template wrapped around it -- a
    system prompt about the reasoning format, then the image, then the question -- and
    `a_t` is a MEAN over `P`, so folding in the boilerplate dilutes precisely the signal
    the weight exists to carry. The question is located by matching its own tokenisation
    inside the prompt, from the right (the template repeats nothing else there). A
    leading space can merge differently at a template boundary, so the match is retried
    without the first token; if both fail, every non-image prompt token is used and
    meta.json records which of the three happened.
    """
    q = list(tok(question, add_special_tokens=False)["input_ids"])
    for cand, how in ((q, "question"), (q[1:], "question_less_first")):
        s = _find_subseq(prompt_ids, cand)
        if s >= 0:
            return list(range(s, s + len(cand))), how
    img = set(int(i) for i in img_positions)
    return [i for i in range(len(prompt_ids)) if i not in img], "prompt_minus_image"


@contextlib.contextmanager
def eager_one_attention(attn_mod):
    """Run ONE attention module in eager, so its softmax weights exist and are in the graph.

    The flag lives on the shared text config, so flipping it is global -- but only the
    replayed layer runs inside the window, and it is handed an explicit causal mask, so
    nothing else can observe either the flag or the mask. This is the same trick
    `overlap_probe.capture_layer_attention` uses to read one layer's attention out of an
    otherwise-sdpa forward; maps 1-3 have been built on it all along.
    """
    cfg = attn_mod.config
    prev = cfg._attn_implementation
    try:
        cfg._attn_implementation = "eager"
        yield
    finally:
        cfg._attn_implementation = prev


@contextlib.contextmanager
def checkpointing_off(model):
    """Gradient checkpointing recomputes each layer during the backward, which would fire
    the capture hooks a second time, on tensors that are not the ones the graph holds."""
    on = bool(getattr(model, "is_gradient_checkpointing", False))
    if on:
        model.gradient_checkpointing_disable()
    try:
        yield
    finally:
        if on:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})


class GlimpseGradCache:
    """One sdpa forward, then `dz/dA` for one layer at a time.

    Holding every layer's `[H, N, N]` attention in the graph at once -- what an all-eager
    forward does -- costs ~9 KB of GPU memory per (query, key) pair across the 36 layers.
    Measured on an A100-80GB: 26 GiB of activations at `N` = 1600, 52 GiB at 2400, OOM by
    3600. Nothing else in this file pays that. Maps 1-3 re-run a single layer in eager
    inside an otherwise-sdpa forward (`overlap_probe.capture_layer_attention`), and map 4
    takes its gradient through sdpa, which never materialises `[H, N, N]` at all. Map 5 is
    now built the same way, in two stages per target token:

      1. one backward on the sdpa graph gives `dz/dh_l`, the gradient w.r.t. every
         propagated layer's OUTPUT hidden state -- `[N, d]` each, ~10 MB, not 400 MB;
      2. layer by layer, that layer alone is re-run in eager from its own recorded input
         and `dz/dh_l` is pushed into it, which is `dz/dA_l`. `E_l` is folded out and the
         `[H, N, N]` freed before the next layer is touched.

    `dz/dA_l = dz/dh_{l+1} . dh_{l+1}/dA_l` is the chain rule, not an approximation. The
    layer is a function of its own input, and that input is RECORDED rather than
    recomputed, so the deepstack features the text model adds between layers are already
    inside the recorded value and need no special handling here.

    What still scales as `N^2` is the `[N, N]` fp32 `E_l` kept per layer while the row is
    propagated: 36 x 4 bytes per pair against the graph's ~9000, a factor of ~60.

    Two hooks. A forward hook per propagated decoder layer records its input, its kwargs
    and its output. A forward PRE-hook on the first propagated layer detaches its input
    into a leaf -- every weight is frozen, so without it the forward would build no graph
    and there would be nothing to differentiate.
    """

    def __init__(self, model, first_layer: int, temp: float):
        self.first_layer, self.temp = int(first_layer), float(temp)
        self.layers, self.handles, self.mods = [], [], {}
        self.h_in, self.kw, self.out = {}, {}, {}
        self.leaf = None
        self.mask = None
        self._reentry = False
        self._checked = self._replay_checked = False
        cut = 0
        for m in model.modules():
            if type(m).__name__ != "Qwen3VLTextDecoderLayer":
                continue
            li = int(m.self_attn.layer_idx)
            if li < self.first_layer:
                continue
            self.mods[li] = m
            self.layers.append(li)
            self.handles.append(
                m.register_forward_hook(self._record(li), with_kwargs=True))
            if li == self.first_layer:
                self.handles.append(
                    m.register_forward_pre_hook(self._cut, with_kwargs=True))
                cut += 1
        self.layers.sort()
        if not self.layers or cut != 1:
            self.close()
            raise RuntimeError(f"glimpse: {len(self.layers)} decoder layers at or above "
                               f"{self.first_layer} and {cut} cut points (want 1)")

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        self.release()

    def release(self):
        self.h_in, self.kw, self.out = {}, {}, {}
        self.leaf = self.mask = None

    def check(self):
        """Every propagated layer must have recorded a forward, and the cut must have
        fired -- a layer that silently failed to fire would drop out of the product."""
        if sorted(self.out) != self.layers:
            raise RuntimeError(f"glimpse: {len(self.out)} of {len(self.layers)} layers "
                               "recorded a forward")
        if self.leaf is None:
            raise RuntimeError("glimpse: the leaf cut did not fire")

    def layer_grads(self, scalar):
        """-> {layer: dz/d(that layer's output)}, from ONE backward over the sdpa graph."""
        outs = [self.out[li] for li in self.layers]
        gs = torch.autograd.grad(scalar, outs, retain_graph=True)
        return dict(zip(self.layers, gs))

    def edge(self, li: int, g_out):
        """Replay layer `li` in eager and turn `dz/dh` into `(E_l, ||sum_h g^h||_1)`."""
        mod, cap = self.mods[li], {}

        def grab(module, args, kwargs, output):
            cap["a"] = output[1]
            return None

        kw = dict(self.kw[li])
        kw.pop("hidden_states", None)
        kw["attention_mask"] = self.mask      # eager with no mask is silently bidirectional
        kw["past_key_values"] = None
        kw["use_cache"] = False
        # The recorded input is detached, so without a leaf here the replay would build no
        # graph either -- same reason as the cut.
        hs = self.h_in[li].detach().requires_grad_(True)
        handle = mod.self_attn.register_forward_hook(grab, with_kwargs=True)
        self._reentry = True                  # the replay re-enters this layer's own hook
        try:
            with eager_one_attention(mod.self_attn), torch.enable_grad():
                out = mod(hs, **kw)
                a = cap.get("a")
                if a is None:
                    raise RuntimeError(f"glimpse: layer {li} returned no attention weights "
                                       "-- it did not run in eager")
                self._check_causal(a)
                self._check_replay(li, out)
                (g,) = torch.autograd.grad(out, a, grad_outputs=g_out)
        finally:
            self._reentry = False
            handle.remove()
        e, g1 = glimpse_edge_matrix(a[0].detach(), g[0], self.temp)
        del a, g, out, cap
        return e, g1

    def _cut(self, module, args, kwargs):
        if self._reentry:            # the replay brings its own leaf; leave it alone
            return None
        hs = args[0] if args else kwargs["hidden_states"]
        leaf = hs.detach().requires_grad_(True)
        self.leaf = leaf
        if args:
            return (leaf,) + tuple(args[1:]), kwargs
        kw = dict(kwargs)
        kw["hidden_states"] = leaf
        return args, kw

    def _record(self, li: int):
        def hook(module, args, kwargs, output):
            if self._reentry:
                return None
            hs = args[0] if args else kwargs["hidden_states"]
            self.h_in[li] = hs.detach()
            self.kw[li] = dict(kwargs)
            self.out[li] = output if isinstance(output, torch.Tensor) else output[0]
            return None
        return hook

    def _check_causal(self, aw):
        """Query 0 may only see key 0. Eager attention with no mask is silently
        bidirectional -- every map would be wrong and nothing would raise."""
        if self._checked:
            return
        self._checked = True
        leak = float(aw[0, :, 0, 1:].detach().abs().sum())
        if leak > 1e-3:
            raise RuntimeError(f"glimpse: attention is not causal (row 0 puts {leak:.3g} "
                               "outside column 0) -- the causal mask did not reach eager")

    def _check_replay(self, li: int, out):
        """The replay must reproduce the layer's own output. A dropped kwarg or a wrong
        mask would otherwise yield a plausible map built on the wrong tensor. The tolerance
        is loose on purpose: eager and sdpa differ in the last bf16 bits, a replay mistake
        differs by order 1."""
        if self._replay_checked:
            return
        self._replay_checked = True
        ref = self.out[li].detach().float()
        rel = float((out.detach().float() - ref).abs().max()) / max(float(ref.abs().max()),
                                                                    1e-6)
        if rel > 0.05:
            raise RuntimeError(f"glimpse: the eager replay of layer {li} differs from the "
                               f"forward by {rel:.3g} relative -- the recorded kwargs or "
                               "the causal mask are wrong")


def glimpse_map(model, processor, inputs, ids, prompt_len, steps, gh, gw, question,
                args, device):
    """-> ([n_steps, gh, gw] float32, the settings that produced it).

    One sdpa forward, then per target token one backward for the layer gradients and one
    eager replay per propagated layer. The relevance matrix of eqs 11-13 is built from
    `d z_t / d A`, which is a different backward for every target token; the forward and
    its graph are retained across the whole sample, so the forward is paid once.

    Per token this costs one backward plus one layer-forward per layer -- roughly one
    extra full forward -- where an all-eager graph would cost one backward and tens of GiB.
    See GlimpseGradCache for the measurements.
    """
    img_cols = (inputs["input_ids"][0] == IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0]
    if img_cols.numel() != gh * gw:
        raise RuntimeError(f"{img_cols.numel()} image tokens, grid is {gh}x{gw}")
    prompt_ids = inputs["input_ids"][0].tolist()
    pos, how = prompt_positions(processor.tokenizer, question, prompt_ids,
                                img_cols.tolist())
    prompt_idx = torch.tensor(pos, device=device)

    n_layers = sum(1 for m in model.modules()
                   if type(m).__name__ == "Qwen3VLTextAttention")
    keep = min(n_layers, max(1, int(round(args.glimpse_layer_frac * n_layers))))
    first = n_layers - keep

    # The row that produced the token at absolute position p is p-1 -- both the logit row
    # and, for the same reason, the relevance row. See the module docstring.
    rows = [prompt_len + a - 1 + i for _t, a, b in steps for i in range(b - a)]
    rows_t = torch.tensor(rows, device=device)
    targets = torch.cat([ids[0, prompt_len + a: prompt_len + b] for _t, a, b in steps])

    out = np.zeros((len(steps), gh, gw), dtype=np.float32)
    fell_back = []
    cap = GlimpseGradCache(model, first, args.glimpse_temp)
    try:
        with checkpointing_off(model), torch.enable_grad():
            res = model(**FC.build_forward(inputs, ids, prompt_len), use_cache=False,
                        logits_to_keep=rows_t)
            z = res.logits[0].float()
            del res
            cap.check()
            # The forward ran under sdpa, which is entitled to build no mask at all; the
            # eager replay needs an explicit one or it would attend bidirectionally.
            cap.mask = IV.causal_mask(ids.shape[1], next(model.parameters()).dtype, device)
            # `_row_scalars` is the reward's own definition of the per-token scalar, so
            # `--glimpse-target` means here exactly what `--grad_target` means there.
            f = GM._row_scalars(z, targets, args.glimpse_target)
            conf = GM._row_scalars(z, targets, "logprob").detach().exp()      # eq 16

            k = 0
            for si, (_t, a, b) in enumerate(steps):
                acc = torch.zeros(img_cols.numel(), dtype=torch.float32, device=device)
                plain = torch.zeros_like(acc)
                wsum = torch.zeros((), dtype=torch.float32, device=device)
                for _ in range(b - a):
                    g_out = cap.layer_grads(f[k])
                    mats, g_l1 = [], []
                    for li in cap.layers:
                        # pop, so each [N, d] gradient dies as its layer is consumed
                        e, g1 = cap.edge(li, g_out.pop(li))
                        mats.append(e)
                        g_l1.append(g1)
                    del g_out
                    alphas = glimpse_layer_alphas(g_l1, cap.layers, args.glimpse_depth_temp)
                    v = glimpse_propagate(rows[k], mats, alphas)
                    w = glimpse_token_weight(conf[k], v[prompt_idx].mean(),
                                             args.glimpse_token_weight)
                    acc += w * v[img_cols]
                    plain += v[img_cols]
                    wsum += w
                    del mats, alphas, v
                    k += 1
                # eq 22, with `Y` restricted to this step so the map stays comparable to
                # the other four. A step whose tokens all weigh zero -- no positive
                # gradient anywhere on the prompt -- would otherwise render as a blank.
                if float(wsum) > 0:
                    m = acc / wsum
                else:
                    fell_back.append(si)
                    m = plain / max(b - a, 1)
                out[si] = m.reshape(gh, gw).cpu().numpy()
    finally:
        cap.close()
    info = {"prompt_tokens": how, "n_prompt_tokens": len(pos), "first_layer": first,
            "n_layers": n_layers, "temp": args.glimpse_temp,
            "depth_temp": args.glimpse_depth_temp, "target": args.glimpse_target,
            "token_weight": args.glimpse_token_weight, "unweighted_steps": fell_back}
    return out, info


# ---------------------------------------------------------------------------
# stage: scan
# ---------------------------------------------------------------------------
def sample_dir(out: Path, i: int, row_index: int) -> Path:
    return out / "samples" / f"sample_{i:03d}_row{row_index:06d}"


def scan(args, device):
    out = Path(args.out_dir)
    (out / "samples").mkdir(parents=True, exist_ok=True)

    rows = PROBE.load_samples(args.dataset, args.n_samples, args.seed,
                              cache_tag=f"_sv{args.shard}", split=args.split)
    todo = list(enumerate(rows))[args.shard::args.num_shards]
    print(f"[scan] shard {args.shard}/{args.num_shards}: {len(todo)} of {len(rows)} samples",
          flush=True)
    if not todo:
        return

    methods = [m for m in args.methods.split(",") if m]
    bad = [m for m in methods if m not in METHODS]
    if bad:
        raise SystemExit(f"unknown method(s) {bad}; pick from {list(METHODS)}")
    heads = [int(h) for h in args.direct_heads.split(",") if h != ""]

    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    processor, model = PROBE.load_model(args.base_model, args.adapter or None, device,
                                        args.attn_impl)
    model.requires_grad_(False)          # only `pixel_values` is ever differentiated
    if args.grad_checkpointing and "grad" in methods:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    clf = OSTEPS.OverlapStepsClassifier.load(args.steps_ckpt, device=device)
    tok = processor.tokenizer

    for i, row in todo:
        d = sample_dir(out, i, row["row_index"])
        if (d / "maps.npz").exists() and not args.overwrite:
            print(f"[scan] {d.name}: already done", flush=True)
            continue
        d.mkdir(parents=True, exist_ok=True)
        try:
            note = scan_one(model, processor, tok, clf, row, d, methods, heads,
                            args, device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            note = "FAILED: out of memory"
        except Exception as e:                    # one bad sample must not kill a shard
            note = f"FAILED: {type(e).__name__}: {e}"
        print(f"[scan] {d.name}: {note}", flush=True)


def scan_one(model, processor, tok, clf, row, d: Path, methods, heads, args, device):
    image = row["image"]
    inputs, prompt_len, comp_ids = IV.greedy_chain(
        processor, model, image, row["question"], args.max_new_tokens, device)

    text, seg, reason = segment_case(tok, clf, row["question"], comp_ids)
    image.save(d / "original.png")
    (d / "question.txt").write_text(row["question"] + "\n")
    (d / "generation.txt").write_text(text + "\n")
    meta = {"row_index": row["row_index"], "dataset": row.get("dataset"),
            "question": row["question"], "gt_answer": row.get("gt_answer"),
            "generation": text, "methods": [], "steps": [],
            "direct_layer": args.direct_layer, "direct_heads": heads,
            "grad_target": args.grad_target, "rollout_layer": args.rollout_layer,
            "image_size": list(image.size)}
    if seg is None:
        meta["dropped"] = reason
        (d / "meta.json").write_text(json.dumps(meta, indent=2))
        return f"no maps ({reason})"
    steps, format_ok = seg
    meta["format_ok"] = format_ok
    if args.max_steps and len(steps) > args.max_steps:
        steps = steps[: args.max_steps]

    gh = int(inputs["image_grid_thw"][0, 1].item()) // 2
    gw = int(inputs["image_grid_thw"][0, 2].item()) // 2
    ids = torch.tensor([inputs["input_ids"][0].tolist() + list(comp_ids)], device=device)

    maps = {}
    for m in methods:
        if m == "direct":
            maps[m] = direct_map(model, inputs, prompt_len, comp_ids, steps, gh, gw,
                                 args.direct_layer, heads, device)
        elif m.startswith("rollout_"):
            maps[m] = rollout_map(model, inputs, ids, prompt_len, steps, gh, gw,
                                  m.split("_", 1)[1], args, device)
        elif m == "grad":
            maps[m] = grad_map(model, processor, inputs, ids, prompt_len, steps, gh, gw,
                               args, device)
        elif m == "glimpse":
            maps[m], meta["glimpse"] = glimpse_map(model, processor, inputs, ids,
                                                   prompt_len, steps, gh, gw,
                                                   row["question"], args, device)
        torch.cuda.empty_cache()

    meta["methods"] = list(maps)
    meta["grid"] = [gh, gw]
    meta["steps"] = [{"index": si, "text": t, "tok_a": int(a), "tok_b": int(b),
                      "n_tokens": int(b - a)}
                     for si, (t, a, b) in enumerate(steps)]
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    np.savez_compressed(d / "maps.npz", **{m: v for m, v in maps.items()})
    return f"{len(steps)} observe step(s), maps: {','.join(maps)}"


# ---------------------------------------------------------------------------
# stage: render
# ---------------------------------------------------------------------------
def normalize_map(m, mode: str, lo: float, hi: float):
    m = np.asarray(m, dtype=np.float64)
    if mode == "rank":
        flat = m.ravel()
        order = flat.argsort(kind="stable")
        ranks = np.empty(flat.size, dtype=np.float64)
        ranks[order] = np.arange(flat.size, dtype=np.float64)
        return (ranks / max(flat.size - 1, 1)).reshape(m.shape)
    if mode == "minmax":
        a, b = float(m.min()), float(m.max())
    else:                                          # percentile (default)
        a, b = float(np.percentile(m, lo)), float(np.percentile(m, hi))
    if not b > a:
        return np.zeros_like(m)
    return np.clip((m - a) / (b - a), 0.0, 1.0)


def overlay(img, m, cmap, args):
    """The image with `m` painted over it, upsampled from the patch grid."""
    from PIL import Image

    x = normalize_map(m, args.norm, args.norm_lo, args.norm_hi)
    rgb = (np.asarray(cmap(x))[..., :3] * 255).astype(np.uint8)
    resample = Image.NEAREST if args.upsample == "nearest" else Image.BILINEAR
    heat = Image.fromarray(rgb).resize(img.size, resample)
    if args.overlay_mode == "alpha":
        # transparency tracks saliency: the picture stays visible where nothing fires
        a = Image.fromarray((x * 255 * args.overlay_alpha).astype(np.uint8)).resize(
            img.size, resample)
        base = img.convert("RGB").copy()
        base.paste(heat, (0, 0), a)
        return base
    return Image.blend(img.convert("RGB"), heat, args.overlay_alpha)


def caption(img, text: str):
    from PIL import Image, ImageDraw, ImageFont

    bar = 16
    canvas = Image.new("RGB", (img.size[0], img.size[1] + bar), (16, 16, 16))
    canvas.paste(img, (0, bar))
    ImageDraw.Draw(canvas).text((3, 3), text[: max(1, img.size[0] // 6)],
                                fill=(235, 235, 235), font=ImageFont.load_default())
    return canvas


def contact_sheet(panels, pad: int = 6):
    from PIL import Image

    w = sum(p.size[0] for p in panels) + pad * (len(panels) + 1)
    h = max(p.size[1] for p in panels) + 2 * pad
    sheet = Image.new("RGB", (w, h), (16, 16, 16))
    x = pad
    for p in panels:
        sheet.paste(p, (x, pad))
        x += p.size[0] + pad
    return sheet


def render_sample(d: Path, cmap, args):
    from PIL import Image

    meta = json.loads((d / "meta.json").read_text())
    if not (d / "maps.npz").exists():
        return meta, 0
    img = Image.open(d / "original.png").convert("RGB")
    z = np.load(d / "maps.npz")
    methods = [m for m in METHODS if m in z]
    steps = meta["steps"]
    gm = meta.get("glimpse", {})
    label = {m: TITLES[m].format(layer=meta.get("direct_layer"),
                                 heads="/".join(str(h) for h in meta.get("direct_heads", [])),
                                 rl=("last" if meta.get("rollout_layer", -1) < 0
                                     else meta["rollout_layer"]),
                                 target=meta.get("grad_target", "logprob"),
                                 gfirst=gm.get("first_layer", "?"),
                                 gdepth=gm.get("depth_temp", "?"))
             for m in METHODS}

    def draw(dst: Path, get):
        dst.mkdir(parents=True, exist_ok=True)
        panels = [caption(img, "original")]
        for m in methods:
            o = overlay(img, get(m), cmap, args)
            o.save(dst / f"sal_{m}.png")
            panels.append(caption(o, label[m]))
        if not args.no_contact_sheet:
            contact_sheet(panels).save(dst / "contact_sheet.png")

    # sample level: the maps averaged over every observe step
    draw(d, lambda m: z[m].mean(axis=0))
    if not args.no_per_step:
        for si, st in enumerate(steps):
            sd = d / "steps" / f"step{si:02d}"
            sd.mkdir(parents=True, exist_ok=True)
            (sd / "step.txt").write_text(
                f"tokens [{st['tok_a']}, {st['tok_b']}) ({st['n_tokens']})\n\n{st['text']}\n")
            draw(sd, lambda m, si=si: z[m][si])
    return meta, len(steps)


def render(args):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import cm

    # colormaps[] is the modern spelling; cm.get_cmap was removed in matplotlib 3.9.
    cmap = (matplotlib.colormaps[args.cmap] if hasattr(matplotlib, "colormaps")
            else cm.get_cmap(args.cmap))
    out = Path(args.out_dir)
    dirs = sorted((out / "samples").glob("sample_*"))
    if not dirs:
        raise SystemExit(f"no samples under {out/'samples'} -- run --stage scan first")
    cards = []
    for d in dirs:
        if not (d / "meta.json").exists():
            continue
        meta, n = render_sample(d, cmap, args)
        cards.append((d, meta, n))
        print(f"[render] {d.name}: {n} step(s)", flush=True)
    write_index(out, cards, args)
    print(f"[render] {len(cards)} sample(s) -> {out/'index.html'}")


def write_index(out: Path, cards, args):
    e = html.escape
    parts = [
        "<!doctype html><meta charset='utf-8'><title>saliency maps</title>",
        "<style>body{background:#111;color:#ddd;font:13px/1.5 -apple-system,sans-serif;"
        "margin:24px}h2{margin:32px 0 4px}img{max-width:100%;border:1px solid #333}"
        "pre{white-space:pre-wrap;background:#181818;padding:8px;border-radius:4px}"
        ".q{color:#9cf}.s{margin:10px 0 10px 18px;border-left:2px solid #333;padding-left:12px}"
        "a{color:#9cf}</style>",
        f"<h1>saliency maps &mdash; {len(cards)} samples</h1>",
        f"<p>norm={e(args.norm)} ({args.norm_lo}&ndash;{args.norm_hi} pct), cmap={e(args.cmap)}, "
        f"overlay={e(args.overlay_mode)} alpha={args.overlay_alpha}. "
        "Left to right in each strip: original, then the maps that were scanned, in "
        "docs/saliency-maps.md order.</p>",
    ]
    for d, meta, n in cards:
        r = d.name
        parts.append(f"<h2>{e(r)} &mdash; {e(str(meta.get('dataset')))}</h2>")
        parts.append(f"<p class='q'>{e(meta['question'])}</p>")
        parts.append(f"<p>gold: <b>{e(str(meta.get('gt_answer')))}</b>"
                     + ("" if meta.get("format_ok", True) else " &nbsp;<i>(malformed completion)</i>")
                     + (f" &nbsp;<i>({e(meta['dropped'])})</i>" if meta.get("dropped") else "")
                     + "</p>")
        parts.append(f"<pre>{e(meta.get('generation',''))}</pre>")
        if (d / "contact_sheet.png").exists():
            parts.append(f"<p><i>mean over all {n} observe step(s)</i><br>"
                         f"<img src='samples/{r}/contact_sheet.png'></p>")
        for st in meta.get("steps", []):
            sp = d / "steps" / f"step{st['index']:02d}" / "contact_sheet.png"
            if sp.exists():
                parts.append(
                    f"<div class='s'><b>step {st['index']}</b> "
                    f"({st['n_tokens']} tokens)<br>{e(st['text'])}<br>"
                    f"<img src='samples/{r}/steps/step{st['index']:02d}/contact_sheet.png'></div>")
    (out / "index.html").write_text("\n".join(parts))


# ---------------------------------------------------------------------------
# stage: selftest -- the pixel -> token regrouping, on the real processor
# ---------------------------------------------------------------------------
def selftest(args):
    from PIL import Image
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.base_model, padding_side="left")
    ip = processor.image_processor
    ps, tps = int(ip.patch_size), int(ip.temporal_patch_size)
    tok_px = ps * 2                       # merge_size 2: one LM token is 2x2 patches

    # Mid-grey, not black: image_mean/std are 0.5/0.5, so black normalises to -1 and a
    # black background would carry as much magnitude as a white square. Grey is ~0.
    rows_t, cols_t = 12, 16               # 384x512 -- above min_pixels, so no resize
    img = Image.new("RGB", (tok_px * cols_t, tok_px * rows_t), (128, 128, 128))
    ti, tj = 7, 11                        # the one bright token
    for y in range(ti * tok_px, (ti + 1) * tok_px):
        for x in range(tj * tok_px, (tj + 1) * tok_px):
            img.putpixel((x, y), (255, 255, 255))

    text = PROBE.build_prompt(processor, "test")
    inputs = processor(text=[text], images=[[img]], return_tensors="pt", padding=True,
                       padding_side="left", add_special_tokens=False)
    grid = inputs["image_grid_thw"][0].tolist()
    gh, gw = grid[1] // 2, grid[2] // 2
    if (gh, gw) != (rows_t, cols_t):
        raise SystemExit(f"selftest FAILED: processor resized the image; grid is {gh}x{gw}, "
                         f"expected {rows_t}x{cols_t}. Pick a size the processor keeps.")

    # pixel_values as a stand-in for a gradient: the regrouping is a pure reindexing,
    # so feeding it the values themselves must localise the bright token exactly.
    pv = inputs["pixel_values"].float()
    m = pixel_regroup(pv, grid, ps, tps).numpy()
    hot = np.unravel_index(int(m.argmax()), m.shape)
    n_img = int((inputs["input_ids"][0] == IMAGE_TOKEN_ID).sum())
    ok = hot == (ti, tj) and n_img == gh * gw
    share = float(m[ti, tj] / m.sum())
    print(f"[selftest] grid {gh}x{gw}, {n_img} image tokens, patch={ps} temporal={tps}")
    print(f"[selftest] brightest token {hot}, expected ({ti}, {tj}); "
          f"it holds {share:.1%} of the total")
    if not ok:
        raise SystemExit("[selftest] FAILED: pixel->token regrouping does not match the "
                         "processor's layout. The grad map would be scrambled.")
    print("[selftest] OK")


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", default="scan", choices=["scan", "render", "selftest"])
    p.add_argument("--out-dir", default="")
    p.add_argument("--base-model", default=str(repo_path(
        "checkpoint/coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged")))
    p.add_argument("--adapter", default="")
    p.add_argument("--dataset", default=str(repo_path("cold_data/grpo_sets/val_natural")))
    p.add_argument("--split", default="all", choices=["train", "holdout", "all"],
                   help="'all' for the val_* sets (they have no holdout to carve)")
    p.add_argument("--n-samples", type=int, default=20)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    # glimpse replays one layer at a time in eager on top of this; `--attn-impl eager`
    # would put all 36 layers' [H, N, N] back in the graph and undo that.
    p.add_argument("--attn-impl", default="sdpa")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--max-steps", type=int, default=0,
                   help="cap observe steps per sample (0 = all)")
    p.add_argument("--steps-ckpt", default=os.environ.get(
        "OVERLAP_STEPS_CKPT", str(repo_path("checkpoint/steps_classifier/best"))))
    p.add_argument("--methods", default=",".join(METHODS))
    p.add_argument("--direct-layer", type=int, default=22)
    p.add_argument("--direct-heads", default="28,31")
    p.add_argument("--alpha", type=float, default=0.5, help="rollout retention constant")
    p.add_argument("--rollout-layer", type=int, default=-1, help="-1 = last layer")
    p.add_argument("--chunk", type=int, default=256, help="wnorm Gram chunk")
    p.add_argument("--grad-target", default="logprob", choices=["logprob", "logit"],
                   help="logprob is docs/saliency-maps.md map 5; logit is the raw score")
    p.add_argument("--grad-checkpointing", action="store_true")
    # glimpse (docs/saliency-maps.md map 6). The defaults are the paper's, including
    # --glimpse-depth-temp: see glimpse_layer_alphas for why 0.36 is the other candidate.
    p.add_argument("--glimpse-temp", type=float, default=0.5,
                   help="lambda, head-fusion temperature (eq 6)")
    p.add_argument("--glimpse-depth-temp", type=float, default=0.2,
                   help="lambda_d, depth prior temperature (eq 9)")
    p.add_argument("--glimpse-layer-frac", type=float, default=1.0,
                   help="propagate the last frac of the stack; the paper's ablation loses "
                        "nothing at 0.6. A method knob, not a memory knob -- peak memory "
                        "no longer scales with the number of propagated layers")
    p.add_argument("--glimpse-target", default="logit", choices=list(GM.GRAD_TARGETS),
                   help="z_t in eqs 5 and 16; the paper's is the raw logit")
    p.add_argument("--glimpse-token-weight", default="full",
                   choices=["full", "confidence", "prompt", "uniform"],
                   help="eq 18, and the paper's token-saliency ablation")
    p.add_argument("--tf32", action="store_true", default=True)
    p.add_argument("--no-tf32", dest="tf32", action="store_false")
    # render-only
    p.add_argument("--norm", default="percentile", choices=["percentile", "minmax", "rank"])
    p.add_argument("--norm-lo", type=float, default=1.0)
    p.add_argument("--norm-hi", type=float, default=99.0)
    p.add_argument("--cmap", default="jet")
    p.add_argument("--overlay-mode", default="blend", choices=["blend", "alpha"])
    p.add_argument("--overlay-alpha", type=float, default=0.5)
    p.add_argument("--upsample", default="bilinear", choices=["bilinear", "nearest"])
    p.add_argument("--no-per-step", action="store_true")
    p.add_argument("--no-contact-sheet", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if args.stage == "selftest":
        selftest(args)
        return
    if not args.out_dir:
        raise SystemExit("--out-dir is required for --stage scan|render")
    if args.stage == "render":
        render(args)
        return
    device = args.device if torch.cuda.is_available() else "cpu"
    scan(args, device)


if __name__ == "__main__":
    main()
