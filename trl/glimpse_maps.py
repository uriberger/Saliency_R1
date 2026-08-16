# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Per-observe-step GLIMPSE maps -- the map the GLIMPSE grounding reward scores.

GLIMPSE (arXiv 2506.18985v1, Shen 2025), docs/saliency-maps.md map 6. The only map here
that is neither pure attention nor pure gradient: the gradient decides which heads and
which layers to believe, and the attention still says where to look. Per target token `t`,
from one backward that yields `g^{l,h} = dz_t/dA^{l,h}` for every layer at once:

    G^{l,h} = ReLU( g^{l,h} * A^{l,h} )                                        (5)
    w^{l,h} = softmax_h( (1/lambda) sum_ij G^{l,h}_ij / sum_ij ReLU(g^{l,h}) ) (6)
    E^l     = sum_h w^{l,h} G^{l,h},  row-normalised                           (7)
    g^l     = || sum_h g^{l,h} ||_1,  alpha^l = g^l s^l / sum_k g^k s^k        (8-10)
    R       <- R + (I + alpha^l E^l) R                                         (11-13)
    beta_t  ~ p_t . mean_{i in P} R(t, i),  normalised over the step's tokens  (14-18)
    map     = sum_t beta_t . R(t, V)                                           (22)

THIS FILE IS THE CANONICAL HOME OF THAT ALGEBRA. `saliency_viz.py` delegates to it, the
way it already delegates `pixel_regroup` to `grad_maps.py`, so the picture the probe draws
and the map the reward scores cannot drift apart. It carries `build_forward` and
`causal_mask` with it on purpose: `trl/` is copied into `trl_repo/` and executes there,
where no probe script exists to import them from.

Equivalence against the pre-rework all-eager implementation was gated on GPU in fp32 at
corr 1.000000 / max deviation 1.3e-06 (`test_glimpse_gpu.py`); it cannot be gated in bf16,
where this map carries 0.063-0.089 of its own rounding noise. `test_glimpse_cpu.py` gates
the algebra, including the chain-rule identity `GlimpseGradCache` rests on.

WHAT IT COSTS, because it is the fact that governs every use of this module. One backward
plus one eager layer replay per PROPAGATED LAYER, per TARGET TOKEN -- against the gradient
map's one vmapped backward per chunk of whole STEPS. Measured on an H100-80GB against
`grad_maps.step_grad_maps` on identical cases (glimpse_cost_probe.py):

    grad                0.20-0.25 s/case    1.8-2.5 s per 8-case optimizer step
    glimpse frac 1.0    11.3-18.3 s/case    100-145 s                 55-59x grad
    glimpse frac 0.6     6.9-11.2 s/case      61-89 s                 34-36x grad

0.195 s per target token at N ~ 450, rising to 0.265 at N=1200 and 0.688 at N=2400.
Peak memory is NOT the constraint -- 19.7 GiB against the gradient map's 20.1 on the same
cases, both including 16.75 GiB of weights. Time is. `layer_frac` and `token_cap` are the
two dials; both are documented at `step_glimpse_maps`.

AND WHAT IT BUYS, which the caller should know before paying: on a 3,471-step screen
(`outputs/flow_corr/glimpse_screen/glimpse/report.txt`) GLIMPSE is the first map in this
repo with above-chance grounding -- auroc level 0.567, and 0.712 on unions <= 0.11 -- but
its correlation with the model being right is null to slightly negative at every level
(`mean_in_v2` r = -0.031 step / -0.020 completion; `auroc` r = -0.056 / -0.064), and
nothing clears Bonferroni (|r| >= 0.0735). The level decays hard with union area
(r(union) = -0.487), so `--max_union_area` is the knob that decides which regime a run is
in. See `rewards/glimpse_rewards.py` for what that means for a reward.
"""

from __future__ import annotations

import contextlib

import numpy as np
import torch

from .grad_maps import GRAD_TARGETS, _at_least_fp32, _row_scalars, frozen_params  # noqa: F401

# Qwen3-VL's image placeholder. The trainer passes its own `processing_class.image_token_id`
# when it has one; this is the fallback the rest of the repo hard-codes.
IMAGE_TOKEN_ID = 151655

GLIMPSE_TARGETS = GRAD_TARGETS
TOKEN_WEIGHTS = ("full", "confidence", "prompt", "uniform")


# ---------------------------------------------------------------------------
# the algebra (eqs 5-18). Pure functions, so test_glimpse_cpu.py can check them
# against a naive [N, N] reference with no model and no GPU.
# ---------------------------------------------------------------------------
def glimpse_edge_matrix(a, g, temp: float, eps: float = 1e-12):
    """[H, N, N] attention + its gradient -> (E, ||sum_h g^h||_1), eqs 5-8.

    Eq 6 divides by the head's total positive gradient mass, so `w` ranks heads by how
    much of the gradient they actually attend *along* rather than by gradient magnitude --
    a head whose positive gradient sits where it does not attend is demoted.

    The head loop keeps the peak at one [N, N] fp32 temporary. The vectorised form would
    need [H, N, N] in fp32 on top of the [H, N, N] the graph already holds, which at H=32
    and a 1700-token sequence is another 370 MB per layer, times 36 layers, per backward.
    """
    h, n, _ = a.shape
    # `_at_least_fp32` rather than `.float()`: bf16 attention has to be promoted, but a
    # float64 reference must not be silently demoted to meet the code halfway.
    up = _at_least_fp32
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
    """Per-layer propagation weights, eqs 9-10: gradient evidence x an exponential depth
    prior, `alpha_l = g_l s_l / sum_k g_k s_k` with `s_l = softmax(lambda_d (l+1))`.

    The paper's lambda_d = 0.2 was tuned on a 64-layer backbone, where it makes the prior
    fall by e every 5 layers -- 7.8% of the depth. On this 36-layer model the same number
    spans 14%, so `depth_temp=0.36` matches the paper's *shape* and 0.2 its *text*. The
    ablation calls this the single most important component (removing it takes their NSS
    from 1.014 to -0.210), which is exactly why the mismatch is worth naming.
    """
    g = _at_least_fp32(torch.stack([torch.as_tensor(v) for v in g_l1]))
    ell = torch.as_tensor(list(layer_ids), dtype=g.dtype, device=g.device) + 1.0
    s = torch.softmax(depth_temp * ell, dim=0)
    tot = g.sum()
    # A backward that produced no positive gradient anywhere would otherwise make every
    # alpha NaN and silently blank the map; fall back to the depth prior alone.
    g = torch.where(tot > 0, g / tot.clamp_min(eps), torch.full_like(g, 1.0 / g.numel()))
    a = g * s
    return a / a.sum().clamp_min(eps)


def glimpse_propagate(row: int, mats, alphas):
    """Row `row` of `R = prod_l (2I + alpha_l E_l)`, eqs 11-13, scaled by 2^-L.

    Eq 13 with eq 12 is `R <- (2I + alpha_l E_l) R`: the identity path DOUBLES at every
    layer, so over 36 layers the row would grow by 2^36 before anything is read off it.
    Every quantity the method takes from R is a ratio and the factor is identical for
    every token, so it is folded in per layer as `v + (alpha/2) v E`.

    R is only ever read one row at a time and `v (2I + alpha E)` is linear in `v`, so the
    row is carried through the product as a vector: O(L*N^2), not the matrix form's
    O(L*N^3). The product applies the LAST layer's factor first, hence the reversed loop.
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
    raise ValueError(f"token weight {mode!r} not in {TOKEN_WEIGHTS}")


def _find_subseq(hay: list[int], needle: list[int]) -> int:
    """Last start index of `needle` in `hay`, or -1."""
    if not needle or len(needle) > len(hay):
        return -1
    for s in range(len(hay) - len(needle), -1, -1):
        if hay[s:s + len(needle)] == needle:
            return s
    return -1


def prompt_positions(tok, question: str, prompt_ids: list[int], img_positions,
                     valid_positions=None):
    """-> (positions, how). The prompt columns `P` of eq 14: the question's own tokens.

    The paper's `P` is the user prompt. Ours is a chat template wrapped around it -- a
    system prompt about the reasoning format, then the image, then the question -- and
    `a_t` is a MEAN over `P`, so folding in the boilerplate dilutes precisely the signal
    the weight exists to carry. The question is located by matching its own tokenisation
    inside the prompt, from the right (the template repeats nothing else there). A leading
    space can merge differently at a template boundary, so the match is retried without
    the first token; if both fail, every non-image prompt token is used.

    `valid_positions` is the attention mask over the prompt, when there is one. The
    trainer left-pads its batch, and a fallback that swept the pad tokens into `P` would
    average `R(t, .)` over columns the forward never attended to.
    """
    q = list(tok(question, add_special_tokens=False)["input_ids"])
    for cand, how in ((q, "question"), (q[1:], "question_less_first")):
        s = _find_subseq(prompt_ids, cand)
        if s >= 0:
            return list(range(s, s + len(cand))), how
    img = set(int(i) for i in img_positions)
    ok = None if valid_positions is None else set(int(i) for i in valid_positions)
    return ([i for i in range(len(prompt_ids))
             if i not in img and (ok is None or i in ok)], "prompt_minus_image")


# ---------------------------------------------------------------------------
# the forward, and the two masks it needs
# ---------------------------------------------------------------------------
def build_forward(inputs, ids, prompt_len):
    """prompt processor output + prompt++completion ids -> one teacher-forced forward."""
    fwd = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    if "pixel_values" in inputs:
        fwd["pixel_values"] = inputs["pixel_values"]
        fwd["image_grid_thw"] = inputs["image_grid_thw"]
    if inputs.get("mm_token_type_ids") is not None:
        pad = torch.zeros(1, ids.shape[1] - prompt_len, dtype=torch.long, device=ids.device)
        fwd["mm_token_type_ids"] = torch.cat([inputs["mm_token_type_ids"], pad], dim=1)
    return fwd


def causal_mask(seq: int, dtype, device, attention_mask=None):
    """Additive [1, 1, N, N] mask for the eager replay: causal, and padding-aware.

    The forward runs under sdpa, which is entitled to build no explicit mask at all, so
    the replayed layer must be handed one or it would attend bidirectionally -- and, in a
    left-padded batch, across the pad. The trainer's cases ARE left-padded (a batch of one
    sliced out of a padded batch keeps its own padding), so a purely causal mask would
    make the replay disagree with the forward it is supposed to reproduce; `_check_replay`
    would then fire, which is the good outcome, but the mask is simply wrong.

    The diagonal is restored after masking the pad columns, because a LEFT-PAD query row
    is otherwise masked everywhere: causality leaves it only pad columns, and the padding
    then takes those away. An all-min row softmaxes to a UNIFORM row -- the pad query
    would read every real token in the sequence -- and one arithmetic step from -inf it is
    NaN, which would spread through `E` into every map. Attending to itself is the cheap
    well-defined answer.

    It is not sdpa's answer, and cannot be: for a fully-masked row torch >= 2.5 returns
    exactly ZERO (pytorch#110213), which no softmax can produce. Those rows therefore
    disagree with the forward BY CONSTRUCTION -- see `valid_row_mask`, which is how the
    replay check is kept off them.
    """
    m = torch.triu(torch.ones(seq, seq, dtype=torch.bool, device=device), diagonal=1)
    add = torch.zeros(seq, seq, dtype=dtype, device=device)
    add.masked_fill_(m, torch.finfo(dtype).min)
    if attention_mask is not None:
        pad = attention_mask.reshape(-1)[:seq].to(torch.bool).logical_not()
        add.masked_fill_(pad[None, :], torch.finfo(dtype).min)
        add.fill_diagonal_(0.0)
    return add[None, None]


def valid_row_mask(seq: int, attention_mask, device):
    """-> bool [N] of the query rows the forward actually computed, or None if unpadded.

    The rows a padded forward does NOT compute are not merely imprecise, they are a
    different quantity: sdpa returns exact zero for a fully-masked row (see `causal_mask`),
    the eager replay returns the row's own value vector, and neither is "the layer's
    output" in any sense. Nothing reads them -- a pad row's `dz/dh` is exactly zero, so it
    contributes nothing to `E`, to the head weights or to the layer alphas, and no
    propagated row ever picks up mass on a pad column -- but `_check_replay` compared them
    and could not tell that garbage from a dropped kwarg. Measured by
    `diag_glimpse_pad_rows.py` on layer 0 of the real checkpoint in fp32, 120 pad rows in
    1700: 4e-7 relative over the real rows, 0.38 over the pad rows, 0.17 over both,
    against a tolerance of 0.05.

    Layer 0 is where this bites, and it is the layer `layer_frac=1.0` -- the trainer's
    default -- replays first: its residual stream peaks in the single digits, while deeper
    layers carry outliers orders of magnitude larger that swamp the same absolute junk.
    """
    if attention_mask is None:
        return None
    return attention_mask.reshape(-1)[:seq].to(torch.bool).to(device)


@contextlib.contextmanager
def eager_one_attention(attn_mod):
    """Run ONE attention module in eager, so its softmax weights exist and are in the graph.

    The flag lives on the shared text config, so flipping it is global -- but only the
    replayed layer runs inside the window, and it is handed an explicit causal mask, so
    nothing else can observe either the flag or the mask.
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
    forward does -- costs ~9 KB of GPU memory per (query, key) pair across the 36 layers:
    42.3 GiB at N=1600, OOM by 3600. So map 6 is built in two stages per target token:

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
    propagated: 36 x 4 bytes per pair against the graph's ~9000.

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
        self.valid_rows = None                # set beside `mask`; see `valid_row_mask`
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
        self.leaf = self.mask = self.valid_rows = None

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

    def _first_row(self) -> int:
        """The first query row the forward actually computed. In a left-padded case row 0
        is a pad row, whose attention this class DEFINES (the restored diagonal) rather
        than reproduces, so it can say nothing about the mask reaching eager."""
        if self.valid_rows is None:
            return 0
        idx = self.valid_rows.nonzero()
        return int(idx[0]) if idx.numel() else 0

    def _check_causal(self, aw):
        """The first real query row may only see itself and what precedes it. Eager
        attention with no mask is silently bidirectional -- every map would be wrong and
        nothing would raise."""
        if self._checked:
            return
        self._checked = True
        r = self._first_row()
        leak = float(aw[0, :, r, r + 1:].detach().abs().sum())
        if leak > 1e-3:
            raise RuntimeError(f"glimpse: attention is not causal (row {r} puts {leak:.3g} "
                               f"after column {r}) -- the causal mask did not reach eager")

    def _check_replay(self, li: int, out):
        """The replay must reproduce the layer's own output ON THE ROWS THE FORWARD
        COMPUTED. A dropped kwarg or a wrong mask would otherwise yield a plausible map
        built on the wrong tensor. The tolerance is loose on purpose: eager and sdpa differ
        in the last bf16 bits, a replay mistake differs by order 1.

        The pad rows are excluded because they are not a reproduction at all: sdpa returns
        exact zero there and the replay returns the row's own value vector. Comparing them
        made this check fire on the first colocated training run -- at 0.084-0.090 relative
        against a tolerance of 0.05, on maps that were correct -- while `valid_rows`
        reports 4e-7 on the rows that carry the map. `valid_row_mask` has the measurement.
        """
        if self._replay_checked:
            return
        self._replay_checked = True
        ref, got = self.out[li].detach().float(), out.detach().float()
        if self.valid_rows is not None:
            ref, got = ref[:, self.valid_rows], got[:, self.valid_rows]
        rel = float((got - ref).abs().max()) / max(float(ref.abs().max()), 1e-6)
        if rel > 0.05:
            raise RuntimeError(f"glimpse: the eager replay of layer {li} differs from the "
                               f"forward by {rel:.3g} relative on the {ref.shape[1]} rows "
                               "the forward computed -- the recorded kwargs or the causal "
                               "mask are wrong")


# ---------------------------------------------------------------------------
# the core, shared by the probe entry and the trainer entry
# ---------------------------------------------------------------------------
def _select_tokens(spans, token_cap: int, rng):
    """-> per-step offsets into the step, honouring `token_cap`.

    The draw is uniform-random WITHOUT REPLACEMENT, not the first `k`. Eq 18 renormalises
    beta inside the step, so a random subset estimates the same weighted mean; the first
    `k` would instead read every step's OPENING tokens, and these maps carry a documented
    reading-order prior (rollout puts 5.8-8.8x of its mass on the top row, monotone in
    sequence order), so a first-k cap would write that prior into the reward.

    Sorted, so the rows handed to `logits_to_keep` stay ascending and the map is
    bit-identical for a given draw whatever order the indices came out in.
    """
    out = []
    for a, b in spans:
        n = b - a
        if token_cap and 0 < token_cap < n:
            out.append(np.sort(rng.choice(n, size=int(token_cap), replace=False)))
        else:
            out.append(np.arange(n))
    return out


def _glimpse_core(model, fwd, spans, *, img_cols, prompt_idx, attention_mask, gh, gw,
                  target, layer_frac, temp, depth_temp, token_weight, token_cap, rng,
                  collect=None):
    """-> ([n_steps, gh, gw] float32, info). `spans` are ABSOLUTE [a, b) into `fwd`."""
    device = fwd["input_ids"].device
    ids = fwd["input_ids"]
    n_layers = sum(1 for m in model.modules()
                   if type(m).__name__ == "Qwen3VLTextAttention")
    keep = min(n_layers, max(1, int(round(float(layer_frac) * n_layers))))
    first = n_layers - keep

    sel = _select_tokens(spans, token_cap, rng)
    # The row that produced the token at absolute position p is p-1 -- both the logit row
    # and, for the same reason, the relevance row. See the module docstring.
    rows = [a + int(o) - 1 for (a, _b), off in zip(spans, sel) for o in off]
    if not rows:
        return np.zeros((len(spans), gh, gw), dtype=np.float32), {"n_layers": n_layers,
                                                                 "first_layer": first,
                                                                 "unweighted_steps": []}
    rows_t = torch.tensor(rows, device=device)
    targets = torch.cat([ids[0, a + int(o)][None] for (a, _b), off in zip(spans, sel)
                         for o in off])

    out = np.zeros((len(spans), gh, gw), dtype=np.float32)
    fell_back = []
    cap = GlimpseGradCache(model, first, temp)
    try:
        with checkpointing_off(model), torch.enable_grad():
            res = model(**fwd, use_cache=False, logits_to_keep=rows_t)
            z = res.logits[0].float()
            del res
            cap.check()
            cap.mask = causal_mask(ids.shape[1], next(model.parameters()).dtype, device,
                                   attention_mask=attention_mask)
            # Set from the SAME attention mask as `cap.mask`: the rows the replay defines
            # rather than reproduces are exactly the rows that mask leaves empty.
            cap.valid_rows = valid_row_mask(ids.shape[1], attention_mask, device)
            # `_row_scalars` is the gradient reward's own definition of the per-token
            # scalar, so `target` means here exactly what `--grad_target` means there.
            f = _row_scalars(z, targets, target)
            conf = _row_scalars(z, targets, "logprob").detach().exp()      # eq 16

            k = 0
            for si, off in enumerate(sel):
                acc = torch.zeros(img_cols.numel(), dtype=torch.float32, device=device)
                plain = torch.zeros_like(acc)
                wsum = torch.zeros((), dtype=torch.float32, device=device)
                for _ in range(len(off)):
                    g_out = cap.layer_grads(f[k])
                    mats, g_l1 = [], []
                    for li in cap.layers:
                        # pop, so each [N, d] gradient dies as its layer is consumed
                        e, g1 = cap.edge(li, g_out.pop(li))
                        mats.append(e)
                        g_l1.append(g1)
                    del g_out
                    alphas = glimpse_layer_alphas(g_l1, cap.layers, depth_temp)
                    v = glimpse_propagate(rows[k], mats, alphas)
                    w = glimpse_token_weight(conf[k], v[prompt_idx].mean(), token_weight)
                    if collect is not None:
                        collect.append((si, float(w),
                                        v[img_cols].detach().float().cpu().numpy()
                                        .reshape(gh, gw)))
                    acc += w * v[img_cols]
                    plain += v[img_cols]
                    wsum += w
                    del mats, alphas, v
                    k += 1
                # eq 22, with `Y` restricted to this step so the map stays comparable to
                # the other maps. A step whose tokens all weigh zero -- no positive
                # gradient anywhere on the prompt -- would otherwise render as a blank.
                if float(wsum) > 0:
                    m = acc / wsum
                else:
                    fell_back.append(si)
                    m = plain / max(len(off), 1)
                out[si] = m.reshape(gh, gw).cpu().numpy()
    finally:
        cap.close()
    info = {"first_layer": first, "n_layers": n_layers, "temp": temp,
            "depth_temp": depth_temp, "target": target, "token_weight": token_weight,
            "token_cap": int(token_cap or 0), "unweighted_steps": fell_back,
            "n_target_tokens": len(rows)}
    return out, info


def glimpse_map(model, processor, inputs, ids, prompt_len, steps, gh, gw, question,
                args, device, *, collect=None):
    """The PROBE entry: one sample, `steps` relative to the completion. -> (maps, info).

    Kept signature-compatible with what `saliency_viz.py --stage scan` and
    `test_glimpse_gpu.py` call, so both gates keep testing the code the reward runs.
    `collect`, if given, receives one `(step_index, beta, [gh, gw] map)` per target token
    -- eq 22's terms before they are summed.
    """
    img_cols = (inputs["input_ids"][0] == IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0]
    if img_cols.numel() != gh * gw:
        raise RuntimeError(f"{img_cols.numel()} image tokens, grid is {gh}x{gw}")
    pos, how = prompt_positions(processor.tokenizer, question,
                                inputs["input_ids"][0].tolist(), img_cols.tolist())
    fwd = build_forward(inputs, ids, prompt_len)
    maps, info = _glimpse_core(
        model, fwd, [(prompt_len + a, prompt_len + b) for _t, a, b in steps],
        img_cols=img_cols, prompt_idx=torch.tensor(pos, device=device),
        attention_mask=None, gh=gh, gw=gw, target=args.glimpse_target,
        layer_frac=args.glimpse_layer_frac, temp=args.glimpse_temp,
        depth_temp=args.glimpse_depth_temp, token_weight=args.glimpse_token_weight,
        token_cap=getattr(args, "glimpse_token_cap", 0),
        rng=np.random.default_rng(getattr(args, "glimpse_seed", 0)), collect=collect)
    info.update(prompt_tokens=how, n_prompt_tokens=len(pos))
    return maps, info


def step_glimpse_maps(
    model,
    forward_inputs: dict,
    spans: list[tuple[int, int]],
    grid_thw,
    *,
    question: str,
    tokenizer,
    prompt_len: int,
    target: str = "clogit",
    layer_frac: float = 1.0,
    temp: float = 0.5,
    depth_temp: float = 0.2,
    token_weight: str = "full",
    token_cap: int = 0,
    seed: int = 0,
    image_token_id: int = IMAGE_TOKEN_ID,
):
    """The TRAINER entry. -> ([n_steps, gh, gw] float32, info).

    Deliberately the same shape as `grad_maps.step_grad_maps`: `forward_inputs` is one
    case (batch of 1) and `spans` are absolute `[a, b)` positions in it, so the trainer's
    per-case loop, its ZeRO-3 safeguards and every probe that reads these maps are
    unchanged -- only the map differs.

    The caller is responsible for `frozen_params(model)`; it wraps the whole per-case loop
    in the trainer, not one call.

    THE TWO COST DIALS, both measured (see the module docstring):

      layer_frac  fraction of the stack propagated, off the TOP. 0.6 costs 1.64x less and
                  the paper's own ablation loses nothing there, but it is a METHOD change,
                  not a memory one -- the screened map is `layer_frac=1.0`.
      token_cap   target tokens scored per step, drawn at random (never the first k, see
                  `_select_tokens`). Cost is exactly linear in this, so a cap of 6 against
                  a median ~20 is ~3.5x. What it does to the SCORE is not measured at
                  scale; 0 (every token) is the default for that reason.
    """
    if not spans:
        return (np.zeros((0, int(grid_thw[1]) // 2, int(grid_thw[2]) // 2),
                         dtype=np.float32), {"n_target_tokens": 0, "unweighted_steps": []})
    if target not in GLIMPSE_TARGETS:
        raise ValueError(f"glimpse target {target!r} not in {GLIMPSE_TARGETS}")
    if token_weight not in TOKEN_WEIGHTS:
        raise ValueError(f"glimpse token weight {token_weight!r} not in {TOKEN_WEIGHTS}")

    ids = forward_inputs["input_ids"]
    device = ids.device
    if any(a <= 0 or b <= a or b > ids.shape[1] for a, b in spans):
        raise ValueError(f"spans {spans} outside 1..{ids.shape[1]} or empty")

    gh, gw = int(grid_thw[1]) // 2, int(grid_thw[2]) // 2
    img_cols = (ids[0] == image_token_id).nonzero(as_tuple=True)[0]
    if img_cols.numel() != gh * gw:
        raise RuntimeError(f"{img_cols.numel()} image tokens but the grid is {gh}x{gw} -- "
                           "the map would be reshaped from the wrong columns")

    attention_mask = forward_inputs.get("attention_mask")
    prompt_ids = ids[0, :prompt_len].tolist()
    valid = None
    if attention_mask is not None:
        valid = attention_mask.reshape(-1)[:prompt_len].nonzero(as_tuple=True)[0].tolist()
    pos, how = prompt_positions(tokenizer, question, prompt_ids,
                                [int(i) for i in img_cols.tolist() if i < prompt_len],
                                valid_positions=valid)
    if not pos:
        # eq 14 would be a mean over nothing; fall back to a flat token weight rather than
        # emit NaN maps that the metric would silently turn into skipped steps.
        token_weight, how = "confidence", "empty_prompt_fallback"
        pos = [0]

    maps, info = _glimpse_core(
        model, forward_inputs, spans, img_cols=img_cols,
        prompt_idx=torch.tensor(pos, device=device), attention_mask=attention_mask,
        gh=gh, gw=gw, target=target, layer_frac=layer_frac, temp=temp,
        depth_temp=depth_temp, token_weight=token_weight, token_cap=token_cap,
        rng=np.random.default_rng(seed))
    info.update(prompt_tokens=how, n_prompt_tokens=len(pos))
    return maps, info
