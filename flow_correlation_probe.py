#!/usr/bin/env python
"""Indirect-flow saliency: does attention that reaches the image THROUGH earlier text
predict correctness better than the direct map?

`head_correlation_probe.py` scores the DIRECT map -- head h of layer L's attention
from a step's own tokens to the image patches. That map assumes the model reads the
patches while it writes the step. The information-flow literature on VLMs says
otherwise: the image is absorbed into text positions in early layers, and later text
reads *those positions* rather than the patches. A direct map at layer 22 cannot see
that path at all, which is a candidate explanation for the layer-level intervention
null, for ID accuracy sitting at 0.534, and for layers 0/1 topping the direct scan.

Four replacement maps, scored by the same two metrics against the same per-step DINO
unions as the direct scan, on the same prepared cases, so the numbers are directly
comparable:

  rollout_mean   layer-wise attention rollout, heads merged by the mean
  rollout_wnorm  the same, heads merged by || sum_h A^h_{n,k} W_O^h v^h_k ||
  grad           || d log P(step's own tokens) / d e_j ||, e_j = image embedding j
  glimpse        GLIMPSE, gradient-weighted attention adaptively propagated (map 6)

---------------------------------------------------------------------------
GLIMPSE, and why it is scored here before it is trained on
---------------------------------------------------------------------------
`saliency_viz.py` owns map 6 and this file borrows it -- see `_glimpse_module`. The
point of running it through this probe is that a GLIMPSE-based grounding reward has
two candidate metrics, `mean_in_v2` and `auroc`, and this scan already computes BOTH
for every map at step and completion level. So one glimpse scan ranks the two reward
variants against each other, and against chance, before either is trained.

Read the LEVEL before the correlation, as with every other map here. Three independent
map families have now come out anti-grounded on this corpus (the rollouts at every
layer, the two rewarded direct heads at 0.410/0.392), and the one map that cleared
chance -- `grad` -- correlates NEGATIVELY with correctness at -0.098. A glimpse column
that lands at 0.5 is not "no result": it says the reward would be rewarding noise.

Cost. GLIMPSE is one backward plus a per-layer eager replay per TARGET TOKEN, where
`grad` amortises one backward over a whole step, so this variant is far and away the
most expensive of the four -- budget hours on 8 GPUs where the other three take
minutes. Run it with --max-cases first.

Precision. The map carries 0.063-0.089 of its own bf16 rounding noise (measured in
docs/glimpse-handoff.md, which is why its equivalence gate had to move to fp32). auroc
is a rank statistic and mean_in_v2 a ratio of means, so both are far more tolerant of
that than a max-deviation check -- but it is noise in the direction of attenuating any
correlation, so read a null column as an upper bound on the effect, not a measurement.

---------------------------------------------------------------------------
The rollout
---------------------------------------------------------------------------
Positions p = 1..P cover the whole sequence: pre-image text, the M image tokens, then
the chain. `sal_p` is a length-M vector, "how much of position p's content is
traceable to each patch".

    sal_p^(0) = one-hot at j   if p is image token j;  0 otherwise
    sal_p^(l) = a * sum_{q<=p} w^(l)_{p,q} sal_q^(l-1)  +  (1-a) * sal_p^(l-1)

The first term is what layer l's attention pulls in -- straight off the patches, or
out of a text position that already carries image content. The second is what p keeps,
because a block ADDS its attention output to the residual stream. a=0.5 is the
convention (Abnar & Zuidema).

Three things this gets right that the single-layer form does not:

  * The inherited map is indexed at l-1, not l. Position q's value vector at layer l is
    built from q's residual after layer l-1, so l-1 is the map that is actually
    readable there. A single-layer recursion instead sums paths of arbitrary hop count
    through one layer -- paths the architecture cannot execute, since h hops need h
    distinct layers.
  * The recursion starts at the IMAGE tokens, not at the first text token, so
    image-to-image mixing is carried. By layer 22 "image token j" is not purely patch j
    and pretending otherwise is an error the direct map also makes.
  * mass(sal_p^(l)) <= 1 at every layer by induction, since attention rows sum to 1 and
    the initial masses are 0 or 1. So the total is interpretable as "fraction of this
    position traceable to the image", and it is the by-layer image-mass curve for free.

Readout is the LAST layer: it already contains every layer below it, so no layer has to
be selected -- unlike the direct map, where layer 22 tells you nothing about layer 5.
Per-layer readouts are recorded anyway as a secondary, because rollout is known to go
diffuse with depth; `report` ranks them on odd row_index and re-scores on even, since
36 nested readouts is still a selection.

Head merge. Heads are SUMMED, not averaged: each writes into the residual through its
own column block of o_proj, and the block output is sum_h W_O^h (sum_k A^h_{n,k} v^h_k).
So the honest edge weight from n to k is the magnitude of that source's total
contribution,

    c_{n,k} = || sum_h A^h_{n,k} W_O^h v^h_k ||_2        (--weighting wnorm)

row-normalised to sum to 1. This is also the value-norm correction: raw attention
overweights sinks, which take a large share of every row while carrying near-zero-norm
values, and it is the only form that can express cancellation between heads. With no
value information the max-entropy default is the plain mean over heads
(--weighting mean); that is a convention, not a fact about the architecture, which is
why both are run.

Increment. sal is cumulative -- step k's map contains steps 1..k-1's objects, plus
whatever the question tokens pulled in -- so a better `r` can come from the map
absorbing a completion-level signal rather than from per-step grounding. The `incL`
columns subtract `sal` at the token immediately BEFORE the step from the step's own
mean map, at the same layer L, isolating what the step's own span newly pulled in from
the image. One `incL` per layer, so the increment has its own layer curve rather than
being read at the last layer alone: the 2026-08-06 run found `inc` at the last layer
to be the only column above chance, which makes the shape of that curve the thing to
know. Read AUROC for them; mean_in_v2 divides by the map's mean, which an increment
can drive to zero, so it is NaN for a non-random subset of steps and the report drops
it.

Controls. Every correlation here is also reported as a PARTIAL correlation holding
three covariates fixed: the step's DINO union area, the completion's step count, and
the column's own image mass. Union area moves `mean_in_v2` mechanically (its ceiling
is n_patches/n_in) and tracks clutter, hence difficulty; image mass is the strongest
correlate of correctness anyone has measured here (+0.22-0.29), and a map's *shape*
predicting correctness is a different claim from its *magnitude* doing so. A column
whose raw r survives but whose partial r does not is measuring difficulty, not
grounding.

Sharpness. The scan also writes the box-free concentration columns of
`saliency_sharpness.py` for every map above -- how peaked it is, never where the peak
is -- plus the covariates they have to be controlled for. Scored by
`sharpness_report.py`, which races them against the DINO columns computed here on the
same steps; `--stage report` below is unchanged and ignores them.

DEEPSTACK CAVEAT. Qwen3-VL re-injects visual features at the image positions at several
decoder layers (`_deepstack_process`). The recursion does not model that addition; it
only pushes those positions further toward their own patch, so `sal` at image positions
stays a valid attribution, but the mixing ratio is not exact.

---------------------------------------------------------------------------
The gradient map
---------------------------------------------------------------------------
No flow model at all -- differentiate the thing directly. For a step spanning tokens
t_a..t_b, teacher-forced on the model's own chain,

    F = sum_{n=a..b-1} log P(t_n | t_<n, image, prompt)
    g_j = || dF / de_j ||_2

where e_j is image embedding j entering the language model. This counts every path
through every layer and head, with no alpha, no head-merge convention and no rollout
approximation, so it is the control on whether the rollout's approximations cost
anything. It cannot be decomposed per head, but it can be a Stage-4 loss unchanged.

Both the merged image embeds AND every deepstack tensor are captured as leaves, so
`gnorm_ds` / `gxi_ds` include the deepstack injection paths and `gnorm` / `gxi` do not.
`gxi` is gradient-times-input, |<e_j, dF/de_j>|, usually the less noisy of the two.

---------------------------------------------------------------------------
The union size
---------------------------------------------------------------------------
The DINO union is the reference region every one of these maps is scored against, and
it is UNCAPPED -- `prepare` applies the per-BOX cap (0.5) only, and N boxes each under
it can cover the image between them. The median step's union covers 54% of the patch
grid, the top decile 89%. Every map here reads lower the larger it gets (r(union,
auroc) = -0.50 for gxi_ds, -0.28 for rollout_wnorm at L22, but only -0.04 for the
increment), and the rollout's below-chance level is carried by the large-union half:
at union < 0.19 rollout_wnorm sits at 0.537 rather than 0.434. Chance is exactly 0.5
for a mask of any size at a random location, so that curve is real map/mask structure
rather than an artefact of the statistic -- but above ~0.5 coverage the union has
stopped localising the thing the step names, so the two ends of the curve answer
different questions. `report` prints the level by union decile before anything else,
and `--max-union` restricts every number after it. Fix that threshold before looking
at a confirmation set.

---------------------------------------------------------------------------
All six maps used in this project, side by side and with the notation stated once:
docs/saliency-maps.md.

    bash launch_flow_correlation.sh --gpus 8 --out-dir DIR --cases-dir <probe out-dir> \
         --maps rollout_mean,rollout_wnorm,grad
    bash launch_flow_correlation.sh --gpus 8 --out-dir DIR --cases-dir <probe out-dir> \
         --maps glimpse --max-cases 8          # hours, not minutes: size it first
    python flow_correlation_probe.py --stage report --out-dir DIR/rollout_mean
    python flow_correlation_probe.py --stage report --out-dir DIR/rollout_mean \
         --all-columns --max-union 0.5
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent


def repo_path(rel: str) -> Path:
    p = REPO / rel
    if p.exists():
        return p
    if REPO.parent.name == ".worktrees":
        alt = REPO.parent.parent / rel
        if alt.exists():
            return alt
    return p


def _load_module(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


PROBE = _load_module("_fc_overlap_probe", "overlap_probe.py")
IV = _load_module("_fc_intervene", "intervene_probe.py")
HC = _load_module("_fc_head_corr", "head_correlation_probe.py")
SHARP = _load_module("_fc_sharpness", "saliency_sharpness.py")
IMAGE_TOKEN_ID = PROBE.IMAGE_TOKEN_ID

MAPS = ("rollout_mean", "rollout_wnorm", "grad", "glimpse")


def _glimpse_module():
    """saliency_viz.py, loaded lazily: it owns map 6 and this file only borrows it.

    The dependency runs this way round because saliency_viz already loads THIS module
    (as `_sv_flow`, for `build_forward`) -- so we register ourselves under that name
    before loading it, and its `_load_module` finds us instead of executing this file a
    second time. Lazy, so the other three maps never pay for the import.

    This is deliberately not the final home. The reward needs the same map and cannot
    import a probe script, so the code belongs in `trl/glimpse_maps.py` beside
    `trl/grad_maps.py`. That move carries a constraint this screen does not: `trl/` is
    copied into `trl_repo/` by patch_trl_qwen3.sh and executes there, where no probe
    script exists, so `build_forward` and `causal_mask` have to travel with it. Doing
    it here would produce a module that works in the probe and breaks in the trainer.
    """
    if "_fc_saliency_viz" not in sys.modules:
        sys.modules.setdefault("_sv_flow", sys.modules[__name__])
        _load_module("_fc_saliency_viz", "saliency_viz.py")
    return sys.modules["_fc_saliency_viz"]


class _NoEngine:
    """Stand-in for a map that installs and removes its own hooks per case.

    `scan` holds one engine for the whole shard and closes it in a finally; glimpse
    builds a GlimpseGradCache inside every call and closes it in its own finally, so
    there is nothing shard-lived to hold.
    """

    def close(self):
        pass


# ---------------------------------------------------------------------------
# the rollout
# ---------------------------------------------------------------------------
class RolloutFlow:
    """Runs the layer-wise rollout inside the forward pass, one layer at a time.

    Each hook re-runs its own attention module in eager to recover the softmax weights
    that sdpa discards -- the trainer's single-layer trick, installed on all 36 -- then
    immediately folds them into `sal` and drops them. Peak memory is one layer's
    [H, P, P], the same as the direct scan, rather than 36 of them.
    """

    def __init__(self, model, weighting: str, alpha: float, chunk: int = 256):
        if weighting not in ("mean", "wnorm"):
            raise ValueError(f"weighting must be mean|wnorm, got {weighting}")
        self.weighting, self.alpha, self.chunk = weighting, float(alpha), int(chunk)
        self.mask = None
        self.sal = None            # [P, M] float32 on device, or None to disarm
        self.rows = None           # positions to snapshot after every layer
        self.snaps = []            # list of [n_rows, M] float32 CPU tensors
        self._reentry = False
        self.handles, self.layers = [], []
        for m in model.modules():
            if type(m).__name__ == "Qwen3VLTextAttention" and hasattr(m, "layer_idx"):
                self.layers.append(int(m.layer_idx))
                self.handles.append(
                    m.register_forward_hook(self._make(int(m.layer_idx)), with_kwargs=True))
        self.layers.sort()
        if not self.layers:
            raise RuntimeError("no Qwen3VLTextAttention modules found")

    def close(self):
        for h in self.handles:
            h.remove()

    def arm(self, n_pos: int, img_cols: torch.Tensor, rows: torch.Tensor,
            mask: torch.Tensor):
        self.sal, self.rows, self.mask, self.snaps = init_sal(n_pos, img_cols), rows, mask, []

    def disarm(self):
        self.sal = self.rows = self.mask = None

    def _make(self, layer_idx):
        def hook(module, args, kwargs, output):
            if self._reentry or self.sal is None:
                return None
            self._reentry = True
            kw = dict(kwargs)
            kw["attention_mask"] = self.mask
            kw["past_key_values"] = None          # never double-update the KV cache
            kw["use_cache"] = False
            prev = module.config._attn_implementation
            module.config._attn_implementation = "eager"
            try:
                _o, attn = module(*args, **kw)
            finally:
                module.config._attn_implementation = prev
                self._reentry = False
            a = attn[0]                                        # [H, P, P]
            if self.weighting == "mean":
                w = a.mean(0, dtype=torch.float32)
            else:
                hs = args[0] if args else kwargs["hidden_states"]
                w = edge_weights_wnorm(module, hs, a, self.chunk)
            w = w / w.sum(-1, keepdim=True).clamp_min(1e-12)
            self.sal = rollout_update(self.sal, w, self.alpha)
            # Stays on the device: .cpu() here would force 36 syncs per case for a
            # couple of MB. One transfer after the forward instead.
            self.snaps.append(self.sal[self.rows].clone())
            del a, w, attn, _o
            return None
        return hook


def init_sal(n_pos: int, img_cols: torch.Tensor):
    """Initial condition: image token j carries exactly patch j, nothing else."""
    m = img_cols.numel()
    sal = torch.zeros(n_pos, m, dtype=torch.float32, device=img_cols.device)
    sal[img_cols, torch.arange(m, device=img_cols.device)] = 1.0
    return sal


def rollout_update(sal, w, alpha):
    """One layer: attention pulls content in, the residual stream keeps what was there."""
    return alpha * (w @ sal) + (1.0 - alpha) * sal


def edge_weights_wnorm(module, hidden_states, a, chunk: int = 256):
    """c[n,k] = || sum_h a[h,n,k] * W_O^h v^h_k ||_2, as a [P, P] float32 tensor.

    Materialising `u[h,k,:] = W_O^h v^h_k` for every (n,k) pair would be P*P*d floats,
    which does not fit. Expand the square instead: with the per-source Gram matrix
    G[k,h,g] = <u[h,k], u[g,k]> (only P*H*H entries),

        c[n,k]^2 = sum_{h,g} a[h,n,k] a[g,n,k] G[k,h,g]

    so the norm is a quadratic form in the length-H vector of per-head attentions, and
    both stages chunk cleanly.
    """
    hh, p, _ = a.shape
    dh = module.head_dim
    dev = a.device
    wo = module.o_proj.weight                                  # [d_model, H*dh]
    wo_b = wo.view(wo.shape[0], hh, dh).permute(1, 0, 2).float()   # [H, d_model, dh]

    v = module.v_proj(hidden_states).view(1, p, -1, dh).transpose(1, 2)
    v = IV.repeat_v(v, module.num_key_value_groups)[0]         # [H, P, dh]

    g = torch.empty(p, hh, hh, dtype=torch.float32, device=dev)
    for k0 in range(0, p, chunk):
        k1 = min(k0 + chunk, p)
        u = torch.einsum("hde,hbe->hbd", wo_b, v[:, k0:k1].float())   # [H, B, d_model]
        g[k0:k1] = torch.einsum("hbd,gbd->bhg", u, u)
        del u
    del wo_b, v

    c = torch.empty(p, p, dtype=torch.float32, device=dev)
    for n0 in range(0, p, chunk):
        n1 = min(n0 + chunk, p)
        ac = a[:, n0:n1].permute(1, 2, 0).float()              # [B, P, H]
        t = torch.einsum("bph,phg->bpg", ac, g)
        c[n0:n1] = (t * ac).sum(-1).clamp_min(0).sqrt()
        del ac, t
    return c


# ---------------------------------------------------------------------------
# the gradient map
# ---------------------------------------------------------------------------
class ImageEmbedLeaves:
    """Turns the vision tower's outputs into autograd leaves.

    Detaching and re-flagging makes them leaves rather than intermediates, which does
    two things at once: `torch.autograd.grad` can be taken with respect to them, and
    the vision tower drops out of the graph entirely (its gradients are never wanted
    and its activations are never kept).
    """

    def __init__(self, model):
        vis = [m for m in model.modules() if type(m).__name__ == "Qwen3VLVisionModel"]
        if len(vis) != 1:
            raise RuntimeError(f"expected exactly one Qwen3VLVisionModel, found {len(vis)}")
        self.embeds = None
        self.deep = []
        self.handle = vis[0].register_forward_hook(self._hook)

    def close(self):
        self.handle.remove()

    def _hook(self, module, args, output):
        if not hasattr(output, "pooler_output"):
            raise RuntimeError("vision output has no pooler_output; return_dict off?")
        leaf = output.pooler_output.detach().requires_grad_(True)
        output.pooler_output = leaf
        self.embeds = leaf
        self.deep = []
        feats = getattr(output, "deepstack_features", None)
        if feats:
            new = []
            for t in feats:
                d = t.detach().requires_grad_(True)
                new.append(d)
                self.deep.append(d)
            output.deepstack_features = new
        return output


def grad_maps(model, leaves, logits, ids, prompt_len, spans):
    """-> maps [4, 1, S, M] in the order gnorm, gnorm_ds, gxi, gxi_ds.

    One forward, S backwards: `retain_graph=True` keeps the single graph alive across
    the steps rather than re-running the model once per step.
    """
    logp = torch.log_softmax(logits[0].float(), dim=-1)
    targets = [leaves.embeds] + list(leaves.deep)
    # The leaves are the tensors being differentiated, so they carry requires_grad.
    # gradient-times-input multiplies by them, and without detaching here that product
    # is a graph node -- which both keeps the forward graph alive past the last
    # backward and makes the final .numpy() raise.
    e = leaves.embeds.detach().float()
    deep = [t.detach().float() for t in leaves.deep]
    out = []
    for si, (a, b) in enumerate(spans):
        # logits_to_keep trimmed the leading positions: absolute position p-1 predicts
        # token p and now lives at index p - prompt_len.
        idx = torch.arange(a, b, device=logits.device)
        f = logp[idx - prompt_len, ids[0, idx]].sum()
        gs = torch.autograd.grad(f, targets, retain_graph=(si < len(spans) - 1),
                                 allow_unused=True)
        ge = gs[0].float()
        gnorm = ge.norm(dim=-1)
        gxi = (e * ge).sum(-1)
        sq = gnorm ** 2
        dot = gxi.clone()
        for lf, gd in zip(deep, gs[1:]):
            if gd is None:
                continue
            gd = gd.float()
            sq = sq + gd.norm(dim=-1) ** 2
            dot = dot + (lf * gd).sum(-1)
        out.append(torch.stack([gnorm, sq.clamp_min(0).sqrt(), gxi.abs(), dot.abs()]))
    return torch.stack(out, dim=1).unsqueeze(1).detach().float().cpu().numpy()  # [4,1,S,M]


# ---------------------------------------------------------------------------
# per-case scan
# ---------------------------------------------------------------------------
@torch.no_grad()
def grade_case(model, processor, case, inputs, prompt_len, ids, answer_max_tokens):
    """The model's own greedy answer to its own chain, for the correctness label."""
    fwd = build_forward(inputs, ids, prompt_len)
    out = model(**fwd, use_cache=True)
    past, nxt = out.past_key_values, out.logits[0, -1].argmax().view(1, 1)
    got = [int(nxt)]
    eos = processor.tokenizer.eos_token_id
    for _ in range(answer_max_tokens - 1):
        if got[-1] == eos:
            break
        o = model(input_ids=nxt, past_key_values=past, use_cache=True)
        past, nxt = o.past_key_values, o.logits[0, -1].argmax().view(1, 1)
        got.append(int(nxt))
    return processor.tokenizer.decode(got, skip_special_tokens=True)


def build_forward(inputs, ids, prompt_len):
    fwd = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    if "pixel_values" in inputs:
        fwd["pixel_values"] = inputs["pixel_values"]
        fwd["image_grid_thw"] = inputs["image_grid_thw"]
    if inputs.get("mm_token_type_ids") is not None:
        pad = torch.zeros(1, ids.shape[1] - prompt_len, dtype=torch.long, device=ids.device)
        fwd["mm_token_type_ids"] = torch.cat([inputs["mm_token_type_ids"], pad], dim=1)
    return fwd


def scan_case(model, processor, engine, case, image, device, args):
    """-> (maps [K,1,S,P], names, model's own answer, kept step indices) or None."""
    text = PROBE.build_prompt(processor, case["question"])
    inputs = processor(text=[text], images=[[image]], return_tensors="pt",
                       padding=True, padding_side="left", add_special_tokens=False).to(device)
    prompt_len = inputs["input_ids"].shape[1]
    chain = case["chain_ids"]
    gh, gw = case["grid"]
    img_cols = (inputs["input_ids"][0] == IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0]
    if img_cols.numel() != gh * gw:
        return None                      # grid and image tokens disagree: skip, not guess

    spans, kept = [], []
    for si, st in enumerate(case["steps"]):
        a, b = prompt_len + st["tok_a"], prompt_len + st["tok_b"]
        if b > prompt_len + len(chain) or b <= a or a <= 0:
            continue
        spans.append((a, b))
        kept.append(si)
    if not spans:
        return None

    ids = torch.tensor([inputs["input_ids"][0].tolist() + chain], device=device)
    answer = grade_case(model, processor, case, inputs, prompt_len, ids,
                        args.answer_max_tokens)

    if args.map == "grad":
        leaves = engine
        with torch.enable_grad():
            fwd = build_forward(inputs, ids, prompt_len)
            out = model(**fwd, use_cache=False, logits_to_keep=len(chain) + 1)
            maps = grad_maps(model, leaves, out.logits, ids, prompt_len, spans)
        del out
        names = ["gnorm", "gnorm_ds", "gxi", "gxi_ds"]
        return maps, names, answer, kept

    if args.map == "glimpse":
        SV = _glimpse_module()
        # glimpse_map takes CHAIN-relative spans in (text, a, b) triples; `spans` here
        # are absolute. The text slot is what saliency_viz captions a panel with and is
        # never read by the map, so it stays empty rather than inventing a second
        # meaning for it -- the probe grounds against the union `prepare` already built,
        # not against text it re-derives here.
        gsteps = [("", a - prompt_len, b - prompt_len) for a, b in spans]
        gmaps, _info = SV.glimpse_map(model, processor, inputs, ids, prompt_len, gsteps,
                                      gh, gw, case["question"], args, device)
        # [S, gh, gw] -> the [K, 1, S, P] this scan scores, with one column. The patch
        # order is row-major over the grid on both sides: `mask_q` is unpacked with the
        # same (gh, gw) reshape a few lines below, and glimpse_map fills `out[si]` from
        # `m.reshape(gh, gw)` over `img_cols`, which is image-token order.
        maps = np.ascontiguousarray(
            gmaps.reshape(1, 1, len(spans), gh * gw), dtype=np.float32)
        return maps, ["glimpse"], answer, kept

    # rollout: snapshot every step token, plus the token before each step for `inc`
    need = sorted({int(p) for a, b in spans for p in range(a, b)}
                  | {int(a - 1) for a, _ in spans})
    pos = {p: i for i, p in enumerate(need)}
    rows = torch.tensor(need, device=device)
    seq = ids.shape[1]
    engine.arm(seq, img_cols, rows,
               IV.causal_mask(seq, next(model.parameters()).dtype, device))
    try:
        with torch.no_grad():
            model(**build_forward(inputs, ids, prompt_len), use_cache=False)
        snaps = torch.stack(engine.snaps).cpu()              # [n_layers, n_need, M]
    finally:
        engine.disarm()

    n_l, _, m = snaps.shape
    maps = np.zeros((2 * n_l, 1, len(spans), m), dtype=np.float32)
    for si, (a, b) in enumerate(spans):
        sel = [pos[p] for p in range(a, b)]
        mean = snaps[:, sel].mean(dim=1)                     # [n_layers, M]
        maps[:n_l, 0, si] = mean.numpy()
        maps[n_l:, 0, si] = (mean - snaps[:, pos[a - 1]]).numpy()
    names = ([f"L{l}" for l in engine.layers]
             + [f"inc{l}" for l in engine.layers])
    return maps, names, answer, kept


# ---------------------------------------------------------------------------
# stage: scan
# ---------------------------------------------------------------------------
def scan(args, device):
    out = Path(args.out_dir)
    dest = out / "scan" / f"shard{args.shard:02d}.npz"
    if dest.exists() and not args.overwrite:
        print(f"[scan] {dest} exists -- nothing to do (--overwrite to redo)")
        return
    cases, cfg, _fp = IV.load_cases(Path(args.cases_dir), args.shard, args.num_shards,
                                    args.max_cases)
    imgs = IV.load_case_images(cfg, f"_fc{args.shard}")
    missing = [c["row_index"] for c in cases if c["row_index"] not in imgs]
    if missing:
        raise SystemExit(f"{len(missing)}/{len(cases)} cases have no image "
                         f"(e.g. row {missing[0]}); cases were prepared with {cfg}")
    dest.parent.mkdir(parents=True, exist_ok=True)

    # The rollout's per-layer [P, P] x [P, M] product is ~6 GFLOP, 36 times per case,
    # and it is the only fp32 matmul in the loop. Without TF32 it runs at ~10 TFLOPS
    # and costs more than the eager attention it is folding; with it, ~3 ms. The
    # ~1e-3 relative error is far below anything a rank statistic or a mean ratio can
    # see. --no-tf32 restores exact fp32 for a precision check.
    torch.backends.cuda.matmul.allow_tf32 = args.tf32

    # The gradient map needs a graph, and from_pretrained leaves every parameter
    # trainable -- a backward would then allocate a full 8B set of .grad buffers for
    # gradients nobody reads. Only the image-embedding leaves should require grad.
    # glimpse wants sdpa for the same reason grad does, from the other direction: it
    # replays ONE layer at a time in eager on top of an sdpa forward, and an all-eager
    # forward would put all 36 layers' [H, N, N] back in the graph and undo that.
    attn_impl = "sdpa" if args.map in ("grad", "glimpse") else args.attn_impl
    processor, model = PROBE.load_model(args.base_model, args.adapter or None, device,
                                        attn_impl)
    model.requires_grad_(False)
    if args.map == "grad":
        engine = ImageEmbedLeaves(model)
        if args.grad_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
    elif args.map == "glimpse":
        # glimpse_map turns checkpointing off around its own forward regardless, since
        # it needs that graph; --grad-checkpointing is a grad-only knob.
        engine = _NoEngine()
    else:
        engine = RolloutFlow(model, args.map.split("_", 1)[1], args.alpha, args.chunk)

    print(f"[scan] shard {args.shard}: {len(cases)} cases, map={args.map}"
          + (f", alpha={args.alpha}" if args.map.startswith("rollout") else ""),
          flush=True)
    prog = IV.Progress(out / "progress" / f"scan{args.shard:02d}.json", len(cases),
                       f"scan{args.shard}", args.log_every)

    V2, AU, ROW, STEP, COR, UNI, MASS = [], [], [], [], [], [], []
    SH, NEG, NPAT, NTOK, DSET = [], [], [], [], []
    names = None
    dropped = 0
    try:
        for case in cases:
            try:
                r = scan_case(model, processor, engine, case,
                              imgs[case["row_index"]]["image"], device, args)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"[scan] OOM on row {case['row_index']}; skipped", flush=True)
                r = None
            if r is None:
                dropped += 1
                prog.tick()
                continue
            maps, names, answer, kept = r
            gh, gw = case["grid"]
            steps = [case["steps"][i] for i in kept]
            masks = np.stack([IV.unb64u8(st["mask_q"], (gh, gw)).astype(bool).reshape(-1)
                              for st in steps])
            v2, au = HC.metrics(maps, masks)                 # each [K, 1, S]
            # The column's own magnitude, as a covariate for the partial correlations.
            # For a rollout column this is exactly mass(sal), the fraction of the
            # step's content traceable to the image; for an increment it is the change
            # in that fraction and may be negative; for the gradient map it is a scale
            # with no such interpretation, kept only so the control can run.
            mass = maps.sum(-1)                              # [K, 1, S]
            # Box-free concentration of the same maps: how sharp, never where. Rides
            # along with the scan for the cost of one sort over the patch axis. The
            # `incL` columns go negative and are rectified first -- `neg_frac` records
            # how much of their absolute mass that threw away.
            sh, neg = SHARP.sharpness(maps, (gh, gw))        # [K,1,S,M], [K,1,S]
            grade = PROBE.accuracy_reward(
                [[{"role": "assistant", "content": f"</think> {answer}"}]],
                [case["gold"]])[0]
            if grade is None:                    # ungradable answer: not "wrong"
                dropped += 1
                prog.tick()
                continue
            for si, st in enumerate(steps):
                V2.append(v2[:, 0, si])
                AU.append(au[:, 0, si])
                MASS.append(mass[:, 0, si])
                SH.append(sh[:, 0, si])
                NEG.append(neg[:, 0, si])
                ROW.append(case["row_index"])
                STEP.append(si)
                COR.append(float(grade))
                UNI.append(st["union_frac"])
                NPAT.append(gh * gw)
                NTOK.append(st["tok_b"] - st["tok_a"])
                DSET.append(case.get("dataset", ""))
            prog.tick()
    finally:
        engine.close()
        prog.close()
    if not V2:
        raise SystemExit("no steps scored")
    np.savez_compressed(
        dest,
        v2=np.stack(V2).astype(np.float32), auroc=np.stack(AU).astype(np.float32),
        row=np.array(ROW), step=np.array(STEP),
        correct=np.array(COR, dtype=np.float32), union=np.array(UNI, dtype=np.float32),
        mass=np.stack(MASS).astype(np.float32),
        sharp=np.stack(SH).astype(np.float32),
        neg_frac=np.stack(NEG).astype(np.float32),
        sharp_names=np.array(SHARP.SHARP_NAMES),
        npatch=np.array(NPAT), ntok=np.array(NTOK), dataset=np.array(DSET),
        names=np.array(names), map=np.array(args.map), alpha=np.array(args.alpha))
    print(f"[scan] shard {args.shard}: {len(V2)} steps from "
          f"{len(set(ROW))} completions, {dropped} cases dropped -> {dest}")
    print(SHARP.describe(np.stack(SH)))


# ---------------------------------------------------------------------------
# stage: report
# ---------------------------------------------------------------------------
def partial_corr(x, y, z):
    """Pearson r between x and y after linearly removing the columns of z from both.

    -> (r, n). NaN when fewer than 12 rows are finite in all of x, y and z. Constant
    covariates are dropped rather than left to make the design rank-deficient.
    """
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z).all(axis=1)
    nn = int(ok.sum())
    if nn < 12:
        return np.nan, nn
    xs = x[ok].astype(np.float64)
    ys = y[ok].astype(np.float64)
    zs = z[ok].astype(np.float64)
    cols = [np.ones(nn)] + [zs[:, j] for j in range(zs.shape[1]) if zs[:, j].std() > 0]
    des = np.column_stack(cols)
    rx = xs - des @ np.linalg.lstsq(des, xs, rcond=None)[0]
    ry = ys - des @ np.linalg.lstsq(des, ys, rcond=None)[0]
    if rx.std() <= 0 or ry.std() <= 0:
        return np.nan, nn
    return float((rx * ry).mean() / (rx.std() * ry.std())), nn


def sample_columns(ix, want=9):
    """A readable spread of column indices, always including the first and the last."""
    if len(ix) <= want:
        return list(ix)
    stride = max(1, (len(ix) - 1) // (want - 1))
    out = list(ix[::stride])
    if ix[-1] not in out:
        out.append(ix[-1])
    return out


def report(args):
    out = Path(args.out_dir)
    files = sorted((out / "scan").glob("shard*.npz"))
    if not files:
        raise SystemExit(f"no scan output under {out / 'scan'}")
    d = [np.load(f, allow_pickle=False) for f in files]
    v2 = np.concatenate([x["v2"] for x in d])
    au = np.concatenate([x["auroc"] for x in d])
    row = np.concatenate([x["row"] for x in d])
    cor = np.concatenate([x["correct"] for x in d])
    uni = np.concatenate([x["union"] for x in d])
    stp = np.concatenate([x["step"] for x in d])
    has_mass = "mass" in d[0].files                 # absent in pre-2026-08-06 scans
    mass = np.concatenate([x["mass"] for x in d]) if has_mass else None
    names = [str(s) for s in d[0]["names"]]
    which = str(d[0]["map"])
    k = v2.shape[1]
    base_i = [i for i, nm in enumerate(names) if not nm.startswith("inc")]
    inc_i = [i for i, nm in enumerate(names) if nm.startswith("inc")]
    primary = names[base_i[-1]] if which.startswith("rollout") else names[0]
    order = (list(range(k)) if args.all_columns
             else sorted(set(sample_columns(base_i) + sample_columns(inc_i))))

    # The union curve goes first and on everything, before any cap -- it is what the
    # cap is chosen from, so restricting it would hide the tail being cut.
    HC.union_decile_table(uni, {names[i]: au[:, i] for i in order}, null=0.5)

    (v2, au, row, cor, uni, stp, mass), keep = HC.apply_union_cap(
        args.max_union, uni, (v2, au, row, cor, uni, stp, mass))
    if not keep.all():
        print(f"\n--max-union {args.max_union}: {int(keep.sum())}/{len(keep)} steps "
              f"and {len(np.unique(row))} completions kept. Everything below is that "
              f"subset -- including the Bonferroni threshold, which rises as the "
              f"completion count falls. The table above is not.")

    n, k = v2.shape
    uniq = np.unique(row)
    acc = cor[np.unique(row, return_index=True)[1]].mean()
    print(f"map {which}   steps {n}   completions {len(uniq)}   columns {k}   "
          f"accuracy {acc:.3f}")
    print(f"chance |r| at n={len(uniq)}: 1.96/sqrt(n-3) = "
          f"{1.96 / np.sqrt(len(uniq) - 3):.4f} (two-sided, single test)")
    if not has_mass:
        print("NOTE: this scan predates the `mass` column, so the partial correlations "
              "control only for union area and step count. Re-run the scan for the "
              "full control set.")

    idx = np.searchsorted(uniq, row)
    ccor = np.zeros(len(uniq))
    np.maximum.at(ccor, idx, cor)              # label is constant within a completion

    def by_completion(arr):
        """Mean over each completion's steps, NaN-aware. [N] or [N,K] -> [C] or [C,K]."""
        shape = (len(uniq),) if arr.ndim == 1 else (len(uniq), arr.shape[1])
        s, cnt = np.zeros(shape), np.zeros(shape)
        np.add.at(s, idx, np.nan_to_num(arr, nan=0.0))
        np.add.at(cnt, idx, np.isfinite(arr).astype(float))
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(cnt > 0, s / cnt, np.nan)

    cagg = {"mean_in_v2": by_completion(v2), "auroc": by_completion(au)}

    # Covariates for the partial correlations. Step count is a completion property, so
    # at step level every step of a completion carries its completion's value.
    cns = np.zeros(len(uniq))
    np.maximum.at(cns, idx, stp + 1.0)
    cov_s = [uni.astype(np.float64), cns[idx]]
    cov_c = [by_completion(uni), cns]
    cmass = by_completion(mass) if has_mass else None

    sel_c, sel_s = (uniq % 2 == 1), (row % 2 == 1)
    # col_corr refuses to report an r from fewer than 8 pairs, so on a small run the
    # select/held-out columns come back all-NaN. That is the guard doing its job, not
    # a failure -- say so, because a wall of `nan` reads like one.
    small = min(sel_c.sum(), (~sel_c).sum())
    if small < 8:
        print(f"NOTE: the smaller odd/even half has {small} completions; col_corr needs "
              f"8, so r(select)/r(HELD OUT) will be NaN at completion level. Expected "
              f"on a --max-cases smoke run, not on the full scan.")
    saved, table = {}, []
    for name, sarr, carr in (("mean_in_v2", v2, cagg["mean_in_v2"]),
                             ("auroc", au, cagg["auroc"])):
        for setup, X, y, sel, cov, mcov in (
                ("step", sarr, cor, sel_s, cov_s, mass),
                ("completion", carr, ccor, sel_c, cov_c, cmass)):
            r_all = HC.col_corr(X[:, :, None], y)[:, 0]
            r_sel = HC.col_corr(X[sel][:, :, None], y[sel])[:, 0]
            r_out = HC.col_corr(X[~sel][:, :, None], y[~sel])[:, 0]
            # Partial out union area, step count, and the column's OWN magnitude: a
            # map's shape predicting correctness is a different claim from its size
            # doing so, and union area moves mean_in_v2 mechanically.
            r_par = np.full(k, np.nan)
            for i in range(k):
                z = cov + ([mcov[:, i]] if mcov is not None else [])
                r_par[i] = partial_corr(X[:, i], y, np.column_stack(z))[0]
            # The LEVEL, which no amount of correlation substitutes for: a column can
            # predict correctness while sitting on the wrong side of chance, in which
            # case "more overlap -> more correct" is really "less anti-overlap -> more
            # correct". null is 0.5 for auroc (the union ranks no higher than the rest
            # of the image) and 1.0 for mean_in_v2 (in-mask mean == overall mean).
            null = 0.5 if name == "auroc" else 1.0
            with np.errstate(invalid="ignore"):
                lvl = np.nanmean(X, axis=0)
                lse = 2 * np.nanstd(X, axis=0) / np.sqrt(np.isfinite(X).sum(axis=0))
            saved[f"{name}_{setup}"] = np.stack([r_all, r_sel, r_out, r_par, lvl, lse])
            print(f"\n=== {name} / {setup}-level (n={len(y)}) ===")
            print(f"   {'column':>10} {'r(all)':>9} {'r(select)':>10} "
                  f"{'r(HELD OUT)':>12} {'r(PARTIAL)':>11} {'level':>8} {'n':>7}")
            for i in order:
                if name == "mean_in_v2" and names[i].startswith("inc"):
                    continue          # undefined wherever the increment's mean <= 0
                star = "  <- PRIMARY" if names[i] == primary else ""
                side = ("" if not np.isfinite(lvl[i]) or abs(lvl[i] - null) <= lse[i]
                        else ("+" if lvl[i] > null else "-"))
                cnt = int(np.isfinite(X[:, i]).sum())
                print(f"   {names[i]:>10} {r_all[i]:>+9.4f} {r_sel[i]:>+10.4f} "
                      f"{r_out[i]:>+12.4f} {r_par[i]:>+11.4f} "
                      f"{lvl[i]:>7.4f}{side:<1} {cnt:>7}{star}")
            print(f"   level = the column's mean; chance is {null:.1f}. "
                  f"'+'/'-' marks a level more than 2 SE above/below it.")
            for i in range(k):
                if np.isfinite(r_all[i]) and not (name == "mean_in_v2"
                                                  and names[i].startswith("inc")):
                    table.append((name, setup, names[i], r_all[i], r_sel[i],
                                  r_out[i], r_par[i]))

    # What the primary-only summary used to hide: rank EVERY column that clears a
    # Bonferroni threshold over every test this report ran, whatever its role. The
    # effective n is the completion count in both set-ups -- steps within a completion
    # share a label and are not independent.
    from scipy.stats import norm
    n_tests = len(table)
    thr = norm.ppf(1 - 0.025 / max(1, n_tests)) / np.sqrt(len(uniq) - 3)
    print(f"\n=== ABOVE THRESHOLD ===")
    print(f"   {n_tests} tests reported; Bonferroni |r| >= {thr:.4f} "
          f"(alpha 0.05, effective n = {len(uniq)} completions)")
    hits = sorted([t for t in table if abs(t[3]) >= thr], key=lambda t: -abs(t[3]))
    if not hits:
        print("   nothing clears it.")
    else:
        print(f"   {'metric':>11} {'setup':>11} {'column':>10} {'r(all)':>9} "
              f"{'r(select)':>10} {'r(HELD OUT)':>12} {'r(PARTIAL)':>11}")
        for m_, s_, c_, ra, rs, ro, rp in hits:
            print(f"   {m_:>11} {s_:>11} {c_:>10} {ra:>+9.4f} {rs:>+10.4f} "
                  f"{ro:>+12.4f} {rp:>+11.4f}")
    np.savez_compressed(out / "corr.npz", names=np.array(names), threshold=thr,
                        max_union=np.array(args.max_union), **saved)
    print(f"\n-> {out}/corr.npz   (rows of each array: r_all, r_select, r_heldout, "
          f"r_partial, level, level_2se)")
    print("PRIMARY is the pre-registered readout: the last layer for a rollout, gnorm "
          "for the gradient map. ABOVE THRESHOLD ignores that and ranks everything, "
          "so a secondary column cannot hide behind it. r(PARTIAL) holds union area, "
          "step count and the column's own mass fixed; a column whose r survives but "
          "whose partial r does not is measuring difficulty, not grounding. And read "
          "`level` alongside every r: a column below chance that still correlates "
          "positively says less anti-grounding goes with being right, which is not the "
          "same claim as grounding going with being right.")


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", default="scan", choices=["scan", "report"])
    p.add_argument("--map", default="rollout_mean", choices=list(MAPS))
    p.add_argument("--out-dir", required=True)
    p.add_argument("--cases-dir", default="",
                   help="an intervene_probe out-dir whose cases/ holds the chains and "
                        "per-step DINO unions; defaults to --out-dir")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="rollout attention/residual mix; 0.5 is the convention")
    p.add_argument("--chunk", type=int, default=256,
                   help="row/source chunk for the wnorm edge weights")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--max-cases", type=int, default=0)
    p.add_argument("--base-model", default=str(repo_path(
        "checkpoint/coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged")))
    p.add_argument("--adapter", default="")
    p.add_argument("--attn-impl", default="sdpa")
    p.add_argument("--grad-checkpointing", action="store_true",
                   help="--map grad only; trades ~30%% compute for much less activation "
                        "memory. Not needed at 80 GB.")
    # --map glimpse only. Every default is saliency_viz.py's, so a column in this scan
    # and a drawn panel are the same object; see docs/saliency-maps.md section 6 and
    # `glimpse_layer_alphas` for why 0.36 is the other candidate depth temperature.
    p.add_argument("--glimpse-temp", type=float, default=0.5,
                   help="lambda, head-fusion temperature (eq 6)")
    p.add_argument("--glimpse-depth-temp", type=float, default=0.2,
                   help="lambda_d, depth prior temperature (eq 9)")
    p.add_argument("--glimpse-layer-frac", type=float, default=1.0,
                   help="propagate the last frac of the stack; the paper's ablation "
                        "loses nothing at 0.6, and here it is the one dial that buys "
                        "back wall clock -- it cuts the per-token eager replay")
    p.add_argument("--glimpse-target", default="logit",
                   choices=["clogit", "logit", "logprob"],
                   help="z_t in eqs 5 and 16; the paper's is the raw logit")
    p.add_argument("--glimpse-token-weight", default="full",
                   choices=["full", "confidence", "prompt", "uniform"],
                   help="eq 18, and the paper's token-saliency ablation")
    p.add_argument("--no-tf32", dest="tf32", action="store_false",
                   help="exact fp32 for the rollout matmul; several times slower")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--answer-max-tokens", type=int, default=16)
    p.add_argument("--all-columns", action="store_true",
                   help="report every per-layer readout, not a sample of them")
    p.add_argument("--max-union", type=float, default=0.0,
                   help="report stage: drop steps whose DINO union covers more than "
                        "this fraction of the patch grid (0 = off, the default and "
                        "what every published number used). Fix it BEFORE looking at "
                        "a confirmation set -- it is a researcher degree of freedom "
                        "otherwise.")
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if not args.cases_dir:
        args.cases_dir = args.out_dir
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    if args.stage == "report":
        return report(args)
    return scan(args, args.device)


if __name__ == "__main__":
    sys.exit(main() or 0)
