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

Three replacement maps, scored by the same two metrics against the same per-step DINO
unions as the direct scan, on the same prepared cases, so the numbers are directly
comparable:

  rollout_mean   layer-wise attention rollout, heads merged by the mean
  rollout_wnorm  the same, heads merged by || sum_h A^h_{n,k} W_O^h v^h_k ||
  grad           || d log P(step's own tokens) / d e_j ||, e_j = image embedding j

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

Increment. sal is cumulative -- step k's map contains steps 1..k-1's objects -- so a
better `r` can come from the map absorbing a completion-level signal rather than from
per-step grounding. The `inc` column subtracts the map at the token immediately before
the step from the step's own mean map, isolating what the step's span added. Read
AUROC for it; mean_in_v2 divides by the map's mean, which an increment can drive to
zero, so it is NaN for a non-random subset of steps and the report drops it.

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
    bash launch_flow_correlation.sh --gpus 8 --out-dir DIR --cases-dir <probe out-dir> \
         --maps rollout_mean,rollout_wnorm,grad
    python flow_correlation_probe.py --stage report --out-dir DIR/rollout_mean
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
IMAGE_TOKEN_ID = PROBE.IMAGE_TOKEN_ID

MAPS = ("rollout_mean", "rollout_wnorm", "grad")


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
            self.snaps.append(self.sal[self.rows].clone().cpu())
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
    out = []
    for si, (a, b) in enumerate(spans):
        # logits_to_keep trimmed the leading positions: absolute position p-1 predicts
        # token p and now lives at index p - prompt_len.
        idx = torch.arange(a, b, device=logits.device)
        f = logp[idx - prompt_len, ids[0, idx]].sum()
        gs = torch.autograd.grad(f, targets, retain_graph=(si < len(spans) - 1),
                                 allow_unused=True)
        ge = gs[0].float()
        e = leaves.embeds.float()
        gnorm = ge.norm(dim=-1)
        gxi = (e * ge).sum(-1)
        sq = gnorm ** 2
        dot = gxi.clone()
        for lf, gd in zip(leaves.deep, gs[1:]):
            if gd is None:
                continue
            gd = gd.float()
            sq = sq + gd.norm(dim=-1) ** 2
            dot = dot + (lf.float() * gd).sum(-1)
        out.append(torch.stack([gnorm, sq.clamp_min(0).sqrt(), gxi.abs(), dot.abs()]))
    return torch.stack(out, dim=1).unsqueeze(1).float().cpu().numpy()   # [4,1,S,M]


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
        snaps = torch.stack(engine.snaps)                    # [n_layers, n_need, M]
    finally:
        engine.disarm()

    n_l, _, m = snaps.shape
    maps = np.zeros((n_l + 1, 1, len(spans), m), dtype=np.float32)
    for si, (a, b) in enumerate(spans):
        sel = [pos[p] for p in range(a, b)]
        mean = snaps[:, sel].mean(dim=1)                     # [n_layers, M]
        maps[:n_l, 0, si] = mean.numpy()
        maps[n_l, 0, si] = (mean[-1] - snaps[-1, pos[a - 1]]).numpy()
    names = [f"L{l}" for l in engine.layers] + ["inc"]
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
    attn_impl = "sdpa" if args.map == "grad" else args.attn_impl
    processor, model = PROBE.load_model(args.base_model, args.adapter or None, device,
                                        attn_impl)
    model.requires_grad_(False)
    if args.map == "grad":
        engine = ImageEmbedLeaves(model)
        if args.grad_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        engine = RolloutFlow(model, args.map.split("_", 1)[1], args.alpha, args.chunk)

    print(f"[scan] shard {args.shard}: {len(cases)} cases, map={args.map}"
          + (f", alpha={args.alpha}" if args.map != "grad" else ""), flush=True)
    prog = IV.Progress(out / "progress" / f"scan{args.shard:02d}.json", len(cases),
                       f"scan{args.shard}", args.log_every)

    V2, AU, ROW, STEP, COR, UNI = [], [], [], [], [], []
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
                ROW.append(case["row_index"])
                STEP.append(si)
                COR.append(float(grade))
                UNI.append(st["union_frac"])
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
        names=np.array(names), map=np.array(args.map), alpha=np.array(args.alpha))
    print(f"[scan] shard {args.shard}: {len(V2)} steps from "
          f"{len(set(ROW))} completions, {dropped} cases dropped -> {dest}")


# ---------------------------------------------------------------------------
# stage: report
# ---------------------------------------------------------------------------
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
    names = [str(s) for s in d[0]["names"]]
    which = str(d[0]["map"])
    n, k = v2.shape
    uniq = np.unique(row)
    acc = cor[np.unique(row, return_index=True)[1]].mean()
    print(f"map {which}   steps {n}   completions {len(uniq)}   columns {k}   "
          f"accuracy {acc:.3f}")
    print(f"chance |r| at n={len(uniq)}: 1.96/sqrt(n-3) = "
          f"{1.96 / np.sqrt(len(uniq) - 3):.4f} (two-sided, single test)")

    idx = np.searchsorted(uniq, row)
    ccor = np.zeros(len(uniq))
    np.maximum.at(ccor, idx, cor)              # label is constant within a completion
    cagg = {}
    for name, arr in (("mean_in_v2", v2), ("auroc", au)):
        s = np.zeros((len(uniq), k))
        cnt = np.zeros((len(uniq), k))
        np.add.at(s, idx, np.nan_to_num(arr, nan=0.0))
        np.add.at(cnt, idx, np.isfinite(arr).astype(float))
        with np.errstate(invalid="ignore", divide="ignore"):
            cagg[name] = np.where(cnt > 0, s / cnt, np.nan)

    sel_c, sel_s = (uniq % 2 == 1), (row % 2 == 1)
    primary = names[-2] if which.startswith("rollout") else names[0]
    saved = {}
    for name, sarr, carr in (("mean_in_v2", v2, cagg["mean_in_v2"]),
                             ("auroc", au, cagg["auroc"])):
        for setup, X, y, sel in (("step", sarr, cor, sel_s),
                                 ("completion", carr, ccor, sel_c)):
            r_all = HC.col_corr(X[:, :, None], y)[:, 0]
            r_sel = HC.col_corr(X[sel][:, :, None], y[sel])[:, 0]
            r_out = HC.col_corr(X[~sel][:, :, None], y[~sel])[:, 0]
            saved[f"{name}_{setup}"] = np.stack([r_all, r_sel, r_out])
            print(f"\n=== {name} / {setup}-level (n={len(y)}) ===")
            print(f"   {'column':>10} {'r(all)':>9} {'r(select)':>10} "
                  f"{'r(HELD OUT)':>12} {'n':>7}")
            order = list(range(k)) if args.all_columns else \
                [i for i, nm in enumerate(names)
                 if nm == primary or nm == names[-1] or i % max(1, k // 8) == 0]
            for i in order:
                if name == "mean_in_v2" and names[i] == "inc":
                    continue          # undefined wherever the increment's mean <= 0
                star = "  <- PRIMARY" if names[i] == primary else ""
                cnt = int(np.isfinite(X[:, i]).sum())
                print(f"   {names[i]:>10} {r_all[i]:>+9.4f} {r_sel[i]:>+10.4f} "
                      f"{r_out[i]:>+12.4f} {cnt:>7}{star}")
            if k > 2:
                best = int(np.nanargmax(np.abs(np.nan_to_num(r_sel, nan=0.0))))
                print(f"   best on the SELECT half: {names[best]}  "
                      f"select {r_sel[best]:+.4f} -> held out {r_out[best]:+.4f}")
    np.savez_compressed(out / "corr.npz", names=np.array(names), **saved)
    print(f"\n-> {out}/corr.npz")
    print("PRIMARY is the last layer's readout (or gnorm for the gradient map); every "
          "other column is a secondary whose held-out value is the one to believe.")


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
    p.add_argument("--no-tf32", dest="tf32", action="store_false",
                   help="exact fp32 for the rollout matmul; several times slower")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--answer-max-tokens", type=int, default=16)
    p.add_argument("--all-columns", action="store_true",
                   help="report every per-layer readout, not a sample of them")
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
