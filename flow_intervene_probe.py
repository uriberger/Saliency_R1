#!/usr/bin/env python
"""Causal test of the INDIRECT flow: make the step read more from the boxed objects.

`flow_correlation_probe.py` is correlational -- it measures how much of a step's
content is traceable to the DINO union and correlates that with correctness. Its one
surviving column (`inc34`/`inc35`, r=+0.12) therefore cannot distinguish grounding from
something upstream producing both. `intervene_probe.py` is causal but acts on the DIRECT
path: it edits one layer's attention from the step's tokens to the IMAGE tokens, which
is exactly the path the flow thesis says carries little traffic. Both nulls it returned
are consistent with never having moved the thing that matters.

This probe intervenes on the indirect path. At every layer up to a cutoff, it holds
fixed how much each head reads from image-CARRYING positions and re-allocates that mass
across them in proportion to how much boxed-object content each one holds. Text
positions that absorbed image content earlier are eligible keys; that is the whole point.

---------------------------------------------------------------------------
The edit
---------------------------------------------------------------------------
Per position, two scalars are carried across layers alongside the forward pass:

    u_q  union-traceable mass     u^(0)_q = 1 for image tokens inside the step's union
    m_q  image-traceable mass     m^(0)_q = 1 for every image token

Both obey the rollout recursion, because it is LINEAR in `sal` and these are linear
functionals of it -- which is what makes this affordable: O(P) per scalar instead of the
O(P*M) the full map would cost.

    x^(l)_q = a * sum_{r<=q} w^(l)_{q,r} x^(l-1)_r  +  (1-a) * x^(l-1)_q

with `w` the head-mean of the EDITED attention, a = 0.5. At layer l, for query p in a
step's span:

    E_p = { q <= p : m^(l-1)_q > 0 }                 eligible keys
    T_q = u^(l-1)_q / sum_{r in E_p} u^(l-1)_r       target, sums to 1 over E_p
    M^h_p = sum_{q in E_p} A^{l,h}_{p,q}             this head's mass on E_p

    A'^{l,h}_{p,q} = (1-alpha) A + alpha * M^h_p * T_q      q in E_p
    A'^{l,h}_{p,q} = A                                      otherwise

Row sums are preserved exactly, and so is M^h_p: nothing is taken from text that never
saw the image, and no mass lands on q > p. Only the split between union-carriers and
other image-carriers moves. alpha=0 is a no-op; alpha=1 puts all of E_p's mass in
proportion to union content.

T is normalised PER QUERY ROW over q <= p, not once over the whole sequence. Normalising
globally would put mass on positions after p, which the causal mask would then have to
discard, silently changing the row sum.

CONDITIONS, each a separate forward:

    box    T built from u, the step's own DINO union             the supervision target
    roll   T built from u', the union rolled to a random offset,  matched-area,
           same area                                             wrong-place control
    (alpha=0)  no edit, but the same rows are rebuilt through the same eager path,
           so the eager-vs-sdpa difference is common to baseline and intervention
           and cancels in the paired readout. One baseline per layer cutoff.

Read box - roll, never box alone: box alone also moves under any large perturbation.

DEEPSTACK. Qwen3-VL adds vision features into the residual stream at the image positions
at decoder layers 0,1,2 (`deepstack_visual_indexes` names the VISION layers the features
are tapped from -- 8,16,24 -- not the LM layers they land in, which are
range(len(...))). Fresh image content therefore enters after those layers, and `u`/`m`
are re-seeded there or every mass downstream is understated.

MANIPULATION CHECK, and this is not optional. For the direct probe, forcing mass into
the box moves the measured map by construction. Here the actuator (attention) and the
measurement (traceable mass) have come apart, so a null is uninterpretable without
evidence the manipulation landed. Every forward reports `ushare` = mean over the step's
own positions of u_p/m_p at the last layer, and `rshare` likewise for the rolled union.
`box` must raise ushare relative to baseline and `roll` must raise rshare; if neither
moves, the run says nothing about grounding.

STAGES
  selftest  alpha=0 must reproduce the un-hooked forward; alpha=1 must move it; the
            edit must preserve row sums and never touch q > p. Gates every run.
  run       the (cutoff, condition, alpha) grid. Append-only JSONL, resumable.
  report    paired bootstrap CIs on box - roll, per (cutoff, alpha), with the
            manipulation check beside every cell.
  monitor   aggregate every shard's heartbeat into one ETA.

This edit alongside the five maps it was built to test: docs/saliency-maps.md.

    bash launch_flow_intervene.sh --stage selftest --gpus 1 --out-dir DIR --cases-dir C
    bash launch_flow_intervene.sh --stage run --gpus 8 --out-dir DIR --cases-dir C
    python flow_intervene_probe.py --stage report --out-dir DIR
"""

from __future__ import annotations

import argparse
import importlib.util
import json
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


PROBE = _load_module("_fi_overlap_probe", "overlap_probe.py")
IV = _load_module("_fi_intervene", "intervene_probe.py")
IMAGE_TOKEN_ID = PROBE.IMAGE_TOKEN_ID

CONDITIONS = ("box", "roll")
ROLLOUT_A = 0.5


# ---------------------------------------------------------------------------
# the algebra -- pure, so the CPU tests reach it without a model
# ---------------------------------------------------------------------------
def build_targets(u_prev, m_prev, rows):
    """Per-row target distributions over the eligible keys.

    u_prev, m_prev  [P]        the previous layer's union / image traceable mass
    rows            [R] long   the query positions being edited

    -> T [R, P] float32, each finite row summing to 1 over { q <= p : m_prev[q] > 0 },
       and keep [R] bool, False where no eligible key carries any union mass (there is
       nothing to aim at, so the row is left alone).

    Normalisation is per row. A single global normalisation would place mass on q > p.
    """
    P = u_prev.shape[0]
    dev = u_prev.device
    elig = m_prev > 0
    base = torch.where(elig, u_prev, torch.zeros_like(u_prev)).to(torch.float32)
    causal = torch.arange(P, device=dev)[None, :] <= rows[:, None]     # [R, P]
    num = base[None, :] * causal                                       # [R, P]
    den = num.sum(-1)                                                  # [R]
    keep = den > 0
    T = torch.where(keep[:, None], num / den.clamp_min(1e-30).unsqueeze(-1),
                    torch.zeros_like(num))
    return T, keep


def edit_rows(a, rows, T, keep, m_prev, alpha):
    """The edit itself, returning ONLY the rewritten rows.

    a     [H, P, P]           attention weights, rows summing to 1, any float dtype
    rows  [R] long            query positions to edit
    T     [R, P] float32      from build_targets
    keep  [R] bool
    m_prev[P] float32
    -> (new [H, R, P] float32, mass [H, R] float32 = each head's pre-edit mass on E_p)

    Row sums and every M^h_p are preserved exactly in float32, because T sums to 1 over
    exactly the entries the (1-alpha) factor is applied to.

    Rows-only because the caller holds a [H, P, P] tensor -- ~300 MB at P=1500 -- and
    materialising a fresh copy per step would multiply that by the step count. The
    arithmetic is in float32 even when `a` is bf16; the write back down to bf16 is the
    same rounding the direct probe's Intervener takes, and it cancels in the paired
    readout because the baseline goes through this identical path at alpha=0.
    """
    P = a.shape[-1]
    dev = a.device
    elig = m_prev > 0
    causal = torch.arange(P, device=dev)[None, :] <= rows[:, None]
    sel = (elig[None, :] & causal) & keep[:, None]                     # [R, P]
    sub = a[:, rows, :].to(torch.float32)                              # [H, R, P]
    mass = (sub * sel[None]).sum(-1)                                   # [H, R]
    new = sub - alpha * sub * sel[None] + alpha * mass[..., None] * T[None]
    return new, mass


def edit_attention(a, rows, T, keep, m_prev, alpha):
    """edit_rows folded back into a full copy. The testable form; the hook uses
    edit_rows directly so it can clone once for all of a layer's steps."""
    new, mass = edit_rows(a, rows, T, keep, m_prev, alpha)
    out = a.clone()
    out[:, rows, :] = new.to(a.dtype)
    return out, mass


def propagate(x_prev, w, alpha=ROLLOUT_A):
    """One rollout layer. x_prev [P, K], w [P, P] row-stochastic -> [P, K]."""
    return alpha * (w @ x_prev) + (1.0 - alpha) * x_prev


def reseed(x, x0, layer_idx, deepstack_layers):
    """Deepstack ADDS visual features back at the image positions at these layers."""
    return x + x0 if int(layer_idx) in deepstack_layers else x


def deepstack_layers_of(model):
    """The LM layers deepstack injects into: range(len(deepstack_visual_indexes)).

    The config value names the VISION layers the features are tapped from. Reading it as
    a set of LM layers puts the re-seed in the wrong place, so it is derived, not read.
    """
    cfg = getattr(model, "config", None)
    vis = getattr(cfg, "vision_config", None)
    idx = getattr(vis, "deepstack_visual_indexes", None) if vis is not None else None
    return set(range(len(idx))) if idx else set()


def rolled_mask(mask_flat, gh, gw, rng):
    """The matched-area wrong-place control: same mask, random cyclic offset."""
    m2 = torch.as_tensor(mask_flat).view(gh, gw)
    m2 = torch.roll(m2, (int(rng.integers(0, gh)), int(rng.integers(0, gw))), (0, 1))
    return m2.reshape(-1)


def init_columns(n_pos, img_cols, step_masks, rolled_masks, device):
    """X0 [P, K]: column 0 is m, then (u_box, u_roll) for each step, in order."""
    K = 1 + 2 * len(step_masks)
    X0 = torch.zeros(n_pos, K, dtype=torch.float32, device=device)
    X0[img_cols, 0] = 1.0
    for i, (b, r) in enumerate(zip(step_masks, rolled_masks)):
        X0[img_cols, 1 + 2 * i] = b.to(torch.float32)
        X0[img_cols, 2 + 2 * i] = r.to(torch.float32)
    return X0


# ---------------------------------------------------------------------------
# the hook
# ---------------------------------------------------------------------------
class FlowIntervener:
    """Edits every attention module up to a cutoff, and carries u/m across them.

    One hook per layer. Each re-runs its own module in eager to recover the softmax
    weights sdpa discards, applies the edit to the steps' query rows, rebuilds those
    rows of the module output from the edited weights, and folds the head-mean of the
    edited attention into the carried scalars. Peak memory is one layer's [H, P, P].

    `spec is None` disarms it completely, so an un-hooked reference forward costs
    nothing.
    """

    def __init__(self, model, dtype):
        self.dtype = dtype
        self.spec = None
        self.mask = None
        self.X = None                # [P, K] carried scalars
        self.X0 = None
        self.deep = deepstack_layers_of(model)
        self._reentry = False
        self.n_rows = 0
        self.audit = False
        self.audit_stats = None
        self.handles, self.layers = [], []
        for m in model.modules():
            if type(m).__name__ == "Qwen3VLTextAttention" and hasattr(m, "layer_idx"):
                li = int(m.layer_idx)
                self.layers.append(li)
                self.handles.append(
                    m.register_forward_hook(self._make(li), with_kwargs=True))
        self.layers.sort()
        if not self.layers:
            raise RuntimeError("no Qwen3VLTextAttention modules found")

    def close(self):
        for h in self.handles:
            h.remove()

    def arm(self, spec, X0, mask):
        self.spec, self.X0, self.X, self.mask = spec, X0, X0.clone(), mask
        self.n_rows, self.audit_stats = 0, None

    def disarm(self):
        self.spec = self.mask = self.X = self.X0 = None

    def _make(self, layer_idx):
        def hook(module, args, kwargs, output):
            if self._reentry or self.spec is None:
                return None
            self._reentry = True
            kw = dict(kwargs)
            kw["attention_mask"] = self.mask
            kw["past_key_values"] = None          # never double-update the KV cache
            kw["use_cache"] = False
            prev = module.config._attn_implementation
            module.config._attn_implementation = "eager"
            try:
                _out, attn = module(*args, **kw)
            finally:
                module.config._attn_implementation = prev
                self._reentry = False

            sp = self.spec
            a = attn[0]                                        # [H, P, P], model dtype
            all_rows, new_rows = [], []
            if layer_idx <= sp["cutoff"]:
                m_prev = self.X[:, 0]
                pend = []
                for i, st in enumerate(sp["steps"]):
                    col = (1 + 2 * i) if sp["kind"] == "box" else (2 + 2 * i)
                    T, keep = build_targets(self.X[:, col], m_prev, st["rows"])
                    if not bool(keep.any()):
                        continue
                    new, _mass = edit_rows(a, st["rows"], T, keep, m_prev, sp["alpha"])
                    pend.append((st["rows"], new))
                    all_rows.append(st["rows"])
                if pend:
                    # Every step is built from the UNEDITED `a`, which is equivalent to
                    # chaining because steps' token spans are disjoint -- no step's edit
                    # can land on another step's query rows.
                    a = a.clone()                  # once per layer, not once per step
                    for r, new in pend:
                        a[:, r, :] = new.to(a.dtype)
            if all_rows:
                rows = torch.cat(all_rows)
                self.n_rows = int(rows.numel())
                hidden = args[0] if args else kwargs["hidden_states"]
                b, s, _ = hidden.shape
                v = module.v_proj(hidden).view(b, s, -1, module.head_dim).transpose(1, 2)
                v = IV.repeat_v(v, module.num_key_value_groups)
                ctx = torch.matmul(a[None][:, :, rows, :].to(v.dtype), v)
                ctx = ctx.transpose(1, 2).reshape(1, rows.numel(), -1)
                new_rows = module.o_proj(ctx)
                if self.audit:
                    # The WORST layer, not the last one. Keeping only the most recent
                    # made this report layer 35 alone and call a 36-layer rebuild
                    # faithful on the strength of one layer.
                    ref = _out[:, rows, :].float()
                    err = (new_rows.float() - ref).abs()
                    rel = float(err.max() / ref.abs().max().clamp_min(1e-12))
                    if self.audit_stats is None or rel > self.audit_stats["rel"]:
                        self.audit_stats = {
                            "rel": rel, "layer": int(layer_idx),
                            "max_abs_err": float(err.max()),
                            "max_abs_ref": float(ref.abs().max()),
                            "n_rows": int(rows.numel()),
                        }

            # float32 accumulation: a bf16 mean over 32 heads loses the small weights
            # that the whole indirect path is made of.
            w = a.mean(0, dtype=torch.float32)
            w = w / w.sum(-1, keepdim=True).clamp_min(1e-12)
            self.X = reseed(propagate(self.X, w), self.X0, layer_idx, self.deep)
            del a, w, attn

            if not all_rows:
                return None
            out0 = (output[0] if isinstance(output, tuple) else output).clone()
            out0[:, rows, :] = new_rows.to(out0.dtype)
            return ((out0,) + tuple(output[1:])) if isinstance(output, tuple) else out0
        return hook


# ---------------------------------------------------------------------------
# scoring one case
# ---------------------------------------------------------------------------
def case_image(imgs, row_index):
    """The PIL image for a row, or None.

    `IV.load_case_images` returns a RECORD per row -- {row_index, dataset, question,
    gt_answer, image} -- not a bare image. Handing the record straight to the processor
    raises deep inside `fetch_images` with "got type=<class 'dict'>", nowhere near the
    call site, so the unwrap lives here where a test can reach it.
    """
    rec = imgs.get(int(row_index))
    if rec is None:
        return None
    return rec["image"] if isinstance(rec, dict) else rec


def case_steps(case, prompt_len, gh, gw, device, seed):
    """Query rows and the box/rolled masks for every usable step of this case."""
    rng = np.random.default_rng(seed)
    steps, boxes, rolls = [], [], []
    for st in case["steps"]:
        a, b = prompt_len + st["tok_a"], prompt_len + st["tok_b"]
        if b > prompt_len + len(case["chain_ids"]) or b <= a or a <= 0:
            continue
        mk = torch.tensor(IV.unb64u8(st["mask_q"], (gh, gw)).astype(np.float32),
                          device=device).reshape(-1)
        if float(mk.sum()) == 0 or float(mk.sum()) == mk.numel():
            continue                       # degenerate union: no in/out contrast
        steps.append({"rows": torch.arange(a, b, device=device)})
        boxes.append(mk)
        rolls.append(rolled_mask(mk, gh, gw, rng).to(device))
    return steps, boxes, rolls


@torch.no_grad()
def score_case(model, processor, fi, case, image, device, cutoff, alpha, kind, seed):
    """One teacher-forced forward with the flow intervention applied."""
    text = PROBE.build_prompt(processor, case["question"])
    inputs = processor(text=[text], images=[[image]], return_tensors="pt",
                       padding=True, padding_side="left",
                       add_special_tokens=False).to(device)
    prompt_len = inputs["input_ids"].shape[1]
    chain, gold = case["chain_ids"], case["gold_ids"]
    ids = torch.tensor([inputs["input_ids"][0].tolist() + chain + gold], device=device)
    seq = ids.shape[1]
    img_cols = (inputs["input_ids"][0] == IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0]
    gh, gw = case["grid"]
    if img_cols.numel() != gh * gw:
        return None                     # grid and image tokens disagree: skip, not guess

    steps, boxes, rolls = case_steps(case, prompt_len, gh, gw, device, seed)
    if not steps:
        return None

    X0 = init_columns(seq, img_cols, boxes, rolls, device)
    fi.arm({"steps": steps, "alpha": float(alpha), "kind": kind, "cutoff": int(cutoff)},
           X0, IV.causal_mask(seq, fi.dtype, device))
    fwd = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    if "pixel_values" in inputs:
        fwd["pixel_values"] = inputs["pixel_values"]
        fwd["image_grid_thw"] = inputs["image_grid_thw"]
    if inputs.get("mm_token_type_ids") is not None:
        pad = torch.zeros(1, seq - prompt_len, dtype=torch.long, device=device)
        fwd["mm_token_type_ids"] = torch.cat([inputs["mm_token_type_ids"], pad], dim=1)
    try:
        out = model(**fwd)
        share = manipulation_check(fi.X, steps)
    finally:
        fi.disarm()

    return IV.answer_readout(out.logits[0].float(), prompt_len + len(chain), case,
                             extra={"n_rows": fi.n_rows, "audit": fi.audit_stats,
                                    **share})


def manipulation_check(X, steps):
    """u/m at the step's own positions after the last layer, averaged over steps.

    This is the quantity the edit actuates. Reported on every forward because the
    actuator and the measurement are no longer the same object: without it a null
    cannot be told apart from never having moved anything.
    """
    m = X[:, 0].clamp_min(1e-12)
    us, rs, um, mm = [], [], [], []
    for i, st in enumerate(steps):
        r = st["rows"]
        us.append(float((X[r, 1 + 2 * i] / m[r]).mean()))
        rs.append(float((X[r, 2 + 2 * i] / m[r]).mean()))
        um.append(float(X[r, 1 + 2 * i].mean()))
        mm.append(float(X[r, 0].mean()))
    # The shares are the normalised claim, but concentrating attention on union-carriers
    # raises the numerator AND the denominator -- a flat share can hide a real move. The
    # raw masses separate the two.
    return {"ushare": float(np.mean(us)), "rshare": float(np.mean(rs)),
            "umass": float(np.mean(um)), "mmass": float(np.mean(mm))}


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------
def build_grid(args):
    """(cutoff, kind, alpha) units. alpha=0 is one baseline per cutoff, not per kind."""
    cuts = [int(c) for c in str(args.cutoffs).split(",") if c.strip() != ""]
    alphas = [float(a) for a in str(args.alphas).split(",") if a.strip() != ""]
    kinds = [k.strip() for k in str(args.conditions).split(",") if k.strip()]
    bad = [k for k in kinds if k not in CONDITIONS]
    if bad:
        raise SystemExit(f"unknown --conditions {bad}; choose from {list(CONDITIONS)}")
    grid = []
    for c in cuts:
        grid.append((c, "base", 0.0))
        for k in kinds:
            for al in alphas:
                if al != 0.0:
                    grid.append((c, k, al))
    return grid


def unit_key(row_index, cutoff, kind, alpha):
    return f"{row_index}|{cutoff}|{kind}|{alpha:.4f}"


def load_model(args, device):
    """Same loader the correlational probe uses, so the two see one model."""
    proc, model = PROBE.load_model(args.base_model, args.adapter or None, device,
                                   args.attn_impl)
    model.requires_grad_(False)
    return proc, model


def run_shard(args, device):
    out = Path(args.out_dir)
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "progress").mkdir(parents=True, exist_ok=True)
    dest = out / "results" / f"shard{args.shard:02d}.jsonl"

    cases, cfg, _fp = IV.load_cases(Path(args.cases_dir), args.shard, args.num_shards,
                                    args.max_cases)
    imgs = IV.load_case_images(cfg, f"_fi{args.shard}")
    grid = build_grid(args)
    units = [(c, g) for c in cases for g in grid]

    done = set()
    if dest.exists():
        with dest.open() as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue            # a torn line from a full filesystem
                done.add(unit_key(r["row_index"], r["cutoff"], r["kind"], r["alpha"]))
    todo = [(c, g) for c, g in units
            if unit_key(c["row_index"], g[0], g[1], g[2]) not in done]
    prog = IV.Progress(out / "progress" / f"run{args.shard:02d}.json", len(units),
                       f"run{args.shard}", args.log_every,
                       already_done=len(units) - len(todo))
    print(f"[run] shard {args.shard}: {len(cases)} cases x {len(grid)} units = "
          f"{len(units)}; {len(todo)} to do")

    proc, model = load_model(args, device)
    fi = FlowIntervener(model, next(model.parameters()).dtype)
    try:
        with dest.open("a") as fh:
            for case, (cut, kind, al) in todo:
                img = case_image(imgs, case["row_index"])
                if img is None:
                    prog.tick()
                    continue
                r = score_case(model, proc, fi, case, img, device, cut, al,
                               "box" if kind == "base" else kind,
                               seed=abs(hash(("fi", case["row_index"]))) % (2 ** 31))
                if r is not None:
                    r.update({"row_index": case["row_index"], "cutoff": cut,
                              "kind": kind, "alpha": al})
                    fh.write(json.dumps(r) + "\n")
                    fh.flush()
                prog.tick()
    finally:
        fi.close()
        prog.close()
    print(f"[run] shard {args.shard} done -> {dest}")


def selftest(args, device):
    """Four gates, three of them AGGREGATE.

    The first version gated per case on |logp(alpha=0, hooked) - logp(un-hooked)|, which
    was wrong twice over.

    It compared a forward whose step rows are rebuilt in eager at all 36 layers against
    one running sdpa throughout, so it measured 36 layers of accumulated eager-vs-sdpa
    drift and called it a rebuild bug. The per-layer audit is what isolates the rebuild:
    it compares the rebuilt rows against that same module's OWN eager output, before
    anything propagates. That is gate 1 now, and `--attn-impl eager` (the default here,
    unlike the correlational probe) removes the drift at the source.

    And it gated per case on the edit's effect size. How much purchase the edit has
    depends on the step's union -- a case whose eligible keys already carry uniform union
    content has nothing to re-allocate, and that is a fact about the case, not a broken
    harness. Gates 3 and 4 are therefore aggregate, with the per-case spread printed
    because it is the first real evidence about how much the indirect path can be moved
    at all.
    """
    cases, cfg, _fp = IV.load_cases(Path(args.cases_dir), 0, 1, args.max_cases or 8)
    imgs = IV.load_case_images(cfg, "_fist")
    cases = [c for c in cases if c["row_index"] in imgs][:8]
    if not cases:
        raise SystemExit("selftest needs at least one case with an image")
    proc, model = load_model(args, device)
    fi = FlowIntervener(model, next(model.parameters()).dtype)
    cut = max(fi.layers)
    rels, drift, edit, dushare, det = [], [], [], [], []
    try:
        print(f"deepstack re-seed layers: {sorted(fi.deep)}   "
              f"attention layers: {len(fi.layers)}   attn_impl {args.attn_impl}")
        fi.audit = True
        print(f"  {'row':>6} {'rebuild rel':>12} {'L':>3} {'drift(a=0)':>11} "
              f"{'repeat':>8} {'edit(a1-a0)':>12} {'d ushare':>9} {'u %':>8} {'m %':>8}")
        for case in cases:
            img = case_image(imgs, case["row_index"])
            ref = IV.score_case_nohook(model, proc, case, img, device)
            a0 = score_case(model, proc, fi, case, img, device, cut, 0.0, "box", 0)
            a0b = score_case(model, proc, fi, case, img, device, cut, 0.0, "box", 0)
            a1 = score_case(model, proc, fi, case, img, device, cut, 1.0, "box", 0)
            if None in (ref, a0, a0b, a1):
                continue
            au = a0.get("audit") or {}
            rels.append(au.get("rel", float("nan")))
            drift.append(abs(a0["logp_gold"] - ref["logp_gold"]))
            det.append(abs(a0["logp_gold"] - a0b["logp_gold"]))
            edit.append(abs(a1["logp_gold"] - a0["logp_gold"]))
            dushare.append(a1["ushare"] - a0["ushare"])
            print(f"  {case['row_index']:>6} {rels[-1]:>12.2e} "
                  f"{au.get('layer', -1):>3} {drift[-1]:>11.5f} {det[-1]:>8.1e} "
                  f"{edit[-1]:>12.5f} {dushare[-1]:>+9.4f} "
                  f"{100.0 * (a1['umass'] - a0['umass']) / max(a0['umass'], 1e-12):>+8.2f}"
                  f"{100.0 * (a1['mmass'] - a0['mmass']) / max(a0['mmass'], 1e-12):>+8.2f}")
    finally:
        fi.close()
    if not rels:
        raise SystemExit("selftest scored no cases")

    g1 = float(np.nanmax(rels)) < 0.05
    g2 = float(np.max(det)) < 1e-6
    g3 = float(np.mean(edit)) > 1e-3 and sum(e > 1e-4 for e in edit) >= len(edit) / 2
    g4 = float(np.mean(dushare)) > 0 and sum(d > 0 for d in dushare) >= len(dushare) / 2
    print(f"\nGATES over {len(rels)} cases")
    print(f"  1 {'ok  ' if g1 else 'FAIL'} the per-layer rebuild matches the module's "
          f"own eager output: worst rel {np.nanmax(rels):.2e} < 5e-2")
    print(f"  2 {'ok  ' if g2 else 'FAIL'} an alpha=0 repeat is deterministic: "
          f"worst {np.max(det):.1e} < 1e-6")
    print(f"  3 {'ok  ' if g3 else 'FAIL'} the edit reaches the answer: mean |d logp| "
          f"{np.mean(edit):.5f}, {sum(e > 1e-4 for e in edit)}/{len(edit)} cases move")
    print(f"  4 {'ok  ' if g4 else 'FAIL'} the manipulation lands: mean d ushare "
          f"{np.mean(dushare):+.4f}, {sum(d > 0 for d in dushare)}/{len(dushare)} up")
    print(f"\n  FYI eager-vs-un-hooked drift at alpha=0: mean {np.mean(drift):.5f}, "
          f"max {np.max(drift):.5f}. Common to baseline, box and roll, so it cancels in\n"
          f"  box - roll -- but it is per-case noise, so it sets the variance floor. "
          f"Large values here mean the run needs its n.")
    ok = g1 and g2 and g3 and g4
    print("\nSELFTEST PASS" if ok else "\nSELFTEST FAIL")
    if not ok:
        raise SystemExit(1)


def report(args):
    out = Path(args.out_dir)
    files = sorted((out / "results").glob("shard*.jsonl"))
    if not files:
        raise SystemExit(f"no results under {out / 'results'}")
    rows = []
    for f in files:
        with f.open() as fh:
            for line in fh:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    if not rows:
        raise SystemExit("no parseable results")

    key = {}
    for r in rows:
        key[(r["row_index"], r["cutoff"], r["kind"], round(r["alpha"], 4))] = r
    cuts = sorted({r["cutoff"] for r in rows})
    alphas = sorted({round(r["alpha"], 4) for r in rows if r["alpha"] != 0})
    print(f"{len(rows)} records   {len({r['row_index'] for r in rows})} cases   "
          f"cutoffs {cuts}   alphas {alphas}")
    print("\nbox - roll on log P(gold), paired per case. The manipulation check is the "
          "right-hand pair:\nbox must raise ushare and roll must raise rshare, or the "
          "logp column is not evidence about grounding.\n")
    print(f"   {'cut':>4} {'alpha':>6} {'n':>5} {'box-roll':>10} {'95% CI':>20} "
          f"{'d ushare(box)':>14} {'d rshare(roll)':>15} {'u %':>8} {'m %':>8}")
    rng = np.random.default_rng(0)
    for c in cuts:
        for al in alphas:
            d, du, dr, ru, rm = [], [], [], [], []
            for ri in {r["row_index"] for r in rows}:
                b = key.get((ri, c, "box", al))
                q = key.get((ri, c, "roll", al))
                z = key.get((ri, c, "base", 0.0))
                if not (b and q and z):
                    continue
                d.append(b["logp_gold"] - q["logp_gold"])
                # The share is a ratio, so it can move because the numerator rose or
                # because the denominator fell -- opposite claims. These are the two
                # relative moves behind it. Absolute masses are ~1e-2 after 36 layers of
                # dilution, so they are reported as percentages, not raw deltas.
                if z.get("umass") and z.get("mmass"):
                    ru.append(100.0 * (b["umass"] - z["umass"]) / abs(z["umass"]))
                    rm.append(100.0 * (b["mmass"] - z["mmass"]) / abs(z["mmass"]))
                du.append(b["ushare"] - z["ushare"])
                dr.append(q["rshare"] - z["rshare"])
            if len(d) < 8:
                continue
            arr = np.array(d)
            bs = np.array([rng.choice(arr, arr.size, replace=True).mean()
                           for _ in range(2000)])
            lo, hi = np.percentile(bs, [2.5, 97.5])
            print(f"   {c:>4} {al:>6.2f} {len(d):>5} {arr.mean():>+10.5f} "
                  f"[{lo:>+8.5f},{hi:>+8.5f}] {np.mean(du):>+14.4f} "
                  f"{np.mean(dr):>+15.4f} "
                  f"{(np.mean(ru) if ru else float('nan')):>+8.2f} "
                  f"{(np.mean(rm) if rm else float('nan')):>+8.2f}")
    print("\nA cell whose manipulation columns are ~0 is not a null about grounding; it "
          "is a failed intervention.")
    print("`u %` / `m %` are the box condition's relative change in union-traceable and "
          "image-traceable mass at the step's own positions. They decompose d ushare: a "
          "share that rises\nbecause `u %` rose is the model reading MORE from the boxes; "
          "one that rises because `m %` fell is it reading less from everywhere else. "
          "Those are different claims and only\nthe first is the one the reward was "
          "trying to buy.")


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", required=True,
                   choices=["selftest", "run", "report", "monitor"])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--cases-dir", default="")
    p.add_argument("--base-model", default=str(repo_path(
        "checkpoint/coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged")))
    p.add_argument("--adapter", default="")
    # eager, unlike the correlational probe. The hook re-runs every layer in eager
    # anyway, so an sdpa model pays for a pass whose step rows are then discarded AND
    # leaves the rebuilt rows differing from the ones the rest of the model produced --
    # 36 layers of that drift reached 0.18 nats per case, which is pure variance in the
    # paired readout.
    p.add_argument("--attn-impl", default="eager")
    p.add_argument("--cutoffs", default="8,16,24,35")
    p.add_argument("--alphas", default="0.25,0.5,1.0")
    p.add_argument("--conditions", default="box,roll")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--max-cases", type=int, default=0)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--interval", type=float, default=30.0)
    p.add_argument("--once", action="store_true")
    args = p.parse_args()

    if args.stage == "report":
        return report(args)
    if args.stage == "monitor":
        return IV.monitor(Path(args.out_dir), args.interval, args.once, "run")
    if not args.cases_dir:
        raise SystemExit("--cases-dir is required (an intervene_probe prepare out-dir)")
    device = torch.device(args.device)
    if args.stage == "selftest":
        return selftest(args, device)
    return run_shard(args, device)


if __name__ == "__main__":
    main()
