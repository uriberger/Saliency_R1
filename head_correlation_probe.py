#!/usr/bin/env python
"""Per-head correlation scan: which of the 1152 heads' step-level overlap predicts
whether the completion was right?

The intervention probe asked a different question at the wrong granularity. Forcing
all 32 heads of a layer at once is NOT an upper bound on what one head does -- heads
can carry opposing contributions that cancel, and o_proj mixes them -- so its
layer-level null says nothing about individual heads. This scans every head directly,
on the property the project cares about: does this head's attention to the objects a
step names predict getting the answer right?

For every observe step of every case, and for EVERY (layer, head), the step's map is
that head's attention over image patches, mean-reduced over the step's tokens (the
trainer's token_reduction=mean), scored against the step's own per-step DINO union:

    mean_in_v2 = mean inside the union / mean over the whole map      (chance = 1.0)
    auroc      = P(in-box patch outranks out-box patch)               (chance = 0.5)

Two correlation set-ups:

  step        one observation per observe step; the label is its COMPLETION's
              correctness, repeated across that completion's steps.
  completion  the completion's steps are averaged into one overlap value; one
              observation per completion.

Steps within a completion share a label and are not independent, so `step`-level
significance is anti-conservative. CIs are therefore bootstrapped over COMPLETIONS in
both set-ups, which is the fix.

Correctness is the trainer's own `accuracy_reward` on the model's own greedy answer,
recovered by decoding the continuation of its own chain -- not a first-token match,
which capitalisation biases to 0.38 against a true 0.55.

Selecting a winner from 1152 candidates is where a ranking becomes an artefact, so
`report` splits cases by row_index parity: heads are ranked on the odd half and
re-scored on the even. A head that survives that is a candidate; one that does not is
selection noise.

    bash launch_head_correlation.sh --gpus 8 --out-dir DIR --cases-dir <probe out-dir>
    python head_correlation_probe.py --stage report --out-dir DIR
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


PROBE = _load_module("_hc_overlap_probe", "overlap_probe.py")
IV = _load_module("_hc_intervene", "intervene_probe.py")
IMAGE_TOKEN_ID = PROBE.IMAGE_TOKEN_ID


# ---------------------------------------------------------------------------
# all-layer, all-head attention capture
# ---------------------------------------------------------------------------
class AllHeadCapture:
    """Hooks every attention layer; keeps only [heads, step_rows, image_patches].

    Each hook re-runs its own module in eager to recover the softmax weights that
    flash/sdpa discard -- the trainer's single-layer trick, installed on all 36. The
    transient [1, H, S, S] is sliced immediately and dropped, so peak memory is one
    layer's worth rather than 36.
    """

    def __init__(self, model):
        self.mask = None
        self.rows = None
        self.cols = None
        self.out = {}
        self._reentry = False
        self.handles = []
        self.layers = []
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

    def _make(self, layer_idx):
        def hook(module, args, kwargs, output):
            if self._reentry or self.rows is None:
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
            sl = attn[0][:, self.rows][:, :, self.cols]     # [H, n_rows, n_patches]
            self.out[layer_idx] = torch.relu(sl).float().cpu().numpy()
            del attn, sl
            return None
        return hook


@torch.no_grad()
def scan_case(model, processor, cap, case, image, device, answer_max_tokens):
    """-> (maps [L,H,n_steps,n_patches], model's own answer, kept step indices)."""
    text = PROBE.build_prompt(processor, case["question"])
    inputs = processor(text=[text], images=[[image]], return_tensors="pt",
                       padding=True, padding_side="left", add_special_tokens=False).to(device)
    prompt_len = inputs["input_ids"].shape[1]
    chain = case["chain_ids"]
    gh, gw = case["grid"]
    cols = (inputs["input_ids"][0] == IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0]
    if cols.numel() != gh * gw:
        return None                       # grid and image tokens disagree: skip, not guess

    spans, kept = [], []
    for si, st in enumerate(case["steps"]):
        a, b = prompt_len + st["tok_a"], prompt_len + st["tok_b"]
        if b > prompt_len + len(chain) or b <= a:
            continue
        spans.append((a, b))
        kept.append(si)
    if not spans:
        return None
    rows = torch.cat([torch.arange(a, b, device=device) for a, b in spans])

    ids = torch.tensor([inputs["input_ids"][0].tolist() + chain], device=device)
    seq = ids.shape[1]
    cap.mask = IV.causal_mask(seq, next(model.parameters()).dtype, device)
    cap.rows, cap.cols, cap.out = rows, cols, {}
    fwd = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    if "pixel_values" in inputs:
        fwd["pixel_values"] = inputs["pixel_values"]
        fwd["image_grid_thw"] = inputs["image_grid_thw"]
    if inputs.get("mm_token_type_ids") is not None:
        pad = torch.zeros(1, seq - prompt_len, dtype=torch.long, device=device)
        fwd["mm_token_type_ids"] = torch.cat([inputs["mm_token_type_ids"], pad], dim=1)
    out = model(**fwd, use_cache=True)
    cap.rows = None                                # disarm before the decode

    # The model's own answer: a greedy continuation of its own chain, so this
    # reproduces what it generated at prepare time. Graded by the trainer's
    # accuracy_reward rather than by a first-token match.
    past, nxt = out.past_key_values, out.logits[0, -1].argmax().view(1, 1)
    got = [int(nxt)]
    eos = processor.tokenizer.eos_token_id
    for _ in range(answer_max_tokens - 1):
        if got[-1] == eos:
            break
        o = model(input_ids=nxt, past_key_values=past, use_cache=True)
        past, nxt = o.past_key_values, o.logits[0, -1].argmax().view(1, 1)
        got.append(int(nxt))
    answer = processor.tokenizer.decode(got, skip_special_tokens=True)

    lens = [b - a for a, b in spans]
    H = cap.out[cap.layers[0]].shape[0]
    maps = np.zeros((len(cap.layers), H, len(spans), gh * gw), dtype=np.float32)
    for li, L in enumerate(cap.layers):
        arr = cap.out[L]
        o = 0
        for si, n in enumerate(lens):
            maps[li, :, si] = arr[:, o:o + n].mean(axis=1)   # token_reduction=mean
            o += n
    cap.out = {}
    return maps, answer, kept


# ---------------------------------------------------------------------------
# metrics, vectorised over (layer, head, step)
# ---------------------------------------------------------------------------
def metrics(maps, masks):
    """maps [L,H,S,P], masks [S,P] bool -> (mean_in_v2, auroc), each [L,H,S].

    Average ranks for ties: attention maps have many near-identical near-zero
    patches, and argsort would break those arbitrarily and bias auroc.
    """
    from scipy.stats import rankdata

    Lc, Hc, Sc, P = maps.shape
    flat = maps.reshape(Lc * Hc * Sc, P).astype(np.float64)
    ranks = rankdata(flat, axis=-1)
    v2 = np.full(Lc * Hc * Sc, np.nan)
    au = np.full(Lc * Hc * Sc, np.nan)
    for si in range(Sc):
        m = masks[si]
        k = int(m.sum())
        if k == 0 or k == P:
            continue                   # degenerate union: no in/out contrast to score
        pos = np.arange(Lc * Hc) * Sc + si
        sub, rk = flat[pos], ranks[pos]
        mean_all = sub.mean(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            v2[pos] = np.where(mean_all > 0, sub[:, m].mean(axis=1) / mean_all, np.nan)
        au[pos] = (rk[:, m].sum(axis=1) - k * (k + 1) / 2.0) / (k * (P - k))
    return v2.reshape(Lc, Hc, Sc), au.reshape(Lc, Hc, Sc)


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
    imgs = IV.load_case_images(cfg, f"_hc{args.shard}")
    missing = [c["row_index"] for c in cases if c["row_index"] not in imgs]
    if missing:
        raise SystemExit(f"{len(missing)}/{len(cases)} cases have no image "
                         f"(e.g. row {missing[0]}); cases were prepared with {cfg}")
    dest.parent.mkdir(parents=True, exist_ok=True)

    processor, model = PROBE.load_model(args.base_model, args.adapter or None, device,
                                        args.attn_impl)
    cap = AllHeadCapture(model)
    print(f"[scan] shard {args.shard}: {len(cases)} cases x {len(cap.layers)} layers",
          flush=True)
    prog = IV.Progress(out / "progress" / f"scan{args.shard:02d}.json", len(cases),
                       f"scan{args.shard}", args.log_every)

    V2, AU, ROW, STEP, COR, UNI = [], [], [], [], [], []
    dropped = 0
    try:
        for case in cases:
            try:
                r = scan_case(model, processor, cap, case,
                              imgs[case["row_index"]]["image"], device,
                              args.answer_max_tokens)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"[scan] OOM on row {case['row_index']}; skipped", flush=True)
                r = None
            if r is None:
                dropped += 1
                prog.tick()
                continue
            maps, answer, kept = r
            gh, gw = case["grid"]
            steps = [case["steps"][i] for i in kept]
            masks = np.stack([IV.unb64u8(st["mask_q"], (gh, gw)).astype(bool).reshape(-1)
                              for st in steps])
            v2, au = metrics(maps, masks)
            grade = PROBE.accuracy_reward(
                [[{"role": "assistant", "content": f"</think> {answer}"}]],
                [case["gold"]])[0]
            if grade is None:                    # ungradable answer: not "wrong"
                dropped += 1
                prog.tick()
                continue
            for si, st in enumerate(steps):
                V2.append(v2[:, :, si])
                AU.append(au[:, :, si])
                ROW.append(case["row_index"])
                STEP.append(si)
                COR.append(float(grade))
                UNI.append(st["union_frac"])
            prog.tick()
    finally:
        cap.close()
        prog.close()
    if not V2:
        raise SystemExit("no steps scored")
    np.savez_compressed(
        dest,
        v2=np.stack(V2).astype(np.float32), auroc=np.stack(AU).astype(np.float32),
        row=np.array(ROW), step=np.array(STEP),
        correct=np.array(COR, dtype=np.float32), union=np.array(UNI, dtype=np.float32),
        layers=np.array(cap.layers))
    print(f"[scan] shard {args.shard}: {len(V2)} steps from "
          f"{len(set(ROW))} completions, {dropped} cases dropped -> {dest}")


# ---------------------------------------------------------------------------
# stage: report
# ---------------------------------------------------------------------------
def col_corr(X, y):
    """Pearson r of every column of X [N, L, H] against y [N], NaN-aware. -> [L, H]."""
    ok = np.isfinite(X) & np.isfinite(y)[:, None, None]
    n = ok.sum(0).astype(np.float64)
    Xs = np.where(ok, X, 0.0).astype(np.float64)
    ys = np.where(ok, y[:, None, None], 0.0).astype(np.float64)
    sx, sy = Xs.sum(0), ys.sum(0)
    sxx, syy, sxy = (Xs * Xs).sum(0), (ys * ys).sum(0), (Xs * ys).sum(0)
    num = n * sxy - sx * sy
    den = np.sqrt(np.maximum(n * sxx - sx ** 2, 0)) * np.sqrt(np.maximum(n * syy - sy ** 2, 0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where((den > 0) & (n >= 8), num / den, np.nan)
    return r


def report(args):
    out = Path(args.out_dir)
    files = sorted((out / "scan").glob("shard*.npz"))
    if not files:
        raise SystemExit(f"no scan output under {out / 'scan'}")
    d = [np.load(f) for f in files]
    v2 = np.concatenate([x["v2"] for x in d])
    au = np.concatenate([x["auroc"] for x in d])
    row = np.concatenate([x["row"] for x in d])
    cor = np.concatenate([x["correct"] for x in d])
    layers = d[0]["layers"]
    N, Lc, Hc = v2.shape
    uniq = np.unique(row)
    print(f"steps {N}   completions {len(uniq)}   layers {Lc}   heads {Hc}   "
          f"accuracy {cor[np.unique(row, return_index=True)[1]].mean():.3f}")

    # completion-level: mean of the completion's steps, one observation per completion
    idx = np.searchsorted(uniq, row)
    ccor = np.zeros(len(uniq))
    np.maximum.at(ccor, idx, cor)          # label is constant within a completion
    cagg = {}
    for name, arr in (("mean_in_v2", v2), ("auroc", au)):
        s = np.zeros((len(uniq), Lc, Hc))
        n = np.zeros((len(uniq), Lc, Hc))
        np.add.at(s, idx, np.nan_to_num(arr, nan=0.0))
        np.add.at(n, idx, np.isfinite(arr).astype(float))
        with np.errstate(invalid="ignore", divide="ignore"):
            cagg[name] = np.where(n > 0, s / n, np.nan)

    sel_c = (uniq % 2 == 1)                # select on odd rows, confirm on even
    sel_s = (row % 2 == 1)
    for name, sarr, carr in (("mean_in_v2", v2, cagg["mean_in_v2"]),
                             ("auroc", au, cagg["auroc"])):
        for setup, X, y, sel in (("step", sarr, cor, sel_s),
                                 ("completion", carr, ccor, sel_c)):
            r_all = col_corr(X, y)
            r_sel = col_corr(X[sel], y[sel])
            r_out = col_corr(X[~sel], y[~sel])
            print(f"\n=== {name} / {setup}-level "
                  f"(n={len(y)}) ===")
            print("  per-LAYER (max |r| over its 32 heads, all data) -- pick layers here:")
            order = np.argsort(-np.nan_to_num(np.nanmax(np.abs(r_all), axis=1)))
            print(f"   {'rank':>4} {'layer':>5} {'max|r|':>8} {'head':>5} {'mean|r|':>8}")
            for k, li in enumerate(order[: args.top_layers]):
                h = int(np.nanargmax(np.abs(r_all[li])))
                print(f"   {k + 1:>4} {int(layers[li]):>5} "
                      f"{np.nanmax(np.abs(r_all[li])):>8.4f} {h:>5} "
                      f"{np.nanmean(np.abs(r_all[li])):>8.4f}")
            print("  TOP HEADS ranked on ODD rows, re-scored on EVEN (held out):")
            flat = np.abs(np.nan_to_num(r_sel, nan=0.0)).ravel()
            print(f"   {'layer':>5} {'head':>5} {'r(select)':>10} {'r(HELD OUT)':>12} "
                  f"{'r(all)':>8}")
            for t in np.argsort(-flat)[: args.top_heads]:
                l, h = divmod(int(t), Hc)
                print(f"   {int(layers[l]):>5} {h:>5} {r_sel[l, h]:>+10.4f} "
                      f"{r_out[l, h]:>+12.4f} {r_all[l, h]:>+8.4f}")
            for h22 in (28, 31):
                li = int(np.where(layers == 22)[0][0])
                rank = int((np.abs(np.nan_to_num(r_all)) > abs(r_all[li, h22])).sum()) + 1
                print(f"   incumbent L22H{h22}: r(all) {r_all[li, h22]:>+.4f}  "
                      f"rank {rank} of {Lc * Hc}")
            np.savez_compressed(out / f"corr_{name}_{setup}.npz",
                                r_all=r_all, r_sel=r_sel, r_out=r_out, layers=layers)
    print(f"\n-> {out}/corr_*.npz")


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", default="scan", choices=["scan", "report"])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--cases-dir", default="",
                   help="an intervene_probe out-dir whose cases/ holds the chains and "
                        "per-step DINO unions; defaults to --out-dir")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--max-cases", type=int, default=0)
    p.add_argument("--base-model", default=str(repo_path(
        "checkpoint/coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged")))
    p.add_argument("--adapter", default="")
    p.add_argument("--attn-impl", default="sdpa")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--answer-max-tokens", type=int, default=16)
    p.add_argument("--top-layers", type=int, default=10)
    p.add_argument("--top-heads", type=int, default=15)
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
