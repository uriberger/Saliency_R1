#!/usr/bin/env python
"""E1 -- the causal arm: insert a gap in the position ids and see what it costs.

E0 showed, observationally, that the positional overlay on the image marches one
patch per generated token.  E1 asks whether that matters, by intervening on
`position_ids` and nothing else.  The tokens, the pixels, the weights and the
causal mask are all byte-identical between arms; only the integers fed to M-RoPE
change.  So any difference is attributable to position and to nothing else.

THE ARMS
--------
Write a text token's M-RoPE position as (t, h, w) = (p, p, p) and a patch's as
(s, s+r, s+c).  "Tail" means every token after the image -- the question and the
completion.  The system prompt and the image never move except in `null`.

  null   +delta to EVERY token, image included.  RoPE depends only on differences,
         so the logits must come back bit-identical.  This is not a formality: it
         is the only check that nothing else in the stack reads absolute position,
         and it validates the whole harness.  Reported as a max abs logit delta.
  full   +delta to all three axes of the tail.  The honest "the query is further
         from the image" condition, and what actually happens as a CoT lengthens.
  t      +delta to the tail's t axis only.  Every patch shares the image's t index,
         so this offset is CONSTANT across patches: it can move image-vs-text mass
         but cannot change the shape of the profile over patches.  This is the
         visual-fading arm.
  hw     +delta to the tail's h and w axes only.  Tail-to-tail offsets are
         unchanged (both query and key move together), image-to-image offsets are
         unchanged, so this touches EXACTLY the cross-modal spatial offsets and
         nothing else.  This is the hypothesis, surgically isolated.
  fix    tail h/w frozen at the value the first post-image token would have had,
         so the cross-modal h/w offset no longer depends on p at all -- the
         invariance the model does not have.  Its drift-vs-delta curve must be
         FLAT.  That is the positive control: it proves the drift comes from the
         h/w coupling rather than from perturbing position ids in general.
         (Roughly what Circle-RoPE and DIPE achieve by other means.)

WHAT IS PREDICTED, BEFORE RUNNING
---------------------------------
  * null: max |logit difference| at fp noise.
  * t: profile shape essentially unchanged (exactly so at layer 0, where hidden
    states cannot yet have drifted), image mass moves.
  * hw: profile centroid drifts with delta, and the drift is periodic -- the
    fastest H channel repeats every 8.0 positions and W every 10.2, so a monotone
    curve means fading and a ripple at those periods means this mechanism.  A
    monotone-decay story cannot produce a ripple with a pre-specified period.
  * fix: flat in delta.

Sweeping delta densely enough to resolve a period of 8 is the point; log-spaced
deltas would miss the entire signature.

READOUTS, KEPT SEPARATE
-----------------------
  nll        mean NLL of the model's own delta=0 completion, teacher-forced.  The
             behavioural cost: how far the intervention pushes the model off its
             own trajectory.
  mass       total attention on image tokens.  The fading channel.
  prof_r     correlation of the mean-centred log-attention profile over patches
             against the same case's delta=0 profile.  The shape channel.
  drift      centroid of the patch profile minus its delta=0 centroid, in patches.
             Under `hw` this is the drift itself, in the units of the claim.

Mass and shape are reported separately throughout, because conflating them is
exactly how this effect gets mistaken for visual fading.

  python rope_phase_e1.py --stage scan   --out-dir DIR --dataset PATH
  python rope_phase_e1.py --stage report --out-dir DIR
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rope_phase_probe as RP  # noqa: E402  (same directory, deliberate)

ARMS = ("null", "full", "t", "hw", "fix")


# ---------------------------------------------------------------------------
# the intervention
# ---------------------------------------------------------------------------
def build_position_ids(base: torch.Tensor, tail_start: int, arm: str, delta: int):
    """base [3, B, S] from get_rope_index -> the arm's position ids.

    `tail_start` is the first token after the image block.  Nothing before it
    moves, so the image keeps its own coordinate frame and only the text's
    relation to it changes -- except in `null`, where everything moves together
    and the output must not change at all.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    pos = base.clone()
    if arm == "null":
        pos += delta
        return pos
    if arm in ("full", "t"):
        rows = slice(0, 3) if arm == "full" else slice(0, 1)
        pos[rows, :, tail_start:] += delta
    elif arm == "hw":
        pos[1:3, :, tail_start:] += delta
    elif arm == "fix":
        pos[0:1, :, tail_start:] += delta
        # Freeze h and w at what the first post-image token carries, so the
        # cross-modal spatial offset stops depending on the text index.
        pos[1:3, :, tail_start:] = base[1:3, :, tail_start: tail_start + 1]
    return pos


def parse_deltas(spec: str) -> list[int]:
    """"0-16,24,32" -> [0..16, 24, 32].  Ranges are inclusive and step 1."""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part.lstrip("-"):
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(dict.fromkeys(out))


# ---------------------------------------------------------------------------
# capture
# ---------------------------------------------------------------------------
class LayerTap:
    """Eager replay on selected layers; returns [H, P] profile and [H] image mass.

    Averaged over the completion tokens, so one forward yields one profile per
    head per layer.
    """

    def __init__(self, model, layers):
        self.want = set(int(l) for l in layers)
        self.out = {}
        self.mask = self.img_cols = None
        self.tail_start = 0
        self._reentry = False
        self.handles = []
        found = []
        for m in model.modules():
            if type(m).__name__ in RP.TEXT_ATTENTION_CLASSES and getattr(m, "layer_idx", None) is not None:
                found.append(int(m.layer_idx))
                if int(m.layer_idx) in self.want:
                    self.handles.append(
                        m.register_forward_hook(self._make(int(m.layer_idx)), with_kwargs=True))
        missing = self.want - set(found)
        if missing:
            raise SystemExit(f"--layers asked for {sorted(missing)}, model has {min(found)}..{max(found)}")

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def arm(self, mask, img_cols, tail_start):
        self.mask, self.img_cols, self.tail_start, self.out = mask, img_cols, tail_start, {}

    def _make(self, layer_idx):
        def hook(module, args, kwargs, output):
            if self._reentry or self.mask is None:
                return None
            self._reentry = True
            kw = dict(kwargs)
            kw["attention_mask"] = self.mask
            kw["past_key_values"] = None
            kw["use_cache"] = False
            prev = module.config._attn_implementation
            module.config._attn_implementation = "eager"
            try:
                attn = module(*args, **kw)[1]
            finally:
                module.config._attn_implementation = prev
                self._reentry = False
            a = attn[0][:, self.tail_start:, :][:, :, self.img_cols].float()   # [H, T, P]
            mass = a.sum(-1).mean(1)                                          # [H]
            prof = a.clamp_min(1e-20).log()
            prof = (prof - prof.mean(-1, keepdim=True)).mean(1)               # [H, P]
            self.out[layer_idx] = (prof.cpu().numpy(), mass.cpu().numpy(),
                                   a.mean(1).cpu().numpy())
            del a, attn, prof
            return None
        return hook


def completion_nll(logits, ids, prompt_len: int) -> float:
    lg = logits[0, prompt_len - 1: -1].float()
    tgt = ids[0, prompt_len:]
    return float(torch.nn.functional.cross_entropy(lg, tgt).item())


def centroid(p: np.ndarray, gh: int, gw: int):
    """Row/col centre of mass of a patch-probability map [.., P]."""
    m = p.reshape(*p.shape[:-1], gh, gw)
    m = m / np.clip(m.sum(axis=(-2, -1), keepdims=True), 1e-30, None)
    r = (m.sum(-1) * np.arange(gh)).sum(-1)
    c = (m.sum(-2) * np.arange(gw)).sum(-1)
    return r, c


# ---------------------------------------------------------------------------
@torch.no_grad()
def scan(args, device):
    out_dir = Path(args.out_dir)
    shard_path = out_dir / "scan" / f"shard{args.shard:02d}.npz"
    if shard_path.exists() and not args.overwrite:
        print(f"[skip] {shard_path} exists")
        return
    shard_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.dataset:
        raise SystemExit("--dataset is required (or set ROPE_PHASE_DATASET)")
    rows = RP.load_samples(args.dataset, args.n_samples, args.seed)
    rows = rows[args.shard:: args.num_shards]
    if args.max_cases:
        rows = rows[: args.max_cases]

    deltas = parse_deltas(args.deltas)
    arms = [a for a in args.arms.split(",") if a]
    layers = [int(x) for x in args.layers.split(",") if x != ""]
    print(f"[e1] shard {args.shard}/{args.num_shards}: {len(rows)} cases, "
          f"{len(arms)} arms x {len(deltas)} deltas = {len(arms) * len(deltas)} forwards/case, "
          f"layers {layers}", flush=True)

    processor, model = RP.load_model(args.base_model, args.adapter or None, device)
    tcfg = model.config.text_config
    th, tw = RP.fastest_theta(tcfg, "h"), RP.fastest_theta(tcfg, "w")
    print(f"[e1] H period {RP.TWO_PI / th:.2f} positions, W period {RP.TWO_PI / tw:.2f} "
          f"-- a ripple at those periods is this mechanism, a monotone curve is fading",
          flush=True)
    image_token_id = int(getattr(model.config, "image_token_id", None) or RP.IMAGE_TOKEN_ID)
    torch.manual_seed(args.seed + args.shard)

    A, D, L = len(arms), len(deltas), len(layers)
    n_heads = tcfg.num_attention_heads
    acc = {k: np.zeros((A, D, L, n_heads), dtype=np.float64)
           for k in ("mass", "prof_r", "drift_r", "drift_c")}
    acc["nll"] = np.zeros((A, D), dtype=np.float64)
    null_max = 0.0
    gh = gw = None
    tap = None
    ncase = 0

    for ci, row in enumerate(rows):
        image = RP.square_image(row["image"], args.image_side)
        try:
            inputs, prompt_len, comp_ids = RP.generate(
                processor, model, image, row["question"],
                args.max_new_tokens, args.temperature, device)
        except Exception as exc:
            print(f"[warn] case {ci} generate failed: {exc}", flush=True)
            continue
        if len(comp_ids) < args.min_tokens:
            continue

        grid = inputs["image_grid_thw"][0].tolist()
        merge = int(model.config.vision_config.spatial_merge_size)
        g_h, g_w = grid[1] // merge, grid[2] // merge
        if gh is None:
            gh, gw = g_h, g_w
            tap = LayerTap(model, layers)
        elif (g_h, g_w) != (gh, gw):
            print(f"[warn] case {ci} grid {(g_h, g_w)} != {(gh, gw)}; skipped", flush=True)
            continue

        case = RP.teacher_forced_case(inputs, comp_ids, device)
        ids = case["input_ids"]
        seq = ids.shape[-1]
        img_cols = (ids[0] == image_token_id).nonzero(as_tuple=True)[0]
        if img_cols.numel() != gh * gw:
            continue
        tail_start = int(img_cols[-1].item()) + 1

        base, _ = (model.model if hasattr(model, "model") else model).get_rope_index(
            ids, case["mm_token_type_ids"], image_grid_thw=case.get("image_grid_thw"),
            video_grid_thw=None, attention_mask=case.get("attention_mask"))

        mdtype = next(model.parameters()).dtype
        add = torch.zeros(seq, seq, dtype=mdtype, device=device)
        add.masked_fill_(torch.triu(torch.ones(seq, seq, dtype=torch.bool, device=device),
                                    diagonal=1), torch.finfo(mdtype).min)
        mask = add[None, None]

        ref_logits = None
        ref = {}                                   # (layer) -> delta=0 profile/probs
        for ai, arm in enumerate(arms):
            for di, delta in enumerate(deltas):
                pos = build_position_ids(base, tail_start, arm, delta)
                tap.arm(mask, img_cols, tail_start)
                out = model(**case, position_ids=pos)
                tap.mask = None
                logits = out.logits
                acc["nll"][ai, di] += completion_nll(logits, ids, prompt_len)
                if arm == "null":
                    if delta == 0:
                        ref_logits = logits[0, prompt_len - 1:].float().clone()
                    else:
                        null_max = max(null_max, float(
                            (logits[0, prompt_len - 1:].float() - ref_logits).abs().max().item()))
                for li, lay in enumerate(layers):
                    prof, mass, probs = tap.out[lay]
                    acc["mass"][ai, di, li] += mass
                    if delta == deltas[0] and arm == arms[0]:
                        ref[lay] = (prof.copy(), probs.copy())
                    p0, q0 = ref[lay]
                    a_ = prof - prof.mean(-1, keepdims=True)
                    b_ = p0 - p0.mean(-1, keepdims=True)
                    den = np.sqrt((a_ * a_).sum(-1) * (b_ * b_).sum(-1))
                    acc["prof_r"][ai, di, li] += np.divide(
                        (a_ * b_).sum(-1), den, out=np.zeros_like(den), where=den > 0)
                    r1, c1 = centroid(probs, gh, gw)
                    r0, c0 = centroid(q0, gh, gw)
                    acc["drift_r"][ai, di, li] += r1 - r0
                    acc["drift_c"][ai, di, li] += c1 - c0
                del out, logits
        del add, mask
        ncase += 1
        if ncase % args.log_every == 0:
            print(f"[e1] {ncase} cases done", flush=True)

    if tap is not None:
        tap.close()
    if ncase == 0:
        raise SystemExit("no usable cases in this shard")

    meta = {"arms": arms, "deltas": deltas, "layers": layers, "ncase": ncase,
            "gh": gh, "gw": gw, "n_heads": n_heads, "null_max_logit_delta": null_max,
            "h_period": RP.TWO_PI / th, "w_period": RP.TWO_PI / tw,
            "shard": args.shard, "num_shards": args.num_shards,
            "n_samples": args.n_samples, "seed": args.seed, "image_side": args.image_side,
            "base_model": args.base_model, "adapter": args.adapter or "",
            "dataset": args.dataset, "max_new_tokens": args.max_new_tokens}
    np.savez(shard_path, __meta__=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8),
             **{k: v for k, v in acc.items()})
    print(f"[e1] wrote {shard_path}  cases={ncase}  null_max_logit_delta={null_max:.3e}",
          flush=True)


# ---------------------------------------------------------------------------
def report(args):
    out_dir = Path(args.out_dir)
    shards = sorted((out_dir / "scan").glob("shard*.npz"))
    if not shards:
        raise SystemExit(f"no shards in {out_dir}/scan")
    acc, meta, ncase, null_max = None, None, 0, 0.0
    for p in shards:
        z = np.load(p, allow_pickle=False)
        m = json.loads(bytes(z["__meta__"]).decode())
        if meta is None:
            meta = m
        elif m["deltas"] != meta["deltas"] or m["arms"] != meta["arms"]:
            raise SystemExit(f"{p.name} has a different sweep; do not merge these")
        ncase += m["ncase"]
        null_max = max(null_max, m["null_max_logit_delta"])
        cur = {k: z[k] for k in z.files if k != "__meta__"}
        acc = cur if acc is None else {k: acc[k] + cur[k] for k in acc}
    acc = {k: v / max(ncase, 1) for k, v in acc.items()}
    arms, deltas, layers = meta["arms"], meta["deltas"], meta["layers"]

    lines = []
    P = lines.append
    P("=" * 78)
    P("E1 -- position-id gap sweep (the causal arm)")
    P("=" * 78)
    P(f"model  : {meta['base_model']}")
    P(f"data   : {meta['dataset']}")
    P(f"scan   : {ncase} cases, {len(arms)} arms x {len(deltas)} deltas, layers {layers}")
    P(f"grid   : {meta['gh']} x {meta['gw']} patches")
    P(f"periods: H {meta['h_period']:.2f} positions, W {meta['w_period']:.2f}")
    P("")
    if "null" not in arms:
        P("NULL CHECK  NOT RUN -- the null arm was excluded from --arms, so nothing here")
        P("            verifies that the harness perturbs only what it claims to.")
    else:
        ok = null_max < args.null_tol
        P(f"NULL CHECK  max |logit delta| when EVERY position id shifts together: {null_max:.3e}")
        P(f"            {'PASS' if ok else 'FAIL'} (tolerance {args.null_tol:.0e}).  "
          + ("RoPE translation invariance holds and nothing else reads absolute position."
             if ok else
             "Something in the stack reads absolute position -- every number below is suspect."))
    P("")

    for name, label, unit in (("nll", "NLL of the model's own completion", ""),
                              ("mass", "attention mass on the image", ""),
                              ("prof_r", "profile correlation vs delta=0", ""),
                              ("drift_r", "profile centroid drift, rows", " patches"),
                              ("drift_c", "profile centroid drift, cols", " patches")):
        P(f"--- {label}{unit}")
        P("    delta : " + " ".join(f"{d:>6d}" for d in deltas))
        for ai, arm in enumerate(arms):
            v = acc[name][ai] if name == "nll" else acc[name][ai].mean(axis=(1, 2))
            P(f"    {arm:5s} : " + " ".join(f"{x:>6.3f}" for x in v))
        P("")

    P("READING IT")
    P("  t vs hw is the whole experiment.  The t axis offset is constant across")
    P("  patches, so `t` can move mass but must leave the profile shape alone; `hw`")
    P("  touches only the cross-modal spatial offsets.  If the damage is in `hw`,")
    P("  the drift is the mechanism.  If it is all in `t`, this is visual fading and")
    P("  the E0 march is epiphenomenal.")
    P("  `fix` must be flat: it removes the p-dependence of the h/w offset outright.")
    P(f"  A ripple in the `hw` curves at period {meta['h_period']:.1f} (rows) or "
      f"{meta['w_period']:.1f} (cols) is")
    P("  this mechanism and nothing else; a monotone curve is a decay story.")

    text = "\n".join(lines)
    print(text)
    (out_dir / "report.txt").write_text(text + "\n")
    print(f"\n[e1] wrote {out_dir / 'report.txt'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["scan", "report"], required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--base-model", default=RP.DEFAULT_MODEL)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--dataset", default=RP.DEFAULT_DATASET)
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--max-cases", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--deltas", default="0-16,20,24,32,48,64,128,256",
                    help="dense enough to resolve a period of 8; log spacing would "
                         "miss the entire signature")
    ap.add_argument("--layers", default="0,7,14,21,28,35")
    ap.add_argument("--image-side", type=int, default=768)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--min-tokens", type=int, default=32)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--null-tol", type=float, default=1e-2)
    args = ap.parse_args()
    scan(args, args.device) if args.stage == "scan" else report(args)


if __name__ == "__main__":
    main()
