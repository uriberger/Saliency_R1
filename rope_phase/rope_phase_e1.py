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
         so the logits must come back unchanged up to arithmetic.  This is the
         only check that nothing else in the stack reads absolute position.
  full   +delta to all three axes of the tail.  The honest "the query is further
         from the image" condition, and what happens as a CoT lengthens.
  t      +delta to the tail's t axis only.  Every patch shares the image's t index,
         so this offset is CONSTANT across patches: it can move image-vs-text mass
         but cannot change the shape of the profile over patches.  The visual
         fading arm, and the rival explanation.
  hw     +delta to the tail's h and w axes only.  Tail-to-tail offsets unchanged,
         image-to-image offsets unchanged, so this touches EXACTLY the cross-modal
         spatial offsets.  The hypothesis, surgically isolated.
  fix    tail h/w frozen at the value the first post-image token would have had,
         so the cross-modal h/w offset stops depending on p at all -- the
         invariance the model does not have.  Its curves must be FLAT in delta.

PHASE BUCKETING, AND WHY THE FIRST BUILD OF THIS WAS BLUNT
----------------------------------------------------------
The first build averaged the attention profile over every post-image token before
comparing anything.  That silently discarded the thing worth measuring.  Tokens in
one case sit at a few hundred different distances from the image, spanning some
thirty turns of the fastest row channel, so their stripe patterns are at thirty
different phases and averaging cancels them.  Adding a gap shifts every token's
phase equally, but it cannot un-cancel what has already cancelled.  So that build
saw only the slow channels, and the periodicity survived as a wobble on a trend
rather than as a signature.

This build buckets tokens by phase first, exactly as E0 does, using each token's
UNMODIFIED distance -- so the same tokens land in the same bucket in every arm at
every gap, and the comparison is exactly paired.  Inside a bucket the fast channel
is coherent, so it survives the averaging.

That turns the experiment into a calibration.  Adding a gap of N shifts every
token's pattern by N patches, so a bucket at gap N should be that same bucket at
gap 0, translated by N.  Fit the shift and you should recover N.  And because the
fast row channel repeats every 8 patches, the recovered shift must WRAP: a gap of
8 is indistinguishable from no gap, a gap of 5 reads as -3.

  The prediction is a sawtooth of period 8, with nothing fitted.

No decay story produces a sawtooth.  It is a far stronger signature than the
turning point the first build could see, and it runs on the same shift machinery
E0 uses, so the two experiments are measured with one ruler.

READOUTS, KEPT SEPARATE
-----------------------
  nll    mean NLL of the model's own delta=0 completion, teacher-forced.  The
         behavioural cost: how far the intervention pushes the model off its own
         trajectory.
  mass   total attention on image tokens.  The fading channel.
  corr   correlation of each phase bucket's profile against the SAME bucket at
         gap 0, at every candidate shift.  Two things come off it: the best-fitting
         shift, which should track the imposed gap and wrap every 8; and the
         correlation at zero shift, which should dip and recover on the same
         period, since a gap of 8 puts the pattern back where it started.

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
SCHEMA = 2          # 1 = token-averaged (blunt), 2 = phase-bucketed


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


def sawtooth(delta: int, period: float, lo: int, hi: int) -> int:
    """Where a shift of `delta` lands once a pattern of this period has wrapped.

    A gap of 8 on a period-8 pattern is indistinguishable from no gap; a gap of 5
    reads as -3.  This is the prediction, and it is fixed by the config alone.
    """
    x = delta % period
    if x > period / 2:
        x -= period
    return int(np.clip(round(x), lo, hi))


# ---------------------------------------------------------------------------
# capture: per phase-bucket profiles, not a token average
# ---------------------------------------------------------------------------
class BucketTap:
    """Eager replay on selected layers; bucket-sums the profile over tokens.

    Returns, per layer, the summed mean-centred log-attention profile per phase
    bucket, plus the mean image mass per head.
    """

    def __init__(self, model, layers):
        self.want = {int(l) for l in layers}
        self.out = {}
        self.mask = self.img_cols = self.binspecs = None
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

    def arm(self, mask, img_cols, tail_start, binspecs):
        """binspecs: {name: (bucket index per token, number of buckets)}.

        Several bucketings share one forward -- the row clock and the column clock
        are just different groupings of the same tokens, so replaying the model
        once per clock would double the cost for nothing.
        """
        self.mask, self.img_cols, self.tail_start = mask, img_cols, tail_start
        self.binspecs, self.out = binspecs, {}

    def disarm(self):
        self.mask = None

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
            x = a.clamp_min(1e-20).log()
            x = x - x.mean(-1, keepdim=True)     # log-softmax minus logZ == the raw logit
            per_bk = {}
            for name, (bins, nb) in self.binspecs.items():
                sums = torch.zeros(x.shape[0], nb, x.shape[2],
                                   dtype=torch.float32, device=x.device)
                sums.index_add_(1, bins, x)
                per_bk[name] = sums.cpu().numpy()
                del sums
            self.out[layer_idx] = (per_bk, mass.cpu().numpy())
            del a, x, attn
            return None
        return hook


def completion_nll(logits, ids, prompt_len: int) -> float:
    lg = logits[0, prompt_len - 1: -1].float()
    tgt = ids[0, prompt_len:]
    return float(torch.nn.functional.cross_entropy(lg, tgt).item())


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
    shifts = list(range(-args.max_shift, args.max_shift + 1))
    print(f"[e1] shard {args.shard}/{args.num_shards}: {len(rows)} cases, "
          f"{len(arms)} arms x {len(deltas)} deltas = {len(arms) * len(deltas)} forwards/case, "
          f"layers {layers}", flush=True)

    processor, model = RP.load_model(args.base_model, args.adapter or None, device)
    tcfg = model.config.text_config
    th, tw = RP.fastest_theta(tcfg, "h"), RP.fastest_theta(tcfg, "w")
    # (name, angular frequency, buckets, which image axis a shift slides)
    BK = [("row", th, args.row_buckets, 2), ("col", tw, args.col_buckets, 3)]
    for name, theta, nb, _ax in BK:
        print(f"[e1] {name}: period {RP.TWO_PI / theta:.2f} positions, {nb} buckets -- "
              f"recovered shift should be the gap, wrapping every "
              f"{RP.TWO_PI / theta:.1f}", flush=True)
    image_token_id = int(getattr(model.config, "image_token_id", None) or RP.IMAGE_TOKEN_ID)
    torch.manual_seed(args.seed + args.shard)

    A, D, S, L = len(arms), len(deltas), len(shifts), len(layers)
    n_heads = tcfg.num_attention_heads
    acc = {f"corr::{n}": np.zeros((A, D, S, L, n_heads), dtype=np.float64) for n, *_ in BK}
    acc["mass"] = np.zeros((A, D, L, n_heads), dtype=np.float64)
    acc["nll"] = np.zeros((A, D), dtype=np.float64)
    acc["dlogit_mean"] = np.zeros((A, D), dtype=np.float64)
    acc["dlogit_max"] = np.zeros((A, D), dtype=np.float64)
    acc["logit_scale"] = np.zeros((), dtype=np.float64)
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
            tap = BucketTap(model, layers)
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

        base_pos, _ = (model.model if hasattr(model, "model") else model).get_rope_index(
            ids, case["mm_token_type_ids"], image_grid_thw=case.get("image_grid_thw"),
            video_grid_thw=None, attention_mask=case.get("attention_mask"))
        # Buckets come from the UNMODIFIED distance, so a token keeps its bucket in
        # every arm at every gap.  Anything else would compare different tokens.
        anchor = base_pos[0, 0, img_cols[0]]
        d_tail = (base_pos[0, 0, tail_start:] - anchor).long()
        bins = {n: RP.phase_bins(d_tail, theta, nb) for n, theta, nb, _ in BK}
        if any(int(torch.bincount(b, minlength=nb).min()) == 0
               for (n, _t, nb, _a), b in zip(BK, bins.values())):
            continue                              # an empty bucket makes the mean undefined
        counts = {n: torch.bincount(bins[n], minlength=nb).cpu().numpy().astype(np.float64)
                  for n, _t, nb, _a in BK}
        binspecs = {n: (bins[n], nb) for n, _t, nb, _a in BK}

        mdtype = next(model.parameters()).dtype
        add = torch.zeros(seq, seq, dtype=mdtype, device=device)
        add.masked_fill_(torch.triu(torch.ones(seq, seq, dtype=torch.bool, device=device),
                                    diagonal=1), torch.finfo(mdtype).min)
        mask = add[None, None]

        ref_logits, baseline = None, {}
        for ai, arm in enumerate(arms):
            for di, delta in enumerate(deltas):
                pos = build_position_ids(base_pos, tail_start, arm, delta)
                tap.arm(mask, img_cols, tail_start, binspecs)
                out = model(**case, position_ids=pos)
                tap.disarm()

                logits = out.logits
                acc["nll"][ai, di] += completion_nll(logits, ids, prompt_len)
                cur = logits[0, prompt_len - 1:].float()
                if ref_logits is None:
                    ref_logits = cur.clone()
                    acc["logit_scale"] += float(ref_logits.abs().max().item())
                dl = (cur - ref_logits).abs()
                acc["dlogit_mean"][ai, di] += float(dl.mean().item())
                acc["dlogit_max"][ai, di] += float(dl.max().item())
                del out, logits

                for li, lay in enumerate(layers):
                    bk_sums, mass = tap.out[lay]
                    acc["mass"][ai, di, li] += mass
                    for n, _t, nb, ax in BK:
                        prof = (bk_sums[n] / counts[n][None, :, None]
                                ).reshape(n_heads, nb, gh, gw)
                        key = (n, lay)
                        if key not in baseline:
                            baseline[key] = prof.copy()      # arm 0, gap 0: unmodified
                        b0 = baseline[key]
                        for si, sh in enumerate(shifts):
                            a_, b_ = RP._crop_pair(prof, b0, int(sh), ax)
                            acc[f"corr::{n}"][ai, di, si, li] += RP._corr(a_, b_).mean(axis=1)
        del add, mask
        ncase += 1
        if ncase % args.log_every == 0:
            print(f"[e1] {ncase} cases done", flush=True)

    if tap is not None:
        tap.close()
    if ncase == 0:
        raise SystemExit("no usable cases in this shard")

    meta = {"schema": SCHEMA, "arms": arms, "deltas": deltas, "layers": layers,
            "shifts": shifts, "ncase": ncase, "gh": gh, "gw": gw, "n_heads": n_heads,
            "bucketings": [[n, float(t), int(nb), int(ax)] for n, t, nb, ax in BK],
            "shard": args.shard, "num_shards": args.num_shards, "n_samples": args.n_samples,
            "seed": args.seed, "image_side": args.image_side, "base_model": args.base_model,
            "adapter": args.adapter or "", "dataset": args.dataset,
            "max_new_tokens": args.max_new_tokens}
    np.savez(shard_path, __meta__=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8),
             **acc)
    print(f"[e1] wrote {shard_path}  cases={ncase}", flush=True)


# ---------------------------------------------------------------------------
def report(args):
    out_dir = Path(args.out_dir)
    shards = sorted((out_dir / "scan").glob("shard*.npz"))
    if not shards:
        raise SystemExit(f"no shards in {out_dir}/scan")
    acc, meta, ncase = None, None, 0
    for p in shards:
        z = np.load(p, allow_pickle=False)
        m = json.loads(bytes(z["__meta__"]).decode())
        if m.get("schema", 1) != SCHEMA:
            raise SystemExit(f"{p.name} was written by schema {m.get('schema', 1)}, this is "
                             f"schema {SCHEMA} (token-averaged vs phase-bucketed). Rescan.")
        if meta is None:
            meta = m
        elif m["deltas"] != meta["deltas"] or m["arms"] != meta["arms"]:
            raise SystemExit(f"{p.name} has a different sweep; do not merge these")
        ncase += m["ncase"]
        cur = {k: z[k] for k in z.files if k != "__meta__"}
        acc = cur if acc is None else {k: acc[k] + cur[k] for k in acc}
    acc = {k: v / max(ncase, 1) for k, v in acc.items()}
    arms, deltas, layers = meta["arms"], meta["deltas"], meta["layers"]
    shifts = np.asarray(meta["shifts"])

    lines = []
    P = lines.append
    P("=" * 78)
    P("E1 -- position-id gap sweep, phase-bucketed")
    P("=" * 78)
    P(f"model  : {meta['base_model']}")
    P(f"data   : {meta['dataset']}")
    P(f"scan   : {ncase} cases, {len(arms)} arms x {len(deltas)} gaps, layers {layers}")
    P(f"grid   : {meta['gh']} x {meta['gw']} patches")
    P("")

    ni = arms.index("null") if "null" in arms else None
    if ni is None:
        P("NULL CHECK  NOT RUN -- the null arm was excluded from --arms.")
    else:
        scale = float(acc["logit_scale"])
        nm = acc["dlogit_mean"][ni].max()
        tm = max(acc["dlogit_mean"][ai].max() for ai, a in enumerate(arms) if a != "null")
        P("NULL CHECK  shift EVERY position id, image included: must be a no-op up to")
        P("            arithmetic.  Judged against the treatments, not an absolute number:")
        P("            a max over ~1e10 bf16 values is an extreme order statistic, and bf16")
        P("            rounding of cos/sin differs at large absolute angles even when the")
        P("            offsets are mathematically identical.")
        P(f"            logit scale {scale:.1f} | null mean |dlogit| {nm:.4f} | "
          f"largest treatment {tm:.4f} | ratio {tm / max(nm, 1e-12):.1f}x")
        P(f"            {'PASS' if tm / max(nm, 1e-12) > 3 else 'FAIL'}")
    P("")

    for name, theta, nb, ax in meta["bucketings"]:
        period = RP.TWO_PI / theta
        axis_word = "rows" if ax == 2 else "columns"
        C = acc[f"corr::{name}"]                     # [arm, delta, shift, layer, head]
        P(f"--- {name} clock: period {period:.2f} positions, {nb} buckets, sliding {axis_word}")
        P(f"    RECOVERED SHIFT per arm.  Predicted for a real translation: the gap itself,")
        P(f"    wrapping every {period:.1f} -- a sawtooth.  Nothing is fitted.")
        pred = [sawtooth(d, period, shifts.min(), shifts.max()) for d in deltas]
        P("    gap       : " + " ".join(f"{d:>4d}" for d in deltas))
        P("    PREDICTED : " + " ".join(f"{v:>+4d}" for v in pred))
        for ai, arm in enumerate(arms):
            best = shifts[C[ai].mean(axis=(2, 3)).argmax(axis=1)]
            P(f"    {arm:9s} : " + " ".join(f"{v:>+4d}" for v in best))
        # At exactly half a period the two signs are the same pattern, so +4 and -4
        # are indistinguishable and scoring them as a miss would be dishonest.
        def agrees(b, p_):
            return b == p_ or (abs(p_) * 2 >= period - 0.5 and abs(b) == abs(p_))
        hit = {}
        for ai, arm in enumerate(arms):
            best = shifts[C[ai].mean(axis=(2, 3)).argmax(axis=1)]
            hit[arm] = float(np.mean([agrees(b, p_) for b, p_ in zip(best, pred)]))
        P("    fraction of gaps where the recovered shift equals the prediction:")
        P("      " + "   ".join(f"{a}={100*hit[a]:.0f}%" for a in arms))
        P("")
        P(f"    SIMILARITY AT ZERO SHIFT -- how much the pattern still looks like gap 0.")
        P(f"    Predicted to dip and recover with period {period:.1f} for a translating")
        P("    pattern, and to stay flat for anything that only changes total attention.")
        z0 = list(shifts).index(0)
        P("    gap       : " + " ".join(f"{d:>5d}" for d in deltas))
        for ai, arm in enumerate(arms):
            v = C[ai, :, z0].mean(axis=(1, 2))
            P(f"    {arm:9s} : " + " ".join(f"{x:>5.2f}" for x in v))
        P("")

    for nm_, lab in (("nll", "NLL of the model's own completion"),
                     ("mass", "attention mass on the image")):
        P(f"--- {lab}")
        P("    gap       : " + " ".join(f"{d:>5d}" for d in deltas))
        for ai, arm in enumerate(arms):
            v = acc[nm_][ai] if nm_ == "nll" else acc[nm_][ai].mean(axis=(1, 2))
            P(f"    {arm:9s} : " + " ".join(f"{x:>5.3f}" for x in v))
        P("")

    P("READING IT")
    P("  The sawtooth is the whole point.  A translating pattern must give back the gap")
    P("  you imposed and wrap when the gap reaches one period.  Nothing that merely")
    P("  weakens attention to the image can produce a non-monotone, periodic recovery.")
    P("  `t` moves an offset that is identical for every patch, so it cannot reshape the")
    P("  profile at all: it should recover shift 0 at every gap.  `fix` removes the")
    P("  p-dependence outright and should also stay at 0.")

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
    ap.add_argument("--deltas", default="0-16",
                    help="dense and covering at least two full periods; log spacing "
                         "would miss the sawtooth entirely")
    ap.add_argument("--layers", default="0,7,14,21,28,35")
    ap.add_argument("--row-buckets", type=int, default=8)
    ap.add_argument("--col-buckets", type=int, default=10)
    ap.add_argument("--max-shift", type=int, default=4)
    ap.add_argument("--image-side", type=int, default=768)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--min-tokens", type=int, default=64)
    ap.add_argument("--log-every", type=int, default=5)
    args = ap.parse_args()
    scan(args, args.device) if args.stage == "scan" else report(args)


if __name__ == "__main__":
    main()
