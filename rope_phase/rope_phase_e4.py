#!/usr/bin/env python
"""E4 -- freeze the cross-modal position, and find the best value to freeze it at.

THE INTERVENTION
----------------
A text token at index p carries M-RoPE position (p, p, p); a patch at row r,
column c of an image anchored at s carries (s, s+r, s+c).  Every patch dependence
reaches the attention logit only through d-r and d-c with d = p-s, so the
positional overlay on the image translates one patch per generated token.

E4 stops that by giving each post-image token TWO position identities: its real
one when it attends to text, and a frozen one -- h = w = s + d0, the same for
every token -- when it attends to image patches.  The overlay then stops moving.

This is not the `fix` arm of E1.  `fix` set every tail token's h/w to one value in
the position ids, which also flattened tail-to-tail spatial offsets and cost +168%
NLL; most of that damage was to text-to-text attention and had nothing to do with
the image.  Here the frozen identity is used ONLY for image columns, so
text-to-text attention is untouched.  That cannot be expressed through
position_ids -- offsets are differences of per-token values, so
"change tail<->image but not tail<->tail" has no solution -- which is why this
file patches attention instead.

d0 IS A FREE PARAMETER, AND THAT IS THE EXPERIMENT
--------------------------------------------------
Any d0 removes the drift, so correctness does not pick one.  What picks one is
accuracy against ground truth.  The natural default is d0 = max(gh, gw), the real
distance of the first token after the image, which makes the intervention an exact
no-op at the start of an answer.  There is no reason that value is best.

Note what d0 is NOT selected on: how little it disturbs the model.  The model's
surprise at its own unmodified text is minimised by doing nothing at all, so that
is a damage alarm (the `guard` stage) and never the objective.

THE READOUT
-----------
Two stimulus families, both with exact ground truth and a continuous readout.

  synthetic  E2's square: one coloured square on grey, a controlled number of
             patches off the canvas midline.
  real       a real image with a ground-truth box, scaled and slid vertically on a
             grey canvas so the box centre sits a controlled number of patches off
             the midline.  Content is identical across offsets -- the image is
             translated, never cropped -- so the only thing an offset changes is
             position.

Both ask "top half or bottom half" and read the two answer tokens' logits.  A
straight line through evidence-vs-offset crosses zero at the model's perceived
midline, in patches.  The true midline is 0 by construction, so the crossing IS
the error, signed, and smaller in absolute value is better.

  python rope_phase_e4.py --stage check  --out-dir DIR --dataset PATH   # CPU
  python rope_phase_e4.py --stage pilot  --out-dir DIR --dataset PATH   # GPU, ~2 min
  python rope_phase_e4.py --stage scan   --out-dir DIR --dataset PATH
  python rope_phase_e4.py --stage guard  --out-dir DIR --dataset PATH
  python rope_phase_e4.py --stage report --out-dir DIR
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rope_phase_probe as RP      # noqa: E402
import rope_phase_e1 as E1         # noqa: E402
import rope_phase_e2 as E2         # noqa: E402

SELF_CONTAINED_STIMULI = False     # the `real` family needs a dataset
SCHEMA = 1

REAL_QUESTION = ('In this image, is "{ref}" in the top half or the bottom half? '
                 'Answer with exactly one word.')
REAL_PREFIX = "It is in the"


# ---------------------------------------------------------------------------
# the intervention
# ---------------------------------------------------------------------------
class FrozenQueryAttention:
    """Patch the text tower so image columns are scored with a frozen query position.

    Every attention module computes its scores twice: once with the token's real
    rotary phase, once with a frozen one, and the image columns take the frozen
    result.  Keys are never touched -- the patches keep their true positions -- so
    the only thing that changes is what distance the query believes it is at when
    it looks at the image.

    `apply_rotary_pos_emb` and `repeat_kv` are taken from the module that defines
    the attention class rather than imported by name, so this follows whatever
    implementation the installed transformers is actually using.
    """

    def __init__(self, model):
        mods = [m for m in model.modules()
                if type(m).__name__ in RP.TEXT_ATTENTION_CLASSES
                and getattr(m, "layer_idx", None) is not None]
        if not mods:
            raise SystemExit(f"no text-tower attention modules; looked for "
                             f"{RP.TEXT_ATTENTION_CLASSES}")
        if any(not (hasattr(m, "q_norm") and hasattr(m, "k_norm")) for m in mods):
            raise SystemExit("this patch is written against Qwen3-VL's attention "
                             "(q_norm/k_norm on the head dim); this model differs")
        defining = sys.modules[type(mods[0]).__module__]
        self.apply_rope = defining.apply_rotary_pos_emb
        self.repeat_kv = defining.repeat_kv
        self.mods, self.originals = mods, [m.forward for m in mods]
        self.enabled = False
        self.img_cols = self.cos_f = self.sin_f = None
        for m in mods:
            m._rp_ctl = self
            m.forward = types.MethodType(_frozen_forward, m)
        self.n_layers = len(mods)

    def arm(self, img_cols, cos_f, sin_f):
        self.img_cols, self.cos_f, self.sin_f, self.enabled = img_cols, cos_f, sin_f, True

    def disarm(self):
        self.enabled = False
        self.img_cols = self.cos_f = self.sin_f = None

    def restore(self):
        for m, f in zip(self.mods, self.originals):
            m.forward = f
            if hasattr(m, "_rp_ctl"):
                del m._rp_ctl


def _resolve_mask(attention_mask, module, q_len, k_len, dtype, device):
    """The decoder may hand attention a tensor, a per-layer-type dict, or nothing."""
    am = attention_mask
    if isinstance(am, dict):
        key = getattr(module, "layer_type", None)
        am = am.get(key, next(iter(am.values()))) if am else None
    if am is None:
        m = torch.full((q_len, k_len), torch.finfo(dtype).min, dtype=dtype, device=device)
        return torch.triu(m, diagonal=k_len - q_len + 1)[None, None]
    if am.dtype == torch.bool:
        return torch.zeros_like(am, dtype=dtype).masked_fill_(~am, torch.finfo(dtype).min)
    return am


def _frozen_forward(self, hidden_states, position_embeddings, attention_mask=None,
                    past_key_values=None, **kwargs):
    ctl = self._rp_ctl
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    q = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    k = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    q_real, k_rot = ctl.apply_rope(q, k, cos, sin)
    if past_key_values is not None:
        # Stock behaviour, kept so the patch cannot silently corrupt a cached
        # generate: without it the model would attend only inside the current
        # chunk and produce fluent nonsense rather than an error.
        k_rot, v = past_key_values.update(k_rot, v, self.layer_idx)
    kk = ctl.repeat_kv(k_rot, self.num_key_value_groups)
    vv = ctl.repeat_kv(v, self.num_key_value_groups)

    scores = torch.matmul(q_real, kk.transpose(2, 3)) * self.scaling
    if ctl.enabled:
        if ctl.cos_f.shape[-2] != q.shape[-2]:
            raise SystemExit(
                f"frozen phase covers {ctl.cos_f.shape[-2]} positions but this "
                f"forward has {q.shape[-2]}; the frozen ids were built for a "
                f"different sequence (incremental decoding is not supported)")
        # The frozen phase differs from the real one only on the h/w channels of
        # post-image tokens, so for every other query row q_frozen == q_real and
        # writing the image columns back is a no-op there.
        q_frozen, _ = ctl.apply_rope(q, k, ctl.cos_f, ctl.sin_f)
        img = ctl.img_cols
        scores[:, :, :, img] = torch.matmul(
            q_frozen, kk[:, :, img, :].transpose(2, 3)) * self.scaling

    mask = _resolve_mask(attention_mask, self, scores.shape[2], scores.shape[3],
                         scores.dtype, scores.device)
    scores = scores + mask[:, :, :, : scores.shape[-1]]
    w = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    out = torch.matmul(w, vv).transpose(1, 2).reshape(*input_shape, -1).contiguous()
    return self.o_proj(out), None


def find_text_rotary(model):
    for m in model.modules():
        if type(m).__name__.endswith("TextRotaryEmbedding"):
            return m
    raise SystemExit("no text rotary embedding module found")


def frozen_position_ids(pos: torch.Tensor, tail_start: int, anchor: int, d0: int):
    """Real ids everywhere except the tail's h/w, which are pinned to anchor + d0.

    The t axis keeps its real value, so image-vs-text mass -- the visual fading
    channel, which cannot reshape anything -- is left exactly as the model made it.
    """
    out = pos.clone()
    out[1:3, :, tail_start:] = anchor + d0
    return out


def parse_d0(spec: str) -> list[int]:
    return E1.parse_deltas(spec)


def arm_labels(d0s):
    return ["none"] + [f"frozen:{d}" for d in d0s]


# ---------------------------------------------------------------------------
# stimuli
# ---------------------------------------------------------------------------
def slack_window(img_h_scaled: float, side: int, max_off_px: float):
    """Where the box centre may sit, in scaled-image pixels, to allow the full sweep.

    Pasting the image at vertical position `top` puts the box centre at top + cy,
    and that must reach side/2 +- max_off_px while the image stays entirely on the
    canvas.  Solving both ends gives this window; outside it some offset in the
    sweep would push the image off the canvas and crop content, which would make
    the offset change content as well as position.
    """
    return (img_h_scaled - side / 2 + max_off_px, side / 2 - max_off_px)


def real_stimulus(image, bbox, side: int, patch_px: int, target_h: int,
                  offset_patches: float):
    """A real image translated on a grey canvas so the box centre lands at `offset`.

    Positive offset is DOWN, matching the row axis and E2's convention.  The image
    is scaled once and only translated between offsets, so every offset shows
    byte-identical content.
    """
    from PIL import Image

    if image.mode != "RGB":
        image = image.convert("RGB")
    w, h = image.size
    sc = min(side / w, target_h / h)
    ws, hs = max(1, round(w * sc)), max(1, round(h * sc))
    small = image.resize((ws, hs), 2)
    cy = (bbox[1] + bbox[3]) / 2 * hs
    canvas = Image.new("RGB", (side, side), (128, 128, 128))
    top = side / 2 + offset_patches * patch_px - cy
    canvas.paste(small, ((side - ws) // 2, int(round(top))))
    return canvas


def usable_real_rows(dataset: str, n: int, seed: int, side: int, patch_px: int,
                     target_h: int, max_off: float, max_box_h: float,
                     max_box_w: float, max_words: int, pool: int):
    """Rows with a small, well-placed box and a quotable answer.

    The box must be small enough to have a location at all, and its centre must sit
    where the image can still be slid over the whole sweep.  That is a selection on
    where things happen to be in the picture, which is unrelated to anything this
    experiment measures.
    """
    rows = RP.load_samples(dataset, pool, seed)
    if rows and "bbox" not in rows[0]:
        raise SystemExit(
            f"{dataset} has no `bbox` column, so there is no ground truth to score "
            f"against; the real family needs one. Columns seen: {sorted(rows[0])}")
    out = []
    for r in rows:
        # A malformed box is skipped, but a MISSING one is a configuration error and
        # is raised above: swallowing it here once turned an empty result set into a
        # cheerful "check passed".
        try:
            b = json.loads(r["bbox"]) if isinstance(r["bbox"], str) else r["bbox"]
        except (TypeError, ValueError):
            continue
        if b is None or len(b) != 4 or not (0 <= b[0] < b[2] <= 1 and 0 <= b[1] < b[3] <= 1):
            continue
        if (b[3] - b[1]) > max_box_h or (b[2] - b[0]) > max_box_w:
            continue
        ref = (r.get("solution") or "").strip()
        if not ref or "\n" in ref or len(ref.split()) > max_words:
            continue
        w, h = r["image"].size
        hs = h * min(side / w, target_h / h)
        lo, hi = slack_window(hs, side, max_off * patch_px)
        if not (lo <= (b[1] + b[3]) / 2 * hs <= hi):
            continue
        out.append({"image": r["image"], "bbox": b, "ref": ref,
                    "dataset": r.get("dataset", "?"), "row_index": r["row_index"]})
        if len(out) >= n:
            break
    return out


def build_case(processor, image, question: str, prefix: str, device):
    messages = [{"role": "user",
                 "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    text = processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True) + prefix
    return processor(text=[text], images=[[image]], return_tensors="pt",
                     padding=True, add_special_tokens=False).to(device)


# ---------------------------------------------------------------------------
# readout
# ---------------------------------------------------------------------------
def fit_zero(offsets, ev):
    """(perceived midline, slope) from a straight line through evidence vs offset.

    A least-squares line rather than E2's two-point interpolation: on real images
    the sweep is short and need not bracket the crossing, and a fit uses every
    point instead of the two nearest a sign change.  The slope comes back too,
    because a flat curve means the stimulus says nothing about position and its
    crossing is noise however tidy the number looks.
    """
    x, y = np.asarray(offsets, float), np.asarray(ev, float)
    if not np.all(np.isfinite(y)):
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(x, y, 1)
    if abs(slope) < 1e-9:
        return float("nan"), float(slope)
    return float(-intercept / slope), float(slope)


# ---------------------------------------------------------------------------
# Keys that define the sweep.  If any of them differs, two files are answers to
# different questions and must not be stitched together -- the same discipline E0's
# report applies to its shards.
RESUME_KEYS = ("family", "offsets", "d0s", "gaps", "arms", "image_side", "square_px",
               "target_h", "base_model", "dataset", "gh", "gw", "colours", "rows")


def _load_partial(out_path, meta, shape):
    """Stimuli already done by an earlier run of the SAME sweep.

    The stimulus list is deterministic -- a fixed seed, a fixed filter, taken in
    order -- so stimulus i is the same picture in every run of one sweep, and the
    finished rows can simply be kept.  Anything that would change what stimulus i
    IS appears in RESUME_KEYS and refuses the merge instead.
    """
    if out_path is None or not out_path.exists():
        return None, 0
    z = np.load(out_path, allow_pickle=False)
    old = json.loads(bytes(z["__meta__"]).decode())
    diff = [k for k in RESUME_KEYS if old.get(k) != meta.get(k)]
    if diff:
        raise SystemExit(
            f"{out_path.name} was written by a different sweep (differs on {diff}). "
            f"Use a fresh --out-dir; stitching two sweeps together would report a "
            f"table whose rows came from different experiments.")
    ev_old = z["ev"]
    if tuple(ev_old.shape) != tuple(shape):
        raise SystemExit(f"{out_path.name} has shape {ev_old.shape}, this sweep needs "
                         f"{tuple(shape)}; use a fresh --out-dir")
    done = int(old.get("done", 0))
    # Trust the counter only as far as the data backs it up: a run killed mid-write
    # could leave the count ahead of the values.
    while done > 0 and not np.all(np.isfinite(ev_old[done - 1])):
        done -= 1
    return ev_old, done


@torch.no_grad()
def _sweep(model, ids, ctl, rotary, cases, offsets, d0s, gaps, device, label,
           out_path=None, meta=None):
    """evidence[stimulus, offset, arm, gap]; arms are `none` then one per d0."""
    image_token_id = int(getattr(model.config, "image_token_id", None) or RP.IMAGE_TOKEN_ID)
    dummy = torch.zeros(1, 1, 1, dtype=next(model.parameters()).dtype, device=device)
    inner = model.model if hasattr(model, "model") else model
    shape = (len(cases), len(offsets), 1 + len(d0s), len(gaps))
    ev_old, start = _load_partial(out_path, meta, shape)
    ev = np.full(shape, np.nan) if ev_old is None else ev_old.copy()
    if start:
        print(f"[e4] {label}: resuming, {start}/{len(cases)} stimuli already done",
              flush=True)
        if start >= len(cases):
            return ev
    t0 = time.time()
    for si, mk in enumerate(cases):
        if si < start:
            continue
        for oi, off in enumerate(offsets):
            case = mk(off)
            input_ids = case["input_ids"]
            img_cols = (input_ids[0] == image_token_id).nonzero(as_tuple=True)[0]
            tail_start = int(img_cols[-1].item()) + 1
            base_pos, _ = inner.get_rope_index(
                input_ids, case["mm_token_type_ids"],
                image_grid_thw=case.get("image_grid_thw"), video_grid_thw=None,
                attention_mask=case.get("attention_mask"))
            anchor = int(base_pos[0, 0, img_cols[0]].item())
            for gi, gap in enumerate(gaps):
                pos = E1.build_position_ids(base_pos, tail_start, "full", gap)
                ctl.disarm()
                ev[si, oi, 0, gi] = E2.evidence(model(**case, position_ids=pos, use_cache=False).logits, ids)
                for ai, d0 in enumerate(d0s, start=1):
                    cos_f, sin_f = rotary(
                        dummy, frozen_position_ids(pos, tail_start, anchor, d0))
                    ctl.arm(img_cols, cos_f, sin_f)
                    ev[si, oi, ai, gi] = E2.evidence(
                        model(**case, position_ids=pos, use_cache=False).logits, ids)
                    ctl.disarm()
        el = time.time() - t0
        # Rate over what THIS run has done, not over the resumed count, or a resumed
        # job reports an ETA based on work it never did.
        rate = el / max(si + 1 - start, 1)
        print(f"[e4] {label}: {si + 1}/{len(cases)} stimuli, {el / 60:.1f} min elapsed, "
              f"~{rate * (len(cases) - si - 1) / 60:.1f} min left", flush=True)
        # Written after every stimulus: a job killed at the wall clock should not
        # cost the whole sweep, which is what E0's all-or-nothing shard write did.
        if out_path is not None:
            np.savez(out_path, ev=ev, __meta__=np.frombuffer(
                json.dumps({**meta, "done": si + 1}).encode(), dtype=np.uint8))
    return ev


@torch.no_grad()
def _null_selfcheck(model, rotary, case, device):
    """The patch must be a no-op when the frozen phase equals the real one.

    Two numbers, both judged against the logit scale rather than a guessed
    tolerance: the hand-written attention against the library's own kernel, and the
    frozen path against the real one when handed identical positions.  A splice
    that indexed the wrong rows or columns would show up here.
    """
    image_token_id = int(getattr(model.config, "image_token_id", None) or RP.IMAGE_TOKEN_ID)
    inner = model.model if hasattr(model, "model") else model
    input_ids = case["input_ids"]
    img_cols = (input_ids[0] == image_token_id).nonzero(as_tuple=True)[0]
    base_pos, _ = inner.get_rope_index(
        input_ids, case["mm_token_type_ids"], image_grid_thw=case.get("image_grid_thw"),
        video_grid_thw=None, attention_mask=case.get("attention_mask"))
    dummy = torch.zeros(1, 1, 1, dtype=next(model.parameters()).dtype, device=device)

    stock = model(**case, position_ids=base_pos, use_cache=False).logits[0, -1].float()
    ctl = FrozenQueryAttention(model)
    mine = model(**case, position_ids=base_pos, use_cache=False).logits[0, -1].float()
    cos_f, sin_f = rotary(dummy, base_pos)
    ctl.arm(img_cols, cos_f, sin_f)
    spliced = model(**case, position_ids=base_pos, use_cache=False).logits[0, -1].float()
    ctl.disarm()
    return ctl, {"logit_scale": float(stock.abs().max().item()),
                 "eager_vs_stock": float((mine - stock).abs().max().item()),
                 "frozen_vs_real": float((spliced - mine).abs().max().item())}


def _families(args, processor, model, device):
    """(name, [stimulus factories], offsets, extra meta) for each family requested."""
    patch_px = int(model.config.vision_config.patch_size) * \
        int(model.config.vision_config.spatial_merge_size)
    out = []
    want = [f for f in args.families.split(",") if f]
    if "synthetic" in want:
        offs = [float(x) for x in args.offsets_synth.split(",")]
        colours = [c for c in args.colours.split(",") if c]
        cases = [(lambda c: (lambda off: build_case(
            processor, E2.make_image(args.image_side, off, c, patch_px, args.square_px),
            E2.QUESTION, E2.ANSWER_PREFIX, device)))(c) for c in colours]
        out.append(("synthetic", cases, offs, {"colours": colours}))
    if "real" in want:
        offs = [float(x) for x in args.offsets_real.split(",")]
        if any(abs(o - round(o)) > 1e-9 for o in offs):
            raise SystemExit(
                f"real offsets must be whole patches, got {offs}: a sub-patch shift "
                f"re-aligns content to the patch grid and changes the features, "
                f"which shows up as a sawtooth in the evidence curve")
        rows = usable_real_rows(args.dataset, args.n_real, args.seed, args.image_side,
                                patch_px, args.target_h, max(abs(o) for o in offs),
                                args.max_box_h, args.max_box_w, args.max_words, args.pool)
        if not rows:
            raise SystemExit("no usable rows in the dataset after filtering")
        cases = [(lambda r: (lambda off: build_case(
            processor, real_stimulus(r["image"], r["bbox"], args.image_side, patch_px,
                                     args.target_h, off),
            REAL_QUESTION.format(ref=r["ref"]), REAL_PREFIX, device)))(r) for r in rows]
        out.append(("real", cases, offs,
                    {"rows": [{k: v for k, v in r.items() if k != "image"} for r in rows]}))
    return out, patch_px


@torch.no_grad()
def run(args, device, pilot: bool):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    processor, model = RP.load_model(args.base_model, args.adapter or None, device,
                                     attn_impl="eager")
    ids = E2.answer_token_ids(processor)
    rotary = find_text_rotary(model)
    fams, patch_px = _families(args, processor, model, device)
    d0s = [] if pilot else parse_d0(args.d0)
    gaps = [0] if pilot else E1.parse_deltas(args.gaps)

    probe_case = fams[0][1][0](0.0)
    ctl, checks = _null_selfcheck(model, rotary, probe_case, device)
    print(f"[e4] null self-check: logit scale {checks['logit_scale']:.1f} | "
          f"eager vs stock {checks['eager_vs_stock']:.4f} | "
          f"frozen vs real {checks['frozen_vs_real']:.4f}", flush=True)
    if checks["frozen_vs_real"] > 0.05 * checks["logit_scale"]:
        raise SystemExit("the frozen path does not reproduce the real one when given "
                         "identical positions; the splice is wrong")

    grid = probe_case["image_grid_thw"][0].tolist()
    merge = int(model.config.vision_config.spatial_merge_size)
    gh, gw = grid[1] // merge, grid[2] // merge
    print(f"[e4] grid {gh}x{gw} patches, one patch = {patch_px} px; natural d0 = "
          f"{max(gh, gw)} (the real distance of the first token after the image)",
          flush=True)
    print(f"[e4] {len(fams)} families x {1 + len(d0s)} arms x {len(gaps)} gaps",
          flush=True)

    for name, cases, offs, extra in fams:
        if pilot:
            ev = _sweep(model, ids, ctl, rotary, cases, offs, [], [0], device, name)
            print(f"\nBASELINE CURVE -- {name}: log P(top) - log P(bottom)")
            print("  offset(patches)       : " + " ".join(f"{o:>7.2f}" for o in offs))
            for si in range(len(cases)):
                lbl = (extra["colours"][si] if name == "synthetic"
                       else f"{extra['rows'][si]['dataset']}/{extra['rows'][si]['ref']}")
                print(f"  {lbl[:22]:<22s}: " + " ".join(
                    f"{ev[si, oi, 0, 0]:>7.2f}" for oi in range(len(offs))))
            print("  mean                  : " + " ".join(
                f"{np.nanmean(ev[:, oi, 0, 0]):>7.2f}" for oi in range(len(offs))))
            zs = [fit_zero(offs, ev[si, :, 0, 0]) for si in range(len(cases))]
            print(f"  perceived midline  : {[round(z, 2) for z, _ in zs]}")
            print(f"  slope, logits/patch: {[round(s, 2) for _, s in zs]}")
            print("  Usable only where the slope is clearly negative: a flat curve means")
            print("  the stimulus carries no positional information and its crossing is")
            print("  noise, however tidy the number looks.")
            continue
        meta = {"schema": SCHEMA, "family": name, "offsets": offs, "d0s": d0s,
                "gaps": gaps, "arms": arm_labels(d0s), "patch_px": patch_px,
                "gh": gh, "gw": gw, "natural_d0": max(gh, gw),
                "image_side": args.image_side, "square_px": args.square_px,
                "target_h": args.target_h, "base_model": args.base_model,
                "dataset": args.dataset, "checks": checks, **extra}
        _sweep(model, ids, ctl, rotary, cases, offs, d0s, gaps, device, name,
               out_path=out_dir / f"scan_{name}.npz", meta=meta)
        print(f"[e4] wrote {out_dir / f'scan_{name}.npz'}", flush=True)
    ctl.restore()


# ---------------------------------------------------------------------------
@torch.no_grad()
def guard(args, device):
    """The damage alarm: does freezing hurt ordinary language modelling?

    Never the objective -- surprise at the model's own unmodified text is minimised
    by not intervening at all -- but a candidate that wrecks the model cannot be
    rescued by fixing the drift, so it is a rejection filter.
    """
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    processor, model = RP.load_model(args.base_model, args.adapter or None, device,
                                     attn_impl="eager")
    rotary = find_text_rotary(model)
    d0s = parse_d0(args.d0)
    image_token_id = int(getattr(model.config, "image_token_id", None) or RP.IMAGE_TOKEN_ID)
    inner = model.model if hasattr(model, "model") else model
    rows = RP.load_samples(args.dataset, args.n_guard, args.seed + 7)
    dummy = torch.zeros(1, 1, 1, dtype=next(model.parameters()).dtype, device=device)

    nll, flip, n = np.zeros(1 + len(d0s)), np.zeros(1 + len(d0s)), 0
    for ci, row in enumerate(rows):
        image = RP.square_image(row["image"], args.image_side)
        try:
            inputs, prompt_len, comp_ids = RP.generate(
                processor, model, image, row["question"], args.max_new_tokens, 1.0, device)
        except Exception as exc:
            print(f"[warn] case {ci} generate failed: {exc}", flush=True)
            continue
        if len(comp_ids) < args.min_tokens:
            continue
        case = RP.teacher_forced_case(inputs, comp_ids, device)
        cids = case["input_ids"]
        img_cols = (cids[0] == image_token_id).nonzero(as_tuple=True)[0]
        tail_start = int(img_cols[-1].item()) + 1
        pos, _ = inner.get_rope_index(
            cids, case["mm_token_type_ids"], image_grid_thw=case.get("image_grid_thw"),
            video_grid_thw=None, attention_mask=case.get("attention_mask"))
        anchor = int(pos[0, 0, img_cols[0]].item())

        ctl = FrozenQueryAttention(model)
        ctl.disarm()
        out = model(**case, position_ids=pos, use_cache=False)
        ref = E1.greedy_tokens(out.logits, prompt_len).clone()
        nll[0] += E1.completion_nll(out.logits, cids, prompt_len)
        flip[0] += E1.flip_rate(out.logits, ref, prompt_len)
        del out
        for ai, d0 in enumerate(d0s, start=1):
            cos_f, sin_f = rotary(dummy, frozen_position_ids(pos, tail_start, anchor, d0))
            ctl.arm(img_cols, cos_f, sin_f)
            o = model(**case, position_ids=pos, use_cache=False)
            nll[ai] += E1.completion_nll(o.logits, cids, prompt_len)
            flip[ai] += E1.flip_rate(o.logits, ref, prompt_len)
            ctl.disarm()
            del o
        ctl.restore()
        n += 1
        print(f"[e4] guard {n} cases done ({len(comp_ids)} completion tokens)", flush=True)
    if n == 0:
        raise SystemExit("no usable guard cases")
    meta = {"schema": SCHEMA, "d0s": d0s, "arms": arm_labels(d0s), "ncase": n,
            "base_model": args.base_model, "dataset": args.dataset}
    np.savez(out_dir / "guard.npz",
             __meta__=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8),
             nll=nll / n, flip=flip / n)
    print(f"[e4] wrote {out_dir / 'guard.npz'}", flush=True)


# ---------------------------------------------------------------------------
def check(args):
    """CPU: the geometry, the position ids and the dataset filter, without a GPU."""
    side, patch = args.image_side, 32
    offs = [float(x) for x in args.offsets_real.split(",")]
    print(f"[e4] real offsets      {offs}")
    print(f"[e4] synthetic offsets {[float(x) for x in args.offsets_synth.split(',')]}")
    print(f"[e4] d0 sweep ({len(parse_d0(args.d0))} values): {parse_d0(args.d0)}")
    print(f"[e4] gaps: {E1.parse_deltas(args.gaps)}")

    from PIL import Image
    img = Image.new("RGB", (600, 400), (10, 200, 30))
    bbox = [0.4, 0.45, 0.6, 0.55]
    for off in offs:
        canv = real_stimulus(img, bbox, side, patch, args.target_h, off)
        rows = [y for y in range(side)
                if canv.getpixel((side // 2, y)) != (128, 128, 128)]
        got = (np.mean(rows) - side / 2) / patch
        assert abs(got - off) < 0.06, (off, got, "image centre landed in the wrong place")
    print(f"[e4] real stimulus geometry: lands within 0.06 patches at all "
          f"{len(offs)} offsets")

    a = real_stimulus(img, bbox, side, patch, args.target_h, min(offs))
    b = real_stimulus(img, bbox, side, patch, args.target_h, max(offs))
    na = sum(a.getpixel((side // 2, y)) != (128, 128, 128) for y in range(side))
    nb = sum(b.getpixel((side // 2, y)) != (128, 128, 128) for y in range(side))
    assert na == nb and na > 0, (na, nb, "an offset must translate the image, not crop it")
    print(f"[e4] content conserved across the sweep ({na} image rows at both extremes)")

    base = torch.zeros(3, 1, 40, dtype=torch.long)
    base[:, 0, :] = torch.arange(40)
    base[1, 0, 10:20] += 3
    fz = frozen_position_ids(base, 20, 5, 24)
    assert torch.equal(fz[0], base[0]), "the t axis must keep its real value"
    assert torch.equal(fz[:, :, :20], base[:, :, :20]), "nothing before the tail moves"
    assert (fz[1, 0, 20:] == 29).all() and (fz[2, 0, 20:] == 29).all()
    assert not torch.equal(fz[1], base[1]), "the tail's h must actually change"
    print("[e4] frozen position ids: t untouched, pre-tail untouched, tail h/w pinned")

    lo, hi = slack_window(512, 768, 1.5 * 32)
    assert (lo, hi) == (512 - 384 + 48, 384 - 48)
    print(f"[e4] slack window, 512px image on a 768px canvas: {lo:.0f}..{hi:.0f} px")

    if args.dataset:
        got = usable_real_rows(args.dataset, args.n_real, args.seed, side, patch,
                               args.target_h, max(abs(o) for o in offs),
                               args.max_box_h, args.max_box_w, args.max_words, args.pool)
        import collections
        print(f"[e4] dataset yields {len(got)}/{args.n_real} usable rows from a pool of "
              f"{args.pool}: {collections.Counter(r['dataset'] for r in got).most_common()}")
        for r in got[:6]:
            print(f"      {r['dataset']:16s} ref={r['ref']!r}")
        # An empty result set is a failure, not a quiet zero.  The first run of this
        # check reported "passed" on nothing at all, because the loader was not
        # returning the annotation columns and every row failed silently.
        assert len(got) >= min(args.n_real, 4), (
            f"only {len(got)} usable rows; the real family cannot run")
    print("[e4] check passed")


# ---------------------------------------------------------------------------
def report(args):
    out_dir = Path(args.out_dir)
    lines = []
    P = lines.append
    P("=" * 78)
    P("E4 -- which pretend location is best?")
    P("=" * 78)

    for path in sorted(out_dir.glob("scan_*.npz")):
        z = np.load(path, allow_pickle=False)
        m = json.loads(bytes(z["__meta__"]).decode())
        ev, offs = z["ev"], np.asarray(m["offsets"])
        arms, gaps = m["arms"], m["gaps"]
        S = int(m.get("done", ev.shape[0]))
        ev = ev[:S]

        mid = np.full((S, len(arms), len(gaps)), np.nan)
        slope = np.full((S, len(arms), len(gaps)), np.nan)
        for si in range(S):
            for ai in range(len(arms)):
                for gi in range(len(gaps)):
                    mid[si, ai, gi], slope[si, ai, gi] = fit_zero(offs, ev[si, :, ai, gi])
        # A stimulus the model cannot read positionally has a meaningless crossing.
        # Judge that on the untouched arm at gap 0 and drop the whole stimulus, so
        # every arm is scored on exactly the same set.
        ok = slope[:, 0, 0] <= -args.min_slope
        # A crossing outside the swept offsets is an extrapolation off the end of the
        # fitted line, not a measurement between two points.  It is still the best
        # estimate available, but the reader should know how much of the table is one.
        inside = np.isfinite(mid) & (mid >= offs.min()) & (mid <= offs.max())
        frac_extrap = 1.0 - float(inside[ok].mean()) if int(ok.sum()) else float("nan")
        P("")
        P(f"--- family: {m['family']}   ({int(ok.sum())}/{S} stimuli with a usable slope)")
        P(f"    {100 * frac_extrap:.0f}% of crossings fall outside the swept offsets "
          f"{offs.min():+.1f}..{offs.max():+.1f} and are extrapolated from the fit")
        P(f"    model {m['base_model']}")
        P(f"    grid {m['gh']}x{m['gw']} patches; natural d0 = {m['natural_d0']}; "
          f"offsets {list(offs)}")
        if m["family"] == "real":
            import collections
            P(f"    sources: "
              f"{collections.Counter(r['dataset'] for r in m['rows'][:S]).most_common()}")
        P(f"    self-check: eager vs stock {m['checks']['eager_vs_stock']:.4f}, "
          f"frozen vs real {m['checks']['frozen_vs_real']:.4f}, "
          f"logit scale {m['checks']['logit_scale']:.1f}")
        if int(ok.sum()) < 2:
            P("    too few usable stimuli to report")
            continue
        P("")
        P("    PERCEIVED MIDLINE in patches.  Ground truth is 0.000: the sign says which")
        P("    way the model is wrong and |value| is the error.  +- is the standard error")
        P("    over stimuli, so a difference smaller than that is not a difference.")
        P("    " + " " * 12 + "".join(f"{'gap ' + str(g):>19s}" for g in gaps))
        rows_out = []
        for ai, arm in enumerate(arms):
            cells = []
            for gi in range(len(gaps)):
                v = mid[ok, ai, gi]
                v = v[np.isfinite(v)]
                cells.append((float(v.mean()) if len(v) else float("nan"),
                              float(v.std(ddof=1) / len(v) ** 0.5) if len(v) > 1
                              else float("nan")))
            rows_out.append((arm, cells))
            P(f"    {arm:>12s}" + "".join(f"{c[0]:>+12.3f} +-{c[1]:<5.3f}" for c in cells))
        P("")
        gl = len(gaps) - 1
        # Every arm is measured on the SAME stimulus, so the per-stimulus content
        # prior -- which is most of the spread above -- cancels in the difference.
        # That makes arm-to-arm comparisons far better powered than the absolute
        # level, and it is how to tell a real difference from a coincidence.
        P("    PAIRED SHIFT vs the untouched model at gap "
          f"{gaps[0]}, same stimulus, same offsets.")
        P("    " + " " * 12 + "".join(f"{'gap ' + str(g):>19s}" for g in gaps))
        paired = {}
        for ai, arm in enumerate(arms):
            cells = []
            for gi in range(len(gaps)):
                d = mid[ok, ai, gi] - mid[ok, 0, 0]
                d = d[np.isfinite(d)]
                cells.append((float(d.mean()) if len(d) else float("nan"),
                              float(d.std(ddof=1) / len(d) ** 0.5) if len(d) > 1
                              else float("nan")))
            paired[arm] = cells
            P(f"    {arm:>12s}" + "".join(f"{c[0]:>+12.3f} +-{c[1]:<5.3f}" for c in cells))
        P("")
        short, long_ = rows_out[0][1][0][0], rows_out[0][1][gl][0]
        P(f"    the drift, measured paired: {paired['none'][gl][0]:+.3f} "
          f"+-{paired['none'][gl][1]:.3f} patches from gap {gaps[0]} to {gaps[gl]}")
        P(f"    REFERENCE  untouched, short answer (gap {gaps[0]}) : {short:+.3f}")
        P(f"    REFERENCE  untouched, long answer  (gap {gaps[gl]}) : {long_:+.3f}"
          f"   <- the drift adds {abs(long_) - abs(short):+.3f} patches of error")
        ranked = sorted((abs(c[gl][0]), a, c[gl]) for a, c in rows_out[1:]
                        if np.isfinite(c[gl][0]))
        P("")
        P(f"    BEST PRETEND LOCATIONS at gap {gaps[gl]}, by absolute error against")
        P("    ground truth.  `paired` is the same arm's shift from the untouched")
        P("    short-answer model, which is the precisely measured quantity.")
        for err, arm, cell in ranked[:8]:
            pc = paired[arm][gl]
            P(f"      {arm:>12s}  {cell[0]:>+7.3f} +-{cell[1]:.3f}   |error| {err:.3f}"
              f"   paired {pc[0]:>+7.3f} +-{pc[1]:.3f}")
        P("    WORST:")
        for err, arm, cell in ranked[-3:]:
            pc = paired[arm][gl]
            P(f"      {arm:>12s}  {cell[0]:>+7.3f} +-{cell[1]:.3f}   |error| {err:.3f}"
              f"   paired {pc[0]:>+7.3f} +-{pc[1]:.3f}")
        if m["family"] == "real":
            P("")
            P("    CAVEAT for real images: the absolute error is dominated by the model's")
            P("    content prior about where a thing belongs -- a horse low, a sombrero")
            P("    high -- which spread the pilot's crossings over ~2.9 patches.  That is")
            P("    a real error against ground truth and it is fair to count it, but it is")
            P("    not positional, no frozen phase should be expected to fix it, and it is")
            P("    why the paired column is the one that resolves arm from arm.  The")
            P("    square has no such prior, which is why E2 was synthetic.")
        P("")
        P("    FLATNESS CHECK -- a frozen arm should barely move with the gap, since its")
        P("    image attention no longer knows the distance.  What is left is the t axis,")
        P("    which changes mass and cannot reshape the profile.")
        for arm, cells in rows_out:
            if arm == "none" or arm == f"frozen:{m['natural_d0']}":
                P(f"      {arm:>12s}  " + "  ".join(f"gap {g}: {c[0]:+.3f}"
                                                    for g, c in zip(gaps, cells)))

    gpath = out_dir / "guard.npz"
    if gpath.exists():
        z = np.load(gpath, allow_pickle=False)
        m = json.loads(bytes(z["__meta__"]).decode())
        P("")
        P("--- GUARD: damage to ordinary language modelling (rejection filter only)")
        P(f"    {m['ncase']} cases, teacher-forced on the untouched model's own text.")
        P("    NOT how the winner is chosen: the arm that changes nothing wins here by")
        P("    construction.  It is here to reject arms that break the model.")
        P(f"    {'arm':>12s} {'NLL':>8s} {'vs none':>9s} {'flips %':>9s}")
        base = float(z["nll"][0])
        for ai, arm in enumerate(m["arms"]):
            P(f"    {arm:>12s} {z['nll'][ai]:>8.3f} "
              f"{100 * (z['nll'][ai] - base) / max(base, 1e-9):>+8.1f}% "
              f"{100 * z['flip'][ai]:>8.1f}")

    P("")
    P("READING IT")
    P("  The question is which frozen distance leaves the smallest error against ground")
    P("  truth on a long answer -- not which one disturbs the model least.  Three things")
    P("  can happen.  If every frozen arm sits at the untouched model's long-answer")
    P("  error, the intervention does nothing.  If they sit at its short-answer error,")
    P("  the fix works and buys back exactly what the drift cost.  If one sits BELOW the")
    P("  short-answer error, part of the model's standing spatial bias was the overlay")
    P("  all along, and freezing it is worth more than merely removing the drift.")
    text = "\n".join(lines)
    print(text)
    (out_dir / "report.txt").write_text(text + "\n")
    print(f"\n[e4] wrote {out_dir / 'report.txt'}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["check", "pilot", "scan", "guard", "report"],
                    required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--base-model", default=RP.DEFAULT_MODEL)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--dataset", default=RP.DEFAULT_DATASET)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--families", default="synthetic,real")
    # 24..47 is three turns of the fast row clock and 2.4 of the column clock.  0 and
    # 12 put the query inside the image's own rows, where no text token ever sits, and
    # are off-distribution controls.  64/96/112 probe the slower channels.
    ap.add_argument("--d0", default="0,12,24-47,64,96,112")
    ap.add_argument("--gaps", default="0,512")
    ap.add_argument("--image-side", type=int, default=768)
    ap.add_argument("--square-px", type=int, default=64)
    # Pass these as --offsets-real=-1.5,... : argparse reads a space-separated value
    # beginning with a minus sign as another flag.
    ap.add_argument("--offsets-synth", default="-3,-2,-1.5,-1,-0.5,0,0.5,1")
    # WHOLE patches only.  A sub-patch translation re-aligns content to the patch
    # grid, which changes the features and not just the position: in the pilot one
    # docvqa page read +5.59 at half-patch offsets and -1.88 at whole-patch ones, a
    # 7.5-logit sawtooth with nothing to do with where the box was.  A whole-patch
    # shift maps patch k onto patch k+n exactly and cannot alias.  The square is
    # insensitive to this, so the synthetic sweep keeps E2's fractional offsets.
    ap.add_argument("--offsets-real", default="-2,-1,0,1,2")
    ap.add_argument("--colours", default="red,blue,green,yellow")
    # The pilot put the between-stimulus spread of the crossing at 2.9 patches --
    # that is the model's content prior about where a horse or a sombrero lives, not
    # measurement noise -- so the ABSOLUTE level needs many stimuli to pin down.
    # Arm-to-arm differences are paired on the same stimulus and are far cheaper.
    ap.add_argument("--n-real", type=int, default=120)
    ap.add_argument("--pool", type=int, default=2500)
    ap.add_argument("--target-h", type=int, default=512)
    ap.add_argument("--max-box-h", type=float, default=0.35)
    ap.add_argument("--max-box-w", type=float, default=0.6)
    # 4 words, not 6: at 6 the flickr30k answers come through as whole sentences
    # ("The children are of African ethnicity."), which do not name a thing that can
    # be in the top half of a picture.
    ap.add_argument("--max-words", type=int, default=4)
    ap.add_argument("--min-slope", type=float, default=0.5)
    ap.add_argument("--n-guard", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--min-tokens", type=int, default=48)
    args = ap.parse_args()
    {"check": lambda: check(args), "report": lambda: report(args),
     "guard": lambda: guard(args, args.device),
     "pilot": lambda: run(args, args.device, True),
     "scan": lambda: run(args, args.device, False)}[args.stage]()


if __name__ == "__main__":
    main()
