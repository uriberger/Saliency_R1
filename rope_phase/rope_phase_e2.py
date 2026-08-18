#!/usr/bin/env python
"""E2 -- does the drift move where the model thinks something IS?

E1 established that the positional overlay reshapes attention enormously and does
not change what the model says on ordinary prompts.  But ordinary prompts never
ask the model to care about one or two patches, and one or two patches is the
size of the drift.  E2 asks the question on a task built to be maximally
sensitive to exactly that.

THE TASK
--------
A grey canvas with one coloured square on it, centred horizontally, sitting a
controlled number of PATCHES above or below the horizontal midline.  The model is
asked whether the square is in the top half or the bottom half, and instead of
letting it generate we read the two answer tokens' logits directly.

The offset is the knob.  Far from the midline the answer is obvious and no
perturbation will touch it.  Near the midline the model is near-indifferent --
and E1 showed that near-indifference is the only place small perturbations change
anything.  Sweeping the offset traces out a psychometric curve, and the quantity
of interest is where that curve CROSSES ZERO: the midline as the model perceives
it, in patch units.

  If the drift moves where the model thinks things are, the crossing point should
  move with the gap -- and, since the overlay wraps every 8 positions, move
  PERIODICALLY rather than monotonically.

That is a displacement measured in the same units as the drift itself, on a task
where a one-patch error is the whole answer.  It is a far more sensitive
instrument than either fluency or a flip count over prose.

WHY SYNTHETIC RATHER THAN A GROUNDING BENCHMARK
-----------------------------------------------
Natural pointing data mixes the effect being measured with recognition
difficulty, annotation slop and the model's own bias, all of which are larger
than one patch.  Here the ground truth is exact, the margin is a dial, and the
comparison is paired: the same image at the same offset, differing only in
position ids.  A null on this is informative; a null on RefCOCO would not be.
Real data is the follow-up, not the first cut.

ARMS
----
The same five as E1, reusing its tested construction: `null` moves every position
id and must do nothing; `t` moves an offset that is identical for every patch and
so cannot reshape anything; `hw` moves exactly the cross-modal spatial offsets;
`full` moves all three; `fix` freezes the tail's spatial coordinates.

STAGES
------
  pilot   (GPU, ~1 min) baseline curve only, no sweep.  Run this FIRST: if the
          model is saturated -- always "top", or at chance everywhere -- there is
          no crossing point to move and the full sweep would measure nothing.
  scan    (GPU) the full offset x arm x gap sweep.
  report  (CPU) crossing points per arm and gap.

  python rope_phase_e2.py --stage pilot  --out-dir DIR
  python rope_phase_e2.py --stage scan   --out-dir DIR
  python rope_phase_e2.py --stage report --out-dir DIR
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rope_phase_probe as RP      # noqa: E402
import rope_phase_e1 as E1         # noqa: E402

QUESTION = ("Is the coloured square in the top half or the bottom half of the image? "
            "Answer with exactly one word.")
ANSWER_PREFIX = "The square is in the"
COLOURS = {"red": (200, 40, 40), "blue": (40, 80, 200),
           "green": (40, 160, 60), "yellow": (220, 190, 40)}


# ---------------------------------------------------------------------------
def make_image(side: int, offset_patches: float, colour: str, patch_px: int,
               square_px: int):
    """Grey canvas, one square, centred horizontally, offset vertically.

    Offset is in PATCHES and positive means DOWN, matching the row axis, so a
    perceived-midline shift can be compared with the drift without converting
    units in the head.
    """
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (side, side), (128, 128, 128))
    d = ImageDraw.Draw(img)
    cx = side // 2
    cy = side / 2 + offset_patches * patch_px
    h = square_px / 2
    d.rectangle([cx - h, cy - h, cx + h, cy + h], fill=COLOURS[colour])
    return img


def answer_token_ids(processor):
    """Single-token ids for the two answers, or fail loudly.

    If either answer is more than one token the first token may not distinguish
    them and the readout would be measuring the wrong thing.
    """
    tok = processor.tokenizer
    out = {}
    for word in ("top", "bottom"):
        ids = tok.encode(" " + word, add_special_tokens=False)
        if len(ids) != 1:
            raise SystemExit(f"' {word}' is {len(ids)} tokens {ids}; pick different "
                             f"answer words or read a multi-token span")
        out[word] = ids[0]
    return out


def build_case(processor, image, device):
    """Prompt ending mid-sentence, so the very next token is the answer."""
    messages = [{"role": "user",
                 "content": [{"type": "image"}, {"type": "text", "text": QUESTION}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    text = text + ANSWER_PREFIX
    return processor(text=[text], images=[[image]], return_tensors="pt",
                     padding=True, add_special_tokens=False).to(device)


def evidence(logits, ids: dict) -> float:
    """log P(top) - log P(bottom) at the answer position.

    Positive means the model says top.  A difference of logits, so the softmax
    normaliser cancels and nothing depends on the rest of the vocabulary.
    """
    row = logits[0, -1].float()
    return float((row[ids["top"]] - row[ids["bottom"]]).item())


def crossing(offsets: np.ndarray, ev: np.ndarray) -> float:
    """Offset in patches where the evidence changes sign -- the perceived midline.

    Linear interpolation between the two offsets that straddle zero.  Returns nan
    if the curve never crosses, which is the saturated case the pilot exists to
    catch.
    """
    order = np.argsort(offsets)
    x, y = np.asarray(offsets)[order], np.asarray(ev)[order]
    for i in range(len(x) - 1):
        if (y[i] > 0) != (y[i + 1] > 0):
            if y[i + 1] == y[i]:
                return float(x[i])
            return float(x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i]))
    return float("nan")


# ---------------------------------------------------------------------------
@torch.no_grad()
def run(args, device, pilot: bool):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    offsets = [float(x) for x in args.offsets.split(",")]
    colours = [c for c in args.colours.split(",") if c]
    arms = ["null"] if pilot else [a for a in args.arms.split(",") if a]
    deltas = [0] if pilot else E1.parse_deltas(args.deltas)

    processor, model = RP.load_model(args.base_model, args.adapter or None, device)
    ids = answer_token_ids(processor)
    image_token_id = int(getattr(model.config, "image_token_id", None) or RP.IMAGE_TOKEN_ID)
    merge = int(model.config.vision_config.spatial_merge_size)
    patch_px = int(model.config.vision_config.patch_size) * merge
    print(f"[e2] one patch = {patch_px} px; offsets {offsets} patches; "
          f"{len(colours)} colours x {len(arms)} arms x {len(deltas)} gaps", flush=True)

    ev = np.zeros((len(colours), len(offsets), len(arms), len(deltas)), dtype=np.float64)
    for oi, off in enumerate(offsets):
        for ci, colour in enumerate(colours):
            image = make_image(args.image_side, off, colour, patch_px, args.square_px)
            case = build_case(processor, image, device)
            input_ids = case["input_ids"]
            img_cols = (input_ids[0] == image_token_id).nonzero(as_tuple=True)[0]
            tail_start = int(img_cols[-1].item()) + 1
            base_pos, _ = (model.model if hasattr(model, "model") else model).get_rope_index(
                input_ids, case["mm_token_type_ids"],
                image_grid_thw=case.get("image_grid_thw"), video_grid_thw=None,
                attention_mask=case.get("attention_mask"))
            for ai, arm in enumerate(arms):
                for di, delta in enumerate(deltas):
                    pos = E1.build_position_ids(base_pos, tail_start, arm, delta)
                    o = model(**case, position_ids=pos)
                    ev[ci, oi, ai, di] = evidence(o.logits, ids)
                    del o
        print(f"[e2] offset {off:+.1f} done", flush=True)

    meta = {"offsets": offsets, "colours": colours, "arms": arms, "deltas": deltas,
            "patch_px": patch_px, "image_side": args.image_side,
            "square_px": args.square_px, "base_model": args.base_model, "pilot": pilot}
    name = "pilot.npz" if pilot else "scan.npz"
    np.savez(out_dir / name,
             __meta__=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8), ev=ev)
    print(f"[e2] wrote {out_dir / name}", flush=True)

    if pilot:
        P = print
        P("")
        P("BASELINE CURVE -- log P(top) - log P(bottom); positive means 'top'")
        P("  offset(patches): " + " ".join(f"{o:>7.1f}" for o in offsets))
        for ci, colour in enumerate(colours):
            P(f"  {colour:<15s}: " + " ".join(f"{ev[ci, oi, 0, 0]:>7.2f}" for oi in range(len(offsets))))
        P("  mean           : " + " ".join(f"{ev[:, oi, 0, 0].mean():>7.2f}" for oi in range(len(offsets))))
        x = np.asarray(offsets)
        cr = crossing(x, ev[:, :, 0, 0].mean(axis=0))
        P("")
        P(f"  perceived midline: {cr:+.3f} patches" if np.isfinite(cr)
          else "  perceived midline: NEVER CROSSES -- the model is saturated on this task")
        P("  Usable only if the curve changes sign and is not saturated at the offsets")
        P("  nearest zero: the whole experiment is whether this crossing point MOVES.")


def report(args):
    out_dir = Path(args.out_dir)
    z = np.load(out_dir / "scan.npz", allow_pickle=False)
    m = json.loads(bytes(z["__meta__"]).decode())
    ev, offsets = z["ev"], np.asarray(m["offsets"])
    arms, deltas = m["arms"], m["deltas"]

    lines = []
    P = lines.append
    P("=" * 78)
    P("E2 -- does the drift move the perceived midline?")
    P("=" * 78)
    P(f"model   : {m['base_model']}")
    P(f"stimulus: {m['image_side']}px canvas, {m['square_px']}px square, "
      f"1 patch = {m['patch_px']}px, colours {m['colours']}")
    P(f"offsets : {list(offsets)} patches")
    P("")
    P("PERCEIVED MIDLINE (patches; where log P(top) - log P(bottom) crosses zero)")
    P("Averaged over colours.  The drift is 1-2 patches, so a real effect is visible")
    P("at this scale; and it should be PERIODIC in the gap, not monotone.")
    P("    gap      : " + " ".join(f"{d:>6d}" for d in deltas))
    base = {}
    for ai, arm in enumerate(arms):
        cr = [crossing(offsets, ev[:, :, ai, di].mean(axis=0)) for di in range(len(deltas))]
        base[arm] = cr
        P(f"    {arm:8s} : " + " ".join(f"{c:>6.2f}" if np.isfinite(c) else "   nan" for c in cr))
    P("")
    P("SHIFT FROM GAP 0 (patches)")
    P("    gap      : " + " ".join(f"{d:>6d}" for d in deltas))
    for arm in arms:
        cr = np.asarray(base[arm])
        P(f"    {arm:8s} : " + " ".join(f"{c - cr[0]:>+6.2f}" if np.isfinite(c) else "   nan"
                                        for c in cr))
    P("")
    P("READING IT")
    P("  `null` is the floor: it changes nothing mathematically, so whatever it moves")
    P("  by is arithmetic noise and nothing smaller than that is a result.")
    P("  `t` cannot reshape the profile, so it should sit with `null`.")
    P("  `hw` is the hypothesis.  A shift comparable to the measured drift (1-2")
    P("  patches for the affected heads) and periodic in the gap would show that the")
    P("  drift moves where the model thinks things are.  A shift indistinguishable")
    P("  from `null` closes the question the other way, on the most sensitive task")
    P("  available: this stimulus has nothing in it BUT position.")
    text = "\n".join(lines)
    print(text)
    (out_dir / "report.txt").write_text(text + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["pilot", "scan", "report"], required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--base-model", default=RP.DEFAULT_MODEL)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--image-side", type=int, default=768)
    ap.add_argument("--square-px", type=int, default=64)
    # Pass this as --offsets=-2,-1,0 (equals form).  A space-separated value that
    # starts with a minus sign is read by argparse as another flag, and the job
    # dies at startup after loading the model.
    ap.add_argument("--offsets", default="-4,-2,-1,-0.5,0,0.5,1,2,4",
                    help="comma-separated, in patches; use --offsets=-2,... form")
    ap.add_argument("--colours", default="red,blue,green,yellow")
    ap.add_argument("--arms", default="null,t,hw,full,fix")
    ap.add_argument("--deltas", default="0-16")
    args = ap.parse_args()
    if args.stage == "report":
        report(args)
    else:
        run(args, args.device, pilot=args.stage == "pilot")


if __name__ == "__main__":
    main()
