#!/usr/bin/env python
"""E3 -- does the drift survive when the distance grows the way it really grows?

E1 and E2 both reached in and rewrote position ids.  That isolates position
perfectly, but it is not what happens in use: in use the distance between the
image and the current token grows because the model has *written more words*.
E3 asks the question that way, with no intervention on the model at all.

THE DESIGN
----------
Two prompts with word-for-word identical content, differing only in the order of
two blocks:

  A   [filler] [IMAGE] [question]      filler BEFORE the image
  B   [IMAGE] [filler] [question]      filler AFTER the image

Same tokens, same count, same image, same question, same answer position.  But
the filler in A sits before the image, so it leaves the image-to-question
distance untouched; in B it sits between them, so that distance grows by exactly
the filler length.  RoPE sees only differences, so A is positionally inert no
matter how long the filler gets, and B is not.

A is therefore the control for the filler's CONTENT.  Whatever the filler does by
merely being present happens in both and cancels in the difference.  What
survives is only what it does by being *between* the image and the question.

THE READOUT
-----------
E2's stimulus and estimator: one coloured square on grey, N patches off the
midline, "top half or bottom half" read off the two answer tokens' logits.
Sweeping the offset gives the model's perceived midline in patch units.  Do that
in both conditions at each filler length and take B minus A.

  flat                        -> distance does not move spatial judgement in use
  smooth growth               -> the general "image is further away" effect,
                                 which E2 already found and largely attributed to
                                 an axis that cannot reshape anything
  growth that DIPS BACK at
  filler lengths 8, 16, 24    -> the marching stripes: only a repeating pattern
                                 can return at one period

That last one is the signature E2 looked for and did not find, tested here
without touching a single position id.

THE DETAIL THAT WOULD SINK IT
-----------------------------
The prediction has a period of eight TOKENS, so the filler length must be counted
in tokens and not words.  If eight words tokenise to nine tokens the sweep drifts
out of phase with the very periodicity being tested and the result looks like a
null.  So the filler unit is chosen by checking, against this tokenizer, that
repeating it n times gives exactly n tokens for every n in the sweep -- and the
scan then asserts, from the model's own position ids, that condition B's
image-to-answer distance exceeds condition A's by exactly the filler length.
Neither is assumed.

  python rope_phase_e3.py --stage check  --out-dir DIR   # CPU, tokenizer only
  python rope_phase_e3.py --stage scan   --out-dir DIR
  python rope_phase_e3.py --stage report --out-dir DIR
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
import rope_phase_e2 as E2         # noqa: E402

SELF_CONTAINED_STIMULI = True   # builds its own images; no dataset needed

# Candidates for a filler unit; the first that repeats token-exactly wins.
FILLER_UNITS = (" word", " item", " thing", " apple", " seven")


def pick_filler_unit(tokenizer, max_n: int, candidates=FILLER_UNITS) -> str:
    """A unit whose n-fold repetition is exactly n tokens, for every n in the sweep.

    Repeated words do not always tokenise one-to-one -- merges and leading-space
    variants both break it -- and a filler that is off by one token puts the whole
    sweep out of phase with the period being measured.
    """
    for unit in candidates:
        if all(len(tokenizer.encode(unit * n, add_special_tokens=False)) == n
               for n in range(1, max_n + 1)):
            return unit
    raise SystemExit(
        f"no candidate in {candidates} repeats token-exactly up to {max_n}; "
        "add one, or the filler length will not equal the token count")


def build_case(processor, image, filler: str, after_image: bool, device):
    """Condition A (after_image=False) or B (after_image=True)."""
    blocks = ([{"type": "image"}, {"type": "text", "text": filler},
               {"type": "text", "text": E2.QUESTION}] if after_image else
              [{"type": "text", "text": filler}, {"type": "image"},
               {"type": "text", "text": E2.QUESTION}])
    text = processor.apply_chat_template([{"role": "user", "content": blocks}],
                                         tokenize=False, add_generation_prompt=True)
    text = text + E2.ANSWER_PREFIX
    return processor(text=[text], images=[[image]], return_tensors="pt",
                     padding=True, add_special_tokens=False).to(device)


def image_to_answer_distance(model, case, image_token_id: int) -> int:
    """Position-id distance from the image anchor to the answer position.

    Read off the model's own get_rope_index rather than counted by hand, because
    the whole design rests on this differing between the two conditions by exactly
    the filler length.
    """
    ids = case["input_ids"]
    img_cols = (ids[0] == image_token_id).nonzero(as_tuple=True)[0]
    pos, _ = (model.model if hasattr(model, "model") else model).get_rope_index(
        ids, case["mm_token_type_ids"], image_grid_thw=case.get("image_grid_thw"),
        video_grid_thw=None, attention_mask=case.get("attention_mask"))
    return int(pos[0, 0, -1].item() - pos[0, 0, img_cols[0]].item())


# ---------------------------------------------------------------------------
def check(args):
    """CPU: does the filler tokenise exactly, and are the two conditions the same text?"""
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.base_model, padding_side="left")
    tok = processor.tokenizer
    ns = list(range(0, args.max_filler + 1))
    unit = pick_filler_unit(tok, max(ns))
    print(f"[e3] filler unit {unit!r} repeats token-exactly to {max(ns)}")

    ids = E2.answer_token_ids(processor)
    print(f"[e3] answer tokens single-token: {ids}")

    bad = 0
    for n in ns:
        filler = unit * n
        assert len(tok.encode(filler, add_special_tokens=False)) == n, n
        texts = {}
        for after in (False, True):
            blocks = ([{"type": "image"}, {"type": "text", "text": filler},
                       {"type": "text", "text": E2.QUESTION}] if after else
                      [{"type": "text", "text": filler}, {"type": "image"},
                       {"type": "text", "text": E2.QUESTION}])
            t = processor.apply_chat_template([{"role": "user", "content": blocks}],
                                              tokenize=False, add_generation_prompt=True)
            texts[after] = tok.encode(t + E2.ANSWER_PREFIX, add_special_tokens=False)
        same_len = len(texts[False]) == len(texts[True])
        same_multiset = sorted(texts[False]) == sorted(texts[True])
        if not (same_len and same_multiset):
            bad += 1
            if bad <= 3:
                print(f"[e3] n={n}: lengths {len(texts[False])} vs {len(texts[True])}, "
                      f"same multiset={same_multiset}")
    if bad:
        print(f"[e3] WARNING: {bad}/{len(ns)} filler lengths do not give the two "
              f"conditions an identical token multiset. The content control is then "
              f"imperfect and the B-A difference is not purely positional.")
    else:
        print(f"[e3] all {len(ns)} lengths: both conditions are the same tokens reordered")
    print("[e3] check passed" if not bad else "[e3] check found problems")


@torch.no_grad()
def scan(args, device):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    offsets = [float(x) for x in args.offsets.split(",")]
    colours = [c for c in args.colours.split(",") if c]
    ns = list(range(0, args.max_filler + 1, args.filler_step))

    processor, model = RP.load_model(args.base_model, args.adapter or None, device)
    tok = processor.tokenizer
    unit = pick_filler_unit(tok, max(ns))
    ids = E2.answer_token_ids(processor)
    image_token_id = int(getattr(model.config, "image_token_id", None) or RP.IMAGE_TOKEN_ID)
    patch_px = int(model.config.vision_config.patch_size) * \
        int(model.config.vision_config.spatial_merge_size)
    print(f"[e3] filler unit {unit!r}; lengths {ns}; one patch = {patch_px} px; "
          f"{len(offsets)} offsets x {len(colours)} colours x 2 conditions", flush=True)

    ev = np.zeros((len(colours), len(offsets), 2, len(ns)), dtype=np.float64)
    dists = np.zeros((2, len(ns)), dtype=np.int64)
    for ni, n in enumerate(ns):
        filler = unit * n
        for ai, after in enumerate((False, True)):
            for oi, off in enumerate(offsets):
                for ci, colour in enumerate(colours):
                    image = E2.make_image(args.image_side, off, colour, patch_px,
                                          args.square_px)
                    case = build_case(processor, image, filler, after, device)
                    if oi == 0 and ci == 0:
                        dists[ai, ni] = image_to_answer_distance(model, case, image_token_id)
                    ev[ci, oi, ai, ni] = E2.evidence(model(**case).logits, ids)
        gap = int(dists[1, ni] - dists[0, ni])
        # The design in one assertion: B must be exactly `n` further from the image.
        if gap != n:
            raise SystemExit(
                f"filler length {n} moved the image-to-answer distance by {gap}, not {n}. "
                f"The sweep would be out of phase with the period being tested; fix the "
                f"filler before trusting anything.")
        print(f"[e3] filler {n:>3d} tokens: distance A={dists[0, ni]} B={dists[1, ni]} "
              f"(+{gap}) ok", flush=True)

    meta = {"offsets": offsets, "colours": colours, "ns": ns, "unit": unit,
            "patch_px": patch_px, "image_side": args.image_side,
            "square_px": args.square_px, "base_model": args.base_model,
            "distances": dists.tolist()}
    np.savez(out_dir / "scan.npz",
             __meta__=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8), ev=ev)
    print(f"[e3] wrote {out_dir / 'scan.npz'}", flush=True)


def report(args):
    out_dir = Path(args.out_dir)
    z = np.load(out_dir / "scan.npz", allow_pickle=False)
    m = json.loads(bytes(z["__meta__"]).decode())
    ev, offsets, ns = z["ev"], np.asarray(m["offsets"]), m["ns"]

    mid = np.zeros((2, len(ns)))
    for ai in range(2):
        for ni in range(len(ns)):
            mid[ai, ni] = E2.crossing(offsets, ev[:, :, ai, ni].mean(axis=0))

    lines = []
    P = lines.append
    P("=" * 78)
    P("E3 -- distance grown with real tokens, not by rewriting position ids")
    P("=" * 78)
    P(f"model   : {m['base_model']}")
    P(f"filler  : {m['unit']!r} repeated; lengths {ns} tokens")
    P(f"stimulus: {m['image_side']}px canvas, 1 patch = {m['patch_px']}px, "
      f"colours {m['colours']}")
    P("")
    P("PERCEIVED MIDLINE (patches)")
    P("  filler tokens : " + " ".join(f"{n:>6d}" for n in ns))
    P("  A before image: " + " ".join(f"{v:>6.2f}" if np.isfinite(v) else "   nan"
                                      for v in mid[0]))
    P("  B after image : " + " ".join(f"{v:>6.2f}" if np.isfinite(v) else "   nan"
                                      for v in mid[1]))
    P("")
    P("B MINUS A (patches) -- the filler's content cancels; only its POSITION remains")
    P("  filler tokens : " + " ".join(f"{n:>6d}" for n in ns))
    P("  difference    : " + " ".join(f"{mid[1, i] - mid[0, i]:>+6.2f}" for i in range(len(ns))))
    P("")
    P("  A on its own is the content control and should be flat: the filler sits")
    P("  before the image, so it cannot change the image-to-question distance.")
    P("  Drift in A is the filler's meaning acting on the judgement, and subtracting")
    P("  it is the point of the design.")
    P("")
    P("READING THE DIFFERENCE")
    P("  flat                     -> distance does not move spatial judgement in use")
    P("  smooth growth            -> the general distance effect (what E2 found)")
    P("  dips back at 8, 16, 24   -> the marching stripes, with no position ids touched")
    text = "\n".join(lines)
    print(text)
    (out_dir / "report.txt").write_text(text + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["check", "scan", "report"], required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--base-model", default=RP.DEFAULT_MODEL)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--image-side", type=int, default=768)
    ap.add_argument("--square-px", type=int, default=64)
    ap.add_argument("--offsets", default="-2.5,-2,-1.5,-1.25,-1,-0.75,-0.5,-0.25,0,0.5,1",
                    help="comma-separated, in patches; use --offsets=-2,... form")
    ap.add_argument("--colours", default="red,blue,green,yellow")
    ap.add_argument("--max-filler", type=int, default=24)
    ap.add_argument("--filler-step", type=int, default=1)
    args = ap.parse_args()
    {"check": lambda: check(args), "report": lambda: report(args),
     "scan": lambda: scan(args, args.device)}[args.stage]()


if __name__ == "__main__":
    main()
