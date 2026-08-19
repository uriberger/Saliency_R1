#!/usr/bin/env python
"""E5 -- run the frozen cross-modal position during generation, and read the text.

Everything measured so far in this directory comes from two answer-token logits at
one position: E2, E3 and E4 all read log P(top) - log P(bottom) off a single
forward.  Not one number came from a completion the model actually wrote.  E5 asks
the question that has been deferred all along: with the drift frozen out, does the
model SAY anything different, and is it better?

WHY THIS NEEDS MORE THAN E4'S PATCH
-----------------------------------
E4 precomputes the frozen phase for a whole fixed sequence, which is exactly wrong
for decoding one token at a time -- it refuses a query length of 1 rather than
silently misaligning.  Here the frozen phase is rebuilt inside a wrapper on the text
rotary module, which the decoder calls once per forward with the real position ids,
so prefill and decode are handled by the same code.

Which rows to freeze is decided from the ids themselves rather than from a
remembered `tail_start`: a token belongs to the tail exactly when its t index
exceeds the image anchor.  Text before the image counts below the anchor, every
patch of the image carries the anchor itself, and text after it resumes above.
That rule cannot drift out of step with the cache the way an index would.

THE ARMS, AND WHY `null` IS THE ONE THAT MATTERS
------------------------------------------------
  none        untouched.
  null        a constant added to EVERY position id, image included.  RoPE reads
              only differences, so this is a mathematical no-op -- and it is the
              control the whole experiment rests on.  E1 measured that bf16
              arithmetic alone flips 1.2% of greedy tokens, and under greedy
              decoding one flipped token forks the rest of the completion.  So
              "the completions changed" is worth nothing without knowing how much
              they change for no reason at all.  This measures that.
  frozen:d0   the intervention, at E4's winner d0 = max(gh, gw).
  frozen:0    d0 = 0 puts the query's row counter inside the image's own rows,
              where no text token ever sits.  E4 found it the worst arm on both
              stimulus families and the most damaging to NLL, so it is the "known
              bad" end of the scale.

WHAT IS MEASURED
----------------
  divergence   whether the completion changes at all, and WHERE it first changes.
               The intervention is an exact no-op at d = d0 and grows with
               distance, so its first divergence should sit later than the floor's.
               If divergence looks the same as `null`, nothing has been shown.
  nll          teacher-forced NLL of the dataset's own reference answer.  Needs no
               judge and is paired per prompt.
  match        exact/substring match against the reference, on the subset whose
               reference is a single noun -- the only place string matching is
               defensible.
  length       the deciding test.  The drift accumulates with distance, so any
               real benefit must GROW with completion length.  Flat in length
               falsifies the attribution however good the average looks.

  python rope_phase_e5.py --stage gen    --out-dir DIR --dataset PATH
  python rope_phase_e5.py --stage report --out-dir DIR
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rope_phase_probe as RP      # noqa: E402
import rope_phase_e4 as E4         # noqa: E402

SELF_CONTAINED_STIMULI = False
SCHEMA = 1


# ---------------------------------------------------------------------------
class RotaryFreezer:
    """Rebuild the frozen phase on every forward, so decoding works.

    Wraps the text rotary module.  It is called once per forward with the real
    position ids -- shape [3, batch, seq] -- which is the one place that sees them
    in both prefill and decode, and early enough for every layer to use the result.
    """

    def __init__(self, model, ctl):
        self.rot = E4.find_text_rotary(model)
        self.ctl = ctl
        self.original = self.rot.forward
        self.arm, self.d0, self.anchor, self.delta = "none", 0, 0, 0
        rf = self

        def forward(x, position_ids):
            pos = position_ids
            if pos.ndim == 2:
                pos = pos[None].expand(3, -1, -1)
            if rf.arm == "null":
                # Every id moves together, so every difference is unchanged and the
                # logits owe the untouched ones up to arithmetic.
                return rf.original(x, pos + rf.delta)
            real = rf.original(x, pos)
            if rf.arm == "none":
                rf.ctl.enabled = False
                return real
            frozen = pos.clone()
            tail = pos[0] > rf.anchor          # [batch, seq]
            frozen[1][tail] = rf.anchor + rf.d0
            frozen[2][tail] = rf.anchor + rf.d0
            cos_f, sin_f = rf.original(x, frozen)
            rf.ctl.cos_f, rf.ctl.sin_f, rf.ctl.enabled = cos_f, sin_f, True
            return real

        self.rot.forward = forward

    def set(self, arm: str, anchor: int, d0: int = 0, delta: int = 0):
        self.arm, self.anchor, self.d0, self.delta = arm, anchor, d0, delta
        if arm in ("none", "null"):
            self.ctl.disarm()

    def restore(self):
        self.rot.forward = self.original


def parse_arms(spec: str):
    """"none,null:1000,frozen:24" -> [(label, kind, value), ...]"""
    out = []
    for part in [p for p in spec.split(",") if p]:
        if part == "none":
            out.append((part, "none", 0))
        elif part.startswith("null"):
            out.append((part, "null", int(part.split(":")[1]) if ":" in part else 1000))
        elif part.startswith("frozen:"):
            out.append((part, "frozen", int(part.split(":")[1])))
        else:
            raise SystemExit(f"unknown arm {part!r}")
    return out


def first_divergence(a: list[int], b: list[int]) -> int:
    """Index of the first differing token, or -1 if one is a prefix of the other."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return -1 if len(a) == len(b) else min(len(a), len(b))


def normalise(s: str) -> str:
    keep = "".join(c.lower() if c.isalnum() or c.isspace() else " " for c in s)
    return " ".join(keep.split())


def matches(completion: str, reference: str) -> bool:
    """Substring match on a normalised single-noun reference.

    Deliberately only used where the reference is one or two words: anything longer
    needs a judge, and a judge is a second model's opinion rather than a measurement.
    """
    return normalise(reference) in normalise(completion)


# ---------------------------------------------------------------------------
@torch.no_grad()
def gen(args, device):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    processor, model = RP.load_model(args.base_model, args.adapter or None, device,
                                     attn_impl="eager")
    image_token_id = int(getattr(model.config, "image_token_id", None) or RP.IMAGE_TOKEN_ID)
    inner = model.model if hasattr(model, "model") else model
    arms = parse_arms(args.arms)
    rows = RP.load_samples(args.dataset, args.n_cases, args.seed)
    ctl = E4.FrozenQueryAttention(model)
    ctl.disarm()
    freezer = RotaryFreezer(model, ctl)
    tok = processor.tokenizer

    results, t0 = [], time.time()
    path = out_dir / "gen.json"
    for ci, row in enumerate(rows):
        image = RP.square_image(row["image"], args.image_side)
        prompt = RP.build_prompt(processor, row["question"])
        inputs = processor(text=[prompt], images=[[image]], return_tensors="pt",
                           padding=True, padding_side="left",
                           add_special_tokens=False).to(device)
        prompt_len = inputs["input_ids"].shape[1]
        img_cols = (inputs["input_ids"][0] == image_token_id).nonzero(as_tuple=True)[0]
        if img_cols.numel() == 0:
            continue
        pos, _ = inner.get_rope_index(
            inputs["input_ids"], inputs.get("mm_token_type_ids"),
            image_grid_thw=inputs.get("image_grid_thw"), video_grid_thw=None,
            attention_mask=inputs.get("attention_mask"))
        anchor = int(pos[0, 0, img_cols[0]].item())
        ctl.img_cols = img_cols

        ref = (row.get("solution") or "").strip()
        rec = {"row_index": row["row_index"], "dataset": row.get("dataset", "?"),
               "question": row["question"], "reference": ref,
               "single_noun": 0 < len(ref.split()) <= 2, "anchor": anchor, "arms": {}}

        for label, kind, val in arms:
            freezer.set(kind if kind != "frozen" else "frozen", anchor,
                        d0=val if kind == "frozen" else 0,
                        delta=val if kind == "null" else 0)
            ids = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens,
                                 pad_token_id=tok.pad_token_id)
            comp = ids[0, prompt_len:].tolist()
            if tok.eos_token_id in comp:
                comp = comp[: comp.index(tok.eos_token_id) + 1]

            # Reference NLL, teacher-forced under this same arm.  A separate forward
            # because the completion the model chose is not the reference.
            nll = float("nan")
            if ref:
                ref_ids = tok.encode(" " + ref, add_special_tokens=False)
                cat = torch.cat([inputs["input_ids"],
                                 torch.tensor([ref_ids], device=device)], dim=1)
                case = {"input_ids": cat,
                        "attention_mask": torch.ones_like(cat)}
                for k in ("pixel_values", "image_grid_thw"):
                    if k in inputs:
                        case[k] = inputs[k]
                if inputs.get("mm_token_type_ids") is not None:
                    case["mm_token_type_ids"] = torch.cat(
                        [inputs["mm_token_type_ids"],
                         torch.zeros(1, len(ref_ids), dtype=torch.long, device=device)], dim=1)
                lg = model(**case, use_cache=False).logits
                nll = float(torch.nn.functional.cross_entropy(
                    lg[0, prompt_len - 1: -1].float(),
                    cat[0, prompt_len:]).item())

            rec["arms"][label] = {"tokens": comp, "n_tokens": len(comp),
                                  "text": tok.decode(comp, skip_special_tokens=True),
                                  "ref_nll": nll}
        results.append(rec)
        el = time.time() - t0
        print(f"[e5] {len(results)}/{len(rows)} cases, {el/60:.1f} min, "
              f"~{el/len(results)*(len(rows)-len(results))/60:.1f} min left", flush=True)
        path.write_text(json.dumps(
            {"schema": SCHEMA, "arms": [a[0] for a in arms], "base_model": args.base_model,
             "dataset": args.dataset, "max_new_tokens": args.max_new_tokens,
             "cases": results}, indent=1))
    freezer.restore()
    ctl.restore()
    print(f"[e5] wrote {path}", flush=True)


# ---------------------------------------------------------------------------
def report(args):
    out_dir = Path(args.out_dir)
    data = json.loads((out_dir / "gen.json").read_text())
    arms, cases = data["arms"], data["cases"]
    base = "none"
    L, P = [], []
    P = L.append
    P("=" * 78)
    P("E5 -- the frozen cross-modal position, live during generation")
    P("=" * 78)
    P(f"model   : {data['base_model']}")
    P(f"data    : {data['dataset']}")
    P(f"cases   : {len(cases)}, greedy, max {data['max_new_tokens']} new tokens")
    P("")
    P("`null` adds a constant to every position id and is a mathematical no-op, so")
    P("whatever it moves is the floor: greedy decoding forks on a single bf16-flipped")
    P("token.  An arm that does not clearly beat `null` has not been shown to do")
    P("anything at all.")
    P("")

    P("--- DID THE TEXT CHANGE?")
    P(f"    {'arm':>12s} {'differs':>9s} {'1st diff':>10s} {'median':>8s} "
      f"{'len':>7s} {'d(len)':>8s}")
    stats = {}
    for arm in arms:
        div, firsts, dlen, lens = 0, [], [], []
        for c in cases:
            a, b = c["arms"][base]["tokens"], c["arms"][arm]["tokens"]
            i = first_divergence(a, b)
            lens.append(len(b))
            dlen.append(len(b) - len(a))
            if i != -1:
                div += 1
                firsts.append(i)
        stats[arm] = {"div": div / max(len(cases), 1),
                      "firsts": firsts, "dlen": float(np.mean(dlen)),
                      "len": float(np.mean(lens))}
        med = float(np.median(firsts)) if firsts else float("nan")
        P(f"    {arm:>12s} {100*stats[arm]['div']:>8.0f}% "
          f"{np.mean(firsts) if firsts else float('nan'):>10.1f} {med:>8.1f} "
          f"{stats[arm]['len']:>7.1f} {stats[arm]['dlen']:>+8.1f}")
    P("    'differs' = share of prompts whose completion is not identical to `none`.")
    P("    '1st diff' = mean token index of the first difference; the intervention is")
    P("    an exact no-op at d = d0 and grows with distance, so it should sit LATER")
    P("    than the floor's.  Equal to `null` on every column = nothing demonstrated.")
    P("")

    P("--- REFERENCE-ANSWER NLL, paired per prompt (lower is better)")
    P(f"    {'arm':>12s} {'NLL':>8s} {'vs none':>10s} {'+-':>7s} {'better':>8s}")
    for arm in arms:
        d = np.array([c["arms"][arm]["ref_nll"] - c["arms"][base]["ref_nll"]
                      for c in cases if np.isfinite(c["arms"][arm]["ref_nll"])])
        v = np.array([c["arms"][arm]["ref_nll"] for c in cases
                      if np.isfinite(c["arms"][arm]["ref_nll"])])
        se = float(d.std(ddof=1) / len(d) ** 0.5) if len(d) > 1 else float("nan")
        P(f"    {arm:>12s} {v.mean():>8.3f} {d.mean():>+10.4f} {se:>7.4f} "
          f"{100*float((d < 0).mean()):>7.0f}%")
    P("")

    sub = [c for c in cases if c["single_noun"] and c["reference"]]
    if sub:
        P(f"--- EXACT MATCH on the {len(sub)} prompts whose reference is 1-2 words")
        P(f"    {'arm':>12s} {'match':>8s}")
        for arm in arms:
            m = float(np.mean([matches(c["arms"][arm]["text"], c["reference"])
                               for c in sub]))
            P(f"    {arm:>12s} {100*m:>7.0f}%")
        P("")

    P("--- THE DECIDING TEST: does the effect grow with completion length?")
    P("    The drift accumulates with distance, so a real benefit must be bigger for")
    P("    long completions.  Flat in length falsifies the attribution however good")
    P("    the average looks.")
    lens = np.array([c["arms"][base]["n_tokens"] for c in cases])
    if len(lens) >= 8:
        cut = float(np.median(lens))
        P(f"    split at the median completion length, {cut:.0f} tokens")
        P(f"    {'arm':>12s} {'short dNLL':>12s} {'long dNLL':>12s} {'long-short':>12s}")
        for arm in arms:
            cell = []
            for sel in (lens <= cut, lens > cut):
                d = np.array([c["arms"][arm]["ref_nll"] - c["arms"][base]["ref_nll"]
                              for c, s in zip(cases, sel)
                              if s and np.isfinite(c["arms"][arm]["ref_nll"])])
                cell.append(d.mean() if len(d) else float("nan"))
            P(f"    {arm:>12s} {cell[0]:>+12.4f} {cell[1]:>+12.4f} "
              f"{cell[1]-cell[0]:>+12.4f}")
    P("")

    P("READING IT")
    P("  Three outcomes.  If every arm sits on `null`, the intervention does not")
    P("  reach generated text and the 0.36 patches E4 measured stay locked inside a")
    P("  logit comparison.  If `frozen` beats `null` on divergence but not on NLL, it")
    P("  changes the text without improving it.  If it beats `null` on NLL AND the")
    P("  gain grows with length, the mechanism reaches the output.")
    text = "\n".join(L)
    print(text)
    (out_dir / "report.txt").write_text(text + "\n")

    # Qualitative: the pairs that diverged earliest, which is where the intervention
    # had the most room to change the rest of the completion.
    arm = args.show if args.show in arms else (arms[-1] if arms else base)
    pairs = []
    for c in cases:
        i = first_divergence(c["arms"][base]["tokens"], c["arms"][arm]["tokens"])
        if i != -1:
            pairs.append((i, c))
    pairs.sort(key=lambda p: p[0])
    lines = [f"E5 qualitative: `none` vs `{arm}`, {len(pairs)} diverging cases, "
             f"earliest divergence first\n"]
    for i, c in pairs[: args.n_show]:
        lines += [
            "=" * 78,
            f"[{c['dataset']}] {c['question']}",
            f"reference: {c['reference']}",
            f"first divergence at token {i}",
            f"--- none      : {c['arms'][base]['text']}",
            f"--- {arm:<10s}: {c['arms'][arm]['text']}",
            "",
        ]
    (out_dir / "qualitative.txt").write_text("\n".join(lines) + "\n")
    print(f"\n[e5] wrote {out_dir/'report.txt'} and {out_dir/'qualitative.txt'}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["gen", "report"], required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--base-model", default=RP.DEFAULT_MODEL)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--dataset", default=RP.DEFAULT_DATASET)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-cases", type=int, default=60)
    ap.add_argument("--arms", default="none,null:1000,frozen:24,frozen:0")
    ap.add_argument("--image-side", type=int, default=768)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--show", default="frozen:24")
    ap.add_argument("--n-show", type=int, default=25)
    args = ap.parse_args()
    gen(args, args.device) if args.stage == "gen" else report(args)


if __name__ == "__main__":
    main()
