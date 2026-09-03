#!/usr/bin/env python
"""Does Grounding-DINO read the reasoning step at all, or just the picture?

`step_box_similarity.py` shows, from data already on disk, that two steps of one chain get
almost the same boxes -- and, decisively, no more alike than two steps of two DIFFERENT
chains about the same picture. That says the sentence is not doing the work. It does not
say how little the sentence is doing, because every text compared there was a real
sentence written by the model about that image, and those are all somewhat alike.

This script settles it by re-grounding the SAME images with texts that are deliberately
wrong, empty, or scrambled, and asking how much the boxes move. It needs a GPU because it
runs Grounding-DINO, but nothing else: no VLM, no vLLM, no generation. Every image, every
step sentence and every stored mask comes out of an existing `probe_merged.json`.

The variants, all scored against the step's REAL mask:

  real      the sentence exactly as the reward sends it. Must reproduce the stored boxes;
            it is the check that this script is grounding the same way training did.
  nouns     the sentence's content words, comma-joined -- the phrase-list format
            Grounding-DINO was actually built for.
  shuffled  the same words in random order. Same vocabulary, no syntax.
  question  the sample's question instead of the step.
  foreign   a step sentence from a DIFFERENT image's chain. Right format, wrong content.
  generic   the single word "object". No image-specific content whatsoever.
  empty     "." -- nothing to ground at all.

Read `generic` first. If grounding a picture on the word "object" already lands most of
the real mask, then the per-step reward was never per-step: DINO was drawing the same
blob every time and the sentence was decoration.

Run (on a GPU node, one card is plenty):

    python dino_text_sensitivity.py \
        outputs/overlap_probe/20260809-021810-crossrun-val_natural-plus/probe_merged.json \
        --models base_coldstart --out-dir outputs/dino_text_sensitivity

`--limit-steps` caps the work; the default covers everything the file has.
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
SEED = 20260903


# ---------------------------------------------------------------------------
# the reward's own DINO path, loaded exactly the way overlap_probe loads it
# ---------------------------------------------------------------------------
def _load_module(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _stub_rewards_package():
    """overlap_rewards does `from . import roll_null`, which needs a parent package."""
    for _n, _p in (("trl", REPO / "trl"), ("trl.rewards", REPO / "trl" / "rewards")):
        if _n not in sys.modules:
            _m = types.ModuleType(_n)
            _m.__path__ = [str(_p)]
            sys.modules[_n] = _m


_stub_rewards_package()
OREW = _load_module("trl.rewards.overlap_rewards", "trl/rewards/overlap_rewards.py")

# The similarity measures and the metric copies live in the offline script; reuse them so
# the two reports are on one scale and there is one definition of `closeness`.
SBS = _load_module("_sbs", "step_box_similarity.py")


VARIANTS = ("real", "nouns", "shuffled", "question", "foreign", "generic", "empty")


def make_texts(step_text: str, question: str, foreign_text: str, rng) -> dict[str, str]:
    words = SBS.content_words(step_text)
    toks = step_text.split()
    rng.shuffle(toks)
    return {
        "real": step_text,
        "nouns": ", ".join(sorted(words)) if words else "object",
        "shuffled": " ".join(toks),
        "question": question or "object",
        "foreign": foreign_text or "object",
        "generic": "object",
        "empty": ".",
    }


# ---------------------------------------------------------------------------
def collect(path: str, keep_models, limit_steps: int, rng):
    """-> (list of step records, probe dir). One record per step, with its stored mask."""
    d = json.load(open(path))
    cfg = d["config"]
    if not cfg.get("store_maps"):
        raise SystemExit(f"{path}: store_maps was off, no masks stored")
    root = Path(path).parent

    recs = []
    for mk, mv in d["models"].items():
        if keep_models and mk not in keep_models:
            continue
        for s in mv["samples"]:
            img = s.get("image_file")
            if not img or not (root / img).exists():
                continue
            for ci, c in enumerate(s["completions"]):
                for st in c.get("observe_steps", []):
                    if not st.get("mask_q") or not st.get("grid") or not st.get("map_q"):
                        continue
                    gh, gw = st["grid"]
                    recs.append({
                        "model": mk,
                        "image": str(root / img),
                        "image_key": img,
                        "question": s.get("question", ""),
                        "text": st.get("text", ""),
                        "gh": gh, "gw": gw,
                        "mask": SBS.decode_mask(st["mask_q"], gh, gw),
                        "smap": SBS.decode_map(st["map_q"], gh, gw),
                        "boxes_stored": st.get("boxes_kept") or [],
                    })
    if not recs:
        raise SystemExit("no steps with stored masks")

    if limit_steps and len(recs) > limit_steps:
        pick = rng.choice(len(recs), size=limit_steps, replace=False)
        recs = [recs[i] for i in sorted(pick.tolist())]

    # `foreign` needs a sentence from a different image, fixed per record so a rerun with
    # the same seed grounds the same pairs.
    keys = [r["image_key"] for r in recs]
    for i, r in enumerate(recs):
        for _try in range(32):
            j = int(rng.integers(len(recs)))
            if keys[j] != r["image_key"]:
                r["foreign_text"] = recs[j]["text"]
                break
        else:
            r["foreign_text"] = "object"
    return recs, cfg


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("merged", help="probe_merged.json written by overlap_probe --store-maps")
    ap.add_argument("--models", default="base_coldstart",
                    help="comma-separated model keys, or 'all'")
    ap.add_argument("--out-dir", default="outputs/dino_text_sensitivity")
    ap.add_argument("--limit-steps", type=int, default=0, help="0 = every step in the file")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-image-side", type=int, default=512,
                    help="must match the probe's MAX_IMAGE_SIDE, or `real` will not "
                         "reproduce the stored boxes")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    from PIL import Image

    rng = np.random.default_rng(args.seed)
    keep = None if args.models == "all" else set(args.models.split(","))
    recs, cfg = collect(args.merged, keep, args.limit_steps, rng)

    # Ground exactly as the run being analysed did.
    OREW.configure(box_threshold=cfg.get("box_threshold", 0.10),
                   max_box_area=cfg.get("max_box_area"),
                   max_union_area=cfg.get("max_union_area"),
                   dino_batch_size=args.batch_size)
    print(f"[dino_text_sensitivity] {len(recs)} steps x {len(VARIANTS)} variants "
          f"= {len(recs) * len(VARIANTS)} groundings; "
          f"box_threshold={OREW._CFG['box_threshold']} "
          f"max_box_area={OREW._CFG['max_box_area']}", flush=True)

    # One decoded copy per image, resized the way the probe resized it before DINO saw it.
    cache: dict[str, "Image.Image"] = {}

    def img_of(path):
        if path not in cache:
            im = Image.open(path)
            im = im.convert("RGB") if im.mode != "RGB" else im
            if args.max_image_side and max(im.size) > args.max_image_side:
                sc = args.max_image_side / max(im.size)
                im = im.resize((max(1, int(im.size[0] * sc)), max(1, int(im.size[1] * sc))))
            cache[path] = im
        return cache[path]

    texts_per_rec = [make_texts(r["text"], r["question"], r["foreign_text"], rng)
                     for r in recs]

    results = {v: [] for v in VARIANTS}
    for v in VARIANTS:
        imgs = [img_of(r["image"]) for r in recs]
        txts = [t[v] for t in texts_per_rec]
        print(f"[dino_text_sensitivity] grounding variant `{v}` ...", flush=True)
        boxes = OREW._dino_boxes(imgs, txts)
        results[v] = boxes

    # ---- score every variant against the step's real mask ----
    rows = []
    for i, r in enumerate(recs):
        gh, gw, real = r["gh"], r["gw"], r["mask"]
        row = {"model": r["model"], "image": r["image_key"], "text": r["text"]}
        for v in VARIANTS:
            b = results[v][i] or []
            kept = [x for x in b if OREW._box_area(x) <= (OREW._CFG.get("max_box_area") or 1e9)]
            m = OREW._union_mask(b, gh, gw)
            if m is None:
                row[v] = {"grounded": False, "n_boxes": len(kept)}
                continue
            o, c, best, z = SBS.closeness(real, m)
            bm, bh = SBS.box_set_match(r["boxes_stored"], kept)
            row[v] = {"grounded": True, "n_boxes": len(kept),
                      "union_frac": float(m.sum()) / m.size,
                      "iou": o, "closeness": z, "identical": bool(np.array_equal(real, m)),
                      "box_match": bm, "box_hit": bh,
                      "mean_in": SBS.m_mean_in(r["smap"], m),
                      "auroc": SBS.m_auroc(r["smap"], m)}
        row["real_mean_in"] = SBS.m_mean_in(r["smap"], real)
        row["real_union_frac"] = float(real.sum()) / real.size
        rows.append(row)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "rows.json"), "w") as f:
        json.dump(rows, f)

    # ---- report ----
    def col(v, k):
        return [r[v][k] for r in rows if r[v].get("grounded") and r[v].get(k) is not None]

    real_mi = np.array([r["real_mean_in"] for r in rows
                        if r["real_mean_in"] is not None], dtype=float)

    L = []
    P = L.append
    P("=" * 92)
    P("DOES GROUNDING-DINO READ THE REASONING STEP, OR JUST THE PICTURE?")
    P("=" * 92)
    P("")
    P(f"{len(rows)} steps from {len({r['image'] for r in rows})} images, "
      f"model(s) {args.models}.")
    P("Each variant re-grounds the SAME image with a different text and is compared to")
    P("the mask the reward actually used.")
    P("")
    P("  IoU        overlap with the real mask (1.0 = the same patches).")
    P("  closeness  IoU rescaled: 0 = unrelated given the two mask sizes, 1 = as")
    P("             identical as masks of those sizes can be.")
    P("  boxes      how many boxes came back (real mask's own count for reference).")
    P("  r(reward)  correlation across steps between the reward this variant would give")
    P("             and the reward the real sentence gave.")
    P("")
    P("  variant     grounded    boxes   union     IoU   closeness  identical  r(reward)")
    for v in VARIANTS:
        n_ok = sum(1 for r in rows if r[v].get("grounded"))
        if n_ok == 0:
            P(f"  {v:11s} {n_ok:8d}        -       -       -           -          -        -")
            continue
        mi = np.array([r[v]["mean_in"] if r[v].get("grounded") and r[v].get("mean_in") is not None
                       else np.nan for r in rows], dtype=float)
        base = np.array([r["real_mean_in"] if r["real_mean_in"] is not None else np.nan
                         for r in rows], dtype=float)
        ok = np.isfinite(mi) & np.isfinite(base)
        rr = float(np.corrcoef(mi[ok], base[ok])[0, 1]) if ok.sum() > 3 else float("nan")
        P(f"  {v:11s} {n_ok/len(rows):7.1%} {np.mean(col(v,'n_boxes')):8.1f} "
          f"{np.mean(col(v,'union_frac')):7.2f} {np.mean(col(v,'iou')):7.3f} "
          f"{np.mean(col(v,'closeness')):11.3f} {np.mean(col(v,'identical')):9.1%} "
          f"{rr:9.3f}")
    P("")
    P(f"  the real mask covers {np.mean([r['real_union_frac'] for r in rows]):.2f} of the "
      f"grid on average and scores mean_in {real_mi.mean():.4f}.")
    P("")
    P("  `real` is the control: its IoU with the stored mask should be 1.000 and")
    P("  `identical` 100%. Anything less means this script is not grounding the way the")
    P("  run did -- check --max-image-side and box_threshold before reading the rest.")
    txt = "\n".join(L)
    with open(os.path.join(args.out_dir, "report.txt"), "w") as f:
        f.write(txt + "\n")
    print(txt)
    print(f"\nwritten to {args.out_dir}/report.txt", file=sys.stderr)


if __name__ == "__main__":
    main()
