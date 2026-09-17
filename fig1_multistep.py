#!/usr/bin/env python
"""Find the Figure-1 panel: one chain whose steps look at DIFFERENT places.

`fig1_step_referent.py` asks "did this step look at the thing it names?" and pairs steps
ACROSS models on a shared referent. This asks the other question, the one a Figure 1 is
actually a picture of:

    within ONE chain, are there two observe steps whose referents are DISJOINT, and does
    the model's own attention move from the first region to the second between them?

That is a crossover, not an overlap, and it needs a referent that is one object. At the
reward's own detector settings -- Grounding-DINO over the step's whole sentence at
box_threshold 0.1, per-box area cap 0.5 -- the union covers a median 52% of the patch
grid (14 boxes), and in the 20-sample sviz-3models run 0 of 59 within-chain step pairs
came out below IoU 0.1. There is nothing to find at those settings, so this script keeps
TWO referents per step and reports both:

  reward   every box the reward would have kept, unioned. The training instrument.
           Quoted so the panel's caption can say what the reward saw.
  tight    only the boxes within --tight-score-frac of the step's best DINO score, each
           under --tight-max-box-area. One object, or one object and its parts. This is
           a FIGURE definition, not a metric: it is chosen to be drawable, and it is not
           what the reward optimised.

The crossover test, for a step pair (i, j) with disjoint tight referents R_i, R_j:

    margin = min( v2(map_i, R_i) - v2(map_i, R_j),
                  v2(map_j, R_j) - v2(map_j, R_i) )

`v2` is `overlap_rewards._mean_in_v2` (chance = 1.0). Taking the MIN means both steps
have to move the right way -- one step that fires everywhere cannot carry the pair. The
same margin is computed for every other model in the run against the SAME two regions,
maximised over all of that model's own step pairs, which is as generous as the comparison
can be made: if a model cannot produce the crossover with any pair of its own steps, that
is not an artefact of which step got matched to which.

Crossover rates over every qualifying pair are printed per model, because a hand-picked
panel means nothing without the rate it was picked out of.

    python fig1_multistep.py --run-dir outputs/saliency_viz/fig1ms-valnat \
        --ours ours --out outputs/fig1-multistep/valnat.json

One GPU for the detector (see launch_fig1_multistep_job.sh); CPU works and is ~15x slower.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import types
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent


def _load_module(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _stub_rewards_package():
    """Give `trl`/`trl.rewards` a __path__ so overlap_rewards' relative imports resolve.

    Same reason as overlap_probe.py and fig1_step_referent.py: importing the real
    trl/__init__.py would drag the whole training stack into a script that only needs
    the detector helpers and the three metrics.
    """
    for name, path in (("trl", REPO / "trl"), ("trl.rewards", REPO / "trl" / "rewards")):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = [str(path)]
            sys.modules[name] = mod


_stub_rewards_package()
OREW = _load_module("trl.rewards.overlap_rewards", "trl/rewards/overlap_rewards.py")


# ---------------------------------------------------------------------------
# Grounding-DINO, keeping the scores
# ---------------------------------------------------------------------------
def dino_scored(images, texts, box_threshold: float, batch_size: int = 8):
    """Like `overlap_rewards._dino_boxes_local`, but the per-box SCORE survives.

    The reward throws the scores away -- it unions everything above the threshold, so a
    rank within the survivors would mean nothing to it. The tight referent is exactly a
    rank within the survivors, so this cannot go through `_dino_boxes`.

    Returns, per item, a list of {box: [x1,y1,x2,y2] relative, score, label}.
    """
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    dev = _DINO.get("device")
    if _DINO.get("model") is None:
        dev = _DINO["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        _DINO["proc"] = AutoProcessor.from_pretrained(OREW.GROUNDING_DINO_HF_ID)
        _DINO["model"] = AutoModelForZeroShotObjectDetection.from_pretrained(
            OREW.GROUNDING_DINO_HF_ID).to(dev).eval()
        print(f"[dino] {OREW.GROUNDING_DINO_HF_ID} on {dev}", flush=True)
    proc, model = _DINO["proc"], _DINO["model"]

    prompts = [(t.strip() + ".") if not t.strip().endswith(".") else t.strip()
               for t in texts]
    out = [None] * len(images)
    start = 0
    while start < len(images):
        n = min(batch_size, len(images) - start)
        while True:
            try:
                imgs = images[start:start + n]
                inputs = proc(images=imgs, text=prompts[start:start + n],
                              return_tensors="pt", padding=True, truncation=True,
                              max_length=256).to(dev)
                with torch.no_grad():
                    res = model(**inputs)
                results = proc.post_process_grounded_object_detection(
                    res, inputs.input_ids, threshold=box_threshold,
                    text_threshold=box_threshold,
                    target_sizes=[(im.size[1], im.size[0]) for im in imgs])
                for j, r in enumerate(results):
                    w, h = imgs[j].size
                    labels = r.get("text_labels", r.get("labels"))
                    labels = ["" for _ in r["boxes"]] if labels is None else list(labels)
                    out[start + j] = [
                        {"box": [x1 / w, y1 / h, x2 / w, y2 / h],
                         "score": float(s), "label": str(lab)}
                        for (x1, y1, x2, y2), s, lab
                        in zip(r["boxes"].tolist(), r["scores"].tolist(), labels)]
                break
            except torch.cuda.OutOfMemoryError:
                if n == 1:
                    raise
                torch.cuda.empty_cache()
                n = max(1, n // 2)
                print(f"[dino] OOM at {start}; batch -> {n}", flush=True)
        start += n
    return out


_DINO: dict = {}


# ---------------------------------------------------------------------------
# referents
# ---------------------------------------------------------------------------
def raster(boxes, gh, gw):
    """Boolean patch-grid union of relative boxes; None if empty or everything.

    Identical rasterisation to `overlap_rewards._union_mask` (every box claims at least
    one row and column), so a tight mask and a reward mask are measured on the same
    footing and their IoU means what it looks like.
    """
    if not boxes:
        return None
    m = np.zeros((gh, gw), dtype=bool)
    for x1, y1, x2, y2 in boxes:
        r0 = max(0, int(y1 * gh))
        r1 = min(gh, max(r0 + 1, round(y2 * gh)))
        c0 = max(0, int(x1 * gw))
        c1 = min(gw, max(c0 + 1, round(x2 * gw)))
        m[r0:r1, c0:c1] = True
    n = int(m.sum())
    return None if n == 0 or n == gh * gw else m


def reward_referent(dets, gh, gw, max_box_area):
    """The mask the overlap reward would have scored this step against."""
    boxes = [d["box"] for d in dets if OREW._box_area(d["box"]) <= max_box_area]
    return raster(boxes, gh, gw), boxes


def tight_referent(dets, gh, gw, score_frac, max_box_area):
    """The step's best-grounded object: boxes within `score_frac` of the top score.

    Not the top box alone -- Grounding-DINO routinely returns the object and a part of
    it ("the man", "his jacket") at nearly the same score, and dropping the second makes
    the region an arbitrary half of the same thing. The score margin keeps both and still
    rejects the long tail that turns the union into a blob.
    """
    keep = [d for d in dets if OREW._box_area(d["box"]) <= max_box_area]
    if not keep:
        return None, []
    top = max(d["score"] for d in keep)
    boxes = [d["box"] for d in keep if d["score"] >= score_frac * top]
    return raster(boxes, gh, gw), boxes


def border_frac(smap):
    inner = np.zeros(smap.shape, dtype=bool)
    inner[1:-1, 1:-1] = True
    total = float(smap.sum())
    return 1.0 - float(smap[inner].sum()) / total if total > 0 else float("nan")


def score_one(smap, mask):
    return {"mean_in": OREW._mean_in(smap, mask),
            "mean_in_v2": OREW._mean_in_v2(smap, mask),
            "auroc": OREW._auroc(smap, mask),
            "area_frac": float(mask.sum()) / float(mask.size),
            "peak_in": bool(mask[np.unravel_index(np.argmax(smap), smap.shape)])}


# ---------------------------------------------------------------------------
# answers
# ---------------------------------------------------------------------------
# Lines base Qwen3-VL-8B-Instruct ends on that are ABOUT its answer rather than the
# answer. Taking the last line literally scores "This is my answer." against the gold
# string and marks a correct model wrong -- which is the artefact this whole comparison
# has to avoid, not commit.
_BOILERPLATE = re.compile(
    r"^\W*(this is my (answer|reasoning)|answer|final answer|in summary|conclusion)\W*$",
    re.IGNORECASE)


def extract_answer(text: str) -> tuple[str, str]:
    """-> (what to print, what to grade on).

    A cold-started chain is `<think> ... </think> ANSWER`, which is what the trainer's
    accuracy_reward parses, and there the two are the same string. Base Qwen3-VL-8B-
    Instruct has no think block at all (0/20 format_ok on val_natural), answers in prose
    and often signs off with a line about the answer rather than the answer; falling back
    to the whole completion -- the trainer's fallback -- would hand the grader four
    paragraphs, and falling back to the last line hands it the sign-off. So: drop the
    sign-off lines, print the last real one, and grade over the last two, which is where
    a "Therefore, ..." conclusion and its restatement both live.
    """
    text = text.replace("<|im_end|>", " ").strip()
    m = re.search(r"</think>\s*(.*)", text, re.DOTALL)
    if m and m.group(1).strip():
        return m.group(1).strip(), m.group(1).strip()
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(1).strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    real = [ln for ln in lines if not _BOILERPLATE.match(ln)]
    if not real:
        return (lines[-1] if lines else text), text
    return real[-1], " ".join(real[-2:])


def mcq_letter(text: str):
    """The option letter a free-form answer is choosing, or None.

    A bare `\\bA\\b` search is not it: prose contains the article "A", and on a
    five-option benchmark that alone would hand a wrong model a 1-in-5 credit. So the
    letter has to be in a position that means a choice -- the whole answer, a
    "the answer is X", or a parenthesised "(X)".
    """
    t = (text or "").strip()
    m = re.fullmatch(r"\W*([A-Ea-e])\W*", t)                  # the answer IS the letter
    if m:
        return m.group(1).upper()
    m = re.match(r"^\W*([A-E])\s*[.):,\-]", t)                # "C. In the upper left area"
    if m:
        return m.group(1)
    m = re.search(r"(?:answer|option|choice)\s*(?:is|are)?\s*[:\-]?\s*[*\(\[]*([A-E])\b(?!['\w])",
                  t, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.findall(r"[\(\[]([A-E])[\)\]]", t)
    return m[-1] if m else None


def grade(answer: str, gold: str) -> dict:
    """Strict is the trainer's rule; soft is the one a prose answer can pass.

    The trainer scores `answer.lower() == gold.lower()`, which a model that writes "The
    cup stands on top of a bathroom vanity." fails even when it is right. Reporting only
    the strict grade would make a verbose baseline look wrong for being verbose -- the
    same artefact as the LogicVista MCQ parser -- so both are carried and the caption
    quotes the soft one.

    A single-letter gold is a multiple-choice benchmark, where neither rule works: exact
    match fails on "The best answer is: C" and the substring rule fires on the article
    "A". Both grades then come from `mcq_letter`.
    """
    a = (answer or "").strip().lower().rstrip(".")
    g = (gold or "").strip().rstrip(".")
    if not g:
        return {"strict": None, "soft": None, "kind": "none"}
    if re.fullmatch(r"[A-Ea-e]", g):
        got = mcq_letter(answer)
        ok = got is not None and got.upper() == g.upper()
        return {"strict": ok, "soft": ok, "kind": "mcq", "parsed": got}
    g = g.lower()
    soft = bool(re.search(rf"(?<![a-z0-9]){re.escape(g)}(?![a-z0-9])", a))
    return {"strict": a == g, "soft": soft, "kind": "text"}


# ---------------------------------------------------------------------------
# collect
# ---------------------------------------------------------------------------
def collect(run_dir: Path, models: dict[str, str], maps: list[str]):
    """One record per (model, sample, step), with every requested map attached."""
    items = []
    for tag, sub in models.items():
        root = run_dir / sub / "samples"
        if not root.is_dir():
            raise SystemExit(f"no samples under {root}")
        for sname in sorted(p.name for p in root.iterdir() if p.is_dir()):
            sdir = root / sname
            if not (sdir / "meta.json").exists():
                continue
            meta = json.loads((sdir / "meta.json").read_text())
            if meta.get("dropped") or not (sdir / "maps.npz").exists():
                continue
            z = np.load(sdir / "maps.npz")
            have = [m for m in maps if m in z.files]
            if not have:
                continue
            gen = meta.get("generation", "")
            ans, span = extract_answer(gen)
            image = Image.open(sdir / "original.png").convert("RGB")
            for i, step in enumerate(meta["steps"]):
                items.append({
                    "model": tag, "sample": sname, "row": meta.get("row_index"),
                    "dataset": meta.get("dataset"), "step": i, "text": step["text"],
                    "question": meta.get("question", ""),
                    "gt_answer": meta.get("gt_answer", ""),
                    "answer": ans, "answer_span": span,
                    "grade": grade(span, str(meta.get("gt_answer", ""))),
                    "format_ok": meta.get("format_ok"), "n_steps": len(meta["steps"]),
                    "image": image, "sdir": str(sdir),
                    "maps": {m: np.clip(z[m][i], 0, None).astype(np.float64) for m in have},
                })
    return items


def pair_margin(rec_i, rec_j, mi, mj, key):
    """min over the two steps of (own region - other region), on map `key`.

    None if either map is missing or either v2 is undefined (an all-zero map).
    """
    try:
        a = OREW._mean_in_v2(rec_i["maps"][key], mi)
        b = OREW._mean_in_v2(rec_i["maps"][key], mj)
        c = OREW._mean_in_v2(rec_j["maps"][key], mj)
        d = OREW._mean_in_v2(rec_j["maps"][key], mi)
    except KeyError:
        return None
    if None in (a, b, c, d):
        return None
    return {"self_i": a, "other_i": b, "self_j": c, "other_j": d,
            "margin": min(a - b, c - d)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, action="append",
                    help="a saliency_viz output root; repeatable to pool several scans")
    ap.add_argument("--model", action="append", default=[], metavar="NAME=SUBDIR",
                    help="repeatable; default is every subdir holding samples/")
    ap.add_argument("--ours", default="ours", help="which model tag is the candidate")
    ap.add_argument("--maps", default="glimpse,direct")
    ap.add_argument("--rank-map", default="glimpse", help="which map ranks the panels")
    ap.add_argument("--out", required=True)
    ap.add_argument("--box-threshold", type=float, default=0.1,
                    help="the reward's detector threshold; both referents start here")
    ap.add_argument("--max-box-area", type=float, default=0.5, help="the reward's per-box cap")
    ap.add_argument("--tight-score-frac", type=float, default=0.8)
    ap.add_argument("--tight-max-box-area", type=float, default=0.35)
    ap.add_argument("--max-referent-area", type=float, default=0.25,
                    help="a tight referent bigger than this fraction of the grid is not one object")
    ap.add_argument("--max-pair-iou", type=float, default=0.05,
                    help="grid IoU below which two referents count as disjoint")
    ap.add_argument("--dino-batch-size", type=int, default=8)
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    maps = [m for m in args.maps.split(",") if m]
    run_dirs = [Path(r) for r in args.run_dir]
    if args.model:
        models = dict(m.split("=", 1) for m in args.model)
    else:
        models = {}
        for r in run_dirs:
            models.update({p.name: p.name for p in sorted(r.iterdir())
                           if (p / "samples").is_dir()})

    items = []
    for r in run_dirs:
        got = collect(r, models, maps)
        print(f"[collect] {r}: {len(got)} (model, sample, step) records", flush=True)
        for it in got:
            it["run"] = r.name
        items += got
    if not items:
        raise SystemExit("nothing collected")

    print(f"[dino] grounding {len(items)} step sentences at threshold {args.box_threshold} ...",
          flush=True)
    dets = dino_scored([it["image"] for it in items], [it["text"] for it in items],
                       args.box_threshold, args.dino_batch_size)

    for it, ds in zip(items, dets):
        gh, gw = next(iter(it["maps"].values())).shape
        it["grid"] = [gh, gw]
        it["rew_mask"], it["rew_boxes"] = reward_referent(ds, gh, gw, args.max_box_area)
        it["tight_mask"], it["tight_boxes"] = tight_referent(
            ds, gh, gw, args.tight_score_frac, args.tight_max_box_area)
        it["n_det"] = len(ds)
        it["scores"] = {}
        for key, mask in (("reward", it["rew_mask"]), ("tight", it["tight_mask"])):
            if mask is None:
                continue
            it["scores"][key] = {m: score_one(it["maps"][m], mask) for m in it["maps"]}
        it["border"] = {m: border_frac(it["maps"][m]) for m in it["maps"]}

    n_t = sum(it["tight_mask"] is not None for it in items)
    n_r = sum(it["rew_mask"] is not None for it in items)
    print(f"[dino] {n_r}/{len(items)} steps have a reward referent, {n_t} a tight one",
          flush=True)

    # ---- within-chain disjoint pairs, per model ------------------------------------
    by_key: dict[tuple, list] = {}
    for it in items:
        by_key.setdefault((it["run"], it["sample"]), []).append(it)

    candidates, rates = [], {t: [0, 0] for t in models}
    for (run, sample), recs in sorted(by_key.items()):
        per_model = {}
        for t in models:
            per_model[t] = [r for r in recs if r["model"] == t and r["tight_mask"] is not None]
        for t, rs in per_model.items():
            for a in range(len(rs)):
                for b in range(a + 1, len(rs)):
                    ri, rj = rs[a], rs[b]
                    mi, mj = ri["tight_mask"], rj["tight_mask"]
                    if mi.shape != mj.shape:
                        continue
                    if mi.mean() > args.max_referent_area or mj.mean() > args.max_referent_area:
                        continue
                    union = float((mi | mj).sum())
                    iou = float((mi & mj).sum()) / union if union else 1.0
                    if iou > args.max_pair_iou:
                        continue
                    pm = pair_margin(ri, rj, mi, mj, args.rank_map)
                    if pm is None:
                        continue
                    rates[t][1] += 1
                    rates[t][0] += int(pm["margin"] > 0)
                    if t != args.ours:
                        continue
                    # The same two regions, scored against every OTHER model's best own
                    # step pair -- the most generous comparison available to it.
                    others = {}
                    for ot, ors in per_model.items():
                        if ot == args.ours:
                            continue
                        best = None
                        for p in range(len(ors)):
                            for q in range(len(ors)):
                                if p == q or ors[p]["tight_mask"].shape != mi.shape:
                                    continue
                                om = pair_margin(ors[p], ors[q], mi, mj, args.rank_map)
                                if om and (best is None or om["margin"] > best["margin"]):
                                    best = {**om, "step_i": ors[p]["step"],
                                            "step_j": ors[q]["step"]}
                        others[ot] = best
                    candidates.append({
                        "run": run, "sample": sample, "row": ri["row"],
                        "dataset": ri["dataset"], "question": ri["question"],
                        "gt_answer": ri["gt_answer"], "iou": iou,
                        "step_i": ri["step"], "step_j": rj["step"],
                        "text_i": ri["text"], "text_j": rj["text"],
                        "area_i": float(mi.mean()), "area_j": float(mj.mean()),
                        "boxes_i": [[round(v, 5) for v in b] for b in ri["tight_boxes"]],
                        "boxes_j": [[round(v, 5) for v in b] for b in rj["tight_boxes"]],
                        "rew_area_i": (None if ri["rew_mask"] is None else float(ri["rew_mask"].mean())),
                        "rew_area_j": (None if rj["rew_mask"] is None else float(rj["rew_mask"].mean())),
                        "border_i": ri["border"], "border_j": rj["border"],
                        "ours_answer": ri["answer"], "ours_grade": ri["grade"],
                        "ours_format_ok": ri["format_ok"],
                        "others_answer": {ot: (ors[0]["answer"] if ors else None)
                                          for ot, ors in per_model.items() if ot != args.ours},
                        "others_grade": {ot: (ors[0]["grade"] if ors else None)
                                         for ot, ors in per_model.items() if ot != args.ours},
                        "crossover": {m: pair_margin(ri, rj, mi, mj, m) for m in maps},
                        "others_crossover": others,
                        "scores_i": ri["scores"], "scores_j": rj["scores"],
                        "sdir_i": ri["sdir"], "sdir_j": rj["sdir"],
                    })

    def rank(c):
        own = c["crossover"].get(args.rank_map) or {"margin": -9e9}
        best_other = max([(v or {"margin": -9e9})["margin"]
                          for v in c["others_crossover"].values()] or [-9e9])
        return (own["margin"] - max(best_other, 0.0), own["margin"])

    candidates.sort(key=rank, reverse=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    steps_json = [{k: v for k, v in it.items()
                   if k not in ("image", "maps", "rew_mask", "tight_mask")} for it in items]
    out.write_text(json.dumps({
        "run_dirs": [str(r) for r in run_dirs], "models": models, "ours": args.ours,
        "maps": maps, "rank_map": args.rank_map,
        "cfg": {k: getattr(args, k) for k in
                ("box_threshold", "max_box_area", "tight_score_frac", "tight_max_box_area",
                 "max_referent_area", "max_pair_iou")},
        "crossover_rates": {t: {"pos": v[0], "n": v[1],
                                "rate": (v[0] / v[1] if v[1] else None)}
                            for t, v in rates.items()},
        "steps": steps_json, "candidates": candidates}, indent=1, default=str))

    print(f"\n[out] {out}  ({len(candidates)} candidate panels)")
    print(f"\n=== crossover rate on `{args.rank_map}`, over every disjoint within-chain "
          f"pair (IoU<={args.max_pair_iou}, each referent <={args.max_referent_area:.0%} "
          f"of the grid) ===")
    for t, (pos, n) in rates.items():
        print(f"  {t:28s} {pos:4d}/{n:4d} pairs move the right way"
              + (f"  ({pos / n:.0%})" if n else "  (no qualifying pair)"))

    print(f"\n=== top {args.top} candidates ===")
    for c in candidates[:args.top]:
        own = c["crossover"].get(args.rank_map) or {}
        line = (f"{c['run']}/{c['sample']} [{c['dataset']}] s{c['step_i']}+s{c['step_j']} "
                f"IoU {c['iou']:.2f} areas {c['area_i']:.2f}/{c['area_j']:.2f} "
                f"margin {own.get('margin', float('nan')):+.2f}")
        print(line)
        print(f"    Q: {c['question'][:100]}   gold: {c['gt_answer']}")
        print(f"    ours -> {str(c['ours_answer'])[:70]!r}  "
              f"strict {c['ours_grade']['strict']} soft {c['ours_grade']['soft']}")
        for ot, g in c["others_grade"].items():
            oc = c["others_crossover"].get(ot) or {}
            print(f"    {ot:6s} -> {str(c['others_answer'][ot])[:70]!r}  "
                  f"strict {g and g['strict']} soft {g and g['soft']}  "
                  f"best margin {oc.get('margin', float('nan')):+.2f}")
        print(f"    s{c['step_i']} v2 {own.get('self_i', float('nan')):.2f} vs "
              f"{own.get('other_i', float('nan')):.2f}  {c['text_i'][:88]}")
        print(f"    s{c['step_j']} v2 {own.get('self_j', float('nan')):.2f} vs "
              f"{own.get('other_j', float('nan')):.2f}  {c['text_j'][:88]}")


if __name__ == "__main__":
    main()
