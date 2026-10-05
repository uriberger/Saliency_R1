#!/usr/bin/env python
"""Does the top-left visual token carry the WHOLE PICTURE? -- H1, the register's definition.

    python register_probe.py labels  --out-dir outputs/register_probe/pope
    python register_probe.py extract --out-dir outputs/register_probe/pope   # 1 GPU, minutes
    python register_probe.py probe   --out-dir outputs/register_probe/pope   # CPU
    python register_probe.py report  --out-dir outputs/register_probe/pope

WHY. Qwen3-VL pours attention onto the top-left square of every picture (the peak is cell
(0,0) on 88.0% of 1,800 pictures), that square's vector is a norm outlier (2.3x a random
row), replacing it costs 1.56x what replacing a random row costs, and destroying its
PIXELS costs 0.87x -- less than random. So its content did not come from underneath it,
and it matters more than its neighbours. That is a register's *signature*. It is not a
register's *definition*.

The definition, from Darcet et al., is that the token holds information about the WHOLE
IMAGE rather than about its own patch. Nobody in this project has tested that. This file
does, by the method that paper used: take ONE square's vector, train the simplest possible
classifier on it, and see whether it can answer a question about the entire picture.

THE QUESTION ASKED OF THE PICTURE. "Is there a <object> in it?" -- ground truth, from
COCO's human annotations, by way of POPE (`lmms-lab/POPE`), whose rows are exactly that
question with a yes/no answer and which is balanced 1500/1500 by construction. 500 distinct
COCO val2014 pictures, and 5,113 (picture, object) labels recovered by unioning its three
splits, which agree with each other on every one of them.

This is a good question for the purpose because the object is almost never in the corner:
a square at (0, 0) has no local evidence for "is there a dog somewhere in this picture".
If it answers anyway, it got the answer from somewhere other than its own pixels.

THE ARMS, all one square, all 4,096 numbers:

    tl      cell (0, 0)                 the suspect
    mid     the middle cell             THE control that matters -- it is also always in
                                        the same place, so beating a RANDOM square could
                                        just mean "a fixed position is easier to learn"
    rand    a random cell per picture
    mean    the average of all squares  a reference, not a ceiling: averaging destroys
                                        information, so this is not an upper bound on
                                        what the picture contains

THREE THINGS THAT WOULD OTHERWISE MAKE THE RESULT WRONG.

  scale     the top-left row's norm is 2.3x a normal row's. A classifier can read scale.
            Every vector is scaled to unit length BEFORE anything else, so the arms differ
            in direction only -- the same control that moved the swap result from 1.89x to
            1.56x and was not a formality there either.
  capacity  one square is 4,096 numbers against a few hundred pictures, so how hard the
            classifier is held back decides the score. The strength is chosen by an inner
            cross-validation INSIDE each training fold, separately for every arm, so no
            arm is reported at another arm's best setting.
  pairing   every arm sees the SAME folds of the SAME pictures, so the arms can be
            compared per class rather than only on average.

Scores are BALANCED accuracy -- the mean of the accuracy on the yes pictures and on the no
pictures -- so always answering "yes" scores 0.5 however lopsided the class is, and
`person` (69% yes) does not have to be thrown away to be read.

WHAT WOULD ANSWER IT. If the top-left square is a register it beats `mid` by a clear margin
(prior work on this comparison reports 20-30 points). If it ties `mid`, it is not a
register and the project's register language has to go.

CAVEAT THIS CANNOT CLOSE. POPE gives presence, not boxes, so pictures where the object
really does sit in the top-left corner cannot be filtered out. The pixel arm argues that
matters little -- destroying that square's own pixels changed the model LESS than
destroying a random square's -- but it is not nothing.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import analysis_stats as ST  # noqa: E402

#: The question POPE asks, and the only form of it this file will parse. Anything else is
#: counted and skipped rather than guessed at.
#:
#: `imange` is not a typo here -- it is a typo in POPE, on 57 of its 9,000 rows, all of
#: the form "Is there an <X> in the imange?" and all on six objects (umbrella, orange,
#: oven, airplane, elephant, apple). Parsing only the correct spelling silently drops
#: real labels from exactly those six classes, which is worse than accepting both.
POPE_Q = re.compile(r"Is there an? (.+) in the ima(?:ge|nge)\?", re.IGNORECASE)

#: A class is probed only if it has this many pictures, and this many on its minority side.
#: Below that the fold-level balanced accuracy is dominated by which picture landed where.
MIN_TOTAL, MIN_SIDE = 60, 20

#: The prompt used while extracting. The vision tower never sees the text, so the rows this
#: file captures do not depend on it -- but it is fixed and recorded anyway, so a later run
#: that taps a LANGUAGE-model layer instead is comparing like with like.
EXTRACT_QUESTION = "Describe the image."


# ---------------------------------------------------------------------------
# labels
# ---------------------------------------------------------------------------
def stage_labels(args):
    """POPE -> one record per picture, with every object it is labelled for.

    The three splits (random / popular / adversarial) differ only in how the ABSENT
    objects were chosen, so the present ones should agree across all three. They are
    unioned and the disagreements counted: a non-zero count would mean the question is not
    being parsed the way it is written, and the labels would be a mixture of two things.
    """
    from datasets import load_dataset

    import overlap_probe as PROBE

    out = Path(args.out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    d = load_dataset(args.pope_repo, args.pope_config)

    pres, unparsed, seen_img = {}, 0, {}
    for split in d:
        s = d[split]
        for i, (q, a, src) in enumerate(zip(s["question"], s["answer"], s["image_source"])):
            m = POPE_Q.fullmatch(q.strip())
            if not m:
                unparsed += 1
                continue
            key = (src, m.group(1).strip().lower())
            v = a.strip().lower() == "yes"
            if key in pres and pres[key] is not None and pres[key] != v:
                pres[key] = None
            else:
                pres.setdefault(key, v)
            seen_img.setdefault(src, (split, i))
    bad = sum(1 for v in pres.values() if v is None)
    print(f"[labels] {len(seen_img)} pictures, {len(pres)} (picture, object) labels, "
          f"{bad} disagreements across splits, {unparsed} questions not parsed", flush=True)
    if bad:
        raise SystemExit("the splits disagree about a present object -- the parse is wrong")

    by_img = collections.defaultdict(dict)
    for (src, obj), v in pres.items():
        by_img[src][obj] = int(v)

    rows = []
    for src, (split, i) in sorted(seen_img.items()):
        im = PROBE.prepare_image(d[split][i]["image"].convert("RGB"))
        im.save(out / "images" / f"{src}.png")
        rows.append({"key": src, "image": f"images/{src}.png",
                     "size": list(im.size), "objects": by_img[src]})
    (out / "labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    per = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        for obj, v in r["objects"].items():
            per[obj][0 if v else 1] += 1
    keep = [(o, y, n) for o, (y, n) in per.items()
            if y + n >= MIN_TOTAL and min(y, n) >= MIN_SIDE]
    print(f"[labels] {len(rows)} pictures written; {len(keep)} classes clear "
          f"{MIN_TOTAL} pictures / {MIN_SIDE} a side", flush=True)
    for o, y, n in sorted(keep, key=lambda t: -(t[1] + t[2])):
        print(f"           {o:<16} {y:>4} with / {n:>4} without  ({y / (y + n):.2f} yes)")
    return 0


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------
def stage_extract(args):
    """One prefill per picture; keep the rows the language model is handed.

    The capture point is the vision tower's output AFTER the 2x2 merge and projection --
    `vlm_family.Family.row_module`, the same place the swap experiment edits -- so a row
    here is exactly one grid cell as the language model receives it, 4,096 numbers wide.
    Nothing from the language model's own layers is in it, which is the point: the claim
    under test is that the ENCODER wrote a summary into that cell.

    ALL rows are kept, not just the three the probe currently asks for. The forward is the
    expensive part and a later arm -- k random cells, a different cell, the ring -- should
    not need the GPU again.
    """
    import torch

    import sink_location as SL
    import sink_location_probe as SLP
    import token_mediation_probe as TMP
    from PIL import Image

    out = Path(args.out_dir)
    rows = [json.loads(l) for l in (out / "labels.jsonl").read_text().splitlines() if l]
    if args.limit:
        rows = rows[: args.limit]
    device = args.device
    processor, model = SLP.load_model(args.model, args.adapter, device, "sdpa")
    fam = SLP.load_family(model, processor, args.system_prompt)
    print(f"[extract] {len(rows)} pictures, family={fam.name}, "
          f"system prompt={'(none)' if not fam.system_prompt else 'set'}", flush=True)

    keys, grids, flat, shapes = [], [], [], []
    cap = TMP.RowCapture(model, family=fam).install()
    try:
        for n, r in enumerate(rows):
            im = Image.open(out / r["image"]).convert("RGB")
            inputs = SLP.build_inputs(fam, processor, [im], EXTRACT_QUESTION, device)
            with torch.no_grad():
                model(**inputs, use_cache=False)
            if cap.rows is None:
                print(f"[extract] {r['key']}: no rows captured, skipped", flush=True)
                continue
            _runs, gr = SL.locate_image_runs(inputs["input_ids"], inputs, fam)
            if len(gr) != 1:
                print(f"[extract] {r['key']}: {len(gr)} grids, skipped", flush=True)
                continue
            _t, gh, gw = gr[0]
            a = cap.rows.float().cpu().numpy()
            if a.shape[0] != gh * gw:
                print(f"[extract] {r['key']}: {a.shape[0]} rows for a {gh}x{gw} grid, "
                      "skipped", flush=True)
                continue
            keys.append(r["key"])
            grids.append([int(gh), int(gw)])
            shapes.append(list(a.shape))
            flat.append(a.astype(np.float16).reshape(-1))
            if (n + 1) % 50 == 0:
                print(f"[extract] {n + 1}/{len(rows)}", flush=True)
    finally:
        cap.uninstall()

    np.savez(out / "rows.npz", keys=np.asarray(keys), grids=np.asarray(grids, dtype=np.int64),
             shapes=np.asarray(shapes, dtype=np.int64), flat=np.concatenate(flat))
    print(f"[extract] {len(keys)} pictures -> {out / 'rows.npz'} "
          f"({sum(s[0] * s[1] for s in shapes) * 2 / 1e6:.0f} MB)", flush=True)
    return 0


def read_rows(out_dir):
    """-> {key: [N, D] float32}, {key: (gh, gw)}."""
    z = np.load(Path(out_dir) / "rows.npz")
    keys = [str(k) for k in z["keys"]]
    flat, off, arrs = z["flat"], 0, {}
    for k, sh in zip(keys, z["shapes"]):
        n = int(sh[0]) * int(sh[1])
        arrs[k] = flat[off:off + n].reshape(int(sh[0]), int(sh[1])).astype(np.float32)
        off += n
    return arrs, {k: tuple(int(x) for x in g) for k, g in zip(keys, z["grids"])}


# ---------------------------------------------------------------------------
# the arms
# ---------------------------------------------------------------------------
def arm_vector(name, rows, gh, gw, rng):
    """One arm's input for one picture. -> [D] float32, scaled to unit length by the caller."""
    if name == "tl":
        return rows[0]
    if name == "mid":
        return rows[(gh // 2) * gw + (gw // 2)]
    if name == "rand":
        return rows[rng.integers(rows.shape[0])]
    if name == "mean":
        return rows.mean(0)
    if name == "br":
        return rows[-1]
    raise ValueError(f"unknown arm {name!r}")


ARMS = ("tl", "mid", "rand", "mean", "br")


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------
def stage_probe(args):
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold

    out = Path(args.out_dir)
    labels = {r["key"]: r["objects"]
              for r in (json.loads(l) for l in
                        (out / "labels.jsonl").read_text().splitlines() if l)}
    arrs, grids = read_rows(out)
    print(f"[probe] {len(arrs)} pictures with rows", flush=True)

    per = collections.defaultdict(lambda: [0, 0])
    for k in arrs:
        for obj, v in labels.get(k, {}).items():
            per[obj][0 if v else 1] += 1
    classes = sorted([o for o, (y, n) in per.items()
                      if y + n >= MIN_TOTAL and min(y, n) >= MIN_SIDE],
                     key=lambda o: -(per[o][0] + per[o][1]))
    print(f"[probe] {len(classes)} classes clear {MIN_TOTAL}/{MIN_SIDE}: "
          f"{', '.join(classes)}", flush=True)

    # One random cell per PICTURE, drawn once and reused by every class, so `rand` is the
    # same arm everywhere rather than a different draw per class.
    rng = np.random.default_rng(args.seed)
    feats = {}
    for arm in ARMS:
        r2 = np.random.default_rng(args.seed)
        X = {}
        for k in sorted(arrs):
            gh, gw = grids[k]
            v = arm_vector(arm, arrs[k], gh, gw, r2).astype(np.float64)
            nrm = np.linalg.norm(v)
            X[k] = v / nrm if nrm > 0 else v          # SCALE CONTROL: direction only
        feats[arm] = X

    def score(X, y, folds):
        """Balanced accuracy per fold. liblinear's DUAL solver because there are far more
        numbers than pictures (4,096 against a few hundred), which is the case it is for:
        identical predictions to lbfgs in a check at this exact shape, 3-4x faster."""
        sc = []
        for tr, te in folds:
            clf = LogisticRegressionCV(
                Cs=np.logspace(-4, 4, 9), cv=args.inner_folds, scoring="balanced_accuracy",
                class_weight="balanced", solver="liblinear", dual=True,
                max_iter=5000, n_jobs=args.jobs)
            clf.fit(X[tr], y[tr])
            sc.append(balanced_accuracy_score(y[te], clf.predict(X[te])))
        return sc

    def make_folds(y, seed):
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=seed)
        return list(skf.split(np.zeros(len(y)), y))

    results = collections.defaultdict(dict)
    for ci, obj in enumerate(classes):
        ks = sorted(k for k in arrs if obj in labels.get(k, {}))
        y = np.array([labels[k][obj] for k in ks])
        folds = make_folds(y, args.seed)                  # SAME folds for every arm
        for arm in ARMS:
            X = np.stack([feats[arm][k] for k in ks])
            sc = score(X, y, folds)
            results[obj][arm] = {"mean": float(np.mean(sc)),
                                 "sd": float(np.std(sc, ddof=1)),
                                 "folds": [float(s) for s in sc]}
        results[obj]["n"] = {"yes": int(y.sum()), "no": int((1 - y).sum())}

        # THE FLOOR, measured rather than assumed. With 60-500 pictures a single arm can
        # read well above 0.50 on a label it cannot possibly know -- a synthetic check at
        # n=240 put an uninformative arm at 0.589. So the same pipeline is run on SHUFFLED
        # labels, which is what chance actually looks like at this sample size, for the
        # two arms the conclusion rests on.
        if args.null_shuffles:
            nrng = np.random.default_rng(args.seed + 991 + ci)
            nulls = collections.defaultdict(list)
            for s in range(args.null_shuffles):
                ysh = nrng.permutation(y)
                fsh = make_folds(ysh, args.seed + s)
                for arm in ("tl", "mid"):
                    X = np.stack([feats[arm][k] for k in ks])
                    nulls[arm].append(float(np.mean(score(X, ysh, fsh))))
            results[obj]["null"] = {a: v for a, v in nulls.items()}
        print(f"[probe] {ci + 1}/{len(classes)} {obj:<16} "
              + "  ".join(f"{a}={results[obj][a]['mean']:.3f}" for a in ARMS)
              + (f"   null tl={np.mean(results[obj]['null']['tl']):.3f}"
                 if args.null_shuffles else ""), flush=True)

    (out / "probe.json").write_text(json.dumps(
        {"classes": classes, "arms": list(ARMS), "folds": args.folds,
         "null_shuffles": args.null_shuffles, "results": results}, indent=1))
    print(f"[probe] -> {out / 'probe.json'}")
    return 0


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def stage_report(args):
    out = Path(args.out_dir)
    d = json.loads((out / "probe.json").read_text())
    res, classes, arms = d["results"], d["classes"], d["arms"]
    lines = []

    def o(s=""):
        print(s, flush=True)
        lines.append(s)

    o("=" * 76)
    o("H1 -- does the top-left visual token carry the WHOLE PICTURE?")
    o("=" * 76)
    o("Balanced accuracy on held-out pictures: 0.50 = no better than always answering")
    o("the commoner way. One square's 4,096 numbers, scaled to unit length, one simple")
    o(f"weighted-sum classifier, {d['folds']} folds, the same folds for every arm.")
    o("")
    has_null = bool(d.get("null_shuffles"))
    o(f"   {'object':<16} {'yes':>5} {'no':>5} " + "".join(f"{a:>9}" for a in arms)
      + (f"{'tl|null':>9}" if has_null else ""))
    for obj in classes:
        r = res[obj]
        o(f"   {obj:<16} {r['n']['yes']:>5} {r['n']['no']:>5} "
          + "".join(f"{r[a]['mean']:>9.3f}" for a in arms)
          + (f"{np.mean(r['null']['tl']):>9.3f}" if has_null else ""))
    o("   " + "-" * (16 + 12 + 9 * (len(arms) + int(has_null))))
    means = {a: float(np.mean([res[o_][a]["mean"] for o_ in classes])) for a in arms}
    o(f"   {'MEAN':<16} {'':>5} {'':>5} " + "".join(f"{means[a]:>9.3f}" for a in arms)
      + (f"{np.mean([np.mean(res[o_]['null']['tl']) for o_ in classes]):>9.3f}"
         if has_null else ""))
    if has_null:
        nt = [v for o_ in classes for v in res[o_]["null"]["tl"]]
        o("")
        o(f"   THE FLOOR, measured: with labels SHUFFLED, the tl arm reads "
          f"{np.mean(nt):.3f} on average,")
        o(f"   spread {np.std(nt, ddof=1):.3f}, highest single class {max(nt):.3f} "
          f"({d['null_shuffles']} shuffles x {len(classes)} classes).")
        o("   Read every number above against THAT, not against 0.500.")

    o("")
    o("THE COMPARISON THAT MATTERS: top-left against the FIXED MIDDLE square.")
    o("Both are always in the same place, so a gap cannot be 'a fixed position is")
    o("easier to learn'. Per class, and the sign test over classes.")
    o("")
    o(f"   {'object':<16} {'tl':>8} {'mid':>8} {'tl - mid':>10}")
    diffs = []
    for obj in classes:
        dlt = res[obj]["tl"]["mean"] - res[obj]["mid"]["mean"]
        diffs.append(dlt)
        o(f"   {obj:<16} {res[obj]['tl']['mean']:>8.3f} {res[obj]['mid']['mean']:>8.3f} "
          f"{dlt:>+10.3f}")
    wins = sum(1 for x in diffs if x > 0)
    o("")
    o(f"   mean difference {np.mean(diffs):+.3f}   top-left wins in {wins}/{len(diffs)} "
      f"classes")
    lo, hi = ST.wilson(wins, len(diffs))
    o(f"   win rate 95% interval [{lo:.2f}, {hi:.2f}]  (0.5 is a coin flip)")
    if has_null:
        nd = [res[o_]["null"]["tl"][i] - res[o_]["null"]["mid"][i]
              for o_ in classes for i in range(len(res[o_]["null"]["tl"]))]
        o(f"   with labels shuffled the same difference is {np.mean(nd):+.3f} "
          f"(spread {np.std(nd, ddof=1):.3f}), so the bar for the row above is roughly "
          f"{2 * np.std(nd, ddof=1) / np.sqrt(len(classes)):.3f}.")

    o("")
    o("READING IT")
    o("  tl >> mid, by the 20-30 points prior work reports for this comparison")
    o("      -> the top-left square is a register in the defining sense.")
    o("  tl ~= mid, both above 0.5")
    o("      -> every square knows roughly this much about the picture; the top-left one")
    o("         is not special, and the project's register language has to go.")
    o("  tl ~= mid ~= 0.5")
    o("      -> one square's vector carries no whole-picture information at all, and the")
    o("         experiment says nothing about registers -- check `mean` before concluding")
    o("         anything, because if THAT is also 0.5 the probe is broken, not the model.")
    o("")
    o("CAVEATS")
    o("  - 500 pictures, 60-500 per class. Sized for a large effect, not a small one.")
    o("  - POPE gives presence, not boxes, so pictures where the object really is in the")
    o("    top-left corner cannot be filtered out.")
    o("  - The rows are the vision tower's output. Nothing here is about what the")
    o("    language model does with them afterwards.")
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    print(f"\n-> {out / 'report.txt'}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["labels", "extract", "probe", "report"])
    ap.add_argument("--out-dir", default="outputs/register_probe/pope")
    ap.add_argument("--pope-repo", default="lmms-lab/POPE")
    ap.add_argument("--pope-config", default="Full")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--system-prompt", default="none")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--inner-folds", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--null-shuffles", type=int, default=3,
                    help="label shuffles per class, on the tl and mid arms, to measure "
                         "what chance looks like at this sample size. 0 disables.")
    ap.add_argument("--seed", type=int, default=20261005)
    args = ap.parse_args()
    os.chdir(REPO)
    return {"labels": stage_labels, "extract": stage_extract,
            "probe": stage_probe, "report": stage_report}[args.stage](args)


if __name__ == "__main__":
    sys.exit(main())
