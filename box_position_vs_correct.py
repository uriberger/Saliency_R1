#!/usr/bin/env python
"""Arm 0: does a model answer WORSE when the answer region sits near the edge of the grid?

    python box_position_vs_correct.py build  --out-dir OUT
    python box_position_vs_correct.py judge  --out-dir OUT        # needs NVIDIA_API_KEY
    python box_position_vs_correct.py report --out-dir OUT

THE QUESTION. Every geometric result in this project so far is about ATTENTION -- the
border draws 2.6x its share, the corner patch 13x, the mark is written by the encoder.
None of it says the bias COSTS anything. This is the cheapest possible test of that: the
boxed corpus already carries a human answer box per picture, four models have already
answered all 1,800 of them, and the completions are on disk. So ask whether correctness
depends on where that box sits on the patch grid, for nothing but CPU.

WHAT THIS CAN AND CANNOT ANSWER -- read this before reading a number.

The corpus was built to measure attention, not to vary position, and Visual-CoT's answer
regions are centrally placed (`sink_three_legs` measures them at 0.71-0.76 against a
translation null, i.e. genuinely central). Measured on Qwen3-VL's own grids:

    box centroid in cell (0,0)       0 of 1800   (0.00%)
    box centroid in ANY corner cell  1 of 1800   (0.06%)
    box centroid anywhere on the ring  84        (4.7%)
    coarse 3x3 centre bin            660         (37%)

So this arm CANNOT test the corner/register claim -- there is no mass to regress on, and
no amount of statistics recovers it. It CAN test the coarse centre-versus-periphery
contrast at 3x3 granularity, where the four corner bins hold 65-93 pictures each, and it
is the only thing here that prices the effect before the interventional arms are built.
Its other job is the power section at the end of the report: Arm 2 should be sized from
the number this arm measures, not from a guess.

AND IT IS OBSERVATIONAL. Where a human put the answer box is not randomised -- a question
whose answer lives at the edge of the frame is a different question, about a different
kind of thing, from one whose answer is in the middle. Two nuisance routes are closed
explicitly and neither closes the confound itself:

  difficulty   an easy picture is easy wherever its box is, and the mix of easy pictures
               differs by bin. Held fixed by stratifying on how many of the OTHER models
               answered that same picture correctly (0..3) and combining with
               Mantel-Haenszel. This OVER-controls if position hurts every model at once,
               so the raw and the stratified numbers are both printed and neither is
               called the answer on its own.
  box size     a small box is harder to answer about and may sit differently. Held fixed
               by a second stratification on the box's area quartile.

THE LABEL IS THE LLM JUDGE, and that is not cosmetic. These scans ran with
`--system-prompt none`, so every model answers in prose: exact match scores Qwen3-VL at
0.006 and a word-boundary substring rule mis-scores in both directions (gold `car` against
"the undercarriage of a vehicle" reads wrong; gold `man` fires on a word elsewhere in the
span). 327 of the 1,800 rows are flickr30k, whose gold is a full sentence -- they are
real VQA and the judge grades them fine, where a string rule scores them 0.003 and would
have forced dropping 18% of the corpus. Both string grades are carried anyway, because a
judge that disagrees with every string match is a judge worth checking.

WHAT IT READS. Nothing is regenerated: the completions are the `completion` field already
stored in each scan's npz as token ids, the grid is the `grid` field in its jsonl, the box
is the corpus manifest's `bbox`, and the gold is recovered by joining
`(set, dataset, question_id)` back into `cold_data/grpo_sets` -- exact on 1800/1800 rows
with zero question or bbox mismatches, which is checked and printed by `build`.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import analysis_stats as ST  # noqa: E402

#: family -> (scan directory under outputs/sink_location/xmodel, tokenizer repo, label).
#: LLaVA-1.5 is absent for the same reason it is absent from the boxed corpus: it has no
#: ring to explain (docs/sink-location-cross-model.md section 6).
DEFAULT_SCANS = {
    "qwen3_vl": ("outputs/sink_location/xmodel/box_qwen3vl",
                 "Qwen/Qwen3-VL-8B-Instruct", "Qwen3-VL-8B"),
    "internvl": ("outputs/sink_location/xmodel/box_internvl35",
                 "OpenGVLab/InternVL3_5-8B-HF", "InternVL3.5-8B"),
    "glm4v": ("outputs/sink_location/xmodel/box_glm4v",
              "zai-org/GLM-4.1V-9B-Thinking", "GLM-4.1V-9B"),
    "nemotron_vl": ("outputs/sink_location/xmodel/box_nemotron",
                    "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16",
                    "Nemotron-3-Nano-Omni-30B"),
}

MANIFEST = "outputs/sink_location/xmodel/boxed/corpus/manifest.jsonl"

#: judge score (0-1, from the 1-5 scale) at or above which an answer counts as correct.
#: 0.75 is 4/5, the cut `human_box_vs_correct.py` uses, kept identical so the two analyses
#: report the same notion of "right".
JUDGE_THRESH = 0.75


# ---------------------------------------------------------------------------
# reading what is already on disk
# ---------------------------------------------------------------------------
def read_gold(manifest_rows, verbose=True):
    """-> ({key: gold}, n_checked) by joining back into cold_data/grpo_sets.

    The manifest records `source` as "<set>:<dataset>" and `ref` as the row's
    `question_id`, and `collect()` in build_boxed_corpus deduplicates on
    (dataset, question_id), so (set, dataset, question_id) is a key. The join is verified
    rather than trusted: the set's `problem` must equal the manifest's `question` and its
    `bbox` must equal the manifest's `bbox`, because a silently wrong join would attach
    the right geometry to the wrong gold and every number below would be noise with a
    plausible shape.
    """
    from datasets import load_from_disk

    want = collections.defaultdict(list)
    for r in manifest_rows:
        s, d = r["source"].split(":", 1)
        want[s].append((d, str(r["ref"]), r))

    gold, q_bad, b_bad, missing = {}, 0, 0, 0
    for s in sorted(want):
        p = REPO / "cold_data" / "grpo_sets" / s
        ds = load_from_disk(str(p))
        ds = ds["train"] if hasattr(ds, "keys") else ds
        lut = {}
        for rec in ds.select_columns(["dataset", "question_id", "problem",
                                      "solution", "bbox"]):
            lut.setdefault((rec["dataset"], str(rec["question_id"])), rec)
        for d, qid, r in want[s]:
            rec = lut.get((d, qid))
            if rec is None:
                missing += 1
                continue
            if (rec["problem"] or "").strip() != (r["question"] or "").strip():
                q_bad += 1
            bb = rec["bbox"]
            if isinstance(bb, str):
                bb = json.loads(bb)
            if max(abs(float(a) - float(b)) for a, b in zip(bb, r["bbox"])) > 2e-3:
                b_bad += 1
            gold[r["key"]] = rec["solution"]
    if verbose:
        print(f"[build] gold joined for {len(gold)}/{len(manifest_rows)} rows "
              f"({missing} missing, {q_bad} question mismatch, {b_bad} bbox mismatch)",
              flush=True)
    if missing or q_bad or b_bad:
        raise SystemExit("the join is not exact -- refusing to build a table on it")
    return gold


def read_scan(scan_dir):
    """-> {unit: {"grid": [gh, gw], "view": [...], "completion": int32 array}}.

    DEDUPLICATED BY UNIT, last write wins, which is `sink_location_probe.read_stage`'s
    rule and matters here: resume is per shard, so re-running a directory at a different
    shard count re-measures the units that changed hands and both copies are on disk.
    box_nemotron has 2,067 jsonl rows and 2,163 stored completions for 1,800 pictures for
    exactly this reason; counting them as separate pictures would double-weight them.
    """
    d = Path(scan_dir)
    meta = {}
    for p in sorted(d.glob("scan_shard*.jsonl")):
        for line in p.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("unit"):
                meta[r["unit"]] = r
    out = {}
    for p in sorted(d.glob("scan_shard*_part*.npz")):
        with np.load(p, allow_pickle=False) as z:
            if "completion" not in z.files:
                continue
            units = [str(u) for u in z["units"]]
            flat, off = z["completion"], 0
            for i, sh in zip(z["completion__idx"], z["completion__shapes"]):
                n = int(np.prod(sh))
                out.setdefault(units[int(i)], {})["completion"] = flat[off:off + n]
                off += n
    for u, m in meta.items():
        if u in out:
            out[u]["grid"] = m.get("grid")
            out[u]["view"] = m.get("view")
            out[u]["n_generated"] = m.get("n_generated")
    return {u: v for u, v in out.items() if "grid" in v}


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
_RING = {}


def ring_grid(gh, gw):
    if (gh, gw) not in _RING:
        import sink_three_legs as T
        _RING[(gh, gw)] = T.ring_grid(gh, gw)
    return _RING[(gh, gw)]


def box_features(bbox, gh, gw):
    """Position of a normalised box on a gh x gw grid, as several size-aware measures.

    `mask_from_boxes` is `sink_three_legs`'s, so a box rasterises here exactly as it does
    in the attention tables -- the patches whose CENTRES the box covers, with a
    sub-patch box snapped to one cell rather than dropped, which would otherwise select
    for large boxes.

    The PRIMARY position measures are `depth_norm` and `bin3`, and both are deliberately
    size-free. `ring_frac` is carried because it is the statistic the attention side uses,
    but a region's overlap with a thin frame is mostly a statement about its size -- see
    the translation-null note in docs/sink-location-cross-model.md -- so it is secondary
    here and must not be read as a position claim on its own.
    """
    import sink_three_legs as T

    x0, y0, x1, y1 = [float(v) for v in bbox]
    m = T.mask_from_boxes([[x0, y0, x1, y1]], gh, gw, max_area=0.0)
    ring = ring_grid(gh, gw)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    ci = min(gw - 1, max(0, int(cx * gw)))
    ri = min(gh - 1, max(0, int(cy * gh)))
    depth = min(ri, gh - 1 - ri, ci, gw - 1 - ci)
    # normalise by the deepest cell this grid has, so 0 = on the border and 1 = dead
    # centre whatever the grid shape. Grids here run 8x10 to 19x14.
    dmax = max(1, (min(gh, gw) - 1) // 2)
    return {
        "centroid_cell": [ri, ci],
        "depth": int(depth),
        "depth_norm": float(min(1.0, depth / dmax)),
        "on_ring": bool(depth == 0),
        "in_corner": bool(ri in (0, gh - 1) and ci in (0, gw - 1)),
        "bin3": [min(2, int(cy * 3)), min(2, int(cx * 3))],
        "ring_frac": float((m & ring).sum() / max(1, m.sum())),
        "cells_frac": float(m.mean()),
        "area_frac": float(max(0.0, x1 - x0) * max(0.0, y1 - y0)),
        "centroid": [float(cx), float(cy)],
    }


# ---------------------------------------------------------------------------
# pulling the answer out, which is a different shape in every family
# ---------------------------------------------------------------------------
#: the `--max-new-tokens` the boxed scans ran with. A completion this long was CUT OFF,
#: not finished, and on the Nemotron that is a quarter of the corpus.
GEN_CAP = 1024


#: End-of-turn markers. The decode keeps special tokens on purpose (GLM-4.1V's
#: `<|begin_of_box|>` is one), so every family's terminator has to come off by hand or it
#: rides into the judge's prompt and into the string grades.
_END_MARKERS = ("<|im_end|>", "<|endoftext|>", "<|end_of_text|>", "<|eot_id|>",
                "<|end|>", "</s>", "<|begin_of_box|>", "<|end_of_box|>",
                "<answer>", "</answer>")


def _clean(s):
    for t in _END_MARKERS:
        s = s.replace(t, " ")
    return " ".join(s.split()).strip()


def extract_for_family(family, text, n_gen):
    """-> (answer, span, truncated, unfinished).

    `answer_grading.extract_answer` is Qwen-shaped -- `</think>` then prose, or bare
    prose -- and on these four models that is right for two of them and wrong for two:

      GLM-4.1V writes `<think>...</think><answer>... <|begin_of_box|>X<|end_of_box|>.</answer>`.
        The generic rule returns everything after `</think>`, i.e. the whole `<answer>`
        paragraph with its opening tag still attached. The model is telling us exactly
        which span is the answer and it costs one regex to listen.
      Nemotron-Omni is handed a prompt that already OPENS `<think>`, so its completion is
        the inside of the chain and then `</think>` ANSWER. Where the 1024-token cap cut
        the chain there is no `</think>` and no answer at all, and the generic rule
        returns the last line of the reasoning -- which is how `'*   Let'` ends up being
        graded as an answer.

    UNFINISHED IS NOT WRONG, AND IS NOT MISSING EITHER. A model that ran out of budget
    produced no answer, which counts as wrong for accuracy; but it is wrong for a reason
    that has nothing to do with where the box is, so the flag is carried and the report
    prices it and re-runs the contrast without those rows.
    """
    import re

    import answer_grading as AG

    truncated = n_gen >= GEN_CAP
    if family == "glm4v":
        for pat in (r"<\|begin_of_box\|>(.*?)<\|end_of_box\|>", r"<answer>(.*?)</answer>"):
            m = re.search(pat, text, re.DOTALL)
            if m and _clean(m.group(1)):
                return _clean(m.group(1)), _clean(m.group(1)), truncated, False
        return "", "", truncated, True
    if family == "nemotron_vl":
        m = re.search(r"</think>\s*(.*)", text, re.DOTALL)
        if m and _clean(m.group(1)):
            return _clean(m.group(1)), _clean(m.group(1)), truncated, False
        return "", "", truncated, True
    ans, span = AG.extract_answer(text)
    if not _clean(ans):
        return "", "", truncated, True
    return _clean(ans), _clean(span), truncated, False


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def stage_build(args):
    import answer_grading as AG
    from transformers import AutoTokenizer

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = [json.loads(l) for l in (REPO / args.manifest).read_text().splitlines() if l]
    print(f"[build] {len(manifest)} corpus rows from {args.manifest}", flush=True)
    gold = read_gold(manifest, verbose=True)
    by_key = {r["key"]: r for r in manifest}

    scans = dict(DEFAULT_SCANS)
    if args.only:
        keep = set(args.only.split(","))
        scans = {k: v for k, v in scans.items() if k in keep}

    rows = []
    for fam, (sdir, tok_repo, label) in scans.items():
        p = REPO / sdir
        if not p.is_dir():
            print(f"[build] {fam}: {sdir} is not there, skipped", flush=True)
            continue
        scan = read_scan(p)
        tok = AutoTokenizer.from_pretrained(tok_repo, trust_remote_code=True)
        n_noview = 0
        for key, s in scan.items():
            r = by_key.get(key)
            if r is None or key not in gold:
                continue
            gh, gw = s["grid"]
            view = s.get("view") or [0.0, 0.0, 1.0, 1.0]
            # Every model here reads the whole picture. A centre-cropping processor would
            # need the box composed through the view box before it means anything on the
            # grid, so refuse rather than quietly measure the wrong frame.
            if max(abs(a - b) for a, b in zip(view, [0.0, 0.0, 1.0, 1.0])) > 1e-6:
                n_noview += 1
                continue
            # skip_special_tokens=False: GLM-4.1V marks its answer with
            # <|begin_of_box|>, which IS a special token, and stripping it throws away
            # the one span the model told us was the answer.
            text = tok.decode(s["completion"].tolist(), skip_special_tokens=False)
            n_gen = int(s["completion"].size)
            ans, span, trunc, unfin = extract_for_family(fam, text, n_gen)
            gr = AG.grade(ans, gold[key], span)
            f = box_features(r["bbox"], gh, gw)
            rows.append(dict(
                family=fam, model=label, key=key, type=r["type"], source=r["source"],
                dev=bool(r["dev"]), natural=bool(r.get("natural", True)),
                question=r["question"], gold=gold[key], answer=ans,
                n_gen=n_gen, truncated=bool(trunc), unfinished=bool(unfin),
                grid=[int(gh), int(gw)], bbox=[float(v) for v in r["bbox"]],
                strict=gr["strict"], soft=gr["soft"], grade_kind=gr["kind"],
                judge=None, **f))
        mine = [x for x in rows if x["family"] == fam]
        print(f"[build] {label:<26} {len(mine):>5} rows  "
              f"{sum(x['truncated'] for x in mine):>4} hit the {GEN_CAP}-token cap, "
              f"{sum(x['unfinished'] for x in mine):>4} produced no answer"
              + (f"  ({n_noview} dropped: grid does not cover the picture)"
                 if n_noview else ""), flush=True)

    tbl = out / "table.jsonl"
    with open(tbl, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"[build] {len(rows)} rows -> {tbl}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# judge
# ---------------------------------------------------------------------------
def stage_judge(args):
    from llm_judge import judge_scores

    out = Path(args.out_dir)
    tbl = out / "table.jsonl"
    rows = [json.loads(l) for l in tbl.read_text().splitlines() if l]
    todo = [r for r in rows if r.get("judge") is None]
    print(f"[judge] {len(rows)} rows, {len(todo)} without a score", flush=True)
    if not todo:
        return 0
    items = [{"question": r["question"], "gt_answer": r["gold"], "answer": r["answer"]}
             for r in todo]
    cache = args.judge_cache or str(out / "judge_cache.json")
    # Judgements are keyed on content, so any cache this project has written is free
    # money. They are merged INTO this run's cache rather than read alongside it, so one
    # file is the record of what was judged.
    if args.seed_from:
        import glob as _glob
        seeded = json.loads(Path(cache).read_text()) if Path(cache).exists() else {}
        n0 = len(seeded)
        for p in sorted(_glob.glob(args.seed_from)):
            if Path(p).resolve() == Path(cache).resolve():
                continue
            seeded.update({k: v for k, v in json.loads(Path(p).read_text()).items()
                           if v is not None})
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        Path(cache).write_text(json.dumps(seeded))
        print(f"[judge] seeded cache {n0} -> {len(seeded)} entries from {args.seed_from}",
              flush=True)
    scores = judge_scores(items, workers=args.judge_workers, cache_path=cache)
    for r, s in zip(todo, scores):
        r["judge"] = s
    with open(tbl, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    got = [r["judge"] for r in rows if r["judge"] is not None]
    print(f"[judge] {len(got)}/{len(rows)} scored, mean {np.mean(got):.3f}, "
          f"share >= {JUDGE_THRESH}: {np.mean([g >= JUDGE_THRESH for g in got]):.3f}",
          flush=True)
    return 0


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
BIN_NAME = {(0, 0): "top-left", (0, 1): "top", (0, 2): "top-right",
            (1, 0): "left", (1, 1): "CENTRE", (1, 2): "right",
            (2, 0): "bottom-left", (2, 1): "bottom", (2, 2): "bottom-right"}


def _correct(r, label):
    if label == "judge":
        return None if r.get("judge") is None else bool(r["judge"] >= JUDGE_THRESH)
    return bool(r.get(label))


def stage_report(args):
    out = Path(args.out_dir)
    rows = [json.loads(l) for l in (out / "table.jsonl").read_text().splitlines() if l]
    label = args.correct_on
    lines = []

    def o(s=""):
        print(s, flush=True)
        lines.append(s)

    by_model = collections.defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)
    models = sorted(by_model)

    o("=" * 78)
    o("ARM 0 -- does accuracy depend on where the human answer box sits on the grid?")
    o("=" * 78)
    o(f"label = {label}" + (f" (judge score >= {JUDGE_THRESH}, i.e. 4/5)"
                            if label == "judge" else ""))
    o(f"{len(rows)} rows over {len(models)} models")

    # ---- 1. what the corpus can reach -------------------------------------
    o("")
    o("1. WHERE THE BOXES ARE -- the ceiling on what this arm can test")
    o("   Each model rasterises the same box onto its OWN grid, so the counts differ.")
    o("")
    o(f"   {'model':<28} {'n':>5} {'cell (0,0)':>11} {'any corner':>11} "
      f"{'on ring':>9} {'3x3 centre':>11}")
    for m in models:
        rs = by_model[m]
        n = len(rs)
        o(f"   {m:<28} {n:>5} "
          f"{sum(r['centroid_cell'] == [0, 0] for r in rs):>11} "
          f"{sum(r['in_corner'] for r in rs):>11} "
          f"{sum(r['on_ring'] for r in rs):>9} "
          f"{sum(tuple(r['bin3']) == (1, 1) for r in rs):>11}")
    o("")
    o("   -> the corner/register claim is UNTESTABLE on this corpus. What follows is the")
    o("      coarse centre-versus-periphery contrast, which is the RING claim.")

    # ---- 2. accuracy, and the grade's own agreement ------------------------
    o("")
    o("2. ACCURACY, and what the string grades would have said")
    o("   `no answer` = the 1024-token cap cut the chain before any answer existed. Those")
    o("   rows are counted WRONG here (the model produced nothing), which is right for an")
    o("   accuracy number and is a reason that has nothing to do with the box -- so")
    o("   section 4 repeats the contrast with them removed.")
    o(f"   {'model':<28} {'n':>5} {'judge':>8} {'soft':>8} {'strict':>8} "
      f"{'hit cap':>8} {'no answer':>10}")
    for m in models:
        rs = by_model[m]
        j = [r["judge"] for r in rs if r["judge"] is not None]
        o(f"   {m:<28} {len(rs):>5} "
          f"{np.mean([x >= JUDGE_THRESH for x in j]) if j else float('nan'):>8.3f} "
          f"{np.mean([bool(r['soft']) for r in rs]):>8.3f} "
          f"{np.mean([bool(r['strict']) for r in rs]):>8.3f} "
          f"{sum(r['truncated'] for r in rs):>8} {sum(r['unfinished'] for r in rs):>10}")
    o("")
    o("   does running out of budget depend on where the box is? (it should not)")
    for m in models:
        rs = by_model[m]
        if not any(r["unfinished"] for r in rs):
            continue
        rho, p = ST.spearman([r["depth_norm"] for r in rs],
                             [float(r["unfinished"]) for r in rs])
        o(f"   {m:<28} rho(depth_norm, no answer) = {rho:+.4f}  p = {p:.3g}")
    j_all = [r for r in rows if r["judge"] is not None]
    if j_all:
        jc = [r["judge"] >= JUDGE_THRESH for r in j_all]
        sc = [bool(r["soft"]) for r in j_all]
        agree = np.mean([a == b for a, b in zip(jc, sc)])
        o(f"   judge and soft agree on {agree:.1%} of rows "
          f"(judge-right/soft-wrong {np.mean([a and not b for a, b in zip(jc, sc)]):.1%}, "
          f"soft-right/judge-wrong {np.mean([b and not a for a, b in zip(jc, sc)]):.1%})")

    # ---- 3. the 3x3 table --------------------------------------------------
    o("")
    o("3. ACCURACY BY COARSE 3x3 BIN of the box centroid  [95% Wilson]")
    for m in models:
        rs = [r for r in by_model[m] if _correct(r, label) is not None]
        o("")
        o(f"   {m}   (n={len(rs)})")
        for br in range(3):
            cells = []
            for bc in range(3):
                g = [r for r in rs if tuple(r["bin3"]) == (br, bc)]
                k = sum(_correct(r, label) for r in g)
                lo, hi = ST.wilson(k, len(g))
                cells.append(f"{BIN_NAME[(br, bc)]:>12} {k}/{len(g)}"
                             f" = {k / len(g):.3f} [{lo:.2f},{hi:.2f}]"
                             if g else f"{BIN_NAME[(br, bc)]:>12}  (empty)")
            o("     " + " | ".join(cells))

    # ---- 4. centre vs periphery, raw and stratified -------------------------
    o("")
    o("4. CENTRE vs PERIPHERY")
    o("   'periphery' = any of the 8 non-centre 3x3 bins; 'corners' = the 4 corner bins.")
    o("   MH holds picture difficulty fixed (how many of the OTHER models got it right),")
    o("   crossed with the box's area quartile. It OVER-controls if position hurts every")
    o("   model at once -- read it next to the raw row, not instead of it.")

    # difficulty: per picture, how many other models were right
    right = collections.defaultdict(dict)
    for r in rows:
        c = _correct(r, label)
        if c is not None:
            right[r["key"]][r["model"]] = c
    areas = np.array([r["area_frac"] for r in rows])
    qs = np.quantile(areas, [0.25, 0.5, 0.75])

    o("")
    o(f"   {'model':<28} {'contrast':<10} {'centre':>14} {'other':>14} "
      f"{'diff':>8} {'Fisher p':>9} {'MH OR':>7} {'MH p':>8}")
    contrasts = (
        ("periphery", lambda r: tuple(r["bin3"]) != (1, 1), lambda r: True),
        ("corners", lambda r: r["bin3"][0] != 1 and r["bin3"][1] != 1, lambda r: True),
        ("periph/fin", lambda r: tuple(r["bin3"]) != (1, 1), lambda r: not r["unfinished"]),
    )
    for m in models:
        all_rs = [r for r in by_model[m] if _correct(r, label) is not None]
        for name, pick, keep in contrasts:
            rs = [r for r in all_rs if keep(r)]
            ctr = [r for r in rs if tuple(r["bin3"]) == (1, 1)]
            oth = [r for r in rs if pick(r)]
            if not ctr or not oth:
                continue
            a = sum(_correct(r, label) for r in ctr); b = len(ctr) - a
            c = sum(_correct(r, label) for r in oth); d = len(oth) - c
            p = ST.fisher_exact_2x2(a, b, c, d)
            tables = collections.defaultdict(lambda: [0, 0, 0, 0])
            for r in ctr + oth:
                others = [v for mm, v in right[r["key"]].items() if mm != m]
                st = (sum(others), int(np.searchsorted(qs, r["area_frac"])))
                is_ctr = tuple(r["bin3"]) == (1, 1)
                idx = (0 if is_ctr else 2) + (0 if _correct(r, label) else 1)
                tables[st][idx] += 1
            orr, mp = ST.mantel_haenszel([tuple(v) for v in tables.values()])
            ctr_s = f"{a}/{a + b}={a / (a + b):.3f}"
            oth_s = f"{c}/{c + d}={c / (c + d):.3f}"
            mh_s = (f"{orr:>7.2f} {mp:>8.3f}" if orr is not None
                    else f"{'--':>7} {'--':>8}")
            o(f"   {m:<28} {name:<10} {ctr_s:>14} {oth_s:>14} "
              f"{a / (a + b) - c / (c + d):>+8.3f} "
              f"{p if p is not None else float('nan'):>9.3f} {mh_s}")

    # ---- 5. the continuous version -----------------------------------------
    o("")
    o("5. THRESHOLD-FREE -- the box's radial depth against the judge's own 0-1 score")
    o("   depth_norm: 0 = centroid on the one-patch border, 1 = dead centre.")
    o("   A threshold throws away how far from the edge each picture sits; if the 3x3")
    o("   table shows something real this should too.")
    o("")
    o(f"   {'model':<28} {'rho(depth, score)':>18} {'p':>9} "
      f"{'AUC right>wrong':>16} {'p':>9}")
    for m in models:
        rs = [r for r in by_model[m] if _correct(r, label) is not None]
        dv = [r["judge"] if label == "judge" and r["judge"] is not None
              else float(_correct(r, label)) for r in rs]
        rho, pr = ST.spearman([r["depth_norm"] for r in rs], dv)
        dr = [r["depth_norm"] for r in rs if _correct(r, label)]
        dw = [r["depth_norm"] for r in rs if not _correct(r, label)]
        auc, pa = ST.mannwhitney(dr, dw)
        o(f"   {m:<28} {rho if rho is not None else float('nan'):>18.4f} "
          f"{pr if pr is not None else float('nan'):>9.4f} "
          f"{auc if auc is not None else float('nan'):>16.4f} "
          f"{pa if pa is not None else float('nan'):>9.4f}")

    # ---- 6. the covariate that could manufacture all of it ------------------
    o("")
    o("6. THE CONFOUND, priced")
    o("   If peripheral boxes are also SMALL boxes, a size effect wears a position")
    o("   effect's clothes. Both correlations are printed so the reader can see which.")
    o("")
    for m in models[:1]:
        rs = by_model[m]
        rho, p = ST.spearman([r["depth_norm"] for r in rs], [r["area_frac"] for r in rs])
        o(f"   rho(depth_norm, box area) = {rho:+.4f}  p = {p:.3g}   (geometry only, "
          f"model-independent up to the grid)")
    for m in models:
        rs = [r for r in by_model[m] if _correct(r, label) is not None]
        rho, p = ST.spearman([r["area_frac"] for r in rs],
                             [float(_correct(r, label)) for r in rs])
        o(f"   {m:<28} rho(box area, correct) = {rho:+.4f}  p = {p:.3g}")
    o("")
    o("   accuracy by box-area quartile (Q1 = smallest):")
    o(f"   {'model':<28} {'Q1':>12} {'Q2':>12} {'Q3':>12} {'Q4':>12}")
    for m in models:
        rs = [r for r in by_model[m] if _correct(r, label) is not None]
        cells = []
        for q in range(4):
            g = [r for r in rs if int(np.searchsorted(qs, r["area_frac"])) == q]
            cells.append(f"{np.mean([_correct(r, label) for r in g]):.3f} (n={len(g)})"
                         if g else "  --")
        o(f"   {m:<28} " + " ".join(f"{c:>12}" for c in cells))

    # ---- 7. power ----------------------------------------------------------
    o("")
    o("7. POWER -- what Arm 2 has to be sized for")
    o("   Using each model's own centre-vs-periphery gap as the effect to detect, and")
    o("   its centre accuracy as the base rate. Two-sided, alpha 0.05, power 0.80, two")
    o("   independent groups -- Arm 2 is PAIRED within picture, so its real requirement")
    o("   is lower than this by roughly the within-picture correlation.")
    o("")
    o(f"   {'model':<28} {'centre':>8} {'periph':>8} {'gap':>8} {'n per group':>12}")
    for m in models:
        rs = [r for r in by_model[m] if _correct(r, label) is not None]
        ctr = [r for r in rs if tuple(r["bin3"]) == (1, 1)]
        oth = [r for r in rs if tuple(r["bin3"]) != (1, 1)]
        if not ctr or not oth:
            continue
        p1 = float(np.mean([_correct(r, label) for r in ctr]))
        p2 = float(np.mean([_correct(r, label) for r in oth]))
        n = ST.n_per_group_for(p1, p2)
        o(f"   {m:<28} {p1:>8.3f} {p2:>8.3f} {p1 - p2:>+8.3f} "
          f"{n if n is not None else '-':>12}")

    o("")
    o("CAVEATS")
    o("  - Observational. Box position is not randomised and is confounded with what the")
    o("    question is about. Sections 4 and 6 close two nuisance routes, not the confound.")
    o("  - The four models answered the SAME pictures, so the four rows are not four")
    o("    independent tests; they are one test read four ways.")
    o("  - The judge does not see the image. It compares the extracted answer with the")
    o("    Visual-CoT gold string, which is what makes it fair across formats and also")
    o("    what makes it blind to a right answer phrased about a different object.")
    o("  - Completions were capped at 1024 tokens by the scan that produced them.")

    (out / "report.txt").write_text("\n".join(lines) + "\n")
    print(f"\n-> {out / 'report.txt'}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["build", "judge", "report"])
    ap.add_argument("--out-dir", default="outputs/box_position/arm0")
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--only", default="", help="comma-separated families to build")
    ap.add_argument("--judge-workers", type=int, default=8)
    ap.add_argument("--judge-cache", default="",
                    help="reuse a cache written by another probe; defaults to "
                         "OUT/judge_cache.json")
    ap.add_argument("--seed-from", default="",
                    help="glob of other *.judge_cache.json files to merge in first, e.g. "
                         "'outputs/human_box_correct/*.judge_cache.json'")
    ap.add_argument("--correct-on", default="judge", choices=["judge", "soft", "strict"])
    args = ap.parse_args()
    os.chdir(REPO)
    return {"build": stage_build, "judge": stage_judge, "report": stage_report}[args.stage](args)


if __name__ == "__main__":
    sys.exit(main())
