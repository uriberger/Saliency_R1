#!/usr/bin/env python
"""Does draining the border sink into the middle, at inference, change the answer?

`sink_shift.py` is the edit; this is the harness that runs it and measures it.

    python sink_shift_probe.py --stage selftest --out-dir DIR --model M
    python sink_shift_probe.py --stage survey   --out-dir DIR --model M
    python sink_shift_probe.py --stage run      --out-dir DIR --model M --shard i --num-shards n
    python sink_shift_probe.py --stage report   --out-dir DIR
    python sink_shift_probe.py --stage monitor  --out-dir DIR

STAGES

  selftest  Gates every run, and must pass.
              - alpha=0 reproduces the un-hooked greedy generation TOKEN FOR TOKEN. This
                is the one check that says the custom attention path is the stock one
                plus an edit, rather than a second model.
              - uninstalling puts the model back: the same prompts give the same words
                again, so a run cannot be contaminated by a previous arm.
              - at alpha>0 the border's share of the picture's attention falls to
                (1-alpha) of what it was, measured on real forwards.
            A run whose selftest was skipped is not evidence.

  survey    Per layer and per head: what share of an attention row lands on the picture,
            and how much of that sits on the border. No edit is applied. This decides
            where the edit has any leverage at all -- at layer 22 heads 28/31 the whole
            picture receives 0.4-1.4% of a row, so emptying its border moves ~0.2-0.7%
            of two heads out of 1,152, and a null there says nothing about the idea.

  run       The (arm, alpha) grid over a validation set, greedily, one row at a time,
            graded with the same `accuracy_reward` the training curves used. Append-only
            JSONL, resumable, shardable across GPUs.

  report    Accuracy per cell, paired against the alpha=0 baseline over the same rows,
            with a bootstrap CI. Every cell carries its landing check beside it.

WHY GREEDY, AND WHY PAIRED. Temperature 0 means a change between two cells is a change
in the model's behaviour and not in the sampling draw. The baseline is this harness's own
alpha=0 run and not the validation curve from training, because that curve was produced
by vLLM and this is HuggingFace: the two differ numerically for reasons that have nothing
to do with the edit, and pairing within one harness removes the question.

WHAT THE ARMS ANSWER. `centre` minus `outward` is the result -- same source, same mass
moved, opposite destinations, so a difference is about WHERE the attention went. `flat`
asks whether evenness alone does it, which matters because the reward this replaces
correlates 0.932 with the box-blind `flatness` statistic. `reverse` is the sign check.
`text` moves the same mass among the words instead, so a gain there would mean the size
of the nudge was the whole story.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent


def _load_module(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


PROBE = _load_module("_ss_overlap_probe", "overlap_probe.py")
IV = _load_module("_ss_intervene", "intervene_probe.py")
sys.path.insert(0, str(REPO))
import sink_shift as SS  # noqa: E402

# The rewarded pair, from the training runs: layer 22, heads 28 and 31, merged by mean.
TRAINED_LAYER, TRAINED_HEADS = 22, (28, 31)
DEFAULT_ARMS = ("centre", "outward", "flat", "reverse", "text")
DEFAULT_ALPHAS = (0.25, 0.5, 0.75, 1.0)
BASELINE_ARM = "none"          # alpha = 0; identical for every arm, so it is run once


# ---------------------------------------------------------------------------
# the grid
# ---------------------------------------------------------------------------
def build_units(args):
    """[(arm, alpha)] -- the baseline once, then every arm at every non-zero alpha.

    alpha=0 is the identity for all six arms, so running it per arm would buy six
    identical numbers and six times the GPU hours.
    """
    units = [(BASELINE_ARM, 0.0)]
    for arm in args.arms:
        for a in args.alphas:
            if a > 0:
                units.append((arm, float(a)))
    return units


def unit_key(split, arm, alpha, row_index):
    return f"{split}|{arm}|{alpha:g}|{row_index}"


def scope_of(args):
    """The (layers, heads) the edit is confined to, and a short name for the run.

    `trained` is the two heads the reward shaped -- the narrow arm, run first because
    it is the configuration the training result came from. `all` is every layer and
    every head, which is the configuration the one intervention that ever moved this
    model used (`flow_intervene_probe.py`, 0.726 nats).
    """
    if args.scope == "trained":
        return [TRAINED_LAYER], list(TRAINED_HEADS), "L22h28-31"
    if args.scope == "all":
        return None, None, "all"
    layers = None if not args.layers else IV.parse_layers(args.layers, 64)
    heads = None if not args.heads else [int(x) for x in args.heads.split(",")]
    name = f"L{args.layers or 'all'}h{args.heads or 'all'}".replace(",", "-")
    return layers, heads, name


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------
def greedy(processor, model, image, question, max_new_tokens, device):
    """One greedy completion, at batch size 1 -- which is what the edit requires."""
    import torch

    text = PROBE.build_prompt(processor, question)
    inputs = processor(text=[text], images=[[image]], return_tensors="pt",
                       padding=True, padding_side="left",
                       add_special_tokens=False).to(device)
    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens,
                             pad_token_id=processor.tokenizer.pad_token_id)
    ids = out[0][prompt_len:].tolist()
    return processor.tokenizer.decode(ids, skip_special_tokens=True), ids


def grade(text, solution):
    """The trainer's accuracy grader, and whether the answer kept its shape.

    `accuracy_reward` lives at module level in overlap_probe.py and is imported rather
    than copied, so this harness grades exactly what the reward curves graded.
    """
    acc = PROBE.accuracy_reward([[{"role": "assistant", "content": text}]], [solution])[0]
    return acc, bool(re.match(PROBE.FORMAT_PATTERN, text, re.DOTALL))


# ---------------------------------------------------------------------------
# stage: run
# ---------------------------------------------------------------------------
def load_rows(args):
    """{split: [row]} -- the validation sets, prepared exactly as overlap_probe does."""
    out = {}
    for split in args.splits:
        path = os.path.join(args.val_sets_dir, split)
        rows = PROBE.load_samples(path, args.rows_per_split, args.seed,
                                  cache_tag=f"_ss{args.shard}", split="all")
        out[split] = rows
        print(f"[rows] {split}: {len(rows)} rows from {path}", flush=True)
    return out


def done_keys(path):
    """Every unit already in the results file. Resume is by KEY, not by line count."""
    seen = set()
    if not path.exists():
        return seen
    with open(path) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue          # a torn last line from a killed job is not a result
            seen.add(unit_key(r["split"], r["arm"], r["alpha"], r["row_index"]))
    return seen


def run_shard(args):
    import torch

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = out_dir / f"results_shard{args.shard}.jsonl"
    seen = done_keys(results)

    rows_by_split = load_rows(args)
    units = build_units(args)
    layers, heads, scope_name = scope_of(args)

    # Progress is counted against THIS grid, not against the length of the results file.
    # A shard whose out-dir also holds an earlier grid's results would otherwise report a
    # percentage over 100 and a nonsense ETA -- the cosmetic bug docs/HANDOFF.md records
    # as having misled twice.
    grid = [(split, arm, alpha, r["row_index"])
            for split, rows in rows_by_split.items()
            for arm, alpha in units
            for r in rows[args.shard::args.num_shards]]
    already = sum(1 for g in grid if unit_key(*g) in seen)
    prog = IV.Progress(out_dir / "progress" / f"run{args.shard}.json",
                       len(grid), f"run/{args.shard}", already_done=already)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, model = PROBE.load_model(args.model, args.adapter, device, "sdpa")
    print(f"[run] scope {scope_name}: layers={layers} heads={heads}", flush=True)

    with open(results, "a") as fh:
        for split, rows in rows_by_split.items():
            mine = rows[args.shard::args.num_shards]
            for arm, alpha in units:
                pending = [r for r in mine
                           if unit_key(split, arm, alpha, r["row_index"]) not in seen]
                if not pending:
                    continue
                ss = None
                if alpha > 0:
                    ss = SS.install(model, arm=arm, alpha=alpha, layers=layers,
                                    heads=heads, rows=args.rows,
                                    rect_frac=args.rect_frac)
                try:
                    for r in pending:
                        t0 = time.time()
                        if ss is not None:
                            ss.reset_diagnostics()
                        text, ids = greedy(processor, model, r["image"], r["question"],
                                           args.max_new_tokens, device)
                        acc, fmt = grade(text, r["gt_answer"])
                        rec = {
                            "split": split, "arm": arm, "alpha": alpha,
                            "row_index": r["row_index"], "scope": scope_name,
                            "accuracy": acc, "format_valid": fmt,
                            "n_tokens": len(ids), "seconds": time.time() - t0,
                            "diag": (ss.diagnostics() if ss is not None else None),
                            "text": text if args.store_text else None,
                        }
                        fh.write(json.dumps(rec) + "\n")
                        fh.flush()
                        prog.tick()
                finally:
                    if ss is not None:
                        ss.uninstall()
    prog.close()


# ---------------------------------------------------------------------------
# stage: selftest
# ---------------------------------------------------------------------------
def selftest(args):
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, model = PROBE.load_model(args.model, args.adapter, device, "sdpa")
    args.rows_per_split = args.selftest_rows      # do not decode 256 images to use 4
    rows = load_rows(args)[args.splits[0]][:args.selftest_rows]
    layers, heads, scope_name = scope_of(args)
    ok = True

    def gen(r):
        return greedy(processor, model, r["image"], r["question"],
                      args.selftest_tokens, device)[1]

    print(f"\nselftest  model={args.model}  scope={scope_name}  "
          f"{len(rows)} rows x {args.selftest_tokens} tokens", flush=True)

    base = [gen(r) for r in rows]

    ss = SS.install(model, arm="centre", alpha=0.0, layers=layers, heads=heads,
                    rows=args.rows, rect_frac=args.rect_frac)
    try:
        zero = [gen(r) for r in rows]
    finally:
        ss.uninstall()
    same = sum(a == b for a, b in zip(base, zero))
    ok &= same == len(rows)
    print(f"  {'PASS' if same == len(rows) else 'FAIL'}  alpha=0 reproduces the "
          f"un-hooked generation: {same}/{len(rows)} identical token sequences")

    after = [gen(r) for r in rows]
    back = sum(a == b for a, b in zip(base, after))
    ok &= back == len(rows)
    print(f"  {'PASS' if back == len(rows) else 'FAIL'}  uninstall restores the model: "
          f"{back}/{len(rows)} identical")

    for alpha in (0.5, 1.0):
        ss = SS.install(model, arm="centre", alpha=alpha, layers=layers, heads=heads,
                        rows=args.rows, rect_frac=args.rect_frac)
        try:
            ss.reset_diagnostics()
            changed = sum(int(gen(r) != base[i]) for i, r in enumerate(rows))
            d = ss.diagnostics()
        finally:
            ss.uninstall()
        want = (1 - alpha) * d["frame_share_before"]
        landed = abs(d["frame_share_after"] - want) < 1e-4 and d["rows_edited"] > 0
        ok &= landed
        print(f"  {'PASS' if landed else 'FAIL'}  alpha={alpha}: border share "
              f"{d['frame_share_before']:.4f} -> {d['frame_share_after']:.4f} "
              f"(want {want:.4f}), {d['rows_edited']} rows on layers "
              f"{d['layers_touched']}")
        print(f"        picture gets {d['image_mass']:.5f} of the row; this edit moves "
              f"{d['row_mass_moved']:.5f} of it; answers changed on "
              f"{changed}/{len(rows)} prompts")

    print(f"\n  {'SELFTEST PASS' if ok else 'SELFTEST FAIL'}")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# stage: survey
# ---------------------------------------------------------------------------
def survey(args):
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, model = PROBE.load_model(args.model, args.adapter, device, "sdpa")
    args.rows_per_split = args.survey_rows
    rows = load_rows(args)[args.splits[0]][:args.survey_rows]

    ss = SS.install(model, arm="centre", alpha=0.0, rows=args.rows,
                    rect_frac=args.rect_frac, survey=True)
    try:
        for r in rows:
            greedy(processor, model, r["image"], r["question"], args.survey_tokens, device)
        table = SS.survey_table(ss)
    finally:
        ss.uninstall()

    out = Path(args.out_dir) / "survey.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": args.model, "rows": len(rows),
                               "tokens": args.survey_tokens, "table": table}, indent=1))
    print_survey(table)
    print(f"\nwritten to {out}")
    return 0


def print_survey(table):
    """Where the edit has leverage, and where it cannot possibly have any."""
    print("\nHow much of an attention row lands on the PICTURE, per layer\n")
    print(f"  {'layer':>5} {'img mass':>10} {'best head':>10} {'that head':>10} "
          f"{'border share':>13} {'movable':>9}")
    tot = []
    for layer, v in table.items():
        im = np.asarray(v["image_mass"])
        fs = np.asarray(v["frame_share"])
        h = int(im.argmax())
        movable = float((im * fs).max())
        tot.append((layer, float(im.mean()), h, float(im[h]), float(fs.mean()), movable))
    for layer, mean_im, h, best, fs, movable in tot:
        print(f"  {layer:>5} {mean_im:>10.5f} {h:>10} {best:>10.5f} {fs:>13.3f} "
              f"{movable:>9.5f}")
    arr = np.asarray([t[5] for t in tot])
    print(f"\n  'movable' is image mass x border share for the best head of the layer: "
          f"the largest\n  fraction of one attention row this edit could shift there at "
          f"alpha=1.")
    print(f"  Across layers it runs {arr.min():.5f} to {arr.max():.5f}, best at layer "
          f"{tot[int(arr.argmax())][0]}.")
    if TRAINED_LAYER in table:
        v = table[TRAINED_LAYER]
        for h in TRAINED_HEADS:
            if h < len(v["image_mass"]):
                print(f"  The rewarded head L{TRAINED_LAYER}h{h}: image mass "
                      f"{v['image_mass'][h]:.5f}, border share {v['frame_share'][h]:.3f}, "
                      f"movable {v['image_mass'][h] * v['frame_share'][h]:.5f}")


# ---------------------------------------------------------------------------
# stage: report
# ---------------------------------------------------------------------------
def read_results(out_dir):
    recs = []
    for p in sorted(Path(out_dir).glob("results_shard*.jsonl")):
        with open(p) as fh:
            for line in fh:
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return recs


def paired_delta(cell, base, n_boot=2000, seed=20260907):
    """Mean of cell - base over the rows BOTH ran, with a bootstrap CI over rows.

    Restricted to shared rows because a cell that crashed on some prompts would
    otherwise be compared against a baseline that answered them, and the difference
    would be about which prompts were attempted.
    """
    shared = sorted(set(cell) & set(base))
    if len(shared) < 2:
        return float("nan"), float("nan"), float("nan"), 0
    d = np.asarray([cell[i] - base[i] for i in shared], float)
    d = d[np.isfinite(d)]
    if d.size < 2:
        return float("nan"), float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, d.size, size=(n_boot, d.size))].mean(axis=1)
    return (float(d.mean()), float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)), int(d.size))


def report(args):
    recs = read_results(args.out_dir)
    if not recs:
        print(f"no results under {args.out_dir}")
        return 1
    scopes = sorted({r.get("scope", "?") for r in recs})
    print(f"{len(recs)} records from {args.out_dir}   scope(s): {', '.join(scopes)}")

    by = {}
    for r in recs:
        by.setdefault((r["split"], r["arm"], r["alpha"]), []).append(r)

    for split in sorted({k[0] for k in by}):
        base = by.get((split, BASELINE_ARM, 0.0), [])
        if not base:
            print(f"\n=== {split}: no alpha=0 baseline, nothing to pair against")
            continue
        b_acc = {r["row_index"]: r["accuracy"] for r in base
                 if r["accuracy"] is not None}
        print(f"\n=== {split}   baseline (alpha=0): "
              f"accuracy {np.mean(list(b_acc.values())):.4f} over {len(b_acc)} rows, "
              f"format valid {np.mean([r['format_valid'] for r in base]):.3f}, "
              f"mean length {np.mean([r['n_tokens'] for r in base]):.0f}")
        print(f"    {'arm':<9} {'alpha':>5} {'acc':>7} {'-base':>8} {'95% CI':>20} "
              f"{'n':>4} {'fmt':>6} {'len':>6} {'ungr':>6} {'border':>15} {'moved':>8}")
        for (sp, arm, alpha), rs in sorted(by.items()):
            if sp != split or arm == BASELINE_ARM:
                continue
            acc = {r["row_index"]: r["accuracy"] for r in rs
                   if r["accuracy"] is not None}
            m, lo, hi, n = paired_delta(acc, b_acc, args.n_boot, args.seed)
            ungraded = np.mean([r["accuracy"] is None for r in rs])
            fmt = np.mean([r["format_valid"] for r in rs])
            ln = np.mean([r["n_tokens"] for r in rs])
            d = [r["diag"] for r in rs if r.get("diag")]
            border = (f"{np.mean([x['frame_share_before'] for x in d]):.3f}"
                      f"->{np.mean([x['frame_share_after'] for x in d]):.3f}"
                      if d else "-")
            moved = f"{np.mean([x['row_mass_moved'] for x in d]):.5f}" if d else "-"
            flag = "" if fmt >= args.min_format else "  <- BROKEN, do not read"
            print(f"    {arm:<9} {alpha:>5.2f} {np.mean(list(acc.values())):>7.4f} "
                  f"{m:>+8.4f} {f'[{lo:+.4f}, {hi:+.4f}]':>20} {n:>4} {fmt:>6.3f} "
                  f"{ln:>6.0f} {ungraded:>6.3f} {border:>15} {moved:>8}{flag}")

        print("\n    the result is centre - outward at matched alpha, not centre alone:")
        for alpha in sorted({k[2] for k in by if k[0] == split and k[2] > 0}):
            c = {r["row_index"]: r["accuracy"]
                 for r in by.get((split, "centre", alpha), []) if r["accuracy"] is not None}
            o = {r["row_index"]: r["accuracy"]
                 for r in by.get((split, "outward", alpha), []) if r["accuracy"] is not None}
            if not c or not o:
                continue
            m, lo, hi, n = paired_delta(c, o, args.n_boot, args.seed)
            print(f"      alpha={alpha:.2f}   centre - outward = {m:+.4f} "
                  f"[{lo:+.4f}, {hi:+.4f}]   over {n} rows")
    print(f"\n  A cell whose format-valid rate fell below {args.min_format} is off the "
          "manifold,\n  not an effect: the model stopped writing answers in the shape "
          "the grader reads.")
    return 0


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True,
                    choices=["selftest", "survey", "run", "report", "monitor"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default=None, help="base model or merged checkpoint")
    ap.add_argument("--adapter", default=None, help="a PEFT adapter to merge in memory")
    ap.add_argument("--val-sets-dir",
                    default=str(REPO / "cold_data" / "grpo_sets"),
                    help="holds val_natural/ and val_nonnatural/; pass an absolute path "
                         "to the central tree when running from a worktree")
    ap.add_argument("--splits", default="val_natural,val_nonnatural")
    ap.add_argument("--rows-per-split", type=int, default=256)
    ap.add_argument("--scope", default="trained", choices=["trained", "all", "custom"],
                    help="trained = layer 22 heads 28/31; all = every layer and head")
    ap.add_argument("--layers", default=None, help="with --scope custom, e.g. 0-21 or 22")
    ap.add_argument("--heads", default=None, help="with --scope custom, e.g. 28,31")
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    ap.add_argument("--alphas", default=",".join(str(a) for a in DEFAULT_ALPHAS))
    ap.add_argument("--rows", default="after_image", choices=list(SS.ROWS_CHOICES),
                    help="which query positions are edited")
    ap.add_argument("--rect-frac", type=float, default=SS.RECT_FRAC)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--store-text", action="store_true",
                    help="keep every completion in the results file (large)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20260907)
    ap.add_argument("--selftest-rows", type=int, default=4)
    ap.add_argument("--selftest-tokens", type=int, default=64)
    ap.add_argument("--survey-rows", type=int, default=4)
    ap.add_argument("--survey-tokens", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--min-format", type=float, default=0.9,
                    help="below this a cell is broken generation, not a result")
    ap.add_argument("--interval", type=float, default=30.0, help="monitor poll seconds")
    ap.add_argument("--once", action="store_true", help="monitor: print once and exit")
    args = ap.parse_args()
    args.splits = [s for s in args.splits.split(",") if s]
    args.arms = [a for a in args.arms.split(",") if a]
    args.alphas = [float(a) for a in args.alphas.split(",") if a]
    bad = [a for a in args.arms if a not in SS.ARMS]
    if bad:
        raise SystemExit(f"unknown arm(s) {bad}; have {sorted(SS.ARMS)}")

    if args.stage == "report":
        return report(args)
    if args.stage == "monitor":
        IV.monitor(Path(args.out_dir), args.interval, args.once, "run")
        return 0
    if not args.model:
        raise SystemExit("--model is required for this stage")
    if args.stage == "selftest":
        return selftest(args)
    if args.stage == "survey":
        return survey(args)
    run_shard(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
