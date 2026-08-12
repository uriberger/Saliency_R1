#!/usr/bin/env python3
"""Would subsampling a step's target tokens survive as a GLIMPSE reward?

GLIMPSE costs 55-59x the gradient reward (glimpse_cost_probe.py), and the cost is exactly
linear in TARGET TOKENS, so the largest single lever is to score a subsample of each
observe step's tokens instead of all of them. Eq 22 restricted to a step is already a
weighted mean over those tokens,

    R~_V = sum_{t in S} beta_t . R(t, V),      beta_t ~ p_t . a_t, normalised over S

so a subsample is a ratio estimator of the same quantity. The question is not whether it
is biased -- at O(1/k) it barely is -- but whether the noise it injects is small next to
the signal the reward actually delivers.

THE COMPARISON THAT DECIDES IT. GRPO normalises within the group
(`advantages = (reward - group_mean) / (group_std + 1e-4)`), so a per-completion reward
reaches the gradient only through its spread ACROSS the completions of one prompt. That
is the same argument overlap_metric_spread.py makes for the weight calibration. So the
number to look at is

    sd of the per-completion score across SUBSAMPLE DRAWS      (the noise the cap adds)
    -------------------------------------------------------
    sd of the per-completion score across the GROUP's 8        (the signal it must not drown)

well under 1 and the cap is nearly free; near or above 1 and the cap turns the reward into
noise however good the map looks. The map correlation is reported too, but it is the
weaker question: the reward never sees the map, it sees `mean_in_v2` over a DINO union,
and a ratio of means over ~a third of the patches averages a lot of per-token noise away.

THE DRAW IS RANDOM, NOT THE FIRST k. A first-k cap would read every step's OPENING tokens,
and these maps carry a documented reading-order prior (rollout puts 5.8-8.8x of its mass
on the top row, monotone in sequence order), so first-k would bake that confound into the
reward. Uniform-random without replacement, seeded per (completion, step, k, draw).

Reported alongside, because the brief requires it and because it is nearly free here:
the full-step `mean_in_v2` itself (chance = 1.0) and the union area, which moves
`mean_in_v2` mechanically since its ceiling is n_patches / n_in.

    srun --jobid=$JOBID --overlap -n1 bash -lc '... python glimpse_token_cap_probe.py \
        --n-samples 2 --out outputs/glimpse_cost/token_cap.json'
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent


def repo_path(rel: str) -> Path:
    p = REPO / rel
    if p.exists():
        return p
    if REPO.parent.name == ".worktrees":
        alt = REPO.parent.parent / rel
        if alt.exists():
            return alt
    return p


def _load_module(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


SV = _load_module("_tc_saliency_viz", "saliency_viz.py")
PROBE, GM, OSTEPS = SV.PROBE, SV.GM, SV.OSTEPS
OREW, GREW = PROBE.OREW, PROBE.GREW


def roll_null(step_map, mask, k, rng):
    """`mean_in_v2` against translates of the SAME union -- same shape, same area.

    The brief's own instruction: detect "the reward moves for text-side reasons" rather
    than assume grounding. `mean_in_v2` alone cannot, because its ceiling is
    n_patches / n_in, so union area moves it mechanically and a step that grounds to half
    the image scores near 1.0 whatever the map does. Scoring the same map against
    translated copies of its own union holds area and shape fixed and leaves only
    location, which is the thing grounding is supposed to mean. `sample_offsets` is the
    gradient reward's own in-frame draw, so this is the same null it uses.
    """
    offs, _tor = GREW.sample_offsets(np.asarray(mask, bool), k, rng)
    vals = []
    for off in offs:
        v = OREW._mean_in_v2(step_map, np.roll(np.asarray(mask, bool), off, axis=(0, 1)))
        if v is not None:
            vals.append(float(v))
    return (float(np.median(vals)), len(vals)) if vals else (float("nan"), 0)


def subsample_map(maps: np.ndarray, w: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """The step's map from a subset of its tokens: beta renormalised inside the draw.

    Renormalising (rather than dividing by the full sum) is what makes this an estimator
    of the same weighted mean instead of a shrunken copy of it -- `mean_in_v2` is
    invariant to m -> c*m, so a shrunken copy would score identically and the comparison
    would be vacuous.
    """
    ws = w[idx]
    tot = float(ws.sum())
    if tot <= 0:
        return maps[idx].mean(axis=0)
    return np.tensordot(ws / tot, maps[idx], axes=(0, 0))


def corr(a: np.ndarray, b: np.ndarray) -> float:
    x, y = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    x, y = x - x.mean(), y - y.mean()
    d = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(x @ y / d) if d > 0 else float("nan")


def run(args, device):
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    OREW.configure(metric="mean_in_v2", box_threshold=args.box_threshold,
                   max_box_area=args.max_box_area, max_union_area=args.max_union_area,
                   dino_device=args.dino_device or device)
    rows = PROBE.load_samples(args.dataset, args.n_samples, args.seed,
                              cache_tag="_tcap", split=args.split)
    processor, model = PROBE.load_model(args.base_model, args.adapter or None, device,
                                        args.attn_impl)
    model.requires_grad_(False)
    clf = OSTEPS.OverlapStepsClassifier.load(args.steps_ckpt, device=device)
    tok = processor.tokenizer
    ks = [int(x) for x in args.ks.split(",") if x]
    gargs = argparse.Namespace(
        glimpse_temp=args.glimpse_temp, glimpse_depth_temp=args.glimpse_depth_temp,
        glimpse_layer_frac=args.glimpse_layer_frac, glimpse_target=args.glimpse_target,
        glimpse_token_weight=args.glimpse_token_weight)

    groups = []
    for si, row in enumerate(rows):
        inputs, prompt_len, seqs = PROBE.generate(
            processor, model, row["image"], row["question"], args.num_generations,
            args.max_new_tokens, args.temperature, device)
        gh = int(inputs["image_grid_thw"][0, 1].item()) // 2
        gw = int(inputs["image_grid_thw"][0, 2].item()) // 2
        torch.cuda.empty_cache()

        # per completion: the full-step maps, and eq 22's terms behind each of them
        per_comp = []
        for c, (comp_ids, _trunc) in enumerate(seqs):
            _text, seg, reason = SV.segment_case(tok, clf, row["question"], comp_ids)
            if seg is None:
                print(f"[cap] s{si} c{c}: skipped ({reason})", flush=True)
                continue
            steps, _fmt = seg
            ids = torch.tensor([inputs["input_ids"][0].tolist() + list(comp_ids)],
                               device=device)
            collect = []
            try:
                with GM.frozen_params(model):
                    maps, _info = SV.glimpse_map(model, processor, inputs, ids, prompt_len,
                                                 steps, gh, gw, row["question"], gargs,
                                                 device, collect=collect)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"[cap] s{si} c{c}: OOM", flush=True)
                continue
            finally:
                del ids
                torch.cuda.empty_cache()
            # The gradient map on the SAME steps, to be scored against the SAME masks with
            # the SAME metric. Without it a near-chance GLIMPSE score is ambiguous between
            # "GLIMPSE is not grounded" and "these steps are not groundable" -- the maps
            # come from the model's own text, and a step like "let me think about this"
            # grounds to a box that means nothing. The incumbent map is the reference that
            # separates the two, and it costs 0.25 s per case.
            gmaps = None
            try:
                ip = processor.image_processor
                case = PROBE.teacher_forced_case(inputs, comp_ids, device)
                spans = [(prompt_len + a, prompt_len + b) for _t, a, b in steps]
                with GM.frozen_params(model):
                    gmaps = GM.step_grad_maps(
                        model, case, spans, inputs["image_grid_thw"][0].tolist(),
                        int(getattr(ip, "patch_size", 16)),
                        int(getattr(ip, "temporal_patch_size", 2)),
                        target=args.glimpse_target)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
            per_comp.append({"completion": c, "steps": steps, "maps": maps,
                             "grad_maps": gmaps, "collect": collect})
            print(f"[cap] s{si} c{c}: {len(steps)} steps, {len(collect)} target tokens",
                  flush=True)

        # One batched DINO call over every (completion, step) of this prompt, the way
        # think_grad_reward grounds a batch.
        flat_imgs, flat_txts, owner = [], [], []
        for pi, pc in enumerate(per_comp):
            for stepi, (text, _a, _b) in enumerate(pc["steps"]):
                flat_imgs.append(row["image"])
                flat_txts.append(text)
                owner.append((pi, stepi))
        boxes = OREW._dino_boxes(flat_imgs, flat_txts) if flat_imgs else []

        rng = np.random.default_rng(args.seed)
        comp_records = []
        for pi, pc in enumerate(per_comp):
            comp_records.append({"completion": pc["completion"], "steps": []})
        for (pi, stepi), bx in zip(owner, boxes):
            pc = per_comp[pi]
            full = pc["maps"][stepi]
            mask = OREW._union_mask(bx, gh, gw)
            if mask is None:
                continue
            terms = [(w, m) for (s, w, m) in pc["collect"] if s == stepi]
            if not terms:
                continue
            w = np.array([t[0] for t in terms], dtype=np.float64)
            tm = np.stack([t[1] for t in terms]).astype(np.float64)
            full_score = OREW._mean_in_v2(full, mask)
            if full_score is None:
                continue

            null, n_off = roll_null(full, mask, args.null_offsets, rng)
            rec = {"step": stepi, "n_tokens": len(terms),
                   "union_frac": float(mask.mean()), "full_score": float(full_score),
                   "ceiling": float(mask.size / max(int(mask.sum()), 1)),
                   "roll_null": null, "n_offsets": n_off,
                   "logratio": GREW.step_logratio(full, mask, rng), "k": {}}
            gm = pc.get("grad_maps")
            if gm is not None and stepi < len(gm):
                g_null, _n = roll_null(gm[stepi], mask, args.null_offsets, rng)
                rec["grad_score"] = OREW._mean_in_v2(gm[stepi], mask)
                rec["grad_roll_null"] = g_null
                rec["grad_logratio"] = GREW.step_logratio(gm[stepi], mask, rng)
            # how much the per-token maps disagree with each other at all
            if len(terms) > 1:
                pw = [corr(tm[i], tm[j]) for i in range(len(tm))
                      for j in range(i + 1, len(tm))]
                rec["token_map_corr"] = float(np.median(pw))
            for k in ks:
                if k >= len(terms):
                    continue
                scores, corrs = [], []
                for _d in range(args.draws):
                    idx = rng.choice(len(terms), size=k, replace=False)
                    sm = subsample_map(tm, w, idx)
                    s = OREW._mean_in_v2(sm, mask)
                    if s is None:
                        continue
                    scores.append(float(s))
                    corrs.append(corr(sm, full))
                if scores:
                    rec["k"][str(k)] = {"scores": scores,
                                        "corr_to_full": float(np.median(corrs))}
            comp_records[pi]["steps"].append(rec)

        groups.append({"sample": si, "row_index": row["row_index"],
                       "question": row["question"], "completions": comp_records})
        for pc in per_comp:
            pc.pop("maps", None)
            pc.pop("grad_maps", None)
            pc.pop("collect", None)
        torch.cuda.empty_cache()
    return {"config": vars(args), "groups": groups}


def summarise(res, ks, draws):
    """The per-completion reward, its draw-to-draw noise, and the group spread it must
    not drown. The completion reward is `think_*_reward`'s own shape: the mean over the
    completion's grounded observe steps."""
    out, all_steps = [], []
    for g in res["groups"]:
        for c in g["completions"]:
            all_steps.extend(c["steps"])
    if not all_steps:
        return "no grounded steps"

    fs = np.array([s["full_score"] for s in all_steps])
    uf = np.array([s["union_frac"] for s in all_steps])
    ce = np.array([s["ceiling"] for s in all_steps])
    tc = [s["token_map_corr"] for s in all_steps if "token_map_corr" in s]
    out.append(f"{len(all_steps)} grounded observe steps over "
               f"{sum(len(g['completions']) for g in res['groups'])} completions")
    out.append(f"mean_in_v2 on the FULL step: median {np.median(fs):.3f}  "
               f"(chance 1.0, median ceiling {np.median(ce):.1f})   "
               f"union area median {np.median(uf):.2f}")

    # Is it grounded at all? Both maps, the SAME steps, the SAME masks, the SAME metric,
    # each against translates of the step's own union -- same area, same shape, only the
    # location differs. The grad row is the reference that separates "this map is not
    # grounded" from "these steps are not groundable".
    out.append("")
    out.append(f"{'map':<8} {'mean_in_v2':>11} {'roll-null':>10} {'d mean':>9} "
               f"{'sem':>8} {'d med':>8} {'beats null':>12} {'logratio':>10} {'sem':>8}")
    for label, sk, nk, lk in (("glimpse", "full_score", "roll_null", "logratio"),
                              ("grad", "grad_score", "grad_roll_null", "grad_logratio")):
        pr = [(s[sk], s[nk]) for s in all_steps
              if s.get(sk) is not None and np.isfinite(s.get(nk, np.nan))]
        if not pr:
            continue
        d = np.array([a - b for a, b in pr])
        sem = float(np.std(d, ddof=1) / np.sqrt(d.size)) if d.size > 1 else float("nan")
        lr = np.array([s[lk] for s in all_steps if s.get(lk) is not None])
        lsem = (float(lr.std(ddof=1) / np.sqrt(lr.size)) if lr.size > 1 else float("nan"))
        out.append(f"{label:<8} {np.median([a for a, _b in pr]):>11.3f} "
                   f"{np.median([b for _a, b in pr]):>10.3f} {np.mean(d):>+9.4f} "
                   f"{sem:>8.4f} {np.median(d):>+8.4f} {int((d > 0).sum()):>6}/{len(pr):<5} "
                   f"{(np.median(lr) if lr.size else float('nan')):>+10.4f} {lsem:>8.4f}")
    out.append("chance is 1.0 for mean_in_v2 and 0 for the log-ratio; 'paired d' is the "
               "step's own union\nminus its translates, so it is the part of the score "
               "that is about LOCATION and not area.")
    out.append(f"per-token maps inside a step: median pairwise corr "
               f"{np.median(tc):.3f}" if tc else "")
    out.append(f"median tokens per step {int(np.median([s['n_tokens'] for s in all_steps]))}")

    # per-completion reward = mean over grounded steps; draw d uses draw d of every step
    out.append("")
    out.append(f"{'k':>4} {'steps':>6} {'corr to full':>13} {'score bias':>11} "
               f"{'draw sd':>9} {'group sd':>9} {'draw/group':>11}")
    group_sds = []
    for g in res["groups"]:
        vals = [np.mean([s["full_score"] for s in c["steps"]])
                for c in g["completions"] if c["steps"]]
        if len(vals) > 1:
            group_sds.append(float(np.std(vals, ddof=1)))
    gsd = float(np.median(group_sds)) if group_sds else float("nan")

    for k in ks:
        kk = str(k)
        used = [s for s in all_steps if kk in s["k"]]
        if not used:
            continue
        cf = float(np.median([s["k"][kk]["corr_to_full"] for s in used]))
        bias = float(np.median([np.mean(s["k"][kk]["scores"]) - s["full_score"]
                                for s in used]))
        draw_sds = []
        for g in res["groups"]:
            for c in g["completions"]:
                ss = [s for s in c["steps"] if kk in s["k"]]
                if not ss:
                    continue
                n = min(len(s["k"][kk]["scores"]) for s in ss)
                # completion reward per draw: mean over this completion's steps
                per_draw = [float(np.mean([s["k"][kk]["scores"][d] for s in ss]))
                            for d in range(n)]
                if len(per_draw) > 1:
                    draw_sds.append(float(np.std(per_draw, ddof=1)))
        dsd = float(np.median(draw_sds)) if draw_sds else float("nan")
        out.append(f"{k:>4} {len(used):>6} {cf:>13.3f} {bias:>+11.3f} {dsd:>9.4f} "
                   f"{gsd:>9.4f} {dsd / gsd:>11.2f}")
    out.append("")
    out.append("draw/group is the number that decides: the noise a k-token cap injects "
               "into a completion's\nreward, over the spread across the group's "
               "completions that is the only thing GRPO can see.")
    return "\n".join(x for x in out if x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=str(repo_path("cold_data/grpo_sets/set_a")))
    p.add_argument("--split", default="train", choices=["train", "holdout", "all"])
    p.add_argument("--n-samples", type=int, default=2)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--base-model",
                   default=str(repo_path("checkpoint/coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged")))
    p.add_argument("--adapter", default="")
    p.add_argument("--num-generations", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--attn-impl", default="sdpa")
    p.add_argument("--steps-ckpt", default=os.environ.get(
        "OVERLAP_STEPS_CKPT", str(repo_path("checkpoint/steps_classifier/best"))))
    p.add_argument("--ks", default="1,2,4,6,8")
    p.add_argument("--draws", type=int, default=32)
    p.add_argument("--null-offsets", type=int, default=16,
                   help="translates of the union forming the grounding control "
                        "(grad_null_offsets' default)")
    p.add_argument("--box-threshold", type=float, default=0.10)
    p.add_argument("--max-box-area", type=float, default=0.5)
    p.add_argument("--max-union-area", type=float, default=None)
    p.add_argument("--dino-device", default=None)
    p.add_argument("--glimpse-temp", type=float, default=0.5)
    p.add_argument("--glimpse-depth-temp", type=float, default=0.2)
    p.add_argument("--glimpse-layer-frac", type=float, default=1.0)
    p.add_argument("--glimpse-target", default="clogit", choices=list(GM.GRAD_TARGETS))
    p.add_argument("--glimpse-token-weight", default="full",
                   choices=["full", "confidence", "prompt", "uniform"])
    p.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default=str(REPO / "outputs/glimpse_cost/token_cap.json"))
    args = p.parse_args()

    res = run(args, args.device)
    ks = [int(x) for x in args.ks.split(",") if x]
    text = summarise(res, ks, args.draws)
    res["summary"] = text
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, default=str))
    print("\n" + text + f"\n\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
