#!/usr/bin/env python
"""Measure the per-sample SPREAD of each overlap metric from a probe run, and derive the
w_overlap that reproduces a reference metric's training pressure.

Why spread and not mean: GRPO normalises within the group
(``advantages = (reward - group_mean) / (group_std + 1e-4)``, scale_rewards defaults
True), so a constant offset in a reward term does nothing and only its VARIATION across
the completions of one prompt reaches the gradient. Two metrics with the same mean but
different sd apply different pressure at the same w_overlap. This is exactly how auroc's
0.11 was derived from mean_in's 0.4 (sd 0.13 vs 0.036 -> 0.4 x 0.036/0.13 ~ 0.11), and
it is the number a new metric needs before it can be trained with.

The probe records every metric ungated on every grounded step (mean_in_raw,
mean_in_v2_raw, auroc_raw), so all of them are measured on IDENTICAL maps, masks and
completions -- a paired comparison, not one across runs that generated different text.

    python overlap_metric_spread.py outputs/overlap_probe/<run>/probe_merged.json

The per-completion reward is reconstructed the way think_overlap_reward builds it:
mean over the completion's grounded observe steps, times the format gate, None when
there is no grounded step (masked, so it never enters the group). The optional mass
floor is NOT applied -- it is a per-step multiplier in [0,1] that would confound a
comparison of the metrics themselves.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# (json key on a step, display name). The probe writes all of these per grounded step.
METRICS = [
    ("mean_in_raw", "mean_in"),
    ("mean_in_v2_raw", "mean_in_v2"),
    ("auroc_raw", "auroc"),
]
CHANCE = {"mean_in": None, "mean_in_v2": 1.0, "auroc": 0.5}


def completion_reward(completion, key):
    """think_overlap_reward for one completion under one metric, mass floor off."""
    vals = [s[key] for s in completion["observe_steps"]
            if s.get("grounded") and s.get(key) is not None]
    if not vals:
        return None                                    # masked: no grounded step
    return float(np.mean(vals)) * (1.0 if completion["format_valid"] else 0.0)


def summarise(samples, key, name):
    per_step, per_completion, group_sds = [], [], []
    for s in samples:
        vals = []
        for c in s["completions"]:
            per_step += [st[key] for st in c["observe_steps"]
                         if st.get("grounded") and st.get(key) is not None]
            r = completion_reward(c, key)
            if r is not None:
                vals.append(r)
        per_completion += vals
        # The quantity that matters: variation WITHIN one prompt's generation group,
        # which is what survives the advantage's group-mean subtraction.
        if len(vals) >= 2:
            group_sds.append(float(np.std(vals, ddof=1)))
    if not per_completion:
        return None
    out = {
        "name": name,
        "n_steps": len(per_step),
        "n_completions": len(per_completion),
        "n_groups": len(group_sds),
        "step_mean": float(np.mean(per_step)) if per_step else float("nan"),
        "reward_mean": float(np.mean(per_completion)),
        "reward_sd_all": float(np.std(per_completion, ddof=1)) if len(per_completion) > 1 else float("nan"),
        "sd_per_sample": float(np.mean(group_sds)) if group_sds else float("nan"),
    }
    chance = CHANCE.get(name)
    if chance is not None and per_step:
        out["frac_steps_above_chance"] = float(np.mean(np.asarray(per_step) > chance))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("probe_json", help="probe_merged.json from a probe run (or its directory)")
    p.add_argument("--reference", default="mean_in",
                   help="metric whose training pressure is being matched (default mean_in)")
    p.add_argument("--reference-weights", default="0.2,0.4",
                   help="w_overlap values already trained with the reference metric")
    args = p.parse_args()

    path = Path(args.probe_json)
    if path.is_dir():
        path = path / "probe_merged.json"
    payload = json.loads(path.read_text())
    cfg = payload.get("config", {})
    print(f"probe: {path}")
    print(f"  dataset={cfg.get('dataset')} n_samples={cfg.get('n_samples')} "
          f"gens={cfg.get('num_generations')} temp={cfg.get('temperature')}")
    print(f"  layer={cfg.get('overlap_layer')} heads={cfg.get('overlap_heads')} "
          f"token_reduction={cfg.get('token_reduction')} "
          f"box_threshold={cfg.get('box_threshold')} max_box_area={cfg.get('max_box_area')}")

    w_refs = [float(w) for w in args.reference_weights.split(",") if w.strip()]

    for model, m in payload["models"].items():
        rows = [r for r in (summarise(m["samples"], k, n) for k, n in METRICS) if r]
        if not rows:
            print(f"\n=== {model}: no scored steps ===")
            continue
        print(f"\n=== {model} ===")
        print(f"{'metric':<12} {'steps':>7} {'compl':>6} {'step mean':>10} {'reward mean':>12} "
              f"{'sd/sample':>10} {'sd all':>8} {'>chance':>8}")
        for r in rows:
            ac = r.get("frac_steps_above_chance")
            print(f"{r['name']:<12} {r['n_steps']:>7} {r['n_completions']:>6} "
                  f"{r['step_mean']:>10.4f} {r['reward_mean']:>12.4f} "
                  f"{r['sd_per_sample']:>10.4f} {r['reward_sd_all']:>8.4f} "
                  f"{'--' if ac is None else f'{ac:>8.3f}'}")

        ref = next((r for r in rows if r["name"] == args.reference), None)
        if not ref or not np.isfinite(ref["sd_per_sample"]) or ref["sd_per_sample"] <= 0:
            continue
        print(f"\n  w_overlap that matches {args.reference}'s pressure "
              f"(w_ref x sd_{args.reference}/sd_metric):")
        head = "  " + f"{'metric':<12}" + "".join(f"{f'w_ref={w}':>14}" for w in w_refs) + f"{'sd ratio':>11}"
        print(head)
        for r in rows:
            if not np.isfinite(r["sd_per_sample"]) or r["sd_per_sample"] <= 0:
                continue
            ratio = ref["sd_per_sample"] / r["sd_per_sample"]
            cells = "".join(f"{w * ratio:>14.3f}" for w in w_refs)
            print(f"  {r['name']:<12}{cells}{ratio:>11.3f}")


if __name__ == "__main__":
    main()
