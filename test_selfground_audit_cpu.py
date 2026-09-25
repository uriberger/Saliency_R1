#!/usr/bin/env python
"""CPU checks for selfground_audit.py.

    python test_selfground_audit_cpu.py

No GPU, no detector, no model. The audit's job is to say whether the policy changed WHAT
IT SAYS to make the saliency reward easier, so the things that have to be pinned are the
ones that would make an arm look different for a reason that is not the arm:

  * `phi` must be the shipped reward's `_mean_in` on the same map and mask, and the stored
    bytes must decode back to the map the probe quantised. If either drifts, every table
    is about a statistic the training run never used.
  * the bootstrap must cluster on the PROMPT. An 8-rollout group resampled as 8
    independent draws narrows every interval by about sqrt(8), which is the difference
    between "starred" and "not".
  * a delta must be paired on the prompt, so arms are compared on the pictures they both
    saw.
  * the flatteners must count what they claim: a duplicate observe step, a sentence label,
    a box that the area cap dropped.
  * the size-matched table must find nothing when an arm differs only in union area, and
    find the shift when the arm really does score higher inside a bin.
"""
import base64
import importlib.util
import json
import os
import sys
import tempfile
import types

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

SG = importlib.import_module("selfground_audit")

# the shipped reward, imported without the `trl` package (it pulls torch + the trainer)
_pkg = types.ModuleType("trl_sg"); _pkg.__path__ = [os.path.join(ROOT, "trl")]
sys.modules["trl_sg"] = _pkg
_sub = types.ModuleType("trl_sg.rewards"); _sub.__path__ = [os.path.join(ROOT, "trl", "rewards")]
sys.modules["trl_sg.rewards"] = _sub


def _imp(name):
    spec = importlib.util.spec_from_file_location(
        f"trl_sg.rewards.{name}", os.path.join(ROOT, "trl", "rewards", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"trl_sg.rewards.{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


_imp("roll_null")
ORW = _imp("overlap_rewards")

FAILED = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}  {detail}")
        FAILED.append(name)


def quantize(smap):
    """overlap_probe.quantize_map, copied so the test does not need torch to import it."""
    mx = float(smap.max())
    q = (np.zeros(smap.shape, np.uint8) if mx <= 0
         else np.clip(np.rint(255.0 * (smap / mx)), 0, 255).astype(np.uint8))
    return base64.b64encode(np.ascontiguousarray(q).tobytes()).decode("ascii")


def b64mask(mask):
    return base64.b64encode(
        np.ascontiguousarray(mask.astype(np.uint8)).tobytes()).decode("ascii")


rng = np.random.default_rng(0)

print("\n1. phi is the shipped reward's mean_in")
bad = 0
for _ in range(200):
    gh, gw = int(rng.integers(3, 12)), int(rng.integers(3, 18))
    m = rng.random((gh, gw)).astype(np.float32)
    mask = rng.random((gh, gw)) < 0.4
    if not mask.any():
        continue
    mine = SG.mean_in(m.astype(np.float64), mask)
    theirs = ORW._mean_in(m, mask)
    if not np.isclose(mine, theirs, rtol=1e-6, atol=1e-9):
        bad += 1
check("200 random (map, mask) pairs agree with trl/rewards/overlap_rewards._mean_in",
      bad == 0, f"{bad} disagreed")
check("an empty mask is nan, not 0",
      not np.isfinite(SG.mean_in(np.ones((4, 4)), np.zeros((4, 4), bool))))

print("\n1b. the fixed-chain statistics")
# The cross pass asks whether the WEIGHTS moved the attention, so its statistics have to
# separate two things the same map can do: send more mass to the image (vis_mass), and
# move mass around inside it (share_in / enr). If `share_in` were not normalised by the
# map's own sum, doubling the attention on every patch would read as a gain.
m = rng.random((10, 16)) * 0.002
mask = rng.random((10, 16)) < 0.35
s = SG.map_stats(m, mask)
check("phi is mean_in", np.isclose(s["phi"], SG.mean_in(m, mask)))
check("vis_mass is the map's sum -- the share of the row that reached the image",
      np.isclose(s["vis_mass"], m.sum()))
check("share_in is the in-region share of the VISUAL mass",
      np.isclose(s["share_in"], m[mask].sum() / m.sum()))
check("enrichment is that share over the union's area share, so chance is 1.0",
      np.isclose(s["enr"], (m[mask].sum() / m.sum()) / mask.mean()))
check("enrichment is also phi / flatness, the two routes agree",
      np.isclose(s["enr"], s["phi"] / s["flat"]))
s2 = SG.map_stats(m * 2.0, mask)
check("doubling every patch moves vis_mass and nothing else",
      np.isclose(s2["vis_mass"], 2 * s["vis_mass"])
      and np.isclose(s2["share_in"], s["share_in"])
      and np.isclose(s2["enr"], s["enr"]) and np.isclose(s2["phi"], s["phi"]),
      f"{s2['share_in']:.6f} vs {s['share_in']:.6f}")
flat_map = np.full((8, 8), 0.001)
check("a uniform map is enrichment 1.0 and phi 1.0 against any union",
      np.isclose(SG.map_stats(flat_map, mask[:8, :8])["enr"], 1.0)
      and np.isclose(SG.map_stats(flat_map, mask[:8, :8])["phi"], 1.0))
check("an empty mask leaves the mask-free columns alive",
      not np.isfinite(SG.map_stats(m, np.zeros_like(mask))["share_in"])
      and np.isclose(SG.map_stats(m, np.zeros_like(mask))["vis_mass"], m.sum()))

print("\n1c. the two matched nulls")
# A map that peaks exactly on the union must beat both nulls; a map that is flat, or that
# peaks somewhere the union does not cover, must not.
grid = np.zeros((8, 10))
union = np.zeros((8, 10), bool)
union[2:5, 3:6] = True
sib = np.zeros((8, 10), bool)
sib[5:8, 0:3] = True
grid[union] = 1.0
grid += 0.01
n = SG.null_stats(grid, union, [sib], seed=7)
s = SG.map_stats(grid, union)
check("a map that peaks on the union beats its own translates",
      s["enr"] - n["enr_roll"] > 1.0, f"{s['enr']:.3f} vs roll {n['enr_roll']:.3f}")
check("...and beats the sibling step's union",
      s["enr"] - n["enr_sib"] > 1.0, f"sib {n['enr_sib']:.3f}")
check("the roll null used offsets and the sibling was counted",
      n["n_roll"] > 0 and n["n_sib"] == 1, str((n["n_roll"], n["n_sib"])))
flat = np.full((8, 10), 0.01)
nf = SG.null_stats(flat, union, [sib], seed=7)
sf = SG.map_stats(flat, union)
check("a flat map scores 1.0 against every union, so both gaps are 0",
      np.isclose(sf["enr"], 1.0) and np.isclose(nf["enr_roll"], 1.0)
      and np.isclose(nf["enr_sib"], 1.0))
elsewhere = np.full((8, 10), 0.01)
elsewhere[sib] = 1.0
ne = SG.null_stats(elsewhere, union, [sib], seed=7)
se = SG.map_stats(elsewhere, union)
check("a map that peaks on the SIBLING's region scores below it",
      se["enr"] - ne["enr_sib"] < -1.0, f"{se['enr']:.3f} vs {ne['enr_sib']:.3f}")
check("the same seed draws the same translates, so two models stay paired",
      SG.null_stats(grid, union, [sib], seed=7)["enr_roll"] == n["enr_roll"]
      and SG.null_stats(grid, union, [sib], seed=8)["enr_roll"] != n["enr_roll"])

# and the table the whole stage exists for: rows in, markdown out, deltas down a block
def _cp_row(t, mp, q, phi, share, vis):
    return dict(text_arm=t, map_arm=mp, qid=q, phi=phi, flat=0.05, union_frac=0.5,
                enr=share / 0.5, share_in=share, vis_mass=vis, n_tokens=20)


cp_rows = []
for q in range(20):
    for j in range(8):
        cp_rows.append(_cp_row("cold", "cold", f"q{q}", 0.04, 0.40, 0.30))
        cp_rows.append(_cp_row("cold", "ours", f"q{q}", 0.04, 0.40, 0.45))
cp = {"rows": cp_rows, "layer": 22, "heads": "28,31", "base": "cold",
      "comp_rows": [dict(text_arm="cold", map_arm=m, qid=f"q{q}", vis_chain=v,
                         vis_chain_all=float("nan"))
                    for q in range(20) for m, v in (("cold", 0.30), ("ours", 0.45))]}
_by = {}
for r in cp_rows:
    _by.setdefault((r["text_arm"], r["map_arm"]), []).append(r)
md = "\n".join(SG._fixed_chain_tables(cp, _by, ["cold"], ["cold", "ours"]))
check("the fixed-chain table finds the visual-attention move", "+0.1500" in md, md[-600:])
check("and reports no in-region redistribution", "+0.0000 [+0.0000, +0.0000]" in md)
check("the chain-level block is there", "reaches the image at all" in md)

print("\n2. the stored bytes decode back")
m = rng.random((10, 16)) * 0.003
q = quantize(m)
back = SG.decode_map(q, 10, 16, float(m.max()))
check("map round-trips to within one quantisation step",
      float(np.abs(back - m).max()) <= m.max() / 255.0 + 1e-12,
      f"max err {float(np.abs(back - m).max()):.2e}")
mask = rng.random((10, 16)) < 0.3
check("mask round-trips exactly", bool((SG.decode_mask(b64mask(mask), 10, 16) == mask).all()))
r = SG.ring_mask(4, 5)
check("the ring is the one-patch border", int(r.sum()) == 2 * 4 + 2 * 5 - 4)

print("\n3. box geometry")
areas, dists = SG.box_stats([[0.0, 0.0, 0.5, 0.5], [0.25, 0.25, 0.75, 0.75]])
check("area is width x height", np.allclose(areas, [0.25, 0.25]), str(areas))
check("a centred box has distance 0", np.isclose(dists[1], 0.0), str(dists))
check("a corner-ish box is further out", dists[0] > dists[1])
check("a full-image box hits area 1", np.isclose(SG.box_stats([[0, 0, 1, 1]])[0][0], 1.0))

print("\n4. the bootstrap clusters on the prompt")
# 20 prompts x 8 rollouts. Every rollout of a prompt carries the SAME value, so the
# information in the sample is 20 numbers, not 160: an unclustered interval would be
# about sqrt(8) too narrow.
rows = [{"qid": f"q{p}", "v": float(p)} for p in range(20) for _ in range(8)]
mean, lo, hi, n = SG.boot_mean(rows, "v")
check("the mean is over every row", np.isclose(mean, np.mean(range(20))) and n == 160)
width = hi - lo
flat = [{"qid": f"q{i}", "v": float(p)} for i, (p, _) in
        enumerate((p, r) for p in range(20) for r in range(8))]
_, flo, fhi, _ = SG.boot_mean(flat, "v")
check("clustering widens the interval against treating rollouts as independent",
      width > 1.6 * (fhi - flo), f"{width:.3f} vs {fhi - flo:.3f}")

print("\n5. a delta is paired on the prompt")
a = [{"qid": f"q{p}", "v": p + 1.0} for p in range(30)]
b = [{"qid": f"q{p}", "v": float(p)} for p in range(30)]
d, dlo, dhi, k = SG.boot_diff(a, b, "v")
check("a constant shift is recovered exactly", np.isclose(d, 1.0) and k == 30)
check("...and its paired interval is degenerate, not wide",
      (dhi - dlo) < 1e-9, f"[{dlo}, {dhi}]")
half = [{"qid": f"q{p}", "v": float(p)} for p in range(15)]
_, _, _, k2 = SG.boot_diff(half, b, "v")
check("only the prompts both arms saw are compared", k2 == 15)

print("\n6. flatten counts what it claims")
gh, gw = 4, 5
smap = rng.random((gh, gw)).astype(np.float32)
mask = np.zeros((gh, gw), bool); mask[1:3, 1:3] = True


def step(text, boxes_raw, boxes_kept, grounded=True):
    return {
        "step_index": 0, "text": text, "tok_a": 0, "tok_b": 4, "n_tokens": 4,
        "n_boxes_raw": len(boxes_raw), "n_boxes_kept": len(boxes_kept),
        "max_box_area": max((SG.box_stats([b])[0][0] for b in boxes_raw), default=None),
        "union_frac_uncapped": float(mask.mean()), "dropped_by_union_cap": False,
        "image_mass": float(smap.sum()), "map_max": float(smap.max()),
        "map_mean": float(smap.mean()), "grid": [gh, gw], "map_q": quantize(smap),
        "boxes_raw": boxes_raw, "boxes_kept": boxes_kept, "mask_q": b64mask(mask),
        "grounded": grounded, "box_area_frac": float(mask.mean()),
        "mean_in_raw": SG.mean_in(smap.astype(np.float64), mask) if grounded else None,
        "auroc_raw": 0.5, "logratio_raw": -0.1, "ecc": 0.2, "note": "",
    }


def completion(idx, steps, sentences):
    return {"index": idx, "text": "<think> x </think> y",
            "n_completion_tokens": 10, "truncated_at_max_tokens": False,
            "format_valid": True,
            "rewards": {"think_overlap_reward": 0.05, "accuracy_reward": 1.0,
                        "openai_reward": None},
            "n_observe_steps_scored": sum(1 for s in steps if s["grounded"]),
            "n_observe_steps_total": len(steps),
            "observe_steps": steps, "all_sentences": sentences}


SENTS = [{"text": "a red cup on the table.", "label": "observe"},
         {"text": "let me check the left.", "label": "plan"},
         {"text": "so it is coffee.", "label": "deduce"},
         {"text": "the user asked a question.", "label": "none"}]
probe = {"config": {"base_model": "/b", "dataset": "d", "split": "holdout"},
         "models": {"armA": {"path": "/b", "adapter": None, "samples": [
             {"sample_index": 0, "row_index": 0, "dataset": "ds", "question_id": "q1",
              "question": "what?", "gt_answer": "a cup", "image_file": "images/a.png",
              "completions": [
                  completion(0, [step("a red cup on the table.",
                                      [[0.1, 0.1, 0.4, 0.4]], [[0.1, 0.1, 0.4, 0.4]]),
                                 step("a red cup on the table.",
                                      [[0.1, 0.1, 0.4, 0.4]], [[0.1, 0.1, 0.4, 0.4]])],
                             SENTS),
                  completion(1, [step("a blue plate.", [[0.0, 0.0, 0.9, 0.9]], [],
                                      grounded=False)], SENTS)]}]}}}
with tempfile.TemporaryDirectory() as td:
    p = os.path.join(td, "probe_merged.json")
    json.dump(probe, open(p, "w"))
    arms = SG.load_arms([p])
    steps_rows, comp_rows = SG.flatten(arms)
check("one row per observe step", len(steps_rows) == 3)
check("one row per completion", len(comp_rows) == 2)
check("the ungrounded step is kept but not marked grounded",
      sum(1 for r in steps_rows if r["grounded"]) == 2)
check("a repeated sentence is a duplicate step",
      np.isclose([c for c in comp_rows if c["comp"] == 0][0]["dup_step_frac"], 0.5))
check("...and a single step is not",
      np.isclose([c for c in comp_rows if c["comp"] == 1][0]["dup_step_frac"], 0.0))
c0 = [c for c in comp_rows if c["comp"] == 0][0]
check("the sentence labels are counted", c0["n_plan"] == 1 and c0["n_deduce"] == 1
      and c0["n_none"] == 1 and np.isclose(c0["frac_observe"], 0.25))
check("a completion's distinct content terms are counted off its observe steps",
      c0["n_terms"] == 3 and
      [c for c in comp_rows if c["comp"] == 1][0]["n_terms"] == 2,
      str([c["n_terms"] for c in comp_rows]))
check("...and the per-100-token rate divides by the completion, not by the step",
      np.isclose(c0["terms_per_100tok"], 30.0) and
      np.isclose(c0["observe_per_100tok"], 20.0),
      f"{c0['terms_per_100tok']}, {c0['observe_per_100tok']}")
r0 = steps_rows[0]
check("the union's ring share is read off the stored mask",
      np.isclose(r0["ring_share"], 0.0), str(r0["ring_share"]))
check("phi comes through unchanged",
      np.isclose(r0["phi"], SG.mean_in(smap.astype(np.float64), mask)))
check("a dropped box shows as kept < raw",
      [r for r in steps_rows if not r["grounded"]][0]["n_boxes_kept"] == 0)

print("\n7. content terms and frames")
t = SG.content_terms("The image shows a red cup on the table.")
check("stopwords and the frame words are dropped", "the" not in t and "image" not in t)
check("the objects survive", {"red", "cup", "table"} <= set(t), str(t))

print("\n8. log-odds picks the shifted term, not the commonest one")
from collections import Counter
A = Counter({"cup": 60, "table": 50, "background": 40})
B = Counter({"cup": 60, "table": 50, "background": 5})
z = SG.logodds_delta(A, B)
check("the term that moved ranks first", max(z, key=z.get) == "background", str(z))
# The measure is compositional: one term rising pushes every other term's share down, so
# the unmoved terms go mildly negative rather than to zero. What has to hold is the
# ordering and the sign, not a null on them.
check("the unmoved terms sit below it and on the other side of zero",
      z["cup"] < 0 and z["table"] < 0 and z["background"] > max(abs(z["cup"]),
                                                               abs(z["table"])), str(z))

print("\n9. the size-matched table")
# arm `big` differs from `base` only by union area; phi is a pure function of the area,
# so inside a bin there is nothing left to find.
def synth(arm, n, lo, hi, gain=0.0, seed=1):
    g = np.random.default_rng(seed)
    out = []
    for i in range(n):
        u = float(g.uniform(lo, hi))
        out.append({"arm": arm, "qid": f"q{i % 25}", "grounded": True,
                    "union_frac": u, "phi": 0.05 + 0.1 * u + gain})
    return out


rows = synth("base", 400, 0.2, 0.8, seed=2) + synth("big", 400, 0.5, 0.9, seed=3)
raw = SG.boot_diff([r for r in rows if r["arm"] == "big"],
                   [r for r in rows if r["arm"] == "base"], "phi")[0]
tab = SG.union_bin_table(rows, "base", ["base", "big"])
overlapping = [r for r in tab if r.get("big") and r["big"][3] > 5]
# Binning cannot remove the confound entirely -- inside a bin the two arms still sit at
# different points of it -- so the claim is that it shrinks an area-only gap by a lot,
# not that it nulls it. With five bins that is a factor of four or better.
check("an area-only difference mostly goes away inside a bin",
      raw > 0.015 and all(abs(r["big"][0]) < raw / 4 for r in overlapping),
      f"raw {raw:.4f} vs {[round(r['big'][0], 4) for r in overlapping]}")
rows2 = synth("base", 400, 0.2, 0.8, seed=2) + synth("up", 400, 0.2, 0.8, gain=0.02, seed=3)
tab2 = SG.union_bin_table(rows2, "base", ["base", "up"])
check("a real within-bin shift is found in every bin",
      all(np.isclose(r["up"][0], 0.02, atol=0.004) for r in tab2 if r.get("up")),
      str([round(r["up"][0], 4) for r in tab2]))

print("\n10. per-image maps average, and only within a grid")
arms = {"a": {"path": "/x/probe_merged.json", "samples": [
    {"sample_index": 0, "question_id": "q1", "image_file": "images/a.png",
     "completions": [{"index": 0, "observe_steps": [
         {"grid": [2, 2], "map_q": quantize(np.array([[1.0, 0.0], [0.0, 0.0]])),
          "map_max": 1.0},
         {"grid": [2, 2], "map_q": quantize(np.array([[0.0, 0.0], [0.0, 1.0]])),
          "map_max": 1.0},
     ]}]}]}}
pm = SG.per_image_maps(arms)
m, grid = pm["a"]["images/a.png"]
check("two step maps average", np.allclose(m, [[0.5, 0.0], [0.0, 0.5]]), str(m))
check("the grid comes with it", grid == (2, 2))

print("\n" + ("ALL PASS" if not FAILED else f"{len(FAILED)} FAILED: {FAILED}"))
sys.exit(1 if FAILED else 0)
