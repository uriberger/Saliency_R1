#!/usr/bin/env python
"""CPU checks for intervene_probe: the intervention algebra, sharding, resume, report.

The GPU selftest stage covers the hook's output rebuild. This covers everything that
can be checked without a model, which is the part most likely to be silently wrong:
the mixture must preserve image mass exactly, must never touch a non-image column,
and every target must be a distribution.
"""
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

WT = str(Path(__file__).resolve().parent)
sys.path.insert(0, WT)
spec = importlib.util.spec_from_file_location("iv", f"{WT}/intervene_probe.py")
IV = importlib.util.module_from_spec(spec)
spec.loader.exec_module(IV)

fails = []


def check(name, cond, extra=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name} {extra}")
    if not cond:
        fails.append(name)


print("\n1. targets are distributions over the image patches")
gh, gw = 10, 16
n = gh * gw
rng = np.random.default_rng(0)
mask = torch.zeros(gh, gw)
mask[3:6, 4:9] = 1.0
mask = mask.reshape(-1)
w = torch.rand(2, 7, n)                      # [H=2, R=7, n_img]
for kind in IV.CONDITIONS:
    t = IV.step_target(mask, w, kind, np.random.default_rng(1), gh, gw, "cpu")
    s = t.sum(-1)
    check(f"{kind}: sums to 1", torch.allclose(s, torch.ones_like(s), atol=1e-5),
          f"(sum={float(s.reshape(-1)[0]):.6f})")
    check(f"{kind}: non-negative", bool((t >= -1e-9).all()))
    if kind in ("box", "roll"):
        check(f"{kind}: support size == box size", int((t > 0).sum(-1).reshape(-1)[0]) == int(mask.sum()))
    if kind == "shape":
        inside = (t.reshape(-1, n)[:, mask.bool()] > 0).any()
        outside = float(t.reshape(-1, n)[:, ~mask.bool()].abs().max())
        check("shape: zero outside the box", outside == 0.0 and bool(inside))

print("\n2. the mixture preserves image mass at every alpha, for every target")
for kind in IV.CONDITIONS:
    ok_mass, ok_pos = True, True
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        t = IV.step_target(mask, w, kind, np.random.default_rng(2), gh, gw, "cpu")
        m = w.sum(-1, keepdim=True)
        new = (1 - alpha) * w + alpha * m * t
        if not torch.allclose(new.sum(-1), w.sum(-1), atol=1e-5):
            ok_mass = False
        if alpha < 1.0 and not bool((new > 0).all()):
            ok_pos = False
    check(f"{kind}: image mass preserved for all alpha", ok_mass)
    check(f"{kind}: strictly positive for alpha<1 (on-manifold)", ok_pos)

print("\n3. alpha=0 is exactly a no-op")
t = IV.step_target(mask, w, "box", np.random.default_rng(3), gh, gw, "cpu")
new = (1 - 0.0) * w + 0.0 * w.sum(-1, keepdim=True) * t
check("alpha=0 identity", torch.equal(new, w))

print("\n4. the scatter writes only image columns of only the selected heads/rows")
H, S = 8, 40
a = torch.rand(1, H, S, S)
before = a.clone()
heads = torch.tensor([2, 5])
rows = torch.arange(11, 15)
img = torch.tensor([20, 21, 22, 23, 24])
a[0, heads[:, None, None], rows[None, :, None], img[None, None, :]] = 0.0
touched = (a != before)
check("only selected heads touched",
      set(torch.nonzero(touched)[:, 1].unique().tolist()) == {2, 5})
check("only selected rows touched",
      set(torch.nonzero(touched)[:, 2].unique().tolist()) == set(rows.tolist()))
check("only image columns touched",
      set(torch.nonzero(touched)[:, 3].unique().tolist()) == set(img.tolist()))

print("\n5. layer/variant grid construction")
check("parse_layers range", IV.parse_layers("0-3", 36) == [0, 1, 2, 3])
check("parse_layers mixed+clamped", IV.parse_layers("2,4-6,99", 36) == [2, 4, 5, 6])


class A:
    head_mode, alphas, conditions = "each", "0.5,1.0", "box,roll,perm"


v = IV.build_variants(A, 4)
check("each-head grid size", len(v) == 4 * (2 + 2 + 1), f"(got {len(v)})")
check("controls are alpha=1 only",
      all(x["alpha"] == 1.0 for x in v if x["kind"] == "perm"))
A.head_mode = "layer"
check("layer mode is one head-set", len({x["hname"] for x in IV.build_variants(A, 32)}) == 1)
A.head_mode = "28,31"
check("explicit head set", IV.build_variants(A, 32)[0]["heads"] == [28, 31])
A.conditions = "box,bogus"
try:
    IV.build_variants(A, 4)
    check("unknown condition rejected", False)
except SystemExit:
    check("unknown condition rejected", True)

print("\n5b. answer readout scores gold, not the separator")

V = 100
torch.manual_seed(0)
# ids = [sep, gold0, gold1]. Non-degenerate logits, so logp is a real number and a
# mistake in the offset shows up as a wrong value rather than a wrong-but-tiny one.
logits = torch.randn(10, V)
logits[3, 7] += 4.0        # position predicting the separator
logits[4, 42] += 4.0       # position predicting gold0
logits[5, 43] += 4.0       # position predicting gold1
case = {"gold_ids": [7, 42, 43], "score_from": 1}
r = IV.answer_readout(logits, answer_pos=4, case=case)
lp = torch.log_softmax(logits[4:6], dim=-1)
want = float(lp[0, 42] + lp[1, 43])
check("scores len(gold) tokens, not the separator", r["n_gold"] == 2, f"(got {r['n_gold']})")
check("top1 taken at the first GOLD token", r["top1_id"] == 42)
check("first_correct compares against gold", r["first_correct"] == 1)
check("logp sums exactly the gold tokens", abs(r["logp_gold"] - want) < 1e-5,
      f"({r['logp_gold']:.4f} vs {want:.4f})")

# score_from=0 is the pre-fix behaviour: the comparison lands one position earlier,
# on the separator, which is what made first_correct identically 0 on real data.
r0 = IV.answer_readout(logits, answer_pos=4, case={"gold_ids": [7, 42, 43]})
lp0 = torch.log_softmax(logits[3:6], dim=-1)
want0 = float(lp0[0, 7] + lp0[1, 42] + lp0[2, 43])
check("score_from=0 scores every token including the separator",
      r0["n_gold"] == 3 and abs(r0["logp_gold"] - want0) < 1e-5)
check("score_from=0 takes top1 at the SEPARATOR position", r0["top1_id"] == 7)
check("score_from shifts the comparison onto the answer",
      r0["top1_id"] != r["top1_id"])

# a separator merged into the first answer token (" B" is one token) -> score_from 0
# gold_ids here start at position 5, so logits[4] (which favours 42) predicts them
r1 = IV.answer_readout(logits, answer_pos=5, case={"gold_ids": [42, 43], "score_from": 0})
check("merged separator: nothing skipped, top1 still on the answer",
      r1["n_gold"] == 2 and r1["top1_id"] == 42)

print("\n6. progress heartbeat + ETA")
with tempfile.TemporaryDirectory() as d:
    p = IV.Progress(Path(d) / "progress" / "run00.json", 100, "t", log_every=1000,
                    already_done=10)
    for _ in range(5):
        p.tick()
    p.close()
    hb = json.loads((Path(d) / "progress" / "run00.json").read_text())
    check("heartbeat counts resumed work", hb["completed"] == 15)
    check("eta present", hb["eta"] is not None)
    check("fmt_dt", IV.fmt_dt(3725) == "1h02m", f"(got {IV.fmt_dt(3725)})")

print("\n7. resume: a torn final line must not crash the loader")
with tempfile.TemporaryDirectory() as d:
    out = Path(d)
    (out / "results").mkdir()
    f = out / "results" / "shard00.jsonl"
    recs = []
    for row in range(60):
        for layer in (10, 22):
            b = {"row_index": row, "layer": layer, "hname": "-", "variant": "base",
                 "logp_gold": -2.0, "first_correct": 0, "union": 0.2 + 0.5 * (row % 2),
                 "n_gold": 2, "top1_id": 5}
            recs.append(b)
            # box beats roll by 0.3 at layer 22 only
            eff = 0.3 if layer == 22 else 0.0
            recs.append({**b, "variant": "box_a1", "logp_gold": -2.0 + eff,
                         "first_correct": 1 if layer == 22 else 0})
            recs.append({**b, "variant": "roll_a1", "logp_gold": -2.0})
    with f.open("w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
        fh.write('{"row_index": 99, "lay')           # killed mid-write
    done = set()
    with f.open() as fh:
        for line in fh:
            try:
                d2 = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((d2["row_index"], d2["layer"], d2["hname"], d2["variant"]))
    check("torn line skipped, rest recovered", len(done) == 60 * 2 * 3, f"(got {len(done)})")

    print("\n8. report reads it and recovers the planted effect")
    r = subprocess.run([sys.executable, f"{WT}/intervene_probe.py", "--stage", "report",
                        "--out-dir", str(out)], capture_output=True, text=True)
    print("     " + "\n     ".join(r.stdout.strip().splitlines()[:14]))
    check("report ran", r.returncode == 0, r.stderr.strip()[-200:])
    check("planted box-roll gap at L22 recovered", "+0.3000" in r.stdout)
    check("no effect at L10", r.stdout.count("+0.0000") >= 1)

print("\n" + ("ALL CHECKS PASSED" if not fails else f"FAILURES: {fails}"))
sys.exit(1 if fails else 0)
