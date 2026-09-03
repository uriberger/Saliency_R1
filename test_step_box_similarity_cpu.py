#!/usr/bin/env python
"""CPU checks for step_box_similarity.py and dino_text_sensitivity.py.

Two things are checked, because two different things can go wrong:

  1. The MEASURES. `closeness` is a rescaling of IoU by two reference points, and if
     either reference is wrong every number in the report is wrong in the same direction
     and nothing looks odd. So they are pinned against cases whose answer is known by
     hand: identical masks, disjoint masks, nested masks, and two masks drawn at random
     (whose closeness must land near 0 by construction, since that is what `chance` is
     defined to be).

  2. The PLUMBING of the GPU script, which is the part that cannot be smoke-tested
     anywhere Claude can run -- Grounding-DINO on a login-node CPU takes minutes per
     image. `_dino_boxes` is stubbed with a fake grounder, so every other line of that
     script -- variant construction, image loading, re-rasterisation, scoring, the
     report -- runs here for real. What is left untested on CPU is one function call.

    python test_step_box_similarity_cpu.py
"""

from __future__ import annotations

import base64
import importlib.util
import json
import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

SBS = importlib.util.spec_from_file_location("_sbs", REPO / "step_box_similarity.py")
_m = importlib.util.module_from_spec(SBS)
sys.modules["_sbs"] = _m
SBS.loader.exec_module(_m)
SBS = _m

OK = []


def check(name, cond, extra=""):
    OK.append(bool(cond))
    print(f"{'ok  ' if cond else 'FAIL'} {name}{(' -- ' + extra) if extra else ''}")


# ---------------------------------------------------------------------------
# 1. the measures
# ---------------------------------------------------------------------------
g = (10, 16)
N = g[0] * g[1]

a = np.zeros(g, dtype=bool); a[:5, :] = True          # 80 patches, top half
b = np.zeros(g, dtype=bool); b[5:, :] = True          # 80 patches, bottom half
c = np.zeros(g, dtype=bool); c[:2, :] = True          # 32 patches, inside a

check("identical masks: IoU 1", SBS.iou(a, a) == 1.0)
check("disjoint halves: IoU 0", SBS.iou(a, b) == 0.0)
check("nested: IoU = small/large", abs(SBS.iou(a, c) - 32 / 80) < 1e-12,
      f"{SBS.iou(a, c):.4f}")

# chance for two 80-patch masks on 160 patches: intersection 40, union 120 -> 1/3.
check("chance IoU, two half-grids", abs(SBS.iou_chance(80, 80, N) - 1 / 3) < 1e-12,
      f"{SBS.iou_chance(80, 80, N):.4f}")
check("best IoU is min/max", SBS.iou_best(80, 32) == 32 / 80)
check("best IoU of equal sizes is 1", SBS.iou_best(80, 80) == 1.0)

_o, _c, _m, z = SBS.closeness(a, a)
check("closeness of a mask with itself is 1", abs(z - 1.0) < 1e-12, f"{z:.4f}")
_o, _c, _m, z = SBS.closeness(a, c)
check("closeness of a nested pair is 1", abs(z - 1.0) < 1e-12, f"{z:.4f}")
_o, _c, _m, z = SBS.closeness(a, b)
check("closeness of disjoint halves is negative", z < -0.4, f"{z:.4f}")

# Random masks of the given sizes must average ~0 closeness: that IS the definition of
# the chance reference, so a bias here means the reference is mis-derived.
rng = np.random.default_rng(0)
zs = []
for _ in range(400):
    na, nb = int(rng.integers(20, 120)), int(rng.integers(20, 120))
    x = np.zeros(N, dtype=bool); x[rng.choice(N, na, replace=False)] = True
    y = np.zeros(N, dtype=bool); y[rng.choice(N, nb, replace=False)] = True
    zs.append(SBS.closeness(x.reshape(g), y.reshape(g))[3])
check("random masks average ~0 closeness", abs(np.mean(zs)) < 0.03, f"{np.mean(zs):+.4f}")

# ---------------------------------------------------------------------------
# 2. rasterise matches the reward's own union, on the reward's own boxes
# ---------------------------------------------------------------------------
for _n in ("trl", "trl.rewards"):
    if _n not in sys.modules:
        _s = types.ModuleType(_n)
        _s.__path__ = [str(REPO / _n.replace(".", "/"))]
        sys.modules[_n] = _s
_spec = importlib.util.spec_from_file_location("trl.rewards.overlap_rewards",
                                               REPO / "trl/rewards/overlap_rewards.py")
OREW = importlib.util.module_from_spec(_spec)
sys.modules["trl.rewards.overlap_rewards"] = OREW
_spec.loader.exec_module(OREW)

boxes = [[0.1, 0.2, 0.4, 0.6], [0.55, 0.05, 0.9, 0.35], [0.0, 0.8, 0.2, 1.0]]
mine = SBS.rasterise(boxes, *g)
theirs = OREW._union_mask(boxes, *g, apply_union_cap=False)
check("rasterise == overlap_rewards._union_mask", np.array_equal(mine, theirs))

# ---------------------------------------------------------------------------
# 3. the metric copies match the reward's originals
# ---------------------------------------------------------------------------
rng = np.random.default_rng(7)
for _ in range(50):
    smap = rng.random(g)
    mask = rng.random(g) < 0.4
    if not mask.any() or mask.all():
        continue
    for name, mine_fn, theirs_fn in (("mean_in", SBS.m_mean_in, OREW._mean_in),
                                     ("mean_in_v2", SBS.m_mean_in_v2, OREW._mean_in_v2),
                                     ("auroc", SBS.m_auroc, OREW._auroc)):
        if abs(mine_fn(smap, mask) - theirs_fn(smap, mask)) > 1e-12:
            check(f"metric copy {name}", False)
            break
else:
    check("metric copies match overlap_rewards (mean_in / mean_in_v2 / auroc)", True)

# ---------------------------------------------------------------------------
# 4. decoding round-trips what overlap_probe wrote
# ---------------------------------------------------------------------------
m = rng.random(g) < 0.5
b64 = base64.b64encode(m.astype(np.uint8).tobytes()).decode()
check("decode_mask round-trips", np.array_equal(SBS.decode_mask(b64, *g), m))

smap = rng.random(g)
q = np.clip(np.rint(255.0 * (smap / smap.max())), 0, 255).astype(np.uint8)
back = SBS.decode_map(base64.b64encode(q.tobytes()).decode(), *g)
check("decode_map recovers mean_in to 1/255",
      abs(SBS.m_mean_in(back, m) - SBS.m_mean_in(smap, m)) < 1 / 255,
      f"{abs(SBS.m_mean_in(back, m) - SBS.m_mean_in(smap, m)):.5f}")

# ---------------------------------------------------------------------------
# 5. content words and text overlap
# ---------------------------------------------------------------------------
check("stop words dropped", SBS.content_words("The dog is on the mat.") == {"dog", "mat"},
      str(SBS.content_words("The dog is on the mat.")))
check("identical sentences overlap 1", SBS.text_overlap("a red car", "a red car") == 1.0)
check("no shared content word overlaps 0",
      SBS.text_overlap("the red car", "a blue bus") == 0.0)

# ---------------------------------------------------------------------------
# 6. the GPU script's plumbing, with DINO stubbed out
# ---------------------------------------------------------------------------
from PIL import Image  # noqa: E402

tmp = tempfile.mkdtemp(prefix="sbs_test_")
imgdir = os.path.join(tmp, "images")
os.makedirs(imgdir)

gh, gw = g
rows = []
for i in range(3):
    Image.new("RGB", (64, 40), (i * 40, 60, 90)).save(os.path.join(imgdir, f"im{i}.png"))
    comps = []
    for cidx in range(2):
        steps = []
        for si in range(3):
            bx = [[0.1 * si, 0.1, 0.4 + 0.1 * si, 0.6]]
            mask = SBS.rasterise(bx, gh, gw)
            smap = rng.random((gh, gw))
            qq = np.clip(np.rint(255.0 * (smap / smap.max())), 0, 255).astype(np.uint8)
            steps.append({
                "step_index": si,
                "text": f"image {i} chain {cidx} step {si} shows a cat and a table",
                "grid": [gh, gw],
                "mask_q": base64.b64encode(mask.astype(np.uint8).tobytes()).decode(),
                "map_q": base64.b64encode(qq.tobytes()).decode(),
                "boxes_kept": bx,
                "grounded": True,
                "mean_in_raw": SBS.m_mean_in(smap, mask),
                "mean_in_v2_raw": SBS.m_mean_in_v2(smap, mask),
                "auroc_raw": SBS.m_auroc(smap, mask),
            })
        comps.append({"index": cidx, "observe_steps": steps})
    rows.append({"sample_index": i, "question": f"what is in image {i}?",
                 "image_file": f"images/im{i}.png", "completions": comps})

merged = os.path.join(tmp, "probe_merged.json")
with open(merged, "w") as f:
    json.dump({"config": {"store_maps": True, "box_threshold": 0.1, "max_box_area": 0.5,
                          "max_union_area": None, "dataset": "fake", "n_samples": 3},
               "models": {"base_coldstart": {"path": "-", "adapter": None, "samples": rows}}},
              f)

# the offline script, end to end on the fake file
res, _ = SBS.analyse_model("fake", SBS.load_model(
    json.load(open(merged))["models"]["base_coldstart"]),
    np.random.default_rng(1), "mean_in", verify=True)
check("offline: reads the fake probe file", res["n_steps"] == 18, str(res["n_steps"]))
check("offline: --verify sees no metric drift",
      max(res["verify_max_abs_err"].values()) < 1 / 255,
      str(res["verify_max_abs_err"]))
check("offline: report renders", "within" in SBS.render([res], "mean_in"))

# the GPU script, with the grounder replaced
sys.argv = ["dino_text_sensitivity.py", merged, "--models", "base_coldstart",
            "--out-dir", os.path.join(tmp, "out"), "--limit-steps", "6"]
_spec = importlib.util.spec_from_file_location("_dts", REPO / "dino_text_sensitivity.py")
DTS = importlib.util.module_from_spec(_spec)
sys.modules["_dts"] = DTS
_spec.loader.exec_module(DTS)

seen = {"texts": []}

# What the fake probe stored, keyed by the sentence that produced it. The fake grounder
# hands these back verbatim for the `real` variant, so the report's own control -- IoU
# 1.000 against the stored mask, identical 100% -- is a genuine end-to-end check that the
# script pairs each step with its own image and its own stored mask. Get the pairing wrong
# anywhere and that control breaks.
STORED = {st["text"]: st["boxes_kept"]
          for r in rows for c in r["completions"] for st in c["observe_steps"]}


def fake_boxes(images, texts):
    """Stands in for Grounding-DINO. Depends on the text, so ignoring the text fails."""
    seen["texts"].append(list(texts))
    out = []
    for t in texts:
        if t in STORED:
            out.append([list(b) for b in STORED[t]])
            continue
        k = len(SBS.content_words(t))
        out.append([] if k == 0 else [[min(0.5, 0.02 * k), 0.1, min(0.5, 0.02 * k) + 0.4, 0.6]])
    return out


DTS.OREW._dino_boxes = fake_boxes
DTS.main()

check("gpu script: grounded every variant", len(seen["texts"]) == len(DTS.VARIANTS),
      f"{len(seen['texts'])} calls")
out_rows = json.load(open(os.path.join(tmp, "out", "rows.json")))
check("gpu script: wrote a row per step", len(out_rows) == 6, str(len(out_rows)))
check("gpu script: the `real` control reproduces the stored mask exactly",
      all(r["real"]["grounded"] and r["real"]["iou"] == 1.0 and r["real"]["identical"]
          for r in out_rows))
check("gpu script: `empty` grounds nothing",
      all(not r["empty"]["grounded"] for r in out_rows))
check("gpu script: the fake grounder's text dependence shows up",
      any(r["generic"]["iou"] != r["real"]["iou"] for r in out_rows
          if r["generic"].get("grounded") and r["real"].get("grounded")))
check("gpu script: report written",
      os.path.exists(os.path.join(tmp, "out", "report.txt")))

print()
print(f"{sum(OK)}/{len(OK)} checks passed")
sys.exit(0 if all(OK) else 1)
