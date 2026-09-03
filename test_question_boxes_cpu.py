#!/usr/bin/env python
"""CPU checks for --overlap_question_boxes: the offline builder and the reward that reads it.

Grounding-DINO cannot run where these tests run (minutes per image on a login-node CPU),
so `_dino_boxes` is stubbed throughout. Everything either side of that one call is real:
the builder's keying, sharding and merge; the loader's validation; the reward's lookup and
scoring path.

The checks worth naming, because each guards a failure that is SILENT:

  * the cached run and a per-step run given the SAME boxes must produce the SAME reward.
    That is what makes "one grounding per question" a change of grounding and of nothing
    else -- if the two paths score differently, any comparison between them is confounded.
  * a row missing from the cache must RAISE. Masking it instead would show up only as a
    quietly smaller reward on part of the batch.
  * a cache built at a different --box_threshold or a different image resolution must be
    refused. Neither can be detected from the boxes themselves.
  * the builder's image resize must still match the trainer's. They are two copies of the
    same six lines in two files, and a detector shown a different picture returns
    different boxes.

    python test_question_boxes_cpu.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

# Importing the builder also imports overlap_rewards (as trl.rewards.overlap_rewards) and
# registers it in sys.modules; take the reward module from there so both this test and the
# builder hold the SAME module object, and a stub set here is seen there.
_spec = importlib.util.spec_from_file_location("_pqb", REPO / "precompute_question_boxes.py")
PQB = importlib.util.module_from_spec(_spec)
sys.modules["_pqb"] = PQB
_spec.loader.exec_module(PQB)
OREW = sys.modules["trl.rewards.overlap_rewards"]

OK = []


def check(name, cond, extra=""):
    OK.append(bool(cond))
    print(f"{'ok  ' if cond else 'FAIL'} {name}{(' -- ' + extra) if extra else ''}")


def raises(fn, want_type=Exception, want_text=None):
    try:
        fn()
    except want_type as e:  # noqa: BLE001
        return want_text is None or want_text.lower() in str(e).lower()
    except Exception:  # noqa: BLE001
        return False
    return False


def reset_cfg():
    OREW.configure(box_threshold=0.10, max_box_area=0.5, max_union_area=None,
                   metric="mean_in", mass_floor_tau=None, natural_only=False)
    OREW._CFG["question_boxes"] = None      # configure() ignores None by design
    OREW._CFG["max_union_area"] = None
    OREW._QBOX.update(path=None, boxes=None, meta=None)


reset_cfg()

# ---------------------------------------------------------------------------
# 1. the key
# ---------------------------------------------------------------------------
check("key: joins the three identity columns",
      OREW.qbox_key("gqa", "validation", 82119) == "gqa|validation|82119")
check("key: an int and a str question_id agree",
      OREW.qbox_key("gqa", "validation", 82119) == OREW.qbox_key("gqa", "validation", "82119"))
check("key: a separator in a field is refused, not silently collided",
      raises(lambda: OREW.qbox_key("g|qa", "validation", 1), ValueError, "separator"))
check("key: the key columns are the ones the trainer forwards",
      OREW.QBOX_KEY_COLUMNS == ("dataset", "split", "question_id"))

# ---------------------------------------------------------------------------
# 2. a tiny corpus, and the builder over it
# ---------------------------------------------------------------------------
from PIL import Image  # noqa: E402

GRID = (10, 16)
N_ROWS = 6


class FakeDataset:
    """The three things precompute_question_boxes.build() asks of a dataset."""

    column_names = ["dataset", "split", "question_id", "problem", "image", "solution"]

    def __init__(self, n):
        self.rows = [{
            "dataset": "gqa" if i % 2 else "cub",
            "split": "validation",
            "question_id": 1000 + i,
            "problem": f"what is in region {i}?",
            "solution": "thing",
            "image": Image.new("RGB", (640, 480), color=(i * 20, 40, 60)),
        } for i in range(n)]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


# A grounder whose boxes depend only on the TEXT, so the test can tell which text was
# grounded, and deterministic, so the builder and the reward can be compared.
def fake_boxes_for(text):
    if "region 4" in text:
        return []                                    # this row grounds nothing
    h = abs(hash(text)) % 5
    x0 = 0.05 + 0.05 * h
    return [[x0, 0.10, x0 + 0.30, 0.55], [0.55, 0.40, 0.95, 0.90]]


GROUNDED = {"images": [], "texts": []}


def fake_dino(images, texts):
    GROUNDED["images"].extend(images)
    GROUNDED["texts"].extend(list(texts))
    return [fake_boxes_for(t) for t in texts]


OREW._dino_boxes = fake_dino
PQB.OREW._dino_boxes = fake_dino
PQB.load_dataset_like_the_trainer = lambda name, split: FakeDataset(N_ROWS)

tmp = tempfile.mkdtemp()


class Args:
    dataset = "fake"
    dataset_split = "train"
    text_column = "problem"
    box_threshold = 0.10
    batch_size = 4
    shard = 0
    num_shards = 1
    limit = 0


built = PQB.build(Args())
check("builder: one entry per row", len(built["boxes"]) == N_ROWS, str(len(built["boxes"])))
check("builder: grounded the QUESTION, once per row",
      sorted(GROUNDED["texts"]) == sorted(r["problem"] for r in FakeDataset(N_ROWS).rows))
check("builder: images were resized to the trainer's cap before grounding",
      all(max(im.size) == PQB.MAX_IMAGE_SIDE for im in GROUNDED["images"]),
      str({im.size for im in GROUNDED["images"]}))
check("builder: records the threshold and resolution the loader checks",
      built["config"]["box_threshold"] == 0.10
      and built["config"]["max_image_side"] == PQB.MAX_IMAGE_SIDE)
check("builder: stores RAW boxes, before the area cap",
      built["boxes"]["cub|validation|1000"] == [
          [round(v, 5) for v in b] for b in fake_boxes_for("what is in region 0?")])
check("builder: a row DINO could not ground is kept, as an empty list",
      built["boxes"]["cub|validation|1004"] == [])

# sharding must partition the corpus, not sample it
shards = []
for s in range(3):
    class A(Args):
        shard = s
        num_shards = 3
    shards.append(PQB.build(A()))
union = {}
for d in shards:
    union.update(d["boxes"])
check("builder: 3 shards partition the corpus exactly",
      len(union) == N_ROWS and sum(len(d["boxes"]) for d in shards) == N_ROWS,
      f"{[len(d['boxes']) for d in shards]}")
check("builder: sharded boxes equal unsharded ones", union == built["boxes"])

# ---------------------------------------------------------------------------
# 3. merge
# ---------------------------------------------------------------------------
shard_paths = []
for i, d in enumerate(shards):
    p = os.path.join(tmp, f"shard{i}.json")
    json.dump(d, open(p, "w"))
    shard_paths.append(p)

merged = PQB.merge(shard_paths, os.path.join(tmp, "merged.json"))
check("merge: reassembles every row", merged["boxes"] == built["boxes"])
check("merge: drops the per-shard bookkeeping", "shard" not in merged)

bad = json.loads(json.dumps(shards[0]))
bad["config"]["box_threshold"] = 0.25
p_bad = os.path.join(tmp, "bad.json")
json.dump(bad, open(p_bad, "w"))
check("merge: refuses shards built with different settings",
      raises(lambda: PQB.merge([shard_paths[0], p_bad], "x"), SystemExit, "different"))

part = json.loads(json.dumps(shards[0]))
p_part = os.path.join(tmp, "part.json")
json.dump(part, open(p_part, "w"))
check("merge: refuses a cache missing rows rather than writing a partial one",
      raises(lambda: PQB.merge([p_part], "x"), SystemExit, "missing"))

# ---------------------------------------------------------------------------
# 4. the loader's validation
# ---------------------------------------------------------------------------
CACHE = os.path.join(tmp, "qboxes.json")
json.dump(merged, open(CACHE, "w"))

OREW._QBOX.update(path=None, boxes=None, meta=None)
loaded = OREW.load_question_boxes(CACHE, box_threshold=0.10,
                                  max_image_side=PQB.MAX_IMAGE_SIDE)
check("loader: reads every row", len(loaded) == N_ROWS)
check("loader: refuses a missing file",
      raises(lambda: OREW.load_question_boxes(os.path.join(tmp, "nope.json")),
             FileNotFoundError, "precompute_question_boxes"))
OREW._QBOX.update(path=None, boxes=None, meta=None)
check("loader: refuses a cache built at another box_threshold",
      raises(lambda: OREW.load_question_boxes(CACHE, box_threshold=0.25),
             ValueError, "box_threshold"))
OREW._QBOX.update(path=None, boxes=None, meta=None)
check("loader: refuses a cache built at another image resolution",
      raises(lambda: OREW.load_question_boxes(CACHE, max_image_side=336),
             ValueError, "max_image_side"))
OREW._QBOX.update(path=None, boxes=None, meta=None)
old = json.loads(json.dumps(merged))
old["version"] = OREW.QBOX_VERSION + 1
p_old = os.path.join(tmp, "old.json")
json.dump(old, open(p_old, "w"))
check("loader: refuses a file of another version",
      raises(lambda: OREW.load_question_boxes(p_old), ValueError, "version"))

# ---------------------------------------------------------------------------
# 5. the reward
# ---------------------------------------------------------------------------
rng = np.random.default_rng(0)


def make_steps(texts):
    return [{"map": rng.random(GRID).astype(np.float32), "text": t} for t in texts]


def kwargs_for(rows):
    return {"dataset": [r["dataset"] for r in rows],
            "split": [r["split"] for r in rows],
            "question_id": [r["question_id"] for r in rows]}


rows = FakeDataset(N_ROWS).rows
# Two completions of ONE row, so every step of both must get that row's question boxes.
row = rows[0]
sal = [make_steps(["a red bird sits here", "the sky above it", "a fence"]),
       make_steps(["something else entirely", "another observation"])]
imgs = [row["image"], row["image"]]
kw = kwargs_for([row, row])

reset_cfg()
OREW._CFG["question_boxes"] = CACHE
GROUNDED["texts"].clear()


def boom(images, texts):
    raise AssertionError("Grounding-DINO was called with a question-box cache configured")


OREW._dino_boxes = boom
try:
    cached_rewards = OREW.think_overlap_reward(
        completions=[None] * 2, saliency_map=sal, valid_list=[True, True], image=imgs, **kw)
    called_dino = False
except AssertionError:
    cached_rewards, called_dino = [None, None], True
check("reward: DINO is never called when the cache is configured", not called_dino)
check("reward: one score per completion",
      len(cached_rewards) == 2 and all(r is not None for r in cached_rewards),
      str(cached_rewards))

# The equivalence that matters: per-step grounding fed the SAME boxes must agree exactly.
OREW._dino_boxes = lambda images, texts: [list(merged["boxes"][
    OREW.qbox_key(row["dataset"], row["split"], row["question_id"])]) for _ in texts]
reset_cfg()
per_step_rewards = OREW.think_overlap_reward(
    completions=[None] * 2, saliency_map=sal, valid_list=[True, True], image=imgs, **kw)
check("reward: cached == per-step given the same boxes",
      np.allclose(cached_rewards, per_step_rewards),
      f"{cached_rewards} vs {per_step_rewards}")

# Every step of a completion shares one mask, so a step's score must not depend on which
# other steps are in the chain.
reset_cfg()
OREW._CFG["question_boxes"] = CACHE
OREW._dino_boxes = boom
single = OREW.think_overlap_reward(completions=[None], saliency_map=[sal[0][:1]],
                                   valid_list=[True], image=[row["image"]],
                                   **kwargs_for([row]))
mask = OREW._union_mask(merged["boxes"]["cub|validation|1000"], *GRID)
check("reward: a step is scored against the row's question mask",
      np.isclose(single[0], OREW._step_score(sal[0][0]["map"], mask)),
      f"{single[0]}")

# A row DINO could not ground: masked (None), not scored 0.
r4 = rows[4]
none_row = OREW.think_overlap_reward(completions=[None], saliency_map=[make_steps(["x", "y"])],
                                     valid_list=[True], image=[r4["image"]],
                                     **kwargs_for([r4]))
check("reward: a row that grounded nothing is masked, not scored 0", none_row == [None],
      str(none_row))

# The format gate still applies, multiplicatively.
gated = OREW.think_overlap_reward(completions=[None], saliency_map=[sal[0]],
                                  valid_list=[False], image=[row["image"]],
                                  **kwargs_for([row]))
check("reward: the format gate still zeroes an invalid completion", gated == [0.0], str(gated))

# A row outside the cache must raise, not mask.
missing = {"dataset": ["nowhere"], "split": ["train"], "question_id": [7]}
check("reward: a row the cache does not cover raises",
      raises(lambda: OREW.think_overlap_reward(
          completions=[None], saliency_map=[sal[0]], valid_list=[True],
          image=[row["image"]], **missing), KeyError, "not in"))

# The identity columns must be present.
check("reward: a corpus without the key columns raises a legible error",
      raises(lambda: OREW.think_overlap_reward(
          completions=[None], saliency_map=[sal[0]], valid_list=[True],
          image=[row["image"]]), KeyError, "columns"))

# --overlap_natural_only still masks first, so a masked row costs no lookup.
reset_cfg()
OREW._CFG["question_boxes"] = CACHE
OREW.configure(natural_only=True)
OREW._dino_boxes = boom
nat = OREW.think_overlap_reward(completions=[None], saliency_map=[sal[0]], valid_list=[True],
                                image=[row["image"]], natural=[False], **kwargs_for([row]))
check("reward: --overlap_natural_only still masks non-natural rows", nat == [None], str(nat))

# A row this call is not scoring must not be able to fail it: the masked row below is
# absent from the cache, and the unmasked one beside it must still be scored.
mixed = OREW.think_overlap_reward(
    completions=[None, None], saliency_map=[sal[0], sal[1]], valid_list=[True, True],
    image=[row["image"], row["image"]], natural=[False, True],
    dataset=["nowhere", row["dataset"]], split=["train", row["split"]],
    question_id=[7, row["question_id"]])
check("reward: a masked row costs no cache lookup",
      mixed[0] is None and mixed[1] is not None, str(mixed))
reset_cfg()

# ---------------------------------------------------------------------------
# 6. the two copies of the image resize must not drift
# ---------------------------------------------------------------------------
trainer_src = (REPO / "trl" / "grpo_vlm_qwen3.py").read_text()
m = re.search(r"^MAX_IMAGE_SIDE\s*=\s*(\d+)\s*$", trainer_src, re.M)
check("resize: the trainer still declares MAX_IMAGE_SIDE at module level", m is not None)
if m:
    check("resize: builder and trainer agree on the cap",
          int(m.group(1)) == PQB.MAX_IMAGE_SIDE, f"{m.group(1)} vs {PQB.MAX_IMAGE_SIDE}")
check("resize: the trainer's prepare_image reads that constant, not a literal",
      re.search(r"max\(width, height\) > MAX_IMAGE_SIDE", trainer_src) is not None)

big = Image.new("RGB", (1024, 512))
small = Image.new("RGB", (300, 200))
check("resize: caps the long side", max(PQB.prepare_image(big).size) == PQB.MAX_IMAGE_SIDE)
check("resize: leaves an already-small image alone",
      PQB.prepare_image(small).size == (300, 200))
check("resize: converts to RGB",
      PQB.prepare_image(Image.new("L", (64, 64))).mode == "RGB")

print()
print(f"{sum(OK)}/{len(OK)} checks passed")
sys.exit(0 if all(OK) else 1)
