#!/usr/bin/env python
"""Ground each dataset row ONCE, on its question, and write the boxes for training to read.

The overlap reward normally calls Grounding-DINO once per observe step, on that step's
own sentence. dino_text_sensitivity.py measured what that buys, on the cold-start policy
over 30 images and 859 steps: re-grounding a step with the sample's QUESTION instead of
its own sentence recovers the step's real mask at IoU 0.649 / closeness 0.785, where a
DIFFERENT REAL STEP OF THE SAME CHAIN gets 0.635 / 0.721 -- at the same mask size (median
union 0.568 against the real 0.578). The per-step call is not buying a per-step mask.

So do it once, offline, per dataset ROW, and let the run read the answer:

    python precompute_question_boxes.py \
        --dataset cold_data/grpo_sets/set_a \
        --out cold_data/question_boxes/set_a.json

then train with `--overlap_question_boxes cold_data/question_boxes/set_a.json`, which
makes the reward skip Grounding-DINO entirely.

What is stored is the RAW box list -- every box above --box-threshold, before any area
filter -- so --max_box_area and --max_union_area stay run-time knobs. --box-threshold is
applied inside the detector and cannot be re-applied later, so it is recorded and the
trainer refuses a cache built at a different one. So is --max-image-side: the detector
sees a different picture at a different resolution.

Rows are keyed by (dataset, split, question_id), which is unique in every corpus this
trainer accepts. The question text is NOT part of the key -- questions repeat across
images (35343 distinct strings over set_a's 50000 rows) and keying on the text would
collapse different pictures onto one box list.

One GPU. ~20 groundings/second on an A100, so ~7 min for the 8k and ~40 min for a 50k
set; --shard/--num-shards splits it across cards and `--merge` puts the pieces back
together:

    python precompute_question_boxes.py --dataset ... --out out.shard0.json --shard 0 --num-shards 8
    ...
    python precompute_question_boxes.py --merge out.shard*.json --out out.json

launch_precompute_question_boxes_job.sh does that fan-out as one SLURM job.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent

# Must match trl/grpo_vlm_qwen3.py's MAX_IMAGE_SIDE and its prepare_image(): the boxes
# are only the boxes the run would have got if the detector sees the same picture. The
# value is written into the cache and the trainer refuses a mismatch, so a change on
# either side is a loud failure rather than a silently different reward.
MAX_IMAGE_SIDE = 512

# Qwen3-VL's vision tower: 16px patches merged 2x2, so one cell of the grid the reward
# scores on is 32px of the (already resized) image, and smart_resize rounds each side to a
# multiple of that. Recorded per row for the SUMMARY ONLY -- the reward reads the true grid
# off the step's own attention map and never looks at this. It is here because the
# alternative, a fixed nominal grid, biases the union fraction: rasterisation gives every
# box at least one row and column, so a coarser grid reports MORE coverage for the same
# boxes, and the summary could not then be compared with the per-step numbers it exists to
# be compared with. Checked against overlap_probe's stored grids: a 500x332 image gives
# (10, 16), which is what the probe recorded.
GRID_CELL_PX = 32


def _load_module(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _stub_rewards_package():
    """overlap_rewards does `from . import roll_null`, which needs a parent package."""
    for _n, _p in (("trl", REPO / "trl"), ("trl.rewards", REPO / "trl" / "rewards")):
        if _n not in sys.modules:
            _m = types.ModuleType(_n)
            _m.__path__ = [str(_p)]
            sys.modules[_n] = _m


_stub_rewards_package()
OREW = _load_module("trl.rewards.overlap_rewards", "trl/rewards/overlap_rewards.py")


# ---------------------------------------------------------------------------
def load_dataset_like_the_trainer(dataset_name: str, split: str):
    """Exactly grpo_vlm_qwen3.py's loader, minus the train/test carve.

    The carve is deliberately not applied: it holds out 100 rows the run never trains on,
    but the cache costs nothing to build for them and a cache that covers the whole corpus
    survives a change of seed or holdout size.
    """
    from datasets import load_dataset, load_from_disk

    if os.path.isfile(os.path.join(dataset_name, "dataset_info.json")) or os.path.isfile(
        os.path.join(dataset_name, "dataset_dict.json")
    ):
        ds = load_from_disk(dataset_name)
        if not hasattr(ds, "train_test_split"):
            ds = ds[split]
    else:
        ds = load_dataset(dataset_name, split=split)
    return ds


def prepare_image(image):
    """trl/grpo_vlm_qwen3.py's prepare_image, verbatim.

    Kept as a copy rather than imported because the original is a closure inside the
    trainer's main(); the two are pinned together by MAX_IMAGE_SIDE being written into
    the cache and checked at load, and by test_question_boxes_cpu.py.
    """
    from PIL import Image

    width, height = image.size
    if max(width, height) > MAX_IMAGE_SIDE:
        scale = MAX_IMAGE_SIDE / max(width, height)
        image = image.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.BICUBIC,
        )
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def patch_grid(image) -> list[int]:
    """(gh, gw) the reward will score on. Diagnostic only -- see GRID_CELL_PX."""
    w, h = image.size
    return [max(1, round(h / GRID_CELL_PX)), max(1, round(w / GRID_CELL_PX))]


# ---------------------------------------------------------------------------
def build(args) -> dict:
    ds = load_dataset_like_the_trainer(args.dataset, args.dataset_split)

    for col in (*OREW.QBOX_KEY_COLUMNS, args.text_column, "image"):
        if col not in ds.column_names:
            raise SystemExit(
                f"{args.dataset}: no `{col}` column (has {ds.column_names}). "
                "The cache is keyed by "
                f"{', '.join(OREW.QBOX_KEY_COLUMNS)} and grounds `{args.text_column}`."
            )

    # Every row of the corpus, then this shard's slice of it. Sharding by stride rather
    # than by block so each shard sees the same mix of source datasets and the timing of
    # one shard predicts the rest.
    idx = list(range(len(ds)))
    if args.limit:
        idx = idx[: args.limit]
    mine = [i for i in idx if i % args.num_shards == args.shard]

    # Two different batch sizes, and conflating them is what makes every call OOM once:
    # --rows-per-call is how many rows are decoded and handed over at a time, and
    # --batch-size is how many of those Grounding-DINO forwards together. _dino_boxes
    # halves its own batch on OOM, so it only has room to do that if it is given more than
    # one batch's worth.
    OREW.configure(box_threshold=args.box_threshold, dino_batch_size=args.batch_size)
    print(f"[question_boxes] {args.dataset}: {len(ds)} rows, shard {args.shard}/"
          f"{args.num_shards} takes {len(mine)}; box_threshold={args.box_threshold} "
          f"max_image_side={MAX_IMAGE_SIDE} dino_batch={args.batch_size} "
          f"rows_per_call={args.rows_per_call}", flush=True)

    out: dict[str, list] = {}
    grids: dict[str, list] = {}
    done = 0
    for start in range(0, len(mine), args.rows_per_call):
        rows = [ds[i] for i in mine[start:start + args.rows_per_call]]
        images = [prepare_image(r["image"]) for r in rows]
        texts = [r[args.text_column] or "object" for r in rows]
        boxes = OREW._dino_boxes(images, texts)
        for r, im, b in zip(rows, images, boxes):
            key = OREW.qbox_key(*(r[c] for c in OREW.QBOX_KEY_COLUMNS))
            if key in out:
                raise SystemExit(
                    f"{args.dataset}: duplicate row key {key!r}. The cache is a mapping, "
                    "so two rows sharing a key would silently share one box list."
                )
            out[key] = [[round(float(v), 5) for v in box] for box in (b or [])]
            grids[key] = patch_grid(im)
        done += len(rows)
        print(f"[question_boxes] {done}/{len(mine)}", flush=True)

    return {
        "grids": grids,
        "version": OREW.QBOX_VERSION,
        "config": {
            "box_threshold": float(args.box_threshold),
            "max_image_side": int(MAX_IMAGE_SIDE),
            "dino_hf_id": OREW.GROUNDING_DINO_HF_ID,
            "text_column": args.text_column,
            "dataset": args.dataset,
            "dataset_rows": len(ds),
            "key_columns": list(OREW.QBOX_KEY_COLUMNS),
        },
        "shard": {"shard": args.shard, "num_shards": args.num_shards, "rows": len(mine)},
        "boxes": out,
    }


def merge(paths, out_path):
    """Combine shard files. Their configs must agree, or the halves are not comparable."""
    merged, grids, cfg, version = {}, {}, None, None
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        if cfg is None:
            cfg, version = d["config"], d["version"]
        elif d["config"] != cfg or d["version"] != version:
            raise SystemExit(
                f"{p} was built with a different configuration than {paths[0]}:\n"
                f"  {json.dumps(d['config'], sort_keys=True)}\n"
                f"  {json.dumps(cfg, sort_keys=True)}"
            )
        for k, v in d["boxes"].items():
            if k in merged:
                raise SystemExit(f"{p}: key {k!r} already came from an earlier shard")
            merged[k] = v
        grids.update(d.get("grids") or {})
        print(f"[merge] {p}: {len(d['boxes'])} rows", flush=True)

    want = cfg.get("dataset_rows")
    if want is not None and len(merged) != want:
        raise SystemExit(
            f"merged {len(merged)} rows but the corpus has {want}. A shard is missing "
            "or was built with --limit; refusing to write a partial cache."
        )
    return {"version": version, "config": cfg, "grids": grids, "boxes": merged}


# ---------------------------------------------------------------------------
def summarise(d) -> str:
    """What the run will actually see, so a bad cache is visible before training on it."""
    boxes = d["boxes"]
    grids = d.get("grids") or {}
    n = len(boxes)
    raw = np.array([len(v) for v in boxes.values()], dtype=float)
    cap = 0.5  # the --max_box_area default; the run's own value may differ
    kept = np.array([sum(1 for b in v if OREW._box_area(b) <= cap) for v in boxes.values()],
                    dtype=float)

    # Union coverage on each row's own patch grid, which is the whole point of recording
    # it -- see GRID_CELL_PX. A row whose union swallows the grid rasterises to None and
    # the REWARD SKIPS IT, so it is counted here rather than folded in as 1.0.
    fracs, degenerate = [], 0
    for k, v in boxes.items():
        g = grids.get(k)
        if not g:
            continue
        m = OREW._union_mask([b for b in v if OREW._box_area(b) <= cap], g[0], g[1],
                             apply_union_cap=False)
        if m is None:
            degenerate += 1
            continue
        fracs.append(float(m.sum()) / m.size)
    fr = np.array(fracs, dtype=float)

    L = []
    P = L.append
    P(f"  rows                              {n}")
    P(f"  grounded nothing at all           {int((raw == 0).sum())}  "
      f"({(raw == 0).mean():.1%} -- these rows are MASKED by the reward, not scored 0)")
    P(f"  boxes per row (raw)               mean {raw.mean():.1f}, median {np.median(raw):.0f}")
    P(f"  boxes per row (after area<={cap})   mean {kept.mean():.1f}, median {np.median(kept):.0f}")
    if not fr.size:
        P("  union coverage                    (no per-row grids stored; rebuild to get it)")
        return "\n".join(L)
    P(f"  union covers the whole grid       {degenerate}  "
      f"({degenerate / n:.1%} -- also MASKED, like an ungroundable row)")
    P(f"  union coverage, per-row grid      mean {fr.mean():.3f}, "
      f"median {np.median(fr):.3f}, p10 {np.percentile(fr, 10):.3f}, "
      f"p90 {np.percentile(fr, 90):.3f}")
    P("  (dino_text_sensitivity.py measured the question mask's median coverage at 0.568")
    P("   and the per-step masks' at 0.578, on the real grids of a val_natural sample --")
    P("   a median far from that on a natural corpus is a bug, not a finding)")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", help="what --dataset_name would be given to the trainer")
    ap.add_argument("--dataset-split", default="train")
    ap.add_argument("--text-column", default="problem",
                    help="the column grounded once per row (default: the question)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--box-threshold", type=float, default=0.10,
                    help="must match the run's --box_threshold; baked into the cache")
    ap.add_argument("--batch-size", type=int, default=16,
                    help="images Grounding-DINO forwards at once. 32 OOMs and halves "
                         "itself on an 80GB card at 512px, which costs a retry per call")
    ap.add_argument("--rows-per-call", type=int, default=256,
                    help="rows decoded and handed to the detector at a time. Larger than "
                         "--batch-size on purpose, so its OOM halving has room to work")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="0 = the whole corpus")
    ap.add_argument("--merge", nargs="+", default=None,
                    help="combine shard files into --out instead of grounding anything")
    args = ap.parse_args()

    if args.merge:
        d = merge(args.merge, args.out)
    else:
        if not args.dataset:
            raise SystemExit("--dataset is required unless --merge is given")
        d = build(args)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f)
    os.replace(tmp, args.out)

    print("")
    print(f"[question_boxes] wrote {args.out}")
    if not d.get("shard") or d["shard"]["num_shards"] == 1:
        print(summarise(d))
    else:
        print(f"  shard {d['shard']['shard']}/{d['shard']['num_shards']}, "
              f"{len(d['boxes'])} rows -- rerun with --merge for the summary")


if __name__ == "__main__":
    main()
