#!/usr/bin/env python
"""Build LASER's 45K RL corpus as verl multimodal parquet.

    python build_laser_data.py --out-dir cold_data/laser/verl
    python build_laser_data.py --out-dir DIR --smoke 64      # a tiny set for the smoke run

LASER trains on "45K RL samples filtered from MMR1-RL and ReVisual-R1" and released
neither the parquet nor a prep script -- `train.sh` just says "point TRAIN_FILES at your
own". This rebuilds it.

THE COMPOSITION IS NOT GUESSWORK, AND IT NEEDS NO FILTERING

    MMR1/MMR1-RL   14,996 rows
    csfufu/mmrl    30,937 rows   (ReVisual-R1's multimodal RL set, arXiv 2506.04207)
                   ------
                   45,933  =  the paper's "45K"

and `mmrl`'s own `dataset` column reproduces the paper's Table 5 almost line for line:
iconqa 8,303 (8.3K), geometry3k 1,484 (1.5K), geoqa 1,253 (1.2K), olympiadbench 478
(0.4K), and an unlabelled bucket of 11,012 against "Revisual-R1 Curated 11.5K". So the
curation the paper describes -- "explicit attention to quality, difficulty, and
diversity" -- was applied when those two artifacts were published, not on top of them.
Both corpora are used IN FULL. Anything that drops rows here is a deviation, and
`--max-rows` says so when it is used.

The one number that does not line up is Table 5's "MMR1 Curated 8K" against MMR1-RL's
14,996. Since the totals land at 45.9K against their 45K, taking both whole is the
reading that reproduces the size; if MMR1-RL was resampled after the paper, this build is
7K larger than theirs and the difference is in MMR1's share.

WHAT verl NEEDS, AND WHY THE ROWS COME OUT THIS SHAPE

`RLHFDataset._build_messages` pops `prompt` and, when an image key is present, splits each
message's **string** content on `<image>` / `<video>` itself, building the content list.
So `content` must stay a plain string with the placeholder in it -- which is exactly the
form `problem` already has in both corpora, so the text passes through untouched. The
`images` column is already HF's `{bytes, path}` struct list, which is what
`process_image` consumes. `reward_model.ground_truth` is what the `dapo` reward manager
reads, and `data_source` is its `reward_fn_key`.

THE PROMPT IS THE ONE THING THEY DID NOT PUBLISH

Their reward requires a specific shape -- `format_reward` in `openr1_verl.py` wants
exactly one `<think></think>`, exactly one `<answer></answer>`, the whole completion to
match `^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$`, and exactly one `\boxed{}`
inside the answer; `_format_reward_think`, which gates the ATTENTION rewards separately,
wants the same tags. A prompt that does not ask for that shape scores 0 on format for
every rollout, which zeroes the attention terms too and would look like a broken port
rather than a broken prompt.

No such prompt exists anywhere in their repo. `SYSTEM_PROMPT` below is written to their
checkers rather than copied from them, and it is the largest single deviation in this
build. It is a module constant so a later correction is one edit and a rebuild.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent

#: Written against `openr1_verl.format_reward` and `dp_actor._format_reward_think`, NOT
#: copied from upstream -- see the module docstring. Both tag pairs and the single
#: \boxed{} are load-bearing: miss any one and every rollout scores format 0.
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question about an image, "
    "and the Assistant solves it. The Assistant first thinks about the reasoning process "
    "in the mind and then provides the user with the answer. The reasoning process is "
    "enclosed within <think> </think> tags and the answer is enclosed within <answer> "
    "</answer> tags. The final answer inside <answer> </answer> must be put in \\boxed{}, "
    "i.e. <think> reasoning process here </think><answer> \\boxed{answer here} </answer>."
)

SOURCES = {
    "mmr1_rl": dict(glob="MMR1-RL/data/*.parquet", source_col=None,
                    note="MMR1/MMR1-RL, 14,996 rows"),
    "revisual_r1": dict(glob="mmrl/*.parquet", source_col="dataset",
                        note="csfufu/mmrl, ReVisual-R1's multimodal RL set, 30,937 rows"),
}


def load_rows(data_dir: Path, name: str, spec: dict):
    """One corpus as a list of verl rows. Images and problem text pass through as-is."""
    import pyarrow.parquet as pq

    files = sorted(data_dir.glob(spec["glob"]))
    if not files:
        raise SystemExit(f"no parquet for {name} under {data_dir / spec['glob']}")
    want = ["problem", "answer", "images"] + ([spec["source_col"]] if spec["source_col"] else [])
    out = []
    for f in files:
        t = pq.read_table(f, columns=want)
        cols = {n: t.column(n).to_pylist() for n in want}
        for i in range(t.num_rows):
            problem = cols["problem"][i]
            images = cols["images"][i]
            if not problem or not images:
                continue          # a row with no picture cannot carry a visual reward
            sub = cols[spec["source_col"]][i] if spec["source_col"] else None
            out.append({
                "data_source": f"{name}/{sub}" if sub else name,
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    # `problem` already carries the <image> placeholder that
                    # _build_messages splits on. Left exactly as published.
                    {"role": "user", "content": problem},
                ],
                "images": images,
                "ability": "visual_reasoning",
                "reward_model": {"style": "rule", "ground_truth": cols["answer"][i]},
                "extra_info": {"corpus": name, "sub_source": sub, "index": len(out)},
            })
    return out


def write_parquet(rows, path: Path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    # An explicit schema, because inferring one from `extra_info` makes `sub_source` a
    # null-typed column on any shard where every row happens to lack it, and two shards
    # with different types for the same field cannot be read back as one dataset.
    schema = pa.schema([
        ("data_source", pa.string()),
        ("prompt", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
        ("images", pa.list_(pa.struct([("bytes", pa.binary()), ("path", pa.string())]))),
        ("ability", pa.string()),
        ("reward_model", pa.struct([("style", pa.string()), ("ground_truth", pa.string())])),
        ("extra_info", pa.struct([("corpus", pa.string()), ("sub_source", pa.string()),
                                  ("index", pa.int64())])),
    ])
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path, compression="zstd")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(REPO / "cold_data" / "laser"),
                    help="where MMR1-RL/ and mmrl/ were downloaded")
    ap.add_argument("--out-dir", default=str(REPO / "cold_data" / "laser" / "verl"))
    ap.add_argument("--val-size", type=int, default=512,
                    help="rows held out for VAL_FILES. Their val set was not released; "
                         "this is carved from the same pool, stratified by source")
    ap.add_argument("--smoke", type=int, default=0,
                    help="write a tiny train/val pair of this many rows instead, for the "
                         "smoke run. NOT a training corpus")
    ap.add_argument("--max-rows", type=int, default=0,
                    help="cap the corpus. A DEVIATION from LASER, reported as one")
    ap.add_argument("--seed", type=int, default=20260910)
    args = ap.parse_args()

    import numpy as np

    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    rows, by_corpus = [], {}
    for name, spec in SOURCES.items():
        got = load_rows(data_dir, name, spec)
        by_corpus[name] = len(got)
        rows += got
        print(f"[{name:<12}] {len(got):>6} rows   {spec['note']}", flush=True)

    total = len(rows)
    print(f"\ntotal {total} rows (the paper's 45K = 45,933 from these two corpora)")
    if total < 45000:
        print("WARNING: fewer than 45,000 rows -- a corpus is short or partly unreadable.")

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(total)
    rows = [rows[i] for i in order]

    if args.smoke:
        n = min(args.smoke, total)
        train, val = rows[:n], rows[: max(8, n // 4)]
        tag = f"smoke{n}"
        print(f"\nSMOKE SET: {len(train)} train / {len(val)} val. This is a mechanical "
              f"check of the port,\nnot a training corpus -- do not report a number off it.")
    else:
        if args.max_rows and args.max_rows < total:
            print(f"\nDEVIATION: capping at {args.max_rows} of {total} rows "
                  f"(--max-rows). LASER uses all of them.")
            rows = rows[: args.max_rows]
        val_n = min(args.val_size, len(rows) // 10)
        val, train = rows[:val_n], rows[val_n:]
        tag = "45k"
        print(f"\ntrain {len(train)} / val {len(val)}  (their val split was not released; "
              f"carved here)")

    tr = write_parquet(train, out_dir / f"train_{tag}.parquet")
    va = write_parquet(val, out_dir / f"val_{tag}.parquet")
    meta = {
        "train": str(tr), "val": str(va), "n_train": len(train), "n_val": len(val),
        "by_corpus": by_corpus, "seed": args.seed, "smoke": bool(args.smoke),
        "max_rows": args.max_rows or None,
        "system_prompt": SYSTEM_PROMPT,
        "note": "system_prompt is written against their reward checkers, not copied from "
                "upstream -- see build_laser_data.py's docstring.",
    }
    (out_dir / f"manifest_{tag}.json").write_text(json.dumps(meta, indent=2))
    print(f"\n  {tr}\n  {va}\n  {out_dir / f'manifest_{tag}.json'}")
    print("\nThen:")
    print(f"  conda activate laser; and env MODEL_PATH=Qwen/Qwen3-VL-8B-Instruct \\")
    print(f"      TRAIN_FILES={tr} VAL_FILES={va} bash laser_fork/train.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
