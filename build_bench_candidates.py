#!/usr/bin/env python
"""Turn the natural mini-benchmark documents into a dataset `saliency_viz.py` can scan.

The saliency path reads `cold_data/grpo_sets/*`-shaped rows -- `problem`, `solution`,
`image`, plus the `(dataset, split, question_id)` triple -- and the benchmarks are
lmms-eval tasks, which the viz has never been able to look at. This writes the bridge:
the SAME 100 documents per benchmark that `bench_eval` scores, in that shape.

"The same documents" is the whole point, so the sample is reproduced rather than redrawn:
`eval_mini/make_mini_tasks.py` emits `dataset.shuffle(seed=1234).select(range(100))`, and
row i of the result is lmms-eval's `doc_id` i. The prompt is copied from each task's own
`doc_to_text` and the profile lmms-eval would pick for `--model qwen3_vl` (mmstar has a
`qwen3_vl` key; the other two do not and fall back to `default`), so the question the scan
asks is the question the benchmark asked.

Two things it is NOT:

  * Not the benchmark's own verdict. `overlap_probe.prepare_image` caps the long side at
    512 px, because that is what the reward and the training saw; lmms-eval passes the
    full image. On MME-RealWorld, whose pictures are remote-sensing frames several
    thousand pixels wide, that cap is not cosmetic -- a scan answer that disagrees with
    the benchmark's is expected, and the scan's own answer is the one to quote.
  * Not a grouped benchmark. `--benchmark mme` is refused: MME is scored over yes/no
    PAIRS and make_mini_tasks draws groups, not rows, so reproducing its sample needs the
    grouped sampler and its doc_ids renumber with n. Nothing here needs it.

    python build_bench_candidates.py --out outputs/fig1-multistep/bench_mini_natural \
        [--benchmark mmstar --benchmark realworldqa --benchmark mmerealworld] [--n 100]
"""

from __future__ import annotations

import argparse
import base64
import io
from pathlib import Path

SEED = 1234                    # eval_mini/make_mini_tasks.py --seed default


def mmstar_rows(doc):
    # mmstar/_default_template_yaml, profile `qwen3_vl`.
    return (f"Question: {doc['question'].strip()}Answer with the option letter only.",
            str(doc["answer"]), doc["image"], str(doc["index"]), "val")


def realworldqa_rows(doc):
    # realworldqa.yaml, profile `default`: both pre_prompt and post_prompt are "".
    return (doc["question"].strip(), str(doc["answer"]), doc["image"],
            str(doc.get("image_path", "")), "test")


def mmerealworld_rows(doc):
    # mme_realworld/utils.py::mme_realworld_doc_to_text, verbatim -- it ignores
    # lmms_eval_specific_kwargs and builds the option block itself.
    from PIL import Image

    option_prompt = ("The choices are listed below:\n"
                     + "\n".join(doc["multi-choice options"]) + "\n")
    q = (doc["question"] + " " + option_prompt
         + "Select the best answer to the above multiple-choice question based on the "
           "image. Respond with only the letter (A, B, C, D, or E) of the correct "
           "option.\nThe best answer is: ")
    # The column is called `bytes` and holds a base64 STRING; lmms-eval's
    # mme_realworld_doc_to_visual base64-decodes it before opening.
    img = doc["bytes"]
    if not hasattr(img, "size"):
        raw = base64.b64decode(img) if isinstance(img, str) else img
        img = Image.open(io.BytesIO(raw))
    return q, str(doc["answer"]), img.convert("RGB"), str(doc["index"]), "train"


BENCH = {
    "mmstar": dict(path="Lin-Chen/MMStar", split="val", fn=mmstar_rows),
    "realworldqa": dict(path="lmms-lab-encoder/RealWorldQA", split="test", fn=realworldqa_rows),
    "mmerealworld": dict(path="yifanzhang114/MME-RealWorld-Lmms-eval", split="train",
                         fn=mmerealworld_rows),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", action="append", default=[],
                    help=f"repeatable; default all of {list(BENCH)}")
    ap.add_argument("--n", type=int, default=100, help="documents per benchmark")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from datasets import Dataset, load_dataset

    names = args.benchmark or list(BENCH)
    bad = [n for n in names if n not in BENCH]
    if bad:
        raise SystemExit(f"unknown benchmark(s) {bad}; pick from {list(BENCH)}")

    rows = []
    for name in names:
        spec = BENCH[name]
        ds = load_dataset(spec["path"], split=spec["split"])
        # make_mini_tasks._take, verbatim: a deterministic permutation, then a prefix.
        sub = ds if len(ds) <= args.n else ds.shuffle(seed=args.seed).select(range(args.n))
        for doc_id in range(len(sub)):
            problem, solution, image, src_id, split = spec["fn"](sub[doc_id])
            rows.append({
                "dataset": f"{name}_mini", "split": split,
                # lmms-eval's doc_id IS the position in the mini sample, so keying on it
                # is what lets a row here be joined back to a bench_eval samples row.
                "question_id": str(doc_id), "source_id": src_id,
                "problem": problem, "solution": solution, "image": image,
                "natural": True,
            })
        print(f"[{name}] {len(sub)} of {len(ds)} documents", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).save_to_disk(str(out))
    print(f"[out] {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
