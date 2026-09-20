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


def _letters(n):
    return [chr(ord("A") + i) for i in range(n)]


def algopuzzlevqa_rows(doc):
    # algopuzzlevqa/utils.py::algopuzzlevqa_doc_to_text, profile `qwen3_vl`. The gold in
    # the dataset is the option TEXT; the prompt asks for a letter, so the letter is what
    # is stored -- algopuzzlevqa_process_results maps it the same way.
    options = [str(o).strip() for o in doc["options"]]
    letters = _letters(len(options))
    block = "\n".join(f"{l}. {o}" for l, o in zip(letters, options))
    answer = str(doc["answer"]).strip()
    gold = next((l for l, o in zip(letters, options) if o == answer),
                next((l for l, o in zip(letters, options)
                      if o.lower() == answer.lower()), answer))
    return (f"Question: {doc['question'].strip()}\n{block}\n"
            "Answer with the option letter only.",
            gold, doc["image"], None, "data")


def pope_rows(doc):
    # pope/pope.yaml, profile `default`.
    return (doc["question"].strip() + "\nAnswer the question using a single word or phrase.",
            str(doc["answer"]), doc["image"], str(doc.get("question_id", "")), "test")


def _hrbench_rows(doc):
    # hrbench/utils.py::hrbench_doc_to_text and ::hrbench_doc_to_visual (base64 -> PIL).
    from PIL import Image

    opts = [(l, doc[l]) for l in "ABCDEFGH"
            if l in doc and doc[l] is not None and str(doc[l]) != "nan"]
    block = "".join(f"{l}. {v}\n" for l, v in opts)
    img = doc["image"]
    if not hasattr(img, "size"):
        img = Image.open(io.BytesIO(base64.b64decode(img)))
    return (f"{doc['question'].strip()}\n{block}Answer the option letter directly.",
            str(doc["answer"]), img.convert("RGB"), str(doc["index"]), None)


def hrbench4k_rows(doc):
    p, a, i, s, _ = _hrbench_rows(doc)
    return p, a, i, s, "hrbench_4k"


def hrbench8k_rows(doc):
    p, a, i, s, _ = _hrbench_rows(doc)
    return p, a, i, s, "hrbench_8k"


def omnispatial_rows(doc):
    """Question + options only.

    `omnispatial_doc_to_text` prepends a ~2,000-character manual-CoT system prompt and a
    format prompt. Those belong in the *system* turn, and this scan already puts the
    cold-start think-format system prompt there -- stacking a second one would change
    what is being looked at rather than reproduce the benchmark. The deviation is
    recorded in the row's `prompt_note`; nothing here claims the benchmark's score.
    """
    opts = list(doc["options"])
    block = "".join(f"\n{chr(65 + i)}. {o}" for i, o in enumerate(opts))
    return (doc["question"].strip() + block, str(doc["gt"]), doc["_image"],
            str(doc.get("image_path", "")), "test")


def p3_rows(doc):
    # salbench/utils.py::p3o3_doc_to_text with no prompt_kwargs (p3.yaml supplies none).
    return (doc["question"].strip() + "\nAnswer the question using a single word or phrase.",
            str(doc["answer"]), doc["image"], str(doc.get("image_id", "")), "test")


def mathvision_rows(doc):
    # mathvision/utils.py::mathvision_doc_to_text, with mathvision_testmini.yaml's
    # mc_prompt. Note the upstream function concatenates query_prompt and question with
    # no separator when there are choices; kept verbatim so the prompt is the real one.
    choices = list(doc["options"] or [])
    letters = _letters(len(choices))
    choices_str = "\n".join(f"{l}. {c}" for l, c in zip(letters, choices))
    q = 'Please solve the problem step by step and put your answer in one "\\boxed{}".'
    if choices_str:
        q += (f"{doc['question']}\nChoices: {choices_str}\n"
              "Answer the question with the option's letter from the given choices directly.")
    else:
        q += doc["question"]
    return q, str(doc["answer"]), doc["decoded_image"], str(doc["id"]), "testmini"


def wemath_rows(doc):
    # wemath/reasoning/utils.py::wemath_doc_to_text_cot. `image_path` holds a PIL image.
    return (doc["question"].strip() + "\n" + str(doc["option"]).strip(),
            str(doc["answer"]), doc["image_path"], str(doc["ID"]), "testmini")


def load_omnispatial(spec, n, seed):
    """The parquet + `image_files/` snapshot, read directly.

    `datasets.load_dataset("pangyyyyy/OmniSpatial")` does not resolve offline, and the
    rows carry an `image_path` rather than an image anyway -- upstream's doc_to_visual
    opens it out of the same snapshot. So the parquet is read and the image attached
    here, which is what `snapshot_download` would have handed the task.
    """
    from datasets import Dataset
    from huggingface_hub import snapshot_download
    from PIL import Image

    root = Path(snapshot_download(repo_id="pangyyyyy/OmniSpatial", repo_type="dataset",
                                  local_files_only=True))
    ds = Dataset.from_parquet(str(root / "test-00000-of-00001.parquet"))
    sub = ds if len(ds) <= n else ds.shuffle(seed=seed).select(range(n))
    rows = []
    for i in range(len(sub)):
        d = dict(sub[i])
        d["_image"] = Image.open(root / d["image_path"]).convert("RGB")
        rows.append(d)
    return rows


BENCH = {
    "mmstar": dict(path="Lin-Chen/MMStar", split="val", fn=mmstar_rows),
    "realworldqa": dict(path="lmms-lab-encoder/RealWorldQA", split="test", fn=realworldqa_rows),
    "mmerealworld": dict(path="yifanzhang114/MME-RealWorld-Lmms-eval", split="train",
                         fn=mmerealworld_rows),
    "algopuzzlevqa": dict(path="declare-lab/AlgoPuzzleVQA", split="data", fn=algopuzzlevqa_rows),
    "pope": dict(path="lmms-lab-encoder/POPE", split="test", fn=pope_rows),
    "hrbench4k": dict(path="DreamMr/HR-Bench", split="hrbench_4k", fn=hrbench4k_rows),
    "hrbench8k": dict(path="DreamMr/HR-Bench", split="hrbench_8k", fn=hrbench8k_rows),
    "omnispatial": dict(loader=load_omnispatial, fn=omnispatial_rows,
                        note="question+options only; the manual-CoT system prompt is dropped"),
    "p3": dict(path="salbench-vlm/salbench", split="test", fn=p3_rows),
    "mathvision": dict(path="MathLLMs/MathVision", split="testmini", fn=mathvision_rows),
    "wemath": dict(path="We-Math/We-Math", split="testmini", fn=wemath_rows),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", action="append", default=[],
                    help=f"repeatable; default all of {list(BENCH)}")
    ap.add_argument("--n", type=int, default=100, help="documents per benchmark")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--max-image-side", type=int, default=0,
                    help="downscale on the way in; 0 keeps the source resolution. HR-Bench "
                         "8K rows are ~8,000 px wide and `datasets` re-encodes every image "
                         "as PNG on save, so 100 of them is tens of GB on disk for pixels "
                         "no scan can afford to feed the model anyway")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from datasets import Dataset, load_dataset
    from PIL import Image

    names = args.benchmark or list(BENCH)
    bad = [n for n in names if n not in BENCH]
    if bad:
        raise SystemExit(f"unknown benchmark(s) {bad}; pick from {list(BENCH)}")

    rows = []
    for name in names:
        spec = BENCH[name]
        if spec.get("loader"):
            sub, total = spec["loader"](spec, args.n, args.seed), None
        else:
            ds = load_dataset(spec["path"], split=spec["split"])
            total = len(ds)
            # make_mini_tasks._take, verbatim: a deterministic permutation, then a prefix.
            sub = ds if total <= args.n else ds.shuffle(seed=args.seed).select(range(args.n))
        for doc_id in range(len(sub)):
            problem, solution, image, src_id, split = spec["fn"](sub[doc_id])
            if image is None:
                print(f"[{name}] doc {doc_id}: no image, skipped", flush=True)
                continue
            if args.max_image_side > 0 and hasattr(image, "size"):
                w, h = image.size
                if max(w, h) > args.max_image_side:
                    s = args.max_image_side / max(w, h)
                    image = image.resize((max(1, round(w * s)), max(1, round(h * s))),
                                         Image.BICUBIC)
            rows.append({
                "dataset": f"{name}_mini", "split": str(split or ""),
                # lmms-eval's doc_id IS the position in the mini sample, so keying on it
                # is what lets a row here be joined back to a bench_eval samples row.
                "question_id": str(doc_id), "source_id": str(src_id or doc_id),
                "problem": problem, "solution": solution,
                "image": image.convert("RGB") if hasattr(image, "convert") else image,
                "natural": True, "prompt_note": spec.get("note", ""),
            })
        print(f"[{name}] {len(sub)}"
              + (f" of {total}" if total else "") + " documents", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).save_to_disk(str(out))
    print(f"[out] {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
