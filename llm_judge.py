#!/usr/bin/env python
"""LLM-as-judge over (question, gold, extracted answer), with a cache on disk.

`trl/rewards/openai_rewards.py` holds the judge the TRAINER uses, and it is not reusable
for analysis: it pulls the prediction out of a completion with a `</think>` regex and
masks anything that has none, which is every base Qwen3-VL completion on a run with no
system prompt. It would score the baseline 0 for its format and call that accuracy -- the
same artefact as the LogicVista and MathVision parsers. This version takes an ALREADY
EXTRACTED answer, so whatever `answer_grading.extract_answer` returned is what the judge
sees, and the string grades and the judge grade are computed on the same object.

THE CACHE KEY IS THE CONTENT, not the row: `[question, gold, answer]`. So two probes that
happen to ask about the same triple share the answer, a rerun costs nothing, and a cache
written by one script is readable by another. `outputs/human_box_correct/*.judge_cache.json`
is in this format and can be passed straight in.

Scores are returned on [0, 1] as ((1-5) - 1) / 4, or None where the judge failed after
retries -- a failure must be distinguishable from a 0, because a run that silently turns
gateway errors into wrong answers reports a model as worse than it is.

    from llm_judge import judge_scores
    scores = judge_scores(items, workers=8, cache_path="OUT/judge_cache.json")

where each item is a dict with `question`, `gt_answer`, `answer`.

Environment: NVIDIA_API_KEY (or OPENAI_API_KEY), OPENAI_BASE_URL, JUDGE_MODEL.

NOTE ON DUPLICATION. `human_box_vs_correct.py`, on the unmerged branch
`analysis/human-box-vs-correct`, carries a copy of this function. When that branch merges
it should import from here instead; the cache format is identical by construction so no
cached judgement is lost in the switch.
"""

from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

JUDGE_SYSTEM = (
    "You are an intelligent chatbot designed for evaluating the correctness of "
    "generative outputs for question-answer pairs. Your task is to compare the "
    "predicted answer with the correct answer and determine if they match "
    "meaningfully. Here's how you can accomplish the task:"
    "------"
    "##INSTRUCTIONS: "
    "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
    "- Consider synonyms or paraphrases as valid matches.\n"
    "- Evaluate the correctness of the prediction compared to the answer."
)


def cache_key(question: str, gold: str, answer: str) -> str:
    return json.dumps([question, gold, answer], sort_keys=True)


def judge_scores(items, workers: int = 8, cache_path=None, verbose: bool = True):
    """-> [score in [0,1] or None], one per item. `items` need question/gt_answer/answer.

    An empty answer is scored 0.0 without a call: the model produced nothing to judge,
    which is wrong rather than unmeasurable, and spending a call to be told so costs money
    and rate limit.
    """
    import openai

    cache = {}
    if cache_path and Path(cache_path).exists():
        cache = json.loads(Path(cache_path).read_text())

    # Everything already cached, or empty, needs no client at all -- so a report can be
    # regenerated from a cache with no key in the environment. And the batch is
    # DEDUPLICATED on the cache key before anything is asked: four models on the same
    # 1,800 pictures produce ~490 repeated (question, gold, answer) triples, and paying
    # for each of them twice is money and rate limit for an identical answer.
    todo = sorted({cache_key(it["question"], it["gt_answer"], (it.get("answer") or "").strip())
                   for it in items
                   if (it.get("answer") or "").strip()} - set(cache))
    if verbose:
        n_empty = sum(1 for it in items if not (it.get("answer") or "").strip())
        print(f"[judge] {len(items)} items, {n_empty} empty (scored 0 without a call), "
              f"{len(todo)} distinct uncached triples to ask", flush=True)

    client = model = None
    if todo:
        key = os.environ.get("NVIDIA_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise SystemExit(
                f"{len(todo)} items are not in the cache and NVIDIA_API_KEY (or "
                "OPENAI_API_KEY) is not set. Set it and rerun, or point --judge-cache at "
                "a cache that covers them.")
        client = openai.OpenAI(
            api_key=key,
            base_url=os.environ.get("OPENAI_BASE_URL", "https://inference-api.nvidia.com"))
        model = os.environ.get("JUDGE_MODEL", "azure/openai/gpt-4o-mini")
        if verbose:
            print(f"[judge] model={model}  workers={workers}", flush=True)

    def ask(ck):
        question, gold, ans = json.loads(ck)
        content = (
            f"I will give you an image and the following text as inputs:\n\n"
            f"1. **Question Related to the Image**: {question}\n"
            f"2. **Ground Truth Answer**: {gold}\n"
            f"3. **Model Predicted Answer**: {ans}\n\n"
            "Your task is to evaluate the model's predicted answer against the ground "
            "truth answer, based on the context provided by the image and the question. "
            "Consider the following criteria for evaluation:"
            "- **Relevance**: Does the predicted answer directly address the question "
            "posed, considering the information provided in the image?"
            "- **Accuracy**: Compare the predicted answer to the ground truth answer. "
            "Does the prediction accurately reflect the information given in the ground "
            "truth answer without introducing factual inaccuracies?"
            "**Output Format**:"
            "Score: <a integer score of quality from 1-5>")
        for attempt in range(5):
            try:
                r = client.chat.completions.create(
                    model=model, temperature=0, max_tokens=512, timeout=120,
                    messages=[{"role": "system", "content": JUDGE_SYSTEM},
                              {"role": "user", "content": content}])
                m = re.search(r"Score:\s*(\d+)", r.choices[0].message.content or "")
                return (int(m.group(1)) - 1.0) / 4.0 if m else None
            except Exception as e:  # transient gateway errors are the common case
                if attempt == 4:
                    print(f"[judge] giving up on one item: {type(e).__name__}: "
                          f"{str(e)[:120]}", flush=True)
                    return None
                time.sleep(0.2 * 2 ** attempt)

    fresh = {}
    if todo:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for ck, s in zip(todo, ex.map(ask, todo)):
                if s is not None:             # never cache a failure as an answer
                    fresh[ck] = s
        if cache_path:
            cache.update(fresh)
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            Path(cache_path).write_text(json.dumps(cache))
            if verbose:
                print(f"[judge] cache now {len(cache)} entries (+{len(fresh)}) at "
                      f"{cache_path}", flush=True)
        else:
            cache.update(fresh)

    out = []
    for it in items:
        ans = (it.get("answer") or "").strip()
        out.append(0.0 if not ans
                   else cache.get(cache_key(it["question"], it["gt_answer"], ans)))
    n_fail = sum(1 for s in out if s is None)
    if n_fail and verbose:
        print(f"[judge] {n_fail} items came back None and are excluded, not counted wrong",
              flush=True)
    return out
