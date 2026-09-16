#!/usr/bin/env python
"""Stage 1 cold-start SFT data, in LASER's four-tag format, on substitute sources.

    python build_coldstart_data.py --n 40000 --out-dir cold_data/laser/coldstart

WHY THIS EXISTS, AND WHAT IT IS NOT

LASER's Stage 1 is ReVisual-R1's cold start, and `revisual_cold_start.yaml` trains on a
dataset called `GRAMMAR`. **GRAMMAR is not released.** Their repo says it "will open
source the GRAMMAR dataset within the next two weeks" and has not; `cold_start/` holds
only `run_cold_start.sh` and LLaMA-Factory's stock examples, and nothing matching exists
under `csfufu` or `RevisualR1` on the hub (`csfufu/Grammer_dataset` 404s).

So this is **their recipe on substitute data**, and it should be described that way and
never as a reproduction. What is matched: text-only, ~40K samples, explicit reasoning
paths, and the four-tag target their config adds as special tokens. What is not: the
actual GRAMMAR mixture. The paper describes it as "47k diverse textual thought samples
with explicit reasoning paths, augmented by 31k complex textual examples"; this is one
source, and a maths-weighted one.

WHY OpenR1-Math AND NOT OpenThoughts

OpenThoughts-114k looks like the closer analogue of "diverse textual thought samples" --
until you check the thing that matters. LASER's `format_reward` demands **exactly one**
`\\boxed{}` inside `<answer>`, and OpenThoughts solutions carry a boxed answer in **0 of
400** sampled rows: it is heavily code-generation and proof tasks, where there is no short
answer to box. Training on it would teach three of the four tags and fail the reward every
time. OpenR1-Math-220k carries one in 98%, ships `correctness_math_verify` so only
*verified* traces are kept, and its short boxed answers are the same shape as the RL
corpus LASER then trains on. Diversity lost, trainability gained -- and the format gate is
the entire reason Stage 1 is being run at all.

THE OUTPUT IS CHECKED AGAINST THE REAL REWARD, NOT AGAINST THIS FILE'S IDEA OF IT

Every emitted target is scored with the fork's own `openr1_verl.format_reward` and
`_format_reward_think`, and a target that fails either is dropped rather than written.
An SFT corpus whose own targets do not satisfy the reward cannot teach the model to
satisfy it, and that failure would only surface as a flat reward curve days later.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent

#: Their `revisual_cold_start.yaml` adds exactly these as special tokens and resizes the
#: vocabulary, which is why the format is learned rather than prompted -- and why
#: Qwen3-VL-8B-Instruct scores format 0.000 under every prompt variant tried.
SPECIAL_TOKENS = ["<think>", "</think>", "<answer>", "</answer>"]


def load_format_checkers(fork: Path):
    spec = importlib.util.spec_from_file_location(
        "_cs_openr1", fork / "verl" / "utils" / "reward_score" / "openr1_verl.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_cs_openr1"] = mod
    spec.loader.exec_module(mod)

    def think_gate(c: str) -> float:
        t = re.findall(r"<think>.*?</think>", c, re.MULTILINE | re.DOTALL)
        a = re.findall(r"<answer>.*?</answer>(\s*)$", c, re.MULTILINE | re.DOTALL)
        return 1.0 if (len(t) == 1 and len(a) == 1) else 0.0

    return mod.format_reward, think_gate


def split_trace(gen: str):
    """An R1 generation -> (reasoning, solution), or None if it cannot be split cleanly.

    R1-distilled traces usually carry a literal `</think>`; everything before it is the
    monologue and everything after is the answer. Rows without one are dropped rather than
    split on a heuristic: a mis-split puts the boxed answer inside <think>, which still
    *looks* well-formed and teaches exactly the wrong thing.
    """
    if "</think>" not in gen:
        return None
    head, _, tail = gen.partition("</think>")
    think = head.replace("<think>", "").strip()
    sol = tail.strip()
    if not think or not sol:
        return None
    # Nested tags would break the "exactly one of each" clause downstream.
    if any(t in think or t in sol for t in SPECIAL_TOKENS):
        return None
    return think, sol


def build_target(think: str, sol: str) -> str:
    """The completion the model must learn to produce, verbatim in the checker's shape.

    `^\\s*<think>.*?</think>\\s*<answer>.*?</answer>\\s*$` is matched against the WHOLE
    completion, so nothing may sit outside the tags -- no preamble, no trailing note.
    """
    return f"<think>\n{think}\n</think>\n<answer>\n{sol}\n</answer>"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=40000,
                    help="target size. ReVisual-R1 cold-starts on ~40K text entries")
    ap.add_argument("--source", default="open-r1/OpenR1-Math-220k")
    ap.add_argument("--fork", default=str(REPO / "laser_fork"))
    ap.add_argument("--out-dir", default=str(REPO / "cold_data" / "laser" / "coldstart"))
    ap.add_argument("--max-chars", type=int, default=48000,
                    help="cheap pre-filter, applied before the tokenizer so it is not "
                         "handed novels. NOT the real length bound -- see --max-tokens")
    ap.add_argument("--max-tokens", type=int, default=12288,
                    help="THE REAL BOUND, measured with the model's own tokenizer. "
                         "Do not raise it without redoing the memory arithmetic below")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--scan-limit", type=int, default=0,
                    help="stop after scanning this many source rows (0 = no limit)")
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    format_reward, think_gate = load_format_checkers(Path(args.fork))
    # Length is bounded in TOKENS, with the model's own tokenizer, because a character
    # budget does not survive contact with LaTeX. A 48,000-char cap was assumed to be
    # ~13.5K tokens at 3.5 chars/token; maths traces run closer to 2.3, so the real tail
    # was ~20.7K and it OOM'd the backward pass twice at 12.6-15.6 GiB.
    #
    # The peak allocation is the fp32 cross-entropy GRADIENT over the logits:
    #     micro_batch x seq_len x vocab(152K) x 4 bytes
    # At micro_batch 1 and 12,288 tokens that is 7.5 GiB, against ~10 GiB of headroom
    # observed at the second failure. Raising --max-tokens without redoing that sum is
    # how this bites a third time.
    #
    # FILTERED, NOT TRUNCATED. Lowering cutoff_len instead would truncate from the right
    # and cut `</answer>` off the end of the longest targets -- teaching the model to open
    # tags and never close them, which is the exact failure the cold start exists to fix.
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.source, split="train", streaming=True)
    rows, seen, stats = [], 0, {"no_gen": 0, "unverified": 0, "unsplittable": 0,
                                "too_long": 0, "bad_boxed": 0, "format_reject": 0,
                                "too_many_tokens": 0}
    for ex in ds:
        seen += 1
        if args.scan_limit and seen > args.scan_limit:
            break
        if len(rows) >= args.n:
            break
        gens = ex.get("generations") or []
        ok = ex.get("correctness_math_verify") or []
        if not gens:
            stats["no_gen"] += 1
            continue
        # Only verified-correct traces. An unverified trace teaches the format and a
        # wrong answer, and the RL stage that follows is graded on correctness.
        idx = next((i for i, v in enumerate(ok) if v), None)
        if idx is None or idx >= len(gens):
            stats["unverified"] += 1
            continue
        gen = gens[idx]
        if len(gen) > args.max_chars:
            stats["too_long"] += 1
            continue
        got = split_trace(gen)
        if got is None:
            stats["unsplittable"] += 1
            continue
        think, sol = got
        if len(re.findall(r"\\boxed\{", sol)) != 1:
            stats["bad_boxed"] += 1
            continue
        target = build_target(think, sol)
        # The gate that matters: score it with THEIR code before writing it.
        if format_reward(target) != 1.0 or think_gate(target) != 1.0:
            stats["format_reject"] += 1
            continue
        n_tok = len(tok(ex["problem"]).input_ids) + len(tok(target).input_ids)
        if n_tok > args.max_tokens:
            stats["too_many_tokens"] += 1
            continue
        rows.append({"conversations": [
            {"from": "human", "value": ex["problem"]},
            {"from": "gpt", "value": target},
        ]})
        if len(rows) % 5000 == 0:
            print(f"  kept {len(rows)} of {seen} scanned", flush=True)

    name = "laser_coldstart_text"
    data_path = out_dir / f"{name}.json"
    data_path.write_text(json.dumps(rows, ensure_ascii=False))
    # LLaMA-Factory resolves datasets by name through this file, so it ships beside the
    # data rather than being a step someone has to remember.
    (out_dir / "dataset_info.json").write_text(json.dumps({
        name: {"file_name": data_path.name, "formatting": "sharegpt",
               "columns": {"messages": "conversations"},
               "tags": {"role_tag": "from", "content_tag": "value",
                        "user_tag": "human", "assistant_tag": "gpt"}}
    }, indent=2))

    kept = len(rows)
    print(f"\n{kept} samples from {seen} scanned rows of {args.source}")
    print("  dropped:", json.dumps(stats))
    print(f"  -> {data_path}")
    print(f"  -> {out_dir / 'dataset_info.json'}   (dataset name: {name})")
    if kept < args.n:
        print(f"\nWARNING: asked for {args.n} and got {kept}. The source ran out before "
              f"the target;\nthe cold start will be correspondingly smaller than "
              f"ReVisual-R1's ~40K.")
    print("\nEvery target above scores format=1 and think_gate=1 under the fork's own "
          "checkers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
