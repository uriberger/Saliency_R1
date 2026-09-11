#!/usr/bin/env python
"""What does the model actually write, and where does LASER's format checker reject it?

    python laser_prompt_probe.py --model Qwen/Qwen3-VL-8B-Instruct --n-prompts 6 --n-gen 4

The smoke run (job 6717917) completed three clean steps with `critic/score/max` exactly
1.0 -- accuracy 1 with format 0 -- so every rollout failed LASER's format gate and the
attention rewards never fired. The gate is the one part of this build that is OURS:
`build_laser_data.py:SYSTEM_PROMPT` is written against their checkers because they never
published a prompt.

This prints completions verbatim and scores each of the checker's clauses SEPARATELY, so
the fix is a reading rather than a guess. `compute_score_vanilla` returns one number for
five different ways to fail:

    tags        exactly one each of <think> </think> <answer> </answer>
    structure   the WHOLE completion matches ^\\s*<think>.*?</think>\\s*<answer>.*?</answer>\\s*$
    boxed       exactly one \\boxed{} inside <answer>
    think_gate  `_format_reward_think` -- laxer, and what gates the ATTENTION rewards
    accuracy    r1v_accuracy_reward against the ground truth

`think_gate` is reported apart from `format` on purpose: it is a *different* function in a
*different* file (dp_actor, not openr1_verl) with different rules, and a prompt could
satisfy one and not the other. That would show up as an attention reward that fires while
the format reward stays zero, which is a confusing thing to debug from a training curve.

Runs the fork's OWN reward code -- imported from laser_fork, not reimplemented -- so the
verdict here is the verdict the trainer would reach.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent


def load_upstream_rewards(fork: Path):
    """`openr1_verl` from the fork, so this scores with their code and not a copy."""
    p = fork / "verl" / "utils" / "reward_score" / "openr1_verl.py"
    spec = importlib.util.spec_from_file_location("_laser_openr1", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_laser_openr1"] = mod
    spec.loader.exec_module(mod)
    return mod


def think_gate(completion: str) -> float:
    """`DataParallelPPOActor._format_reward_think`, transcribed -- it gates R_vis/R_supp.

    Copied rather than imported because dp_actor pulls in the whole FSDP/ray stack at
    import time. `test_laser_capture_cpu.py` is where transcriptions get checked; this one
    is four lines and is reproduced verbatim from dp_actor.py:930.
    """
    t = re.findall(r"<think>.*?</think>", completion, re.MULTILINE | re.DOTALL)
    a = re.findall(r"<answer>.*?</answer>(\s*)$", completion, re.MULTILINE | re.DOTALL)
    return 1.0 if (len(t) == 1 and len(a) == 1) else 0.0


def clause_report(completion: str):
    """Every clause of their format checker, scored on its own."""
    n = {k: len(re.findall(re.escape(f"<{k}>"), completion)) for k in ("think", "answer")}
    nc = {k: len(re.findall(re.escape(f"</{k}>"), completion)) for k in ("think", "answer")}
    structure = bool(re.search(r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$",
                               completion, re.DOTALL))
    m = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
    boxed = len(re.findall(r"\\boxed\{", m.group(1))) if m else 0
    return {
        "tags_ok": all(n[k] == 1 and nc[k] == 1 for k in ("think", "answer")),
        "counts": {f"<{k}>": n[k] for k in n} | {f"</{k}>": nc[k] for k in nc},
        "structure_ok": structure,
        "boxed_in_answer": boxed,
        "think_gate": think_gate(completion),
    }


#: The instruction, in the three places a chat model can be told to follow a format.
#: Round 1 (job 6722516) established that the SYSTEM variant scores format 0.000 with
#: tags failing 16/16 -- the model emits no tags at all -- while the prompt demonstrably
#: renders into the chat. So the instruction reaches the model and the model ignores it,
#: which is a placement problem rather than a wording one. Every R1-style multimodal RL
#: recipe puts it in the USER turn for exactly this reason; these variants measure that
#: rather than assume it.
_RULE = (
    "Think step by step inside <think> </think>, then give the final answer inside "
    "<answer> </answer> with the answer in \\boxed{}. Respond with exactly this and "
    "nothing else:\n<think> your reasoning </think><answer> \\boxed{your answer} </answer>"
)

PROMPT_VARIANTS = {
    # the one in the corpus today, and the one that scored 0.000
    "system_only": dict(system=None, suffix=None),
    # the instruction moved into the user turn, system dropped
    "user_only": dict(system="", suffix="\n\n" + _RULE),
    # both: system sets the persona, the user turn carries the hard requirement
    "system_and_user": dict(system=None, suffix="\n\n" + _RULE),
    # a terser user-turn rule, to check the long one is not itself the problem
    "user_terse": dict(system="", suffix="\n\nAnswer in the format: <think> reasoning "
                                         "</think><answer> \\boxed{answer} </answer>"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--parquet",
                    default=str(REPO / "cold_data/laser/verl/train_smoke64.parquet"))
    ap.add_argument("--fork", default=str(REPO / "laser_fork"))
    ap.add_argument("--n-prompts", type=int, default=6)
    ap.add_argument("--n-gen", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--show-chars", type=int, default=900)
    ap.add_argument("--system-prompt", default=None,
                    help="override the prompt baked into the parquet, to test a fix "
                         "WITHOUT rebuilding 1.7 GB of data first. '' drops it entirely")
    ap.add_argument("--user-suffix", default=None,
                    help="text appended to the user turn, after the question")
    ap.add_argument("--variants", action="store_true",
                    help="sweep PROMPT_VARIANTS in one job and rank them by format rate")
    args = ap.parse_args()

    import pyarrow.parquet as pq
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor

    rew = load_upstream_rewards(Path(args.fork))
    rows = pq.read_table(args.parquet).slice(0, args.n_prompts).to_pylist()

    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda").eval()

    variants = (PROMPT_VARIANTS if args.variants
                else {"cli": dict(system=args.system_prompt, suffix=args.user_suffix)})
    summary = {}
    for v_name, v in variants.items():
        summary[v_name] = run_variant(args, rows, rew, proc, model, v_name,
                                      v["system"], v["suffix"])
    if len(summary) > 1:
        print("\n" + "=" * 78)
        print("VARIANT RANKING -- format rate is what gates BOTH the accuracy-weighted")
        print("attention terms and _format_reward_think, so it is the number to maximise.")
        print(f"  {'variant':<18} {'format':>7} {'think_gate':>11} {'accuracy':>9}")
        for k, s in sorted(summary.items(), key=lambda kv: -kv[1]["format"]):
            print(f"  {k:<18} {s['format']:>7.3f} {s['think_gate']:>11.3f} "
                  f"{s['accuracy']:>9.3f}")
        print(json.dumps(summary, indent=2))
    return 0


def run_variant(args, rows, rew, proc, model, name, system, suffix):
    import io
    import re as _re

    import torch
    from PIL import Image

    print("\n" + "#" * 78)
    print(f"# VARIANT {name}   system={'(from parquet)' if system is None else repr(system[:60])}"
          f"   suffix={repr(suffix[:60]) if suffix else None}")
    print("#" * 78)
    tally = {"format": 0, "think_gate": 0, "acc": 0, "n": 0}
    clause_fail = {"tags": 0, "structure": 0, "boxed": 0}
    for r_i, row in enumerate(rows):
        msgs = [dict(m) for m in row["prompt"]]
        if system is not None:
            msgs = [m for m in msgs if m["role"] != "system"]
            if system:
                msgs.insert(0, {"role": "system", "content": system})
        if suffix:
            for m in msgs:
                if m["role"] == "user":
                    m["content"] = m["content"] + suffix
                    break
        # verl's own _build_messages contract: content is a STRING with <image> in it,
        # split into parts here exactly as RLHFDataset does.
        chat = []
        for m in msgs:
            parts = [s for s in _re.split("(<image>|<video>)", m["content"]) if s]
            chat.append({"role": m["role"], "content": [
                {"type": "image"} if s == "<image>" else {"type": "text", "text": s}
                for s in parts]})
        images = [Image.open(io.BytesIO(im["bytes"])).convert("RGB") for im in row["images"]]
        text = proc.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=[images], return_tensors="pt",
                      add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, do_sample=True, temperature=args.temperature,
                                 top_p=1.0, top_k=0, num_return_sequences=args.n_gen,
                                 max_new_tokens=args.max_new_tokens,
                                 pad_token_id=proc.tokenizer.pad_token_id)
        gt = row["reward_model"]["ground_truth"]
        comps = [proc.tokenizer.decode(o[inputs["input_ids"].shape[1]:],
                                       skip_special_tokens=True) for o in out]
        print("=" * 78)
        print(f"PROMPT {r_i}  [{row['data_source']}]  ground_truth={gt!r}")
        for g, c in enumerate(comps):
            res = rew.compute_score_vanilla(row["data_source"], c, gt, {})
            cl = clause_report(c)
            tally["n"] += 1
            tally["format"] += res["format"]
            tally["acc"] += res["accuracy"]
            tally["think_gate"] += cl["think_gate"]
            if not cl["tags_ok"]:
                clause_fail["tags"] += 1
            elif not cl["structure_ok"]:
                clause_fail["structure"] += 1
            elif cl["boxed_in_answer"] != 1:
                clause_fail["boxed"] += 1
            print(f"\n--- rollout {g}   acc={res['accuracy']:.0f} format={res['format']:.0f} "
                  f"think_gate={cl['think_gate']:.0f} boxed_in_answer={cl['boxed_in_answer']} "
                  f"tags={cl['counts']} structure={cl['structure_ok']}")
            print(repr(c[: args.show_chars]))
            if len(c) > args.show_chars:
                print(f"   ... [{len(c) - args.show_chars} more chars] tail: "
                      f"{c[-160:]!r}")

    n = max(1, tally["n"])
    print("\n" + "=" * 78)
    print(f"{tally['n']} rollouts:  format {tally['format'] / n:.3f}   "
          f"think_gate {tally['think_gate'] / n:.3f}   accuracy {tally['acc'] / n:.3f}")
    print("FIRST failing clause, counted in the checker's own order:")
    for k, v in clause_fail.items():
        print(f"   {k:<10} {v:>4}  ({v / n:.3f})")
    print("\n  tags       -> the prompt is not getting the tags emitted at all")
    print("  structure  -> tags are there but something sits outside them (preamble,")
    print("                markdown fence, trailing commentary)")
    print("  boxed      -> tags and structure fine; \\boxed{} missing or duplicated")
    return {"format": tally["format"] / n, "think_gate": tally["think_gate"] / n,
            "accuracy": tally["acc"] / n, "first_fail": clause_fail, "n": tally["n"]}


if __name__ == "__main__":
    raise SystemExit(main())
