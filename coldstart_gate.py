#!/usr/bin/env python
"""Did this cold start teach the format, and leave chains the overlap reward can score?

    python coldstart_gate.py --model checkpoint/coldstart_glm4v_..._merged
    python coldstart_gate.py --model <merged> --reference checkpoint/coldstart_qwen3_..._merged

WHY THIS EXISTS

The GLM and InternVL cold starts are 2,000 steps rather than the Qwen3-VL arm's 2 epochs,
because that arm reproduces Saliency-R1's published recipe and these two have no published
baseline to match (`train/cold_start/*/train.yaml` carries the loss-drop table that
justifies the cut). A shortened cold start is a bet, and the bet has exactly two ways to
fail -- both of which are INVISIBLE until days into GRPO, where they surface as
`think_overlap_reward = nan` with nothing saying why:

  1. **The format is not learned.** `grpo_trainer_qwen3.py` gates every reward on
     `^<think>\\s*(.+?)\\s*</think>\\s*(.+?)\\s*$`. A completion that fails it is scored as
     malformed and contributes nothing.
  2. **The chains have no observe steps.** The overlap reward scores per observe step, and
     the steps come from a FLAN-T5 classifier over the think block's sentences. A chain
     that is one terse sentence is perfectly well-formatted and yields zero steps, so the
     reward silently has nothing to attach to.
  3. **`<answer>` tags come back, and the FORMAT GATE DOES NOT CATCH THEM.** This recipe
     has no `<answer>` tags -- 0% of both cold-start corpora carry them -- but the regex
     above does not reject them either: `<answer> flower </answer>` matches the second
     group perfectly well. What breaks is downstream, in `accuracy_reward`, which takes
     everything after `</think>` and -- for the non-LaTeX gold that is most of this
     corpus -- falls back to `answer_text.lower() == solution.lower()`. "<answer> flower
     </answer>" never equals "flower", so accuracy is 0.0 on every row, forever, with no
     error anywhere.

     **This is the specific risk of shortening GLM-4.1V's cold start**, because
     `<think>...</think><answer>...</answer>` is that checkpoint's NATIVE format -- it was
     trained upstream to write it. Fewer cold-start steps is less pressure to unlearn it.
     So this gate counts the tags explicitly rather than trusting the format regex.

This measures all three, on the corpus GRPO actually trains on, through the same
classifier the reward loads. ~15 minutes on one GPU.

WHAT IT IS NOT: a quality metric. It does not check whether the answer is right -- that is
`accuracy_reward`'s job during training, and a cold start is not expected to be accurate.
It checks that the two MECHANISMS the reward depends on are present.

Sampled at temperature 1.0, not greedy, because that is what GRPO generates at and greedy
overstates format compliance.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))


def _load_module(name: str, relpath: str):
    """Import a leaf module by path, bypassing the `trl` package __init__.

    The idiom `overlap_probe.py` uses, and for the same two reasons: `sys.path` already
    has the repo root on it, so a plain `import trl.…` would resolve to the repo's own
    `trl/` directory and SHADOW the pip-installed trl the trainer runs from; and
    `trl/__init__.py` pulls in the whole reward stack to reach one leaf module.
    """
    spec = importlib.util.spec_from_file_location(name, REPO / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


#: The reward's own step classifier and sentence splitter, by path. `trl/overlap_steps.py`
#: is the tracked source; `patch_trl_qwen3.sh` copies it into the shared clone, so this is
#: the same code the training run loads.
STEPS = _load_module("_gate_overlap_steps", "trl/overlap_steps.py")

# The live pattern, copied from `grpo_trainer_qwen3.py`. `_check_pattern_still_matches`
# below reads it back out of that file and refuses to run if the two have drifted -- a
# gate scoring a format the trainer no longer wants is worse than no gate.
FORMAT_PATTERN = r"^<think>\s*([^\s].*?)\s*</think>\s*([^\s].*?)\s*$"

# The system prompt the GRPO stage sends, from `trl/grpo_vlm_qwen3.py`. Same deal: it is
# checked against that file rather than trusted, because a cold start measured under a
# different prompt is measuring a different model.
SYSTEM_PROMPT = (
    "A conversation between user and assistant. The user asks a question, and the assistant solves it. "
    "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think></think> tags, "
    "i.e., <think>\nThis is my reasoning.\n</think>\nThis is my answer."
)

#: Below these, do not go on to head selection -- re-run the cold start with a bigger
#: max_steps instead. Deliberately not tight: the question is "is the mechanism there",
#: not "is it as good as it can get". All three are overridable for a deliberate experiment.
MIN_FORMAT_OK = 0.95
MIN_MEDIAN_STEPS = 2
#: Any more than a trace of `<answer>` and `accuracy_reward` is dead -- see the module
#: docstring, failure 3. Not a soft target; the cold-start corpus is 0%.
MAX_ANSWER_TAGS = 0.02


def answer_text_of(completion):
    """What `accuracy_reward` will compare against the gold, character for character.

    Reproduces `grpo_vlm_qwen3.accuracy_reward`'s own extraction rather than
    approximating it, because the whole point is to see what that function will see.
    """
    m = re.search(r"</think>\s*(.*)", completion, re.DOTALL)
    return m.group(1).strip() if m else completion.strip()


def _check_pattern_still_matches():
    """Fail loudly if the trainer's format regex or system prompt has moved."""
    trainer = (REPO / "trl" / "grpo_trainer_qwen3.py").read_text()
    if FORMAT_PATTERN not in trainer:
        raise SystemExit(
            "coldstart_gate: the format pattern in this file is not in "
            "trl/grpo_trainer_qwen3.py any more. The trainer's gate moved; update "
            "FORMAT_PATTERN here to match it rather than scoring the old one.")
    script = (REPO / "trl" / "grpo_vlm_qwen3.py").read_text()
    # Compared on the distinctive tail, so reflowed string literals do not false-alarm.
    if "The reasoning process and answer are enclosed within <think></think> tags" not in script:
        raise SystemExit(
            "coldstart_gate: the system prompt in trl/grpo_vlm_qwen3.py no longer looks "
            "like the one this gate sends. Measuring under a different prompt measures a "
            "different model.")


def observe_steps(think_text, question, clf):
    """How many sentences of this chain the reward's own classifier calls `observe`.

    `segment_observe_steps` is the trainer's version and additionally maps each sentence
    to TOKEN spans, which needs the trainer's tokenisation object. The classification --
    which is the part being measured -- is identical: the same splitter, the same batched
    `predict_many`, the same chain and question context.
    """
    sentences = STEPS.split_sentences_with_spans(think_text)
    if not sentences:
        return 0, 0
    labels = clf.predict_many([s for s, _, _ in sentences], think_text, question)
    return sum(1 for lab in labels if lab == "observe"), len(sentences)


def run_one(model_path, rows, args, device):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    import vlm_family as VF

    print(f"\n{'=' * 78}\n{model_path}\n{'=' * 78}", flush=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa").to(device).eval()

    # The family is what makes this work on all three backbones unchanged: it supplies the
    # chat template wiring, the image token, and -- for InternVL -- the `crop_to_patches`
    # keyword without which the processor tiles to 13 tiles and the model sees a picture
    # 13x the size of the one it was trained on.
    fam = VF.family_for(model=model, processor=processor)
    fam.bind(model=model, processor=processor, config=model.config)
    fam.system_prompt = SYSTEM_PROMPT
    print(f"family={fam.name}  proc_defaults={fam.proc_defaults or '{}'}", flush=True)

    clf = STEPS.OverlapStepsClassifier.load(None, device=device)

    n_fmt, steps_per_chain, n_sent, failures = 0, [], [], []
    n_answer_tag, n_exact = 0, 0
    for i, row in enumerate(rows):
        inputs = fam.build_inputs(processor, [row["image"].convert("RGB")],
                                  row["problem"], device)
        with torch.no_grad():
            out = model.generate(**fam.generate_inputs(inputs),
                                 max_new_tokens=args.max_new_tokens,
                                 do_sample=True, temperature=1.0, top_p=1.0)
        comp = out[0][inputs["input_ids"].shape[1]:]
        text = processor.tokenizer.decode(comp, skip_special_tokens=True).strip()

        # What accuracy_reward will see, measured on EVERY completion -- including the
        # malformed ones, because that function has no format gate of its own.
        ans = answer_text_of(text)
        if "<answer>" in ans or "</answer>" in ans:
            n_answer_tag += 1
        if ans.lower() == str(row.get("solution", "")).strip().lower():
            n_exact += 1

        m = re.match(FORMAT_PATTERN, text, re.DOTALL | re.MULTILINE)
        one_each = text.count("<think>") == 1 and text.count("</think>") == 1
        if m and one_each:
            n_fmt += 1
            k, total = observe_steps(m.group(1), row["problem"], clf)
            steps_per_chain.append(k)
            n_sent.append(total)
        elif len(failures) < 3:
            failures.append(text[:300])
        if (i + 1) % 20 == 0:
            print(f"  ...{i + 1}/{len(rows)}  format {n_fmt}/{i + 1}", flush=True)

    fmt_rate = n_fmt / len(rows)
    tag_rate = n_answer_tag / len(rows)
    med = statistics.median(steps_per_chain) if steps_per_chain else 0
    res = {
        "model": str(model_path),
        "family": fam.name,
        "n": len(rows),
        "format_ok": round(fmt_rate, 4),
        "answer_tag_rate": round(tag_rate, 4),
        "exact_match": round(n_exact / len(rows), 4),
        "median_observe_steps": med,
        "mean_observe_steps": round(statistics.mean(steps_per_chain), 2) if steps_per_chain else 0.0,
        "chains_with_zero_steps": sum(1 for k in steps_per_chain if k == 0),
        "median_sentences_in_chain": statistics.median(n_sent) if n_sent else 0,
    }
    print(f"\n  format_ok               {fmt_rate:.1%}   (need >= {MIN_FORMAT_OK:.0%})")
    print(f"  <answer> tags in answer {tag_rate:.1%}   (need <= {MAX_ANSWER_TAGS:.0%}"
          " -- these pass the format gate and zero out accuracy_reward)")
    print(f"  median observe steps    {med}     (need >= {MIN_MEDIAN_STEPS})")
    print(f"  mean observe steps      {res['mean_observe_steps']}")
    print(f"  chains with ZERO steps  {res['chains_with_zero_steps']}/{len(steps_per_chain)}"
          "   <- these are scored as nothing by the overlap reward")
    print(f"  median sentences/chain  {res['median_sentences_in_chain']}")
    print(f"  exact-match accuracy    {res['exact_match']:.1%}   (context only, not a "
          "gate: a cold start is not expected to be accurate, but 0.0% means the "
          "accuracy signal is dead)")
    if failures:
        print("\n  first format failures, verbatim -- a pass/fail number cannot tell "
              "'ignored the tags' from 'emitted them twice':")
        for f in failures:
            print(f"    {f!r}")
    del model
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="a MERGED cold-start checkpoint")
    ap.add_argument("--reference", default=None,
                    help="a second model to measure alongside, as the bar. The merged "
                         "Qwen3-VL cold start is the one with a known-good GRPO run "
                         "behind it, so its numbers are what 'good enough' means.")
    ap.add_argument("--dataset", default="cold_data/grpo_sets/val_natural",
                    help="held out from training, and the distribution GRPO trains on")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--out", default=None, help="write the result dicts here as JSON")
    ap.add_argument("--min-format", type=float, default=MIN_FORMAT_OK)
    ap.add_argument("--min-steps", type=float, default=MIN_MEDIAN_STEPS)
    ap.add_argument("--max-answer-tags", type=float, default=MAX_ANSWER_TAGS)
    args = ap.parse_args()

    _check_pattern_still_matches()

    import torch
    from datasets import load_from_disk

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU visible. This will be very slow.", file=sys.stderr)
    ds = load_from_disk(args.dataset)
    rows = [ds[i] for i in range(min(args.n, len(ds)))]
    print(f"{len(rows)} prompts from {args.dataset}, sampled at temperature 1.0")

    results = [run_one(args.model, rows, args, device)]
    if args.reference:
        results.append(run_one(args.reference, rows, args, device))

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.out}")

    got = results[0]
    reasons = []
    if got["format_ok"] < args.min_format:
        reasons.append(f"format_ok {got['format_ok']:.1%} < {args.min_format:.0%}")
    if got["median_observe_steps"] < args.min_steps:
        reasons.append(f"median observe steps {got['median_observe_steps']} "
                       f"< {args.min_steps}")
    if got["answer_tag_rate"] > args.max_answer_tags:
        reasons.append(f"<answer> tags on {got['answer_tag_rate']:.1%} of completions "
                       f"> {args.max_answer_tags:.0%} -- accuracy_reward will score "
                       "those 0.0 and say nothing")
    print(f"\n{'PASS' if not reasons else 'FAIL'}: {got['model']}")
    for r in reasons:
        print(f"  - {r}")
    if reasons:
        print("\n  Do not go on to head selection. A weak cold start does not announce\n"
              "  itself in GRPO -- it looks like a reward that does not work.\n"
              "  Format or steps too low: re-run with a larger max_steps (4000 is +1.8 h).\n"
              "  <answer> tags: that is the model reverting to its own native format, most\n"
              "  likely on GLM-4.1V. More cold-start steps is the fix there too.")
    return 0 if not reasons else 1


if __name__ == "__main__":
    raise SystemExit(main())
