#!/usr/bin/env python
"""The cold-start gate's judgement, without a GPU or a cold start.

`coldstart_gate.py` decides whether a 2,000-step cold start is good enough to go on to
head selection. Everything in it except the generate() call is a pure function of a string,
and those are what this pins -- because the gate is the only thing standing between a
shortened cold start and discovering the problem a week later inside GRPO.

THE CASE THAT MATTERS MOST is `<think> a </think> <answer> flower </answer>`. It PASSES
the trainer's format regex -- the second capture group matches the tags happily -- and it
is nonetheless fatal, because `accuracy_reward` string-compares everything after
`</think>` against the gold and "<answer> flower </answer>" is never "flower". GLM-4.1V
writes that format natively, so it is the live risk of shortening its cold start. An
earlier draft of the gate's own docstring claimed the regex rejected it; running these
cases is what showed it does not.

    python test_coldstart_gate_cpu.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import coldstart_gate as G  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{('   ' + detail) if detail else ''}")


def format_ok(text):
    """The gate's own test, restated: the regex AND exactly one of each tag."""
    m = re.match(G.FORMAT_PATTERN, text, re.DOTALL | re.MULTILINE)
    return bool(m) and text.count("<think>") == 1 and text.count("</think>") == 1


def test_the_format_gate():
    print("\nThe format gate: what the trainer accepts, and what it quietly does not")
    good = "<think> I look at the bush. There is a flower on it. </think> flower"
    check("the recipe's own shape passes", format_ok(good))
    check("no tags at all fails", not format_ok("The answer is flower."))
    check("two think blocks fail",
          not format_ok("<think> a </think> <think> b </think> flower"))
    check("an empty answer fails", not format_ok("<think> a </think>"))
    check("a missing closing tag fails", not format_ok("<think> a flower"))
    # Multi-line chains are the normal case, and `re.match` without DOTALL would stop at
    # the first newline -- the gate passes both flags, and this is what says so.
    check("a multi-line chain passes",
          format_ok("<think>\nFirst I look.\nThen I see it.\n</think>\nflower"))


def test_answer_tags_pass_the_regex_and_still_break_accuracy():
    print("\nThe trap: <answer> tags are FORMAT-legal and ACCURACY-fatal")
    tagged = "<think> I see a flower. </think> <answer> flower </answer>"
    check("it passes the format gate", format_ok(tagged),
          "which is why the gate counts the tags separately")
    ans = G.answer_text_of(tagged)
    check("answer_text_of extracts what accuracy_reward sees",
          ans == "<answer> flower </answer>", repr(ans))
    check("and that never equals the gold", ans.lower() != "flower")
    # The same chain without the tags is what the cold start is supposed to teach.
    plain = "<think> I see a flower. </think> flower"
    check("the untagged form does equal the gold",
          G.answer_text_of(plain).lower() == "flower")
    check("a completion with no </think> falls back to the whole string",
          G.answer_text_of("flower") == "flower")
    check("the threshold is near zero, not lenient", G.MAX_ANSWER_TAGS <= 0.02)


def test_the_drift_guard():
    print("\nThe drift guard: the gate refuses to score a format the trainer abandoned")
    G._check_pattern_still_matches()
    check("the pattern and the system prompt are both still in the trainer", True)
    trainer = (REPO / "trl" / "grpo_trainer_qwen3.py").read_text()
    check("the pattern really is the trainer's live one, not a lookalike",
          trainer.count(G.FORMAT_PATTERN) >= 1)
    # The guard has to FIRE, not just pass when things are fine.
    saved = G.FORMAT_PATTERN
    try:
        G.FORMAT_PATTERN = r"^<thonk>(.*)$"
        fired = False
        try:
            G._check_pattern_still_matches()
        except SystemExit:
            fired = True
        check("a drifted pattern raises SystemExit", fired)
    finally:
        G.FORMAT_PATTERN = saved


def test_thresholds_are_the_ones_documented():
    print("\nThe thresholds")
    check("format >= 95%", G.MIN_FORMAT_OK == 0.95)
    check("median observe steps >= 2", G.MIN_MEDIAN_STEPS == 2)
    check("the system prompt matches the training script's, character for character",
          "The reasoning process and answer are enclosed within <think></think> tags"
          in G.SYSTEM_PROMPT)


def test_observe_steps_against_the_real_classifier():
    print("\nObserve steps, through the classifier the reward itself loads")
    try:
        clf = G.STEPS.OverlapStepsClassifier.load(None, device="cpu")
    except Exception as exc:  # no checkpoint on this machine
        print(f"  skip  classifier unavailable ({type(exc).__name__})")
        return
    chain = ("I need to find what is on the bush. Looking at the image, there is a bush "
             "in the lower left of the frame. On top of the bush I can see a small pink "
             "flower. The flower appears to be a rose. Therefore the answer is a flower.")
    k, total = G.observe_steps(chain, "What's on the bush?", clf)
    check("a five-sentence chain splits into five", total == 5, f"got {total}")
    check("and some of them are observe steps", k >= 1, f"got {k}")
    check("a one-sentence chain is the zero-step case the gate exists to catch",
          G.observe_steps("The answer is flower.", "What's on the bush?", clf)[0] == 0)
    check("an empty chain yields nothing rather than raising",
          G.observe_steps("", "q", clf) == (0, 0))


def main():
    test_the_format_gate()
    test_answer_tags_pass_the_regex_and_still_break_accuracy()
    test_the_drift_guard()
    test_thresholds_are_the_ones_documented()
    test_observe_steps_against_the_real_classifier()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for n in FAIL:
            print(f"  FAILED: {n}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
