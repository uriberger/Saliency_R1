#!/usr/bin/env python
"""Grading a free-form answer against a short gold string, strictly and softly.

THE ARTEFACT THIS EXISTS FOR. `grpo_vlm_qwen3.accuracy_reward` parses the gold with
math_verify and, when that yields nothing -- which it does for `C`, for `horses`, for
`Yes` -- falls back to

    float(answer_text.lower() == solution.strip().lower())

an EXACT STRING MATCH. A model that answers `A` scores; the same model answering
`(C) water supply`, `\\boxed{A}` or `The cup stands on a shelf.` scores zero for being
verbose. That is the same failure as the LogicVista and MathVision parser artefacts
(`docs/` and the memory entries of both), and on a model that was never cold-started into
the terse format it is not an edge case: on the Nemotron-Omni it moved measured accuracy
from 0.167 to ~0.44, because the base checkpoint also echoes the SYSTEM PROMPT's own
illustration -- it literally writes "This is my answer." before answering.

Written for `fig1_multistep.py`, which needed it to keep a verbose BASELINE from looking
wrong for being verbose. It lives here because `head_correlation_probe` needs the identical
rule for the identical reason, and a second copy would be a second thing to fix: that probe
selects a saliency head on "does this head's attention predict getting it RIGHT", and a
label that is half grading noise attenuates every correlation it ranks on.

BOTH GRADES ARE ALWAYS CARRIED. `strict` is the trainer's rule and stays the primary
number, because it is what the reward actually optimises; `soft` is what a prose answer can
pass. Reporting only one of them is how this becomes a claim rather than a measurement.
"""

from __future__ import annotations

import re

# Lines a model ends on that are ABOUT its answer rather than the answer. Taking the last
# line literally scores "This is my answer." against the gold string and marks a correct
# model wrong -- which is the artefact to avoid, not commit.
_BOILERPLATE = re.compile(
    r"^\W*(this is my (answer|reasoning)|answer|final answer|in summary|conclusion)\W*$",
    re.IGNORECASE)


def extract_answer(text: str) -> tuple[str, str]:
    """-> (what to print, what to grade on).

    A cold-started chain is `<think> ... </think> ANSWER`, which is what the trainer's
    accuracy_reward parses, and there the two are the same string. Base Qwen3-VL-8B-
    Instruct has no think block at all (0/20 format_ok on val_natural), answers in prose
    and often signs off with a line about the answer rather than the answer; falling back
    to the whole completion -- the trainer's fallback -- would hand the grader four
    paragraphs, and falling back to the last line hands it the sign-off. So: drop the
    sign-off lines, print the last real one, and grade over the last two, which is where
    a "Therefore, ..." conclusion and its restatement both live.
    """
    text = text.replace("<|im_end|>", " ").strip()
    m = re.search(r"</think>\s*(.*)", text, re.DOTALL)
    if m and m.group(1).strip():
        return m.group(1).strip(), m.group(1).strip()
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(1).strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    real = [ln for ln in lines if not _BOILERPLATE.match(ln)]
    if not real:
        return (lines[-1] if lines else text), text
    return real[-1], " ".join(real[-2:])


def mcq_letter(text: str):
    """The option letter a free-form answer is choosing, or None.

    A bare `\\bA\\b` search is not it: prose contains the article "A", and on a
    five-option benchmark that alone would hand a wrong model a 1-in-5 credit. So the
    letter has to be in a position that means a choice -- the whole answer, a
    "the answer is X", or a parenthesised "(X)".
    """
    t = (text or "").strip()
    m = re.fullmatch(r"\W*([A-Ea-e])\W*", t)                  # the answer IS the letter
    if m:
        return m.group(1).upper()
    m = re.match(r"^\W*([A-E])\s*[.):,\-]", t)                # "C. In the upper left area"
    if m:
        return m.group(1)
    m = re.search(r"(?:answer|option|choice)\s*(?:is|are)?\s*[:\-]?\s*[*\(\[]*([A-E])\b(?!['\w])",
                  t, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.findall(r"[\(\[]([A-E])[\)\]]", t)
    return m[-1] if m else None


def grade(answer: str, gold: str, span: str | None = None) -> dict:
    """Strict is the trainer's rule; soft is the one a prose answer can pass.

    `answer` is the model's last real line, `span` the couple of lines around it. The
    two-line span is what a free-text gold needs (the conclusion and its restatement are
    often on different lines) and is exactly wrong for a multiple-choice gold: base
    answers a whole paragraph and then "B" on its own line, and searching the paragraph
    too finds no option letter at all and scores a correct answer wrong. So MCQ reads
    the last line first and only falls back to the span.

    The trainer scores `answer.lower() == gold.lower()`, which a model that writes "The
    cup stands on top of a bathroom vanity." fails even when it is right. Reporting only
    the strict grade would make a verbose baseline look wrong for being verbose -- the
    same artefact as the LogicVista MCQ parser -- so both are carried and the caption
    quotes the soft one.

    A single-letter gold is a multiple-choice benchmark, where neither rule works: exact
    match fails on "The best answer is: C" and the substring rule fires on the article
    "A". Both grades then come from `mcq_letter`.
    """
    span = answer if span is None else span
    a = (answer or "").strip().lower().rstrip(".")
    s = (span or "").strip().lower().rstrip(".")
    g = (gold or "").strip().rstrip(".")
    if not g:
        return {"strict": None, "soft": None, "kind": "none"}
    if re.fullmatch(r"[A-Ea-e]", g):
        got = mcq_letter(answer) or mcq_letter(span)
        ok = got is not None and got.upper() == g.upper()
        return {"strict": ok, "soft": ok, "kind": "mcq", "parsed": got}
    g = g.lower()
    soft = bool(re.search(rf"(?<![a-z0-9]){re.escape(g)}(?![a-z0-9])", s))
    return {"strict": a == g, "soft": soft, "kind": "text"}


def grade_completion(answer_text: str, gold: str) -> dict:
    """`extract_answer` then `grade`, for a caller holding the answer as one string.

    The probes store the model's own answer already separated from its chain, so the
    `</think>` split inside `extract_answer` is a no-op for them -- but going through it
    anyway is what keeps this one rule rather than two that drift.
    """
    ans, span = extract_answer(answer_text or "")
    return grade(ans, gold, span)
