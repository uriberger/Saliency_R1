#!/usr/bin/env python
"""CPU gates for fig1_multistep.py -- everything except the detector call.

The expensive half of that script is one Grounding-DINO pass; the half that can be wrong
without anyone noticing is the rest: which boxes become the tight referent, whether the
crossover margin has the sign it claims, and what counts as the model's answer. All three
are pure functions of already-computed arrays, so they are gated here rather than
discovered on a GPU node twenty minutes in.

    python test_fig1_multistep_cpu.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("_fig1ms", REPO / "fig1_multistep.py")
M = importlib.util.module_from_spec(_spec)
sys.modules["_fig1ms"] = M
_spec.loader.exec_module(M)

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAILED.append(name)


def test_raster():
    print("raster")
    m = M.raster([[0.0, 0.0, 0.25, 0.25]], 8, 8)
    check("a quarter-width box claims a quarter of each axis", m.sum() == 4, f"{m.sum()} patches")
    # A box thinner than one patch still has to claim one, or a small object would be
    # scored against an empty mask -- the reward's rasterisation does the same.
    thin = M.raster([[0.5, 0.5, 0.501, 0.501]], 8, 8)
    check("a sub-patch box still claims one patch", thin is not None and thin.sum() == 1)
    check("no boxes -> None", M.raster([], 8, 8) is None)
    check("whole image -> None (degenerate, like the reward)",
          M.raster([[0, 0, 1, 1]], 8, 8) is None)


def test_tight_referent():
    print("tight_referent")
    dets = [{"box": [0, 0, 0.2, 0.2], "score": 0.90, "label": "cup"},
            {"box": [0, 0, 0.22, 0.22], "score": 0.75, "label": "rim"},
            {"box": [0.7, 0.7, 0.95, 0.95], "score": 0.40, "label": "shelf"}]
    mask, boxes = M.tight_referent(dets, 10, 10, 0.8, 0.35)
    check("keeps the top box and its near-tie, drops the tail", len(boxes) == 2,
          f"kept {len(boxes)} of 3")
    check("the tight mask is one object", mask.mean() <= 0.10, f"area {mask.mean():.2f}")
    rmask, rboxes = M.reward_referent(dets, 10, 10, 0.5)
    check("the reward referent keeps everything above threshold", len(rboxes) == 3)
    check("and is larger than the tight one", rmask.sum() > mask.sum(),
          f"{rmask.mean():.2f} vs {mask.mean():.2f}")
    # The per-box area cap is what stops a whole-image detection from becoming "the object".
    big = [{"box": [0.0, 0.0, 0.95, 0.95], "score": 0.9, "label": "scene"},
           {"box": [0.1, 0.1, 0.2, 0.2], "score": 0.5, "label": "cup"}]
    mask2, boxes2 = M.tight_referent(big, 10, 10, 0.8, 0.35)
    check("a box over the area cap cannot be the referent",
          len(boxes2) == 1 and boxes2[0][2] == 0.2)
    check("nothing grounded -> None", M.tight_referent([], 10, 10, 0.8, 0.35)[0] is None)


def test_crossover_sign():
    print("pair_margin")
    gh = gw = 12
    mi = M.raster([[0.0, 0.0, 0.3, 0.3]], gh, gw)
    mj = M.raster([[0.7, 0.7, 1.0, 1.0]], gh, gw)
    hot_i = np.full((gh, gw), 0.01); hot_i[:4, :4] = 1.0
    hot_j = np.full((gh, gw), 0.01); hot_j[8:, 8:] = 1.0
    good = M.pair_margin({"maps": {"g": hot_i}}, {"maps": {"g": hot_j}}, mi, mj, "g")
    check("each step on its own region -> positive margin", good["margin"] > 0,
          f"{good['margin']:+.2f}")
    swapped = M.pair_margin({"maps": {"g": hot_j}}, {"maps": {"g": hot_i}}, mi, mj, "g")
    check("swap the maps -> negative margin", swapped["margin"] < 0,
          f"{swapped['margin']:+.2f}")
    # One step doing the right thing must not carry the pair: that is the whole reason
    # the margin is a min and not a mean.
    flat = np.full((gh, gw), 0.5)
    half = M.pair_margin({"maps": {"g": hot_i}}, {"maps": {"g": flat}}, mi, mj, "g")
    check("one good step and one flat step -> margin ~0, not half of the good one",
          abs(half["margin"]) < 1e-9, f"{half['margin']:+.3f}")
    check("a missing map -> None", M.pair_margin({"maps": {}}, {"maps": {}}, mi, mj, "g") is None)
    zero = M.pair_margin({"maps": {"g": np.zeros((gh, gw))}},
                         {"maps": {"g": hot_j}}, mi, mj, "g")
    check("an all-zero map -> None rather than a fake 0.0", zero is None)


def test_answers():
    print("extract_answer")
    shown, span = M.extract_answer("<think> reasoning </think> Counter <|im_end|>")
    check("cold-start chain -> what follows </think>", (shown, span) == ("Counter", "Counter"))
    shown, span = M.extract_answer("Step 1 ...\n\nThe cup stands on a shelf.<|im_end|>")
    check("base prose -> its last line, not four paragraphs",
          shown == "The cup stands on a shelf.")
    # The sign-off is the failure the pilot actually hit: base ends on "This is my
    # answer." and the last line is then a sentence ABOUT the answer.
    shown, span = M.extract_answer(
        "Looking at the image...\n\nTherefore, the man in the foreground wears them.\n\n"
        "This is my answer.<|im_end|>")
    check("a sign-off line is not the answer",
          shown == "Therefore, the man in the foreground wears them.", shown)
    check("and the graded span still reaches back past it", "man" in span)
    shown, _ = M.extract_answer("blah <answer>goat</answer> blah")
    check("<answer> tags are honoured when there is no think block", shown == "goat")

    print("grade -- free text")
    check("strict is the trainer's rule", M.grade("Counter", "counter")["strict"] is True)
    g = M.grade("The cup stands on a shelf", "shelf")
    check("a prose answer fails strict and passes soft",
          (g["strict"], g["soft"], g["kind"]) == (False, True, "text"))
    # Word boundaries, or "shelf" would match "shelves" and every plural would read right.
    check("soft is word-bounded, not a substring test",
          M.grade("shelves", "shelf")["soft"] is False)
    check("soft does not fire on a longer word",
          M.grade("the goatherd is here", "goat")["soft"] is False)

    print("grade -- multiple choice")
    check("a bare letter answer", M.grade("C", "C")["soft"] is True)
    check("the benchmark's own phrasing",
          M.grade("The best answer is: C", "C")["soft"] is True)
    check("a parenthesised choice", M.grade("I would pick (B) here.", "B")["soft"] is True)
    check("a wrong letter is wrong", M.grade("The best answer is: D", "C")["soft"] is False)
    # The whole reason mcq_letter exists: the article "A" must not score.
    check("the article 'a' does not count as choosing A",
          M.grade("A man is standing on the left.", "A")["soft"] is False)
    check("an unparseable answer is wrong, not None",
          M.grade("I cannot tell from this image.", "A")
          == {"strict": False, "soft": False, "kind": "mcq", "parsed": None})
    # The one that was actually wrong: base reasons for a paragraph and then puts "B" on
    # its own line. Graded over the two-line span, the paragraph swamps the letter and a
    # correct answer reads wrong -- which is the direction this must never fail in.
    para = ("The bike is tilted and the rider is losing balance, so it will crash.")
    check("MCQ reads the last line, not the paragraph before it",
          M.grade("B", "B", span=f"{para} B")["soft"] is True)
    check("and still finds a letter when the last line is prose",
          M.grade(para, "B", span=f"the answer is B. {para}")["soft"] is True)
    # Free text is the opposite: the span is what a two-line conclusion needs.
    check("free text still grades over the span",
          M.grade("It is underneath it.", "shelf",
                  span="The cup is on a shelf. It is underneath it.")["soft"] is True)


def main():
    for t in (test_raster, test_tight_referent, test_crossover_sign, test_answers):
        t()
    print()
    if FAILED:
        raise SystemExit(f"FAILED: {', '.join(FAILED)}")
    print("all CPU gates pass")


if __name__ == "__main__":
    main()
