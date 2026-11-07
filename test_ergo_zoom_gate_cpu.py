#!/usr/bin/env python
"""The ERGO zoom gate's judgement, without a GPU and without a checkpoint.

`ergo_zoom_gate.py` decides two things that a GRPO run cannot tell you: whether turn 1
will ever produce a box, and which coordinate frame that box is written in. Everything in
it except the two `generate()` calls is a pure function of a string and four numbers, and
those are what this pins.

THE CASE THAT MATTERS MOST is `test_a_wrong_frame_still_looks_healthy`. A box read in the
wrong frame is not an error and does not raise: it is a perfectly valid rectangle,
somewhere else in the picture. `map_box` returns it, `crop_with_zoom`'s equivalent crops
it, turn 2 runs, `r_box` (crop area <= gamma) fires, and only `r_region` -- the reward
model shown the crop alone -- quietly stops firing. That is the failure the frame
measurement exists to prevent, and it has no symptom. The test asserts the measurement can
separate the frames at all on V*-shaped geometry, because a gate that cannot is worse than
no gate: it returns a verdict either way.

    python test_ergo_zoom_gate_cpu.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import ergo_zoom_gate as G  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{('   ' + detail) if detail else ''}")


# ---------------------------------------------------------------------------
def test_strict_parser_is_ergos():
    print("\nThe strict parser: what ERGO's reward will and will not accept")
    check("the canonical shape parses",
          G.parse_zoom_strict("<think>a</think><zoom>[10, 20, 30, 40]</zoom>")
          == [10, 20, 30, 40])
    check("no tags -> None", G.parse_zoom_strict("the box is [10, 20, 30, 40]") is None)
    check("FLOATS are rejected by ERGO's own regex -- this is why the lenient rate "
          "is reported next to it",
          G.parse_zoom_strict("<zoom>[10.0, 20.0, 30.0, 40.0]</zoom>") is None)
    check("two boxes inside one tag -> None (their rule, not ours)",
          G.parse_zoom_strict("<zoom>[1, 2, 3, 4] and [5, 6, 7, 8]</zoom>") is None)
    check("a box wrapped in prose inside the tag still parses",
          G.parse_zoom_strict("<zoom>the region [1, 2, 3, 4]</zoom>") == [1, 2, 3, 4])


def test_lenient_parser_separates_the_near_misses():
    print("\nThe lenient parser: telling 'no box' from 'a box the regex would not take'")
    check("floats are caught",
          G.parse_zoom_lenient("<zoom>[10.5, 20, 30, 40]</zoom>") == [10.5, 20, 30, 40])
    check("a bare box outside any tag is caught",
          G.parse_zoom_lenient("the region is [1, 2, 3, 4].") == [1.0, 2.0, 3.0, 4.0])
    check("<zoom> wins over a box written earlier in the chain",
          G.parse_zoom_lenient("maybe [9, 9, 9, 9] <zoom>[1, 2, 3, 4]</zoom>")
          == [1.0, 2.0, 3.0, 4.0])
    check("prose with no numbers -> None",
          G.parse_zoom_lenient("I should zoom in on the glove.") is None)


# ---------------------------------------------------------------------------
def test_turn_format_checks():
    print("\nThe two turns' format gates")
    check("turn 1, the shape ERGO asks for",
          G.turn1_flags("<think>look left</think><zoom>[1, 2, 3, 4]</zoom>")["format_ok"])
    check("turn 1 with a box but no think block fails",
          not G.turn1_flags("<zoom>[1, 2, 3, 4]</zoom>")["format_ok"])
    check("turn 1 with a think block but no box fails",
          not G.turn1_flags("<think>look left</think> the glove is rubber")["format_ok"])
    check("turn 1 with two think blocks fails",
          not G.turn1_flags("<think>a</think><think>b</think><zoom>[1,2,3,4]</zoom>")
          ["format_ok"])

    good2 = "<think>the glove is black</think><answer>A</answer>"
    check("turn 2, the shape ERGO asks for", G.turn2_flags(good2)["strict"])
    # Their QUESTION_TEMPLATE2 literally says "<think> ... <think>" -- the closing tag is
    # not a closing tag. A model that obeys it writes something the strict reading
    # rejects, which is why the lenient rate exists and why both are reported.
    typo = "<think>the glove is black<think><answer>A</answer>"
    check("ERGO's own typo'd instruction fails strict but passes lenient",
          (not G.turn2_flags(typo)["strict"]) and G.turn2_flags(typo)["lenient"])
    check("an answer with no tags fails both",
          not G.turn2_flags("The answer is A.")["lenient"])
    check("the letter is read out of <answer>", G.turn2_pred(good2) == "A")
    check("a letter in the chain does not leak into the prediction",
          G.turn2_pred("<think>not B</think><answer>C</answer>") == "C")


# ---------------------------------------------------------------------------
def test_map_box_geometry():
    print("\nmap_box: one emitted box, three readings")
    W, H, rw, rh = 2000, 1500, 1312, 992
    box = [500, 250, 700, 450]
    res = G.map_box(box, "resized", W, H, rw, rh)
    rel = G.map_box(box, "rel1000", W, H, rw, rh)
    org = G.map_box(box, "original", W, H, rw, rh)
    check("the three readings are three different rectangles",
          len({res, rel, org}) == 3, f"{res} {rel} {org}")
    check("'original' is the identity", org == (500.0, 250.0, 700.0, 450.0))
    check("'rel1000' scales by W/1000, H/1000",
          abs(rel[0] - 1000.0) < 1e-6 and abs(rel[1] - 375.0) < 1e-6)
    check("'resized' scales by W/rw, H/rh",
          abs(res[0] - 500 * W / rw) < 1e-6 and abs(res[1] - 250 * H / rh) < 1e-6)
    check("everything is clipped into the picture",
          G.map_box([0, 0, 99999, 99999], "original", W, H, rw, rh) == (0.0, 0.0, 2000.0, 1500.0))
    check("a degenerate box is None, not a zero-area crop",
          G.map_box([10, 10, 10, 10], "original", W, H, rw, rh) is None)
    check("reversed corners are repaired rather than charged to the frame",
          G.map_box([700, 450, 500, 250], "original", W, H, rw, rh)
          == (500.0, 250.0, 700.0, 450.0))


def test_iou_and_coverage():
    print("\nIoU, coverage and the reason both are reported")
    a = (0.0, 0.0, 10.0, 10.0)
    check("identical boxes are IoU 1", abs(G.iou(a, a) - 1.0) < 1e-9)
    check("disjoint boxes are IoU 0", G.iou(a, (20.0, 20.0, 30.0, 30.0)) == 0.0)
    # The V* situation: a crop that CONTAINS the whole answer scores an IoU near zero
    # purely because the target is tiny. This is the confound the centre statistics exist
    # to go around, and it is a fact about the benchmark, not about the frame.
    crop = (0.0, 0.0, 1000.0, 1000.0)
    tiny = (500.0, 500.0, 510.0, 510.0)
    check("a crop containing the target entirely still scores IoU ~0",
          G.iou(crop, tiny) < 0.001 and G.coverage(crop, tiny) == 1.0,
          f"iou={G.iou(crop, tiny):.5f} cover={G.coverage(crop, tiny)}")
    check("centre-hit sees what IoU cannot", G.center_hit(crop, tiny))
    check("a crop next to the target does not", not G.center_hit((0.0, 0.0, 100.0, 100.0), tiny))


def test_union_xywh():
    print("\nV* ships [x, y, w, h] per target; a two-object question needs both")
    check("one box converts to xyxy", G.union_xywh([[10, 20, 30, 40]]) == (10.0, 20.0, 40.0, 60.0))
    check("two boxes union", G.union_xywh([[10, 20, 30, 40], [100, 5, 10, 10]])
          == (10.0, 5.0, 110.0, 60.0))


# ---------------------------------------------------------------------------
def _rec(box, W=2000, H=1500, rw=1312, rh=992, gt=(0.0, 0.0, 10.0, 10.0)):
    return {"strict_box": box, "row": {"_wh": (W, H), "_rwh": (rw, rh), "gt_box": gt}}


def test_a_wrong_frame_still_looks_healthy():
    print("\nThe failure with no symptom: a 0-1000 box read as resized pixels")
    W, H, rw, rh = 2000, 1500, 1312, 992
    # A model on a 0-1000 grid points at the centre of the picture.
    box = [450, 450, 550, 550]
    right = G.map_box(box, "rel1000", W, H, rw, rh)
    wrong = G.map_box(box, "resized", W, H, rw, rh)
    check("the wrong reading is a perfectly valid rectangle -- nothing raises",
          wrong is not None and wrong[2] > wrong[0] and wrong[3] > wrong[1], f"{wrong}")
    check("and it is NOT where the model pointed",
          abs(wrong[0] - right[0]) > 100, f"right={right} wrong={wrong}")
    check("both crops are small enough to satisfy r_box at gamma=0.6 -- which is why "
          "training looks fine while r_region quietly dies",
          ((wrong[2] - wrong[0]) * (wrong[3] - wrong[1])) / (W * H) <= 0.6
          and ((right[2] - right[0]) * (right[3] - right[1])) / (W * H) <= 0.6)

    # Now the gate's job: given a cohort of such boxes, does it name rel1000?
    gt = (900.0, 650.0, 1100.0, 850.0)       # the centre of the picture, V*-sized
    recs = [_rec([440 + i, 440 + i, 560 + i, 560 + i], gt=gt) for i in range(20)]
    stats = G.score_frames(recs)
    check("score_frames names rel1000 on a cohort of 0-1000 boxes",
          stats["winner"] == "rel1000", f"winner={stats['winner']} "
          f"margin={stats['iou_margin']}")
    check("and the three statistics agree, so there is no disagreement warning",
          "disagreement" not in stats, stats.get("disagreement", ""))


def test_structural_evidence_needs_no_ground_truth():
    print("\nThe structural test: where the numbers live, independent of model quality")
    W, H, rw, rh = 2000, 1500, 1312, 992
    # A model emitting resized-absolute pixels on a 1312x992 frame can exceed 1000.
    recs = [_rec([100, 100, 1200, 900])]
    stats = G.score_frames(recs)
    check("a coordinate above 1000 rules OUT the 0-1000 reading",
          stats["over_1000"] == 1 and stats["over_resized"] == 0)
    # A model on a 0-1000 grid never does.
    recs = [_rec([100, 100, 900, 900])]
    stats = G.score_frames(recs)
    check("a box entirely under 1000 is consistent with both, and says so",
          stats["over_1000"] == 0 and stats["over_resized"] == 0)
    check("max coordinate is reported so the two can be eyeballed",
          stats["max_coord_p50"] == 900)


def test_no_boxes_is_undecided_not_a_verdict():
    print("\nZero parseable boxes: the frame is UNDECIDED, and the report must say so")
    stats = G.score_frames([_rec(None), _rec(None)])
    check("n_boxes is 0", stats["n_boxes"] == 0)
    check("a note replaces the verdict", "winner_note" in stats)
    check("turn 2 still has a frame to crop under", stats["winner"] in G.FRAMES)


# ---------------------------------------------------------------------------
def test_smart_resize_is_on_32():
    print("\nsmart_resize: the 28 that is a 32 on this backbone")
    check("the module constant is 32, not ERGO's 28", G.IMAGE_FACTOR == 32)
    h, w = G.smart_resize(1500, 2000, factor=32, min_pixels=64 * 1024,
                          max_pixels=1280 * 32 * 32)
    check("both sides land on a multiple of 32", h % 32 == 0 and w % 32 == 0, f"{w}x{h}")
    check("the budget is respected", (h * w) <= 1280 * 32 * 32, f"{h * w} px")
    check("a 1280-token budget really is ~1280 tokens, not ~980",
          1200 <= (h * w) // 1024 <= 1280, f"{(h * w) // 1024} tokens")
    # The trap, stated as a test: ERGO's literal 28 buys 23% fewer tokens here.
    h28, w28 = G.smart_resize(1500, 2000, factor=28, min_pixels=4 * 28 * 28,
                              max_pixels=1280 * 28 * 28)
    check("ERGO's literal N*28*28 budget is ~23% short on Qwen3-VL",
          0.70 < ((h28 * w28) / 1024) / ((h * w) / 1024) < 0.82,
          f"{(h28 * w28) // 1024} vs {(h * w) // 1024} tokens")


def test_prompts_match_upstream():
    print("\nThe vendored prompts, diffed against the ERGO checkout when it is present")
    try:
        checked = G._check_ergo_sources("/home/uberger/scratch/research/ERGO/src/ergo")
    except SystemExit as e:
        check("prompts match nota-github/ERGO src/ergo/const.py", False, str(e))
        return
    check("prompts match nota-github/ERGO src/ergo/const.py (or no checkout to diff)",
          True, "diffed" if checked else "no checkout -- skipped")
    check("QUESTION_TEMPLATE2's unclosed <think> is preserved verbatim",
          "<think> ... <think>" in G.ERGO_QUESTION_TEMPLATE2)


def main():
    test_strict_parser_is_ergos()
    test_lenient_parser_separates_the_near_misses()
    test_turn_format_checks()
    test_map_box_geometry()
    test_iou_and_coverage()
    test_union_xywh()
    test_a_wrong_frame_still_looks_healthy()
    test_structural_evidence_needs_no_ground_truth()
    test_no_boxes_is_undecided_not_a_verdict()
    test_smart_resize_is_on_32()
    test_prompts_match_upstream()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  FAILED: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
