#!/usr/bin/env python
"""CPU checks for deepeyes_coord_format_probe.py.

    python test_deepeyes_coord_format_cpu.py

No GPU, no model, no network. Everything here is the half of the probe that decides what
the numbers MEAN -- the three coordinate conversions, DeepEyes' format rule, and the
tool-call parser. The GPU half only produces strings; if this file is green, a wrong
headline can only come from the model.

The section worth reading before trusting the result is 5, *the readings are separable*.
It plants a box in each of the three frames in turn and asserts the probe picks that one
back out. That is not free: under arm A's processor caps `abs_orig` and `abs_resized`
coincide exactly, and a probe that cannot separate its own hypotheses will quietly report
whichever is listed first -- and Prompt 3 would then wire the tool to it.
"""
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
FAILED = []


def check(label, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))
    if not ok:
        FAILED.append(label)


def close(a, b, tol=1e-6):
    return abs(a - b) <= tol


spec = importlib.util.spec_from_file_location(
    "_decf", REPO / "deepeyes_coord_format_probe.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)


# --------------------------------------------------------------------------------------
print("\n1. DeepEyes' format rule, transcribed verbatim from vl_agent.compute_score")
# --------------------------------------------------------------------------------------
for text, want, why in [
    ("<think>a</think><tool_call>{}</tool_call>", False, "balanced think, no answer tags"),
    ("<think>a</think><answer>b</answer>", False, "the intended shape"),
    ("<think>a<answer>b</answer>", True, "unbalanced think"),
    ("<think>a</think><answer>b", True, "unbalanced answer after </think>"),
    ("just an answer, no tags at all", False,
     "THE TRAP: 0 == 0 balances, so their rule gives this no penalty"),
    ("<think>a</think>" + "x" * 1200, True,
     "no <answer>, so answer_text is the whole tail and trips the 1000-char clause"),
    ("<think>a</think><answer>" + "x" * 1200 + "</answer>", True,
     "their anti-judge-hacking clause, which doubles as a format error"),
]:
    check(why, P.deepeyes_is_format_error(text) is want)

check("balanced vision splices are not an error",
      P.deepeyes_is_format_error(
          "<think>a</think><|vision_start|><|image_pad|><|image_pad|><|vision_end|>")
      is False)
check("a half-typed vision splice is",
      P.deepeyes_is_format_error(
          "<think>a</think><|vision_start|><|image_pad|> and nothing closing it") is True)


# --------------------------------------------------------------------------------------
print("\n1b. the answer parse -- what a tagless model actually loses, which is NOT -1")
# --------------------------------------------------------------------------------------
# Prompt 4's brief said R_format is "-1 on any tag mismatch", so a model at format 0.000
# is penalised on every rollout. Their code says otherwise, and these pin the difference.
check("a tagless completion is not a format error at all",
      P.deepeyes_is_format_error("The flag is red.") is False)
check("with no <answer>, the judge is handed the WHOLE completion",
      P.deepeyes_answer_text("Okay, so the flag looks white. The answer is A.")
      == "Okay, so the flag looks white. The answer is A.")
check("with tags, it is handed the answer",
      P.deepeyes_answer_text("<think>x</think><answer> A </answer>") == "A")
check("the last <answer> wins, as their split does",
      P.deepeyes_answer_text("<answer>a</answer><answer>b</answer>") == "b")
long_chain = "Okay, so " + "I need to look at the image. " * 50
check("a long tagless chain is >= 1000 chars", len(long_chain) >= 1000)
check("... so acc_reward is forced to 0.0 and the format error is set too",
      P.deepeyes_is_format_error(long_chain) is True)
check("a SHORT tagless chain costs nothing at all -- no penalty, no zero",
      P.deepeyes_is_format_error("A.") is False
      and P.deepeyes_answer_text("A.") == "A.")


# --------------------------------------------------------------------------------------
print("\n2. tag_structure_ok is strictly harder than their rule -- the reason both "
      "numbers are reported")
# --------------------------------------------------------------------------------------
bare = "The flag is red."
check("a bare untagged answer costs no R_format penalty",
      P.deepeyes_is_format_error(bare) is False)
check("... and is nevertheless no protocol at all", P.tag_structure_ok(bare) is False)
check("think + tool_call passes",
      P.tag_structure_ok('<think>I should look closer.</think>\n<tool_call>\n'
                         '{"name": "image_zoom_in_tool", "arguments": '
                         '{"bbox_2d": [1,2,3,4]}}\n</tool_call>') is True)
check("think + answer passes",
      P.tag_structure_ok("<think>I can see it.</think><answer>red</answer>") is True)
check("two think blocks do not",
      P.tag_structure_ok("<think>a</think><think>b</think><answer>c</answer>") is False)


# --------------------------------------------------------------------------------------
print("\n3. the tool-call parser")
# --------------------------------------------------------------------------------------
t = ('<think>x</think><tool_call>{"name": "image_zoom_in_tool", '
     '"arguments": {"bbox_2d": [10, 20, 100, 200], "label": "flag"}}</tool_call>')
check("json tool call", P.first_bbox(P.parse_tool_calls(t)) == [10.0, 20.0, 100.0, 200.0])

# eval_vstar.py uses eval(), so a python-dict-shaped call is in distribution.
t = ("<tool_call>{'name': 'image_zoom_in_tool', "
     "'arguments': {'bbox_2d': [1, 2, 3, 4]}}</tool_call>")
check("single-quoted tool call", P.first_bbox(P.parse_tool_calls(t)) == [1.0, 2.0, 3.0, 4.0])

t = ('<tool_call>{"name":"image_zoom_in_tool","arguments":{"bbox_2d":[1,1,2,2]}}</tool_call>'
     '<tool_call>{"name":"image_zoom_in_tool","arguments":{"bbox_2d":[9,9,10,10]}}</tool_call>')
calls = P.parse_tool_calls(t)
check("two calls are both parsed", len(calls) == 2, f"{len(calls)}")
check("... and the FIRST zoom is the one scored",
      P.first_bbox(calls) == [1.0, 1.0, 2.0, 2.0])

check("malformed json yields no call", P.parse_tool_calls("<tool_call>{not json</tool_call>") == [])
check("three numbers is not a box",
      P.first_bbox(P.parse_tool_calls(
          '<tool_call>{"name":"z","arguments":{"bbox_2d":[1,2,3]}}</tool_call>')) is None)
check("no calls at all", P.first_bbox([]) is None)


# --------------------------------------------------------------------------------------
print("\n4. geometry")
# --------------------------------------------------------------------------------------
check("IoU of a box with itself", close(P.iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0))
check("IoU of disjoint boxes", P.iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0)
check("IoU of a half overlap", close(P.iou([0, 0, 10, 10], [5, 0, 15, 10]), 50 / 150))
check("a crop 100x the target still contains all of it",
      close(P.covered([10, 10, 20, 20], [0, 0, 100, 100]), 1.0))
check("half-covered target", close(P.covered([10, 10, 20, 20], [15, 10, 20, 20]), 0.5))
check("out_of_frame: inside", P.out_of_frame([0, 0, 10, 10], 100, 100) == 0.0)
check("out_of_frame: half out", close(P.out_of_frame([95, 0, 105, 10], 100, 100), 0.5))
check("out_of_frame: entirely out", close(P.out_of_frame([200, 200, 300, 300], 100, 100), 1.0))

W, H, rW, rH = 2000, 1000, 1000, 500      # a 2x linear downscale
b = [100, 200, 300, 400]
check("abs_orig is the identity", P.convert(b, "abs_orig", W, H, rW, rH) == [100, 200, 300, 400])
check("rel1000 scales by W/1000, H/1000",
      P.convert(b, "rel1000", W, H, rW, rH) == [200.0, 200.0, 600.0, 400.0])
check("abs_resized scales by W/rW, H/rH",
      P.convert(b, "abs_resized", W, H, rW, rH) == [200.0, 400.0, 600.0, 800.0])
try:
    P.convert(b, "nonsense", W, H, rW, rH)
    check("an unknown reading raises", False)
except ValueError:
    check("an unknown reading raises", True)

# Arm A's honest limitation, pinned so it cannot be forgotten in the writeup.
S = 1536
check("with no downscale, abs_orig and abs_resized are the SAME reading",
      P.convert(b, "abs_orig", S, S, S, S) == P.convert(b, "abs_resized", S, S, S, S))


# --------------------------------------------------------------------------------------
print("\n5. the readings are separable -- the test the headline rests on")
# --------------------------------------------------------------------------------------
def rec(completion, W=2000, H=1000, rW=1000, rH=500, gt=((900, 400, 1100, 600),),
        n_tok=42, budget=320):
    return {"uid": "t/0", "split": "direct_attributes", "model": "m", "arm": "B",
            "W": W, "H": H, "rW": rW, "rH": rH, "n_completion_tokens": n_tok,
            "max_new_tokens": budget, "proc_max_pixels": 327680,
            "gt_boxes": [list(g) for g in gt], "completion": completion}


def call(box):
    return ('<think>look closer</think><tool_call>{"name": "image_zoom_in_tool", '
            '"arguments": {"bbox_2d": %s, "label": "x"}}</tool_call>' % list(box))


gt = (900, 400, 1100, 600)                 # original pixels, in a 2000x1000 picture
for truth, box in [("rel1000", [450, 400, 550, 600]),       # x/W*1000, y/H*1000
                   ("abs_orig", [900, 400, 1100, 600]),
                   ("abs_resized", [450, 200, 550, 300])]:  # x*rW/W, y*rH/H
    s = P.score_record(rec(call(box), gt=(gt,)))
    ious = {k: v["iou"] for k, v in s["readings"].items()}
    best = max(ious, key=ious.get)
    check(f"a box planted in {truth} is read back as {truth}", best == truth, str(ious))
    check(f"... at IoU 1.0", close(ious[truth], 1.0), f"{ious[truth]:.4f}")

# The frame signal that needs no ground truth: an original-pixel box near the right edge
# of a 2000x1000 picture, misread as a 0-1000 grid, lands off the bottom-right corner.
s = P.score_record(rec(call([1800, 900, 1900, 950])))
check("off-image flags the wrong frame with no ground truth",
      close(s["readings"]["abs_orig"]["oof"], 0.0) and s["readings"]["rel1000"]["oof"] > 0.5,
      f"abs {s['readings']['abs_orig']['oof']:.2f} rel {s['readings']['rel1000']['oof']:.2f}")


# --------------------------------------------------------------------------------------
print("\n6. a box written without the tags is a format failure, not a tool call")
# --------------------------------------------------------------------------------------
s = P.score_record(rec("<think>I will zoom</think> bbox_2d: [10, 20, 100, 200]",
                       W=1000, H=1000, rW=1000, rH=1000))
check("no tool call is recorded", s["tool_call_ok"] is False)
check("the loose box is counted separately", s["loose_bbox_only"] is True)
check("and it is never scored for IoU", s["readings"] == {})


# --------------------------------------------------------------------------------------
print("\n6b. their ENV discards a zoom that shares a turn with a closed <answer>")
# --------------------------------------------------------------------------------------
both = call([450, 400, 550, 600]) + "  <answer> B </answer>"
s = P.score_record(rec(both))
check("the tool call is still parsed", s["tool_call_ok"] is True)
check("but the env sees an answer first", s["env_sees_answer"] is True)
check("so the zoom is discarded", s["env_zoom_discarded"] is True)
check("and env_would_zoom is False -- this is what gates R_tool",
      s["env_would_zoom"] is False)

# Their extract_answer needs BOTH tags; the format rule counts the unclosed one. The two
# disagree on exactly this string, and that disagreement is a real training condition.
unclosed = call([450, 400, 550, 600]) + "  <answer> B"
s = P.score_record(rec(unclosed))
check("an UNCLOSED <answer> is invisible to the env, so the zoom happens",
      s["env_would_zoom"] is True)
check("... while their format rule still scores it -1",
      s["format_ok_deepeyes"] is False)


# --------------------------------------------------------------------------------------
print("\n7. truncation is OUR failure, not the model's, and is reported apart")
# --------------------------------------------------------------------------------------
cut = rec("<think>I was still thinking when the budget ran", n_tok=320, budget=320)
check("a completion at the budget is flagged truncated",
      P.score_record(cut)["truncated"] is True)
check("... and their rule scores it a format error (unbalanced <think>)",
      P.score_record(cut)["format_ok_deepeyes"] is False)
check("one under the budget is not flagged",
      P.score_record(rec("<think>a</think><answer>b</answer>", n_tok=319,
                         budget=320))["truncated"] is False)
cell = P.summarise([P.score_record(r) for r in [
    cut, rec("<think>a</think><answer>b</answer>", n_tok=50, budget=320)]])
check("truncated_rate", close(cell["truncated_rate"], 0.5))
check("format_ok over everything", close(cell["format_ok_deepeyes"], 0.5))
check("format_ok over the ones that finished", close(cell["format_ok_untruncated"], 1.0))


# --------------------------------------------------------------------------------------
print("\n8. the summary cell")
# --------------------------------------------------------------------------------------
cell = P.summarise([P.score_record(r) for r in [
    rec(call([450, 400, 550, 600])),
    rec("<think>I can see it already.</think><answer>red</answer>"),
    rec("no tags at all"),
]])
check("n", cell["n"] == 3)
check("n_boxes", cell["n_boxes"] == 1)
check("tool_call_rate", close(cell["tool_call_rate"], 1 / 3))
check("their rule passes all three, including the one with no protocol",
      close(cell["format_ok_deepeyes"], 1.0))
check("tag_structure_ok passes two", close(cell["tag_structure_ok"], 2 / 3))
check("the winning reading's mean IoU", close(cell["readings"]["rel1000"]["mean_iou"], 1.0))
check("the cell records the arm's pixel cap", cell["proc_max_pixels"] == 327680)
check("one of the three has no <answer> tag at all",
      close(cell["answer_is_whole_completion_rate"], 2 / 3),
      f"{cell['answer_is_whole_completion_rate']:.3f}")

cell = P.summarise([P.score_record(r) for r in [
    rec("Okay, so " + "I need to look at the image. " * 50),   # tagless and long
    rec("<think>x</think><answer> A </answer>"),
]])
check("acc_zeroed_by_length_rate counts the long tagless chain",
      close(cell["acc_zeroed_by_length_rate"], 0.5))
check("... and their format rule half-agrees, for the same single reason",
      close(cell["format_ok_deepeyes"], 0.5))


# --------------------------------------------------------------------------------------
print("\n9. the prompt is theirs, faults included")
# --------------------------------------------------------------------------------------
check("their trailing double-space survives", "**Example**:  \n" in P.SYSTEM_PROMPT)
check("their `required: [bbox]` typo survives (the parameter is bbox_2d)",
      '"required":["bbox"]' in P.SYSTEM_PROMPT)
check("the user prompt's trailing space survives", P.USER_PROMPT.endswith("</answer> "))
check("... and its leading newline", P.USER_PROMPT.startswith("\n"))
rendered = P.USER_TEMPLATE.format(question="Is the flag red or white?",
                                  options="\nA. white\nB. red\n")
check("the rendered user turn is their layout",
      rendered.startswith("Question: Is the flag red or white?\nOptions: \n")
      and rendered.endswith(P.USER_PROMPT))

if (P.UPSTREAM / "eval" / "eval_vstar.py").exists():
    P._check_prompt_matches_upstream()
    check("the embedded prompt matches the clone, character for character", True)
else:
    print(f"  skip  no DeepEyes clone at {P.UPSTREAM}; prompt left unverified")


print("\n" + ("ALL PASS" if not FAILED else f"{len(FAILED)} FAILED: {FAILED}"))
sys.exit(1 if FAILED else 0)
