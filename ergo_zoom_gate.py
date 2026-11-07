#!/usr/bin/env python
"""Will a Qwen3-VL checkpoint run ERGO's two turns, and in WHICH coordinate frame?

    python ergo_zoom_gate.py --model Qwen/Qwen3-VL-8B-Instruct
    python ergo_zoom_gate.py --model checkpoint/coldstart_..._merged \
        --reference Qwen/Qwen3-VL-8B-Instruct --out outputs/ergo_gate_zoom/gate.json

WHY THIS EXISTS

ERGO (arXiv:2509.21991) is two turns. Turn 1 must emit `<think>…</think>` followed by
`<zoom>[x1,y1,x2,y2]</zoom>`; the box is cropped out of the ORIGINAL image and shown back
to the model, which must then answer in `<think>…</think><answer>…</answer>`. Both of
those are gated by `r_format`, and ERGO's rule is that a turn-1 completion with no
parseable box scores ZERO on every term. Two independent things can therefore look like
"it works" right up until the run is a space heater:

  1. **The format is never produced.** `build_coldstart_data.py:52` records Qwen3-VL-8B-
     Instruct at format 0.000 under every `<think>` prompt variant tried -- that is the
     whole reason our cold start exists. If turn 1 is malformed on most rollouts, every
     reward in the GRPO group is 0, the group advantage is 0, and there is no gradient at
     all. Nothing errors. The loss curve looks like a loss curve.

  2. **The box is read in the wrong coordinate frame.** Qwen2.5-VL -- the backbone ERGO
     was built on -- emits absolute pixels in the POST-`smart_resize` frame, which is why
     `src/ergo/utils.py:map_box` rescales a box by `original/resized`. Qwen3-VL is widely
     documented to emit a 0-1000 RELATIVE grid against the original image instead. Port
     `map_box` unchanged onto Qwen3-VL and every crop lands somewhere else in the picture,
     while the pipeline stays perfectly healthy: `r_region` (the reward model, shown only
     the crop) just fails more often, `r_box` (crop area <= gamma) still fires, and the
     policy quietly optimises box AREA -- i.e. the paper's one contribution is gone and
     the training curve does not say so.

This measures both, on V* -- which is the benchmark ERGO leads with AND the only one of
its three that ships ground-truth boxes, so it is the only place the frame question can
be answered from data rather than from a docs page.

WHAT IT MEASURES

  * turn-1 zoom-parse rate, strict (ERGO's own `parse_zoom_bbox_from_text`, verbatim) and
    lenient (any four numbers anywhere). A single rate cannot tell "ignored the tag" from
    "wrote the box without the tag" from "wrote floats", and those have different fixes.
  * turn-1 think-tag rate, and the joint `r_format` turn-1 condition.
  * turn-2 `<think>…</think><answer>…</answer>` rate, strict and lenient.
  * the coordinate frame, EMPIRICALLY: every emitted box is mapped into the original
    image under all three candidate readings and scored against the GT box. Reported with
    structural evidence (where the coordinate values actually live) alongside the task
    evidence (IoU / coverage), because the two can disagree and when they do that is the
    interesting finding, not a tie to be broken silently.
  * median crop area as a fraction of the original, so ERGO's gamma = 0.6 can be checked
    against what this policy actually emits.

WHAT IT IS NOT: an accuracy measurement. Turn-2 answer accuracy is printed for context
only -- a cold start is not expected to be accurate and this gate has no opinion about it.

TWO PASSES, ON PURPOSE. Turn 1 is generated for every row first; the frame is decided from
the AGGREGATE; only then is turn 2 generated, cropping under the winning frame. Choosing
the frame per row from that row's own IoU would be reading the answer key, and the trainer
cannot do it either -- it has one global convention or it has nothing.

Sampled at temperature 1.0, not greedy, for the same reason `coldstart_gate.py` is: that
is what GRPO generates at, and greedy overstates format compliance.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

# ---------------------------------------------------------------------------
# ERGO's released prompts, verbatim from nota-github/ERGO src/ergo/const.py (Apache 2.0).
# Copied rather than imported because the clone lives in scratch on one machine and this
# repo runs on two; `_check_ergo_sources` below diffs them back against the clone whenever
# it IS present, so the copy cannot drift in silence.
#
# The `<think> ... <think>` in QUESTION_TEMPLATE2 is THEIR typo (the closing tag is not a
# closing tag). It is reproduced exactly: fixing it would measure a prompt ERGO-7B was
# never trained under, and the point of this gate is to predict what their recipe does on
# our checkpoint. The turn-2 rate is reported under both the strict reading and a lenient
# one precisely because of it.
# ---------------------------------------------------------------------------
ERGO_SYSTEM_PROMPT = (
    "You are a helpful assistant. Given an image and one question. First, identify the coordinates "
    "of the key image area relevant to solving the problem. Append the coordinates in [x1, y1, x2, y2] "
    "format at the end of your response and stop. This will trigger cropping of the corresponding area in "
    "the original image and enlarge it for improved clarity. Once the enlarged image is available, provide the final answer."
)

ERGO_QUESTION_TEMPLATE1 = "{Question} First, consider how to zoom in on the image to include only the region containing all the information necessary to answer the question. Output the thinking process in <think> </think> tags and then output the region's bounding box coordinates in the following format <zoom>[x1, y1, x2, y2]</zoom> tags."

ERGO_QUESTION_TEMPLATE2 = "{Question}\nCarefully analyze both the original image and the enlarged sub-image to solve the question step by step. Present your reasoning clearly the following format strictly <think> ... <think>, and provide the final answer enclosed within <answer> ... </answer>"

#: Appended to QUESTION_TEMPLATE1 by `--coord-hint`. Not part of ERGO and NOT used for the
#: headline measurement -- the frame has to be measured under the verbatim prompt or the
#: prompt is the thing being measured. This exists only so that a low parse rate can be
#: told apart from a rescuable one in the same job.
COORD_HINTS = {
    "none": "",
    "rel1000": (" Give the coordinates on a 0-1000 relative grid, where 0 is the left/top "
                "edge of the image and 1000 is the right/bottom edge."),
    "abs": (" Give the coordinates in absolute pixels of the image as you see it."),
}

# ---------------------------------------------------------------------------
# Qwen3-VL geometry. patch 16, merge 2x2 -> one visual token covers 32x32 px, NOT the 28
# that is hard-coded through every line of ERGO's code and every row of its tables. A
# budget copied literally as `N*28*28` buys 23% fewer tokens on this backbone than the
# number says, which is how an efficiency paper ends up reporting the wrong operating
# point. `vlm_family.Qwen3VL.patch_px` is the same constant, independently.
# ---------------------------------------------------------------------------
IMAGE_FACTOR = 32
#: ERGO's default `--max_vision_token_num`, restated in this backbone's token size.
DEFAULT_VISION_TOKENS = 1280
#: ERGO uses 4 tokens as its floor. Qwen3-VL's own image processor refuses to go below
#: `size.shortest_edge = 65536` px = 64 tokens and silently UPSCALES anything smaller, so
#: a 4-token floor here would mean our recorded crop geometry is not the crop the model
#: got. Set to the processor's real floor instead, and `processor_resized_under_us` in the
#: report is the assertion that the two agree.
MIN_VISION_TOKENS = 64

#: The three readings of an emitted box. Each maps (x1,y1,x2,y2) into ORIGINAL pixels.
#:   resized  -- absolute pixels in the post-smart_resize frame. Qwen2.5-VL's convention
#:               and the one `ergo/utils.py:map_box` implements.
#:   rel1000  -- a 0-1000 grid relative to the image. Qwen3-VL's documented convention.
#:   original -- absolute pixels in the original frame; the model ignoring the resize.
FRAMES = ("resized", "rel1000", "original")

#: Below this on the cold start, ERGO needs a two-turn cold start of its own (~2-5k SFT
#: rows in its format) before GRPO can start. That is +3-5 days and it is the single
#: thing that blows this arm's budget, so the number is a loud gate and not a footnote.
MIN_ZOOM_PARSE = 0.60
#: `r_format` needs BOTH turns. A turn-2 rate this low has the same consequence as a
#: turn-1 one -- zero reward, zero advantage -- but is cheaper to fix (it is our own
#: cold-start format plus `<answer>` tags, which the cold start already knows how to
#: suppress; see coldstart_gate.py failure 3, which is this one inverted).
MIN_TURN2_FORMAT = 0.60


# ---------------------------------------------------------------------------
# ERGO's own parser and resizer, transcribed verbatim. Checked back against the clone.
# ---------------------------------------------------------------------------
def round_by_factor(number, factor):
    return round(number / factor) * factor


def ceil_by_factor(number, factor):
    return math.ceil(number / factor) * factor


def smart_resize(height, width, factor=IMAGE_FACTOR, min_pixels=256, max_pixels=1024):
    """ERGO's `src/ergo/utils.py:smart_resize`, with `factor` defaulted to 32 for Qwen3-VL.

    Returns (h_bar, w_bar). Theirs omits the MAX_RATIO guard Qwen's official helper has;
    kept omitted, because V*'s widest row is 5759x1440 (ratio 4.0) and adding a check they
    do not have would be a silent behavioural difference rather than a fix.
    """
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = round_by_factor(height / beta, factor)
        w_bar = round_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def parse_zoom_strict(content):
    """ERGO's `parse_zoom_bbox_from_text`, verbatim. INTEGERS ONLY, inside <zoom> tags.

    This is the function the reward would call, so this is the rate that decides whether
    there is a gradient. Everything looser is diagnosis, not the gate.
    """
    try:
        answer_match = re.search(r"<zoom>(.*?)</zoom>", content, re.DOTALL)
        if not answer_match:
            return None
        answer_content = answer_match.group(1)
        if re.match(r"^\s*\[\d+,\s*\d+,\s*\d+,\s*\d+\]\s*$", answer_content):
            return [int(num.strip()) for num in answer_content.strip().strip("[]").split(",")]
        bbox_patterns = re.findall(r"\[\d+,\s*\d+,\s*\d+,\s*\d+\]", answer_content)
        if len(bbox_patterns) == 1:
            bbox_str = bbox_patterns[0]
            return [int(num.strip()) for num in bbox_str.strip("[]").split(",")]
        return None
    except Exception:
        return None


_NUM = r"-?\d+(?:\.\d+)?"
_FOUR = re.compile(r"\[\s*(%s)\s*,\s*(%s)\s*,\s*(%s)\s*,\s*(%s)\s*\]" % (_NUM, _NUM, _NUM, _NUM))


def parse_zoom_lenient(content):
    """Any four numbers in brackets, floats allowed, tags optional.

    The three ways a well-meaning model misses the strict parser, separated so the fix is
    obvious: wrote floats (`[12.0, …]`), wrote the box outside the tags, or wrote several
    boxes. Prefers whatever is inside <zoom> when <zoom> exists.
    """
    m = re.search(r"<zoom>(.*?)</zoom>", content, re.DOTALL)
    scope = m.group(1) if m else content
    hits = _FOUR.findall(scope)
    if not hits:
        return None
    return [float(v) for v in hits[0]]


def _norm_prompt(s):
    """A prompt reduced to its words, so a source file and a Python value can be compared.

    Three differences are not differences: `const.py` writes SYSTEM_PROMPT as adjacent
    string literals (so the quote-newline-indent-quote between them is not in the value),
    writes `\\n` as a two-character escape (the value has a real newline), and wraps lines
    where it likes. Everything else -- any changed or dropped word -- survives this and
    trips the check.
    """
    return re.sub(r"\s+", " ", s.replace("\\n", " ").replace('"', "").replace("'", "")).strip()


def _check_ergo_sources(ergo_src):
    """Diff the vendored prompts and parser against the clone, when the clone is here.

    `coldstart_gate._check_pattern_still_matches` is the precedent and the reason: a gate
    scoring a prompt the method no longer uses is worse than no gate, and a copied
    constant is exactly the thing that drifts. Absent clone -> skip, with a line saying so,
    because this must still run on the cluster that has no scratch checkout.
    """
    const = Path(ergo_src) / "const.py"
    utils = Path(ergo_src) / "utils.py"
    if not const.exists() or not utils.exists():
        print(f"NOTE: no ERGO checkout at {ergo_src}; prompts not diffed against upstream.\n"
              f"      git clone https://github.com/nota-github/ERGO to re-enable the check.",
              file=sys.stderr)
        return False
    text = const.read_text()
    got = _norm_prompt(text)
    for name, value in (("SYSTEM_PROMPT", ERGO_SYSTEM_PROMPT),
                        ("QUESTION_TEMPLATE1", ERGO_QUESTION_TEMPLATE1),
                        ("QUESTION_TEMPLATE2", ERGO_QUESTION_TEMPLATE2)):
        # Compared on the whole prompt, normalised on both sides, so that their wrapped
        # adjacent string literals and their `\n` escapes do not false-alarm while a
        # changed WORD still does.
        want = _norm_prompt(value)
        if want not in got:
            raise SystemExit(
                f"ergo_zoom_gate: {name} in this file is not in {const} any more.\n"
                "ERGO's released prompt moved; update the copy here rather than measuring\n"
                "a prompt their checkpoint was never trained under.")
    src = utils.read_text()
    if "def parse_zoom_bbox_from_text" not in src or '<zoom>(.*?)</zoom>' not in src:
        raise SystemExit(f"ergo_zoom_gate: {utils} no longer defines the zoom parser this "
                         "file transcribes. Re-transcribe parse_zoom_strict from it.")
    return True


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def map_box(box, frame, W, H, rw, rh):
    """One emitted box, read under `frame`, in ORIGINAL-image pixels. None if degenerate.

    Clipping happens here and not in the caller because a box that only becomes valid
    after clipping (e.g. 0-1000 coordinates read as resized pixels on a 640px-wide frame)
    is still a box the trainer would crop, and excluding it would flatter that frame.
    """
    x1, y1, x2, y2 = (float(v) for v in box)
    if frame == "resized":
        sx, sy = W / float(rw), H / float(rh)
    elif frame == "rel1000":
        sx, sy = W / 1000.0, H / 1000.0
    elif frame == "original":
        sx = sy = 1.0
    else:
        raise ValueError(frame)
    x1, x2 = x1 * sx, x2 * sx
    y1, y2 = y1 * sy, y2 * sy
    # Some models emit the corners in the other order; ERGO's own mapper does not fix
    # this, but leaving it broken would charge a frame for a formatting habit.
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1 = max(0.0, min(x1, W))
    x2 = max(0.0, min(x2, W))
    y1 = max(0.0, min(y1, H))
    y2 = max(0.0, min(y2, H))
    if x2 - x1 < 1.0 or y2 - y1 < 1.0:
        return None
    return (x1, y1, x2, y2)


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def coverage(pred, gt):
    """Fraction of the GT box that the crop contains.

    Reported next to IoU because V*'s targets are tiny -- the median GT box is well under
    1% of the picture -- so a crop that contains the whole answer still scores an IoU near
    zero on box-size grounds alone. Coverage is what `r_region` actually depends on: the
    reward model is shown only the crop, so what matters is whether the evidence is in it.
    """
    ix1, iy1 = max(pred[0], gt[0]), max(pred[1], gt[1])
    ix2, iy2 = min(pred[2], gt[2]), min(pred[3], gt[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ga = (gt[2] - gt[0]) * (gt[3] - gt[1])
    return inter / ga if ga > 0 else 0.0


def center_hit(pred, gt):
    cx, cy = (gt[0] + gt[2]) / 2.0, (gt[1] + gt[3]) / 2.0
    return pred[0] <= cx <= pred[2] and pred[1] <= cy <= pred[3]


def union_xywh(boxes):
    """V* ships one [x, y, w, h] per target object; a two-object question needs both."""
    xs1 = [b[0] for b in boxes]
    ys1 = [b[1] for b in boxes]
    xs2 = [b[0] + b[2] for b in boxes]
    ys2 = [b[1] + b[3] for b in boxes]
    return (float(min(xs1)), float(min(ys1)), float(max(xs2)), float(max(ys2)))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_vstar(root, n):
    """V* bench: 191 questions, GT boxes, and the benchmark ERGO's Table 1 leads with.

    `craigwu/vstar_bench` and NOT the lmms-eval mirror `lmms-lab-encoder/vstar-bench`,
    because only the original ships the per-image `.json` carrying `bbox`. HR-Bench has no
    boxes at all (checked: its parquet is index/answer/question/A-D/category/image), so it
    can contribute parse and format rates but never a frame verdict -- which is why this
    gate is V*-only by default.
    """
    root = Path(root)
    ann = root / "test_questions.jsonl"
    if not ann.exists():
        raise SystemExit(
            f"ergo_zoom_gate: {ann} not found. Fetch it once with\n"
            "  python -c \"from huggingface_hub import snapshot_download as s; "
            "print(s('craigwu/vstar_bench', repo_type='dataset'))\"")
    rows = []
    for line in ann.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        img = root / r["image"]
        meta = img.with_suffix(".json")
        if not img.exists() or not meta.exists():
            continue
        a = json.loads(meta.read_text())
        lines = r["text"].split("\n")
        rows.append({
            "image_path": str(img),
            # ERGO's `preprocess_data(dataset="vstar")`: first line is the question, the
            # middle lines are the options and the last is the "answer with the letter"
            # instruction. Turn 1 gets the bare question (that is what ERGO sends);
            # turn 2 gets question + options + postfix.
            "question": lines[0],
            "options": lines[1:-1],
            "postfix": lines[-1],
            "label": r["label"],
            "category": r["category"],
            "question_id": r["question_id"],
            "gt_box": union_xywh(a["bbox"]),
        })
    if n:
        rows = rows[:n]
    return rows


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def build_turn1(question, hint):
    return [
        {"role": "system", "content": ERGO_SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": ERGO_QUESTION_TEMPLATE1.format(Question=question) + hint},
        ]},
    ]


def build_turn2(msgs1, turn1_text, full_question):
    return msgs1 + [
        {"role": "assistant", "content": [{"type": "text", "text": turn1_text}]},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": ERGO_QUESTION_TEMPLATE2.format(Question=full_question)},
        ]},
    ]


def generate_batch(model, processor, convs, images, max_new_tokens, temperature, device):
    """One left-padded batch. `images` is one list of PIL images per conversation.

    Left padding and `add_special_tokens=False` match the trainer's own prompt build; the
    nested image list is what the Qwen3-VL processor's batching expects (see
    `vlm_family.Qwen3VL.batch_image_arg`).
    """
    import torch

    texts = [processor.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
             for c in convs]
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True,
                       padding_side="left", add_special_tokens=False).to(device)
    gen_kw = dict(max_new_tokens=max_new_tokens)
    if temperature and temperature > 0:
        gen_kw.update(do_sample=True, temperature=temperature, top_p=1.0)
    else:
        gen_kw.update(do_sample=False)
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kw)
    n_prompt = inputs["input_ids"].shape[1]
    texts_out = processor.tokenizer.batch_decode(out[:, n_prompt:], skip_special_tokens=True)
    grids = inputs.get("image_grid_thw")
    return [t.strip() for t in texts_out], grids


# ---------------------------------------------------------------------------
# Format checks
# ---------------------------------------------------------------------------
#: One <think> block and one <zoom> block, in that order, and nothing before the think.
TURN1_STRICT = re.compile(r"^\s*<think>\s*(.+?)\s*</think>\s*<zoom>\s*(.+?)\s*</zoom>\s*$",
                          re.DOTALL)
#: ERGO's turn-2 target. Their own template asks for `<think> ... <think>` (sic), so the
#: strict form below is the SENSIBLE reading of the intent, and `turn2_lenient` is the
#: reading their typo would actually accept. Both are reported.
TURN2_STRICT = re.compile(r"^\s*<think>\s*(.+?)\s*</think>\s*<answer>\s*(.+?)\s*</answer>\s*$",
                          re.DOTALL)


def turn1_flags(text):
    has_think = text.count("<think>") == 1 and text.count("</think>") == 1
    one_zoom = text.count("<zoom>") == 1 and text.count("</zoom>") == 1
    strict_box = parse_zoom_strict(text)
    lenient_box = parse_zoom_lenient(text)
    return {
        "think_ok": has_think,
        "zoom_tag": "<zoom>" in text and "</zoom>" in text,
        "strict_box": strict_box,
        "lenient_box": lenient_box,
        # The tag COUNTS are part of the gate and not a nicety. `TURN1_STRICT` alone
        # accepts `<think>a</think><think>b</think><zoom>…</zoom>`: the non-greedy group
        # swallows the inner pair happily. coldstart_gate.py pairs its regex with the same
        # count check for the same reason -- a repeated tag is a different failure from a
        # missing one and must not be scored as a pass.
        "format_ok": (bool(TURN1_STRICT.match(text)) and strict_box is not None
                      and has_think and one_zoom),
    }


def turn2_flags(text):
    has_answer = text.count("<answer>") == 1 and text.count("</answer>") == 1
    has_think_open = text.count("<think>") >= 1
    return {
        "strict": bool(TURN2_STRICT.match(text)),
        # What ERGO's own (typo'd) instruction describes: some <think>, one <answer> pair.
        "lenient": has_answer and has_think_open,
        "answer_tag": has_answer,
        "think_open": has_think_open,
    }


ANSWER_LETTER = re.compile(r"\b([A-D])\b")


def turn2_pred(text):
    """The option letter, read the way ERGO's eval reads it.

    `extract_answer_text` then keeps only letters, so "(A) rubber" becomes "Arubber" and
    their comparison against a one-letter gold fails. Reproduced loosely rather than
    literally: this number is context, and a scorer bug in their eval is not what this
    gate is for. Takes the first standalone A-D inside <answer> when there is one.
    """
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    scope = m.group(1) if m else text
    hit = ANSWER_LETTER.search(scope)
    return hit.group(1) if hit else None


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------
def run_one(model_path, rows, args, device):
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"\n{'=' * 78}\n{model_path}\n{'=' * 78}", flush=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa").to(device).eval()

    max_pixels = args.vision_tokens * IMAGE_FACTOR * IMAGE_FACTOR
    min_pixels = MIN_VISION_TOKENS * IMAGE_FACTOR * IMAGE_FACTOR
    hint = COORD_HINTS[args.coord_hint]
    print(f"vision tokens {args.vision_tokens} -> max_pixels {max_pixels} "
          f"({args.vision_tokens}x{IMAGE_FACTOR}x{IMAGE_FACTOR}; ERGO writes this 28x28)",
          flush=True)
    print(f"coord hint: {args.coord_hint!r}"
          + ("  <- NOT ERGO's prompt; frame verdict is only valid at 'none'"
             if args.coord_hint != "none" else ""), flush=True)

    # -- pass 1: turn 1 for every row -------------------------------------
    recs = []
    for start in range(0, len(rows), args.batch_size):
        chunk = rows[start:start + args.batch_size]
        convs, imgs = [], []
        for r in chunk:
            im = Image.open(r["image_path"]).convert("RGB")
            W, H = im.size
            rh, rw = smart_resize(H, W, factor=IMAGE_FACTOR,
                                  min_pixels=min_pixels, max_pixels=max_pixels)
            small = im.resize((rw, rh), Image.BICUBIC)
            r["_wh"] = (W, H)
            r["_rwh"] = (rw, rh)
            convs.append(build_turn1(r["question"], hint))
            imgs.append([small])
        texts, grids = generate_batch(model, processor, convs, imgs,
                                      args.max_new_tokens, args.temperature, device)
        for j, (r, t) in enumerate(zip(chunk, texts)):
            # The frame named "resized" is defined by what the MODEL saw, so it is read
            # back off image_grid_thw rather than trusted from our own smart_resize. If
            # the processor re-resized under us, every frame number would be wrong and
            # nothing would say so.
            if grids is not None:
                gh, gw = int(grids[j][1]), int(grids[j][2])
                r["_proc_rwh"] = (gw * 16, gh * 16)
            recs.append({"row": r, "turn1": t, **turn1_flags(t)})
        if (start + len(chunk)) % 40 < args.batch_size:
            ok = sum(1 for x in recs if x["strict_box"] is not None)
            print(f"  turn1 ...{len(recs)}/{len(rows)}  parsed {ok}", flush=True)

    # -- the frame verdict, from the aggregate ----------------------------
    frame_stats = score_frames(recs)
    winner = frame_stats["winner"]

    # -- pass 2: turn 2, cropping under the winning frame -----------------
    for start in range(0, len(recs), args.batch_size):
        chunk = recs[start:start + args.batch_size]
        convs, imgs = [], []
        for rec in chunk:
            r = rec["row"]
            im = Image.open(r["image_path"]).convert("RGB")
            W, H = r["_wh"]
            rw, rh = r["_rwh"]
            small = im.resize((rw, rh), Image.BICUBIC)
            box = rec["strict_box"]
            mapped = map_box(box, winner, W, H, rw, rh) if box is not None else None
            if mapped is None:
                # ERGO's `crop_with_zoom` falls back to the whole picture when the box does
                # not parse. The reward still scores that rollout 0 -- the fallback is only
                # so turn 2 has an image -- so this is reproduced, not short-circuited.
                mapped = (0.0, 0.0, float(W), float(H))
            x1, y1, x2, y2 = (int(v) for v in mapped)
            cw, ch = smart_resize(x2 - x1, y2 - y1, factor=IMAGE_FACTOR,
                                  min_pixels=min_pixels, max_pixels=max_pixels)
            crop = im.crop((x1, y1, x2, y2)).resize((cw, ch), Image.Resampling.LANCZOS)
            rec["crop_px"] = (x2 - x1) * (y2 - y1)
            full_q = r["question"] + "\n" + "\n".join(r["options"]) + "\n" + r["postfix"]
            convs.append(build_turn2(build_turn1(r["question"], hint), rec["turn1"], full_q))
            imgs.append([small, crop])
        texts, _ = generate_batch(model, processor, convs, imgs,
                                  args.max_new_tokens, args.temperature, device)
        for rec, t in zip(chunk, texts):
            rec["turn2"] = t
            rec["turn2_flags"] = turn2_flags(t)
            rec["pred"] = turn2_pred(t)
        if (start + len(chunk)) % 40 < args.batch_size:
            print(f"  turn2 ...{start + len(chunk)}/{len(recs)}", flush=True)

    res = summarise(model_path, recs, frame_stats, args)
    del model
    torch.cuda.empty_cache()
    return res, recs


def score_frames(recs):
    """Every parseable box, read three ways, scored against the GT box.

    Two kinds of evidence, deliberately kept apart:

      * STRUCTURAL -- where the emitted numbers live. A 0-1000 grid cannot produce a
        coordinate above 1000; a resized-pixel frame cannot produce one above the resized
        side. This needs no ground truth and no model quality, so it survives a checkpoint
        that cannot find the object at all.
      * TASK -- IoU / coverage / centre-hit against the GT box. This is what the reward
        will experience, and it is the tie-breaker, but it is only meaningful if the
        policy is pointing at roughly the right thing in SOME frame.

    They are reported side by side because a disagreement between them is a real finding
    (a model emitting 0-1000 coordinates at the wrong place), not a number to average.
    """
    per = {f: {"iou": [], "cov": [], "hit": 0, "area": [], "cerr": [], "n": 0}
           for f in FRAMES}
    over_1000 = 0
    over_resized = 0
    maxcoords = []
    n_box = 0
    for rec in recs:
        box = rec["strict_box"]
        if box is None:
            continue
        n_box += 1
        r = rec["row"]
        W, H = r["_wh"]
        rw, rh = r["_rwh"]
        mx = max(box)
        maxcoords.append(mx)
        if mx > 1000:
            over_1000 += 1
        if mx > max(rw, rh):
            over_resized += 1
        gt = r["gt_box"]
        for f in FRAMES:
            m = map_box(box, f, W, H, rw, rh)
            if m is None:
                continue
            per[f]["n"] += 1
            per[f]["iou"].append(iou(m, gt))
            per[f]["cov"].append(coverage(m, gt))
            per[f]["hit"] += 1 if center_hit(m, gt) else 0
            per[f]["area"].append(((m[2] - m[0]) * (m[3] - m[1])) / float(W * H))
            # Centre error, normalised by the diagonal. The one statistic here that is
            # not confounded by crop SIZE: a frame read at the wrong scale moves the
            # centre toward the top-left (every coordinate shrinks) and this sees it,
            # where IoU just reads it as "a smaller box that missed".
            dx = ((m[0] + m[2]) - (gt[0] + gt[2])) / 2.0
            dy = ((m[1] + m[3]) - (gt[1] + gt[3])) / 2.0
            per[f]["cerr"].append(math.hypot(dx, dy) / math.hypot(W, H))

    out = {"n_boxes": n_box, "over_1000": over_1000, "over_resized": over_resized,
           "max_coord_p50": statistics.median(maxcoords) if maxcoords else None,
           "max_coord_max": max(maxcoords) if maxcoords else None,
           "frames": {}}
    for f in FRAMES:
        d = per[f]
        out["frames"][f] = {
            "n": d["n"],
            "mean_iou": round(statistics.mean(d["iou"]), 4) if d["iou"] else 0.0,
            "median_iou": round(statistics.median(d["iou"]), 4) if d["iou"] else 0.0,
            "mean_gt_coverage": round(statistics.mean(d["cov"]), 4) if d["cov"] else 0.0,
            "gt_fully_contained": (round(sum(1 for c in d["cov"] if c >= 0.999) / d["n"], 4)
                                   if d["n"] else 0.0),
            "center_hit_rate": round(d["hit"] / d["n"], 4) if d["n"] else 0.0,
            "mean_center_err": round(statistics.mean(d["cerr"]), 4) if d["cerr"] else None,
            "median_crop_area_frac": (round(statistics.median(d["area"]), 4)
                                      if d["area"] else 0.0),
        }
    ranked = sorted(FRAMES, key=lambda f: out["frames"][f]["mean_iou"], reverse=True)
    out["winner"] = ranked[0] if n_box else "rel1000"
    out["runner_up"] = ranked[1] if n_box else None
    out["iou_margin"] = (round(out["frames"][ranked[0]]["mean_iou"]
                               - out["frames"][ranked[1]]["mean_iou"], 4)
                         if n_box else None)
    if not n_box:
        out["winner_note"] = ("no parseable boxes: the frame is UNDECIDED and 'rel1000' "
                              "below is a placeholder so turn 2 could run, not a verdict")
    # The three statistics are not the same measurement and are allowed to disagree. When
    # they do, say so loudly rather than letting the IoU column be read as settled: IoU is
    # size-confounded on V*, whose targets are a fraction of a percent of the picture.
    by_hit = max(FRAMES, key=lambda f: out["frames"][f]["center_hit_rate"])
    by_err = min(FRAMES, key=lambda f: (out["frames"][f]["mean_center_err"]
                                        if out["frames"][f]["mean_center_err"] is not None
                                        else 1e9))
    if n_box and (by_hit != out["winner"] or by_err != out["winner"]):
        out["disagreement"] = (f"IoU picks {out['winner']}, centre-hit picks {by_hit}, "
                               f"centre-error picks {by_err}")
    return out


def summarise(model_path, recs, frame_stats, args):
    n = len(recs)
    strict = sum(1 for r in recs if r["strict_box"] is not None)
    lenient = sum(1 for r in recs if r["lenient_box"] is not None)
    tagged = sum(1 for r in recs if r["zoom_tag"])
    think1 = sum(1 for r in recs if r["think_ok"])
    fmt1 = sum(1 for r in recs if r["format_ok"])
    t2s = sum(1 for r in recs if r.get("turn2_flags", {}).get("strict"))
    t2l = sum(1 for r in recs if r.get("turn2_flags", {}).get("lenient"))
    t2a = sum(1 for r in recs if r.get("turn2_flags", {}).get("answer_tag"))
    both = sum(1 for r in recs
               if r["format_ok"] and r.get("turn2_flags", {}).get("strict"))
    graded = [r for r in recs if r.get("pred")]
    acc = (sum(1 for r in graded if r["pred"] == r["row"]["label"]) / n) if n else 0.0
    areas = [r["crop_px"] / float(r["row"]["_wh"][0] * r["row"]["_wh"][1])
             for r in recs if r["strict_box"] is not None and "crop_px" in r]
    proc_mismatch = sum(1 for r in recs
                        if r["row"].get("_proc_rwh") and r["row"]["_proc_rwh"] != r["row"]["_rwh"])

    res = {
        "model": str(model_path),
        "n": n,
        "vision_tokens": args.vision_tokens,
        "coord_hint": args.coord_hint,
        "temperature": args.temperature,
        "zoom_parse_strict": round(strict / n, 4) if n else 0.0,
        "zoom_parse_lenient": round(lenient / n, 4) if n else 0.0,
        "zoom_tag_present": round(tagged / n, 4) if n else 0.0,
        "turn1_think_ok": round(think1 / n, 4) if n else 0.0,
        "turn1_format_ok": round(fmt1 / n, 4) if n else 0.0,
        "turn2_format_strict": round(t2s / n, 4) if n else 0.0,
        "turn2_format_lenient": round(t2l / n, 4) if n else 0.0,
        "turn2_answer_tag": round(t2a / n, 4) if n else 0.0,
        "both_turns_format": round(both / n, 4) if n else 0.0,
        "turn2_letter_acc": round(acc, 4),
        "median_crop_area_frac": round(statistics.median(areas), 4) if areas else None,
        "mean_crop_area_frac": round(statistics.mean(areas), 4) if areas else None,
        "crop_area_le_0.6": (round(sum(1 for a in areas if a <= 0.6) / len(areas), 4)
                             if areas else None),
        "processor_resized_under_us": proc_mismatch,
        "frames": frame_stats,
    }

    f = frame_stats
    print(f"\n  turn 1")
    print(f"    <zoom> tag present      {res['zoom_tag_present']:.1%}")
    print(f"    zoom box parses (ERGO)  {res['zoom_parse_strict']:.1%}   "
          f"(gate: >= {MIN_ZOOM_PARSE:.0%})")
    print(f"    zoom box parses lenient {res['zoom_parse_lenient']:.1%}   "
          "(floats / outside the tag / several boxes)")
    print(f"    <think>...</think> once {res['turn1_think_ok']:.1%}")
    print(f"    full turn-1 format      {res['turn1_format_ok']:.1%}")
    print(f"  turn 2")
    print(f"    <think>..</think><answer>..</answer>  {res['turn2_format_strict']:.1%}   "
          f"(gate: >= {MIN_TURN2_FORMAT:.0%})")
    print(f"    lenient (one <answer> pair, any think) {res['turn2_format_lenient']:.1%}")
    print(f"    <answer> tags at all                   {res['turn2_answer_tag']:.1%}")
    print(f"  BOTH turns well-formed    {res['both_turns_format']:.1%}   "
          "<- r_format, and the floor under every other reward")
    print(f"  turn-2 letter accuracy    {res['turn2_letter_acc']:.1%}   (context only)")
    print(f"\n  coordinate frame, over {f['n_boxes']} parsed boxes")
    print(f"    coords above 1000:          {f['over_1000']}  "
          "(impossible if the model is on a 0-1000 grid)")
    print(f"    coords above the resized side: {f['over_resized']}  "
          "(impossible if the model is in resized pixels)")
    print(f"    max coordinate, median {f['max_coord_p50']}   max {f['max_coord_max']}")
    print(f"    {'frame':<10} {'n':>4} {'meanIoU':>8} {'medIoU':>7} {'GTcover':>8} "
          f"{'GTinside':>9} {'ctr-hit':>8} {'ctr-err':>8} {'areafrac':>9}")
    for name in FRAMES:
        d = f["frames"][name]
        mark = "  <-- winner" if name == f["winner"] else ""
        cerr = d["mean_center_err"]
        print(f"    {name:<10} {d['n']:>4} {d['mean_iou']:>8.4f} {d['median_iou']:>7.4f} "
              f"{d['mean_gt_coverage']:>8.3f} {d['gt_fully_contained']:>9.3f} "
              f"{d['center_hit_rate']:>8.3f} "
              f"{('%8.3f' % cerr) if cerr is not None else '       -'} "
              f"{d['median_crop_area_frac']:>9.3f}{mark}")
    print(f"    winner {f['winner']} over {f['runner_up']} by {f['iou_margin']} mean IoU")
    if f.get("winner_note"):
        print(f"    NOTE: {f['winner_note']}")
    if f.get("disagreement"):
        print(f"    DISAGREEMENT: {f['disagreement']}\n"
              "      IoU is confounded by crop size on V* (the targets are tiny); "
              "centre-error is not.\n"
              "      Do not port a frame into the trainer until these three agree.")
    print(f"\n  median crop area / original  {res['median_crop_area_frac']}   "
          f"(ERGO's gamma is 0.6; {res['crop_area_le_0.6']} of crops are under it)")
    if proc_mismatch:
        print(f"\n  WARNING: the processor re-resized {proc_mismatch}/{n} images under us. "
              "The 'resized' frame is then not the frame the model saw -- fix the "
              "min/max pixel plumbing before trusting any number above.")
    return res


def dump_examples(recs, k=3):
    print("\n  first completions, verbatim -- a rate cannot tell 'ignored the tag' from "
          "'wrote the box bare':")
    shown_ok, shown_bad = 0, 0
    for rec in recs:
        ok = rec["strict_box"] is not None
        if ok and shown_ok < k:
            shown_ok += 1
        elif not ok and shown_bad < k:
            shown_bad += 1
        else:
            continue
        tag = "PARSED" if ok else "NO BOX"
        print(f"\n    [{tag}] turn1: {rec['turn1'][:400]!r}")
        print(f"             turn2: {rec.get('turn2', '')[:300]!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    help="the checkpoint under test; for this gate that is the cold start")
    ap.add_argument("--reference", default=None,
                    help="a second model measured alongside. Qwen3-VL-8B-Instruct is the "
                         "one to use: it is the backbone the cold start departs from, and "
                         "its grounding ability is what the frame question is about.")
    ap.add_argument("--vstar-root", default=None,
                    help="craigwu/vstar_bench snapshot dir; found in HF_HOME by default")
    ap.add_argument("--n", type=int, default=0, help="0 = all 191 V* rows")
    ap.add_argument("--vision-tokens", type=int, default=DEFAULT_VISION_TOKENS,
                    help="ERGO's --max_vision_token_num. Converted at 32x32 px per token, "
                         "not 28x28 -- see the module header.")
    ap.add_argument("--coord-hint", default="none", choices=sorted(COORD_HINTS),
                    help="append a coordinate-convention sentence to ERGO's turn-1 "
                         "template. Leave at 'none' for the headline measurement.")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="1.0 to match GRPO sampling; 0 for greedy, which overstates "
                         "format compliance")
    ap.add_argument("--ergo-src", default="/home/uberger/scratch/research/ERGO/src/ergo",
                    help="the ERGO checkout the vendored prompts are diffed against")
    ap.add_argument("--out", default=None, help="write the result dicts here as JSON")
    ap.add_argument("--dump-completions", default=None,
                    help="write every turn-1/turn-2 completion here as JSONL")
    ap.add_argument("--min-zoom-parse", type=float, default=MIN_ZOOM_PARSE)
    ap.add_argument("--min-turn2-format", type=float, default=MIN_TURN2_FORMAT)
    args = ap.parse_args()

    _check_ergo_sources(args.ergo_src)

    root = args.vstar_root
    if root is None:
        hf = os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface"))
        cands = sorted(Path(hf).glob("hub/datasets--craigwu--vstar_bench/snapshots/*"))
        if not cands:
            raise SystemExit(
                "ergo_zoom_gate: craigwu/vstar_bench is not in HF_HOME. Fetch it with\n"
                "  python -c \"from huggingface_hub import snapshot_download as s; "
                "print(s('craigwu/vstar_bench', repo_type='dataset'))\"\n"
                "or pass --vstar-root.")
        root = cands[-1]
    rows = load_vstar(root, args.n)
    print(f"{len(rows)} V* rows from {root}, sampled at temperature {args.temperature}")

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU visible. This will be very slow.", file=sys.stderr)

    results, dumps = [], []
    for path in [args.model] + ([args.reference] if args.reference else []):
        res, recs = run_one(path, [dict(r) for r in rows], args, device)
        dump_examples(recs)
        results.append(res)
        if args.dump_completions:
            for rec in recs:
                dumps.append({
                    "model": str(path),
                    "question_id": rec["row"]["question_id"],
                    "category": rec["row"]["category"],
                    "image": rec["row"]["image_path"],
                    "wh": rec["row"]["_wh"], "resized_wh": rec["row"]["_rwh"],
                    "gt_box": rec["row"]["gt_box"],
                    "turn1": rec["turn1"], "turn2": rec.get("turn2"),
                    "strict_box": rec["strict_box"], "lenient_box": rec["lenient_box"],
                    "pred": rec.get("pred"), "label": rec["row"]["label"],
                })

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.out}")
    if args.dump_completions:
        Path(args.dump_completions).parent.mkdir(parents=True, exist_ok=True)
        with open(args.dump_completions, "w") as fh:
            for d in dumps:
                fh.write(json.dumps(d) + "\n")
        print(f"wrote {args.dump_completions}")

    got = results[0]
    reasons = []
    if got["zoom_parse_strict"] < args.min_zoom_parse:
        reasons.append(f"zoom-parse rate {got['zoom_parse_strict']:.1%} < "
                       f"{args.min_zoom_parse:.0%}")
    if got["turn2_format_strict"] < args.min_turn2_format:
        reasons.append(f"turn-2 format {got['turn2_format_strict']:.1%} < "
                       f"{args.min_turn2_format:.0%}")
    print(f"\n{'PASS' if not reasons else 'FAIL'}: {got['model']}")
    for r in reasons:
        print(f"  - {r}")
    if reasons:
        print("""
  A two-turn ERGO cold start is needed before any GRPO run is worth starting:
  ~2-5k SFT rows in ERGO's own format, turn-1 boxes taken from the GT annotations
  already in cold_data/grpo_sets (`_boxes_abs` / `union_bbox` in build_grpo_sets.py).

  THAT IS +3-5 DAYS AND IT IS THE SINGLE THING THAT BLOWS THIS ARM'S BUDGET.
  Say so before starting it. Without it the symptom in GRPO is not an error: every
  reward in the group is 0, the advantage is exactly 0, and the run trains nothing
  while the loss curve looks ordinary.""")
    return 0 if not reasons else 1


if __name__ == "__main__":
    raise SystemExit(main())
