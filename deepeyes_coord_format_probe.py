#!/usr/bin/env python
"""Which coordinate frame does Qwen3-VL emit a DeepEyes zoom box in, and can it hit the tags?

    # one GPU, one model, one arm -- the shape every shard runs
    python deepeyes_coord_format_probe.py --stage generate --out-dir DIR \\
        --model checkpoint/coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged
    python deepeyes_coord_format_probe.py --stage report --out-dir DIR

    # the whole thing, sharded over a node
    bash launch_deepeyes_coord_format.sh --out-dir DIR

WHY THIS EXISTS

`docs/deepeyes-build-prompts.md` Prompt 4. Two numbers that Prompt 3 (the zoom tool) and
Prompt 5 (the rewards) both read, and that are expensive to discover late because neither
announces itself: a wrong coordinate frame crops the wrong region on every call while the
loss curve stays healthy, and a format rate near zero means `R_format = -1` on nearly
every rollout while `R_tool` -- which pays out only on a correct answer -- never fires.

THE COORDINATE QUESTION, AND WHY IT IS NOT SETTLED BY READING THE COOKBOOK

DeepEyes' tool reads `bbox_2d` and crops the ORIGINAL image with it as absolute pixels
(`visual_toolbox_v2.VisualToolBoxV2.reset` stores `origin_multi_modal_data` and takes
`self.width/self.height` from it). That is right for Qwen2.5-VL. Qwen3-VL's cookbook says
grounding output is a 0-1000 relative grid against the original image. Both are claims
about a model under ITS OWN prompt; what the policy does under DEEPEYES' prompt is a third
thing, and it is the only one that matters here. So this measures it.

Three readings of the same four numbers, scored against V*'s ground-truth boxes:

    rel1000      x * W/1000, y * H/1000           Qwen3-VL's documented grounding frame
    abs_orig     x, y                             DeepEyes' tool's assumption
    abs_resized  x * W/rW, y * H/rH               post-smart_resize pixels, Qwen2.5-VL's

`rW, rH` are read back out of `image_grid_thw` rather than recomputed, so they are the
frame the model actually saw and not this file's idea of it.

TWO ARMS, BECAUSE IoU ALONE CANNOT SEPARATE ALL THREE

Under Qwen3-VL's own processor defaults (`size.longest_edge = 16777216`, i.e. max_pixels
16.7 MP) a V* photo is not downscaled at all, so `abs_resized` and `abs_orig` coincide to
within the 32 px rounding and no IoU can tell them apart. Arm B forces a real downscale by
capping max_pixels, which turns the question into an INVARIANCE test and answers it
cleanly:

  * if the model emits a 0-1000 relative grid, the raw numbers barely move between arms;
  * if it emits post-resize absolute pixels, they shrink by the resize factor;
  * if it emits original-image pixels, they also barely move -- but then `abs_orig`, not
    `rel1000`, is the reading with the high IoU in BOTH arms.

So: arm A says relative-or-absolute (the IoU gap is ~a factor of two in each linear
dimension, which IoU resolves decisively); arm B says which absolute frame.

THE FORMAT QUESTION, AND THE TRAP IN IT

`R_format` is **-1 on any tag mismatch**, and `vl_agent.compute_score` defines "mismatch"
as COUNTS NOT BALANCING, not as the structure being present:

    count("<think>") != count("</think>")                                   -> error
    count("<|vision_start|><|image_pad|>") != count("<|image_pad|><|vision_end|>")
    count("<answer>") != count("</answer>")   (after the last </think>)     -> error
    len(answer_text) >= 1000                                                -> error

A completion with NO tags at all balances perfectly (0 == 0) and scores format 0.0, no
penalty -- and also never zooms, so `R_tool` never fires and the arm is plain GRPO. A
format rate measured against their literal rule therefore OVERSTATES how well the model
has learned their protocol. This script reports both:

    format_ok_deepeyes   their rule, verbatim -- what decides the -1
    tag_structure_ok     one <think></think> pair AND a closing <tool_call> or <answer>
    tool_call_rate       a parseable zoom call with four numbers -- what gates R_tool

WHAT IT IS NOT: an accuracy measurement, and not a multi-turn rollout. Only the FIRST
assistant turn is generated. The coordinate question is entirely a turn-1 question, and
in a real rollout the vision-token clause of the format rule is satisfied by construction
because the tool splice, not the model, writes those tokens.

Sampled at temperature 1.0 by default, not greedy, because that is what GRPO rolls out at
and greedy overstates format compliance. `coldstart_gate.py` is the precedent for all of
the above.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent

# --------------------------------------------------------------------------------------
# Their prompt, verbatim.
#
# Copied from `eval/eval_vstar.py` in Visual-Agent/DeepEyes, which is character-identical
# to `verl/workers/agent/envs/mm_process_engine/prompt.py:PROMPT.SYSTEM_PROMPT_V2` and
# `.USER_PROMPT_V2` -- the pair the released 47k corpus was rolled out under. Using our
# own paraphrase here would measure a prompt DeepEyes never ran, which is the one thing
# this probe must not do. `_check_prompt_matches_upstream` re-reads both out of a local
# clone when there is one and refuses to run if they have drifted.
#
# Note the two faults that are theirs and are kept: `"required":["bbox"]` names a
# parameter that does not exist (the parameter is `bbox_2d`), and the trailing
# double-spaces in USER_PROMPT_V2 are load-bearing only in the sense that they are what
# the model was trained under.
# --------------------------------------------------------------------------------------
#
# Written as a tuple of per-line literals, not a triple-quoted block, for one reason: the
# last three lines END IN TWO SPACES in their source, and an editor or a pre-commit hook
# that strips trailing whitespace would silently change the prompt. Inside quotes it
# cannot be stripped by accident, and `_check_prompt_matches_upstream` catches it if it is
# changed on purpose.
SYSTEM_PROMPT = (
    'You are a helpful assistant.\n'
    '\n'
    '# Tools\n'
    'You may call one or more functions to assist with the user query.\n'
    'You are provided with function signatures within <tools></tools> XML tags:\n'
    '<tools>\n'
    '{"type":"function","function":{"name":"image_zoom_in_tool","description":"Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) and an optional object label.","parameters":{"type":"object","properties":{"bbox_2d":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4,"description":"The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner."},"label":{"type":"string","description":"The name or label of the object in the specified bounding box (optional)."}},"required":["bbox"]}}}\n'  # noqa: E501
    '</tools>\n'
    '\n'
    '# How to call a tool\n'
    'Return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n'  # noqa: E501
    '<tool_call>\n'
    '{"name": <function-name>, "arguments": <args-json-object>}\n'
    '</tool_call>\n'
    '\n'
    '**Example**:  \n'
    '<tool_call>  \n'
    '{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [10, 20, 100, 200], "label": "the apple on the desk"}}  \n'  # noqa: E501
    '</tool_call>'
)

USER_PROMPT = "\nThink first, call **image_zoom_in_tool** if needed, then answer. Format strictly as:  <think>...</think>  <tool_call>...</tool_call> (if tools needed)  <answer>...</answer> "  # noqa: E501

USER_TEMPLATE = """Question: {question}
Options: {options}
""" + USER_PROMPT

#: A clone of Visual-Agent/DeepEyes, reference only. Absent is fine -- the check is skipped
#: with a warning rather than failing, because this probe has to run on a node with no
#: network. Present and DIFFERENT is a hard stop.
UPSTREAM = Path(os.environ.get("DEEPEYES_REF",
                               "/home/uberger/scratch/deepeyes_ref/DeepEyes"))

#: V* bench, as `eval/eval_vstar.py` expects it on disk: one .jpg and one .json per item,
#: under two scored subdirectories. The .json carries `bbox` as [x, y, w, h] in ORIGINAL
#: image pixels, which is the ground truth the three readings are scored against.
VSTAR_SPLITS = ("direct_attributes", "relative_position")
VSTAR_ROOT = Path(os.environ.get(
    "VSTAR_BENCH_PATH",
    "/home/uberger/scratch/cache/hf_cache/hub/datasets--craigwu--vstar_bench/"
    "snapshots/d9ae62c903da0c98336e85c5ee89cd863b04b4da"))

ABC = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F"}

#: The three candidate frames. Each maps an emitted box to ORIGINAL image pixels, which is
#: the frame V*'s ground truth lives in.
READINGS = ("rel1000", "abs_orig", "abs_resized")

#: Arm B's cap. 2048 * 28 * 28 = 1,605,632 -- verl's and Qwen2.5-VL's classic default, and
#: roughly a 1.5x linear downscale on a V* photo, which is far more than the 32 px
#: rounding that separates `abs_orig` from `abs_resized` under arm A.
ARM_B_MAX_PIXELS = 2048 * 28 * 28


# --------------------------------------------------------------------------------------
# Their format rule, verbatim.
# --------------------------------------------------------------------------------------
def deepeyes_is_format_error(predict_str: str) -> bool:
    """`vl_agent.compute_score`'s `is_format_error`, line for line.

    Reproduced rather than approximated: this boolean is the whole of `R_format`, and
    `R_format` is -1, so a paraphrase that is 5% off reports a penalty rate that is 5%
    off on every rollout of an 80-step run.
    """
    is_format_error = False
    if predict_str.count("<think>") != predict_str.count("</think>"):
        is_format_error = True
    if (predict_str.count("<|vision_start|><|image_pad|>")
            != predict_str.count("<|image_pad|><|vision_end|>")):
        is_format_error = True
    predict_no_think = predict_str.split("</think>")[-1].strip()
    if predict_no_think.count("<answer>") != predict_no_think.count("</answer>"):
        is_format_error = True
    # Their own anti-judge-hacking clause, which doubles as a format error. It bites a
    # long chain that never opens <answer>, because then `answer_text` is the WHOLE
    # completion.
    answer_text = predict_str.split("<answer>")[-1].split("</answer>")[0].strip()
    if len(answer_text) >= 1000:
        is_format_error = True
    return is_format_error


def _upstream_string_consts(path: Path):
    """Every module- or class-level `NAME = "literal"` in a file, as a dict.

    Parsed rather than grepped because `prompt.py` carries a COMMENTED-OUT earlier
    `SYSTEM_PROMPT_V2` directly above the live one, and a regex happily returns the dead
    one -- which would let a drifted prompt pass this check.
    """
    out = {}
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.Assign):
            continue
        if not (isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)):
            continue
        for t in node.targets:
            if isinstance(t, ast.Name):
                out[t.id] = node.value.value
    return out


def _check_prompt_matches_upstream():
    """Refuse to run if the embedded prompt is not the one in the local clone.

    Trailing whitespace and the `"required":["bbox"]` typo are part of the prompt. This is
    the check that keeps a tidied copy from quietly becoming a different experiment.
    """
    ev = UPSTREAM / "eval" / "eval_vstar.py"
    pr = UPSTREAM / "verl" / "workers" / "agent" / "envs" / "mm_process_engine" / "prompt.py"
    if not ev.exists():
        print(f"WARNING: no DeepEyes clone at {UPSTREAM}; the prompt is unverified. "
              "Set DEEPEYES_REF to a clone to check it.", file=sys.stderr)
        return
    want = [(ev, "instruction_prompt_system", SYSTEM_PROMPT),
            (ev, "USER_PROMPT_V2", USER_PROMPT)]
    if pr.exists():
        # Their eval script and their training env must also still agree with each other;
        # if they diverge, which one we are measuring becomes a choice to make out loud.
        want += [(pr, "SYSTEM_PROMPT_V2", SYSTEM_PROMPT),
                 (pr, "USER_PROMPT_V2", USER_PROMPT)]
    for path, name, embedded in want:
        got = _upstream_string_consts(path).get(name)
        if got is None:
            raise SystemExit(f"deepeyes_coord_format_probe: no {name} in {path}")
        if got != embedded:
            raise SystemExit(
                f"deepeyes_coord_format_probe: {name} in {path} is not the string this "
                "file embeds. Measuring under a prompt DeepEyes never ran measures "
                f"nothing.\n  upstream: {got!r}\n  here:     {embedded!r}")
    # The format rule is transcribed too, so check the source it came from.
    vl = UPSTREAM / "verl" / "utils" / "reward_score" / "vl_agent.py"
    if vl.exists():
        vsrc = vl.read_text()
        for needle in ('count_vision_1 = predict_str.count("<|vision_start|><|image_pad|>")',
                       'if len(answer_text) >= 1000:',
                       'format_reward = -1.0 if is_format_error else 0.0'):
            if needle not in vsrc:
                raise SystemExit(
                    f"deepeyes_coord_format_probe: {vl} no longer contains {needle!r}. "
                    "Their format rule moved; update deepeyes_is_format_error to match "
                    "rather than scoring the old one.")


# --------------------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------------------
def xywh_to_xyxy(b):
    x, y, w, h = b
    return [float(x), float(y), float(x) + float(w), float(y) + float(h)]


def iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    aa = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    ba = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = aa + ba - inter
    return inter / union if union > 0 else 0.0


def covered(gt, box):
    """Fraction of the GT box the emitted box contains.

    Reported alongside IoU because a ZOOM box is allowed to be much bigger than its
    target -- a crop that swallows the object is a success and an IoU of 0.2. If the two
    metrics disagree about which reading wins, the frame question is not settled and the
    disagreement belongs in the writeup rather than being averaged away.
    """
    gx0, gy0, gx1, gy1 = gt
    bx0, by0, bx1, by1 = box
    iw = max(0.0, min(gx1, bx1) - max(gx0, bx0))
    ih = max(0.0, min(gy1, by1) - max(gy0, by0))
    ga = max(0.0, gx1 - gx0) * max(0.0, gy1 - gy0)
    return (iw * ih) / ga if ga > 0 else 0.0


def convert(box, reading, W, H, rW, rH):
    """An emitted [x1,y1,x2,y2] -> original-image pixels, under one reading."""
    x0, y0, x1, y1 = (float(v) for v in box)
    if reading == "rel1000":
        return [x0 * W / 1000.0, y0 * H / 1000.0, x1 * W / 1000.0, y1 * H / 1000.0]
    if reading == "abs_orig":
        return [x0, y0, x1, y1]
    if reading == "abs_resized":
        return [x0 * W / rW, y0 * H / rH, x1 * W / rW, y1 * H / rH]
    raise ValueError(reading)


def out_of_frame(box, W, H):
    """Fraction of the box's area that falls outside the image.

    A reading that puts half of every box off the picture is wrong independently of any
    ground truth, which makes this the one frame signal that survives a model whose boxes
    are simply in the wrong PLACE.
    """
    x0, y0, x1, y1 = box
    area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if area <= 0:
        return 1.0
    iw = max(0.0, min(x1, W) - max(x0, 0.0))
    ih = max(0.0, min(y1, H) - max(y0, 0.0))
    return 1.0 - (iw * ih) / area


# --------------------------------------------------------------------------------------
# Parsing what the model emitted
# --------------------------------------------------------------------------------------
TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
#: The loose fallback. Counted separately and never used for the IoU table: a model that
#: writes the box but not the tags produces a number here and ZERO tool calls in verl,
#: and conflating the two would hide exactly the failure Prompt 9 is told to watch for.
LOOSE_BBOX_RE = re.compile(r"\"?bbox(?:_2d)?\"?\s*:\s*\[([^\]]*)\]")


def parse_tool_calls(text):
    """Every well-formed `<tool_call>` block, as dicts. Their `extract_action` is laxer.

    `visual_toolbox_v2.execute` takes the LAST block and `json.loads` it; `eval_vstar.py`
    takes the FIRST and `eval()`s it. We keep all of them and score the first, because the
    first zoom is the one whose placement the method depends on.
    """
    out = []
    for blob in TOOL_CALL_RE.findall(text):
        blob = blob.strip()
        for loader in (json.loads, ast.literal_eval):
            try:
                obj = loader(blob)
            except Exception:
                continue
            if isinstance(obj, dict):
                out.append(obj)
            break
    return out


def first_bbox(calls):
    """The first four-number `bbox_2d` in a list of parsed tool calls, or None."""
    for obj in calls:
        args = obj.get("arguments") if isinstance(obj.get("arguments"), dict) else obj
        for key in ("bbox_2d", "bbox"):
            b = args.get(key) if isinstance(args, dict) else None
            if isinstance(b, (list, tuple)) and len(b) == 4:
                try:
                    return [float(v) for v in b]
                except (TypeError, ValueError):
                    continue
    return None


def loose_bbox(text):
    m = LOOSE_BBOX_RE.search(text)
    if not m:
        return None
    try:
        vals = [float(v.strip()) for v in m.group(1).split(",")]
    except ValueError:
        return None
    return vals if len(vals) == 4 else None


def tag_structure_ok(text):
    """One `<think></think>` pair, closed by exactly one tool call or exactly one answer.

    The strict reading of their protocol, and the one that predicts whether `R_tool` can
    ever fire. Their own rule (`deepeyes_is_format_error`) passes a bare untagged answer.
    """
    if text.count("<think>") != 1 or text.count("</think>") != 1:
        return False
    tail = text.split("</think>")[-1]
    n_tool = len(TOOL_CALL_RE.findall(tail))
    n_ans_open, n_ans_close = tail.count("<answer>"), tail.count("</answer>")
    has_answer = n_ans_open == 1 and n_ans_close == 1
    return bool(n_tool >= 1 or has_answer)


# --------------------------------------------------------------------------------------
# Corpus
# --------------------------------------------------------------------------------------
def load_vstar(root: Path):
    """V* rows, exactly as `eval/eval_vstar.py` builds them, plus the GT boxes.

    HR-Bench is NOT here on purpose: its parquet carries `index/question/A..D/answer/
    category/image`, and no box. The coordinate half therefore rests on V* alone rather
    than on a label we would have had to invent; the format half could use either and
    does not need the extra 1,000 rows.
    """
    rows = []
    for split in VSTAR_SPLITS:
        d = root / split
        if not d.is_dir():
            raise SystemExit(f"deepeyes_coord_format_probe: no V* split at {d}")
        for jpg in sorted(d.glob("*.jpg")):
            meta = json.loads((d / (jpg.stem + ".json")).read_text())
            opts = meta["options"]
            option_str = "\n" + "".join(f"{ABC[i]}. {o}\n" for i, o in enumerate(opts))
            rows.append({
                "uid": f"{split}/{jpg.stem}",
                "split": split,
                "image_path": str(jpg),
                "question": meta["question"],
                "options": option_str,
                "gt_boxes": [xywh_to_xyxy(b) for b in meta["bbox"]],
                "target_object": meta.get("target_object", []),
            })
    return rows


# --------------------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------------------
def _assert_arm_b_downscales(processor, image_path, proc_kw, patch, tag, min_ratio=1.2):
    """Arm B must actually make the picture smaller, or it is arm A run twice.

    The gate exists because the first attempt at this override passed `max_pixels` alone
    and changed nothing: the grid came back identical and the "measurement" would have
    reported two frames that are the same frame. Nothing in the output would have looked
    wrong -- `abs_orig` and `abs_resized` would simply have tied, which is also what a
    genuine result looks like under arm A.
    """
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    text = ("<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>x<|im_end|>\n"
            "<|im_start|>assistant\n")

    def grid(**kw):
        got = processor(text=[text], images=[img], return_tensors="pt",
                        add_special_tokens=False, **kw)
        return tuple(int(v) for v in got["image_grid_thw"][0][1:])

    a_h, a_w = grid()
    b_h, b_w = grid(**proc_kw)
    ratio = a_w / b_w if b_w else 0.0
    print(f"[{tag}] arm-B gate: {img.size} -> arm A {(a_h * patch, a_w * patch)} "
          f"vs arm B {(b_h * patch, b_w * patch)}  ({ratio:.3f}x linear)", flush=True)
    if ratio < min_ratio:
        raise SystemExit(
            f"deepeyes_coord_format_probe: arm B is not downscaling ({ratio:.3f}x < "
            f"{min_ratio}). The processor is ignoring {proc_kw}, so arm B would be arm A "
            "under a different name and the two absolute readings could never separate.")


def run_generate(args):
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    out_dir = Path(args.out_dir)
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    tag = f"{args.model_tag}__{args.arm}__shard{args.shard}of{args.num_shards}"
    sink = out_dir / "raw" / f"{tag}.jsonl"

    done = set()
    if sink.exists() and not args.overwrite:
        for line in sink.read_text().splitlines():
            try:
                done.add(json.loads(line)["uid"])
            except Exception:
                pass

    rows = load_vstar(VSTAR_ROOT)
    if args.n:
        rows = rows[: args.n]
    rows = [r for i, r in enumerate(rows) if i % args.num_shards == args.shard]
    todo = [r for r in rows if r["uid"] not in done]
    print(f"[{tag}] {len(todo)} of {len(rows)} rows to do "
          f"({len(done)} already in {sink.name})", flush=True)
    if not todo:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa").to(device).eval()

    size = getattr(processor.image_processor, "size", None)
    min_px = int(getattr(size, "shortest_edge", None) or size["shortest_edge"])
    proc_kw = {}
    if args.arm == "B":
        # BOTH bounds, and this is not optional. The fast image processor's
        # `_preprocess` only honours them under `if min_pixels is not None and
        # max_pixels is not None` -- pass max_pixels alone and the override is a SILENT
        # NO-OP that leaves arm B identical to arm A. Measured, not assumed:
        # `_assert_arm_b_downscales` below refuses to run if it ever stops working.
        proc_kw["min_pixels"] = min_px
        proc_kw["max_pixels"] = args.arm_b_max_pixels
    print(f"[{tag}] arm {args.arm}  proc_kw={proc_kw or '{}'}  "
          f"default size={size}", flush=True)

    patch = int(getattr(processor.image_processor, "patch_size", 16))
    if args.arm == "B":
        _assert_arm_b_downscales(processor, todo[0]["image_path"], proc_kw, patch, tag)
    t0 = time.time()
    with sink.open("a") as fh:
        for i, row in enumerate(todo):
            img = Image.open(row["image_path"]).convert("RGB")
            W, H = img.size
            prompt = USER_TEMPLATE.format(question=row["question"],
                                          options=row["options"])
            msgs = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": prompt}]},
            ]
            text = processor.apply_chat_template(msgs, tokenize=False,
                                                 add_generation_prompt=True)
            inputs = processor(text=[text], images=[img], return_tensors="pt",
                               padding=True, padding_side="left",
                               add_special_tokens=False, **proc_kw).to(device)
            # The frame the model SAW, read back rather than recomputed. grid_thw counts
            # patch-size units, so this is the post-smart_resize pixel size exactly.
            gh, gw = (int(v) for v in inputs["image_grid_thw"][0][1:])
            rH, rW = gh * patch, gw * patch

            gen_kw = dict(max_new_tokens=args.max_new_tokens)
            if args.temperature > 0:
                gen_kw.update(do_sample=True, temperature=args.temperature, top_p=1.0)
            else:
                gen_kw.update(do_sample=False)
            with torch.no_grad():
                out = model.generate(**inputs, **gen_kw)
            comp = out[0][inputs["input_ids"].shape[1]:]
            completion = processor.tokenizer.decode(
                comp, skip_special_tokens=False).strip()
            # The literal special tokens matter to their format rule, but the end-of-turn
            # marker is scaffolding rather than output.
            for eot in ("<|im_end|>", "<|endoftext|>"):
                completion = completion.replace(eot, "")
            completion = completion.strip()

            rec = dict(row)
            rec.pop("options", None)
            rec.update({
                "model": args.model_tag, "arm": args.arm,
                "W": W, "H": H, "rW": rW, "rH": rH,
                "n_prompt_tokens": int(inputs["input_ids"].shape[1]),
                "n_completion_tokens": int(comp.shape[0]),
                "completion": completion,
            })
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if (i + 1) % 10 == 0:
                el = time.time() - t0
                print(f"[{tag}] {i + 1}/{len(todo)}  {el / (i + 1):.1f}s/row", flush=True)
    print(f"[{tag}] done in {(time.time() - t0) / 60:.1f} min -> {sink}", flush=True)


# --------------------------------------------------------------------------------------
# Scoring and the report
# --------------------------------------------------------------------------------------
def score_record(rec):
    """Everything derivable from one completion, with no model in the room.

    Split out from generation so the report can be re-derived -- and the CPU test
    exercised -- without a GPU.
    """
    text = rec["completion"]
    calls = parse_tool_calls(text)
    box = first_bbox(calls)
    loose = loose_bbox(text) if box is None else None
    W, H, rW, rH = rec["W"], rec["H"], rec["rW"], rec["rH"]

    res = {
        "uid": rec["uid"], "split": rec["split"], "model": rec["model"],
        "arm": rec["arm"], "W": W, "H": H, "rW": rW, "rH": rH,
        "format_ok_deepeyes": not deepeyes_is_format_error(text),
        "tag_structure_ok": tag_structure_ok(text),
        "n_tool_calls": len(calls),
        "tool_call_ok": box is not None,
        "loose_bbox_only": loose is not None,
        "has_think": text.count("<think>") == 1 and text.count("</think>") == 1,
        "has_answer_tags": "<answer>" in text and "</answer>" in text,
        "n_completion_tokens": rec.get("n_completion_tokens"),
        "raw_bbox": box,
        "readings": {},
    }
    if box is None:
        return res
    # Scale-free summary of the raw numbers. This is what separates the two arms: a
    # relative grid keeps `raw_long_frac` fixed when max_pixels moves, an absolute one
    # does not.
    res["raw_max"] = max(box)
    res["raw_w"] = abs(box[2] - box[0])
    res["raw_h"] = abs(box[3] - box[1])
    for reading in READINGS:
        conv = convert(box, reading, W, H, rW, rH)
        gts = rec["gt_boxes"]
        res["readings"][reading] = {
            "box": [round(v, 1) for v in conv],
            "iou": max(iou(g, conv) for g in gts),
            "covered": max(covered(g, conv) for g in gts),
            "oof": out_of_frame(conv, W, H),
            "area_frac": (max(0.0, conv[2] - conv[0]) * max(0.0, conv[3] - conv[1]))
                         / float(W * H),
        }
    return res


def _mean(xs):
    return statistics.mean(xs) if xs else 0.0


def _median(xs):
    return statistics.median(xs) if xs else 0.0


def summarise(scored):
    """One (model, arm) cell of the report."""
    n = len(scored)
    boxed = [s for s in scored if s["tool_call_ok"]]
    cell = {
        "n": n,
        "format_ok_deepeyes": _mean([s["format_ok_deepeyes"] for s in scored]),
        "tag_structure_ok": _mean([s["tag_structure_ok"] for s in scored]),
        "tool_call_rate": _mean([s["tool_call_ok"] for s in scored]),
        "loose_bbox_only_rate": _mean([s["loose_bbox_only"] for s in scored]),
        "has_think_rate": _mean([s["has_think"] for s in scored]),
        "has_answer_tags_rate": _mean([s["has_answer_tags"] for s in scored]),
        "median_completion_tokens": _median(
            [s["n_completion_tokens"] for s in scored if s["n_completion_tokens"]]),
        "n_boxes": len(boxed),
        "readings": {},
    }
    if boxed:
        # The invariance statistic. Normalised by the ORIGINAL width so it is comparable
        # across arms; under `rel1000` it should sit near (box width / image width) * 1000.
        cell["median_raw_max"] = _median([s["raw_max"] for s in boxed])
        cell["median_raw_w_over_W"] = _median([s["raw_w"] / s["W"] for s in boxed])
        cell["median_rW_over_W"] = _median([s["rW"] / s["W"] for s in boxed])
        for reading in READINGS:
            r = [s["readings"][reading] for s in boxed]
            cell["readings"][reading] = {
                "mean_iou": _mean([x["iou"] for x in r]),
                "median_iou": _median([x["iou"] for x in r]),
                "iou_ge_0.5": _mean([x["iou"] >= 0.5 for x in r]),
                "mean_covered": _mean([x["covered"] for x in r]),
                "covered_ge_0.9": _mean([x["covered"] >= 0.9 for x in r]),
                "mean_out_of_frame": _mean([x["oof"] for x in r]),
                "mostly_off_image": _mean([x["oof"] > 0.5 for x in r]),
                "median_area_frac": _median([x["area_frac"] for x in r]),
            }
    return cell


def run_report(args):
    out_dir = Path(args.out_dir)
    raw = sorted((out_dir / "raw").glob("*.jsonl"))
    if not raw:
        raise SystemExit(f"deepeyes_coord_format_probe: nothing under {out_dir}/raw")

    recs, seen = [], set()
    for p in raw:
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            key = (r["model"], r["arm"], r["uid"])
            if key in seen:
                continue
            seen.add(key)
            recs.append(r)

    scored = [score_record(r) for r in recs]
    (out_dir / "scored.jsonl").write_text(
        "".join(json.dumps(s) + "\n" for s in scored))

    cells, keys = {}, []
    for s in scored:
        k = (s["model"], s["arm"])
        if k not in cells:
            cells[k] = []
            keys.append(k)
        cells[k].append(s)
    summary = {f"{m}|{a}": summarise(v) for (m, a), v in cells.items()}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    W = 96
    print("=" * W)
    print("DeepEyes Prompt 4 -- coordinate frame and format rate, on V* "
          f"({len(scored)} completions)")
    print("=" * W)

    print("\nFORMAT, under DeepEyes' own prompt (first assistant turn, "
          f"temperature {args.temperature})")
    print(f"  {'model / arm':<34} {'R_format ok':>12} {'tag struct':>11} "
          f"{'tool call':>10} {'<think>':>8} {'<answer>':>9}")
    for m, a in keys:
        c = summary[f"{m}|{a}"]
        print(f"  {m + ' / arm ' + a:<34} {c['format_ok_deepeyes']:>11.1%} "
              f"{c['tag_structure_ok']:>10.1%} {c['tool_call_rate']:>9.1%} "
              f"{c['has_think_rate']:>7.1%} {c['has_answer_tags_rate']:>8.1%}")
    print("\n  R_format ok  = their literal rule: counts balance. A completion with NO")
    print("                 tags passes it -- and never zooms. Read it next to the two")
    print("                 columns on its right, never alone.")
    print("  tool call    = a parseable <tool_call> with four numbers. This, not the")
    print("                 format column, is what gates R_tool.")

    print("\nCOORDINATE FRAME -- IoU against V*'s ground-truth boxes")
    for m, a in keys:
        c = summary[f"{m}|{a}"]
        if not c["readings"]:
            print(f"\n  {m} / arm {a}: no boxes emitted; nothing to score.")
            continue
        print(f"\n  {m} / arm {a}   ({c['n_boxes']} boxes, "
              f"median resized/original width = {c['median_rW_over_W']:.3f}, "
              f"median raw max coord = {c['median_raw_max']:.0f})")
        print(f"    {'reading':<13} {'mean IoU':>9} {'med IoU':>8} {'IoU>=.5':>8} "
              f"{'covers GT':>10} {'off-image':>10} {'med area':>9}")
        for reading in READINGS:
            r = c["readings"][reading]
            print(f"    {reading:<13} {r['mean_iou']:>9.3f} {r['median_iou']:>8.3f} "
                  f"{r['iou_ge_0.5']:>7.1%} {r['covered_ge_0.9']:>9.1%} "
                  f"{r['mostly_off_image']:>9.1%} {r['median_area_frac']:>8.3f}")

    print("\n  covers GT = the converted crop contains >=90% of the target box. A zoom")
    print("              box may legitimately be far bigger than its target, so this")
    print("              and IoU can disagree; if they pick different winners, say so.")
    print("  off-image = more than half the converted box falls outside the picture.")
    print("              A frame-only signal: it needs no ground truth.")

    # The invariance test, stated rather than left to be eyeballed.
    print("\nARM A vs ARM B -- does the emitted number move when max_pixels does?")
    models = sorted({m for m, _ in keys})
    for m in models:
        a_cell, b_cell = summary.get(f"{m}|A"), summary.get(f"{m}|B")
        if not (a_cell and b_cell and a_cell.get("readings") and b_cell.get("readings")):
            continue
        ra, rb = a_cell["median_raw_max"], b_cell["median_raw_max"]
        sa, sb = a_cell["median_rW_over_W"], b_cell["median_rW_over_W"]
        print(f"  {m}: median raw max coord {ra:.0f} -> {rb:.0f} "
              f"({rb / ra if ra else float('nan'):.3f}x) while the resized frame went "
              f"{sa:.3f} -> {sb:.3f} of the original ({sb / sa if sa else float('nan'):.3f}x)")
    print("  A ratio near 1.000 on the left means the numbers do not follow the "
          "processor:\n  the model is emitting a frame-independent grid (rel1000, or "
          "original pixels).\n  A ratio tracking the right-hand one means post-resize "
          "absolute pixels.")

    print(f"\nwrote {out_dir / 'summary.json'} and {out_dir / 'scored.jsonl'}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="generate", choices=("generate", "report"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--model-tag", default=None,
                    help="short name for the tables; defaults to the model dir's name")
    ap.add_argument("--arm", default="A", choices=("A", "B"),
                    help="A = the processor's own caps; B = max_pixels forced down, "
                         "which is what separates abs_orig from abs_resized")
    ap.add_argument("--arm-b-max-pixels", type=int, default=ARM_B_MAX_PIXELS)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--n", type=int, default=0, help="first N corpus rows (0 = all)")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="1.0 matches what GRPO rolls out at; 0 is greedy and "
                         "overstates format compliance")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    _check_prompt_matches_upstream()

    if args.stage == "report":
        return run_report(args)

    if not args.model:
        raise SystemExit("--model is required for --stage generate")
    args.model_tag = args.model_tag or Path(args.model).name
    run_generate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
