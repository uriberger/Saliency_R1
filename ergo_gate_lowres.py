#!/usr/bin/env python
"""Gate A for the ERGO port: does Qwen3-VL lose accuracy when the image is small?

ERGO's entire headroom is the accuracy that downsampling destroys and a crop gives
back. Their Table 1 on Qwen2.5-VL-7B / V* reads 77.0 at 16384 visual tokens, 64.9 at
1280, 56.5 at 640, and 77.0 again at 640 once the ground-truth crop is appended as a
second image. If Qwen3-VL-8B does not lose those points there is nothing for the
method to recover, and the port is cancelled before it is written.

There is specific reason to doubt it does. DeepStack fuses multi-level ViT features
precisely so that detail survives a smaller token budget, and per the Qwen3-VL tech
report the backbone already saw tool-integrated RL with an `image_zoom_in_tool` in
its own post-training. Both would show up here as a flat resolution curve.

THE FACTOR IS 32, NOT 28
------------------------
Qwen3-VL is patch-16 with a 2x2 spatial merge, so one visual token covers 32x32
pixels, not Qwen2.5-VL's 28x28. A budget copied literally as `N*28*28` buys 23%
fewer tokens than the number says. Every cap here is `N * 32 * 32` and is verified
against the processor by --token-census, which reports the visual tokens the model
actually received rather than the ones we asked for.

WHY --force_simple
------------------
`--model qwen3_vl` resolves to the CHAT wrapper (lmms_eval/models/chat/qwen3_vl.py),
and in that wrapper `max_pixels` is INERT. It calls
`ChatMessages.to_hf_messages(video_kwargs=...)` with no `image_kwargs`, so no
per-image budget reaches qwen_vl_utils; and it passes `do_resize=False` to the
processor, so the processor's own `max_pixels` never applies either. Measured on a
2000x1500 image with max_pixels set to 640 tokens:

    chat path   -> 2914 visual tokens   (native, capped only by qwen_vl_utils' 16384)
    simple path ->  609 visual tokens   (the cap applied)

So the chat wrapper has no resolution knob at all, and `--force_simple` (which
lmms-eval forwards to `models.get_model(model, force_simple)`) is the only way to
set one. The simple wrapper attaches `max_pixels`/`min_pixels` to each image element
explicitly. Consequence worth stating out loud: every `--max-pixels` we have ever
passed through launch_lmms_eval_job.sh in chat mode -- including the 3211264 default
it applies to mmerealworld, hrbench4k, hrbench8k and vstar_bench -- did nothing.

NO --use_cache
--------------
lmms-eval's response cache keys on (context, gen_kwargs, {task, split, doc_id}).
The pixel budget is in NONE of those, so a cache shared across caps would serve the
640-token answers back for the 16384-token arm and the gate would read exactly 0.0
everywhere. The runner never passes --use_cache; do not add it.

USAGE (fish)
------------
    # generate the task configs (writes <out>/tasks/)
    python ergo_gate_lowres.py --make-tasks --out-dir outputs/ergo_gate_lowres

    # what would run, and in what order
    python ergo_gate_lowres.py --plan --out-dir outputs/ergo_gate_lowres

    # measure the visual tokens each cap really buys -- CPU only, no GPU needed
    python ergo_gate_lowres.py --token-census --out-dir outputs/ergo_gate_lowres

    # the grid itself (inside an allocation)
    python ergo_gate_lowres.py --run --out-dir outputs/ergo_gate_lowres --num-gpus 8

    # the table and the verdict
    python ergo_gate_lowres.py --report --out-dir outputs/ergo_gate_lowres

Normally you do not call --run by hand; `launch_ergo_gate_lowres_job.sh` submits it.
"""

import argparse
import datetime
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent

# Qwen3-VL: patch 16, spatial merge 2 -> one visual token is 32x32 pixels.
# qwen_vl_utils derives the same number as image_patch_size * SPATIAL_MERGE_SIZE,
# and both lmms-eval wrappers pass image_patch_size=16, so this is not an
# assumption about the config -- it is the factor the resize actually uses.
VISUAL_TOKEN_PX = 32 * 32

# ERGO's three operating points, read off their Table 1.
CAPS = (640, 1280, 16384)

# The crop arm answers "is the information recoverable at all", so it sits at the
# cap where the penalty is largest. ERGO's own oracle row is at 640.
CROP_CAP = 640

# The simple wrapper's own default floor, passed explicitly so the runs record it.
# It matters for exactly one thing: a V* ground-truth crop is a few thousand pixels,
# and this upsamples it to ~256 tokens instead of the ~25 its native size would buy.
# ERGO re-encodes its crop the same way (src/ergo/utils.py:crop_with_zoom calls
# smart_resize on the cropped region), so the oracle arm is not quietly weaker than
# the method it stands in for.
MIN_PIXELS = 256 * VISUAL_TOKEN_PX

LMMS_EVAL_DIR = Path(os.environ.get("LMMS_EVAL_DIR", "/home/uberger/scratch/research/lmms-eval"))

# The r1 system prompt the cold start was trained to answer under. Same file
# --r1-mode uses, so the cold-start rows here are set up like every other
# benchmark row we report.
R1_SYSTEM_PROMPT = Path(
    "/home/uberger/scratch/research/vlm_reasoning/configs/prompts/saliency_r1_system.txt"
)

MODELS = {
    # Vanilla Qwen3-VL answers an MCQ in a couple of tokens and has no reasoning
    # format to protect, so it gets the stock assistant prompt and a short cap.
    # Its row is NOT comparable with the cold start's in absolute terms -- the gate
    # is read DOWN each model's own column, never across them.
    "base": dict(
        path="Qwen/Qwen3-VL-8B-Instruct",
        system_prompt=None,
        max_new_tokens=512,
    ),
    "coldstart": dict(
        path=str(REPO / "checkpoint" / "coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged"),
        system_prompt=str(R1_SYSTEM_PROMPT),
        max_new_tokens=4096,
    ),
}

# V* is 191 documents and is run whole. HR-Bench is 800 per split and is
# subsampled, because the gate needs a number today and three caps x two models x
# two splits of 800 does not fit one allocation.
DEFAULT_HRBENCH_N = 300
SAMPLE_SEED = 1234

VSTAR_METRIC = "vstar_overall_acc,none"
HRBENCH_METRIC = "average,none"
PARSE_FAIL_METRIC = "parse_fail,none"

TASK_METRIC = {
    "vstar_bench": VSTAR_METRIC,
    "vstar_bench_gtcrop": VSTAR_METRIC,
    "hrbench4k_gate": HRBENCH_METRIC,
    "hrbench8k_gate": HRBENCH_METRIC,
}

# Pretty names for the report, in the order the table reads.
TASK_LABEL = {
    "vstar_bench": "V*",
    "hrbench4k_gate": "HR-Bench 4K",
    "hrbench8k_gate": "HR-Bench 8K",
    "vstar_bench_gtcrop": "V* + GT crop",
}


# ---------------------------------------------------------------------------
# Generated task configs
# ---------------------------------------------------------------------------
# Written into <out>/tasks/ and reached with --include_path, so the shared
# lmms-eval clone is never touched. The absolute `include:` paths are why these
# are generated per run rather than committed: a stale path would silently
# evaluate a different task file (the same reason eval_mini/make_mini_tasks.py
# regenerates its configs on every job).

SAMPLER_MODULE = '''"""Seeded subsample for the HR-Bench gate tasks.

`datasets.shuffle` with a fixed seed is a deterministic permutation that does not
depend on how many rows are then taken, so select(range(300)) is a PREFIX of
select(range(800)): raising the sample later adds documents and moves none, and a
300-item result stays valid for the items it covers.

A plain `--limit N` would not do. HR-Bench stores its two categories in runs --
the first 400 rows of each split are 320 `single` and 80 `cross` -- so the first N
rows are not a sample of the benchmark, they are a sample of one half of it.
"""

SAMPLE_N = {n}
SAMPLE_SEED = {seed}


def take(dataset):
    if len(dataset) <= SAMPLE_N:
        return dataset
    return dataset.shuffle(seed=SAMPLE_SEED).select(range(SAMPLE_N))
'''


# Raw, and formatted with nothing: the generated file is regex-heavy, and a
# non-raw template would halve every backslash on the way out.
HRBENCH_SCORER_MODULE = r'''"""Offline scoring for the HR-Bench gate tasks.

The upstream `hrbench_process_results` hands every prediction to
`HRBenchEval.get_chat_response`, which falls through to a GPT judge whenever it
cannot read an option letter locally. On a compute node that is fatal twice over:
there is no route to the API, so each unreadable item burns ten 30-second POSTs,
and when they all fail the function returns WITHOUT writing `gpt_prediction`, so
the caller raises KeyError and takes the whole task's aggregation with it.

This replacement keeps upstream's local matcher (`can_infer`, which is what
decides the score for every item the judge never sees, so the numbers are the
same rules) and replaces the judge with a letter regex over the post-reasoning
segment. Anything still unreadable scores 0 and is counted.

`parse_fail` is reported as a metric on purpose. A resolution arm that looks worse
because its chains stopped ending in a letter is a grading accident, not a
resolution effect, and this is the column that tells the two apart -- see
LogicVista and MathVision, both of which were parser artifacts that survived a
full results table before anyone noticed.
"""

import re
from collections import defaultdict

from lmms_eval.tasks.hrbench.utils import hrbench_doc_to_options, hrbench_evaluator

_LETTER_PATTERNS = (
    r"(?:THE\s+)?(?:ANSWER|CHOICE|OPTION)(?:\s+IS)?[\s:]*\**\s*\(?([A-D])\)?\b",
    r"\\boxed\{\s*\(?([A-D])\)?\s*\}",
    r"^\(?([A-D])\)?\s*[\.\):\]]",
    r"^\(?([A-D])\)?\s*$",
)


def _letter(text):
    """Last-resort option letter, read only from the segment after the reasoning.

    Searching the whole response would read the letter out of the thinking: a chain
    that says "at first glance A ... but" scores A even when it concludes B. The
    caller has already cut everything up to </think>; this never sees it.

    The final fallback takes the LAST standalone letter rather than the first,
    because a tail that enumerates before concluding ends on its answer.
    """
    text = text.strip().upper()
    for pattern in _LETTER_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()
    letters = re.findall(r"\b([A-D])\b", text)
    if letters:
        return letters[-1]
    return None


def process_results(doc, results):
    pred = results[0] if results else ""
    if "</think>" in pred:
        pred = pred.rsplit("</think>", 1)[1]
    pred = pred.strip()

    options = hrbench_doc_to_options(doc)
    inferred = hrbench_evaluator.can_infer(pred, options) or _letter(pred)
    failed = 0 if inferred else 1
    score = 1 if inferred and str(inferred).lower() == str(doc["answer"]).lower() else 0

    row = {"index": doc["index"], "cycle_category": doc["cycle_category"], "gpt_score": score}
    return {
        doc["category"]: dict(row),
        "average": dict(row),
        "parse_fail": failed,
    }


def aggregate(results, args):
    """Upstream's aggregation: mean over cycle categories of their mean score.

    Copied rather than imported because importing it would drag in the judge's
    module-level client construction for no benefit.
    """
    scores = defaultdict(list)
    for result in results:
        scores[result["cycle_category"]].append(result["gpt_score"])
    per_cycle = [sum(v) / len(v) for v in scores.values()]
    return sum(per_cycle) / len(per_cycle) if per_cycle else 0.0


def aggregate_parse_fail(results, args):
    return sum(results) / len(results) if results else 0.0
'''


GTCROP_MODULE = '''"""V* with the ground-truth region appended as a second image.

The oracle arm. ERGO's turn 2 shows the model an enlarged crop of the region it
asked for; this shows it the crop it SHOULD have asked for, so the row measures
the ceiling the method is chasing rather than the method.

Where the boxes come from
-------------------------
Neither HF release of V* ships them: `lmms-lab-encoder/vstar-bench` and
`craigwu/vstar_bench` both expose only (image, text, category, question_id, label).
The boxes live in the per-image annotation JSONs of the craigwu repo
(`direct_attributes/sa_4690.json` and friends, 238 of them), as absolute
[x, y, w, h] in ORIGINAL image pixels. They were fetched once from a host with
network and committed to assets/vstar_bench_boxes.json, keyed by question_id; all
191 documents are covered, 137 with one box and 54 (the relative_position half,
which asks about two objects) with two.

Verified against the images the task actually loads: the encoder release preserves
the original resolution, so the coordinates apply unchanged, and no box falls
outside its image. The median box is 0.06% of the image area -- which is the whole
point of V*, and the reason a 640-token encode of the full scene cannot see it.

The crop
--------
The union of the item's boxes, taken tight with no padding, matching ERGO's
`crop_with_zoom` (which crops the predicted box exactly and re-encodes it). On the
two-box items the union is whatever it takes to contain both, which is sometimes
most of the picture; that is the same thing a single zoom box would have to do, so
it is not a handicap being added here.

HR-Bench gets no equivalent arm. It ships no boxes at all -- its columns are
(index, question, answer, A-D, category, cycle_category, image) and its paper
annotates none -- so there is nothing to crop to.
"""

import json
from pathlib import Path

from lmms_eval.tasks.vstar_bench.utils import vstar_doc_to_text

BOXES = json.loads(Path("{boxes}").read_text())

# Derived from ERGO's own turn-2 wording (src/ergo/const.py: "Carefully analyze
# both the original image and the enlarged sub-image"), trimmed to a statement of
# fact. It must not ask for a format: the task's own post_prompt already does,
# each model answers in its own, and a second instruction would change what is
# being measured from resolution to prompt-following.
CROP_NOTE = (
    "The second image is an enlarged crop of the region of the first image that "
    "contains the information needed to answer the question.\\n"
)


def union_box(boxes):
    """Tight union of [x, y, w, h] boxes as (x1, y1, x2, y2)."""
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[0] + b[2] for b in boxes)
    y2 = max(b[1] + b[3] for b in boxes)
    return x1, y1, x2, y2


def doc_to_visual(doc):
    image = doc["image"].convert("RGB")
    entry = BOXES.get(str(doc["question_id"]))
    if not entry or not entry.get("bbox"):
        # Never silently degrade to the single-image arm: a missing box would make
        # this row a copy of the plain 640 arm and quietly shrink the oracle gap.
        raise KeyError(f"no ground-truth box for V* question_id {{doc['question_id']}}")
    x1, y1, x2, y2 = union_box(entry["bbox"])
    width, height = image.size
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(width, int(round(x2))), min(height, int(round(y2)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"degenerate V* box for question_id {{doc['question_id']}}")
    return [image, image.crop((x1, y1, x2, y2))]


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return CROP_NOTE + vstar_doc_to_text(doc, lmms_eval_specific_kwargs)
'''


HRBENCH_CONFIG = """# Generated by ergo_gate_lowres.py -- do not edit by hand.
# {parent_task}, cut to a seeded sample and scored without the GPT judge.
include: {parent}
task: "{task}"
process_docs: !function gatesample.take
process_results: !function hrbench_gate.process_results
metric_list:
  - metric: single
    aggregation: !function hrbench_gate.aggregate
    higher_is_better: true
  - metric: cross
    aggregation: !function hrbench_gate.aggregate
    higher_is_better: true
  - metric: average
    aggregation: !function hrbench_gate.aggregate
    higher_is_better: true
  - metric: parse_fail
    aggregation: !function hrbench_gate.aggregate_parse_fail
    higher_is_better: false
"""

GTCROP_CONFIG = """# Generated by ergo_gate_lowres.py -- do not edit by hand.
# V* with the ground-truth region appended as a second image. Only the two media
# keys are overridden; the metric, the aggregation and the answer parser are the
# real benchmark's, so this row is scored by exactly the same rules as the others.
include: {parent}
task: "vstar_bench_gtcrop"
doc_to_visual: !function gtcrop.doc_to_visual
doc_to_text: !function gtcrop.doc_to_text
"""


def make_tasks(out_dir, hrbench_n=DEFAULT_HRBENCH_N, seed=SAMPLE_SEED):
    """Write the generated task configs to <out_dir>/tasks and return that path."""
    tasks_root = LMMS_EVAL_DIR / "lmms_eval" / "tasks"
    if not tasks_root.is_dir():
        raise SystemExit(f"not an lmms-eval clone: {LMMS_EVAL_DIR} (no lmms_eval/tasks/)")

    boxes = REPO / "assets" / "vstar_bench_boxes.json"
    if not boxes.is_file():
        raise SystemExit(f"missing {boxes}; the V* ground-truth boxes are not optional")

    task_dir = Path(out_dir) / "tasks"
    task_dir.mkdir(parents=True, exist_ok=True)

    (task_dir / "gatesample.py").write_text(SAMPLER_MODULE.format(n=hrbench_n, seed=seed))
    (task_dir / "hrbench_gate.py").write_text(HRBENCH_SCORER_MODULE)
    (task_dir / "gtcrop.py").write_text(GTCROP_MODULE.format(boxes=boxes))

    for parent_task, task in (("hrbench4k", "hrbench4k_gate"), ("hrbench8k", "hrbench8k_gate")):
        parent = tasks_root / "hrbench" / f"{parent_task}.yaml"
        if not parent.is_file():
            raise SystemExit(f"{task}: no such task yaml: {parent}")
        (task_dir / f"{task}.yaml").write_text(
            HRBENCH_CONFIG.format(parent=parent, parent_task=parent_task, task=task)
        )

    vstar_parent = tasks_root / "vstar_bench" / "vstar_bench.yaml"
    if not vstar_parent.is_file():
        raise SystemExit(f"no such task yaml: {vstar_parent}")
    (task_dir / "vstar_bench_gtcrop.yaml").write_text(GTCROP_CONFIG.format(parent=vstar_parent))

    print(f"wrote 3 task configs + 3 modules to {task_dir}", file=sys.stderr)
    print(f"  HR-Bench sample: {hrbench_n} per split, seed {seed}", file=sys.stderr)
    return task_dir


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------
def plan():
    """The units, cheapest and most decisive first.

    V* is 191 documents against HR-Bench's 2 x N, and it is the benchmark ERGO's
    Table 1 is quoted from, so every V* arm runs before any HR-Bench arm. If the
    allocation runs out the gate still has its own numbers; HR-Bench is
    corroboration.

    One unit is one `accelerate launch`, which means one model load (~2 min on 8
    ranks). The two HR-Bench splits therefore share a unit -- they take the same
    model at the same cap -- while V* keeps its own so it can carry --log_samples
    without dragging HR-Bench's base64 image column into the log (see run_unit).
    """
    units = []
    for model in ("coldstart", "base"):
        for cap in CAPS:
            units.append(dict(id=f"{model}_vstar_px{cap}", model=model, cap=cap,
                              tasks=["vstar_bench"], log_samples=True))
        units.append(dict(id=f"{model}_vstarcrop_px{CROP_CAP}", model=model, cap=CROP_CAP,
                          tasks=["vstar_bench_gtcrop"], log_samples=True))
    for model in ("coldstart", "base"):
        for cap in CAPS:
            units.append(dict(id=f"{model}_hrbench_px{cap}", model=model, cap=cap,
                              tasks=["hrbench4k_gate", "hrbench8k_gate"], log_samples=False))
    return units


def unit_command(unit, out_dir, num_gpus):
    """The lmms-eval command line for one unit."""
    spec = MODELS[unit["model"]]
    max_pixels = unit["cap"] * VISUAL_TOKEN_PX
    model_args = ",".join([
        f"pretrained={spec['path']}",
        f"max_pixels={max_pixels}",
        f"min_pixels={MIN_PIXELS}",
        "attn_implementation=sdpa",
    ])

    if num_gpus > 1:
        launch = ["accelerate", "launch", f"--num_processes={num_gpus}", "-m", "lmms_eval"]
    else:
        launch = [sys.executable, "-m", "lmms_eval"]

    cmd = launch + [
        "--model", "qwen3_vl",
        # The chat wrapper has no resolution knob at all; see the module docstring.
        "--force_simple",
        "--model_args", model_args,
        "--tasks", ",".join(unit["tasks"]),
        "--batch_size", "1",
        "--gen_kwargs", f"max_new_tokens={spec['max_new_tokens']},repetition_penalty=1.05",
        "--output_path", str(Path(out_dir) / "units" / unit["id"]),
        "--include_path", str(Path(out_dir) / "tasks"),
    ]
    # HR-Bench keeps its image as a base64 STRING, which lmms-eval's saved_doc
    # filter does not recognise as media (it checks for PIL/bytes/ndarray), so
    # --log_samples would write a few megabytes of base64 per row. V*'s image is a
    # PIL object and is stripped, so it can afford the samples -- and wants them,
    # because a per-item file is what makes the 640-vs-16384 comparison paired.
    if unit["log_samples"]:
        cmd.append("--log_samples")
    return cmd


def run_unit(unit, out_dir, num_gpus, dry_run=False):
    """Run one unit and harvest its metrics into <out>/units/<id>.json."""
    out_dir = Path(out_dir)
    marker = out_dir / "units" / f"{unit['id']}.json"
    if marker.is_file():
        print(f"[skip] {unit['id']} (already has {marker.name})", file=sys.stderr)
        return json.loads(marker.read_text())

    cmd = unit_command(unit, out_dir, num_gpus)
    print(f"[run ] {unit['id']}\n       {' '.join(shlex.quote(c) for c in cmd)}", file=sys.stderr)
    if dry_run:
        return None

    env = dict(os.environ)
    # lmms-eval reroutes the datasets cache to local scratch whenever HF_HOME is
    # on a remote filesystem, which ours is. The rerouted directory is empty, so
    # with HF_HUB_OFFLINE=1 every task dies on a ConnectionError before the model
    # is even built. /nonexistent is the documented way to say "write straight to
    # HF_HOME"; launch_lmms_eval_job.sh exports the same value for the same reason.
    env.setdefault("LMMS_LOCAL_CACHE", "/nonexistent")

    spec = MODELS[unit["model"]]
    if spec["system_prompt"]:
        env["LMMS_SYSTEM_PROMPT_FILE"] = spec["system_prompt"]
    else:
        env.pop("LMMS_SYSTEM_PROMPT_FILE", None)
        env.pop("LMMS_SYSTEM_PROMPT", None)

    started = time.time()
    result = subprocess.run(cmd, cwd=str(LMMS_EVAL_DIR), env=env)
    elapsed = time.time() - started
    if result.returncode != 0:
        print(f"[FAIL] {unit['id']} exited {result.returncode} after {elapsed/60:.1f} min",
              file=sys.stderr)
        return None

    harvested = harvest(unit, out_dir, elapsed)
    if harvested is None:
        print(f"[FAIL] {unit['id']} produced no results json", file=sys.stderr)
        return None
    marker.write_text(json.dumps(harvested, indent=1, sort_keys=True))
    print(f"[done] {unit['id']} in {elapsed/60:.1f} min: "
          + ", ".join(f"{t}={v:.3f}" for t, v in sorted(harvested["metrics"].items())),
          file=sys.stderr)
    return harvested


def harvest(unit, out_dir, elapsed=None):
    """Pull the aggregate metric for each of a unit's tasks out of lmms-eval's json."""
    unit_dir = Path(out_dir) / "units" / unit["id"]
    candidates = sorted(unit_dir.rglob("*_results.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        return None
    payload = json.loads(candidates[-1].read_text())
    results = payload.get("results", {})

    metrics, extras = {}, {}
    for task in unit["tasks"]:
        row = results.get(task)
        if not row:
            return None
        key = TASK_METRIC[task]
        if key not in row:
            return None
        metrics[task] = row[key]
        for name in ("single,none", "cross,none", PARSE_FAIL_METRIC):
            if name in row:
                extras.setdefault(task, {})[name.split(",")[0]] = row[name]
    return dict(
        unit=unit["id"], model=unit["model"], cap=unit["cap"], tasks=unit["tasks"],
        max_pixels=unit["cap"] * VISUAL_TOKEN_PX, min_pixels=MIN_PIXELS,
        metrics=metrics, extras=extras, n_samples=payload.get("n-samples", {}),
        minutes=round(elapsed / 60, 1) if elapsed else None,
        results_json=str(candidates[-1]),
    )


def run(out_dir, num_gpus, only=None, dry_run=False, deadline=None):
    out_dir = Path(out_dir)
    (out_dir / "units").mkdir(parents=True, exist_ok=True)
    units = [u for u in plan() if not only or u["id"] in only]
    done, consecutive_failures = 0, 0
    for unit in units:
        if deadline and time.time() > deadline:
            print(f"[stop] out of time before {unit['id']}; rerun to resume", file=sys.stderr)
            break
        if run_unit(unit, out_dir, num_gpus, dry_run=dry_run) is not None:
            done += 1
            consecutive_failures = 0
            continue
        consecutive_failures += 1
        # Every unit runs the same command with two substitutions, so two in a
        # row failing is a configuration fault, not a bad checkpoint or a flaky
        # node. Marching through the remaining twelve would turn one readable
        # error into fourteen and spend the allocation discovering it again.
        if consecutive_failures >= 2:
            print("[abort] two units failed in a row; this is a config fault, not a flake",
                  file=sys.stderr)
            break
    print(f"\n{done}/{len(units)} units complete", file=sys.stderr)
    return done == len(units)


# ---------------------------------------------------------------------------
# Token census -- CPU only
# ---------------------------------------------------------------------------
def token_census(out_dir, n=24):
    """How many visual tokens each cap really buys, per benchmark.

    The point of the table this writes is that a cap is a request, not a result.
    `floor_by_factor` in qwen_vl_utils rounds DOWN to whole 32-px steps, so a 640
    cap lands a little under 640; and an image smaller than the cap is never
    enlarged to meet it, so V*'s ~3 MP photos cannot reach 16384 tokens at all and
    that arm is native resolution, not a 16384-token arm. Saying so here is the
    difference between reporting a resolution curve and reporting a fiction.
    """
    from PIL import Image  # noqa: F401  (imported for the error message if absent)
    from qwen_vl_utils.vision_process import smart_resize

    import datasets

    rows = []
    sources = [
        ("V*", "lmms-lab-encoder/vstar-bench", "test", None),
        ("HR-Bench 4K", "DreamMr/HR-Bench", "hrbench_4k", "hrbench_version_split"),
        ("HR-Bench 8K", "DreamMr/HR-Bench", "hrbench_8k", "hrbench_version_split"),
    ]
    for label, path, split, name in sources:
        kwargs = {"name": name} if name else {}
        data = datasets.load_dataset(path, split=split, **kwargs)
        sizes = []
        for i in range(min(n, len(data))):
            doc = data[i]
            image = doc["image"]
            if isinstance(image, str):
                import base64
                import io

                image = Image.open(io.BytesIO(base64.b64decode(image)))
            sizes.append(image.size)
        for cap in CAPS:
            tokens = []
            for width, height in sizes:
                h, w = smart_resize(height, width, factor=32,
                                    min_pixels=MIN_PIXELS, max_pixels=cap * VISUAL_TOKEN_PX)
                tokens.append(h * w // VISUAL_TOKEN_PX)
            tokens.sort()
            rows.append(dict(benchmark=label, cap=cap, n=len(tokens),
                             median_tokens=tokens[len(tokens) // 2],
                             min_tokens=tokens[0], max_tokens=tokens[-1],
                             native_px=f"{sizes[0][0]}x{sizes[0][1]}"))
            print(f"{label:<12} cap {cap:>5} -> median {tokens[len(tokens)//2]:>6} tokens "
                  f"(min {tokens[0]}, max {tokens[-1]})")
    path = Path(out_dir) / "token_census.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {path}", file=sys.stderr)
    return rows


def target_census(out_dir):
    """How big V*'s answer-bearing region is, in visual tokens, at each cap.

    The accuracy table says whether the penalty exists. This says whether it CAN,
    and it is the half the gate can produce without a GPU.

    A visual token is 32x32 pixels of the RESIZED image, so the question "is the
    object still there at 640 tokens" has an arithmetic answer: scale the box by
    the same factor the image was scaled by and count. A target that lands under
    one token cannot be read at any temperature, and one that is still several
    tokens wide at 640 was never going to be the thing a crop recovers.

    Run over all 191 documents and all three caps, from the committed boxes.
    """
    from qwen_vl_utils.vision_process import smart_resize

    import datasets

    boxes = json.loads((REPO / "assets" / "vstar_bench_boxes.json").read_text())
    data = datasets.load_dataset("lmms-lab-encoder/vstar-bench", split="test")

    rows = []
    for cap in list(CAPS) + ["crop"]:
        tokens, sub_token = [], 0
        for i in range(len(data)):
            doc = data[i]
            width, height = doc["image"].size
            entry = boxes[str(doc["question_id"])]
            x1 = min(b[0] for b in entry["bbox"])
            y1 = min(b[1] for b in entry["bbox"])
            x2 = max(b[0] + b[2] for b in entry["bbox"])
            y2 = max(b[1] + b[3] for b in entry["bbox"])
            box_w, box_h = max(1, x2 - x1), max(1, y2 - y1)

            if cap == "crop":
                # The oracle arm: the crop IS the region, so it is re-encoded on
                # its own and min_pixels is what decides its size.
                rh, rw = smart_resize(box_h, box_w, factor=32,
                                      min_pixels=MIN_PIXELS, max_pixels=CROP_CAP * VISUAL_TOKEN_PX)
                target_tokens = rh * rw / VISUAL_TOKEN_PX
            else:
                rh, rw = smart_resize(height, width, factor=32, min_pixels=MIN_PIXELS,
                                      max_pixels=cap * VISUAL_TOKEN_PX)
                scale_x, scale_y = rw / width, rh / height
                target_tokens = (box_w * scale_x) * (box_h * scale_y) / VISUAL_TOKEN_PX
            tokens.append(target_tokens)
            sub_token += target_tokens < 1.0
        tokens.sort()
        row = dict(cap=cap, n=len(tokens),
                   median_target_tokens=round(tokens[len(tokens) // 2], 2),
                   p10=round(tokens[len(tokens) // 10], 2),
                   p90=round(tokens[9 * len(tokens) // 10], 2),
                   share_under_one_token=round(sub_token / len(tokens), 3))
        rows.append(row)
        print(f"V* target at cap {str(cap):>5}: median {row['median_target_tokens']:>8.2f} tokens "
              f"(p10 {row['p10']}, p90 {row['p90']}); "
              f"{100*row['share_under_one_token']:.1f}% under one token")

    path = Path(out_dir) / "target_census.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {path}", file=sys.stderr)
    return rows


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def load_results(out_dir):
    out = {}
    for path in sorted((Path(out_dir) / "units").glob("*.json")):
        payload = json.loads(path.read_text())
        for task, value in payload["metrics"].items():
            out[(payload["model"], task, payload["cap"])] = dict(
                value=value, extras=payload.get("extras", {}).get(task, {}),
                minutes=payload.get("minutes"))
    return out


def verdict(gap):
    """The gate's own rule, applied to the 16384-minus-640 gap on V*."""
    if gap is None:
        return "NO VERDICT", "V* did not produce both ends of the curve."
    if gap >= 10:
        return "PROCEED", (
            f"{gap:+.1f} points between 16384 and 640 visual tokens on V*. ERGO's "
            "premise holds on this backbone: there is accuracy for a crop to recover.")
    if gap >= 3:
        return "PROCEED ONLY ON EFFICIENCY", (
            f"{gap:+.1f} points between 16384 and 640 visual tokens on V*. Too small to "
            "sell as an accuracy result; the claim has to be the efficiency axis -- same "
            "accuracy at a fraction of the tokens -- or not at all.")
    return "STOP", (
        f"{gap:+.1f} points between 16384 and 640 visual tokens on V*. ERGO's premise "
        "does not hold on this backbone: there is nothing for the crop to recover, and "
        "no amount of making the method work can produce a number it cannot reach.")


def report(out_dir, markdown=None):
    data = load_results(out_dir)
    if not data:
        raise SystemExit(f"no harvested units under {Path(out_dir)/'units'}")

    tasks = [t for t in ("vstar_bench", "hrbench4k_gate", "hrbench8k_gate") if
             any(k[1] == t for k in data)]
    lines = []
    header = "| model | benchmark | " + " | ".join(f"{c} tok" for c in CAPS)
    header += " | 640 + GT crop | 16384 - 640 |"
    lines.append(header)
    lines.append("|" + "---|" * (len(CAPS) + 4))

    gaps = {}
    for model in ("base", "coldstart"):
        for task in tasks:
            cells = []
            for cap in CAPS:
                entry = data.get((model, task, cap))
                cells.append(f"{100*entry['value']:.1f}" if entry else "--")
            crop = data.get((model, "vstar_bench_gtcrop", CROP_CAP)) if task == "vstar_bench" else None
            crop_cell = f"{100*crop['value']:.1f}" if crop else ("--" if task == "vstar_bench" else "n/a")
            lo = data.get((model, task, min(CAPS)))
            hi = data.get((model, task, max(CAPS)))
            gap = 100 * (hi["value"] - lo["value"]) if lo and hi else None
            gaps[(model, task)] = gap
            gap_cell = f"{gap:+.1f}" if gap is not None else "--"
            lines.append(f"| {model} | {TASK_LABEL[task]} | " + " | ".join(cells)
                         + f" | {crop_cell} | {gap_cell} |")

    table = "\n".join(lines)
    gap = gaps.get(("coldstart", "vstar_bench"))
    if gap is None:
        gap = gaps.get(("base", "vstar_bench"))
    call, reasoning = verdict(gap)

    print(table)
    print()
    print(f"VERDICT: {call} -- {reasoning}")

    if markdown:
        Path(markdown).write_text(render_markdown(out_dir, table, call, reasoning, data))
        print(f"\nwrote {markdown}", file=sys.stderr)
    return call, gap


def render_markdown(out_dir, table, call, reasoning, data):
    census = Path(out_dir) / "token_census.json"
    census_table = ""
    if census.is_file():
        rows = json.loads(census.read_text())
        census_table = "\n".join(
            ["| benchmark | cap asked | median tokens received |", "|---|---|---|"]
            + [f"| {r['benchmark']} | {r['cap']} | {r['median_tokens']} |" for r in rows])
    stamp = datetime.datetime.now().strftime("%Y-%m-%d")
    parse = []
    for (model, task, cap), entry in sorted(data.items()):
        fail = entry["extras"].get("parse_fail")
        if fail:
            parse.append(f"| {model} | {TASK_LABEL.get(task, task)} | {cap} | {100*fail:.1f}% |")
    parse_table = ""
    if parse:
        parse_table = "\n".join(
            ["| model | benchmark | cap | unreadable answers |", "|---|---|---|---|"] + parse)
    return (f"# Gate A: Qwen3-VL's low-resolution penalty\n\n_Generated {stamp} by "
            f"`ergo_gate_lowres.py --report`._\n\n## Accuracy (%)\n\n{table}\n\n"
            f"## Visual tokens actually received\n\n{census_table}\n\n"
            f"## Unreadable answers\n\n{parse_table or '_none_'}\n\n"
            f"## Verdict\n\n**{call}** -- {reasoning}\n")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--make-tasks", action="store_true")
    p.add_argument("--plan", action="store_true")
    p.add_argument("--run", action="store_true")
    p.add_argument("--report", action="store_true")
    p.add_argument("--token-census", action="store_true")
    p.add_argument("--num-gpus", type=int, default=8)
    p.add_argument("--hrbench-n", type=int, default=DEFAULT_HRBENCH_N)
    p.add_argument("--seed", type=int, default=SAMPLE_SEED)
    p.add_argument("--only", action="append", help="run only this unit id (repeatable)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--job-minutes", type=int, help="stop starting units after this many minutes")
    p.add_argument("--markdown", help="also write the report to this file")
    args = p.parse_args()

    if args.make_tasks:
        make_tasks(args.out_dir, args.hrbench_n, args.seed)
    if args.plan:
        for unit in plan():
            cmd = unit_command(unit, args.out_dir, args.num_gpus)
            print(f"{unit['id']}\t{','.join(unit['tasks'])}\t{unit['cap']*VISUAL_TOKEN_PX} px")
            print("    " + " ".join(shlex.quote(c) for c in cmd))
    if args.token_census:
        token_census(args.out_dir)
        print()
        target_census(args.out_dir)
    if args.run:
        if not (Path(args.out_dir) / "tasks" / "vstar_bench_gtcrop.yaml").is_file():
            make_tasks(args.out_dir, args.hrbench_n, args.seed)
        deadline = time.time() + args.job_minutes * 60 if args.job_minutes else None
        complete = run(args.out_dir, args.num_gpus, only=args.only,
                       dry_run=args.dry_run, deadline=deadline)
        if not args.dry_run and not complete:
            # 17, like run_bench_eval_steps.sh: "there is more owed", not "it broke".
            sys.exit(17)
    if args.report:
        report(args.out_dir, markdown=args.markdown)
    if not any([args.make_tasks, args.plan, args.run, args.report, args.token_census]):
        p.error("nothing to do: pass one of --make-tasks/--plan/--token-census/--run/--report")


if __name__ == "__main__":
    main()
