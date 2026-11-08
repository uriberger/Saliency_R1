#!/usr/bin/env python
"""CPU tests for the four visual-search benchmarks in eval_mini/benchmarks.py.

    python test_search_benchmarks_cpu.py              # table + plan + reduction
    python test_search_benchmarks_cpu.py --lmms-eval  # also load the real tasks

The default run needs nothing but this repo: it checks the table, the banking
plan, the sample profile and the reduction of a results.json. `--lmms-eval` adds
the expensive half -- generating the configs against the real clone, constructing
the tasks offline out of the HF cache, and driving each task's OWN
process_results + aggregation to re-derive `scale` from the code rather than from
a file. No GPU either way.

What is being defended here is mostly invisible when it breaks:

  * a benchmark scored at a neighbour's resolution still writes a normal-looking
    results.json, and on these four that is the whole difference between a
    benchmark number and a number;
  * adding a 191-row benchmark to a 300-document suite renames every existing
    run's sample profile, which orphans the files rather than erroring;
  * spatialtreebench's aggregate scores a leaf the sample MISSED as 0, so an
    unstratified draw reads low in a way that looks exactly like a worse model.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(HERE / "eval_mini"))
sys.path.insert(0, str(HERE))

from benchmarks import (BENCHMARKS, DEFAULT_SUITE_N, NATIVE_MAX_PIXELS,  # noqa: E402
                        as_suite_n, base_task, extra_args, plan_units,
                        profile_dir, profile_name, suite_tasks, task_sample_n)
import bench_eval  # noqa: E402
import bench_samples  # noqa: E402

SEARCH = ("vstar_bench", "hrbench_4k", "hrbench_8k", "spatialtreebench")
LMMS_EVAL_DIR = os.environ.get("LMMS_EVAL_DIR", "/home/uberger/scratch/research/lmms-eval")

PASSED = FAILED = 0


def check(name, got, want):
    global PASSED, FAILED
    if got == want:
        PASSED += 1
        print(f"  ok    {name}")
    else:
        FAILED += 1
        print(f"  FAIL  {name}\n          got  {got!r}\n          want {want!r}")


def check_true(name, condition, detail=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  ok    {name}")
    else:
        FAILED += 1
        print(f"  FAIL  {name}   {detail}")


# ------------------------------------------------------------------ the table

def test_table():
    print("the table")
    check("all four are in it", [t for t in SEARCH if t in BENCHMARKS], list(SEARCH))
    check("none of them enters a suite mean",
          [t for t in SEARCH if BENCHMARKS[t]["in_mean"]], [])
    check("all four are on demand",
          [t for t in SEARCH if not BENCHMARKS[t].get("on_demand")], [])

    # scale, from the results.json these tasks actually emitted. The yamls say
    # nothing about it and the two families disagree, which is the entire reason
    # the field exists.
    check("V* reports a percentage", BENCHMARKS["vstar_bench"]["scale"], 100.0)
    check("HR-Bench reports a fraction",
          {BENCHMARKS[t]["scale"] for t in ("hrbench_4k", "hrbench_8k")}, {1.0})
    check("spatialtreebench's weighted tree is already 0-1",
          BENCHMARKS["spatialtreebench"]["scale"], 1.0)

    # The three image benchmarks run at the model's own ceiling. mme-realworld
    # keeps the 3.2 M it was always measured at, because every number on every
    # existing curve was taken there.
    check("the three image benchmarks ask for native resolution",
          {t: BENCHMARKS[t].get("max_pixels")
           for t in ("vstar_bench", "hrbench_4k", "hrbench_8k")},
          {t: NATIVE_MAX_PIXELS for t in ("vstar_bench", "hrbench_4k", "hrbench_8k")})
    check("mme-realworld is left where it was",
          BENCHMARKS["mmerealworld"]["max_pixels"], 3211264)
    check("the native figure is Qwen3-VL's preprocessor ceiling, not Qwen2.5-VL's",
          NATIVE_MAX_PIXELS, 16384 * 32 * 32)

    # lmms-eval's own task names have to reduce too: the full 800-row HR-Bench
    # runs already on disk are the baseline table, and they are called hrbench4k.
    check("lmms-eval's spelling maps onto ours",
          [base_task(t) for t in ("hrbench4k", "hrbench8k", "hrbench_4k_mini")],
          ["hrbench_4k", "hrbench_8k", "hrbench_4k"])


# ------------------------------------------------------------------- the plan

def test_plan():
    print("the banking plan")
    default = plan_units("suite", DEFAULT_SUITE_N)
    check("the default plan is the three units it has always been",
          [u[0] for u in default], ["natural", "nonnatural", "mmerw"])
    check("and no search benchmark is in any of them",
          [t for _tag, tasks, _x, _m in default for t in tasks
           if base_task(t) in SEARCH], [])

    asked = plan_units("suite", DEFAULT_SUITE_N, on_demand=True)
    check("asking for them adds exactly four units", len(asked), len(default) + 4)
    solo = {tag: (tasks, extra) for tag, tasks, extra, _m in asked}
    for task in SEARCH:
        check_true(f"{task} gets a unit of its own", solo.get(task, ([],))[0] == [f"{task}_mini"],
                   repr(solo.get(task)))
    check("and each image one carries its resolution",
          {t: solo[t][1] for t in ("vstar_bench", "hrbench_4k", "hrbench_8k")},
          {t: f"--max-pixels {NATIVE_MAX_PIXELS}"
           for t in ("vstar_bench", "hrbench_4k", "hrbench_8k")})

    # --max-pixels is per lmms-eval invocation and applies to every task in it,
    # so a unit whose tasks disagree would score one of them at the other's
    # resolution and say nothing. Nothing above can build one; this is the guard
    # that a future edit meets instead of a plausible wrong number.
    for tag, tasks, _extra, _m in plan_units("task", DEFAULT_SUITE_N, on_demand=True):
        resolutions = {BENCHMARKS[base_task(t)].get("max_pixels") for t in tasks}
        check_true(f"unit {tag} is internally consistent about resolution",
                   len(resolutions) == 1, repr(resolutions))

    check("extra_args is empty for a benchmark with no resolution of its own",
          extra_args("pope_mini"), "")


# ---------------------------------------------------------------- the profile

def test_profile_is_unchanged():
    print("the sample profile")
    sizes = task_sample_n({"natural": 300, "nonnatural": 100})
    check("V* is clamped to its 191-row split", sizes["vstar_bench_mini"], 191)
    check("HR-Bench is not, its split is 800", sizes["hrbench_4k_mini"], 300)

    # The destructive case. vstar's 191 inside the natural suite would make
    # as_suite_n report the pair (191, 300) -- a tuple, which is unequal to any
    # profile -- and every existing n300_100 result would be orphaned.
    check("an on-demand benchmark does not define the profile",
          as_suite_n(sizes), {"natural": 300, "nonnatural": 100})
    check("so the profile keeps its name", profile_name(sizes), "n300_100")
    check("and its directory", str(profile_dir("RUN/bench_eval", sizes)),
          "RUN/bench_eval/n300_100")
    check("the 100/100 default still has no subdirectory",
          str(profile_dir("RUN/bench_eval", task_sample_n(DEFAULT_SUITE_N))),
          "RUN/bench_eval")

    check("they are not offered as suite work either",
          [t for t in suite_tasks("natural") if base_task(t) in SEARCH], [])
    check("unless asked for",
          sorted(base_task(t) for t in suite_tasks("natural", on_demand=True)
                 if base_task(t) in SEARCH), sorted(SEARCH))

    check("an override spelled lmms-eval's way lands on our key",
          task_sample_n(DEFAULT_SUITE_N, {"hrbench4k": 50})["hrbench_4k_mini"], 50)


# -------------------------------------------------------------- the reduction

RESULTS = {
    "results": {
        "vstar_bench": {
            "alias": "vstar_bench",
            # The sub-columns the overall task emits as empty LISTS. They must be
            # skipped rather than crashing or being written as a scalar.
            "vstar_direct_attributes_acc,none": [],
            "vstar_relative_position_acc,none": [],
            "vstar_overall_acc,none": 79.58115183246073,
            "vstar_overall_acc_stderr,none": "N/A",
        },
        "hrbench4k": {"alias": "hrbench4k", "single,none": 0.8925,
                      "cross,none": 0.645, "average,none": 0.76875,
                      "average_stderr,none": "N/A"},
        "hrbench_8k_mini": {"alias": "hrbench_8k_mini", "average,none": 0.5,
                            "single,none": 0.6, "cross,none": 0.4},
    },
    "n-samples": {"vstar_bench": {"original": 191, "effective": 191},
                  "hrbench4k": {"original": 800, "effective": 800},
                  "hrbench_8k_mini": {"original": 300, "effective": 300}},
}


def test_reduction():
    print("the reduction")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "x_results.json"
        path.write_text(json.dumps(RESULTS))
        metrics = bench_eval.flatten([str(path)])

    check("V* is divided by 100", metrics["bench/natural/vstar_bench"],
          0.7958115183246073)
    check("HR-Bench is not", metrics["bench/natural/hrbench_4k"], 0.76875)
    check("the mini task reduces under the same key as the full one",
          metrics["bench/natural/hrbench_8k"], 0.5)
    check("the empty-list sub-columns are dropped",
          [k for k in metrics if "direct_attributes" in k], [])
    check("the sub-columns that ARE numbers are kept",
          metrics["bench/natural/hrbench_4k/cross"], 0.645)
    check("and none of them joins the suite mean",
          "bench/natural/mean" in metrics, False)
    check("n_benchmarks counts all three", metrics["bench/natural/n_benchmarks"], 3.0)

    # HR-Bench names its per-row verdict gpt_score, which has nothing to do with
    # GPT on a row the local parser could read. Without this every hrbench row
    # would be stored unscored, and bench_samples would silently have no items.
    check("a HR-Bench row is scorable",
          bench_samples.dict_score({"index": 3, "cycle_category": 0, "gpt_score": 1}), 1.0)
    check("a V* row is scorable",
          bench_samples.dict_score({"question_id": 1, "category": "direct_attributes",
                                    "score": 0.0}), 0.0)
    check("HR-Bench's headline is not the mean of those rows, and says so",
          BENCHMARKS["hrbench_4k"]["item_decomposes"], False)


# -------------------------------------------- the real tasks (--lmms-eval only)

def test_generated_configs():
    print(f"generating against {LMMS_EVAL_DIR}")
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            [sys.executable, str(HERE / "eval_mini" / "make_mini_tasks.py"),
             "--out-dir", tmp, "--lmms-eval-dir", LMMS_EVAL_DIR,
             "--natural-n", "300", "--nonnatural-n", "100"],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        written = sorted(p.name for p in Path(tmp).glob("*_mini.yaml"))
        check("a config is written for every search benchmark",
              [f"{t}_mini.yaml" for t in SEARCH if f"{t}_mini.yaml" in written],
              [f"{t}_mini.yaml" for t in SEARCH])

        sizes = json.loads((Path(tmp) / "sample_n.json").read_text())
        check("V*'s config samples its whole split", sizes["vstar_bench_mini"], 191)

        sampler = (Path(tmp) / "minisample.py").read_text()
        check_true("spatialtreebench gets the JSON-stratified sampler",
                   "take_for_spatialtreebench" in sampler
                   and '_take_json_strata(dataset, 300, "extra_info", "spatree")' in sampler,
                   sampler[-200:])
        check_true("V*'s config includes the real task yaml by absolute path",
                   f"{LMMS_EVAL_DIR}/lmms_eval/tasks/vstar_bench/vstar_bench.yaml"
                   in (Path(tmp) / "vstar_bench_mini.yaml").read_text())
        return tmp


def test_real_tasks():
    """Construct the tasks offline and re-derive `scale` from their own code.

    This is the check a results.json cannot give: it drives each task's own
    process_results and aggregation over a run that is right on every document
    and a run that is wrong on every document, so the top and the bottom of the
    scale are observed rather than inferred from one mid-range number.
    """
    import random

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    # lmms-eval redirects the datasets build cache to node-local /tmp when
    # $HF_HOME/datasets is on a remote filesystem, and offline that is a
    # ConnectionError before the first document. See run_bench_eval.sh.
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    os.environ.setdefault("LMMS_EVAL_DATASETS_CACHE", os.path.join(hf_home, "datasets"))

    from lmms_eval.tasks import TaskManager, get_task_dict

    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            [sys.executable, str(HERE / "eval_mini" / "make_mini_tasks.py"),
             "--out-dir", tmp, "--lmms-eval-dir", LMMS_EVAL_DIR, "--n", "300"],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        manager = TaskManager(model_name="qwen3_vl", include_path=tmp)

        print("the real tasks, offline")
        random.seed(0)
        for name, rows in (("vstar_bench_mini", 191), ("hrbench_4k_mini", 300)):
            task = list(get_task_dict([name], manager).values())[0]
            spec = BENCHMARKS[base_task(name)]
            key = spec["metric"].split(",")[0]
            check(f"{name} constructs and samples {rows} rows", len(task.eval_docs), rows)
            check_true(f"{name} inherits its metric list",
                       key in list(task._metric_fn_list), list(task._metric_fn_list))

            docs = list(task.eval_docs)[:40]
            for label, want in (("correct", 1.0), ("wrong", 0.0)):
                scored = []
                for doc in docs:
                    gold = (doc["label"] if name.startswith("vstar") else doc["answer"])
                    gold = str(gold).strip().upper()
                    pred = gold if label == "correct" else ("B" if gold != "B" else "C")
                    out = task.process_results(doc, [pred])
                    if key in out:
                        scored.append(out[key])
                aggregate = task.aggregation()[key]
                try:
                    value = aggregate(scored)
                except TypeError:          # hrbench's takes (results, args)
                    value = aggregate(scored, None)
                check(f"{name} all-{label} / scale", round(value / spec["scale"], 6), want)


def test_the_cap_is_actually_off():
    """What the model sees, in visual tokens, at each of the three settings.

    The table is about resolution and every other check here is about plumbing,
    so this is the one that says the plumbing achieved anything. Sizes are the
    real ones, measured off row 0 of each benchmark; the arithmetic is Qwen's own
    smart_resize at its 32 px merged patch, which is what process_vision_info
    calls.
    """
    from qwen_vl_utils.vision_process import smart_resize

    merge = 32

    def tokens(w, h, max_pixels):
        height, width = smart_resize(h, w, factor=merge,
                                     min_pixels=256 * 32 * 32, max_pixels=max_pixels)
        return (width // merge) * (height // merge)

    print("what the cap costs, in visual tokens")
    lmms_default = 2048 * 32 * 32
    for name, (w, h), floor in (("V*", (2000, 1500), 1.4),
                                ("HR-Bench 4K", (4032, 4032), 7.0),
                                ("HR-Bench 8K", (4992, 7680), 7.0)):
        at_default = tokens(w, h, lmms_default)
        at_native = tokens(w, h, NATIVE_MAX_PIXELS)
        check_true(f"{name}: native is >= {floor}x the default ({at_native} vs {at_default} tok)",
                   at_native >= floor * at_default, f"{at_native / at_default:.2f}x")
    # And the other direction: native must not EXCEED what the model can hold,
    # or the processor would be asked for an input the backbone cannot embed.
    check_true("native never exceeds the model's 16,384-token ceiling",
               tokens(4992, 7680, NATIVE_MAX_PIXELS) <= 16384,
               str(tokens(4992, 7680, NATIVE_MAX_PIXELS)))


def test_spatialtree_missing_leaf():
    """The number the stratified sampler exists to prevent.

    spatialtreebench's root score is a weighted mean over 27 leaves, and a leaf
    the sample missed contributes 0 rather than being left out. So the penalty
    for an unbalanced draw is not noise, it is a bias of tens of points, in the
    direction that reads as a worse model.
    """
    import contextlib
    import io

    sys.path.insert(0, LMMS_EVAL_DIR)
    from lmms_eval.tasks.spatialtreebench.utils import (spatialtreebench_aggregate_results,
                                                        spatree_hierarchy)

    print("spatialtreebench's weighted tree")
    leaves = []

    def walk(node, path):
        children = list(node.children.values())
        if not children:
            leaves.append(path + [node.name])
            return
        for child in children:
            walk(child, path + [node.name])

    walk(spatree_hierarchy, [])
    check("the hierarchy has 27 leaves", len(leaves), 27)

    def rows(score, keep=None):
        out = []
        for path in leaves[:keep]:
            tags = {f"spatree{i}": v.replace(" ", "") for i, v in enumerate(path[1:])}
            out.append({"score": score, **tags})
        return out

    def aggregate(items):            # its printer is a tree dump, not a result
        with contextlib.redirect_stdout(io.StringIO()):
            return spatialtreebench_aggregate_results(items)

    check("a perfect run over every leaf is 1.0", aggregate(rows(1.0)), 1.0)
    check("an empty run is 0.0", aggregate(rows(0.0)), 0.0)
    check_true("a perfect run missing 7 of 27 leaves reads far below 1.0 -- which is "
               "what stratify_json is for", aggregate(rows(1.0, keep=20)) < 0.7,
               f"{aggregate(rows(1.0, keep=20)):.4f}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lmms-eval", action="store_true",
                   help="also generate against the real clone and construct the tasks")
    args = p.parse_args()

    for test in (test_table, test_plan, test_profile_is_unchanged, test_reduction,
                 test_the_cap_is_actually_off):
        test()
    if args.lmms_eval:
        test_generated_configs()
        test_real_tasks()
        test_spatialtree_missing_leaf()

    print(f"\n{PASSED} passed, {FAILED} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
