#!/usr/bin/env python
"""CPU tests for the sample-profile machinery and the per-item store.

    python test_bench_precision_cpu.py

Everything here is about one property: a result measured on 100 documents and a
result measured on 300 must never end up on the same curve, in the same file, or
joined to each other. The failure is silent by nature -- both numbers are valid
measurements of something, they just are not measurements of the same thing -- so
the checks are on the refusals, not only on the happy paths.

Nothing here needs a GPU, an lmms-eval clone, or the cluster.
"""

import gzip
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(HERE / "eval_mini"))
sys.path.insert(0, str(HERE))

from benchmarks import (DEFAULT_SUITE_N, as_suite_n, estimate_minutes,  # noqa: E402
                        plan_units, profile_dir, profile_name, task_sample_n,
                        wandb_prefix)
import bench_eval  # noqa: E402
import bench_samples  # noqa: E402

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


# --------------------------------------------------------------- the profile

def test_profiles():
    print("profiles")
    n300 = {"natural": 300, "nonnatural": 100}
    check("default profile is named n100_100", profile_name(DEFAULT_SUITE_N), "n100_100")
    check("300/100 profile is named n300_100", profile_name(n300), "n300_100")

    # The default profile must keep the flat layout: 258 results already live
    # there and the trainer's callback globs it non-recursively while training.
    check("default keeps the flat directory",
          profile_dir("run/bench_eval", DEFAULT_SUITE_N), Path("run/bench_eval"))
    check("300/100 gets its own directory",
          profile_dir("run/bench_eval", n300), Path("run/bench_eval/n300_100"))

    # Key namespaces must not be reachable by one glob.
    check("default logs under bench/", wandb_prefix(DEFAULT_SUITE_N), "bench")
    check("300/100 logs under its own top level", wandb_prefix(n300), "bench_n300_100")

    # Per-task sizes and per-suite sizes are the same statement.
    sizes = task_sample_n(n300)
    check("per-task sizes round-trip to the suite profile", as_suite_n(sizes), n300)
    check("natural tasks get 300", sizes["mmstar_mini"], 300)
    check("non-natural tasks stay at 100", sizes["chartqa_mini"], 100)

    # A step file written before sample sizes were recorded is a 100/100 result.
    check("a legacy payload reads as the default profile",
          bench_eval.profile_of_payload({"step": 1, "metrics": {}}), DEFAULT_SUITE_N)

    # An override is a profile of its own, not a rounding of a uniform one.
    capped = task_sample_n(n300, {"mmstar": 200})
    check("a capped benchmark is honoured", capped["mmstar_mini"], 200)
    check("a capped benchmark makes the suite size a set",
          as_suite_n(capped)["natural"], (200, 300))
    check("and therefore a directory of its own",
          profile_name(capped), "n200-300_100")


# ------------------------------------------------------------- the refusals

def test_refuses_to_mix():
    print("refusing to mix sample sizes")
    hundred = {t: 100 for t in task_sample_n(DEFAULT_SUITE_N)}
    three = task_sample_n({"natural": 300, "nonnatural": 100})

    # All five natural benchmarks complain, and only those: the non-natural half
    # is asked for at the size it already has.
    check("100-document results into a 300 profile: refused",
          len(bench_eval.check_profile(hundred, three)), 5)
    check("300-document results into a 100 profile: refused",
          len(bench_eval.check_profile(three, hundred)), 5)
    check("matching sizes: accepted", bench_eval.check_profile(three, three), [])
    check("the non-natural half is unchanged, so it never complains",
          [p for p in bench_eval.check_profile(hundred, three) if "chartqa" in p], [])
    # "fewer than asked for" must NOT be waved through: a suite banked at 100 and
    # reused by a job wanting 300 looks exactly like a small split.
    check("fewer documents than asked for is still a refusal",
          len(bench_eval.check_profile({"mmstar_mini": 100}, {"mmstar_mini": 300})), 1)
    check("...unless explicitly allowed",
          bench_eval.check_profile({"mmstar_mini": 100}, {"mmstar_mini": 300},
                                   allow_short=True), [])


def test_partial_collect_cannot_clobber():
    """Every natural unit failing must not file the result as a 100-document one.

    The dangerous path: the non-natural suite is carried over from the existing
    n=100 results, every natural unit fails, and the only sizes observed are 100.
    Inferring the profile from those would write a partial 300-document result
    into bench_eval/step-N.json -- on top of the real 100-document result.
    """
    print("a partial collect cannot overwrite the 100-document curve")
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "run"
        (run / "bench_eval").mkdir(parents=True)
        (run / "bench_eval" / "step-500.json").write_text(
            json.dumps({"step": 500, "metrics": {"bench/natural/mean": 0.71}}))

        results = Path(tmp) / "nonnatural_results.json"
        results.write_text(json.dumps({
            "results": {"chartqa_mini": {"relaxed_overall,none": 0.5}},
            "n-samples": {"chartqa_mini": {"original": 100, "effective": 100}}}))

        wanted = task_sample_n({"natural": 300, "nonnatural": 100})
        subprocess.run(
            [sys.executable, str(HERE / "bench_eval.py"), "--collect",
             "--run-dir", str(run), "--step", "500",
             "--sample-n", json.dumps(wanted), "--results", str(results)],
            check=True, capture_output=True)

        survivor = json.loads((run / "bench_eval" / "step-500.json").read_text())
        check("the 100-document result is untouched",
              survivor["metrics"].get("bench/natural/mean"), 0.71)
        check("the partial result went to the 300-document directory",
              (run / "bench_eval" / "n300_100" / "step-500.json").exists(), True)


# --------------------------------------------------------------- the samples

def test_item_scores():
    """Every per-row shape lmms-eval emits has to yield a verdict."""
    print("per-item scores")
    check("plain {'score': ...}", bench_samples.dict_score({"score": 1.0}), 1.0)
    check("mme-realworld's pred_answer/answer",
          bench_samples.dict_score({"pred_answer": "A", "answer": "A"}), 1.0)
    check("...and when it is wrong",
          bench_samples.dict_score({"pred_answer": "B", "answer": "A"}), 0.0)
    check("mmmu_pro's parsed_pred/answer",
          bench_samples.dict_score({"parsed_pred": "E", "answer": "B"}), 0.0)
    check("mathvision's scores list",
          bench_samples.dict_score({"scores": [True], "response": ["C"]}), 1.0)
    check("a shape with no verdict is unscored, not zero",
          bench_samples.dict_score({"pred": {"true_pos": 0}}), None)

    # p3's headline is a corpus F1 whose per-row value is a confusion table; the
    # table declares sample_f1 as the per-item judgement instead.
    row = {"doc_id": 0, "all_cat_f1": {"pred": {}}, "sample_f1": {"score": 0.25},
           "exact_match": {"score": 0}, "input": "q", "target": "size"}
    from benchmarks import BENCHMARKS
    metric, score, _cat = bench_samples.primary_score(row, BENCHMARKS["p3"])
    check("p3 falls back to its declared item metric", (metric, score), ("sample_f1", 0.25))

    # MME's rows carry perception OR cognition depending on the document, and the
    # aggregate is read from perception; the single verdict present is the answer.
    row = {"doc_id": 0, "mme_cognition_score": {"score": 1.0, "category": "code_reasoning"},
           "input": "q", "target": "Yes"}
    metric, score, cat = bench_samples.primary_score(row, BENCHMARKS["mme"])
    check("mme uses whichever half the document is in",
          (metric, score, cat), ("mme_cognition_score", 1.0, "code_reasoning"))


def test_fingerprint():
    print("item keys")
    a = {"doc_id": 0, "input": "Is there a snowboard?", "target": "yes",
         "pope_f1_score": {"question_id": "2575", "score": 1.0}}
    b = dict(a, doc_id=97, resps="a completely different answer", pope_f1_score={
        "question_id": "2575", "score": 0.0})
    check("the key is the document, not the position or the answer",
          bench_samples.fingerprint(a), bench_samples.fingerprint(b))
    c = dict(a, input="Is there a skateboard?")
    check_true("a different document gets a different key",
               bench_samples.fingerprint(a) != bench_samples.fingerprint(c))


def test_join_refuses_mismatched_items():
    """Two harvests of different samples must not be silently paired."""
    print("joining")
    left = [{"task": "pope_mini", "doc_id": i, "item": f"same{i}", "score": 1.0}
            for i in range(5)]
    right = [dict(r) for r in left]
    pairs, _, _ = bench_samples.joined(left, right, {"pope_mini"})
    check("identical samples join on every item", len(pairs), 5)

    right[3]["item"] = "a different document"
    try:
        bench_samples.joined(left, right, {"pope_mini"})
        check("a position holding different documents is refused", "no error", "SystemExit")
    except SystemExit as exc:
        check_true("a position holding different documents is refused",
                   "nothing here is paired" in str(exc), str(exc))


def test_harvest_round_trip():
    print("the per-item store")
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "run"
        (run / "bench_eval").mkdir(parents=True)
        stamp = "20260101_000000"
        results = Path(tmp) / f"{stamp}_results.json"
        results.write_text(json.dumps({
            "results": {"pope_mini": {"pope_f1_score,none": 0.84}},
            "n-samples": {"pope_mini": {"original": 100, "effective": 100}}}))
        samples = Path(tmp) / f"{stamp}_samples_pope_mini.jsonl"
        samples.write_text("\n".join(json.dumps({
            "doc_id": i, "input": f"q{i}", "target": "yes", "resps": "x" * 5000,
            "filtered_resps": "yes",
            "pope_f1_score": {"question_id": str(i), "score": float(i % 2)}})
            for i in range(100)))

        dest, missing = bench_samples.harvest(run, 700, [str(results)], quiet=True)
        check("nothing was left unharvested", missing, [])
        header, rows = bench_samples.read_harvest(dest)
        check("every row is stored", len(rows), 100)
        check("the sizes come from the results file", header["sample_n"], {"pope_mini": 100})
        check("responses are truncated", len(rows[0]["resp"]), bench_samples.RESP_CHARS)
        check("scores survive", sum(r["score"] for r in rows), 50.0)
        check("it lands beside the step file, under the profile",
              dest.parent.parent, run / "bench_eval")

        # A sample file from a DIFFERENT invocation in the same directory must not
        # be picked up: output directories accumulate one set per re-run.
        other = Path(tmp) / "20260202_000000_samples_pope_mini.jsonl"
        other.write_text(json.dumps({"doc_id": 0, "input": "q0", "target": "no"}))
        check("only this invocation's rows are read",
              sorted(bench_samples.sample_files(str(results))), ["pope_mini"])


# ------------------------------------------------------------------ the plan

def test_units_and_cost():
    print("banking units and their cost")
    suite = plan_units("suite", DEFAULT_SUITE_N)
    check("suite banking gives three units", [u[0] for u in suite],
          ["natural", "nonnatural", "mmerw"])
    check("mme-realworld keeps its own resolution",
          [u[2] for u in suite if u[0] == "mmerw"], ["--max-pixels 3211264"])
    check("units are ordered largest first",
          [u[3] for u in suite], sorted([u[3] for u in suite], reverse=True))

    task = plan_units("task", DEFAULT_SUITE_N)
    check("task banking gives one unit per benchmark", len(task), 13)

    # The estimator is checked against what was actually measured on 1 GPU: the
    # natural suite's median is 29.8 min and its p90 is 39.6, and the guard is
    # meant to sit at the p90 -- a unit started and killed banks nothing.
    natural = estimate_minutes(["mme_mini", "mmstar_mini", "pope_mini", "realworldqa_mini"])
    check_true(f"the natural suite at n=100 is budgeted near its measured p90 ({natural:.0f} vs 39.6)",
               38 <= natural <= 42, f"{natural:.1f}")
    mmerw = estimate_minutes(["mmerealworld_mini"])
    check_true(f"mme-realworld at n=100 is budgeted above its median ({mmerw:.0f} vs 11.2)",
               11 <= mmerw <= 17, f"{mmerw:.1f}")
    # And the finding that forced task banking in the first place.
    tripled = estimate_minutes(["mme_mini", "mmstar_mini", "pope_mini", "realworldqa_mini"],
                               {"natural": 300, "nonnatural": 100})
    check_true(f"the natural suite at n=300 cannot fit a one-hour job ({tripled:.0f} min)",
               tripled > 55, f"{tripled:.1f}")


def main():
    for test in (test_profiles, test_refuses_to_mix, test_partial_collect_cannot_clobber,
                 test_item_scores, test_fingerprint, test_join_refuses_mismatched_items,
                 test_harvest_round_trip, test_units_and_cost):
        test()
    print(f"\n{PASSED} passed, {FAILED} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
