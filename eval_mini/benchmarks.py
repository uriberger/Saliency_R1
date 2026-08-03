"""The mini test suites: which benchmark is in which suite, and how it is scored.

Single source of truth, imported by both make_mini_tasks.py (which generates the
lmms-eval configs) and bench_eval.py (which reduces the results to WandB scalars).
Splitting this table in two would let a benchmark be evaluated as part of one suite
and reported under the other, which is exactly the kind of error a training curve
would never make obvious.

Fields:
    suite    natural | nonnatural -- the imagery type the benchmark tests
    yaml     the real task's config, relative to lmms_eval/tasks/. A task's yaml is
             not always under a directory named after it (mmerealworld lives in
             mme_realworld/, p3 in salbench/).
    metric   the results.json key the benchmark is headlined by
    scale    divisor that puts `metric` on 0-1. Several of these report a
             percentage while their neighbours report a fraction; the values are
             taken from what the tasks actually emitted in results/lmms_eval, not
             from the yaml.
    in_mean  whether it contributes to the suite average. MME does not: its score
             is a SUM over 14 sub-categories, so on a 100-document sample it is not
             on the 0-2000 scale at all and normalising it would contribute a
             number that is simply wrong. It is still logged -- comparable from
             checkpoint to checkpoint, just not to a published MME score.
"""

BENCHMARKS = {
    # --- natural imagery ---
    "dailyclue": dict(
        suite="natural", yaml="dailyclue/dailyclue.yaml",
        metric="dailyclue_overall,none", scale=100.0, in_mean=True),
    "mme": dict(
        suite="natural", yaml="mme/mme.yaml",
        metric="mme_perception_score,none", scale=1.0, in_mean=False),
    "mmerealworld": dict(
        suite="natural", yaml="mme_realworld/mme_realworld.yaml",
        metric="mme_realworld_score,none", scale=1.0, in_mean=True),
    "omnispatial_test": dict(
        suite="natural", yaml="omnispatial/omnispatial.yaml",
        metric="omnispatial,none", scale=1.0, in_mean=True),
    "pope": dict(
        suite="natural", yaml="pope/pope.yaml",
        metric="pope_f1_score,none", scale=1.0, in_mean=True),
    "realworldqa": dict(
        suite="natural", yaml="realworldqa/realworldqa.yaml",
        metric="exact_match,none", scale=1.0, in_mean=True),
    "mmstar": dict(
        suite="natural", yaml="mmstar/mmstar.yaml",
        metric="average,none", scale=1.0, in_mean=True),

    # --- non-natural imagery ---
    "algopuzzlevqa": dict(
        suite="nonnatural", yaml="algopuzzlevqa/algopuzzlevqa.yaml",
        metric="exact_match,none", scale=1.0, in_mean=True),
    "chartqa": dict(
        suite="nonnatural", yaml="chartqa/chartqa.yaml",
        metric="relaxed_overall,none", scale=1.0, in_mean=True),
    "illusionvqa_soft_localization": dict(
        suite="nonnatural", yaml="illusionvqa/illusionvqa_soft_localization.yaml",
        metric="exact_match,flexible-extract", scale=1.0, in_mean=True),
    "mathvision_testmini": dict(
        suite="nonnatural", yaml="mathvision/mathvision_testmini.yaml",
        metric="mathvision_standard_eval,none", scale=100.0, in_mean=True),
    "mathvista_testmini_cot": dict(
        suite="nonnatural", yaml="mathvista/mathvista_testmini_cot.yaml",
        metric="llm_as_judge_eval,none", scale=100.0, in_mean=True),
    "mmmu_pro_standard": dict(
        suite="nonnatural", yaml="mmmu_pro/mmmu_pro_standard.yaml",
        metric="mmmu_acc,none", scale=1.0, in_mean=True),
    "p3": dict(
        suite="nonnatural", yaml="salbench/p3.yaml",
        metric="all_cat_f1,none", scale=1.0, in_mean=True),
    "scienceqa_img": dict(
        suite="nonnatural", yaml="scienceqa/scienceqa_img.yaml",
        metric="exact_match,none", scale=1.0, in_mean=True),
    "visulogic": dict(
        suite="nonnatural", yaml="visulogic/visulogic.yaml",
        metric="visulogic_acc,none", scale=1.0, in_mean=True),
}

SUITES = ("natural", "nonnatural")

MINI_SUFFIX = "_mini"


def suite_tasks(suite):
    """Mini task names for one suite, in table order."""
    return [f"{t}{MINI_SUFFIX}" for t, spec in BENCHMARKS.items() if spec["suite"] == suite]


def base_task(task):
    """The real benchmark a mini task name refers to."""
    return task[: -len(MINI_SUFFIX)] if task.endswith(MINI_SUFFIX) else task
