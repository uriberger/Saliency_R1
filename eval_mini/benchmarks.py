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
             is a SUM over 14 sub-categories, and with stratify_by below the mini
             sample does cover all 14, so it lands on the published 0-2000 scale --
             but on 3 or 4 image pairs per category, where one flipped pair moves a
             category by tens of points. It is logged and watchable as a curve;
             it is too coarse to average into a suite headline.
    group_by (optional) a column whose rows must be sampled together. Absent for
             every benchmark that scores each row on its own, which is all of them
             but MME: MME scores an image from the yes/no pair sharing its
             question_id and asserts both are present, so a row-wise sample splits
             the pairs and the task raises during aggregation.
    stratify_by
             (optional) a column the sample is balanced across. Only meaningful
             with group_by. MME needs it because its score sums per-category
             averages: a proportional draw of 50 pairs from 14 unequal categories
             misses the small ones outright, and a missing category is not a
             smaller contribution but no contribution.
    item_metric
             (optional) the per-ROW key that carries this benchmark's verdict on a
             single document, when it is not `metric`. Only bench_samples.py reads
             it. p3 needs it because its headline `all_cat_f1` is a corpus-level
             F1 whose per-row value is a confusion-count dict, not a score, while
             `sample_f1` is the same judgement made one document at a time.
    item_decomposes
             (optional, default True) whether the mean of the per-item scores IS
             the headline metric. False where the aggregate is not an average:
             pope and p3 report corpus F1 while their rows are right or wrong, and
             MME sums per-category averages. A paired test over items is still the
             right test for "did this checkpoint get different documents right",
             but for these three it is a test on per-item accuracy, NOT on the
             number the curve plots -- and bench_samples.py says so rather than
             letting the two be read as the same quantity.
    local_data
             (optional) load the rows from a file inside the repo's snapshot
             instead of through `datasets`' hub resolution, as
             {repo_id, builder, file}. Only visulogic needs it, and only because
             the eval jobs run with HF_HUB_OFFLINE=1 on nodes with no internet:
             its yaml says `dataset_kwargs: {data_files: data.jsonl}`, and offline
             `datasets` does not resolve that against the hub snapshot -- it looks
             the build up in the cache under the literal config name
             `default-data_files=data.jsonl`, which nothing ever writes. The build
             that IS cached is called `default-<hash>`, so the lookup raises and,
             because lmms-eval constructs every task in a suite before running
             any, it took the whole non-natural suite with it. See
             make_mini_tasks.py for what is generated instead.

             Note the plain `default` config in that cache is NOT the benchmark
             (1003 rows of bare images, from the repo's assets/ and images.zip);
             the questions live only in data.jsonl. Dropping data_files to dodge
             the offline lookup would score a different dataset, quietly.

Every benchmark here scores itself: none calls an LLM judge, so the eval job needs
no API key and cannot be skewed by one being absent or rate-limited. Three of the
test benchmarks were left out for that reason, because all three fail *quietly* --
the judge call errors, the item is scored wrong, and the benchmark simply reads
low, which on a training curve is indistinguishable from a model that got worse:

    mathvista_testmini_cot   llm_as_judge_eval IS its primary metric
    dailyclue                exact/numeric/date shortcuts first, judges the rest
    omnispatial_test         calls server.evaluate on every item

All three stay in the full test suite, where a key is always supplied, so the
headline numbers are unaffected -- only the per-checkpoint monitor is narrower.
mathvision_testmini looks like a fourth case but is not: it uses
mathvision_process_results, while the judged function belongs to the
mathvision_reason_* tasks.
"""

BENCHMARKS = {
    # --- natural imagery ---
    "mme": dict(
        suite="natural", yaml="mme/mme.yaml",
        metric="mme_perception_score,none", scale=1.0, in_mean=False,
        group_by="question_id", stratify_by="category", item_decomposes=False),
    "mmerealworld": dict(
        suite="natural", yaml="mme_realworld/mme_realworld.yaml",
        metric="mme_realworld_score,none", scale=1.0, in_mean=True),
    "pope": dict(
        suite="natural", yaml="pope/pope.yaml",
        metric="pope_f1_score,none", scale=1.0, in_mean=True,
        item_metric="pope_accuracy", item_decomposes=False),
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
    # mathvista_testmini_cot is deliberately absent. Its primary metric IS
    # llm_as_judge_eval, so it cannot be scored at all without a judge API key, and
    # a per-checkpoint monitor should not depend on an external service. It stays in
    # the full test suite, where the key is always supplied.
    "mmmu_pro_standard": dict(
        suite="nonnatural", yaml="mmmu_pro/mmmu_pro_standard.yaml",
        metric="mmmu_acc,none", scale=1.0, in_mean=True),
    "p3": dict(
        suite="nonnatural", yaml="salbench/p3.yaml",
        metric="all_cat_f1,none", scale=1.0, in_mean=True,
        item_metric="sample_f1", item_decomposes=False),
    "scienceqa_img": dict(
        suite="nonnatural", yaml="scienceqa/scienceqa_img.yaml",
        metric="exact_match,none", scale=1.0, in_mean=True),
    "visulogic": dict(
        suite="nonnatural", yaml="visulogic/visulogic.yaml",
        metric="visulogic_acc,none", scale=1.0, in_mean=True,
        local_data=dict(repo_id="VisuLogic/VisuLogic", builder="json", file="data.jsonl")),
}

SUITES = ("natural", "nonnatural")

MINI_SUFFIX = "_mini"

# How many documents each suite is scored on. This is the SAMPLE PROFILE, and it
# is the one thing that must never be mixed within a curve: `bench/natural/mean`
# at 100 documents per benchmark and the same key at 300 are two different
# measurements of two different samples, and a plot with both on one line is the
# most likely way the precision work goes wrong.
#
# So the profile is not a loose convention -- it names where results are stored
# (profile_dir), what WandB key they are logged under (wandb_prefix), and it is
# recorded inside every step file (bench_eval.py) and every per-item harvest
# (bench_samples.py). Two results can only be compared when their profiles are
# equal, and every consumer checks rather than assumes.
DEFAULT_SUITE_N = {"natural": 100, "nonnatural": 100}


def as_suite_n(sample_n):
    """Normalise a sample-size mapping to {suite: n}.

    Accepts either spelling, because both occur: the per-SUITE profile a job is
    launched with, and the per-TASK sizes lmms-eval records in results.json
    (`n-samples`). Taking either means a caller never has to convert, which is
    what stops the two drifting into different answers about which profile a
    result belongs to.

    A suite whose tasks disagree about their size has no single number, so it is
    reported as the sorted list of what was seen -- unequal, therefore unequal to
    any profile, therefore refused by whoever compares them. That is the intended
    outcome: a half-resampled suite is not a sample size.
    """
    if not sample_n:
        return dict(DEFAULT_SUITE_N)
    if set(sample_n) <= set(SUITES):
        return {s: sample_n.get(s, DEFAULT_SUITE_N[s]) for s in SUITES}
    out = {}
    for suite in SUITES:
        sizes = sorted({n for task, n in sample_n.items()
                        if (BENCHMARKS.get(base_task(task)) or {}).get("suite") == suite})
        if not sizes:
            out[suite] = DEFAULT_SUITE_N[suite]
        elif len(sizes) == 1:
            out[suite] = sizes[0]
        else:
            out[suite] = tuple(sizes)
    return out


def profile_name(sample_n):
    """A short, sortable name for one sample profile, e.g. `n300_100`."""
    suite_n = as_suite_n(sample_n)
    parts = []
    for suite in SUITES:
        value = suite_n[suite]
        parts.append("-".join(str(v) for v in value) if isinstance(value, tuple) else str(value))
    return "n" + "_".join(parts)


def profile_dir(bench_dir, sample_n):
    """Where results for one sample profile live, under a run's bench_eval/.

    The default profile keeps the flat layout it has always had --
    `bench_eval/step-<N>.json` -- and every other profile gets a subdirectory of
    its own. That asymmetry is deliberate and is the whole reason nothing had to
    be migrated: the trainer's BenchmarkResultsCallback globs
    `bench_eval/step-*.json` non-recursively while a run is training, so a new
    profile is invisible to it, and 258 existing results (plus whatever an
    in-flight eval job is writing right now) stay exactly where every existing
    reader looks for them.
    """
    from pathlib import Path

    bench_dir = Path(bench_dir)
    if as_suite_n(sample_n) == DEFAULT_SUITE_N:
        return bench_dir
    return bench_dir / profile_name(sample_n)


def wandb_prefix(sample_n):
    """The metric namespace one profile logs under.

    The default profile keeps `bench/*`, so every panel that exists goes on
    working. Anything else gets its own top level -- `bench_n300_100/*` -- which
    no `bench/*` glob can reach. Putting the size deeper in the key
    (`bench/n300/natural/mean`) would have been tidier to read and would have let
    a `bench/*` panel pick up both sizes at once, which is exactly the plot this
    is here to make impossible.
    """
    if as_suite_n(sample_n) == DEFAULT_SUITE_N:
        return "bench"
    return f"bench_{profile_name(sample_n)}"


def profile_of_payload(payload):
    """The profile a step file was written at.

    Files written before sample sizes were recorded carry no `sample_n`, and they
    are all n=100: it was the only value either dispatcher ever passed, and every
    results.json on disk agrees. Reading them as the default profile is therefore
    a statement of fact, not a fallback.
    """
    return as_suite_n((payload or {}).get("sample_n"))


def task_sample_n(suite_n=None, overrides=None):
    """{mini task: n} for every benchmark, from a per-suite profile.

    `overrides` is keyed by task name with or without the _mini suffix, and wins
    over the suite default. It exists because the natural suite's cost is not
    spread evenly -- mmstar generates ~4x the tokens per document that pope does,
    so at n=300 it alone approaches a one-hour allocation -- and capping one task
    is a better answer than lengthening the job.
    """
    suite_n = as_suite_n(suite_n or DEFAULT_SUITE_N)
    out = {}
    for task, spec in BENCHMARKS.items():
        value = suite_n[spec["suite"]]
        out[f"{task}{MINI_SUFFIX}"] = value if not isinstance(value, tuple) else value[0]
    for task, n in (overrides or {}).items():
        name = task if task.endswith(MINI_SUFFIX) else f"{task}{MINI_SUFFIX}"
        if base_task(name) not in BENCHMARKS:
            raise SystemExit(f"no such benchmark to override: {task}")
        out[name] = int(n)
    return out


# ---------------------------------------------------------------- what it costs
#
# Measured, not guessed: over the 295 mini-suite results.json on disk, single-GPU
# wall clock fits `minutes = 1.5 + output_tokens / 3310` with the SAME token rate
# for all three suites (natural 3323, mme-realworld 3304, non-natural 3311). So
# the cost of a task is its generated tokens, and nothing else -- document count
# only matters through them.
#
# This exists because the allocation is one GPU for one hour and the banking unit
# has to fit inside it. A unit started and killed at the wall clock banks nothing
# and is redone from scratch, so the guard that decides whether to start one needs
# a real number, and "a suite takes 15-25 minutes" stopped being one the moment
# the natural suite tripled.
TOKENS_PER_MINUTE = 3300
# Loading the merged model into the worker, from the regression intercept
# (natural 1.1, mme-realworld 2.1, non-natural 0.8). Independent of the allocation
# size, which is why sharding the banking unit finer is not free.
MODEL_LOAD_MINUTES = 1.5

# Generation minutes per 100 documents, on 1 GPU. For the five natural benchmarks
# these come from co-measured `efficiency.by_task` token counts over 104-111
# invocations, divided by the rate above; the four that share an invocation sum to
# 29.0 against a measured suite median of 29.8 including load, and mme-realworld's
# 9.1 matches its measured 11.2 minus its own 2.1 of load.
#
# The non-natural benchmarks are given the suite's measured time spread evenly,
# NOT a token figure. Six of the eight keep only `filtered_resps`, so lmms-eval
# counts no tokens for them and the per-task medians that do exist contradict the
# suite total -- algopuzzlevqa's alone exceeds it. An even split of a number that
# was actually observed is worth more than three precise figures and five zeros.
MINUTES_PER_100 = {
    "mme": 4.3,
    "mmerealworld": 9.1,
    "mmstar": 16.3,
    "pope": 3.5,
    "realworldqa": 4.9,
}
NONNATURAL_MINUTES_PER_100 = 22.9 / 8

# How far above the median to plan. The measured p90 of a suite is ~1.3x its
# median, and the two errors are not symmetric: a unit started and killed at the
# wall clock banks nothing and is repeated from scratch, while a unit not started
# costs an early exit and a resubmission, which the dispatcher does anyway.
SAFETY_FACTOR = 1.3


def estimate_minutes(tasks, sample_n=None, num_gpus=1, load=True):
    """Wall clock one lmms-eval invocation over `tasks` should need, pessimistically."""
    sizes = sample_n if isinstance(sample_n, dict) and not set(sample_n) <= set(SUITES) \
        else task_sample_n(sample_n)
    minutes = MODEL_LOAD_MINUTES if load else 0.0
    for task in tasks:
        spec = BENCHMARKS.get(base_task(task))
        if spec is None:
            continue
        n = sizes.get(task if task.endswith(MINI_SUFFIX) else f"{task}{MINI_SUFFIX}", 100)
        rate = MINUTES_PER_100.get(base_task(task), NONNATURAL_MINUTES_PER_100)
        minutes += rate * n / 100 / max(1, num_gpus)
    return minutes * SAFETY_FACTOR


# MME-RealWorld's images are ~36 MP and the wrapper's default downsamples them
# ~22x, which makes the model over-abstain. The test suite runs it at 3.2 M pixels
# and so does this, which a combined invocation cannot do -- hence its own unit.
MMERW_ARGS = "--max-pixels 3211264"


def plan_units(bank="suite", sample_n=None, num_gpus=1):
    """The list of things a job can finish and bank, largest first.

    A unit is one lmms-eval invocation whose results are recorded the moment it
    succeeds, so a job killed at the wall clock loses only whatever it was in the
    middle of. Which is why the granularity is a parameter: at 100 documents a
    suite is 30 minutes and fits an hour, but the natural suite at 300 is 105 and
    can never finish inside one -- it would be started, killed and repeated
    forever, banking nothing. Splitting it per task turns that into five units of
    16-65 minutes at a cost of four extra model loads, ~6 minutes.

    Largest first because the merge is charged to whichever unit runs first: the
    packing that follows (take the next unit that fits, rather than stopping at
    the first that does not) then puts the expensive units in the jobs that can
    hold them and fills the rest with the cheap ones.

    Returns [(tag, [tasks], extra_args, minutes)].
    """
    natural = [t for t in suite_tasks("natural") if base_task(t) != "mmerealworld"]
    if bank == "suite":
        groups = [("natural", natural, ""),
                  ("mmerw", ["mmerealworld_mini"], MMERW_ARGS),
                  ("nonnatural", suite_tasks("nonnatural"), "")]
    elif bank == "task":
        groups = [(base_task(t), [t], MMERW_ARGS if base_task(t) == "mmerealworld" else "")
                  for t in suite_tasks("natural") + suite_tasks("nonnatural")]
    else:
        raise SystemExit(f"unknown banking unit: {bank!r} (suite | task)")
    units = [(tag, tasks, extra, estimate_minutes(tasks, sample_n, num_gpus))
             for tag, tasks, extra in groups]
    return sorted(units, key=lambda u: -u[3])


def suite_tasks(suite):
    """Mini task names for one suite, in table order."""
    return [f"{t}{MINI_SUFFIX}" for t, spec in BENCHMARKS.items() if spec["suite"] == suite]


def base_task(task):
    """The real benchmark a mini task name refers to."""
    return task[: -len(MINI_SUFFIX)] if task.endswith(MINI_SUFFIX) else task
