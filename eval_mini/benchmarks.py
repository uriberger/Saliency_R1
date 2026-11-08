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
    max_pixels
             (optional) the input resolution this benchmark MUST be scored at, in
             pixels, passed through as `--max-pixels`. See "the resolution trap"
             below -- this is the single most consequential field in the table and
             the one whose absence silently produces a number that looks like a
             benchmark score and is not.
    own_unit (optional, default True where `max_pixels` is set) this benchmark
             cannot share an lmms-eval invocation with its neighbours, so it gets
             a banking unit of its own. `--max-pixels` and `--model_args` are
             per-INVOCATION and apply to every task in it, so two benchmarks
             wanting different resolutions in one `--tasks a,b` call cannot both
             be served; plan_units() refuses rather than picking one.
    on_demand
             (optional, default False) not part of the default plan. The four
             visual-search benchmarks are here so that bench_eval.py can reduce
             them and a baseline row has somewhere to live, but they are NOT run
             on every checkpoint: at native resolution HR-Bench alone is ~4x the
             cost of the whole natural suite, and a one-hour eval job that cannot
             finish its own unit banks nothing. Ask for them with
             `run_bench_eval.sh --search` (bench_eval.py --plan --on-demand).

             An on-demand benchmark also does NOT define the sample profile --
             see as_suite_n(). That is load-bearing: vstar_bench's split is 191
             rows, so at the standard `--natural-n 300` the natural suite would
             contain two different sizes, as_suite_n would report the pair as a
             tuple, and every existing run's profile would silently rename from
             `n300_100` to `n191-300_100` -- orphaning 339 step files and the
             `bench_n300_100/*` WandB namespace.
    split_n  (optional) the number of rows in the benchmark's test split, where
             that is smaller than a sample profile might ask for. The sampler
             returns a short split whole rather than padding it, so `wanted` has
             to expect the short number or bench_eval.py's profile check refuses
             the collect. Only vstar_bench needs it (191 rows); check_profile()
             pre-registered this case and said it would be an explicit act.
    stratify_json
             (optional) `(column, prefix)`: balance the sample across a value
             buried inside a JSON-encoded column, taking the deepest key whose
             name starts with `prefix`. stratify_by cannot reach it because it
             indexes `dataset[key]` directly. spatialtreebench needs it -- see
             its entry.
    needs_judge
             (optional) this benchmark calls an LLM judge on some rows, so the
             job needs OPENAI_API_KEY / NVIDIA_API_KEY and the number is only as
             good as the gateway was that day. Recorded rather than forbidden,
             because the alternative for spatialtreebench is not having it.

Almost every benchmark here scores itself: it needs no API key and cannot be
skewed by one being absent or rate-limited. Three of the test benchmarks were left
out for that reason, because all three fail *quietly* -- the judge call errors, the
item is scored wrong, and the benchmark simply reads low, which on a training curve
is indistinguishable from a model that got worse:

    mathvista_testmini_cot   llm_as_judge_eval IS its primary metric
    dailyclue                exact/numeric/date shortcuts first, judges the rest
    omnispatial_test         calls server.evaluate on every item

All three stay in the full test suite, where a key is always supplied, so the
headline numbers are unaffected -- only the per-checkpoint monitor is narrower.
mathvision_testmini looks like a fourth case but is not: it uses
mathvision_process_results, while the judged function belongs to the
mathvision_reason_* tasks.

The two exceptions are flagged with `needs_judge` and are both on-demand:
hrbench_4k/8k fall through to a judge only when they cannot read an option letter
out of the text (their utils.py strips the <think> block first, which is what makes
that rare), and spatialtreebench judges 374 of its 6,300 rows outright.

---------------------------------------------------------------- the resolution trap

The four visual-search benchmarks exist to test whether a model can find a small
thing in a large picture. Scored at a reduced resolution they measure something
else, and they do it without complaining:

    V*               191 rows, median 3.4 MP, max 8.3 MP
    HR-Bench 4K      800 rows, 16.3 MP   (NOT 3840x2160 -- the HF rows are larger)
    HR-Bench 8K      800 rows, 38.8 MP median, up to 59.0 MP

There are two caps on the path, and only one of them is obvious:

 1. `lmms_eval/models/simple/qwen3_vl.py:89` defaults `max_pixels` to
    2048*32*32 = 2,097,152. This is the one that bites. It applies to every task
    that does not ask for something else, and at 2.1 MP HR-Bench 8K is
    downsampled ~18x and every single V* image is downsampled.
 2. `vlm_reasoning/scripts/slurm/launch_lmms_eval_job.sh` raises it to 3,211,264
    for `--tasks mmerealworld|hrbench4k|hrbench8k|vstar_bench` -- matched against
    the WHOLE --tasks string, so it never fires for a mini task (`vstar_bench_mini`)
    and never fires for a combined invocation.

So `max_pixels` below is set to 16,777,216, which is Qwen3-VL-8B-Instruct's own
`preprocessor_config.json: size.longest_edge` -- i.e. no cap beyond the one the
model itself imposes. At that setting V* and HR-Bench 4K are not resized at all,
and HR-Bench 8K is reduced to the largest input the model can represent (16,384
visual tokens). That is what "native resolution" can mean for this backbone.

A third cap exists elsewhere and is NOT on this path, but is worth naming because
it has already produced numbers that were read as benchmark scores: the Figure-1
chain scans (`build_bench_candidates.py`, `--max-image-side 1024`) grade
`hrbench4k_mini` / `vstar_mini` at a 1,024 px long side. Those are a valid
ours-vs-coldstart signal under a fixed cap and nothing more. See
docs/deepeyes-gate.md §3.
"""

# Qwen3-VL-8B-Instruct's own ceiling, from its preprocessor_config.json
# (`size.longest_edge`), i.e. 16384 visual tokens at its 32x32 px merged patch.
# Passing this as --max-pixels means "do not cap below what the model can read",
# which is the only defensible meaning of "native resolution" for a model whose
# processor caps the input itself. It is NOT Qwen2.5-VL's 12,845,056 (16384*28*28)
# -- that figure appears in launch_lmms_eval_job.sh's comments and belongs to the
# previous generation's 28x28 patch.
NATIVE_MAX_PIXELS = 16_777_216

# MME-RealWorld's images are ~36 MP and the wrapper's default downsamples them
# ~22x, which makes the model over-abstain. 3.2 M is the value the full test suite
# has always used and the one its published 58.7 was replicated at; it is left
# exactly where it was rather than raised to NATIVE_MAX_PIXELS, because every
# mmerealworld number on every existing curve was measured there.
MMERW_MAX_PIXELS = 3_211_264

BENCHMARKS = {
    # --- natural imagery ---
    "mme": dict(
        suite="natural", yaml="mme/mme.yaml",
        metric="mme_perception_score,none", scale=1.0, in_mean=False,
        group_by="question_id", stratify_by="category", item_decomposes=False),
    "mmerealworld": dict(
        suite="natural", yaml="mme_realworld/mme_realworld.yaml",
        metric="mme_realworld_score,none", scale=1.0, in_mean=True,
        max_pixels=MMERW_MAX_PIXELS),
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

    # --- visual search: natural imagery, but on demand and at native resolution ---
    #
    # These four are the canonical visual-search benchmarks and the ones the
    # DeepEyes / ERGO / Mini-o3 / Monet tables are headlined by. They are in this
    # table so that bench_eval.py reduces them and a baseline row sits beside the
    # arms, NOT so that they join the per-checkpoint curve: `in_mean=False` keeps
    # `bench/natural/mean` the same four benchmarks it has always been (changing
    # it would redefine the headline on 339 existing step files), and
    # `on_demand=True` keeps them out of every routine one-hour eval job.
    #
    # `scale` is from a real results.json in each case -- vstar reports a
    # PERCENTAGE while hrbench reports a fraction, and the yamls say nothing about
    # either. Full-split r1-mode runs at 3.2 MP are on disk under
    # vlm_reasoning/results/lmms_eval/ for every arm; see docs/search-benchmarks.md
    # for the table they produce.
    "vstar_bench": dict(
        suite="natural", yaml="vstar_bench/vstar_bench.yaml",
        metric="vstar_overall_acc,none", scale=100.0, in_mean=False,
        on_demand=True, split_n=191, max_pixels=NATIVE_MAX_PIXELS),
    # 71.2 is V*'s OVERALL figure and 73.9 is its direct_attributes sub-column;
    # vstar_overall_acc is the former, which is also what DeepEyes' own
    # judge_result.py pools. The two sub-metrics (vstar_direct_attributes_acc,
    # vstar_relative_position_acc) are emitted as EMPTY LISTS by the overall task
    # and are skipped by flatten()'s isinstance check, which is why they do not
    # need an entry of their own.

    # HR-Bench's headline `average` is the mean over cycle_category of the
    # per-cycle means, not the row mean -- hence item_decomposes=False. Its rows
    # carry `gpt_score` rather than `score`, which bench_samples.dict_score was
    # taught to read.
    "hrbench_4k": dict(
        suite="natural", yaml="hrbench/hrbench4k.yaml",
        metric="average,none", scale=1.0, in_mean=False,
        on_demand=True, item_decomposes=False, needs_judge=True,
        max_pixels=NATIVE_MAX_PIXELS),
    "hrbench_8k": dict(
        suite="natural", yaml="hrbench/hrbench8k.yaml",
        metric="average,none", scale=1.0, in_mean=False,
        on_demand=True, item_decomposes=False, needs_judge=True,
        max_pixels=NATIVE_MAX_PIXELS),

    # spatialtreebench is here because the build brief names it, and it comes
    # with four caveats that have to travel with the number:
    #
    #  1. It is NOT the TreeBench of ERGO's table. lmms-eval's own README claims
    #     the name `TreeBench` maps here, and that claim is wrong for this
    #     purpose: ERGO reports HaochenWang/TreeBench (arXiv:2507.07999, <1k rows,
    #     csv, image-only), while this task is LongfeiLi/SpatialTree-Bench
    #     (arXiv:2512.20617, 6,300 rows, 8 GB of images+video). lmms-eval has no
    #     task for the former. A row produced here must not be put in a TreeBench
    #     column.
    #  2. Its aggregate is a WEIGHTED TREE over 27 leaves, and a leaf the sample
    #     missed scores 0 rather than being absent (`avg_leaf_scores.get(name, 0)`
    #     in its utils.py). An unstratified 100-row draw would leave ~7 leaves
    #     empty and read systematically low -- hence stratify_json, which balances
    #     the draw over the deepest `spatree*` key inside `extra_info`. The leaf
    #     is not a column, so stratify_by cannot reach it.
    #  3. 374 of its 6,300 rows are scored by an LLM judge (`gpteval`), so it
    #     needs an API key; without one those rows score 0 silently.
    #  4. 814 of its rows are VIDEO (up to 32 decoded frames each), and 5,486 are
    #     images. max_pixels does not govern video sampling, so it is deliberately
    #     unset here and the benchmark still gets a unit of its own.
    #
    # The data is not staged: `video: True` in its yaml makes lmms-eval
    # snapshot_download the repo and unzip images.zip + videos.zip (8.0 GB) into
    # $HF_HOME/SpatialTreeBench/ on first use, which a compute node with
    # HF_HUB_OFFLINE=1 cannot do. Stage it from a login node first.
    "spatialtreebench": dict(
        suite="natural", yaml="spatialtreebench/spatialtreebench.yaml",
        metric="SpaTreeBench,none", scale=1.0, in_mean=False,
        on_demand=True, item_decomposes=False, needs_judge=True, own_unit=True,
        stratify_json=("extra_info", "spatree")),

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

# lmms-eval's spelling of a benchmark, where it differs from this table's key.
#
# Two things made the keys differ, and both are worth keeping. First, the whole
# FULL task has been run on every arm already, under lmms-eval's own name
# (`hrbench4k`, 800 rows, in vlm_reasoning/results/lmms_eval/), and those files are
# the baseline table -- without this map flatten() would skip every one of them,
# because `hrbench4k` is not a key here. Second, the Figure-1 chain scans already
# use the names `hrbench4k_mini`, `hrbench8k_mini` and `vstar_mini` for something
# that is NOT a benchmark score (they are graded at --max-image-side 1024; see
# docs/deepeyes-gate.md §3). Keeping our mini tasks at `hrbench_4k_mini` and
# `vstar_bench_mini` means the two can never be confused in a filename, a WandB
# key or a conversation.
TASK_ALIASES = {"hrbench4k": "hrbench_4k", "hrbench8k": "hrbench_8k"}

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

    On-demand benchmarks are excluded from that reduction. They are not part of
    the default plan, so they are not part of what "this profile" means -- and
    including them would be destructive rather than merely untidy: vstar_bench's
    split is 191 rows, so at the standard `--natural-n 300` the natural suite
    would hold {191, 300}, every run's profile would rename from `n300_100` to
    `n191-300_100`, and 339 step files plus the `bench_n300_100/*` WandB
    namespace would be orphaned by a benchmark nobody asked that job to run.
    """
    if not sample_n:
        return dict(DEFAULT_SUITE_N)
    if set(sample_n) <= set(SUITES):
        return {s: sample_n.get(s, DEFAULT_SUITE_N[s]) for s in SUITES}
    out = {}
    for suite in SUITES:
        sizes = sorted({n for task, n in sample_n.items()
                        if (BENCHMARKS.get(base_task(task)) or {}).get("suite") == suite
                        and not is_on_demand(task)})
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

    A benchmark whose split is shorter than the profile asks for is clamped to
    its split (`split_n`). The sampler returns a short split whole rather than
    padding it, so this is a statement of what WILL be scored -- and it has to be,
    because bench_eval.py's profile check compares the two and refuses a collect
    where they differ. An explicit override still wins, including one that asks
    for more than the split holds, which then fails that check loudly.
    """
    suite_n = as_suite_n(suite_n or DEFAULT_SUITE_N)
    out = {}
    for task, spec in BENCHMARKS.items():
        value = suite_n[spec["suite"]]
        value = value if not isinstance(value, tuple) else value[0]
        split_n = spec.get("split_n")
        out[f"{task}{MINI_SUFFIX}"] = min(value, split_n) if split_n else value
    for task, n in (overrides or {}).items():
        # Normalised through base_task, so `--task-n hrbench4k=100` (lmms-eval's
        # spelling) and `--task-n hrbench_4k_mini=100` both land on the one key
        # the generated configs use. Accepting a spelling and then writing it
        # verbatim would add an entry no config reads.
        if base_task(task) not in BENCHMARKS:
            raise SystemExit(f"no such benchmark to override: {task}")
        out[f"{base_task(task)}{MINI_SUFFIX}"] = int(n)
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
#
# The four visual-search figures are a MEASUREMENT TIMES A MULTIPLIER, and the
# multiplier is the uncertain half. The measurement is `total_evaluation_time_seconds`
# from the full-split r1-mode runs on the cold start in
# vlm_reasoning/results/lmms_eval/coldstart_..._mnt4096_r1/, at 3.2 M pixels:
# V* 328 s / 191 docs = 2.9 min per 100, HR-4K 2013 s / 800 = 4.2, HR-8K 2118 s /
# 800 = 4.4. At NATIVE_MAX_PIXELS the pixels rise 1.0x for V* (its images are
# already under 3.4 MP median and are then not resized at all), 5.1x for HR-4K and
# 5.2x for HR-8K -- and the vision tower's attention is quadratic in patches, so
# the pessimistic end is used until a real job replaces these.
#
# spatialtreebench has never been run, here or anywhere in this repo. Its 25.0 is
# a placeholder sized off the non-natural rate plus video decoding, and it is
# flagged rather than disguised: the first job that runs it should replace it.
MINUTES_PER_100 = {
    "mme": 4.3,
    "mmerealworld": 9.1,
    "mmstar": 16.3,
    "pope": 3.5,
    "realworldqa": 4.9,
    "vstar_bench": 4.0,
    "hrbench_4k": 15.0,
    "hrbench_8k": 16.0,
    "spatialtreebench": 25.0,
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


# The tag a solo benchmark's banking unit carries under `--bank suite`. A banked
# unit is keyed by its tag on disk (`bench_eval/[<profile>/]partial/step-N/<tag>.json`),
# so mme-realworld keeps the `mmerw` it has always had rather than being renamed to
# its task name and silently re-running work an in-flight job already paid for.
# Everything else is named after itself.
SUITE_UNIT_TAG = {"mmerealworld": "mmerw"}


def is_on_demand(task):
    """Whether this benchmark is left out of the default plan. See the docstring."""
    return bool((BENCHMARKS.get(base_task(task)) or {}).get("on_demand"))


def needs_own_unit(task):
    """Whether this benchmark cannot share an lmms-eval invocation.

    True for anything carrying `max_pixels`, because the flag is per invocation
    and applies to every task in it, and for anything that sets `own_unit`
    outright. The whole reason mme-realworld has had a unit of its own since the
    beginning is this and nothing else.
    """
    spec = BENCHMARKS.get(base_task(task)) or {}
    return bool(spec.get("own_unit", spec.get("max_pixels")))


def extra_args(task):
    """The lmms-eval flags one benchmark needs, as a string for the shell plan."""
    max_pixels = (BENCHMARKS.get(base_task(task)) or {}).get("max_pixels")
    return f"--max-pixels {max_pixels}" if max_pixels else ""


def plan_units(bank="auto", sample_n=None, num_gpus=1, window=0, on_demand=False):
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

    `bank="auto"` (the default) picks suite when the suites fit the job's window
    and task when they do not. That is not a convenience: the two settings are
    coupled to the sample size, and getting the pair wrong is silent and fatal.
    A job asked for 300 documents with suite banking would budget 115 minutes for
    the natural suite, never satisfy its own guard in a one-hour allocation, bank
    nothing, and be resubmitted forever -- a checkpoint that never finishes and
    never says why. Deriving it from the measured cost removes the possibility.

    A benchmark needing its own resolution is pulled out of its suite's unit and
    given one of its own, under BOTH banking granularities. This is not an
    optimisation: `--max-pixels` is per invocation, so a combined `--tasks a,b`
    call can only be issued at one resolution, and putting V* in the same call as
    pope would score V* at pope's resolution without saying so. That is exactly
    the fault docs/deepeyes-gate.md §3 documents, one layer down.

    `on_demand=False` (the default) leaves the visual-search benchmarks out
    entirely, which is what every per-checkpoint job wants.

    Returns [(tag, [tasks], extra_args, minutes)].
    """
    wanted = {suite: suite_tasks(suite, on_demand=on_demand) for suite in SUITES}
    solo = [t for suite in SUITES for t in wanted[suite] if needs_own_unit(t)]

    by_suite = [(suite, [t for t in wanted[suite] if not needs_own_unit(t)], "")
                for suite in SUITES]
    by_suite = [u for u in by_suite if u[1]]
    by_suite += [(SUITE_UNIT_TAG.get(base_task(t), base_task(t)), [t], extra_args(t))
                 for t in solo]
    by_task = [(base_task(t), [t], extra_args(t))
               for suite in SUITES for t in wanted[suite]]

    if bank == "auto":
        groups = by_suite
        if window and any(estimate_minutes(tasks, sample_n, num_gpus) > window
                          for _tag, tasks, _extra in by_suite):
            groups = by_task
    elif bank == "suite":
        groups = by_suite
    elif bank == "task":
        groups = by_task
    else:
        raise SystemExit(f"unknown banking unit: {bank!r} (auto | suite | task)")

    # Belt and braces. Nothing above can build a unit whose tasks disagree about
    # resolution, but a future edit could, and the failure mode is a benchmark
    # scored at a neighbour's resolution with a results.json that looks perfectly
    # normal. Refuse instead.
    for tag, tasks, extra in groups:
        resolutions = {(BENCHMARKS.get(base_task(t)) or {}).get("max_pixels") for t in tasks}
        if len(resolutions) > 1:
            raise SystemExit(
                f"unit {tag!r} would score {', '.join(tasks)} in one lmms-eval "
                f"invocation at {sorted(r or 0 for r in resolutions)} pixels. "
                f"--max-pixels is per invocation; give each its own unit.")

    units = [(tag, tasks, extra, estimate_minutes(tasks, sample_n, num_gpus))
             for tag, tasks, extra in groups]
    return sorted(units, key=lambda u: -u[3])


def suite_tasks(suite, on_demand=False):
    """Mini task names for one suite, in table order.

    On-demand benchmarks are omitted unless asked for -- they are in the table so
    that bench_eval.py can reduce them, not so that every eval job runs them.
    """
    return [f"{t}{MINI_SUFFIX}" for t, spec in BENCHMARKS.items()
            if spec["suite"] == suite and (on_demand or not spec.get("on_demand"))]


def base_task(task):
    """The real benchmark a task name refers to, mini or full, ours or lmms-eval's."""
    name = task[: -len(MINI_SUFFIX)] if task.endswith(MINI_SUFFIX) else task
    return TASK_ALIASES.get(name, name)
