#!/usr/bin/env python
"""Generate the lmms-eval task configs for the two mini test suites.

The mini suites are the benchmarks we report on test, cut down to a fixed random
sample per benchmark so they can be run against every training checkpoint instead
of once at the end.

Each generated config is three lines: `include` the real task's yaml, rename the
task, and add a seeded subsample as `process_docs`. Everything else -- the prompt,
the visual handler, the metric list, the aggregation -- is inherited, so a mini
benchmark scores by exactly the same rules as the full one. None of the 16 tasks
defines `process_docs` of its own, so nothing is being overridden (the generator
checks this and refuses if that ever changes upstream).

The one exception is a benchmark carrying `local_data` in benchmarks.py, which
also gets its `dataset_path`/`dataset_kwargs` rewritten to read the same rows
from the repo snapshot already in the HF cache. That is not a change of data; it
is the only spelling that loads at all with HF_HUB_OFFLINE=1, which is how these
jobs run because the eval nodes have no internet.

lmms-eval resolves `include` against an absolute path and resolves `!function`
against the *including* file's directory, so this works from a --include_path dir
without touching the lmms-eval clone at all.

Usage:
    python eval_mini/make_mini_tasks.py --out-dir <dir> [--lmms-eval-dir DIR] [--n 100]
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmarks import BENCHMARKS, MINI_SUFFIX, SUITES, suite_tasks  # noqa: E402

SAMPLER = '''"""Seeded subsample, applied by every mini task config.

Fixed seed and fixed size: the point of the mini suites is to compare checkpoints
against each other, which only means anything if every checkpoint is scored on the
same documents. `datasets.shuffle` with an explicit seed is a deterministic
permutation, so the sample is reproducible across processes, machines and reruns.

Splits smaller than the sample size are returned whole rather than padded.
"""

import random

SAMPLE_N = {n}
SAMPLE_SEED = {seed}


def take(dataset):
    if len(dataset) <= SAMPLE_N:
        return dataset
    return dataset.shuffle(seed=SAMPLE_SEED).select(range(SAMPLE_N))


def take_groups(dataset, key, stratify=None):
    """Subsample whole groups of rows that share `dataset[key]`.

    Some benchmarks do not score a row on its own. MME scores an image from the
    yes/no PAIR that shares its question_id, and asserts it has both; a sample
    drawn row-wise splits nearly every pair and the task dies in aggregation,
    taking the whole suite invocation with it (aggregation runs after every task
    has generated, so one broken task discards all of them).

    Groups are drawn, not rows, so the sample is always whole. The count is still
    measured in rows -- SAMPLE_N=100 over pairs means 50 images -- so a grouped
    benchmark costs the same as an ungrouped one rather than 2x.

    With `stratify`, groups are taken round-robin over that column instead of in
    one shuffled sequence, so every value of it appears. This is not cosmetic for
    MME: its score SUMS per-category averages, so a category the sample missed
    contributes nothing at all, and a proportional draw from 14 unequal categories
    reliably misses the small ones -- existence, count and position, which are the
    three a grounding run most wants to watch.

    Selection order comes from `random.Random(SAMPLE_SEED)` over the group keys,
    not `datasets.shuffle`: the same groups must be chosen whatever order the rows
    arrive in. Row indices are restored to dataset order at the end so the result
    reads like a contiguous subset.
    """
    groups = {{}}
    for index, value in enumerate(dataset[key]):
        groups.setdefault(value, []).append(index)
    if len(dataset) <= SAMPLE_N or len(groups) <= 1:
        return dataset

    rng = random.Random(SAMPLE_SEED)
    if stratify is None:
        order = sorted(groups)
        rng.shuffle(order)
    else:
        # One group belongs to one stratum: the column is constant within a group
        # for the case this exists for (both rows of an MME pair share a category).
        # Taking the first row's value is therefore exact, not an approximation.
        column = dataset[stratify]
        strata = {{}}
        for value, indices in groups.items():
            strata.setdefault(column[indices[0]], []).append(value)
        for keys in strata.values():
            rng.shuffle(keys)
        order = []
        for row in zip(*(strata[name] for name in sorted(strata))):
            order.extend(row)
        # zip() stops at the smallest stratum; the rest are appended so a large
        # budget still fills up rather than capping at n_strata * smallest.
        taken = set(order)
        for name in sorted(strata):
            order.extend(k for k in strata[name] if k not in taken)

    chosen = []
    for value in order:
        if len(chosen) + len(groups[value]) > SAMPLE_N:
            continue
        chosen.extend(groups[value])
    # Every group is larger than the budget: keep the smallest whole one rather
    # than returning nothing, which would read as a benchmark that scored zero.
    if not chosen:
        chosen = min(groups.values(), key=len)
    return dataset.select(sorted(chosen))
'''

# Each grouped benchmark gets its own wrapper appended to minisample.py: lmms-eval
# resolves `!function` to a bare name and hands it only the dataset, so the column
# names cannot travel through the yaml. Keyed by task, not by column, so two
# benchmarks grouping on the same column name cannot collide.
GROUP_SAMPLER = '''

def take_for_{task}(dataset):
    return take_groups(dataset, "{key}", {stratify})
'''

CONFIG = """# Generated by eval_mini/make_mini_tasks.py -- do not edit by hand.
# {task}, cut to a fixed random sample. Everything except the task name and the
# subsample is inherited from the real benchmark's config.
include: {parent}
task: "{task}{suffix}"
process_docs: !function minisample.{sampler}
{extra}"""

# Written for a benchmark carrying `local_data` (see benchmarks.py). It replaces
# the inherited `dataset_path`/`dataset_kwargs` -- lmms-eval's load_yaml_config
# merges the included config first and then `.update()`s the including one over
# it, so these two keys win outright -- and points them at the file inside the
# repo snapshot that is already on disk. The rows are the same rows: this is the
# file the hub-resolved build was made from, so the sample and the score do not
# move. What changes is that no part of loading it consults the hub, which is the
# only way it can work on a node with no internet and HF_HUB_OFFLINE=1.
LOCAL_DATA_CONFIG = """
# {repo_id} is resolved to its snapshot on disk rather than left to `datasets` to
# resolve against the hub, which offline it cannot do (see benchmarks.py).
dataset_path: {builder}
dataset_kwargs:
  data_files:
    {split}: {path}
"""


def parent_split(task, parent):
    """The split name the real task evaluates, i.e. its `test_split`.

    It has to become the data_files KEY: the generic builders name each split
    after the key it was given, and lmms-eval then asks for `test_split`. Get it
    wrong and the task loads a dataset whose only split is called something else,
    which surfaces as a KeyError far from here.

    Read line-wise rather than with yaml.safe_load, because these configs carry
    `!function` tags that need lmms-eval's own constructor to parse -- the same
    reason the process_docs check below is a substring test.
    """
    for line in parent.read_text().splitlines():
        key, sep, value = line.partition(":")
        if sep and key.strip() == "test_split":
            return value.strip().strip("\"'")
    raise SystemExit(f"{task}: {parent} declares no test_split, so its rows cannot be "
                     f"pointed at a local file (see local_data in benchmarks.py)")


def local_data_config(task, spec, split):
    """The dataset_path/dataset_kwargs override for a benchmark with `local_data`.

    The snapshot is located with the same `snapshot_download` call the task's own
    utils.py already makes at import time -- visulogic reads its images out of
    images.zip that way -- so this adds no requirement the suite did not already
    have. Offline it is a cache lookup and no network call.
    """
    from huggingface_hub import snapshot_download

    local = spec["local_data"]
    try:
        root = Path(snapshot_download(repo_id=local["repo_id"], repo_type="dataset"))
    except Exception as exc:
        raise SystemExit(
            f"{task}: cannot locate the {local['repo_id']} snapshot ({exc}).\\n"
            f"The eval nodes have no internet, so it has to be in the HF cache "
            f"(HF_HOME={os.environ.get('HF_HOME', '<unset>')}) already. Fetch it "
            f"from a host that can reach the hub, then rerun."
        ) from exc

    path = root / local["file"]
    if not path.is_file():
        raise SystemExit(f"{task}: {path} is missing from the {local['repo_id']} snapshot")
    return LOCAL_DATA_CONFIG.format(repo_id=local["repo_id"], builder=local["builder"],
                                    split=split, path=path)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", help="where to write the generated configs")
    p.add_argument("--lmms-eval-dir", default=os.environ.get("LMMS_EVAL_DIR", ""),
                   help="lmms-eval clone (default $LMMS_EVAL_DIR)")
    p.add_argument("--n", type=int, default=100, help="samples per benchmark (default 100)")
    p.add_argument("--seed", type=int, default=1234, help="subsample seed (default 1234)")
    p.add_argument("--print-tasks", choices=sorted(SUITES),
                   help="print the comma-separated mini task names for one suite and exit")
    args = p.parse_args()

    if args.print_tasks:
        print(",".join(suite_tasks(args.print_tasks)))
        return

    if not args.out_dir:
        raise SystemExit("--out-dir is required unless --print-tasks is given")
    if not args.lmms_eval_dir:
        raise SystemExit("pass --lmms-eval-dir or set LMMS_EVAL_DIR")
    tasks_root = Path(args.lmms_eval_dir) / "lmms_eval" / "tasks"
    if not tasks_root.is_dir():
        raise SystemExit(f"not an lmms-eval clone: {args.lmms_eval_dir} (no lmms_eval/tasks/)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sampler = SAMPLER.format(n=args.n, seed=args.seed)
    for task, spec in BENCHMARKS.items():
        if spec.get("group_by"):
            sampler += GROUP_SAMPLER.format(task=task, key=spec["group_by"],
                                            stratify=repr(spec.get("stratify_by")))
    (out_dir / "minisample.py").write_text(sampler)

    written = 0
    for task, spec in BENCHMARKS.items():
        parent = tasks_root / spec["yaml"]
        if not parent.is_file():
            raise SystemExit(f"{task}: no such task yaml: {parent}")
        # Adding process_docs on top of a task that already has one would silently
        # replace the task's own document preparation, not compose with it.
        if "process_docs" in parent.read_text():
            raise SystemExit(
                f"{task}: {parent} now defines process_docs; the mini config would "
                f"override it and change what the benchmark measures. Compose the two "
                f"explicitly before regenerating."
            )
        extra = ""
        if spec.get("local_data"):
            extra = local_data_config(task, spec, parent_split(task, parent))
        (out_dir / f"{task}{MINI_SUFFIX}.yaml").write_text(
            CONFIG.format(task=task, parent=parent, suffix=MINI_SUFFIX, extra=extra,
                          sampler=f"take_for_{task}" if spec.get("group_by") else "take")
        )
        written += 1

    print(f"wrote {written} mini task configs + minisample.py to {out_dir}", file=sys.stderr)
    print(f"  {args.n} samples per benchmark, seed {args.seed}", file=sys.stderr)


if __name__ == "__main__":
    main()
