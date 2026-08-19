#!/usr/bin/env python
"""Per-item benchmark results, bound to (run, step), and the paired tests they buy.

`bench_eval.py` reduces an lmms-eval run to one number per benchmark. That number
is all the WandB curve needs, and it is also why two checkpoints 0.03 apart cannot
be told from noise: an unpaired difference of two 100-item means carries
se ~= 0.028. But every checkpoint is scored on the SAME fixed-seed sample, so the
comparison is naturally paired -- and lmms-eval has been writing the per-item rows
all along, next to the results.json the aggregate came from. Pairing them takes
se on a difference to ~0.017 at the current sample size, for no GPU time at all.

This module harvests those rows, stores a reduced copy beside `step-<N>.json`, and
runs the paired tests.

    python bench_samples.py --harvest --run-dir DIR --step N --results A.json ...
    python bench_samples.py --retro [--dry-run]      # everything already on disk
    python bench_samples.py --index                  # what is harvested, what is not
    python bench_samples.py --compare RUN@STEP RUN@STEP [--suite natural]

Three things about the data are worth knowing before trusting any of this.

**`doc_hash` is a constant.** Every row of every task carries
`doc_hash = 74234e98...`, so it identifies nothing. The item key here is `item`, a
digest of the row's own content (question_id, prompt, media, target), which was
checked to be identical across checkpoints for all five natural benchmarks -- that
identity is what makes the pairing legitimate, and it is re-checked at compare time
rather than assumed.

**`doc_id` is a position, but a stable one.** The sampler is
`dataset.shuffle(seed=1234).select(range(n))`, a fixed permutation truncated to n,
so doc_id i is the same document in every run at the same n, and the n=100 sample
is a prefix of the n=300 one. It is the primary join key because it is exact and
cheap; `item` is carried alongside and every join asserts the two agree.

Grouped benchmarks are the exception: MME draws whole yes/no pairs and then
restores dataset order, so its doc_id renumbers when n changes. Across sample sizes
MME must join on `item`, which is why `item` is stored at all.

**Some tasks have no usable content key.** illusionvqa asks the same question of
1000 different images and records neither the image nor an id, so its `item`
collapses to 4 distinct values in 100 rows; scienceqa and pope each have one
genuinely duplicated item. All of these are joinable by doc_id at a fixed n, which
is the only comparison they are ever used for -- both live in the non-natural
suite, which stays at n=100. `--index` prints the degenerate ones so this stays a
known limitation rather than a silent one.
"""

import argparse
import glob
import gzip
import hashlib
import json
import os
import random
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_mini"))
from benchmarks import (BENCHMARKS, SUITES, base_task, profile_dir,  # noqa: E402
                        profile_name, profile_of_payload)

# Row keys that are never a metric. Everything else whose value is a float, or a
# dict carrying "score", is treated as one -- which is how MME is handled without
# naming it: its rows carry mme_cognition_score or mme_perception_score depending
# on the document, and neither is the headline key the aggregate is read from.
NON_METRIC_KEYS = {"doc_id", "target", "resps", "filtered_resps", "token_counts",
                   "doc_hash", "input", "input_media", "submission"}

# How much of the model's response to keep. Enough to see WHAT changed on an item
# that flipped; not enough to make the store unmanageable. 13 tasks x 300 items x
# 30 checkpoints x several runs of untruncated reasoning traces would not survive
# the /lustre quota, and the quota breaks running jobs, not just this.
RESP_CHARS = 400

SAMPLES_SUBDIR = "samples"


# ---------------------------------------------------------------- reading rows

def fingerprint(row):
    """A content key for one evaluated document, stable across checkpoints.

    Built from the fields that describe the QUESTION rather than the answer:
    whatever question_id the task's own metric dict carries, the rendered prompt,
    the media list, and the target. Not from `resps`, obviously, and not from
    doc_id, which is a position rather than an identity.

    Returns a short digest. Collisions within a task are real duplicates, not hash
    accidents, and `--index` reports them per task.
    """
    qid = None
    for key, value in row.items():
        if key not in NON_METRIC_KEYS and isinstance(value, dict) and "question_id" in value:
            qid = value["question_id"]
            break
    payload = json.dumps([qid, row.get("input"), row.get("input_media"), row.get("target")],
                         sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def metric_name(spec):
    """The results.json key a benchmark is headlined by, without its filter suffix.

    lmms-eval names an aggregate `<metric>,<filter>` (e.g. `pope_f1_score,none`)
    but names the per-row key `<metric>`, so the table's `metric` field has to be
    split before it can be looked up in a sample row.
    """
    return spec["metric"].split(",")[0]


def dict_score(value):
    """The 0-1 verdict inside one metric dict, whatever the task chose to call it.

    Four shapes occur, and only the first says "score":

        {"score": 0.0, ...}                       pope, mmstar, mme, p3's sample_*
        {"pred_answer": "A", "answer": "A", ...}  mme-realworld
        {"parsed_pred": "E", "answer": "B", ...}  mmmu_pro
        {"scores": [false], "response": [...]}    mathvision

    The last three are tasks that compare the prediction to the answer during
    AGGREGATION rather than per row, so the row carries the two strings and no
    verdict. Reproducing the comparison here is exact for the multiple-choice
    tasks these are (equality of the parsed choice), which is what they all are.
    Anything else returns None and the item is stored unscored rather than
    counted as wrong -- a silent zero would read as a model failure.
    """
    if isinstance(value.get("score"), (int, float)) and not isinstance(value.get("score"), bool):
        return float(value["score"])
    for predicted, actual in (("pred_answer", "answer"), ("parsed_pred", "answer")):
        if predicted in value and actual in value:
            got, want = value[predicted], value[actual]
            if isinstance(want, (list, tuple)):
                return 1.0 if got in want else 0.0
            return 1.0 if str(got).strip() == str(want).strip() else 0.0
    scores = value.get("scores")
    if isinstance(scores, list) and scores and all(isinstance(s, bool) for s in scores):
        return float(sum(scores)) / len(scores)
    return None


def row_metrics(row):
    """{name: (score, category)} for every metric this row carries a verdict for."""
    out = {}
    for key, value in row.items():
        if key in NON_METRIC_KEYS:
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[key] = (float(value), None)
        elif isinstance(value, dict):
            score = dict_score(value)
            if score is not None:
                out[key] = (score, value.get("category") or value.get("l2_category"))
    return out


def primary_score(row, spec):
    """(metric, score, category) for this benchmark's verdict on one document.

    `item_metric` first where the table declares one (p3's headline is a corpus
    F1 whose per-row value is a confusion table, so `sample_f1` is the per-item
    judgement); then the headline metric; then, if the row carries exactly one
    verdict, that one -- which is how MME is handled, its rows carrying
    mme_perception_score or mme_cognition_score depending on the document while
    the aggregate is read from the former.

    A row with several verdicts and no declared preference is left unscored
    rather than guessed at: picking one arbitrarily would put a number in the
    store that no aggregate agrees with.
    """
    metrics = row_metrics(row)
    if not metrics:
        return None, None, None
    for name in ((spec or {}).get("item_metric"), metric_name(spec) if spec else None):
        if name in metrics:
            return (name,) + metrics[name]
    if len(metrics) == 1:
        name, (score, category) = next(iter(metrics.items()))
        return name, score, category
    return None, None, None


def truncate(value):
    if isinstance(value, list):
        value = value[0] if value else ""
    if not isinstance(value, str):
        value = json.dumps(value, default=str)
    return value[:RESP_CHARS]


def reduce_rows(path, task):
    """One results row per evaluated document, as stored."""
    spec = BENCHMARKS.get(base_task(task))
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            metric, score, category = primary_score(row, spec)
            record = {
                "task": task,
                "doc_id": row.get("doc_id"),
                "item": fingerprint(row),
                "metric": metric,
                "score": score,
                "pred": truncate(row.get("filtered_resps")),
                "target": truncate(row.get("target")),
                "resp": truncate(row.get("resps") or row.get("filtered_resps")),
            }
            if category:
                record["cat"] = category
            rows.append(record)
    return rows


def sample_files(results_path):
    """The per-item jsonl files written by the same lmms-eval invocation.

    An output directory accumulates one set of files per invocation, all named
    `<timestamp>_...`, and the timestamp is what ties a results.json to its own
    rows. Matching on the directory alone would pick up a re-run's rows and
    silently attribute another checkpoint's answers to this one.
    """
    directory = os.path.dirname(results_path)
    stamp = os.path.basename(results_path).split("_results.json")[0]
    out = {}
    for path in sorted(glob.glob(os.path.join(directory, f"{stamp}_samples_*.jsonl"))):
        out[os.path.basename(path).split("_samples_", 1)[1][: -len(".jsonl")]] = path
    return out


# ------------------------------------------------------------------- the store

def samples_file(run_dir, step, sample_n):
    return profile_dir(Path(run_dir) / "bench_eval", sample_n) / SAMPLES_SUBDIR / f"step-{step}.jsonl.gz"


def write_harvest(dest, header, rows):
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Write-then-rename, as for the step file: a reader that finds a half-written
    # gzip gets an unrecoverable CRC error, not a retry.
    tmp = dest.with_suffix(".tmp")
    with gzip.open(tmp, "wt") as fh:
        fh.write(json.dumps(header, sort_keys=True) + "\n")
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    tmp.replace(dest)


def read_harvest(path):
    with gzip.open(path, "rt") as fh:
        header = json.loads(fh.readline())
        rows = [json.loads(line) for line in fh if line.strip()]
    return header, rows


def harvest(run_dir, step, results_paths, sample_n=None, quiet=False):
    """Reduce every sample file belonging to these results into one store file.

    `sample_n` is read out of the results.json (`n-samples`) rather than taken on
    trust from the caller: it is the only record of what was actually evaluated,
    and it is what stops an n=100 harvest and an n=300 one being pooled by
    mistake.
    """
    rows, sources, sizes, missing = [], [], {}, []
    for results_path in results_paths:
        try:
            payload = json.load(open(results_path))
        except Exception as exc:
            missing.append(f"{results_path}: unreadable ({type(exc).__name__})")
            continue
        found = sample_files(results_path)
        for task in (payload.get("results") or {}):
            counts = (payload.get("n-samples") or {}).get(task) or {}
            if counts.get("original") is not None:
                sizes[task] = int(counts["original"])
            path = found.get(task)
            if path is None:
                missing.append(f"{results_path}: no sample file for {task}")
                continue
            rows.extend(reduce_rows(path, task))
            sources.append(path)

    if not rows:
        return None, missing

    header = {
        "run": os.path.basename(os.path.abspath(run_dir)),
        "run_dir": os.path.abspath(run_dir),
        "step": int(step),
        "sample_n": sizes,
        "suite_n": sample_n or {},
        "n_rows": len(rows),
        "resp_chars": RESP_CHARS,
        "sources": sources,
        "harvested": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    dest = samples_file(run_dir, step, sample_n or sizes)
    write_harvest(dest, header, rows)
    if not quiet:
        print(f"harvested {len(rows)} items over {len(sources)} task(s) -> {dest}")
        for note in missing:
            print(f"  NOT harvested: {note}")
    return dest, missing


# ------------------------------------------------- finding results after the fact

def repo_root():
    return Path(os.path.dirname(os.path.abspath(__file__)))


def step_files(repo):
    """Every recorded benchmark result, as (run_dir, step, path).

    Three shapes, all of them `{"metrics": ..., "step": N}`:
    the in-training curve under checkpoint/<run>/, the one-off evals
    run_bench_eval_steps.sh leaves in outputs/bench_one/<run>-cp<step>/, and the
    baselines under outputs/bench_baselines/<label>/.

    A bench_one result is a COPY of a curve point -- that script writes it into
    the real run directory as its last act -- so it is reported but never
    harvested on its own: the canonical location is the run directory, and
    harvesting both would store the same rows twice under two names.
    """
    out = []
    patterns = [(repo / "checkpoint", "curve"),
                (repo / "outputs" / "bench_baselines", "baseline"),
                (repo / "outputs" / "bench_one", "bench_one")]
    for root, kind in patterns:
        for path in sorted(glob.glob(str(root / "*" / "bench_eval" / "step-*.json"))):
            match = re.search(r"step-(\d+)\.json$", path)
            if not match:
                continue
            out.append((Path(path).parent.parent, int(match.group(1)), Path(path), kind))
    return out


def results_index(quiet=True):
    """Map every mini-suite results.json on disk back to the model it scored.

    lmms-eval records the resolved command line inside results.json, so
    `config.resolved_cli_args.model_args` still names the exact `pretrained=`
    directory the run was given. For a checkpoint that is
    `checkpoint/_bench_eval/<run>_cp<step>_merged`, which names the run and the
    step outright; for a baseline it is the standalone model, matched later
    against the base_model.txt that recorded it.

    So the binding that looked irrecoverable is not: it survives in the results
    file itself, and does not depend on the banked markers (deleted once a step
    file is written) or on output-directory mtimes (rewritten by every re-run).

    Returns ({(run, step): [paths]}, {model_path: [paths]}).
    """
    root = Path(os.environ.get("LMMS_EVAL_RESULTS",
                               "/home/uberger/scratch/research/vlm_reasoning/results/lmms_eval"))
    by_step, by_model = defaultdict(list), defaultdict(list)
    for path in glob.glob(str(root / "*" / "*" / "*_results.json")):
        try:
            payload = json.load(open(path))
        except Exception:
            continue
        tasks = list((payload.get("results") or {}).keys())
        # Mini tasks only. The same tree holds the full test suite, whose results
        # are for the same models but a different (much larger) sample.
        if not tasks or not all(t.endswith("_mini") for t in tasks):
            continue
        args = ((payload.get("config") or {}).get("resolved_cli_args") or {}).get("model_args") or ""
        match = re.search(r"pretrained=([^,]+)", args)
        if not match:
            continue
        model = match.group(1).rstrip("/")
        merged = re.fullmatch(r"(.+)_cp(\d+)_merged", os.path.basename(model))
        if merged and f"{os.sep}checkpoint{os.sep}_bench_eval{os.sep}" in model + os.sep:
            run, step = merged.group(1), int(merged.group(2))
            # A bench_one shadow directory is named <run>-cp<step>, so its merged
            # model comes out as <run>-cp<step>_cp<step>_merged. Unwrap it, or
            # every one-off eval files itself under a run that does not exist.
            shadow = re.fullmatch(r"(.+)-cp(\d+)", run)
            if shadow and int(shadow.group(2)) == step:
                run = shadow.group(1)
            by_step[(run, step)].append(path)
        else:
            by_model[model].append(path)
    if not quiet:
        print(f"indexed {sum(len(v) for v in by_step.values())} checkpoint results "
              f"and {sum(len(v) for v in by_model.values())} standalone-model results")
    return by_step, by_model


def headline_metrics(payload):
    """{benchmark: normalised score} from one results.json, as bench_eval reads it."""
    out = {}
    for task, res in (payload.get("results") or {}).items():
        spec = BENCHMARKS.get(base_task(task))
        if spec is None:
            continue
        value = res.get(spec["metric"])
        if isinstance(value, (int, float)):
            out[base_task(task)] = float(value) / spec["scale"]
    return out


def resolve_results(run_dir, step, kind, by_step, by_model):
    """The results.json files that produced this step file, proved by their values.

    Candidates come from the path (the merged model names the run and step) or,
    for a step 0, from the base_model.txt that recorded which standalone model was
    scored. Every candidate is then CHECKED against the step file: a results.json
    is accepted only if a benchmark's score in it equals the one recorded. That
    turns "probably this one" into "this file contains that number", which matters
    because an output directory accumulates one set of files per re-run and mtimes
    do not say which set was collected.
    """
    step_path = Path(run_dir) / "bench_eval" / f"step-{step}.json"
    try:
        want = json.loads(step_path.read_text())["metrics"]
    except Exception:
        return [], ["step file unreadable"]
    want = {k.rsplit("/", 1)[1]: v for k, v in want.items()
            if k.count("/") == 2 and k.rsplit("/", 1)[1] in BENCHMARKS}

    run = os.path.basename(os.path.abspath(run_dir))
    if kind == "bench_one":
        run = re.sub(rf"-cp{step}$", "", run)
    candidates = list(by_step.get((run, step), []))
    if step == 0:
        base_file = Path(run_dir) / "bench_eval" / "base_model.txt"
        if base_file.exists():
            candidates += by_model.get(base_file.read_text().strip().rstrip("/"), [])

    accepted, covered = [], set()
    for path in candidates:
        try:
            got = headline_metrics(json.load(open(path)))
        except Exception:
            continue
        hit = {b for b, v in got.items() if b in want and abs(v - want[b]) < 1e-9}
        if hit and not hit <= covered:
            accepted.append(path)
            covered |= hit
    notes = []
    if not candidates:
        notes.append("no results.json names this model")
    elif not accepted:
        notes.append(f"{len(candidates)} candidate results.json, none matching the recorded scores")
    for benchmark in sorted(set(want) - covered):
        notes.append(f"{benchmark}: no results.json carries the recorded score")
    return accepted, notes


# ---------------------------------------------------------------- the commands

def do_harvest(args):
    harvest(args.run_dir, args.step, args.results,
            sample_n=json.loads(args.sample_n) if args.sample_n else None)


def do_retro(args):
    """Harvest every step file already on disk whose rows can still be located."""
    repo = Path(args.repo) if args.repo else repo_root()
    by_step, by_model = results_index(quiet=False)

    done = failed = skipped = copies = 0
    unrecoverable = []
    for run_dir, step, path, kind in step_files(repo):
        if kind == "bench_one":
            copies += 1
            continue
        payload = json.loads(path.read_text())
        sample_n = profile_of_payload(payload)
        dest = samples_file(run_dir, step, sample_n)
        if dest.exists() and not args.force:
            skipped += 1
            continue
        results, notes = resolve_results(run_dir, step, kind, by_step, by_model)
        label = f"{os.path.basename(run_dir)[:56]:56s} step {step:<5d}"
        if not results:
            unrecoverable.append((label, notes))
            failed += 1
            continue
        if args.dry_run:
            print(f"  would harvest {label} from {len(results)} results file(s)")
            done += 1
            continue
        written, missing = harvest(run_dir, step, results, sample_n=sample_n, quiet=True)
        if written is None:
            unrecoverable.append((label, missing or ["no sample rows found"]))
            failed += 1
            continue
        done += 1
        if args.verbose:
            print(f"  {label}  -> {written}")

    print("")
    print(f"harvested       {done}")
    print(f"already present {skipped}" + ("" if not skipped else "   (pass --force to redo)"))
    print(f"bench_one       {copies}   (copies of curve points; harvested under the run itself)")
    print(f"NOT harvested   {failed}")
    for label, notes in unrecoverable:
        print(f"  {label}  {'; '.join(notes[:3])}")


def do_index(args):
    """What is harvested, at what sample size, and which tasks cannot be joined."""
    repo = Path(args.repo) if args.repo else repo_root()
    by_run = defaultdict(list)
    stored = []
    for run_dir, step, path, kind in step_files(repo):
        if kind == "bench_one":
            continue
        payload = json.loads(path.read_text())
        sample_n = profile_of_payload(payload)
        dest = samples_file(run_dir, step, sample_n)
        by_run[(os.path.basename(run_dir), kind)].append((step, dest.exists(), sample_n))
        if dest.exists():
            stored.append(dest)

    print(f"{'run':58s} {'kind':9s} {'steps':>6s} {'harvested':>10s}  profile")
    for (run, kind), entries in sorted(by_run.items()):
        harvested = sum(1 for _, ok, _ in entries if ok)
        shown = ", ".join(sorted({profile_name(n) for _, _, n in entries}))
        print(f"{run[:58]:58s} {kind:9s} {len(entries):6d} {harvested:10d}  {shown}")

    # Item-key health, measured on what is actually stored rather than assumed.
    print("")
    if not stored:
        print("nothing harvested yet -- run --retro")
        return
    print(f"item-key uniqueness, from {stored[0]}:")
    _, rows = read_harvest(stored[0])
    per_task = defaultdict(list)
    for row in rows:
        per_task[row["task"]].append(row["item"])
    for task, items in sorted(per_task.items()):
        spec = BENCHMARKS.get(base_task(task)) or {}
        flag = "" if len(set(items)) == len(items) else "   <-- join across sample sizes NOT possible"
        print(f"  {task:34s} {spec.get('suite','?'):11s} {len(set(items)):4d} distinct / {len(items):4d} rows{flag}")


# --------------------------------------------------------------- paired testing

def find_harvest(repo, spec, profile):
    """Resolve `name@step` to a stored harvest, matching names loosely.

    Run names here are 90 characters of hyperparameters, so an exact match is not
    a usable interface. A substring is accepted as long as it picks out exactly
    one run; anything ambiguous is refused with the candidates listed, because
    quietly taking the first would compare the wrong pair of runs.
    """
    name, _, step = spec.partition("@")
    roots = [(repo / "checkpoint", "curve"), (repo / "outputs" / "bench_baselines", "baseline")]
    matches = []
    for root, kind in roots:
        for directory in sorted(glob.glob(str(root / "*"))):
            if not os.path.isdir(os.path.join(directory, "bench_eval")):
                continue
            if name.lower() in os.path.basename(directory).lower():
                matches.append((Path(directory), kind))
    if not matches:
        raise SystemExit(f"no run or baseline matching '{name}'")
    if len(matches) > 1:
        listing = "\n  ".join(os.path.basename(m) for m, _ in matches)
        raise SystemExit(f"'{name}' matches {len(matches)} runs:\n  {listing}")
    run_dir = matches[0][0]

    stored = sorted(glob.glob(str(profile_dir(run_dir / "bench_eval", profile) /
                                  SAMPLES_SUBDIR / "step-*.jsonl.gz")),
                    key=lambda p: int(re.search(r"step-(\d+)", p).group(1)))
    if not stored:
        raise SystemExit(f"{os.path.basename(run_dir)} has nothing harvested at "
                         f"natural={profile['natural']}/nonnatural={profile['nonnatural']} "
                         f"-- run --retro, or check --profile")
    if step:
        want = str(profile_dir(run_dir / "bench_eval", profile) / SAMPLES_SUBDIR / f"step-{int(step)}.jsonl.gz")
        if want not in stored:
            steps = " ".join(re.search(r"step-(\d+)", p).group(1) for p in stored)
            raise SystemExit(f"{os.path.basename(run_dir)} has no harvest at step {step}; has: {steps}")
        return Path(want)
    return Path(stored[-1])


def joined(rows_a, rows_b, tasks):
    """Item-aligned score pairs, over the documents both sides actually scored.

    Joined on (task, doc_id) -- exact, because both sides drew the same fixed-seed
    prefix -- and then CHECKED on `item`: if the two sides disagree about what
    document that position holds, the pairing is wrong and the whole comparison is
    refused rather than reported with a caveat nobody will read.
    """
    def index(rows):
        out = {}
        for row in rows:
            if row["task"] in tasks and row["score"] is not None:
                out[(row["task"], row["doc_id"])] = row
        return out

    left, right = index(rows_a), index(rows_b)
    shared = sorted(set(left) & set(right))
    mismatched = [k for k in shared if left[k]["item"] != right[k]["item"]]
    if mismatched:
        raise SystemExit(
            f"{len(mismatched)} of {len(shared)} joined positions hold different documents on the two "
            f"sides (e.g. {mismatched[0]}). These two harvests were not drawn from the same sample, "
            f"so nothing here is paired. Compare points scored at the same sample size.")
    return [(k[0], left[k]["score"], right[k]["score"]) for k in shared], left, right


def paired_bootstrap(diffs, iterations=10000, seed=0):
    """CI on the mean paired difference, resampling ITEMS."""
    if not diffs:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(diffs)
    mean = sum(diffs) / n
    draws = []
    for _ in range(iterations):
        draws.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    draws.sort()
    return mean, draws[int(0.025 * iterations)], draws[int(0.975 * iterations)]


def mcnemar(pairs):
    """Exact two-sided McNemar over the items that are scored 0/1 on both sides."""
    b = sum(1 for _, x, y in pairs if x == 1.0 and y == 0.0)
    c = sum(1 for _, x, y in pairs if x == 0.0 and y == 1.0)
    n = b + c
    if n == 0:
        return b, c, 1.0
    # Exact binomial tail at p=0.5; n is at most a few hundred, so the sum is cheap
    # and needs no normal approximation to defend.
    from math import comb
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return b, c, min(1.0, 2.0 * tail)


def do_compare(args):
    repo = Path(args.repo) if args.repo else repo_root()
    profile = {"natural": args.natural_n, "nonnatural": args.nonnatural_n}
    path_a = find_harvest(repo, args.compare[0], profile)
    path_b = find_harvest(repo, args.compare[1], profile)
    header_a, rows_a = read_harvest(path_a)
    header_b, rows_b = read_harvest(path_b)

    if header_a["sample_n"] != header_b["sample_n"]:
        only_a = {t: n for t, n in header_a["sample_n"].items() if header_b["sample_n"].get(t) != n}
        raise SystemExit(
            "these two harvests were scored at different sample sizes and cannot share a curve:\n"
            f"  {header_a['run'][:60]} step {header_a['step']}: {header_a['sample_n']}\n"
            f"  {header_b['run'][:60]} step {header_b['step']}: {header_b['sample_n']}\n"
            f"  differing: {only_a}")

    suites = [args.suite] if args.suite else list(SUITES)
    print("=" * 78)
    print(f"A  {header_a['run'][:66]}  step {header_a['step']}")
    print(f"B  {header_b['run'][:66]}  step {header_b['step']}")
    n_natural = header_a["sample_n"].get("pope_mini", "?")
    print(f"   paired over items, sample sizes {sorted(set(header_a['sample_n'].values()))} "
          f"(natural benchmarks at n={n_natural})")
    print("=" * 78)

    for suite in suites:
        tasks = {f"{t}{'_mini'}" for t, spec in BENCHMARKS.items()
                 if spec["suite"] == suite and spec["in_mean"]}
        pairs, left, right = joined(rows_a, rows_b, tasks)
        if not pairs:
            print(f"\n{suite}: nothing joined")
            continue

        print(f"\n{suite}   {len(pairs)} items over {len({t for t, _, _ in pairs})} benchmarks")
        indirect = sorted(t for t in {t for t, _, _ in pairs}
                          if not (BENCHMARKS.get(base_task(t)) or {}).get("item_decomposes", True))
        if indirect:
            print(f"  NOTE: {', '.join(indirect)} report a corpus-level score (F1, or a sum of "
                  f"per-category averages) that is NOT the mean of the per-item verdicts below. "
                  f"For these the test asks whether different documents were answered correctly, "
                  f"which is not the same question as whether bench/{suite}/mean moved.")
        print(f"  {'benchmark':30s} {'A':>8s} {'B':>8s} {'B-A':>8s} {'n':>5s}")
        by_task = defaultdict(list)
        for task, x, y in pairs:
            by_task[task].append((x, y))
        task_diffs = []
        for task in sorted(by_task):
            values = by_task[task]
            mean_a = sum(x for x, _ in values) / len(values)
            mean_b = sum(y for _, y in values) / len(values)
            task_diffs.append(mean_b - mean_a)
            print(f"  {task:30s} {mean_a:8.4f} {mean_b:8.4f} {mean_b - mean_a:+8.4f} {len(values):5d}")

        # Pooled over items: every document weighted equally, which is the endpoint
        # whose standard error can be stated honestly. The mean-of-means is printed
        # beside it because that is what bench/<suite>/mean has always been, and a
        # comparison against the existing curve has to be readable.
        diffs = [y - x for _, x, y in pairs]
        mean, lo, hi = paired_bootstrap(diffs, iterations=args.iterations)
        binary = [(t, x, y) for t, x, y in pairs if x in (0.0, 1.0) and y in (0.0, 1.0)]
        b, c, p = mcnemar(binary)
        mom = sum(task_diffs) / len(task_diffs)
        print(f"  {'-' * 62}")
        print(f"  pooled over items   B-A = {mean:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]"
              f"   {'significant' if lo * hi > 0 else 'not distinguishable from 0'}")
        print(f"  mean of benchmarks  B-A = {mom:+.4f}   (what bench/{suite}/mean moves by)")
        print(f"  McNemar over {len(binary)} binary items: A-only {b}, B-only {c}, "
              f"discordant {(b + c) / max(1, len(binary)):.1%}, exact p = {p:.4f}")

    if args.flips:
        print("\n" + "=" * 78)
        print(f"items that flipped (up to {args.flips} per direction)")
        print("=" * 78)
        tasks = {f"{t}_mini" for t, spec in BENCHMARKS.items()
                 if (not args.suite or spec["suite"] == args.suite) and spec["in_mean"]}
        pairs, left, right = joined(rows_a, rows_b, tasks)
        for direction, want in (("B fixed what A got wrong", (0.0, 1.0)),
                                ("B broke what A got right", (1.0, 0.0))):
            print(f"\n{direction}:")
            shown = 0
            for task, x, y in pairs:
                if (x, y) != want or shown >= args.flips:
                    continue
                key = (task, [k for k in left if k[0] == task][0][1])
                shown += 1
            # Re-walk with keys so the record can be printed, not just the scores.
            shown = 0
            for key in sorted(set(left) & set(right)):
                if key[0] not in tasks:
                    continue
                a_row, b_row = left[key], right[key]
                if (a_row["score"], b_row["score"]) != want or shown >= args.flips:
                    continue
                shown += 1
                print(f"  [{key[0]} #{key[1]}] target={a_row['target'][:40]!r} "
                      f"A={a_row['pred'][:30]!r} B={b_row['pred'][:30]!r}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--harvest", action="store_true", help="store one eval's per-item rows")
    p.add_argument("--retro", action="store_true", help="harvest every step file already on disk")
    p.add_argument("--index", action="store_true", help="report what is stored and what can be joined")
    p.add_argument("--compare", nargs=2, metavar="RUN@STEP",
                   help="paired comparison of two harvested checkpoints")
    p.add_argument("--run-dir", help="the training run's output_dir (--harvest)")
    p.add_argument("--step", type=int, help="checkpoint step (--harvest)")
    p.add_argument("--results", nargs="*", default=[], help="lmms-eval results.json files (--harvest)")
    p.add_argument("--sample-n", help="JSON {suite: n} this eval was run at (--harvest)")
    p.add_argument("--repo", help="repo root (default: this script's directory)")
    p.add_argument("--natural-n", type=int, default=100, help="which stored profile to read (default 100)")
    p.add_argument("--nonnatural-n", type=int, default=100, help="which stored profile to read (default 100)")
    p.add_argument("--suite", choices=sorted(SUITES), help="restrict a comparison to one suite")
    p.add_argument("--iterations", type=int, default=10000, help="bootstrap resamples (default 10000)")
    p.add_argument("--flips", type=int, default=0, help="print up to N flipped items per direction")
    p.add_argument("--force", action="store_true", help="re-harvest points already stored (--retro)")
    p.add_argument("--dry-run", action="store_true", help="report the work and write nothing (--retro)")
    p.add_argument("--verbose", action="store_true", help="one line per harvested point (--retro)")
    args = p.parse_args()

    if args.harvest:
        if args.run_dir is None or args.step is None or not args.results:
            raise SystemExit("--harvest needs --run-dir, --step and --results")
        do_harvest(args)
    elif args.retro:
        do_retro(args)
    elif args.index:
        do_index(args)
    elif args.compare:
        do_compare(args)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
