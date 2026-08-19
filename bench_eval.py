#!/usr/bin/env python
"""Turn one checkpoint's lmms-eval output into WandB-ready scalars, and ship them.

The periodic benchmark eval runs the two mini suites on a checkpoint and drops the
raw lmms-eval results.json files. This module reduces those to a flat dict of
scalars keyed for WandB and writes `<run-dir>/bench_eval/step-<N>.json`:

    {"step": 1000, "metrics": {"bench/natural/mmstar_mini/average": 0.61, ...}}

Keeping the file flat is deliberate: the trainer-side callback that logs these live
then needs no knowledge of benchmarks, metric names or scales -- it reads the dict
and logs it. All of that knowledge lives here, in the repo, next to the table it
depends on.

    python bench_eval.py --collect --run-dir DIR --step N --results A.json B.json
    python bench_eval.py --backfill --run-dir DIR --wandb-run-id ID
    python bench_eval.py --publish-baseline --run-dir DIR --name LABEL --span N

Which suite each benchmark belongs to, and how it is scored, comes from
eval_mini/benchmarks.py -- the same table that generates the lmms-eval configs.

Every result also carries the SAMPLE PROFILE it was measured at, and no command
here will put two profiles on one curve. A 100-document mean and a 300-document
mean of the same run are two estimates of the same quantity with different
standard errors, and a line drawn through both is not a curve of anything. So the
profile decides the directory (`bench_eval/` for the 100/100 default,
`bench_eval/n300_100/` otherwise), the WandB key namespace (`bench/*` against
`bench_n300_100/*`, chosen so no glob can reach both), and it is checked against
what the results files say was actually scored rather than taken on trust.
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_mini"))
from benchmarks import (BENCHMARKS, DEFAULT_SUITE_N, SUITES, as_suite_n,  # noqa: E402
                        base_task, profile_dir, profile_name, profile_of_payload,
                        wandb_prefix)


def flatten(results_paths):
    """Reduce lmms-eval results.json files to {wandb_key: value}.

    Which suite a benchmark belongs to comes from the benchmarks table, not from
    which file it arrived in. Deriving it from the caller's file grouping would
    misfile a benchmark the moment two suites shared an output directory, and a
    curve under the wrong heading is worse than a missing one.

    Every non-stderr scalar is kept, so a sub-category that moves before the
    headline metric does stays visible rather than being averaged away.
    """
    metrics, primaries, seen = {}, defaultdict(list), defaultdict(set)
    for path in results_paths:
        with open(path) as fh:
            payload = json.load(fh)
        for task, res in (payload.get("results") or {}).items():
            spec = BENCHMARKS.get(base_task(task))
            if spec is None:
                continue
            base, suite = base_task(task), spec["suite"]
            seen[suite].add(base)
            for key, value in res.items():
                if not isinstance(value, (int, float)) or "_stderr" in key:
                    continue
                metrics[f"bench/{suite}/{base}/{key.split(',')[0]}"] = float(value)

            value = res.get(spec["metric"])
            if isinstance(value, (int, float)):
                normalized = float(value) / spec["scale"]
                metrics[f"bench/{suite}/{base}"] = normalized
                if spec["in_mean"]:
                    primaries[suite].append(normalized)

    for suite in SUITES:
        if not seen[suite]:
            continue
        if primaries[suite]:
            metrics[f"bench/{suite}/mean"] = sum(primaries[suite]) / len(primaries[suite])
        metrics[f"bench/{suite}/n_benchmarks"] = float(len(seen[suite]))
    return metrics


def step_file(run_dir, step, sample_n=None):
    return profile_dir(Path(run_dir) / "bench_eval", sample_n) / f"step-{step}.json"


def observed_sample_n(results_paths):
    """{mini task: documents evaluated}, read out of the results files themselves.

    lmms-eval records `n-samples` per task, so what a run actually scored is on
    record and does not have to be inferred from the flags the job was launched
    with. That is the difference between believing a directory name and checking
    it: the mini configs are regenerated on every job, and a job that regenerated
    them at one size and then reused a suite banked at another would otherwise
    produce a step file whose five benchmarks are not all the same benchmark.
    """
    sizes = {}
    for path in results_paths:
        payload = json.load(open(path))
        for task, counts in (payload.get("n-samples") or {}).items():
            if base_task(task) in BENCHMARKS and counts.get("original") is not None:
                sizes[task] = int(counts["original"])
    return sizes


def check_profile(observed, wanted, allow_short=False):
    """Refuse a collect whose results were not all scored at the intended size.

    Any mismatch is an error, in either direction. The tempting exception --
    "fewer is fine, the split must be smaller than the sample" -- is the exact
    shape of the mistake this is here to prevent: a suite banked at 100 and reused
    by a job asking for 300 also shows up as fewer, and would sail through. No
    benchmark in the table has a split under 300 (the smallest, mathvision's
    testmini, has 304), so today there is nothing legitimate to let through.

    `allow_short` exists for the day one of them does, and is deliberately an
    explicit act rather than a default.
    """
    problems = []
    for task, got in sorted(observed.items()):
        want = wanted.get(task)
        if want is None or got == want:
            continue
        if got < want and allow_short:
            continue
        problems.append(f"{task}: {got} documents scored, profile asks for {want}")
    return problems


def wandb_target():
    return (os.environ.get("WANDB_ENTITY", "nvr-israel"),
            os.environ.get("WANDB_PROJECT", "vlm_reasoning"))


def slug(name):
    return "".join(c if c.isalnum() else "-" for c in name.lower()).strip("-")


def do_collect(args):
    metrics = flatten(args.results)
    if not metrics:
        raise SystemExit("no recognised benchmark results in the given files")

    observed = observed_sample_n(args.results)
    wanted = json.loads(args.sample_n) if args.sample_n else dict(observed)
    problems = check_profile(observed, wanted, allow_short=args.allow_short)
    if problems:
        raise SystemExit(
            "refusing to collect: these results were not scored at the intended sample size, "
            "and a step file mixing two sizes puts two different measurements on one curve.\n  "
            + "\n  ".join(problems))

    # The profile comes from what was ASKED FOR, not from what happened to arrive.
    # The difference matters in exactly one situation, and it is a destructive one:
    # if every natural unit fails and only the carried-over non-natural results are
    # in hand, the observed sizes are all 100 -- so inferring the profile would
    # file a partial 300-document result as a 100-document one, on top of the real
    # 100-document result that is already there. `observed` is what validates;
    # `wanted` is what decides where this goes.
    suite_n = as_suite_n(wanted if args.sample_n else observed)
    uneven = {s: v for s, v in suite_n.items() if isinstance(v, tuple)}
    # Not an error: a per-task override (`--task-n mmstar=200`, which is how the
    # one benchmark too expensive to triple is kept inside a one-hour job) is a
    # legitimate profile, and it gets a name of its own so it cannot be confused
    # with a uniform one. What it does mean is that `bench/<suite>/mean` averages
    # benchmarks measured on different numbers of documents, so its standard error
    # is no longer one number -- which is what the pooled item-level endpoint in
    # bench_samples.py is for.
    for suite, sizes in uneven.items():
        print(f"  NOTE: {suite} benchmarks were scored at {sizes} documents, not one size. "
              f"bench/{suite}/mean weights them equally regardless; for a defensible "
              f"standard error use `bench_samples.py --compare`, which pools items.")

    payload = {"step": args.step, "metrics": metrics,
               "sample_n": suite_n, "sample_n_per_task": observed}
    if args.carried:
        # A suite reused from an earlier profile rather than re-generated. Recorded
        # so that nobody later reads these numbers as an independent re-measurement:
        # they are the SAME evaluation of the same model on the same documents, and
        # the only honest thing to do with a repeated measurement is say so.
        payload["carried_over"] = json.loads(args.carried)

    dest = step_file(args.run_dir, args.step, suite_n)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Write-then-rename: the trainer polls this directory, and a half-written file
    # would be read as a corrupt one exactly once and then never retried.
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True))
    tmp.replace(dest)

    print(f"wrote {dest} ({len(metrics)} scalars)")
    print(f"  profile {profile_name(suite_n)}: "
          + ", ".join(f"{s}={suite_n[s]}" for s in SUITES)
          + (f"   [{', '.join(sorted(payload['carried_over']))} carried over]"
             if args.carried else ""))
    for suite in SUITES:
        mean = metrics.get(f"bench/{suite}/mean")
        n = metrics.get(f"bench/{suite}/n_benchmarks")
        expected = sum(1 for s in BENCHMARKS.values() if s["suite"] == suite)
        if mean is None:
            print(f"  {suite:11s} MISSING -- no benchmark of this suite produced results")
            continue
        short = "" if int(n) == expected else f"   [{expected - int(n)} benchmark(s) missing]"
        print(f"  {suite:11s} mean {mean:.4f} over {int(n)}/{expected} benchmarks{short}")


def do_backfill(args):
    """Push every recorded step into the training run, for points that landed late.

    The trainer logs these live while it is running. Anything finished after it
    exited has to be appended by re-attaching to the same run, which is safe only
    because there is no longer a second writer.
    """
    import wandb

    wanted = as_suite_n(json.loads(args.sample_n)) if args.sample_n else DEFAULT_SUITE_N
    directory = profile_dir(Path(args.run_dir) / "bench_eval", wanted)
    files = sorted(glob.glob(str(directory / "step-*.json")),
                   key=lambda p: int(Path(p).stem.split("-")[1]))
    if not files:
        raise SystemExit(f"no {profile_name(wanted)} results under {directory}")

    # One profile per push, and the profile decides the key namespace. A step file
    # sitting in the wrong directory is dropped rather than logged under the
    # neighbouring name: the whole point of the namespace is that a panel cannot
    # reach two sample sizes, and that guarantee is worth more than a stray point.
    prefix = wandb_prefix(wanted)
    payloads, skipped = [], []
    for path in files:
        payload = json.loads(Path(path).read_text())
        if profile_of_payload(payload) != wanted:
            skipped.append((path, profile_name(profile_of_payload(payload))))
            continue
        payloads.append(payload)
    if not payloads:
        raise SystemExit(f"no step file under {directory} was scored at {profile_name(wanted)}")

    entity, project = wandb_target()
    run = wandb.init(id=args.wandb_run_id, project=project, entity=entity, resume="allow")
    run.define_metric(f"{prefix}/step")
    run.define_metric(f"{prefix}/*", step_metric=f"{prefix}/step")
    for payload in payloads:
        metrics = {f"{prefix}/{k.split('/', 1)[1]}": v for k, v in payload["metrics"].items()}
        run.log({**metrics, f"{prefix}/step": payload["step"]})
        print(f"  logged step {payload['step']} ({len(metrics)} scalars)")
    run.finish()
    print(f"backfilled {len(payloads)} checkpoints into run {args.wandb_run_id}")
    print(f"  sample profile {profile_name(wanted)} "
          + ", ".join(f"{s}={wanted[s]}" for s in SUITES)
          + f"   -> WandB keys {prefix}/*")
    for path, got in skipped:
        print(f"  SKIPPED {path}: scored at {got}, not {profile_name(wanted)}")


def do_publish_baseline(args):
    """Publish one already-scored baseline as its own WandB run, drawn as a flat line.

    A baseline has a single score, not a curve, so it belongs on the bench panels as
    a horizontal reference line. WandB line plots have no reference-line primitive,
    so the line is drawn the only way the panel understands: the SAME `bench/*` keys
    the training run logs, at two x-values, both carrying the same value. Two points
    at the ends of the run's x-range render as a horizontal line spanning it.

    It goes in a run of its own rather than as extra keys on the training run so the
    label comes from the run name and every existing bench panel picks it up at once
    -- adding the run to the report's run set is the whole edit, with no panel-by-
    panel key list to maintain. `job_type=bench_baseline` is there to filter on.

    The run id is derived from the label, so re-publishing a re-scored baseline lands
    in the same run (and keeps the report's run set valid) instead of accumulating
    near-duplicates. That makes an existing run an error rather than something to
    silently append a second pair of points to: two points at x=0 with different
    values would draw the baseline as a step, not a line.
    """
    import wandb

    wanted = as_suite_n(json.loads(args.sample_n)) if args.sample_n else DEFAULT_SUITE_N
    path = step_file(args.run_dir, 0, wanted)
    if not path.exists():
        raise SystemExit(f"no {profile_name(wanted)} baseline result at {path} -- score the model first")
    payload = json.loads(path.read_text())
    got = profile_of_payload(payload)
    if got != wanted:
        raise SystemExit(f"{path} was scored at {profile_name(got)}, not {profile_name(wanted)}")
    prefix = wandb_prefix(wanted)
    metrics = {f"{prefix}/{k.split('/', 1)[1]}": v for k, v in payload["metrics"].items()}

    model_file = Path(args.run_dir) / "bench_eval" / "base_model.txt"
    model = model_file.read_text().strip() if model_file.exists() else "unknown"

    entity, project = wandb_target()
    # The sample size is part of the identity of a baseline, not a detail of it: a
    # 100-item line and a 300-item line of the same model are two different
    # estimates and must not overwrite one another, nor share a legend entry.
    #
    # Which is why the DISPLAY NAME carries it too, not just the run id. The id
    # keeps them apart in the API; the name is what a panel legend shows, and two
    # reference lines both labelled `baseline/overlap-8k` differing only in an
    # invisible sample size is precisely the confusion all of this exists to
    # prevent.
    suffix = "" if wanted == DEFAULT_SUITE_N else f"-{profile_name(wanted)}"
    run_id = f"bench-baseline-{slug(args.name)}{suffix}"
    run_name = f"baseline/{args.name}" + ("" if not suffix else f"@{profile_name(wanted)}")
    # Anything other than a clean "no such run" aborts rather than being read as
    # "does not exist": a transient API error would otherwise send this down the
    # create path and append a second pair of points to a run that already has one.
    existing = None
    if os.environ.get("WANDB_MODE") not in ("offline", "disabled", "dryrun"):
        try:
            existing = wandb.Api().run(f"{entity}/{project}/{run_id}")
        except Exception as exc:
            if "not find" not in str(exc).lower():
                raise SystemExit(f"could not check whether {run_id} exists: {exc}")
    if existing is not None:
        if not args.overwrite:
            raise SystemExit(
                f"{entity}/{project}/{run_id} already exists ({existing.name}).\n"
                f"Pass --overwrite to delete and republish it, or --name something else.")
        print(f"  --overwrite: deleting existing run {run_id} ({existing.url})")
        existing.delete()

    run = wandb.init(
        id=run_id, name=run_name, project=project, entity=entity,
        job_type="bench_baseline", group="bench_baselines", resume="allow",
        config={"baseline": args.name, "baseline_model": model, "bench_step_span": args.span,
                "sample_profile": profile_name(wanted), "sample_n": wanted},
    )
    run.define_metric(f"{prefix}/step")
    run.define_metric(f"{prefix}/*", step_metric=f"{prefix}/step")
    # sorted(set(...)): --span 0 (or omitted) is a legitimate single point rather
    # than the same x logged twice, which WandB would draw as a zero-length line.
    for x in sorted({0, args.span}):
        run.log({**metrics, f"{prefix}/step": x})
    url = run.url or f"{entity}/{project}/{run_id} (offline)"
    run.finish()

    print(f"published baseline '{run_name}' ({len(metrics)} scalars) spanning "
          f"{prefix}/step 0..{args.span}")
    print(f"  model:   {model}")
    print(f"  profile: {profile_name(wanted)} "
          + ", ".join(f"{s}={wanted[s]}" for s in SUITES) + f"   -> keys {prefix}/*")
    print(f"  run:     {url}")


def do_carried_from(directory):
    """{unit tag: profile} for the banked units in this directory that were reused.

    Printed as JSON for run_bench_eval.sh to hand back to --collect. It lives here
    rather than as a shell heredoc because the marker format is this file's
    business, and a JSON writer spelled in bash is a JSON writer that will one day
    emit something unparseable when a path contains a quote.
    """
    out = {}
    for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
        try:
            marker = json.load(open(path))
        except Exception:
            continue
        if marker.get("carried_from"):
            out[Path(path).stem] = marker["carried_from"]
    print(json.dumps(out, sort_keys=True))


def do_plan(args):
    """Print the banking units a job should attempt, largest first, as TSV.

    Called by run_bench_eval.sh so the shell never has to know the cost table or
    which benchmark needs its own resolution. One line per unit:

        tag <TAB> task,task <TAB> minutes <TAB> extra lmms-eval args
    """
    from benchmarks import plan_units

    sample_n = json.loads(args.sample_n) if args.sample_n else None
    for tag, tasks, extra, minutes in plan_units(args.bank, sample_n, args.num_gpus, args.window):
        print(f"{tag}\t{','.join(tasks)}\t{minutes:.0f}\t{extra}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--collect", action="store_true", help="reduce lmms-eval output to one step file")
    p.add_argument("--plan", action="store_true",
                   help="print the banking units and their minute estimates as TSV")
    p.add_argument("--bank", choices=("auto", "suite", "task"), default="auto",
                   help="the unit a job banks progress in; auto sizes it to --window (--plan)")
    p.add_argument("--window", type=int, default=0,
                   help="usable minutes in the allocation, for --bank auto (--plan)")
    p.add_argument("--num-gpus", type=int, default=1, help="allocation size, for the estimate (--plan)")
    p.add_argument("--carried-from", metavar="DIR",
                   help="print {unit: profile} for the reused banked units in DIR")
    p.add_argument("--backfill", action="store_true", help="push recorded steps into a WandB run")
    p.add_argument("--publish-baseline", action="store_true",
                   help="publish this dir's step-0 score as a flat-line WandB run")
    p.add_argument("--run-dir", help="the training run's output_dir")
    p.add_argument("--step", type=int, help="checkpoint step (--collect)")
    p.add_argument("--results", nargs="*", default=[],
                   help="lmms-eval results.json files, any suite (--collect)")
    p.add_argument("--wandb-run-id", help="WandB run id to append to (--backfill)")
    p.add_argument("--name", help="baseline label, shown in the legend (--publish-baseline)")
    p.add_argument("--span", type=int, default=0,
                   help="last bench/step the line should reach (--publish-baseline)")
    p.add_argument("--overwrite", action="store_true",
                   help="delete and republish an existing baseline run (--publish-baseline)")
    p.add_argument("--sample-n", metavar="JSON",
                   help='sample profile, e.g. \'{"natural": 300, "nonnatural": 100}\' or the '
                        'per-task sample_n.json the configs were generated with. Selects which '
                        'results to read and which WandB keys to write; defaults to 100/100.')
    p.add_argument("--allow-short", action="store_true",
                   help="permit a benchmark scored on fewer documents than the profile asks for")
    p.add_argument("--carried", metavar="JSON",
                   help="{suite: source} for suites reused from another profile (--collect)")
    args = p.parse_args()

    if args.carried_from:
        do_carried_from(args.carried_from)
        return
    if args.plan:
        do_plan(args)
        return
    if not args.run_dir:
        raise SystemExit("--run-dir is required")

    if args.collect:
        if args.step is None or not args.results:
            raise SystemExit("--collect needs --step and --results")
        do_collect(args)
    elif args.backfill:
        if not args.wandb_run_id:
            raise SystemExit("--backfill needs --wandb-run-id")
        do_backfill(args)
    elif args.publish_baseline:
        if not args.name:
            raise SystemExit("--publish-baseline needs --name")
        do_publish_baseline(args)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
