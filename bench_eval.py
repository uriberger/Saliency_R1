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
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_mini"))
from benchmarks import BENCHMARKS, SUITES, base_task  # noqa: E402


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


def step_file(run_dir, step):
    return Path(run_dir) / "bench_eval" / f"step-{step}.json"


def wandb_target():
    return (os.environ.get("WANDB_ENTITY", "nvr-israel"),
            os.environ.get("WANDB_PROJECT", "vlm_reasoning"))


def slug(name):
    return "".join(c if c.isalnum() else "-" for c in name.lower()).strip("-")


def do_collect(args):
    metrics = flatten(args.results)
    if not metrics:
        raise SystemExit("no recognised benchmark results in the given files")

    dest = step_file(args.run_dir, args.step)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Write-then-rename: the trainer polls this directory, and a half-written file
    # would be read as a corrupt one exactly once and then never retried.
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"step": args.step, "metrics": metrics}, indent=1, sort_keys=True))
    tmp.replace(dest)

    print(f"wrote {dest} ({len(metrics)} scalars)")
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

    files = sorted(glob.glob(str(Path(args.run_dir) / "bench_eval" / "step-*.json")),
                   key=lambda p: int(Path(p).stem.split("-")[1]))
    if not files:
        raise SystemExit(f"no results under {args.run_dir}/bench_eval")

    entity, project = wandb_target()
    run = wandb.init(id=args.wandb_run_id, project=project, entity=entity, resume="allow")
    run.define_metric("bench/step")
    run.define_metric("bench/*", step_metric="bench/step")
    for path in files:
        payload = json.loads(Path(path).read_text())
        run.log({**payload["metrics"], "bench/step": payload["step"]})
        print(f"  logged step {payload['step']} ({len(payload['metrics'])} scalars)")
    run.finish()
    print(f"backfilled {len(files)} checkpoints into run {args.wandb_run_id}")


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

    path = step_file(args.run_dir, 0)
    if not path.exists():
        raise SystemExit(f"no baseline result at {path} -- score the model first")
    metrics = json.loads(path.read_text())["metrics"]

    model_file = Path(args.run_dir) / "bench_eval" / "base_model.txt"
    model = model_file.read_text().strip() if model_file.exists() else "unknown"

    entity, project = wandb_target()
    run_id = f"bench-baseline-{slug(args.name)}"
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
        id=run_id, name=f"baseline/{args.name}", project=project, entity=entity,
        job_type="bench_baseline", group="bench_baselines", resume="allow",
        config={"baseline": args.name, "baseline_model": model, "bench_step_span": args.span},
    )
    run.define_metric("bench/step")
    run.define_metric("bench/*", step_metric="bench/step")
    # sorted(set(...)): --span 0 (or omitted) is a legitimate single point rather
    # than the same x logged twice, which WandB would draw as a zero-length line.
    for x in sorted({0, args.span}):
        run.log({**metrics, "bench/step": x})
    url = run.url or f"{entity}/{project}/{run_id} (offline)"
    run.finish()

    print(f"published baseline '{args.name}' ({len(metrics)} scalars) spanning "
          f"bench/step 0..{args.span}")
    print(f"  model: {model}")
    print(f"  run:   {url}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--collect", action="store_true", help="reduce lmms-eval output to one step file")
    p.add_argument("--backfill", action="store_true", help="push recorded steps into a WandB run")
    p.add_argument("--publish-baseline", action="store_true",
                   help="publish this dir's step-0 score as a flat-line WandB run")
    p.add_argument("--run-dir", required=True, help="the training run's output_dir")
    p.add_argument("--step", type=int, help="checkpoint step (--collect)")
    p.add_argument("--results", nargs="*", default=[],
                   help="lmms-eval results.json files, any suite (--collect)")
    p.add_argument("--wandb-run-id", help="WandB run id to append to (--backfill)")
    p.add_argument("--name", help="baseline label, shown in the legend (--publish-baseline)")
    p.add_argument("--span", type=int, default=0,
                   help="last bench/step the line should reach (--publish-baseline)")
    p.add_argument("--overwrite", action="store_true",
                   help="delete and republish an existing baseline run (--publish-baseline)")
    args = p.parse_args()

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
