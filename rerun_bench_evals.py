#!/usr/bin/env python
"""Re-score every checkpoint that already has a benchmark result, at a new sample size.

Raising the natural benchmarks from 100 documents to 300 invalidates every number
on the existing curve -- not because the old ones are wrong, but because a 100-item
mean and a 300-item mean are different estimators and a line through both is a
plot of nothing. So the historical curve has to be regenerated, and this is the
tool that works out what that means, what it costs, and dispatches it.

    python rerun_bench_evals.py                      # the work list and the bill
    python rerun_bench_evals.py --carry-over         # reuse what needs no GPU
    python rerun_bench_evals.py --dispatch           # submit one job
    python rerun_bench_evals.py --watch 300          # keep submitting as they finish

It writes no dispatcher of its own. `--dispatch` calls watch_bench_evals.sh, which
submits the same single-GPU, one-hour, resumable job that produces every other
point on the curve -- so a re-scored checkpoint is scored by exactly the recipe
that scored it the first time, and this file only decides the order.

**Nothing here overwrites an existing result.** The new profile writes to
`bench_eval/n300_100/`, the old one stays in `bench_eval/`, and bench_eval.py
refuses to collect results whose recorded document counts do not match the profile
it is writing into. Running this tool twice is a no-op for work already done.

**The non-natural suite is carried over, not re-run.** It stays at 100 documents,
so its result for the new profile is the SAME evaluation of the same model on the
same documents that already exists on disk -- re-generating it would spend more
than half the GPU time of the whole sweep reproducing a number we have. Instead
`--carry-over` writes the banking markers that point the job at the existing
results, and the step file records `carried_over` so the reuse is on the record
rather than implied. Pass --no-carry-over to generate them again anyway.
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_mini"))
from benchmarks import (DEFAULT_SUITE_N, SUITES, base_task, estimate_minutes,  # noqa: E402
                        plan_units, profile_dir, profile_name, task_sample_n)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_samples import (resolve_results, results_index, shorten,  # noqa: E402
                           step_files)

# Usable minutes in a one-hour allocation, after run_bench_eval.sh's SAFETY_MARGIN
# for container staging and teardown. Only used to turn minutes into a job count
# for the estimate printed below; the job itself measures its own clock.
USABLE_MINUTES = 55


def resolve_model(repo, run_dir, step, kind):
    """(model path, why it cannot be re-run).

    A checkpoint that has a result may no longer have weights: CKPT_KEEP_EVERY
    prunes adapters as training goes, so parts of a curve are unrepeatable. That
    is reported rather than skipped -- a curve with holes nobody was told about is
    worse than a shorter curve.
    """
    if step == 0 or kind == "baseline":
        base_file = Path(run_dir) / "bench_eval" / "base_model.txt"
        if not base_file.exists():
            return None, "no bench_eval/base_model.txt recording which model was scored"
        model = base_file.read_text().strip()
        if not os.path.isdir(model):
            return None, f"the model it was scored from is gone: {model}"
        return model, None

    ckpt = Path(repo) / "checkpoint" / os.path.basename(run_dir) / f"checkpoint-{step}"
    if not (ckpt / "adapter_config.json").is_file():
        return None, "checkpoint pruned (CKPT_KEEP_EVERY); no adapter on disk"
    weights = ckpt / "adapter_model.safetensors"
    if not (weights.is_file() and weights.stat().st_size > 0):
        return None, "adapter_model.safetensors missing or empty"
    return str(ckpt), None


def thin(entries, every):
    """Keep one result in every `every` steps, plus each run's first and last.

    Re-scoring all 186 existing results is ~500 single-GPU hours, i.e. ~500
    one-hour jobs, which is a different kind of decision from "regenerate the
    curve". Most of that buys resolution along an axis -- training step -- that
    the precision work is not about: the question is whether two RUNS differ, and
    that is answered by their endpoints and a handful of interior points.

    The ends are always kept. Step 0 is the baseline the run started from and the
    last checkpoint is what the run is reported by, and a curve missing either is
    missing the comparison it exists to support.
    """
    if not every or every <= 1:
        return entries
    last = {}
    for entry in entries:
        last[entry["run"]] = max(last.get(entry["run"], 0), entry["step"])
    return [e for e in entries
            if e["step"] == 0 or e["step"] == last[e["run"]] or e["step"] % every == 0]


def work_list(repo, sample_n, first=(), runs=None):
    """Every (run, step) that has a result, in the order it should be redone.

    Baselines first overall: without a re-scored baseline none of the new numbers
    can be compared to anything, and they are five cheap jobs. Then whatever
    --first names -- the doc's one measurement worth more than all of this is the
    seed-variance replicate, and it is only worth having early. Then the training
    runs, newest checkpoint first within each, so an interrupted sweep leaves the
    end of every run scored rather than the beginning of one.
    """
    seen, entries = {}, []
    for run_dir, step, path, kind in step_files(Path(repo)):
        # A bench_one result is a copy of a curve point that run_bench_eval_steps.sh
        # wrote into the run directory as its last act. Re-running it would score
        # the same checkpoint twice under two names.
        if kind == "bench_one":
            continue
        key = (os.path.basename(run_dir), step)
        if key in seen:
            continue
        seen[key] = True
        if runs and not any(r.lower() in key[0].lower() for r in runs):
            continue
        model, blocked = resolve_model(repo, run_dir, step, kind)
        entries.append(dict(run_dir=str(run_dir), run=key[0], step=step, kind=kind,
                            model=model, blocked=blocked,
                            done=(profile_dir(Path(run_dir) / "bench_eval", sample_n)
                                  / f"step-{step}.json").exists()))

    def rank(entry):
        if entry["kind"] == "baseline":
            group = 0
        elif any(f.lower() in entry["run"].lower() for f in first):
            group = 1
        else:
            group = 2
        return (group, entry["run"], -entry["step"])

    return sorted(entries, key=rank)


def carry_over(entries, sample_n, bank, by_step, by_model, dry_run=False):
    """Bank the units whose sample size did not change, from results already on disk.

    A unit is carryable when every task in it is asked for at exactly the size it
    was already scored at. With the natural suite at 300 and the non-natural at
    100, that is the whole non-natural half -- 800 of the 1500 documents per
    checkpoint, and rather more than half the generation time, since the
    non-natural benchmarks are the long-answer ones.

    What is written is a banking marker in run_bench_eval.sh's own format,
    pointing at the results.json that already contains those tasks. The job then
    finds the unit banked and skips it, exactly as it would skip a unit an earlier
    job of its own had finished.

    This is a reuse, not a re-measurement, and it is recorded as one: the marker
    carries `carried_from` and the step file ends up with a `carried_over` field
    naming the profile the numbers came from.
    """
    old_profile = profile_name(DEFAULT_SUITE_N)
    banked = skipped = 0
    carried = set()
    for entry in entries:
        if entry["done"] or entry["blocked"]:
            continue
        run_dir = Path(entry["run_dir"])
        partial = profile_dir(run_dir / "bench_eval", sample_n) / "partial" / f"step-{entry['step']}"
        results, _ = resolve_results(run_dir, entry["step"], entry["kind"], by_step, by_model)
        # {task: the results.json that carries it}, so a unit is only banked
        # against a file that actually contains every one of its tasks.
        holder, sizes = {}, {}
        for path in results:
            payload = json.load(open(path))
            for task in (payload.get("results") or {}):
                counts = (payload.get("n-samples") or {}).get(task) or {}
                if counts.get("original") is not None:
                    holder[task] = path
                    sizes[task] = int(counts["original"])

        for tag, tasks, _extra, _minutes in plan_units(bank, sample_n):
            if not all(sizes.get(t) == sample_n.get(t) for t in tasks):
                continue
            files = {holder[t] for t in tasks}
            if len(files) != 1:
                # The unit's tasks came from two different invocations, so no single
                # results.json covers it. Banking one of them would silently drop
                # the rest of the unit from the step file.
                skipped += 1
                continue
            marker = partial / f"{tag}.json"
            if marker.exists():
                continue
            if dry_run:
                banked += 1
                carried.add((entry["run"], entry["step"], tag))
                continue
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(json.dumps({
                "sample_n": {t: sizes[t] for t in tasks},
                "results": files.pop(),
                "carried_from": old_profile,
            }))
            banked += 1
            carried.add((entry["run"], entry["step"], tag))
    return banked, skipped, carried


def cost(entries, sample_n, bank, carried=frozenset()):
    """Minutes of single-GPU time the outstanding work needs, after carry-over.

    `carried` covers the --dry-run case, where the markers that would make these
    units free have not been written yet. Without it the bill would be quoted
    including work the very next command removes.
    """
    units = plan_units(bank, sample_n)
    minutes = 0.0
    per_unit = defaultdict(float)
    for entry in entries:
        if entry["done"] or entry["blocked"]:
            continue
        partial = (profile_dir(Path(entry["run_dir"]) / "bench_eval", sample_n)
                   / "partial" / f"step-{entry['step']}")
        for tag, tasks, _extra, unit_minutes in units:
            if (partial / f"{tag}.json").exists():
                continue
            if (entry["run"], entry["step"], tag) in carried:
                continue
            minutes += unit_minutes
            per_unit[tag] += unit_minutes
        if entry["step"] != 0:
            minutes += 10  # the LoRA merge, once per checkpoint
    return minutes, per_unit


def in_flight():
    """Any eval job of ours queued or running, whatever run or profile it serves."""
    try:
        out = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%j"],
                             capture_output=True, text=True, timeout=60).stdout
    except Exception:
        # squeue unreachable: treat as busy rather than as free. Being wrong the
        # other way floods the queue with duplicate jobs.
        return ["<squeue unavailable>"]
    return [line for line in out.splitlines() if line.startswith("bencheval")]


def dispatch(args, entries, sample_n):
    """Submit one job, for the highest-priority run that still owes work.

    watch_bench_evals.sh is the dispatcher; this only chooses which run it points
    at and which steps it is allowed to touch. One job at a time across the whole
    sweep, because that is the arrangement the queue rewards and because a second
    job merging the same checkpoint would fight the first for the merge directory.
    """
    running = in_flight()
    if running:
        print(f"a bench eval job is already queued or running ({running[0]}) -- not submitting")
        return False

    owed = defaultdict(list)
    for entry in entries:
        if not (entry["done"] or entry["blocked"]):
            owed[(entry["run_dir"], entry["kind"])].append(entry["step"])
    if not owed:
        print("nothing left to re-score")
        return False

    (run_dir, _kind), steps = next(iter(owed.items()))
    cmd = ["bash", os.path.join(os.path.dirname(os.path.abspath(__file__)), "watch_bench_evals.sh"),
           "--run-dir", run_dir, "--once", "--bank", args.bank,
           "--steps", ",".join(str(s) for s in sorted(steps)),
           "--num-gpus", str(args.num_gpus), "--duration", str(args.duration)]
    for suite in SUITES:
        cmd += [f"--{suite.replace('nonnatural', 'nonnatural')}-n", str(args.suite_n[suite])]
    for task, n in (args.overrides or {}).items():
        cmd += ["--task-n", f"{task}={n}"]
    if args.partition:
        cmd += ["--partition", args.partition]

    print(f"dispatching {os.path.basename(run_dir)}   steps {sorted(steps)[:8]}"
          f"{' ...' if len(steps) > 8 else ''}")
    if args.dry_run:
        print("  " + " ".join(cmd))
        return False
    return subprocess.run(cmd).returncode == 0


def report(entries, sample_n, bank, args, carried=frozenset()):
    profile = profile_name(sample_n)
    todo = [e for e in entries if not (e["done"] or e["blocked"])]
    blocked = [e for e in entries if e["blocked"] and not e["done"]]

    print("=" * 78)
    print(f"Re-scoring {len(entries)} existing results at profile {profile}")
    for suite in SUITES:
        print(f"  {suite:11s} {args.suite_n[suite]} documents per benchmark"
              f"   (was {DEFAULT_SUITE_N[suite]})")
    for task, n in sorted(sample_n.items()):
        from benchmarks import BENCHMARKS
        if n != args.suite_n[BENCHMARKS[base_task(task)]["suite"]]:
            print(f"  override    {task} = {n}")
    print("=" * 78)

    by_run = defaultdict(lambda: [0, 0, 0])
    for entry in entries:
        slot = 1 if entry["done"] else (2 if entry["blocked"] else 0)
        by_run[(entry["kind"], entry["run"])][slot] += 1
    labels = shorten([run for kind, run in by_run if kind != "baseline"])
    print(f"\n{'':2s}{'run':56s} {'todo':>5s} {'done':>5s} {'stuck':>6s}")
    for (kind, run), (a, b, c) in sorted(by_run.items()):
        mark = "*" if kind == "baseline" else " "
        print(f"{mark} {labels.get(run, run)[:56]:56s} {a:5d} {b:5d} {c:6d}")
    print("  * baselines, dispatched first -- without them the new numbers compare to nothing")

    if blocked:
        print(f"\nNOT re-runnable ({len(blocked)}), so the {profile} curve will have holes here:")
        reasons = defaultdict(list)
        for entry in blocked:
            reasons[entry["blocked"]].append(f"{entry['run'][:40]}@{entry['step']}")
        for reason, who in sorted(reasons.items()):
            print(f"  {reason}   ({len(who)})")
            for name in who[:6]:
                print(f"    {name}")
            if len(who) > 6:
                print(f"    ... and {len(who) - 6} more")

    minutes, per_unit = cost(entries, sample_n, bank, carried)
    print(f"\nOutstanding: {len(todo)} checkpoints, {minutes / 60:.0f} GPU-hours on 1 GPU")
    print(f"  banking per {bank}; {math.ceil(minutes / USABLE_MINUTES)} jobs of "
          f"{args.duration}h at {USABLE_MINUTES} usable minutes each")
    for tag, unit_minutes in sorted(per_unit.items(), key=lambda kv: -kv[1])[:6]:
        print(f"    {tag:32s} {unit_minutes / 60:6.0f} GPU-hours")

    # What thinning would cost instead. Re-scoring every point of every curve is a
    # far bigger commitment than "regenerate the curve" sounds, and the resolution
    # it buys is along the training-step axis, which is not the axis the precision
    # work is about. Shown rather than chosen, because which points matter is a
    # judgement about the experiment.
    if not args.every:
        print("\n  Thinning the curve, if the whole history is more than this is worth")
        print("  (step 0 and each run's final checkpoint are always kept):")
        for every in (200, 500, 1000):
            kept = thin(entries, every)
            thinned, _ = cost(kept, sample_n, bank, carried)
            print(f"    --every {every:<5d} {len([e for e in kept if not (e['done'] or e['blocked'])]):3d}"
                  f" checkpoints, {thinned / 60:4.0f} GPU-hours, "
                  f"{math.ceil(thinned / USABLE_MINUTES):3d} jobs")

    # The one unit that cannot be banked inside the allocation, said plainly.
    for tag, tasks, _extra, unit_minutes in plan_units(bank, sample_n):
        if unit_minutes > USABLE_MINUTES:
            print(f"\n  WARNING: unit '{tag}' is budgeted {unit_minutes:.0f} min against a "
                  f"{USABLE_MINUTES} min window.")
            print(f"           Median is ~{unit_minutes / 1.3:.0f} min, so it will usually finish "
                  f"and sometimes be killed and repeated.")
            print(f"           To stop that costing jobs, cap it: "
                  f"--task-n {base_task(tasks[0])}={int(sample_n[tasks[0]] * 0.66)}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", default=os.path.dirname(os.path.abspath(__file__)))
    p.add_argument("--natural-n", type=int, default=300, help="documents per natural benchmark")
    p.add_argument("--nonnatural-n", type=int, default=100, help="documents per non-natural benchmark")
    p.add_argument("--task-n", action="append", metavar="TASK=N",
                   help="cap one benchmark, e.g. --task-n mmstar=200")
    p.add_argument("--bank", choices=("suite", "task"), default="task",
                   help="banking unit (default task: a 300-document suite cannot finish in 1h)")
    p.add_argument("--runs", action="append", help="only runs matching this substring")
    p.add_argument("--every", type=int, default=0, metavar="N",
                   help="re-score one checkpoint in every N steps, plus each run's ends")
    p.add_argument("--first", action="append", default=[],
                   help="put runs matching this substring at the head of the list")
    p.add_argument("--carry-over", dest="carry", action="store_true", default=True,
                   help="bank the unchanged units from existing results (default)")
    p.add_argument("--no-carry-over", dest="carry", action="store_false",
                   help="re-generate the non-natural suite instead of reusing it")
    p.add_argument("--dispatch", action="store_true", help="submit one job and exit")
    p.add_argument("--watch", type=int, metavar="SECONDS",
                   help="keep dispatching, polling this often")
    p.add_argument("--num-gpus", type=int, default=1)
    p.add_argument("--duration", type=int, default=1, help="hours per job (default 1)")
    p.add_argument("--partition")
    p.add_argument("--dry-run", action="store_true", help="print what would happen, change nothing")
    args = p.parse_args()

    args.overrides = {}
    for value in args.task_n or []:
        task, _, n = value.partition("=")
        args.overrides[task.strip()] = int(n)
    args.suite_n = {"natural": args.natural_n, "nonnatural": args.nonnatural_n}
    sample_n = task_sample_n(args.suite_n, args.overrides)

    entries = thin(work_list(args.repo, sample_n, first=args.first, runs=args.runs), args.every)
    if not entries:
        raise SystemExit("no existing results to re-score (check --runs / --every)")

    carried = frozenset()
    if args.carry:
        by_step, by_model = results_index()
        banked, skipped, carried = carry_over(entries, sample_n, args.bank, by_step, by_model,
                                              dry_run=args.dry_run)
        print(f"carried over {banked} unit(s) from {profile_name(DEFAULT_SUITE_N)} "
              f"-- no GPU time needed for those"
              + (f"; {skipped} unit(s) not carryable (tasks split across invocations)"
                 if skipped else "")
              + ("   [--dry-run: nothing written]" if args.dry_run else ""))

    report(entries, sample_n, args.bank, args, carried)

    if args.watch:
        import time
        while True:
            entries = thin(work_list(args.repo, sample_n, first=args.first, runs=args.runs),
                           args.every)
            if not [e for e in entries if not (e["done"] or e["blocked"])]:
                print("every re-runnable checkpoint is scored -- done")
                break
            dispatch(args, entries, sample_n)
            time.sleep(args.watch)
    elif args.dispatch:
        dispatch(args, entries, sample_n)
    else:
        print("\nNothing was submitted. --dispatch submits one job; --watch 300 keeps going.")


if __name__ == "__main__":
    main()
