#!/usr/bin/env python
"""Per-GPU utilization report for finished (or running) jobs, from the local wandb log.

Why this works retroactively: every run launched with `--report_to wandb` has a
system-metrics sampler running alongside training, polling NVML about every 15 s and
writing the samples into the run's binary `.wandb` transaction log under
`trl_repo/wandb/run-*/`. That file is written whether wandb is online or offline, so
the per-GPU history of a job that finished weeks ago is already sitting on disk --
nothing had to be instrumented in advance.

The keys used here, per GPU index i:

    gpu.i.gpu                   SM utilization %  -- fraction of sampled time at least
                                one kernel was resident. NOT how *well* the SMs are fed.
    gpu.i.memory                memory-controller utilization % (read/write activity)
    gpu.i.memoryAllocated       HBM used %
    gpu.i.powerWatts            board power draw -- the honest occupancy proxy, since a
                                GPU spinning in an NCCL all-reduce reads as 100% SM util
                                but draws far below its power limit.

Usage:
    python gpu_util_report.py                      # newest run
    python gpu_util_report.py --list               # what runs are available
    python gpu_util_report.py -r wov0.8            # newest run whose name contains this
    python gpu_util_report.py -r <path/to/run-dir>
    python gpu_util_report.py --skip-warmup 10     # drop the first 10 min (model load)
    python gpu_util_report.py --timeline           # coarse util-over-time per GPU
    python gpu_util_report.py --csv out.csv        # raw samples for your own analysis
"""

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.realpath(__file__))
WANDB_DIR = os.path.join(REPO, "trl_repo", "wandb")

# What we pull out of each sample, and how it is labelled in the table.
FIELDS = [
    ("gpu", "SM util %"),
    ("memory", "MemBW %"),
    ("memoryAllocated", "HBM %"),
    ("powerWatts", "Power W"),
]


def find_run(spec):
    """Resolve a run spec to a .wandb file: a path, a name substring, or None for newest."""
    if spec and os.path.isdir(spec):
        candidates = [spec]
    elif spec and os.path.isfile(spec):
        return spec
    else:
        candidates = sorted(
            (d for d in glob.glob(os.path.join(WANDB_DIR, "run-*")) if os.path.isdir(d)),
            key=os.path.getmtime,
            reverse=True,
        )
        if spec:
            candidates = [d for d in candidates if spec in os.path.basename(d)]
        if not candidates:
            sys.exit(f"no run matching {spec!r} under {WANDB_DIR}")

    for d in candidates:
        files = glob.glob(os.path.join(d, "*.wandb"))
        if files:
            return files[0]
    sys.exit(f"no .wandb file in {candidates[0]}")


def read_samples(wandb_file):
    """Yield (timestamp, {key: float}) for every system-stats record in the log."""
    from wandb.proto import wandb_internal_pb2 as pb
    from wandb.sdk.internal.datastore import DataStore

    ds = DataStore()
    ds.open_for_scan(wandb_file)
    record = pb.Record()
    while True:
        try:
            data = ds.scan_data()
        except Exception:
            break  # truncated tail: a killed/timed-out job leaves a partial last block
        if data is None:
            break
        record.Clear()
        try:
            record.ParseFromString(data)
        except Exception:
            continue
        if record.WhichOneof("record_type") != "stats":
            continue
        sample = {}
        for item in record.stats.item:
            if not item.key.startswith("gpu."):
                continue
            try:
                sample[item.key] = float(json.loads(item.value_json))
            except (ValueError, TypeError):
                pass
        if sample:
            yield record.stats.timestamp.seconds, sample


def pct(values, q):
    """Percentile by nearest rank, on an already-sorted list."""
    if not values:
        return float("nan")
    return values[min(len(values) - 1, int(q / 100.0 * len(values)))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-r", "--run", help="run dir, .wandb file, or name substring (default: newest)")
    ap.add_argument("--list", action="store_true", help="list available runs and exit")
    ap.add_argument("--skip-warmup", type=float, default=0.0,
                    help="ignore the first N minutes (model load / vLLM warmup)")
    ap.add_argument("--busy-threshold", type=float, default=5.0,
                    help="SM util %% above which a sample counts as busy (default 5)")
    ap.add_argument("--timeline", action="store_true", help="print util over time, per GPU")
    ap.add_argument("--timeline-bins", type=int, default=40)
    ap.add_argument("--csv", help="write raw per-sample rows here")
    args = ap.parse_args()

    if args.list:
        runs = sorted(glob.glob(os.path.join(WANDB_DIR, "run-*")), key=os.path.getmtime, reverse=True)
        for d in runs[:40]:
            print(os.path.basename(d))
        return

    wandb_file = find_run(args.run)
    print(f"run: {os.path.basename(os.path.dirname(wandb_file))}\n")

    samples = list(read_samples(wandb_file))
    if not samples:
        sys.exit("no GPU system metrics in this run "
                 "(job died before the first sample, or wandb monitoring was disabled)")

    t0 = samples[0][0]
    kept = [(t, s) for t, s in samples if (t - t0) / 60.0 >= args.skip_warmup]
    if not kept:
        sys.exit(f"--skip-warmup {args.skip_warmup} discarded all {len(samples)} samples "
                 f"(run only spans {(samples[-1][0] - t0) / 60.0:.1f} min)")

    span_min = (kept[-1][0] - kept[0][0]) / 60.0
    interval = span_min * 60.0 / max(1, len(kept) - 1)
    gpus = sorted({int(k.split(".")[1]) for _, s in kept for k in s})

    print(f"{len(kept)} samples over {span_min:.1f} min (~{interval:.0f}s apart), {len(gpus)} GPUs"
          + (f", first {args.skip_warmup:g} min skipped" if args.skip_warmup else ""))
    print()

    series = defaultdict(lambda: defaultdict(list))  # gpu -> field -> [values]
    for _, s in kept:
        for i in gpus:
            for field, _label in FIELDS:
                v = s.get(f"gpu.{i}.{field}")
                if v is not None:
                    series[i][field].append(v)

    header = f"{'GPU':>4} | {'SM util %':>26} | {'MemBW':>6} | {'HBM':>6} | {'Power':>6} | {'busy':>6}"
    print(header)
    print(f"{'':>4} | {'mean':>6} {'p10':>6} {'p50':>6} {'p90':>6} | {'mean':>6} | {'peak':>6} | {'mean':>6} | {'frac':>6}")
    print("-" * len(header))
    for i in gpus:
        util = sorted(series[i]["gpu"])
        if not util:
            continue
        mean_util = sum(util) / len(util)
        membw = series[i]["memory"]
        hbm = series[i]["memoryAllocated"]
        power = series[i]["powerWatts"]
        busy = sum(1 for v in util if v > args.busy_threshold) / len(util)
        print(f"{i:>4} | {mean_util:>6.1f} {pct(util, 10):>6.1f} {pct(util, 50):>6.1f} {pct(util, 90):>6.1f} | "
              f"{(sum(membw) / len(membw) if membw else float('nan')):>6.1f} | "
              f"{(max(hbm) if hbm else float('nan')):>6.1f} | "
              f"{(sum(power) / len(power) if power else float('nan')):>6.0f} | "
              f"{busy * 100:>5.1f}%")

    means = {i: sum(series[i]["gpu"]) / len(series[i]["gpu"]) for i in gpus if series[i]["gpu"]}
    if len(means) > 1:
        lo, hi = min(means, key=means.get), max(means, key=means.get)
        print(f"\nspread: GPU {hi} busiest at {means[hi]:.1f}%, GPU {lo} idlest at {means[lo]:.1f}% "
              f"(fleet mean {sum(means.values()) / len(means):.1f}%)")

    if args.timeline:
        print("\ntimeline (mean SM util % per bin; each column ~"
              f"{span_min / args.timeline_bins:.1f} min)")
        ramp = " .:-=+*#%@"
        for i in gpus:
            vals = series[i]["gpu"]
            if not vals:
                continue
            width = max(1, len(vals) // args.timeline_bins)
            bins = [vals[j:j + width] for j in range(0, len(vals), width)][:args.timeline_bins]
            row = "".join(ramp[min(len(ramp) - 1, int(sum(b) / len(b) / 100 * len(ramp)))] for b in bins)
            print(f"  gpu{i} |{row}|")
        print(f"  legend: '{ramp[0]}'=0%  '{ramp[-1]}'=100%")

    if args.csv:
        keys = [f"gpu.{i}.{f}" for i in gpus for f, _ in FIELDS]
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["elapsed_min"] + keys)
            for t, s in kept:
                w.writerow([f"{(t - kept[0][0]) / 60.0:.3f}"] + [s.get(k, "") for k in keys])
        print(f"\nwrote {len(kept)} rows to {args.csv}")


if __name__ == "__main__":
    main()
