#!/usr/bin/env python
"""Read a GRPO run's step-time budget straight out of the local `.wandb` files.

TRL already profiles the step -- `trl/extras/profiling.py` wraps `_prepare_inputs`,
`_calculate_rewards`, every individual reward function, `_compute_overlap_step_maps`,
`_move_model_to_vllm`, `vLLM.generate` and `compute_loss` -- but it logs the durations
ONLY to wandb, never to stdout. So a run whose job log is on disk still looks like an
unexplained 215 s a step, while the breakdown is sitting in the run directory in binary.

This reads it. No network, no wandb API key: the `.wandb` file is a local transaction log
and `wandb.sdk.internal.datastore` is the reader that ships with the client.

Two things about the shape of that log that this has to handle, because getting either
wrong silently halves a number:

  * The profiling records are NOT one-per-step. `profiling_context` calls `wandb.log()`
    with no `step=`, so each duration lands on its own auto-incremented `_step` with the
    trainer's real `train/global_step` carried alongside. Grouping is therefore by
    `train/global_step`, and a key can legitimately appear many times inside one of them
    (`compute_loss` runs once per micro-step, so 8 times at grad_accum 8).
  * The same wandb run id is reused across requeues (`WANDB_RESUME=allow`), so one file
    can hold several allocations and `_step` keeps climbing across them.

Usage:
    python omni_step_budget.py <run-dir-or-.wandb> [more ...] [--per-step] [--csv OUT]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import sys


def iter_history(path):
    """Yield {key: value} dicts, one per history record, from a .wandb file."""
    from wandb.sdk.internal import datastore
    from wandb.proto import wandb_internal_pb2 as pb

    ds = datastore.DataStore()
    ds.open_for_scan(path)
    while True:
        try:
            data = ds.scan_data()
        except Exception:
            return
        if data is None:
            return
        rec = pb.Record()
        try:
            rec.ParseFromString(data)
        except Exception:
            continue
        if rec.WhichOneof("record_type") != "history":
            continue
        row = {}
        for item in rec.history.item:
            # `nested_key` is a REPEATED string field, so it comes back as a protobuf
            # container even when it holds a single flat key like "train/global_step".
            # Joining it (rather than using it directly) is what makes it hashable.
            key = ".".join(item.nested_key) if len(item.nested_key) else item.key
            try:
                row[key] = json.loads(item.value_json)
            except Exception:
                row[key] = item.value_json
        yield row


PREFIX = "profiling/Time taken: "
# The SR1_LAP spans. They are NOT a decomposition of the profiled methods and must not be
# added to them: they cover the inline code BETWEEN those methods, plus (for
# `saliency_block`) the method itself. Reported in their own section for that reason.
LAP_PREFIX = "lap/"


def collect(paths):
    """-> per_step[global_step][short_key] = [durations], and per_step_scalars."""
    per_step = collections.defaultdict(lambda: collections.defaultdict(list))
    scalars = collections.defaultdict(dict)
    for path in paths:
        for row in iter_history(path):
            gs = row.get("train/global_step")
            if gs is None:
                continue
            gs = int(gs)
            for k, v in row.items():
                if k.startswith(PREFIX) and isinstance(v, (int, float)):
                    short = k[len(PREFIX):].split(".", 1)[-1]
                    per_step[gs][short].append(float(v))
                elif k.startswith(LAP_PREFIX) and isinstance(v, (int, float)):
                    per_step[gs]["lap: " + k[len(LAP_PREFIX):]].append(float(v))
                elif k.startswith(("completions/", "train/")) and isinstance(v, (int, float)):
                    scalars[gs][k] = float(v)
    return per_step, scalars


def summarise(per_step, scalars, lo=None, hi=None):
    steps = sorted(s for s in per_step if s > 0)
    if lo is not None:
        steps = [s for s in steps if s >= lo]
    if hi is not None:
        steps = [s for s in steps if s <= hi]
    totals = collections.defaultdict(list)   # key -> per-step summed seconds
    counts = collections.defaultdict(list)   # key -> per-step call count
    for s in steps:
        for k, vals in per_step[s].items():
            totals[k].append(sum(vals))
            counts[k].append(len(vals))
    return steps, totals, counts


def median(xs):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--per-step", action="store_true")
    ap.add_argument("--from-step", type=int, default=None)
    ap.add_argument("--to-step", type=int, default=None)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    files = []
    for p in args.paths:
        if os.path.isdir(p):
            files.extend(sorted(glob.glob(os.path.join(p, "*.wandb"))))
            files.extend(sorted(glob.glob(os.path.join(p, "run-*", "*.wandb"))))
        else:
            files.append(p)
    if not files:
        sys.exit("no .wandb files found")
    for f in files:
        print(f"# {f}", file=sys.stderr)

    per_step, scalars = collect(files)
    steps, totals, counts = summarise(per_step, scalars, args.from_step, args.to_step)
    if not steps:
        sys.exit("no profiled steps found")

    print(f"\nprofiled steps: {len(steps)}  (global_step {steps[0]}..{steps[-1]})\n")
    rows = sorted(totals.items(), key=lambda kv: -median(kv[1]))
    prof = [(k, v) for k, v in rows if not k.startswith("lap: ")]
    laps = [(k, v) for k, v in rows if k.startswith("lap: ")]

    def table(title, entries):
        if not entries:
            return
        print(f"\n{title}")
        print(f"{'phase':<44} {'calls/step':>10} {'median s':>9} {'mean s':>8} {'max s':>8}")
        print("-" * 84)
        for k, vals in entries:
            c = counts[k]
            print(f"{k:<44} {median(c):>10.0f} {median(vals):>9.1f} "
                  f"{sum(vals) / len(vals):>8.1f} {max(vals):>8.1f}")

    table("TRL's profiler (methods):", prof)

    # `_prepare_inputs` is the whole generation+reward block, so it must not be added to
    # its own children. Report it as the envelope and the children as its decomposition;
    # what is left over is the inline code SR1_LAP exists to name.
    env = "_prepare_inputs"
    if env in totals:
        # The direct children of `_prepare_inputs`. `_get_per_token_logps_and_entropies`
        # is excluded because on this configuration it is called from `compute_loss`, and
        # the individual reward funcs because `_calculate_rewards` already contains them.
        children = ("_compute_overlap_step_maps", "_compute_grad_step_maps",
                    "_compute_glimpse_step_maps", "vLLM.generate",
                    "transformers.generate", "transformers.generate_batch",
                    "_calculate_rewards", "_move_model_to_vllm")
        env_med = median(totals[env])
        kid_med = sum(median(totals[k]) for k in children if k in totals)
        print("-" * 84)
        print(f"{'envelope (_prepare_inputs)':<44} {'':>10} {env_med:>9.1f}")
        print(f"{'its profiled children, summed medians':<44} {'':>10} {kid_med:>9.1f}")
        print(f"{'UNATTRIBUTED inside the envelope':<44} {'':>10} {env_med - kid_med:>9.1f}")

    table("SR1_LAP spans (inline code; NOT additive with the table above):", laps)

    if args.per_step or args.csv:
        keys = [k for k, _ in rows]
        hdr = ["step", "mean_length", "max_length", "clipped_ratio"] + keys
        lines = [",".join(hdr)]
        for s in steps:
            sc = scalars.get(s, {})
            row = [str(s),
                   f"{sc.get('completions/mean_length', float('nan')):.1f}",
                   f"{sc.get('completions/max_length', float('nan')):.0f}",
                   f"{sc.get('completions/clipped_ratio', float('nan')):.4f}"]
            row += [f"{sum(per_step[s].get(k, [])):.2f}" for k in keys]
            lines.append(",".join(row))
        out = "\n".join(lines)
        if args.csv:
            with open(args.csv, "w") as fh:
                fh.write(out + "\n")
            print(f"\nwrote {args.csv}")
        else:
            print("\n" + out)


if __name__ == "__main__":
    main()
