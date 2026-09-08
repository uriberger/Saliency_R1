#!/usr/bin/env python
"""CPU checks for the BASELINES table in run_bench_baselines.sh. No GPU, no model.

    python test_bench_baselines_cpu.py

The invariant worth a test is not the parsing, it is the CACHE. lmms-eval derives both
its results directory and its `--use_cache` response cache from the model path's
BASENAME. Two baselines scored through the same path therefore share a cache, and the
second is handed the first's answers: it finishes fast, produces a plausible number, and
that number is the other baseline's. Nothing in the output says so.

That is not hypothetical here. `sft-coldstart` and both `sinkshift-*` arms are the same
checkpoint -- the arms differ only in an environment variable read inside the model
wrapper -- so all three would collide. The scoring loop symlinks every config variant
under its own label to keep them apart, and this file is what stops that being deleted as
redundant.

The table is read out of the shell script rather than duplicated here, so a row added to
one is checked by the other.
"""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
FAILURES = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def baselines():
    """[(label, model, env)] parsed out of the BASELINES=( ... ) block."""
    src = open(os.path.join(ROOT, "run_bench_baselines.sh")).read()
    block = re.search(r"^BASELINES=\((.*?)^\)", src, re.S | re.M)
    assert block, "no BASELINES=( ... ) block in run_bench_baselines.sh"
    out = []
    for line in block.group(1).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r'^"(.*)"$', line)
        assert m, f"entry is not a quoted string: {line}"
        parts = m.group(1).split("|")
        assert 2 <= len(parts) <= 3, f"entry has {len(parts)} fields: {line}"
        out.append((parts[0], parts[1], parts[2] if len(parts) == 3 else ""))
    return out


def scoring_path(label, model, env):
    """The path the loop actually scores through -- the label symlink for a variant."""
    return f"_models/{label}" if env else model


def main():
    print("run_bench_baselines.sh BASELINES checks")
    rows = baselines()
    print(f"  {len(rows)} baselines: {', '.join(r[0] for r in rows)}\n")

    labels = [r[0] for r in rows]
    check("labels are unique", len(set(labels)) == len(labels),
          str([x for x in labels if labels.count(x) > 1]))

    # THE cache check. Every baseline must be scored through a path of its own.
    paths = [scoring_path(*r) for r in rows]
    dupes = sorted({p for p in paths if paths.count(p) > 1})
    check("no two baselines are scored through the same model path", not dupes,
          f"would share a results dir and a response cache: {dupes}")

    # The same, stated the way it actually bites: variants of ONE checkpoint.
    shared = {}
    for label, model, env in rows:
        shared.setdefault(model, []).append((label, env))
    for model, group in shared.items():
        if len(group) < 2:
            continue
        check(f"{len(group)} baselines share {model.split('/')[-1]} -- all but one carry env",
              sum(1 for _l, e in group if not e) <= 1,
              str([l for l, e in group if not e]))

    # lmms-eval derives the slug from the basename, so two DIFFERENT paths whose last
    # component matches would still collide.
    bases = [p.rstrip("/").split("/")[-1] for p in paths]
    b_dupes = sorted({b for b in bases if bases.count(b) > 1})
    check("no two scoring paths share a basename", not b_dupes, str(b_dupes))

    for label, _model, env in rows:
        if not env:
            continue
        kvs = env.split()
        check(f"{label}: every config token is KEY=VALUE",
              all(re.fullmatch(r"[A-Z_][A-Z0-9_]*=\S+", kv) for kv in kvs), env)
        # An env-carrying baseline that never switches the wrapper is the stock model
        # under an experiment's name -- the exact mislabelling docs/per-completion-masks.md
        # records as having wasted two training runs.
        check(f"{label}: selects a model wrapper that reads its config",
              any(kv.startswith("BENCH_MODEL_TYPE=") for kv in kvs), env)
        if any(kv.startswith("BENCH_MODEL_TYPE=qwen3_vl_sinkshift") for kv in kvs):
            alpha = [kv for kv in kvs if kv.startswith("SINK_SHIFT_ALPHA=")]
            check(f"{label}: sets a non-zero alpha",
                  bool(alpha) and float(alpha[0].split("=")[1]) > 0,
                  f"{alpha or 'unset'} -- alpha 0 is the identity, i.e. the stock model")

    print(f"\n{'ALL PASS' if not FAILURES else f'{len(FAILURES)} FAILED: {FAILURES}'}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
