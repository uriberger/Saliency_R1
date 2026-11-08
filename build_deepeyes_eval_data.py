#!/usr/bin/env python
"""Stage everything `eval/eval_vstar.py` and `eval/eval_hrbench.py` need, on CPU.

    python build_deepeyes_eval_data.py --out-dir outputs/deepeyes_gate/data

Their eval scripts are OpenAI-API clients with the bbox/crop loop client-side, so they
really are model-agnostic -- `docs/deepeyes-build-prompts.md` is right about that. What
it is wrong about is "need zero changes". FOUR things stop them running here, all four in
the HR-Bench leg, and all four silent-at-clone-time and loud-at-minute-40-of-a-GPU-
allocation, which is the worst place to find them. The V* leg really is clean.

  1. `eval_hrbench.py` USES `pd.read_csv` (:321) and `copy.deepcopy` (:149) and IMPORTS
     NEITHER. The released file simply raises NameError. `pd` goes first, in __main__
     before any worker starts, so you lose the allocation immediately; `copy` would go
     next, inside the pool. This patches both in, idempotently, and touches nothing else.

  2. `judge_result_hrbench.py:39` reads `args.vstar_bench_path` on a parser that only
     defines `--hrbench_path`. AttributeError at module level, before argparse can even
     print help. Taken with (1) this means the published HR-Bench legs cannot have been
     run from the published files -- worth knowing before trusting their HR-Bench column.

  3. `eval_hrbench.py` reads `<hrbench_path>/hr_bench_{4k,8k}.tsv` with a `category`
     column and an `image` column of base64. The HF cache has `DreamMr/HR-Bench` as
     PARQUET. The columns are otherwise an exact match -- index, answer, question,
     A, B, C, D, category, cycle_category, image -- so this is a format conversion and
     not a reformatting of the benchmark. Parquet compresses the base64 and TSV does not,
     so the TSVs are ~4x the parquet on disk (1.9 GB + 4.6 GB); budget the space.

V* needs no conversion at all: `craigwu/vstar_bench`'s snapshot is already the
`direct_attributes/` + `relative_position/` tree of `<id>.jpg` + `<id>.json` that
`eval_vstar.py` walks with `os.listdir`. 115 + 76 = 191 questions, which is the count
the paper reports. We only resolve and print the path.

WHY THE TSV IS VERIFIED BY READING IT BACK. HR-Bench questions are free text and may
carry a tab or a newline; pandas quotes those on write and unquotes them on read, but
only if both sides use the default dialect, which their `pd.read_csv(sep='\t')` does.
A silent mangle here would change the question the model is asked and show up as a
benchmark number rather than as an error, so the round-trip is asserted rather than
assumed.

Everything here is CPU-only and offline apart from the one-off `git clone`. Send it to
cpu_short; it does not belong behind a GPU allocation.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

DEEPEYES_GIT = "https://github.com/Visual-Agent/DeepEyes.git"

#: Their repo is REFERENCE ONLY (docs/deepeyes-build-prompts.md §"Practical notes"):
#: never cloned into the tree, never executed for training. The eval scripts are the
#: one part we do run, and we run them from scratch, out of the way of the central tree.
DEFAULT_REF_ROOT = Path("/home/uberger/scratch/deepeyes_ref")

#: The two missing imports, as (line to insert, anchor it is inserted after). The anchor
#: is matched exactly and the insert is skipped if the import is already present, so
#: re-running this on an already-patched clone is a no-op rather than a duplicate import.
EVAL_IMPORT_FIXES = {
    "eval/eval_hrbench.py": [
        ("import pandas as pd", "import requests\n"),
        ("import copy", "import requests\n"),
    ],
}

#: The third fault, which is not a missing import but a wrong attribute.
#: `judge_result_hrbench.py` was made from `judge_result.py` by renaming the benchmark
#: path argument to `--hrbench_path` and then NOT renaming the one place it is read, so
#: :39 still asks for `args.vstar_bench_path` on a parser that no longer defines it. It
#: raises AttributeError at module level, before argparse help, before any row -- the
#: HR-Bench judge leg of their release cannot have been run as published. The variable it
#: assigns is dead afterwards, so pointing it at the right attribute is the whole fix.
EVAL_LINE_FIXES = {
    "eval/judge_result_hrbench.py": [
        ("hrbench_path = args.vstar_bench_path", "hrbench_path = args.hrbench_path"),
    ],
}

HRBENCH_REPO = "DreamMr/HR-Bench"
HRBENCH_SPLITS = ("hr_bench_4k", "hr_bench_8k")
#: The columns eval_hrbench.py's `process()` actually reads off a row. If the upstream
#: parquet ever loses one of these the conversion must fail here, not produce a TSV that
#: KeyErrors inside a multiprocessing worker where the traceback is swallowed.
HRBENCH_REQUIRED = ("image", "question", "answer", "A", "B", "C", "D", "category")

VSTAR_REPO = "craigwu/vstar_bench"
VSTAR_SPLITS = ("direct_attributes", "relative_position")


def _hf_snapshot(repo_id: str) -> Path:
    """Resolve a dataset snapshot already in the cache, without touching the network.

    `local_files_only=True` is the point: the GPU nodes run with HF_HUB_OFFLINE=1 and a
    builder that quietly succeeds here by downloading would be a builder that fails
    there. If it is not cached, say so now.
    """
    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(repo_id, repo_type="dataset", local_files_only=True))
    except Exception as exc:  # noqa: BLE001 - the message matters more than the type
        raise SystemExit(
            f"ERROR: {repo_id} is not in the HF cache at HF_HOME={os.environ.get('HF_HOME')}.\n"
            f"       Pre-stage it from a login node (which has internet) before submitting.\n"
            f"       {exc}"
        ) from exc


def stage_eval_scripts(ref_root: Path) -> Path:
    """Clone (or reuse) their repo under scratch and patch in the two missing imports."""
    repo = ref_root / "DeepEyes"
    if not repo.exists():
        ref_root.mkdir(parents=True, exist_ok=True)
        print(f"[stage] cloning {DEEPEYES_GIT} -> {repo}", flush=True)
        subprocess.run(["git", "clone", "--depth", "1", DEEPEYES_GIT, str(repo)], check=True)
    else:
        print(f"[stage] reusing {repo}", flush=True)

    for rel, fixes in EVAL_IMPORT_FIXES.items():
        path = repo / rel
        text = path.read_text()
        for missing, anchor in fixes:
            if missing in text:
                print(f"[stage] {rel}: '{missing}' already present", flush=True)
                continue
            if anchor not in text:
                raise SystemExit(
                    f"ERROR: cannot patch {rel}: anchor {anchor!r} not found. Their file "
                    f"changed upstream; re-read it before trusting this script."
                )
            text = text.replace(anchor, anchor + missing + "\n", 1)
            print(f"[stage] {rel}: inserted '{missing}'", flush=True)
        path.write_text(text)

    for rel, fixes in EVAL_LINE_FIXES.items():
        path = repo / rel
        text = path.read_text()
        for bad, good in fixes:
            if good in text:
                print(f"[stage] {rel}: already fixed ({good})", flush=True)
                continue
            if bad not in text:
                raise SystemExit(
                    f"ERROR: cannot patch {rel}: expected line {bad!r} not found. Their "
                    f"file changed upstream; re-read it before trusting this script."
                )
            text = text.replace(bad, good, 1)
            print(f"[stage] {rel}: {bad!r} -> {good!r}", flush=True)
        path.write_text(text)

    # Fail now if a patch did not take: a NameError inside a multiprocessing worker is
    # reported as a dead pool, not as a missing import.
    for rel, fixes in EVAL_IMPORT_FIXES.items():
        src = (repo / rel).read_text()
        for missing, _ in fixes:
            assert missing in src, f"{rel} still missing {missing}"
    for rel, fixes in EVAL_LINE_FIXES.items():
        src = (repo / rel).read_text()
        for bad, good in fixes:
            assert good in src and bad not in src, f"{rel} still carries {bad}"
    return repo


def build_hrbench_tsv(out_dir: Path) -> dict:
    """Parquet -> the TSV `eval_hrbench.py` opens. Verified by reading it back."""
    import pandas as pd

    snap = _hf_snapshot(HRBENCH_REPO)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = {}
    for split in HRBENCH_SPLITS:
        src = snap / f"{split}.parquet"
        dst = out_dir / f"{split}.tsv"
        if not src.exists():
            raise SystemExit(f"ERROR: {src} missing from the {HRBENCH_REPO} snapshot.")

        df = pd.read_parquet(src)
        missing = [c for c in HRBENCH_REQUIRED if c not in df.columns]
        if missing:
            raise SystemExit(f"ERROR: {split} parquet lacks {missing}; got {list(df.columns)}")

        if dst.exists() and dst.stat().st_size > 0:
            print(f"[hrbench] {dst} exists ({dst.stat().st_size / 1e9:.2f} GB), reusing", flush=True)
        else:
            df.to_csv(dst, sep="\t", index=False)
            print(f"[hrbench] wrote {dst} ({dst.stat().st_size / 1e9:.2f} GB, {len(df)} rows)", flush=True)

        back = pd.read_csv(dst, sep="\t")
        assert len(back) == len(df), f"{split}: TSV round-trip changed the row count"
        for col in HRBENCH_REQUIRED:
            if not (back[col].astype(str) == df[col].astype(str)).all():
                raise SystemExit(
                    f"ERROR: {split}: column '{col}' did not survive the TSV round-trip. "
                    f"The model would be asked a different question than the parquet holds."
                )
        info[split] = dict(rows=int(len(df)),
                           categories={str(k): int(v) for k, v in df["category"].value_counts().items()},
                           tsv=str(dst))
    return info


def check_vstar() -> dict:
    """V* needs no conversion; assert the tree is the shape eval_vstar.py walks."""
    snap = _hf_snapshot(VSTAR_REPO)
    info = {"root": str(snap)}
    total = 0
    for split in VSTAR_SPLITS:
        d = snap / split
        if not d.is_dir():
            raise SystemExit(f"ERROR: {d} missing; eval_vstar.py iterates exactly {VSTAR_SPLITS}.")
        jpgs = sorted(p for p in d.iterdir() if p.suffix == ".jpg")
        # eval_vstar.py derives the annotation path by `img.replace('.jpg', '.json')`,
        # so a picture without its json is not a skipped row, it is a crashed worker.
        orphans = [p.name for p in jpgs if not p.with_suffix(".json").exists()]
        if orphans:
            raise SystemExit(f"ERROR: {split}: {len(orphans)} jpg without a json, e.g. {orphans[:3]}")
        info[split] = len(jpgs)
        total += len(jpgs)
    info["total"] = total
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True,
                    help="where the HR-Bench TSVs and the manifest go")
    ap.add_argument("--ref-root", default=str(DEFAULT_REF_ROOT),
                    help="scratch directory holding the read-only DeepEyes clone")
    ap.add_argument("--skip-clone", action="store_true",
                    help="assume the clone is already staged and patched")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {}
    if not args.skip_clone:
        manifest["deepeyes_repo"] = str(stage_eval_scripts(Path(args.ref_root)))
    manifest["vstar"] = check_vstar()
    manifest["hrbench"] = build_hrbench_tsv(out_dir)
    manifest["hrbench_path"] = str(out_dir)

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("\n" + "=" * 74)
    print(json.dumps(manifest, indent=2))
    print("=" * 74)
    print(f"\n--vstar-bench-path {manifest['vstar']['root']}")
    print(f"--hrbench-path     {manifest['hrbench_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
