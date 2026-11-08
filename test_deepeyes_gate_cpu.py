#!/usr/bin/env python
"""CPU test for the DeepEyes gate: the staging patches, and both judge legs end to end.

    python test_deepeyes_gate_cpu.py              # rule-based paths only, no network
    python test_deepeyes_gate_cpu.py --judge      # also drive the real NVIDIA gateway

WHAT THIS ACTUALLY TESTS, and why it is worth having rather than waiting for a GPU.
The gate has three legs: serve, generate, judge. Only the first two need a GPU. The
JUDGE leg is the one substitution the build brief asks for -- their self-hosted
Qwen2.5-72B replaced by our hosted gateway -- and it is a pure text call, so it can be
proved today, on a login node, against the real endpoint. If it is going to be wrong it
is wrong here, not after a 191-row generation has been paid for.

It also pins the three faults build_deepeyes_eval_data.py patches, by asserting the
patched clone imports and runs. Those are regression guards: a future `git pull` in the
scratch clone silently restores all three.

THE EXPECTED SCORES ARE HAND-COMPUTED FROM THEIR RULES, not from a previous run. Their
two graders do NOT share a rule set and the difference is easy to miss:

    judge_result.py (V*)   gold is always option A, so it tests the LETTER 'A' and the
                           string 'A. <gold text>'.
    judge_result_hrbench.py  gold is a letter column, and its third rule is
                           `answer in pred_ans` -- a bare substring test on a ONE-CHARACTER
                           needle. "A red car" scores correct against gold 'A' for the
                           wrong reason. That is their grader's leniency, reproduced here
                           deliberately so the test fails if we ever accidentally change it.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REF_ROOT = Path(os.environ.get("DEEPEYES_REF_ROOT", "/home/uberger/scratch/deepeyes_ref"))
EVAL_SRC = REF_ROOT / "DeepEyes" / "eval"
PYTHON = sys.executable

JUDGE_URL = os.environ.get("OPENAI_BASE_URL", "https://inference-api.nvidia.com")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "azure/openai/gpt-4o-mini")


def judge_key() -> str:
    key = os.environ.get("NVIDIA_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    path = Path.home() / ".config" / "nvidia" / "api_key"
    return path.read_text().strip() if path.exists() else ""


def _msgs(text: str) -> list:
    """The `pred_output` shape judge_result.py walks. Content is never scored; it only
    has to have the role/content structure, because the V* grader iterates it."""
    return [{"role": "system", "content": "sys"},
            {"role": "user", "content": [{"type": "text", "text": "q"}]},
            {"role": "assistant", "content": text}]


# (pred_ans, expected_acc, which rule fires). Only rows marked needs_judge touch the net.
VSTAR_CASES = [
    ("A",                     1.0, "len-1 letter"),
    ("B",                     0.0, "len-1 letter"),
    ("A.",                    1.0, "len-2 with a dot"),
    ("C.",                    0.0, "len-2 with a dot"),
    ("A. a red fire hydrant", 1.0, "'A. <gold>' substring"),
    ("the hydrant is red",    1.0, "JUDGE: paraphrase of the gold"),
    ("it is a blue bicycle",  0.0, "JUDGE: contradicts the gold"),
]
VSTAR_NEEDS_JUDGE = {"the hydrant is red", "it is a blue bicycle"}

HRBENCH_CASES = [
    ("B",                      1.0, "len-1 letter == gold letter"),
    ("D",                      0.0, "len-1 letter != gold letter"),
    ("B.",                     1.0, "len-2 with a dot"),
    ("the sign reads 37B",     1.0, "'B' substring -- THEIR leniency, not a real match"),
    ("the sign reads 27 only", 0.0, "JUDGE: no 'B' anywhere, falls through"),
]
HRBENCH_NEEDS_JUDGE = {"the sign reads 27 only"}


def check_staging() -> list:
    """The three faults build_deepeyes_eval_data.py repairs must be repaired."""
    out = []
    hr = (EVAL_SRC / "eval_hrbench.py").read_text()
    jh = (EVAL_SRC / "judge_result_hrbench.py").read_text()
    out.append(("eval_hrbench.py imports pandas", "import pandas as pd" in hr))
    out.append(("eval_hrbench.py imports copy", "import copy" in hr))
    out.append(("judge_result_hrbench.py reads args.hrbench_path",
                "args.hrbench_path" in jh and "args.vstar_bench_path" not in jh))
    return out


def run_leg(tmp: Path, name: str, script: str, patharg: str,
            cases: list, needs_judge: set, write_rows, split_names: list,
            use_judge: bool) -> list:
    """Write a synthetic result jsonl, run their judge on it, read back final_acc.json."""
    tag = f"test-{name}"
    save_path = tmp / name
    (save_path / tag).mkdir(parents=True, exist_ok=True)

    rows = [c for c in cases if use_judge or c[0] not in needs_judge]
    # Their grader iterates a fixed list of splits and opens a file per split, so every
    # split has to exist even when the interesting rows are all in the first one.
    for i, split in enumerate(split_names):
        path = save_path / tag / f"result_{split}_{tag}.jsonl"
        chunk = rows if i == 0 else rows[:1]
        with path.open("w") as f:
            for r in chunk:
                f.write(json.dumps(write_rows(r[0])) + "\n")

    cmd = [PYTHON, str(EVAL_SRC / script),
           "--model_name", tag,
           "--api_key", judge_key() or "EMPTY",
           "--api_url", JUDGE_URL,
           "--eval_model_name", JUDGE_MODEL,
           patharg, str(tmp),
           "--save_path", str(save_path),
           "--num_workers", "4"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    results = []
    final = save_path / tag / "final_acc.json"
    if proc.returncode != 0 or not final.exists():
        results.append((f"{name}: judge script ran", False,
                        (proc.stderr or proc.stdout).strip().splitlines()[-3:]))
        return results
    results.append((f"{name}: judge script ran", True, ""))

    # Grade the first split, where every case lives, against the hand-computed table.
    graded = {}
    acc_file = save_path / tag / f"result_{split_names[0]}_{tag}_acc.jsonl"
    for line in acc_file.read_text().splitlines():
        d = json.loads(line)
        graded[d["pred_ans"]] = d["acc"]
    for pred, want, why in cases:
        if pred not in graded:
            continue
        got = graded[pred]
        results.append((f"{name}: {pred!r:28s} -> {got}  ({why})", got == want,
                        "" if got == want else f"expected {want}"))
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", action="store_true",
                    help="also run the cases that fall through to the hosted judge")
    args = ap.parse_args()

    if not EVAL_SRC.is_dir():
        print(f"SKIP: {EVAL_SRC} not staged. Run build_deepeyes_eval_data.py first.")
        return 77

    use_judge = args.judge and bool(judge_key())
    if args.judge and not use_judge:
        print("WARNING: --judge asked for but no NVIDIA_API_KEY/OPENAI_API_KEY; "
              "running rule-based cases only.")

    checks = [(n, ok, "") for n, ok in check_staging()]

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        checks += run_leg(
            tmp, "vstar", "judge_result.py", "--vstar_bench_path",
            VSTAR_CASES, VSTAR_NEEDS_JUDGE,
            lambda p: {"image": "x.jpg", "question": "What colour is the hydrant?",
                       "answer": "a red fire hydrant", "pred_ans": p,
                       "pred_output": _msgs(p), "status": "success"},
            ["direct_attributes", "relative_position"], use_judge)
        checks += run_leg(
            tmp, "hrbench", "judge_result_hrbench.py", "--hrbench_path",
            HRBENCH_CASES, HRBENCH_NEEDS_JUDGE,
            lambda p: {"question": "What number is above the entrance?",
                       "answer": "B", "answer_str": "37B", "pred_ans": p,
                       "pred_output": _msgs(p), "category": "single", "status": "success"},
            ["hr_bench_4k", "hr_bench_8k"], use_judge)

    print("=" * 74)
    bad = 0
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
        bad += not ok
    print("=" * 74)
    print(f"judge leg: {'LIVE against ' + JUDGE_MODEL + ' @ ' + JUDGE_URL if use_judge else 'rule-based only'}")
    print(f"{len(checks) - bad}/{len(checks)} passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
