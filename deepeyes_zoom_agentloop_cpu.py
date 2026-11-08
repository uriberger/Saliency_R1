#!/usr/bin/env python
"""The new zoom tool inside verl's real ToolAgentLoop, on CPU, via P2's harness.

    bash launch_deepeyes_zoom_agentloop_cpu_job.sh --name deepeyes-p3-cpu

WHY THIS IS A WRAPPER AND NOT A SECOND HARNESS

`deepeyes_agentloop_cpu.py` (P2, branch probe/deepeyes-verl-smoke) already drives
`ToolAgentLoop` + `ImageZoomInTool` + `AgentLoopWorkerBase._postprocess` for real, with
vLLM replaced by a scripted two-turn generation. Everything this prompt changed is on
the CPU side of that line, so the right move is to run P2's driver with two
substitutions rather than to write a parallel copy that drifts from it.

The two substitutions, and why each is needed:

  1  THE TOOL CONFIG. P2's `build_tool_config` predates `bbox_frame`, which now has no
     default -- so P2's harness run unchanged fails at tool construction with the
     explanatory error. That is the feature working, not a breakage, but it means this
     wrapper has to supply a frame. It picks one explicitly from --frame; there is no
     "whatever".

  2  THE DATA. `build_deepeyes_smoke_data.py` writes
     `create_kwargs = {"image": <data URI>}` and nothing else, which the new
     crop-source check refuses: the tool cannot tell an original from an
     already-resized copy by looking at it, so the builder has to say. This rebuilds
     the parquet's `tools_kwargs` through
     `deepeyes/zoom_tool_config.py::build_tools_kwargs`, which is what P6's real
     conversion must do too.

Both substitutions are monkeypatches applied to P2's module, deliberately: nothing on
P2's branch is edited, and when P2 and P3 are both on main the two fixes belong in
P2's files and this wrapper can go away.

WHAT IT ESTABLISHES
  does      the tool still satisfies ToolAgentLoop's contract after the rewrite: the
            crop reaches `multi_modal_data`, the observation is masked, the grid
            survives -- P2's A1/A2/A3, re-run on the new tool.
  does      the raw-vs-converted log is produced from inside a real rollout.
  does NOT  which coordinate frame Qwen3-VL actually emits. The tool call here is
            SCRIPTED. That is P4, and it needs a GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent

#: Where P2's driver lives. It is on an unmerged branch, so look in the worktree too.
#: The central tree is found through git's common dir rather than hardcoded, because
#: this file runs from a worktree whose own path is not the central one.
P2_CANDIDATES = (
    "",  # same directory, i.e. after P2 merges to main
    ".worktrees/probe-deepeyes-verl-smoke",
)


def find_p2_dir(explicit: str = "") -> Path:
    import subprocess

    if explicit:
        p = Path(explicit).resolve()
        if (p / "deepeyes_agentloop_cpu.py").exists():
            return p
        raise SystemExit(f"--p2-dir {p} has no deepeyes_agentloop_cpu.py")

    roots = [REPO]
    try:
        common = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        if common.endswith("/.git"):
            roots.append(Path(common[:-5]))
    except Exception:
        pass

    for root in roots:
        for rel in P2_CANDIDATES:
            p = (root / rel) if rel else root
            if (p / "deepeyes_agentloop_cpu.py").exists():
                return p.resolve()
    raise SystemExit(
        "cannot find deepeyes_agentloop_cpu.py. It is P2's, on branch "
        "probe/deepeyes-verl-smoke; check out its worktree or pass --p2-dir."
    )


def rebuild_tools_kwargs(src_parquet: Path, dst_parquet: Path) -> int:
    """Re-derive `extra_info.tools_kwargs` from the row's RAW image.

    This is the shape P6's conversion has to produce. Note what it does NOT do: it
    never touches `images`, and it never reuses a processed copy -- `original_size`
    and `view_size` both come out of the untouched parquet bytes.
    """
    import datasets

    from deepeyes.zoom_tool_config import build_tools_kwargs

    ds = datasets.load_dataset("parquet", data_files=str(src_parquet), split="train")

    def fn(example):
        extra = dict(example["extra_info"])
        extra["tools_kwargs"] = build_tools_kwargs(example["images"][0])
        extra["need_tools_kwargs"] = True
        out = dict(example)
        out["extra_info"] = extra
        return out

    ds = ds.map(fn, remove_columns=ds.column_names)
    dst_parquet.parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(str(dst_parquet))
    sample = ds[0]["extra_info"]["tools_kwargs"]["image_zoom_in_tool"]["create_kwargs"]
    print(
        f"rebuilt tools_kwargs for {len(ds)} rows -> {dst_parquet}\n"
        f"  create_kwargs keys={sorted(sample)} original_size={sample['original_size']} "
        f"view_size={sample['view_size']}",
        flush=True,
    )
    return len(ds)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True, choices=["rel1000", "abs_original", "abs_smartresize"],
                    help="no default anywhere in this arm; P4 measures which one is right")
    ap.add_argument("--data", required=True, help="P2's zoom_smoke.parquet")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--p2-dir", default="")
    args = ap.parse_args()

    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    zoom_log_dir = out / "zoom_log"
    zoom_log_dir.mkdir(parents=True, exist_ok=True)

    p2 = find_p2_dir(args.p2_dir)
    print(f"P2 harness: {p2}", flush=True)
    # REPO last-inserted, so FIRST on the path: P2's worktree is a checkout of this same
    # repo and also has a `deepeyes/` directory. Namespace packages would aggregate the
    # two, which works but means the resolution order of a shared module name depends on
    # which branch is checked out over there. Ours wins, explicitly.
    sys.path.insert(0, str(p2))
    sys.path.insert(0, str(REPO))

    data = out / "zoom_smoke_with_original.parquet"
    rebuild_tools_kwargs(Path(args.data).resolve(), data)

    import deepeyes_agentloop_cpu as p2mod

    from deepeyes import zoom_tool_config as ztc

    # [1] the tool config. Patched on P2's MODULE, not on deepeyes_tool_smoke: P2 did
    #     `from deepeyes_tool_smoke import build_tool_config`, so the name it calls is
    #     its own and patching the source module would do nothing.
    def build(path):
        return ztc.build_tool_config(
            path, bbox_frame=args.frame, log_dir=str(zoom_log_dir), log_first_n_calls=32, log_crops=4
        )

    p2mod.build_tool_config = build

    # [2] the scripted tool call, restated in the frame under test. P2's literal box
    #     [40, 30, 300, 240] is in no particular frame -- it predates the question --
    #     and under rel1000 it would address the top-left eighth of the figure, which
    #     is a fine crop and a bad test.
    boxes = {
        "rel1000": [200, 150, 800, 700],
        "abs_original": [60, 40, 420, 330],
        "abs_smartresize": [60, 40, 420, 330],
    }
    p2mod.TOOL_CALL_TURN = (
        "I need a closer look at the labelled region.\n"
        '<tool_call>\n{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": '
        f'{json.dumps(boxes[args.frame])}, "label": "angle labels"}}}}\n</tool_call>'
    )
    print(f"scripted tool call ({args.frame}): {boxes[args.frame]}", flush=True)

    sys.argv = [
        "deepeyes_agentloop_cpu.py",
        "--model", args.model,
        "--data", str(data),
        "--out-dir", str(out),
        "--rows", str(args.rows),
    ]
    rc = p2mod.main()

    # ------------------------------------------------------------- our own check
    records = []
    for p in sorted(zoom_log_dir.glob("zoom_calls.*.jsonl")):
        records += [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]
    ok = bool(records)
    ok = ok and all(r.get("status") == "ok" for r in records)
    ok = ok and all(r.get("frame") == args.frame for r in records)
    if args.frame != "abs_original":
        # abs_original is the identity map; for the other two a converted box equal to
        # the raw one would mean the conversion silently did not happen.
        ok = ok and all(r.get("bbox_converted") != r.get("bbox_raw") for r in records)
    print()
    print("=" * 74)
    print(f"{'PASS' if ok else 'FAIL'}  A4_zoom_tool: {len(records)} tool calls logged under frame "
          f"{args.frame}, all status=ok")
    for r in records[:4]:
        print("    " + json.dumps({k: r.get(k) for k in
                                   ("bbox_raw", "bbox_converted", "bbox_final", "original_size",
                                    "view_size", "crop_size", "downscale")}))
    print("=" * 74)
    (out / "zoom_tool_report.json").write_text(
        json.dumps({"frame": args.frame, "n_calls": len(records), "pass": ok, "calls": records}, indent=2)
    )
    print("NOTE: the tool call was SCRIPTED. This proves the tool's contract with")
    print("      ToolAgentLoop, not which frame Qwen3-VL emits -- that is P4.")
    return 0 if (rc == 0 and ok) else 1


if __name__ == "__main__":
    os.environ.setdefault("HF_HOME", "/home/uberger/scratch/cache/hf_cache")
    sys.exit(main())
