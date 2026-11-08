#!/usr/bin/env python
"""Reduce P2's logs to the handful of numbers the report is actually about.

    python deepeyes_smoke_report.py --stage grpo --out-dir outputs/deepeyes/...
    python deepeyes_smoke_report.py --stage all  --out-dir outputs/deepeyes/...

Stage `grpo` reads verl's console logger, whose per-step line is built by
`verl/utils/logger/aggregate_logger.py::concat_dict_to_str` as

    step:1 - timing_s/step:123.4 - timing_s/gen:88.2 - critic/score/mean:0.12 - ...

so the parse is a split on " - " and then on ":". The number this exists to produce is
`timing_s/step`, and it is wanted for its own sake: baseline-port-estimates.md section 9
states both "3-7 A100-hours per step" and "~400-1,200 A100-hours for one 80-step run",
and 7 x 80 = 560, so the two disagree by about 2x.

THE MEASURED NUMBER HERE IS NOT THAT NUMBER, and the report must say so. This is a
16 x 8 single-turn batch; the real run is 256 x 16 multi-turn with up to five sequential
generate rounds. Scaling by the sequence count is a floor, not an estimate: multi-turn
rollouts are latency-bound rather than FLOP-bound. Step 1 is also excluded from the
mean -- it carries torch.compile, the first CUDA graph capture and vLLM's first wake-up.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

STEP_RE = re.compile(r"^step:(\d+) - (.*)$")


def parse_grpo_log(path: Path) -> dict:
    steps: list[dict] = []
    if not path.is_file():
        return {"error": f"missing {path}"}
    for line in path.read_text(errors="replace").splitlines():
        m = STEP_RE.match(line.strip())
        if not m:
            continue
        rec: dict = {"step": int(m.group(1))}
        for field in m.group(2).split(" - "):
            if ":" not in field:
                continue
            k, _, v = field.partition(":")
            try:
                rec[k] = float(v)
            except ValueError:
                pass
        steps.append(rec)
    return {"steps": steps}


def summarise_grpo(out_dir: Path) -> dict:
    d = parse_grpo_log(out_dir / "grpo.log")
    steps = d.get("steps", [])
    out: dict = {"n_steps_logged": len(steps), "steps": steps}

    wall = out_dir / "grpo.wall"
    if wall.is_file():
        try:
            out["wall_clock_s"] = int(wall.read_text().strip().split("=")[1])
        except Exception:
            pass

    per_step = [s["timing_s/step"] for s in steps if "timing_s/step" in s]
    if per_step:
        out["step_time_s"] = {
            "all": [round(x, 1) for x in per_step],
            "first": round(per_step[0], 1),
            # Step 1 pays torch.compile, the first CUDA-graph capture and vLLM's first
            # wake-up; it is not the steady-state cost and averaging it in flatters
            # nothing and misleads everyone.
            "steady_mean": round(sum(per_step[1:]) / max(1, len(per_step[1:])), 1) if len(per_step) > 1 else None,
        }
        gen = [s.get("timing_s/gen") for s in steps if s.get("timing_s/gen") is not None]
        if gen:
            out["gen_time_s"] = {"all": [round(x, 1) for x in gen]}
        out["fraction_in_generation"] = (
            round(sum(gen[1:]) / sum(per_step[1:]), 3) if len(per_step) > 1 and len(gen) > 1 else None
        )

    for key in ("critic/score/mean", "critic/rewards/mean", "actor/pg_loss", "response_length/mean"):
        vals = [s[key] for s in steps if key in s]
        if vals:
            out.setdefault("series", {})[key] = [round(v, 4) for v in vals]

    # Non-zero reward variance is the cheapest proof that the reward actually scored
    # something rather than returning 0.0 for every rollout, which is what a broken
    # data_source route looks like from the outside.
    scores = out.get("series", {}).get("critic/score/mean", [])
    out["reward_is_live"] = bool(scores) and any(s != 0.0 for s in scores)
    return out


def collect_markers(out_dir: Path) -> dict:
    marks: dict[str, int] = {}
    pids = 0
    for p in sorted((out_dir / "mrope").glob("mrope.*.json")):
        pids += 1
        try:
            for k, v in json.loads(p.read_text()).get("counts", {}).items():
                marks[k] = marks.get(k, 0) + int(v)
        except Exception:
            pass
    return {"processes": pids, "counts": marks}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["grpo", "tool", "all"])
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir).resolve()

    report: dict = {"out_dir": str(out)}
    if args.stage in ("grpo", "all"):
        report["grpo"] = summarise_grpo(out)
    if args.stage in ("tool", "all"):
        p = out / "tool_smoke_report.json"
        report["tool"] = json.loads(p.read_text()) if p.is_file() else {"error": f"missing {p}"}
    report["mrope_markers"] = collect_markers(out)

    (out / f"summary_{args.stage}.json").write_text(json.dumps(report, indent=2, default=str))

    print()
    print("=" * 74)
    g = report.get("grpo")
    if g and "step_time_s" in g:
        st = g["step_time_s"]
        print(f"STAGE A  plain verl GRPO on geo3k: {g['n_steps_logged']} steps logged")
        print(f"         step time  : {st['all']} s   (first {st['first']}, steady mean {st['steady_mean']})")
        if g.get("gen_time_s"):
            print(f"         generation : {g['gen_time_s']['all']} s  "
                  f"({g.get('fraction_in_generation')} of the steady step)")
        print(f"         wall clock : {g.get('wall_clock_s')} s including model load and vLLM init")
        print(f"         reward live: {g.get('reward_is_live')}   scores "
              f"{g.get('series', {}).get('critic/score/mean')}")
    elif g:
        print(f"STAGE A  no step lines parsed: {g}")
    m = report["mrope_markers"]
    print(f"MROPE    {m['processes']} processes reported: {m['counts']}")
    t = report.get("tool")
    if t and "A1_position_ids" in t:
        for a in ("A1_position_ids", "A2_response_mask", "A3_image_grid_thw"):
            print(f"{'PASS' if t[a]['pass'] else 'FAIL'}  {a}: {t[a]['detail']}")
    print("=" * 74)
    print(f"summary -> {out / f'summary_{args.stage}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
