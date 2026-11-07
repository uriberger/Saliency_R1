#!/usr/bin/env python
"""Build the two parquet files the DeepEyes verl SMOKE run reads.

This is P2's data, and P2 is a STACK PROOF, not an experiment: the point is to run
plain verl GRPO on Qwen3-VL-8B with nothing of ours in the loop, so the dataset is
upstream verl's own `geo3k` conversion (hiyouga/geometry3k) reproduced verbatim from
`examples/data_preprocess/geo3k.py`, cut down to a handful of rows.

Two outputs, because the job has two stages:

  geo3k_train.parquet / geo3k_val.parquet
      Stage A. `data_source="hiyouga/geometry3k"`, which routes to verl's BUILT-IN
      `verl/utils/reward_score/geo3k.py`. No reward of ours is reachable from here --
      that is the whole point. Keep the data_source string exactly as upstream writes
      it or `default_compute_score` raises NotImplementedError.

  zoom_smoke.parquet
      Stage B. The same rows, plus the `extra_info.tools_kwargs` block that
      `ImageZoomInTool.create()` needs:

          extra_info.tools_kwargs.image_zoom_in_tool.create_kwargs.image = <PIL image>

      and `agent_name="tool_agent"` so verl routes the row through
      `verl/experimental/agent_loop/tool_agent_loop.py` rather than the single-turn
      loop. Without BOTH of those the tool is configured and never called, and the
      run looks healthy while proving nothing.

IMAGES ARE NOT RESIZED. geo3k ships small diagrams already, but P6 will carry the
real corpus and `build_grpo_sets.py:109` caps our own images at 512 px -- the cap
that must never reach a zoom corpus. Nothing here introduces one, deliberately.

    python build_deepeyes_smoke_data.py --out-dir outputs/deepeyes/probe-verl-smoke/data
    python build_deepeyes_smoke_data.py --n-train 64 --n-val 16 --n-zoom 16
"""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path

# Upstream verl's instruction block, copied verbatim from
# examples/data_preprocess/geo3k.py so the prompt is theirs and not ours.
INSTRUCTION_FOLLOWING = (
    r"You FIRST think about the reasoning process as an internal monologue and then "
    r"provide the final answer. The reasoning process MUST BE enclosed within <think> "
    r"</think> tags. The final answer MUST BE put in \boxed{}."
)

DATA_SOURCE = "hiyouga/geometry3k"

# Stage B's prompt. It names the tool explicitly because the smoke run has to PROVE a
# tool turn happens -- a prompt that merely permits zooming would leave "no tool call"
# and "tool call broken" indistinguishable, and those are the two outcomes this stage
# exists to tell apart.
ZOOM_INSTRUCTION = (
    "Look at the figure. You MUST first call the `image_zoom_in_tool` exactly once on "
    "the region of the figure that carries the labels, so you can read them clearly. "
    "After you receive the zoomed image, answer the question. "
    "The final answer MUST BE put in \\boxed{}."
)


def _as_png_bytes(img) -> dict:
    """verl's RLHFDataset accepts {'bytes': ..., 'path': ...} or a PIL image.

    Parquet round-trips the dict form losslessly; a PIL object does not survive
    `to_parquet` without datasets' Image feature, which is what we get for free by
    leaving the column as the HF `Image` feature. This helper is only used for the
    tools_kwargs copy, which is a plain struct and has no Image feature on it.
    """
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return {"bytes": buf.getvalue(), "path": None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/deepeyes/probe-verl-smoke/data")
    ap.add_argument("--n-train", type=int, default=64)
    ap.add_argument("--n-val", type=int, default=16)
    ap.add_argument("--n-zoom", type=int, default=16)
    args = ap.parse_args()

    import datasets

    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    ds = datasets.load_dataset(DATA_SOURCE)
    train = ds["train"].select(range(min(args.n_train, len(ds["train"]))))
    val = ds["test"].select(range(min(args.n_val, len(ds["test"]))))

    def plain(split):
        def fn(example, idx):
            problem = example["problem"]
            return {
                "data_source": DATA_SOURCE,
                "prompt": [{"role": "user", "content": problem + " " + INSTRUCTION_FOLLOWING}],
                "images": example["images"],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": example["answer"]},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": example["answer"],
                    "question": problem,
                },
            }

        return fn

    cols = train.column_names
    tr = train.map(plain("train"), with_indices=True, remove_columns=cols)
    va = val.map(plain("test"), with_indices=True, remove_columns=val.column_names)
    tr.to_parquet(str(out / "geo3k_train.parquet"))
    va.to_parquet(str(out / "geo3k_val.parquet"))

    # ---- Stage B: the same rows with the tool wiring attached ------------------
    zoom_src = ds["train"].select(range(min(args.n_zoom, len(ds["train"]))))

    def zoomed(example, idx):
        problem = example["problem"]
        img = example["images"][0]
        return {
            "data_source": DATA_SOURCE,
            "prompt": [{"role": "user", "content": problem + " " + ZOOM_INSTRUCTION}],
            "images": example["images"],
            "ability": "math",
            # `agent_name` is what AgentLoopWorker dispatches on. "tool_agent" is the
            # registered name of ToolAgentLoop; omit it and verl silently uses
            # single_turn_agent and never calls a tool.
            "agent_name": "tool_agent",
            "reward_model": {"style": "rule", "ground_truth": example["answer"]},
            "extra_info": {
                "split": "train",
                "index": idx,
                "answer": example["answer"],
                "question": problem,
                "need_tools_kwargs": True,
                "tools_kwargs": {
                    "image_zoom_in_tool": {
                        # ImageZoomInTool.create() reads create_kwargs["image"] and
                        # pushes it through qwen_vl_utils.fetch_image, which accepts a
                        # PIL image, a path, a URL or a data: URI. The bytes dict is
                        # what survives parquet, so we hand it a data URI instead.
                        "create_kwargs": {"image": _data_uri(img)},
                    }
                },
            },
        }

    zs = zoom_src.map(zoomed, with_indices=True, remove_columns=zoom_src.column_names)
    zs.to_parquet(str(out / "zoom_smoke.parquet"))

    manifest = {
        "data_source": DATA_SOURCE,
        "n_train": len(tr),
        "n_val": len(va),
        "n_zoom": len(zs),
        "files": sorted(p.name for p in out.glob("*.parquet")),
        "note": "geo3k verbatim from verl examples/data_preprocess/geo3k.py; no image resize",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    print(f"wrote -> {out}")
    return 0


def _data_uri(img) -> str:
    import base64

    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


if __name__ == "__main__":
    os.environ.setdefault("HF_HOME", "/home/uberger/scratch/cache/hf_cache")
    raise SystemExit(main())
