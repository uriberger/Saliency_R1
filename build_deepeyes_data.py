#!/usr/bin/env python
"""Convert ChenShawn/DeepEyes-Datasets-47k into a parquet modern verl can train on.

    python build_deepeyes_data.py --stats-only          # measure, write nothing
    python build_deepeyes_data.py                       # the full 47k build
    python build_deepeyes_data.py --smoke 64            # a tiny set, same code path
    python build_deepeyes_data.py --verify-only         # re-check an existing build

Run it through `launch_deepeyes_data_job.sh`, which sends it to `cpu_short`.

THE CORPUS IS ALREADY IN verl's SCHEMA, SO THIS IS RE-ROUTING, NOT CONVERSION
----------------------------------------------------------------------------
Three released parquet files, 47,052 rows, images embedded as bytes:

    data_0.1.2_visual_toolbox_v2.parquet   22,362  data_source=vstar             tool ON
    data_v0.8_visual_toolbox_v2.parquet    13,659  data_source=chart             tool ON
    data_thinklite_reasoning_acc.parquet   11,031  data_source=thinklite_eureka  tool OFF

Every row already carries `data_source`, `prompt`, `images`, `ability`, `env_name`,
`reward_model.{ground_truth,style}` and `extra_info.{answer,index,question,split}`.
`prompt` is a two-turn chat -- a system turn holding DeepEyes' hand-written tool block
and a user turn whose content starts with the `<image>` placeholder that verl's
`RLHFDataset._build_messages` splits on. All of that passes through untouched.

What does NOT transfer is `env_name`. Their verl reads it via
`actor_rollout_ref.rollout.agent.tool_name_key=env_name` and, in
`parallel_env.py:474`, does `tool_name = non_tensor_batch.pop('env_name', '')` --
a non-empty string gets a tool, the empty string gets none. Modern verl deleted that
machinery. Its equivalent is the per-row `agent_name` column read in
`verl/experimental/agent_loop/agent_loop.py:348`, plus `extra_info.tools_kwargs`.
So the single job of this script is:

    env_name="visual_toolbox_v2"  ->  agent_name="tool_agent"       + tools_kwargs
    env_name=""                   ->  agent_name="single_turn_agent", tools_kwargs=None

`env_name` is kept in `extra_info` for provenance; nothing reads it.

TRAP 1 -- THE TOOL MUST BE OFF FOR ThinkLite, AND `agent_name` IS THE ONLY SWITCH
---------------------------------------------------------------------------------
23% of the corpus is math, and their config gives it no tool: `env_name` is the empty
string on all 11,031 rows (measured, not assumed -- see `stats.json`). The composite
reward differs too: `vstar`/`chart` route to `vl_agent.compute_score`
(0.8*acc + 0.2*format + 1.2*tool) and `thinklite_eureka` to
`vl_agent.compute_score_math` (1.2*acc + 0.4*format, no tool term).

In modern verl there is no `tool_name_key`. A row reaches `ToolAgentLoop` if and only
if its `agent_name` says so, and `rollout.agent.default_agent_loop` is
`single_turn_agent` anyway -- but the default is a config knob somebody can move, so
the math rows name `single_turn_agent` EXPLICITLY. That is the whole of trap 1: if the
math rows ever reach `ToolAgentLoop`, the policy can zoom on them, and P5's reward --
which routes on `data_source`, not on whether a tool ran -- scores the result with the
math composite. The run would look healthy and a quarter of the corpus would be wrong.

`--verify` asserts it: zero `thinklite_eureka` rows with `agent_name != single_turn_agent`,
zero with a non-null `tools_kwargs`.

TRAP 2 -- NATIVE RESOLUTION
---------------------------
`build_grpo_sets.py:109` caps our own corpora at `MAX_IMAGE_SIDE = 512`. Applying that
here would delete the detail the zoom tool exists to recover and quietly turn the whole
arm into a no-op. So the `images` column is copied BYTE FOR BYTE: no decode, no
re-encode, no resize. `--verify` hashes a sample of source and output image bytes and
asserts they are identical, which is a stronger statement than "we did not call
resize()".

Two resolution facts the launcher needs, and `stats.json` reports both:

  * the pixel-count distribution (median / p99) of the images as published, and
  * the IMAGE TOKEN count each row costs after Qwen3-VL's `smart_resize`, computed
    analytically on the patch-16 / merge-2 grid (32 px per token). That second number
    is what `data.max_prompt_length` has to hold, not the first.

WHERE THE ORIGINAL IMAGE GOES, AND WHY IT IS A FILE
---------------------------------------------------
`ImageZoomInTool.create()` (deepeyes_fork, :332) requires `create_kwargs["image"]` and
raises `ValueError` without it; `tool_agent_loop.py:444` passes exactly
`tools_kwargs[tool_name]["create_kwargs"]`. So every tool row must carry an image a
second time, because the one in the `images` column has by then been through
`process_image` -> `fetch_image` and is smart-resized.

Default `--image-mode file` writes those originals to `<out>/images/<source>/` and
points `create_kwargs.image` at a `file://` URI. The alternative, `--image-mode
datauri`, inlines a base64 copy and is what P2's smoke data did; it costs ~+4.3 GB of
parquet and makes every row carry its image twice through ray. The file mode keeps the
parquet at the published size and -- the actual reason -- leaves the pre-resize
original as an inspectable artifact on disk, which is what P3 has to assert it crops.

A WARNING FOR P3 THAT IS NOT FIXABLE FROM THE DATA SIDE
--------------------------------------------------------
`ImageZoomInTool.create()` does `img = fetch_image({"image": image})`, and
`qwen_vl_utils.fetch_image` ALWAYS calls `smart_resize` -- for a PIL object, a
`file://` path and a data URI alike. So the tool stores a resized image whatever we
hand it, and P3's requirement "crop the original, pre-processor-resize image" cannot be
met by choosing a different `create_kwargs` encoding. It needs a change inside
`create()`. The `file://` path at least makes the true original reachable.

Related, and also P3/P7's: `fetch_image`'s grid is `image_patch_size * 2`, and verl's
`data.image_patch_size` defaults to **14**. Qwen3-VL is patch-16, so the default
resizes onto a 28 px grid and caps at 12,845,056 px instead of 16,777,216.
`data.image_patch_size=16` belongs in the launcher.

TWO MORE THINGS THE LAUNCHER MUST SET, BOTH FOUND HERE
-------------------------------------------------------
  * `data.tool_config_path=null`. It defaults to
    `${oc.select:actor_rollout_ref.rollout.multi_turn.tool_config_path, null}`, so
    configuring the tool makes `RLHFDataset.__init__` build tool schemas and pass them
    as `tools=` to `apply_chat_template` (:192). DeepEyes' prompt ALREADY contains the
    tool block as a system turn, so the schema would be rendered twice. Leaving it set
    also constructs `ImageZoomInTool` -- and therefore a ray actor and an implicit
    `ray.init()` with no runtime_env -- inside dataset construction, which is the
    ordering trap P2 recorded.
  * `ToolAgentLoop._handle_pending_state` passes `tools=self.tool_schemas`
    unconditionally (:194). That one cannot be turned off from the data side; the
    duplicate-tool-block question is P7's and is recorded in the manifest.

A SHARD CAN BE VALID PARQUET AND STILL UNREADABLE BY verl
----------------------------------------------------------
The first full build wrote a 2.03 GB chart shard that `pq.read_table` opens happily
and `datasets.load_dataset("parquet", ...)` -- which is what
`RLHFDataset._read_files_and_tokenize` calls -- cannot open at all:

    pyarrow.lib.ArrowNotImplementedError:
    Nested data conversions not implemented for chunked array outputs

Over 2 GiB of binary in one scanned batch forces arrow to chunk the `images` column's
child array, and nested columns cannot be chunked. `ROW_GROUP_SIZE` below is the fix
and the comment there is the explanation. The reason `--verify` loads through
`datasets` and not through pyarrow is precisely this: pyarrow's opinion of the file is
not the one that decides whether training starts.

THE ONE FIELD THIS SCRIPT REWRITES: `extra_info.index`
-------------------------------------------------------
In the release, `extra_info.index` is the **constant string "0" on all 11,031
ThinkLite rows** (22,362 distinct on V*, 13,659 distinct on chart, 1 distinct on
ThinkLite). verl copies it to `row_dict["index"]`, and
`agent_loop.py:352` uses it for rollout-trace sampling: `np.unique(index)` over a
batch of math rows yields one value, so tracing sees a single "sample" and the
per-sample trace selection collapses on a quarter of the corpus.

Nothing load-bearing reads it -- grouping and advantages key on `uid`, not `index` --
so this is a logging defect, not a training one. It is still cheap to fix and
expensive to debug later, so `index` is rewritten to `"<corpus>:<row>"`
(`vstar:0`, `chart:9`, `thinklite:11030`), unique across the whole corpus and stable
under re-runs because the sources are streamed in file order. The published value is
kept verbatim as `extra_info.orig_index`, and that is what the verify pass and any
comparison against their numbers should use.

THE VALIDATION SPLIT IS OURS
-----------------------------
Their script validates on the training math file
(`data.val_files=[EUREKA_DATASET_TRAIN]`). This carves a stratified val set from the
same pool instead and excludes those rows from train, so the number means something.
`--val-size 0` reproduces their arrangement.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent

HF_REPO = "ChenShawn/DeepEyes-Datasets-47k"

#: One entry per released file. `data_source` and `env_name` are what we EXPECT; the
#: build asserts them row by row rather than trusting this table, because P5 routes its
#: reward on the real strings and a silent drift here would misroute a whole split.
SOURCES = {
    "vstar": dict(
        file="data_0.1.2_visual_toolbox_v2.parquet",
        data_source="vstar",
        env_name="visual_toolbox_v2",
        rows=22362,
        note="V* / visual search, zoom tool ON",
    ),
    "chart": dict(
        file="data_v0.8_visual_toolbox_v2.parquet",
        data_source="chart",
        env_name="visual_toolbox_v2",
        rows=13659,
        note="ArxivQA / chart, zoom tool ON",
    ),
    "thinklite": dict(
        file="data_thinklite_reasoning_acc.parquet",
        data_source="thinklite_eureka",
        env_name="",
        rows=11031,
        note="ThinkLite-VL math reasoning, zoom tool OFF",
    ),
}

TOOL_NAME = "image_zoom_in_tool"
TOOL_AGENT = "tool_agent"
PLAIN_AGENT = "single_turn_agent"

#: Qwen3-VL geometry: patch 16, spatial merge 2 -> one visual token per 32x32 px.
#: These are the numbers `qwen_vl_utils.fetch_image` uses once `data.image_patch_size`
#: is 16, and they are what the image-token estimate below is computed against.
PATCH_FACTOR = 32
IMAGE_MIN_TOKENS = 4
IMAGE_MAX_TOKENS = 16384


# ---------------------------------------------------------------------------
# Locating the download
# ---------------------------------------------------------------------------
def resolve_snapshot(explicit: str | None) -> Path:
    """The directory holding the three released parquet files.

    Prefers an explicit --hf-dir, else the local HF cache snapshot. Never downloads:
    the compute nodes may not have egress, and the login node already did it.
    """
    if explicit:
        p = Path(explicit).resolve()
        if not p.is_dir():
            raise SystemExit(f"--hf-dir {p} is not a directory")
        return p

    from huggingface_hub import snapshot_download

    p = Path(snapshot_download(HF_REPO, repo_type="dataset", local_files_only=True))
    return p.resolve()


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def smart_resize_hw(height: int, width: int) -> tuple[int, int]:
    """Qwen's smart_resize, on the Qwen3-VL grid, without importing qwen_vl_utils.

    Kept local on purpose: the stats pass runs over 47k rows on a CPU node and must not
    depend on which qwen_vl_utils happens to be installed, and the numbers it reports
    are quoted in a handoff. Mirrors `qwen_vl_utils.vision_process.smart_resize`.
    """
    import math

    factor = PATCH_FACTOR
    min_pixels = IMAGE_MIN_TOKENS * factor * factor
    max_pixels = IMAGE_MAX_TOKENS * factor * factor

    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def image_token_count(width: int, height: int) -> int:
    """Visual tokens Qwen3-VL spends on one image of this size, after smart_resize."""
    h_bar, w_bar = smart_resize_hw(height, width)
    return (h_bar // PATCH_FACTOR) * (w_bar // PATCH_FACTOR)


def percentile(sorted_vals, q: float):
    if not sorted_vals:
        return None
    i = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[i]


def summarize(vals, name: str) -> dict:
    s = sorted(vals)
    return {
        "name": name,
        "n": len(s),
        "min": percentile(s, 0.0),
        "p25": percentile(s, 0.25),
        "median": percentile(s, 0.50),
        "p75": percentile(s, 0.75),
        "p95": percentile(s, 0.95),
        "p99": percentile(s, 0.99),
        "max": percentile(s, 1.0),
        "mean": (sum(s) / len(s)) if s else None,
    }


# ---------------------------------------------------------------------------
# The output schema, written explicitly
# ---------------------------------------------------------------------------
def output_schema():
    """An EXPLICIT schema, for two reasons that both bite silently.

    1. `tools_kwargs` is null on every math row. Inferred from a shard that happens to
       hold only math rows, arrow types it null and the shards stop concatenating --
       the same failure `build_laser_data.py` documents for `sub_source`.
    2. `extra_info.index` is a STRING upstream ("0", "physics-8049"), not an int.
       Inference would make it an int on the two splits whose indices all parse and a
       string on the third.

    MapType was the obvious fit for `tools_kwargs` and is unusable: HF `datasets`
    raises "Arrow type map<...> does not have a datasets dtype equivalent" at load.
    A nullable struct round-trips to `None`, which is falsy, which is what verl's
    `extra_info.get("tools_kwargs", {})` wants.
    """
    import pyarrow as pa

    create_kwargs = pa.struct([("image", pa.string())])
    tools_kwargs = pa.struct([(TOOL_NAME, pa.struct([("create_kwargs", create_kwargs)]))])
    return pa.schema(
        [
            ("data_source", pa.string()),
            ("prompt", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
            ("images", pa.list_(pa.struct([("bytes", pa.binary()), ("path", pa.string())]))),
            ("ability", pa.string()),
            # Read by verl/experimental/agent_loop/agent_loop.py:348. THIS is the
            # tool on/off switch; see trap 1 in the module docstring.
            ("agent_name", pa.string()),
            ("reward_model", pa.struct([("style", pa.string()), ("ground_truth", pa.string())])),
            (
                "extra_info",
                pa.struct(
                    [
                        ("answer", pa.string()),
                        # REWRITTEN, and the only field here that is. See rewrite_index().
                        ("index", pa.string()),
                        ("orig_index", pa.string()),
                        ("question", pa.string()),
                        ("split", pa.string()),
                        ("corpus", pa.string()),
                        ("env_name", pa.string()),
                        ("need_tools_kwargs", pa.bool_()),
                        ("tools_kwargs", tools_kwargs),
                    ]
                ),
            ),
        ]
    )


# ---------------------------------------------------------------------------
# The conversion
# ---------------------------------------------------------------------------
IMAGE_EXT = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp", "GIF": "gif", "BMP": "bmp", "TIFF": "tif"}


def convert_source(
    name: str,
    spec: dict,
    src: Path,
    image_dir: Path | None,
    image_mode: str,
    max_rows: int,
    batch_size: int,
    stats_only: bool,
):
    """Stream one released parquet; yield output rows and collect the stats.

    Images are never decoded into pixels -- `Image.open` on a BytesIO reads the header,
    which is all `.size` and `.format` need, and the bytes object is passed through
    unmodified. That is the mechanical guarantee behind trap 2.
    """
    import pyarrow.parquet as pq
    from PIL import Image

    cols = ["data_source", "prompt", "images", "ability", "env_name", "reward_model", "extra_info"]
    want_tool = bool(spec["env_name"])

    stats = {
        "corpus": name,
        "file": spec["file"],
        "note": spec["note"],
        "rows": 0,
        "data_source": {},
        "env_name": {},
        "ability": {},
        "reward_style": {},
        "n_images": {},
        "image_format": {},
        "pixels": [],
        "width": [],
        "height": [],
        "image_tokens_per_row": [],
        "image_bytes": [],
        "prompt_roles": {},
        "prompt_chars": [],
        "rows_without_image_tag": 0,
        "rows_without_image": 0,
    }

    if image_dir is not None:
        image_dir.mkdir(parents=True, exist_ok=True)

    pf = pq.ParquetFile(src)
    out_rows = []
    i = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
        for row in batch.to_pylist():
            if max_rows and i >= max_rows:
                break
            ds, env = row["data_source"], row["env_name"]
            # Assert rather than trust SOURCES: P5 routes its reward on these strings.
            if ds != spec["data_source"]:
                raise SystemExit(f"{name}: row {i} has data_source={ds!r}, expected {spec['data_source']!r}")
            if env != spec["env_name"]:
                raise SystemExit(f"{name}: row {i} has env_name={env!r}, expected {spec['env_name']!r}")

            stats["data_source"][ds] = stats["data_source"].get(ds, 0) + 1
            stats["env_name"][env] = stats["env_name"].get(env, 0) + 1
            stats["ability"][row["ability"]] = stats["ability"].get(row["ability"], 0) + 1
            style = (row["reward_model"] or {}).get("style")
            stats["reward_style"][style] = stats["reward_style"].get(style, 0) + 1

            prompt = row["prompt"] or []
            roles = "+".join(m["role"] for m in prompt)
            stats["prompt_roles"][roles] = stats["prompt_roles"].get(roles, 0) + 1
            stats["prompt_chars"].append(sum(len(m["content"] or "") for m in prompt))
            if not any("<image>" in (m["content"] or "") for m in prompt):
                stats["rows_without_image_tag"] += 1

            images = row["images"] or []
            stats["n_images"][len(images)] = stats["n_images"].get(len(images), 0) + 1
            if not images:
                stats["rows_without_image"] += 1
                i += 1
                continue

            tokens = 0
            first_uri = None
            for k, im in enumerate(images):
                b = im["bytes"]
                with Image.open(io.BytesIO(b)) as pim:  # header only; no decode
                    w, h = pim.size
                    fmt = pim.format or "?"
                stats["image_format"][fmt] = stats["image_format"].get(fmt, 0) + 1
                stats["width"].append(w)
                stats["height"].append(h)
                stats["pixels"].append(w * h)
                stats["image_bytes"].append(len(b))
                tokens += image_token_count(w, h)
                if k == 0 and want_tool and not stats_only:
                    if image_mode == "file":
                        ext = IMAGE_EXT.get(fmt, "img")
                        p = image_dir / f"{i:06d}.{ext}"
                        if not p.exists():
                            p.write_bytes(b)  # verbatim: no decode, no re-encode
                        first_uri = "file://" + str(p)
                    elif image_mode == "datauri":
                        mime = {"JPEG": "jpeg", "PNG": "png", "WEBP": "webp"}.get(fmt, "png")
                        first_uri = f"data:image/{mime};base64," + base64.b64encode(b).decode()
                    elif image_mode == "none":
                        first_uri = None
                    else:
                        raise SystemExit(f"unknown --image-mode {image_mode}")
            stats["image_tokens_per_row"].append(tokens)

            if not stats_only:
                xi = row["extra_info"] or {}
                tk = None
                if want_tool and first_uri is not None:
                    tk = {TOOL_NAME: {"create_kwargs": {"image": first_uri}}}
                out_rows.append(
                    {
                        "data_source": ds,
                        "prompt": [{"role": m["role"], "content": m["content"]} for m in prompt],
                        "images": [{"bytes": im["bytes"], "path": im["path"]} for im in images],
                        "ability": row["ability"],
                        "agent_name": TOOL_AGENT if want_tool else PLAIN_AGENT,
                        "reward_model": {
                            "style": (row["reward_model"] or {}).get("style"),
                            "ground_truth": (row["reward_model"] or {}).get("ground_truth"),
                        },
                        "extra_info": {
                            "answer": xi.get("answer"),
                            "index": f"{name}:{i}",
                            "orig_index": xi.get("index"),
                            "question": xi.get("question"),
                            "split": xi.get("split"),
                            "corpus": name,
                            "env_name": env,
                            "need_tools_kwargs": bool(want_tool),
                            "tools_kwargs": tk,
                        },
                    }
                )
            i += 1
        if max_rows and i >= max_rows:
            break

    stats["rows"] = i
    return out_rows, stats


def finalize_stats(stats: dict) -> dict:
    """Replace the raw per-image lists with distributions. The lists are 47k long and
    nobody reads them; the percentiles are what the launcher sizes against."""
    out = dict(stats)
    out["pixels"] = summarize(stats["pixels"], "pixels_per_image")
    out["width"] = summarize(stats["width"], "width")
    out["height"] = summarize(stats["height"], "height")
    out["image_tokens_per_row"] = summarize(stats["image_tokens_per_row"], "image_tokens_per_row")
    out["image_bytes"] = summarize(stats["image_bytes"], "image_bytes")
    out["prompt_chars"] = summarize(stats["prompt_chars"], "prompt_chars")
    out["n_images"] = {str(k): v for k, v in sorted(stats["n_images"].items())}
    return out


#: Rows per parquet row group. SMALL ON PURPOSE, and this is not a tuning knob.
#:
#: verl reads the corpus with `datasets.load_dataset("parquet", data_files=...)`
#: (`rl_dataset.py:_read_files_and_tokenize`), which scans through pyarrow's dataset
#: API. When one scanned batch holds more than 2 GiB of binary, arrow has to chunk the
#: `images` column's child array, and nested columns cannot be chunked:
#:
#:     pyarrow.lib.ArrowNotImplementedError:
#:     Nested data conversions not implemented for chunked array outputs
#:
#: Batches do not span row groups, so bounding the row group bounds the batch. At
#: pyarrow's default (one row group for the whole table) the 2.03 GB chart shard was
#: unreadable -- it wrote cleanly, and `datasets` and therefore verl could not open it.
#: That is why `--verify` loads every shard through `datasets` rather than pyarrow: it
#: is the only loader whose opinion matters.
#:
#: 512 rows is ~77 MB on the heaviest corpus (chart averages 150 KB a row), two orders
#: of magnitude under the limit, and costs only a little footer metadata.
ROW_GROUP_SIZE = 512


def write_parquet(rows, path: Path, schema, row_group_size: int = ROW_GROUP_SIZE):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(rows, schema=schema),
        path,
        compression="zstd",
        row_group_size=row_group_size,
    )
    return path


# ---------------------------------------------------------------------------
# Verification: everything that is silent when wrong
# ---------------------------------------------------------------------------
def verify(out_dir: Path, hf_dir: Path, model: str, n_rows: int, strict_template: bool) -> dict:
    """Reload what was written and assert the things that fail quietly.

    1. ThinkLite rows carry `single_turn_agent` and a null `tools_kwargs` -- trap 1.
    2. Tool rows carry `tool_agent` and a reachable `create_kwargs.image` -- without it
       `ImageZoomInTool.create()` raises and every tool call in the run dies.
    3. Output image bytes are byte-identical to the published ones -- trap 2, stated as
       a hash rather than as "we did not resize".
    4. The prompt renders through Qwen3-VL's chat template, `<image>` becomes
       `<|vision_start|><|image_pad|><|vision_end|>`, and the number of `<|image_pad|>`
       tokens the processor emits equals the product of the grid it returns divided by
       merge_size**2. Qwen3-VL is supposed to share Qwen2.5-VL's vision token ids; this
       confirms it on this corpus instead of assuming it.
    """
    import datasets
    import pyarrow.parquet as pq

    res = {"checks": [], "ok": True}

    def check(name, ok, detail=""):
        res["checks"].append({"check": name, "ok": bool(ok), "detail": detail})
        if not ok:
            res["ok"] = False
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)

    files = sorted(out_dir.glob("*.parquet"))
    if not files:
        raise SystemExit(f"no parquet under {out_dir}")
    # Loaded the way verl loads it -- `datasets.load_dataset("parquet", ...)`, not
    # `pq.read_table`. The two disagree: a shard with an oversized row group writes and
    # reads back fine under pyarrow and raises ArrowNotImplementedError under datasets.
    # See ROW_GROUP_SIZE.
    parts = []
    for f in files:
        try:
            parts.append(datasets.load_dataset("parquet", data_files=str(f))["train"])
        except Exception as e:  # noqa: BLE001
            check(f"datasets can load {f.name}", False, f"{type(e).__name__}: {e}")
            return res
    ds = datasets.concatenate_datasets(parts)
    rg = {f.name: pq.ParquetFile(f).metadata.num_row_groups for f in files}
    check("all shards load under datasets and concatenate", True,
          f"{len(files)} files, {len(ds)} rows, row groups {rg}")

    # ---- 1, 2, 3: ONE pass over the rows ----------------------------------------
    # One pass on purpose: every `for r in ds` materializes 47k rows of image bytes out
    # of arrow, so three loops cost three times what one does.
    bad_math, bad_tool, missing_img, missing_files = 0, 0, 0, 0
    by_src, out_hashes, idx, orig = {}, {}, [], []
    checked_files = 0
    for r in ds:
        src = r["data_source"]
        by_src[src] = by_src.get(src, 0) + 1
        xi = r["extra_info"] or {}
        tk = xi.get("tools_kwargs")
        idx.append(xi.get("index"))
        orig.append(xi.get("orig_index"))
        if src == "thinklite_eureka":
            if r["agent_name"] != PLAIN_AGENT or tk is not None:
                bad_math += 1
        else:
            if r["agent_name"] != TOOL_AGENT:
                bad_tool += 1
            img = ((tk or {}).get(TOOL_NAME) or {}).get("create_kwargs", {}).get("image") or ""
            if not img:
                missing_img += 1
            elif img.startswith("file://") and checked_files < n_rows:
                checked_files += 1
                if not Path(img[7:]).exists():
                    missing_files += 1
        out_hashes[xi.get("index")] = [
            hashlib.sha256(im["bytes"]).hexdigest() for im in (r["images"] or [])
        ]
    check("ThinkLite rows have the tool disabled", bad_math == 0, f"{bad_math} violations")
    check("toolbox rows route to tool_agent", bad_tool == 0, f"{bad_tool} violations")
    check("toolbox rows carry create_kwargs.image", missing_img == 0, f"{missing_img} missing")
    check("create_kwargs file:// targets exist", missing_files == 0,
          f"{missing_files} missing of {checked_files} sampled")
    res["rows_by_data_source"] = by_src

    # ---- 3: byte identity against the published parquet --------------------------
    # Keyed on the REWRITTEN `extra_info.index` ("<corpus>:<row>"), which is the
    # source's own row order and is unique. The published `orig_index` cannot be the
    # key: it is "0" on all 11,031 ThinkLite rows, and keying on it silently compares
    # each output row against whichever source row landed in the dict last -- which is
    # exactly the false failure this check produced before the rewrite.
    compared, mismatched, unseen = 0, 0, 0
    for name, spec in SOURCES.items():
        p = hf_dir / spec["file"]
        if not p.exists():
            continue
        pf = pq.ParquetFile(p)
        b = next(pf.iter_batches(batch_size=n_rows, columns=["images"]))
        for i, row in enumerate(b.to_pylist()):
            key = f"{name}:{i}"
            if key not in out_hashes:  # trimmed away by --smoke/--max-rows
                unseen += 1
                continue
            want = [hashlib.sha256(im["bytes"]).hexdigest() for im in (row["images"] or [])]
            compared += 1
            if want != out_hashes[key]:
                mismatched += 1
    check("image bytes identical to the published ones", mismatched == 0 and compared > 0,
          f"{compared} rows compared, {mismatched} mismatched, {unseen} not in this build")

    # ---- 3b: the index rewrite ---------------------------------------------------
    check("extra_info.index is unique", len(set(idx)) == len(idx),
          f"{len(set(idx))} distinct of {len(idx)}")
    check("extra_info.orig_index preserved", all(o is not None for o in orig),
          f"{sum(o is None for o in orig)} missing")

    # ---- 4: the chat template and the vision tokens -----------------------------
    tmpl = {"ran": False}
    try:
        from transformers import AutoProcessor

        proc = AutoProcessor.from_pretrained(model, trust_remote_code=True)
        tok = proc.tokenizer
        ids = {
            "vision_start": tok.convert_tokens_to_ids("<|vision_start|>"),
            "vision_end": tok.convert_tokens_to_ids("<|vision_end|>"),
            "image_pad": tok.convert_tokens_to_ids("<|image_pad|>"),
        }
        res["vision_token_ids"] = ids
        check("vision token ids are Qwen2.5-VL's (151652/151653/151655)",
              ids == {"vision_start": 151652, "vision_end": 151653, "image_pad": 151655}, json.dumps(ids))

        merge = getattr(proc.image_processor, "merge_size", 2)
        res["image_processor"] = {
            "patch_size": getattr(proc.image_processor, "patch_size", None),
            "merge_size": merge,
            "class": type(proc.image_processor).__name__,
        }

        from PIL import Image

        n_bad_tag, n_bad_count, checked = 0, 0, 0
        for r in ds.select(range(min(n_rows, len(ds)))):
            msgs = []
            for m in r["prompt"]:
                content = m["content"] or ""
                if "<image>" in content:
                    parts, cl = content.split("<image>"), []
                    for j, seg in enumerate(parts):
                        if j:
                            cl.append({"type": "image"})
                        if seg:
                            cl.append({"type": "text", "text": seg})
                    msgs.append({"role": m["role"], "content": cl})
                else:
                    msgs.append({"role": m["role"], "content": content})
            raw = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            if "<|vision_start|><|image_pad|><|vision_end|>" not in raw:
                n_bad_tag += 1
                continue
            pil = [Image.open(io.BytesIO(im["bytes"])).convert("RGB") for im in r["images"]]
            mi = proc(text=[raw], images=pil, return_tensors="pt")
            grid = mi["image_grid_thw"]
            want = int((grid.prod(dim=-1) // (merge * merge)).sum())
            got = int((mi["input_ids"][0] == ids["image_pad"]).sum())
            checked += 1
            if want != got:
                n_bad_count += 1
        check("<image> renders as <|vision_start|><|image_pad|><|vision_end|>", n_bad_tag == 0,
              f"{n_bad_tag} of {min(n_rows, len(ds))} rows failed")
        check("image_pad count == image_grid_thw / merge^2", n_bad_count == 0,
              f"{checked} rows checked, {n_bad_count} mismatched")
        tmpl["ran"] = True
    except Exception as e:  # noqa: BLE001
        msg = f"{type(e).__name__}: {e}"
        if strict_template:
            check("chat-template render", False, msg)
        else:
            print(f"  [SKIP] chat-template render -- {msg}", flush=True)
            res["checks"].append({"check": "chat-template render", "ok": None, "detail": msg})
    res["chat_template_checked"] = tmpl["ran"]
    return res


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--hf-dir", default=None,
                    help="directory holding the three released parquet files; "
                         "defaults to the local HF cache snapshot")
    ap.add_argument("--out-dir", default=str(REPO / "cold_data" / "deepeyes" / "verl"))
    ap.add_argument("--image-mode", default="file", choices=["file", "datauri", "none"],
                    help="how the tool's own copy of the original image is carried")
    ap.add_argument("--val-size", type=int, default=384,
                    help="rows held out, stratified by corpus. 0 reproduces their "
                         "arrangement of validating on a training file")
    ap.add_argument("--smoke", type=int, default=0,
                    help="rows per corpus. A MECHANICAL check, not a training corpus")
    ap.add_argument("--max-rows", type=int, default=0,
                    help="cap rows per corpus. A DEVIATION from DeepEyes, reported as one")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--stats-only", action="store_true", help="measure, write no parquet")
    ap.add_argument("--verify-only", action="store_true", help="re-check an existing build")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--verify-rows", type=int, default=16)
    ap.add_argument("--strict-template", action="store_true",
                    help="fail if the chat-template check cannot run (no model, no cache)")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--seed", type=int, default=20261107)
    args = ap.parse_args()

    hf_dir = resolve_snapshot(args.hf_dir)
    out_dir = Path(args.out_dir).resolve()
    print(f"HF snapshot : {hf_dir}")
    print(f"out dir     : {out_dir}\n", flush=True)

    if args.verify_only:
        r = verify(out_dir, hf_dir, args.model, args.verify_rows, args.strict_template)
        (out_dir / "verify.json").write_text(json.dumps(r, indent=2))
        return 0 if r["ok"] else 1

    per_corpus = args.smoke or args.max_rows
    tag = f"smoke{args.smoke}" if args.smoke else ("47k" if not args.max_rows else f"cap{args.max_rows}")
    if args.max_rows and not args.smoke:
        print(f"DEVIATION: --max-rows {args.max_rows} per corpus. DeepEyes uses all 47,052 rows.\n")

    all_stats, all_rows = {}, {}
    combined = {"pixels": [], "width": [], "height": [], "image_tokens_per_row": [],
                "image_bytes": [], "prompt_chars": []}
    for name, spec in SOURCES.items():
        src = hf_dir / spec["file"]
        if not src.exists():
            raise SystemExit(f"missing {src} -- download {HF_REPO} first")
        image_dir = (out_dir / "images" / name) if args.image_mode == "file" else None
        rows, st = convert_source(
            name, spec, src, image_dir, args.image_mode, per_corpus, args.batch_size, args.stats_only
        )
        all_rows[name] = rows
        for k in combined:
            combined[k] += st[k]
        all_stats[name] = finalize_stats(st)
        s = all_stats[name]
        print(
            f"[{name:<10}] {st['rows']:>6} rows  {spec['note']}\n"
            f"             data_source={list(st['data_source'])} env_name={list(st['env_name'])}\n"
            f"             pixels median={s['pixels']['median']:,} p99={s['pixels']['p99']:,}"
            f"  image tokens/row median={s['image_tokens_per_row']['median']:,}"
            f" p99={s['image_tokens_per_row']['p99']:,}",
            flush=True,
        )

    total = sum(s["rows"] for s in all_stats.values())
    print(f"\ntotal {total} rows (the release is 47,052)")
    if not per_corpus and total != 47052:
        print("WARNING: row count is not 47,052 -- a file is short or partly unreadable.")

    corpus_wide = {k: summarize(v, k) for k, v in combined.items()}
    cw = corpus_wide["pixels"]
    ct = corpus_wide["image_tokens_per_row"]
    print(
        f"corpus-wide  pixels/image median={cw['median']:,} p99={cw['p99']:,} max={cw['max']:,}\n"
        f"             image tokens/row median={ct['median']:,} p99={ct['p99']:,} max={ct['max']:,}"
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    stats_doc = {
        "hf_repo": HF_REPO,
        "hf_snapshot": str(hf_dir),
        "total_rows": total,
        "corpus_wide": corpus_wide,
        "per_corpus": all_stats,
        "geometry": {
            "patch_factor_px": PATCH_FACTOR,
            "note": "Qwen3-VL patch-16 / merge-2 -> 32 px per visual token. "
                    "verl's data.image_patch_size defaults to 14 and MUST be set to 16.",
            "image_token_cap": IMAGE_MAX_TOKENS,
        },
        "data_source_routing": {
            "vstar": "vl_agent.compute_score  (0.8*acc + 0.2*format + 1.2*tool)",
            "chart": "vl_agent.compute_score  (0.8*acc + 0.2*format + 1.2*tool)",
            "thinklite_eureka": "vl_agent.compute_score_math  (1.2*acc + 0.4*format, no tool term)",
            "source": "DeepEyes verl/utils/reward_score/__init__.py:64-74, verbatim",
        },
        "release_defect_index": (
            "extra_info.index is the constant string '0' on all 11,031 "
            "data_thinklite_reasoning_acc rows. Rewritten here to '<corpus>:<row>'; the "
            "published value is kept as extra_info.orig_index."
        ),
    }
    (out_dir / f"stats_{tag}.json").write_text(json.dumps(stats_doc, indent=2))
    print(f"\nwrote {out_dir / f'stats_{tag}.json'}")

    if args.stats_only:
        print("\n--stats-only: no parquet written.")
        return 0

    # ---- the stratified val carve ------------------------------------------------
    import numpy as np

    rng = np.random.default_rng(args.seed)
    schema = output_schema()
    manifest_files, n_train, n_val = {}, 0, 0
    val_rows = []
    for name in SOURCES:
        rows = all_rows[name]
        order = rng.permutation(len(rows))
        rows = [rows[i] for i in order]
        share = int(round(args.val_size * len(rows) / max(1, total))) if args.val_size else 0
        # Never let the holdout swallow the split -- it does on --smoke, where
        # val_size is larger than the corpus.
        share = min(share, len(rows) // 4)
        val_rows += rows[:share]
        train = rows[share:]
        p = write_parquet(train, out_dir / f"train_{name}_{tag}.parquet", schema)
        manifest_files[f"train_{name}"] = {"path": str(p), "rows": len(train),
                                           "bytes": p.stat().st_size}
        n_train += len(train)
        print(f"  {p}  ({len(train)} rows, {p.stat().st_size / 1e9:.2f} GB)")
    if val_rows:
        rng.shuffle(val_rows)
        p = write_parquet(val_rows, out_dir / f"val_mixed_{tag}.parquet", schema)
        manifest_files["val"] = {"path": str(p), "rows": len(val_rows), "bytes": p.stat().st_size}
        n_val = len(val_rows)
        print(f"  {p}  ({n_val} rows, {p.stat().st_size / 1e9:.2f} GB)")

    manifest = {
        "hf_repo": HF_REPO,
        "hf_snapshot": str(hf_dir),
        "tag": tag,
        "n_train": n_train,
        "n_val": n_val,
        "seed": args.seed,
        "image_mode": args.image_mode,
        "image_dir": str(out_dir / "images") if args.image_mode == "file" else None,
        "max_rows_per_corpus": per_corpus or None,
        "files": manifest_files,
        "rows_by_data_source": {s["data_source"]: all_stats[n]["rows"] for n, s in SOURCES.items()},
        "agent_name_by_data_source": {
            "vstar": TOOL_AGENT,
            "chart": TOOL_AGENT,
            "thinklite_eureka": PLAIN_AGENT,
        },
        "images_resized": False,
        "note": (
            "Images copied byte for byte from the release; build_grpo_sets.py's 512 px "
            "cap is NOT applied and must never be. env_name is preserved in extra_info "
            "for provenance only -- modern verl routes on agent_name."
        ),
        "launcher_must_set": {
            "data.image_patch_size": 16,
            "data.tool_config_path": None,
            "actor_rollout_ref.rollout.multi_turn.tool_config_path": "<the zoom tool yaml>",
            "actor_rollout_ref.rollout.agent.default_agent_loop": PLAIN_AGENT,
            "why": (
                "image_patch_size defaults to 14, which is Qwen2.5-VL's grid. "
                "data.tool_config_path inherits the rollout one by oc.select and then "
                "RLHFDataset renders the tool schema into the prompt a second time and "
                "constructs a ray actor during dataset build."
            ),
        },
        "known_deviation": (
            "ToolAgentLoop._handle_pending_state passes tools=self.tool_schemas "
            "unconditionally, so the rendered prompt carries the tool block twice: once "
            "from DeepEyes' own system turn, once from the chat template. Not fixable "
            "from the data side."
        ),
    }
    (out_dir / f"manifest_{tag}.json").write_text(json.dumps(manifest, indent=2))
    print(f"  {out_dir / f'manifest_{tag}.json'}")

    if not args.no_verify:
        print("\nverifying:")
        r = verify(out_dir, hf_dir, args.model, args.verify_rows, args.strict_template)
        (out_dir / f"verify_{tag}.json").write_text(json.dumps(r, indent=2))
        if not r["ok"]:
            print("\nVERIFY FAILED -- see the checks above.", file=sys.stderr)
            return 1

    print(f"\ntrain {n_train} / val {n_val}")
    print("\nThen, in the launcher:")
    print(f"  data.train_files=[{','.join(manifest_files[k]['path'] for k in manifest_files if k.startswith('train_'))}]")
    if n_val:
        print(f"  data.val_files=[{manifest_files['val']['path']}]")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("HF_HOME", "/home/uberger/scratch/cache/hf_cache")
    raise SystemExit(main())
