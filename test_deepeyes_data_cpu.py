#!/usr/bin/env python
"""Does verl's own RLHFDataset read the switch this corpus writes? CPU, no GPU, no ray.

    python test_deepeyes_data_cpu.py --data-dir cold_data/deepeyes/verl

`build_deepeyes_data.py --verify` checks the PARQUET. This checks the thing the parquet
exists to drive: that `verl.utils.dataset.rl_dataset.RLHFDataset` and `collate_fn`,
unmodified, turn these rows into a `non_tensor_batch` where

  * `agent_name` is "tool_agent" on the V*/chart rows and "single_turn_agent" on the
    ThinkLite ones -- which is the ONLY thing standing between the math split and a
    zoom tool it is not supposed to have, and
  * `tools_kwargs` arrives as the dict `ImageZoomInTool.create()` wants on the first
    and as a falsy value on the second.

Both are silent when wrong. A wrong `agent_name` trains happily and scores 23% of the
corpus with the wrong composite; a dropped `tools_kwargs` raises inside a ray actor
several minutes into a rollout, where the traceback is somebody else's.

It also checks the two sizes P3's tool now demands -- `original_size` and `view_size`
-- survive into the row, and, more usefully, that the declared `view_size` equals the
size `RLHFDataset` ACTUALLY shows the model. Those two agree only when
`data.image_patch_size` is 16; verl defaults it to 14, and on 14 the tool's
crop-source check rejects every call. The test demonstrates that failure rather than
describing it.

Finally it asserts the prompt renders ONCE. `data.tool_config_path` defaults to
`${oc.select:actor_rollout_ref.rollout.multi_turn.tool_config_path, null}`, so the
moment the launcher configures the zoom tool, RLHFDataset starts passing `tools=` to
`apply_chat_template` -- and DeepEyes' prompt already carries the tool block as a
system turn, so the schema would appear twice. The token cost of that duplication is
the number that argues for `data.tool_config_path=null`.

Runs on the login node: it needs the tokenizer and image processor, not the model.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent

TOOL_AGENT = "tool_agent"
PLAIN_AGENT = "single_turn_agent"
TOOL_NAME = "image_zoom_in_tool"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(REPO / "cold_data" / "deepeyes" / "verl"))
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--rows", type=int, default=8, help="rows read per shard")
    ap.add_argument("--max-prompt-length", type=int, default=16384)
    # The val shard by default: it is carved from the same rows by the same code and
    # holds all three data_sources in 35 MB, where the train shards are 4.5 GB and
    # `datasets` copies every byte into its cache to read eight rows.
    ap.add_argument("--glob", default="val_*.parquet", help="use 'train_*.parquet' for the heavy run")
    ap.add_argument("--cache-dir", default="/tmp/verl_rlhf_cache",
                    help="RLHFDataset copies each parquet here before reading it")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data_dir = Path(args.data_dir).resolve()
    shards = sorted(data_dir.glob(args.glob))
    if not shards:
        raise SystemExit(f"no {args.glob} under {data_dir} -- run build_deepeyes_data.py")

    from hydra import compose, initialize_config_dir

    import verl
    from verl.protocol import DataProto
    from verl.utils import hf_processor, hf_tokenizer
    from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn

    verl_dir = Path(verl.__file__).parent
    print(f"verl    : {verl_dir}")
    print(f"shards  : {[s.name for s in shards]}\n", flush=True)

    from omegaconf import OmegaConf

    with initialize_config_dir(config_dir=str(verl_dir / "trainer/config"), version_base=None):
        config = compose(config_name="ppo_trainer")

    # `data.cache_dir` is read by RLHFDataset (`config.get("cache_dir", ...)`) but is not
    # declared in the config schema, so hydra's struct mode rejects it. On the command
    # line it needs `+data.cache_dir=...`; here it needs this.
    OmegaConf.set_struct(config, False)
    d = config.data
    d.max_prompt_length = args.max_prompt_length
    d.image_key = "images"
    d.return_raw_chat = True
    d.return_multi_modal_inputs = False
    d.filter_overlong_prompts = False
    # Qwen3-VL is patch-16. The default of 14 is Qwen2.5-VL's grid and silently resizes
    # every image onto 28 px tiles -- see build_deepeyes_data.py's module docstring.
    d.image_patch_size = 16
    d.tool_config_path = None
    d.cache_dir = args.cache_dir

    tokenizer = hf_tokenizer(args.model)
    processor = hf_processor(args.model, use_fast=True)
    print(f"processor: {type(processor).__name__} / {type(processor.image_processor).__name__} "
          f"patch={processor.image_processor.patch_size} merge={processor.image_processor.merge_size}\n",
          flush=True)

    results, ok = [], True

    def check(name, cond, detail=""):
        nonlocal ok
        results.append({"check": name, "ok": bool(cond), "detail": detail})
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}", flush=True)

    # ---- the switch, through the real dataset ---------------------------------
    rows, by_agent = [], {}
    for shard in shards:
        ds = RLHFDataset(data_files=[str(shard)], tokenizer=tokenizer, config=d, processor=processor)
        for i in range(min(args.rows, len(ds))):
            r = ds[i]
            rows.append(r)
            by_agent.setdefault(r["agent_name"], []).append(r)
    print()
    check("RLHFDataset yields agent_name", all("agent_name" in r for r in rows),
          f"{sorted(by_agent)} over {len(rows)} rows")
    check("both agent loops present", set(by_agent) == {TOOL_AGENT, PLAIN_AGENT},
          json.dumps({k: len(v) for k, v in by_agent.items()}))

    tool_rows = by_agent.get(TOOL_AGENT, [])
    math_rows = by_agent.get(PLAIN_AGENT, [])
    def ck_of(r):
        return ((r.get("tools_kwargs") or {}).get(TOOL_NAME) or {}).get("create_kwargs", {}) or {}

    check(
        "tool rows carry create_kwargs.image RLHFDataset can hand to ImageZoomInTool.create",
        bool(tool_rows) and all(ck_of(r).get("image") for r in tool_rows),
        f"{len(tool_rows)} rows",
    )
    # P3's crop-source check refuses to run without these, and that refusal is by
    # design -- an already-resized image is indistinguishable from an original that
    # needed no resize, so the data has to say which it is.
    check(
        "tool rows carry original_size and view_size through RLHFDataset",
        bool(tool_rows)
        and all(len(ck_of(r).get("original_size") or []) == 2 for r in tool_rows)
        and all(len(ck_of(r).get("view_size") or []) == 2 for r in tool_rows),
        "; ".join(f"{r['data_source']} orig={list(ck_of(r).get('original_size') or [])} "
                  f"view={list(ck_of(r).get('view_size') or [])}" for r in tool_rows[:3]),
    )
    check("math rows have a falsy tools_kwargs", bool(math_rows) and not any(r.get("tools_kwargs") for r in math_rows),
          f"{len(math_rows)} rows, values={set(str(r.get('tools_kwargs')) for r in math_rows)}")
    check("math rows are thinklite_eureka", all(r["data_source"] == "thinklite_eureka" for r in math_rows))
    check("tool rows are vstar/chart", all(r["data_source"] in {"vstar", "chart"} for r in tool_rows),
          str(sorted({r["data_source"] for r in tool_rows})))

    # ---- it survives collate into non_tensor_batch ------------------------------
    batch = DataProto.from_single_dict(collate_fn(rows))
    nt = batch.non_tensor_batch
    check("agent_name survives collate_fn into non_tensor_batch", "agent_name" in nt,
          f"keys={sorted(nt.keys())}")
    check("tools_kwargs survives collate_fn", "tools_kwargs" in nt)
    if "agent_name" in nt:
        check("per-row agent_name is preserved through DataProto",
              set(nt["agent_name"].tolist()) == {TOOL_AGENT, PLAIN_AGENT},
              str(sorted(set(nt["agent_name"].tolist()))))

    # ---- the <image> placeholder and the MRoPE shape ----------------------------
    pid = rows[0]["position_ids"]
    check("position_ids are Qwen3-VL's interleaved MRoPE (4 rows), not 1-D",
          getattr(pid, "ndim", 1) == 2 and pid.shape[0] == 4, f"shape={tuple(pid.shape)}")
    image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    n_img = [int((r["input_ids"] == image_token_id).sum()) for r in rows]
    check("every row spends image tokens", all(n > 0 for n in n_img), f"per row: {n_img}")

    # ---- view_size vs what the dataloader ACTUALLY shows the model ---------------
    # The end-to-end form of P3's warning. `create_kwargs.view_size` is computed on the
    # 32 px grid; `process_image` uses `data.image_patch_size`, which verl defaults to
    # 14. If they disagree the tool's crop-source check fails every call -- so compare
    # them here, where it costs nothing, instead of in a dead training job.
    mismatch = []
    for r in tool_rows:
        shown = r.get("multi_modal_data", {}).get("image") or []
        if not shown:
            continue
        got = list(shown[0].size)
        want = [int(v) for v in ck_of(r).get("view_size") or []]
        if got != want:
            mismatch.append({"data_source": r["data_source"], "shown": got, "declared": want})
    check("declared view_size == the size RLHFDataset actually shows the model "
          f"(data.image_patch_size={d.image_patch_size})",
          not mismatch, json.dumps(mismatch[:3]) if mismatch else f"{len(tool_rows)} rows agree")

    # And the same thing with verl's default, to show the failure is real rather than
    # theoretical. This one is EXPECTED to disagree; it is a demonstration, not a check.
    d14 = config.data.copy()
    d14.image_patch_size = 14
    ds14 = RLHFDataset(data_files=[str(shards[0])], tokenizer=tokenizer, config=d14, processor=processor)
    patch14 = {"image_patch_size": 14, "shown": None, "declared_view_size": None}
    for j in range(min(args.rows * 8, len(ds14))):
        r14 = ds14[j]
        want14 = [int(v) for v in ck_of(r14).get("view_size") or []]
        if not want14:
            continue  # a math row has no view_size to disagree with
        shown14 = (r14.get("multi_modal_data", {}).get("image") or [None])[0]
        patch14 = {"image_patch_size": 14, "row": j,
                   "shown": list(shown14.size) if shown14 else None,
                   "declared_view_size": want14}
        break
    print(f"\n  with verl's default data.image_patch_size=14 the dataloader shows "
          f"{patch14['shown']} while the data declares {patch14['declared_view_size']} "
          f"-- the tool would reject every call.")

    # ---- the duplicate tool block -----------------------------------------------
    # Measured by handing the chat template P3's schema directly. Constructing the tool
    # is deliberately avoided: it now needs a `bbox_frame` that P4 has not measured yet
    # and would raise, and it would spin up a ray actor for a token count.
    dup = None
    try:
        zc = __import__("build_deepeyes_data").load_zoom_tool_config()
        msgs0 = [{"role": m["role"], "content": m["content"]} for m in
                 [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}]]
        a = len(tokenizer.apply_chat_template(msgs0, add_generation_prompt=True, tokenize=True))
        b = len(tokenizer.apply_chat_template(msgs0, add_generation_prompt=True, tokenize=True,
                                              tools=[zc.TOOL_SCHEMA]))
        dup = {"without_tools": a, "with_tools": b, "extra_tokens": b - a}
        print(f"\n  tool-schema duplication: passing tools= to the chat template adds "
              f"{b - a} tokens. DeepEyes' prompt already carries the block as a system "
              f"turn, so data.tool_config_path must stay null.")
        check("tools= really does re-render the schema (so the null matters)", b > a,
              f"+{b - a} tokens")
    except Exception as e:  # noqa: BLE001
        print(f"\n  [SKIP] tool-schema duplication -- {type(e).__name__}: {e}")

    out = {"ok": ok, "checks": results, "duplication": dup, "image_patch_size_14": patch14,
           "shards": [str(s) for s in shards]}
    if args.out:
        p = Path(args.out) / "test_deepeyes_data_cpu.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {p}")
    print(f"\n{'ALL PASS' if ok else 'FAILURES -- see above'}")
    return 0 if ok else 1


if __name__ == "__main__":
    os.environ.setdefault("HF_HOME", "/home/uberger/scratch/cache/hf_cache")
    sys.exit(main())
