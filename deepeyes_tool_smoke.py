#!/usr/bin/env python
"""Stage B of P2: one multi-turn rollout through ImageZoomInTool, with the three
assertions that are silent when wrong.

Adapted from upstream verl's own end-to-end test,
`tests/experimental/agent_loop/test_multi_modal.py::test_multimodal_tool_agent` --
which this fork does not carry, because LASER stripped `tests/` and `examples/` from
their clone. The upstream file was fetched for reference and the three things it
checks are kept; everything else (model, tool, data, the MRoPE provenance check) is
ours.

Run through launch_deepeyes_smoke_job.sh; it wants GPUs.

    python deepeyes_tool_smoke.py --model Qwen/Qwen3-VL-8B-Instruct \
        --data outputs/.../zoom_smoke.parquet --out-dir outputs/... --n-gpus 8

WHAT IT ASSERTS, AND WHY EACH ONE IS INVISIBLE WHEN IT FAILS
------------------------------------------------------------
A1  position_ids are Qwen3-VL's.
    Shape (B, 4, T), not (B, T) -- and the 4-row tensor must not be four copies of
    the same ramp. Transformers accepts a 2-D tensor and broadcasts it into three
    identical rows, so the ONLY symptom of the 1-D fall-through is a model that
    trains happily on positionless images. Provenance is checked too: the
    `[DEEPEYES-MROPE]` markers the guard drops must say `qwen3_vl` and must not say
    `qwen2_vl`.

A2  observation tokens are masked out of the policy loss.
    Reports the zero fraction of `response_mask` on turns that carried a tool image,
    and checks every `<|vision_start|> ... <|vision_end|>` span inside the RESPONSE
    is fully zero. A tool image that reaches the policy loss is a model trained to
    reproduce its own crops.

A3  image_grid_thw survives the tool turn.
    `multi_modal_inputs[i]` must carry `pixel_values` and `image_grid_thw`, and the
    grid must have one row per image -- prompt image plus every crop. One row means
    the crop was dropped on the floor and the second <|image_pad|> run has no pixels
    behind it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

TAG = "[DEEPEYES-MROPE]"


def build_tool_config(path: Path) -> Path:
    """The ImageZoomInTool entry, with the schema copied from the tool's own docstring.

    `max_pixels`/`min_pixels` are NOT set: P3 owns the geometry (28 -> 32) and a
    stack proof must not quietly introduce a resize, which is the one thing that
    would turn the whole method into a no-op.
    """
    import yaml

    cfg = {
        "tools": [
            {
                "class_name": "verl.tools.image_zoom_in_tool.ImageZoomInTool",
                "config": {"type": "native", "num_workers": 8, "rate_limit": 32},
                "tool_schema": {
                    "type": "function",
                    "function": {
                        "name": "image_zoom_in_tool",
                        "description": (
                            "Zoom in on a specific region of an image by cropping it based on a "
                            "bounding box (bbox) and an optional object label."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "bbox_2d": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "minItems": 4,
                                    "maxItems": 4,
                                    "description": (
                                        "The bounding box of the region to zoom in, as "
                                        "[x1, y1, x2, y2], where (x1, y1) is the top-left corner "
                                        "and (x2, y2) is the bottom-right corner."
                                    ),
                                },
                                "label": {
                                    "type": "string",
                                    "description": "The name or label of the object in the bbox (optional).",
                                },
                            },
                            "required": ["bbox_2d"],
                        },
                    },
                },
            }
        ]
    }
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def env_vars() -> dict:
    """What every ray actor needs. The guard has to reach the ACTORS, not the driver:
    the agent loop worker and the vLLM replicas are separate processes, and the
    get_rope_index call happens in them."""
    ev = {
        "TOKENIZERS_PARALLELISM": "true",
        "NCCL_DEBUG": "WARN",
        "VLLM_LOGGING_LEVEL": "WARN",
        "VLLM_USE_V1": "1",
        "DEEPEYES_MROPE_DIR": os.environ["DEEPEYES_MROPE_DIR"],
        "DEEPEYES_ASSERT_MROPE": os.environ.get("DEEPEYES_ASSERT_MROPE", "1"),
    }
    for k in ("HF_HOME", "HF_HUB_OFFLINE", "PYTHONPATH"):
        if os.environ.get(k):
            ev[k] = os.environ[k]
    return ev


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-gpus", type=int, default=8)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--rollout-n", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--max-prompt-length", type=int, default=4096)
    ap.add_argument("--max-response-length", type=int, default=2048)
    ap.add_argument("--gpu-mem-util", type=float, default=0.6)
    # Everything up to ray.init is CPU work -- hydra compose, the tool config, the
    # dataset, the batch. Being able to run that on a login node is how a typo in a
    # config key stops costing an 8-GPU allocation to discover.
    ap.add_argument("--dry-run", action="store_true", help="build config and batch, then stop")
    args = ap.parse_args()

    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    mrope_dir = out / "mrope"
    mrope_dir.mkdir(parents=True, exist_ok=True)
    os.environ["DEEPEYES_MROPE_DIR"] = str(mrope_dir)
    os.environ.setdefault("DEEPEYES_ASSERT_MROPE", "1")

    import numpy as np
    import ray
    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    import verl
    from verl.experimental.agent_loop import AgentLoopManager
    from verl.protocol import DataProto
    from verl.utils import hf_processor, hf_tokenizer
    from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn

    verl_dir = Path(verl.__file__).parent
    print(f"verl: {verl_dir}")

    tool_config_path = build_tool_config(out / "image_zoom_in_tool_config.yaml")

    with initialize_config_dir(config_dir=str(verl_dir / "trainer/config"), version_base=None):
        config = compose(config_name="ppo_trainer")

    c = config.actor_rollout_ref
    c.model.path = args.model
    c.rollout.name = "vllm"
    c.rollout.mode = "async"
    c.rollout.enforce_eager = True
    c.rollout.prompt_length = args.max_prompt_length
    c.rollout.response_length = args.max_response_length
    c.rollout.n = args.rollout_n
    c.rollout.tensor_model_parallel_size = args.tp
    c.rollout.gpu_memory_utilization = args.gpu_mem_util
    c.rollout.skip_tokenizer_init = True
    c.rollout.agent.num_workers = args.num_workers
    # The row carries agent_name="tool_agent", but set the default too: if the column
    # were ever dropped in conversion the run would silently use single_turn_agent and
    # "the tool was never called" would be indistinguishable from "the tool is broken".
    c.rollout.agent.default_agent_loop = "tool_agent"
    c.rollout.multi_turn.enable = True
    c.rollout.multi_turn.tool_config_path = str(tool_config_path)
    c.rollout.multi_turn.max_parallel_calls = 1
    c.rollout.multi_turn.max_user_turns = 2
    c.rollout.multi_turn.max_assistant_turns = 3
    # Turn-by-turn vs all-at-once tokenisation differs on Qwen chat templates; upstream
    # names Qwen explicitly as a model where this is expected. A strict check here
    # would abort the rollout over a known-benign difference.
    c.rollout.multi_turn.tokenization_sanity_check_mode = "ignore_strippable"
    config.data.max_prompt_length = args.max_prompt_length
    config.data.max_response_length = args.max_response_length
    config.data.image_key = "images"
    config.data.return_raw_chat = True
    config.data.return_multi_modal_inputs = False
    config.trainer.n_gpus_per_node = args.n_gpus
    config.trainer.nnodes = 1
    config.reward_model.enable = False

    tokenizer = hf_tokenizer(args.model)
    processor = hf_processor(args.model, use_fast=True)
    print(f"processor: {type(processor).__name__} / {type(processor.image_processor).__name__}")

    # RAY MUST BE UP BEFORE THE DATASET IS BUILT, and the reason is not obvious.
    # `data.tool_config_path` defaults to
    #     ${oc.select:actor_rollout_ref.rollout.multi_turn.tool_config_path, null}
    # so setting the rollout's tool config also hands it to RLHFDataset, which
    # constructs the tools at __init__ to put their schemas in the prompt. Constructing
    # ImageZoomInTool runs `init_visual_execution_pool`, which creates a ray actor --
    # and an implicit `ray.init()` with NO runtime_env. Build the dataset first and the
    # agent loop workers never see DEEPEYES_MROPE_DIR, so the guard reports nothing and
    # assertion A1's provenance check silently has no evidence to look at.
    # (It is also why this script cannot dry-run its dataset on a login node: ray's
    #  raylet will not come up there. --dry-run drops the tool config for that reason
    #  and that reason only.)
    if args.dry_run:
        config.data.tool_config_path = None
        print("[dry-run] data.tool_config_path cleared so the dataset does not need ray")
    else:
        ray.init(runtime_env={"env_vars": env_vars()}, ignore_reinit_error=True)

    dataset = RLHFDataset(
        data_files=[args.data],
        tokenizer=tokenizer,
        config=config.data,
        processor=processor,
    )
    n = args.num_workers * 2
    rows = [dataset[i] for i in range(min(n, len(dataset)))]
    batch_dict = collate_fn(rows)
    batch = DataProto.from_single_dict(batch_dict)
    # Mirror RayPPOTrainer._get_gen_batch: everything except the reward-model keys is
    # popped into the generation batch, and in async mode the reward keys are copied
    # back so the agent loop can see extra_info.
    reward_keys = {"data_source", "reward_model", "extra_info", "uid"} & set(batch.non_tensor_batch.keys())
    gen_batch = batch.pop(
        batch_keys=["input_ids", "attention_mask", "position_ids"],
        non_tensor_batch_keys=list(set(batch.non_tensor_batch.keys()) - reward_keys),
    )
    gen_batch.non_tensor_batch.update(batch.non_tensor_batch)
    gen_batch = gen_batch.repeat(repeat_times=args.rollout_n, interleave=True)
    gen_batch.meta_info = {
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "recompute_log_prob": False,
        "do_sample": True,
        "validate": False,
        "global_steps": 0,
    }
    print(f"batch: {len(gen_batch)} rows, keys={sorted(gen_batch.non_tensor_batch.keys())}")
    for key in ("raw_prompt", "multi_modal_data", "tools_kwargs", "agent_name"):
        assert key in gen_batch.non_tensor_batch, (
            f"the generation batch has no '{key}'. ToolAgentLoop.run reads all four out "
            "of non_tensor_batch; without tools_kwargs the zoom tool has no image to "
            "crop, and without agent_name verl silently uses single_turn_agent."
        )
    tk = gen_batch.non_tensor_batch["tools_kwargs"][0]
    assert "image_zoom_in_tool" in tk and tk["image_zoom_in_tool"].get("create_kwargs", {}).get("image"), (
        f"tools_kwargs[0] = {tk} -- ImageZoomInTool.create() reads create_kwargs['image']"
    )
    print(f"  agent_name[0]={gen_batch.non_tensor_batch['agent_name'][0]}  "
          f"tools_kwargs[0] keys={list(tk['image_zoom_in_tool'].keys())}")
    if args.dry_run:
        print("[dry-run] config and batch built; stopping before ray.init")
        return 0

    t0 = time.time()
    manager = AgentLoopManager(config)
    t_init = time.time() - t0
    print(f"AgentLoopManager up in {t_init:.1f}s")

    t0 = time.time()
    result = manager.generate_sequences(prompts=gen_batch)
    t_gen = time.time() - t0
    print(f"generate_sequences: {t_gen:.1f}s for {len(result)} rollouts")
    print("timing: " + json.dumps({k: round(float(v), 2) for k, v in result.meta_info.get("timing", {}).items()}))

    report = assess(result, tokenizer, mrope_dir, t_init, t_gen)
    report["config"] = {
        "model": args.model,
        "data": args.data,
        "n_gpus": args.n_gpus,
        "tp": args.tp,
        "rollout_n": args.rollout_n,
        "max_prompt_length": args.max_prompt_length,
        "max_response_length": args.max_response_length,
    }
    (out / "tool_smoke_report.json").write_text(json.dumps(report, indent=2, default=str))

    print()
    print("=" * 74)
    for a in ("A1_position_ids", "A2_response_mask", "A3_image_grid_thw"):
        r = report[a]
        print(f"{'PASS' if r['pass'] else 'FAIL'}  {a}: {r['detail']}")
    print("=" * 74)
    print(f"report -> {out / 'tool_smoke_report.json'}")

    ray.shutdown()
    return 0 if all(report[a]["pass"] for a in ("A1_position_ids", "A2_response_mask", "A3_image_grid_thw")) else 1


def assess(result, tokenizer, mrope_dir: Path, t_init: float, t_gen: float) -> dict:
    import torch

    vs = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    ve = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    pad = tokenizer.convert_tokens_to_ids("<|image_pad|>")

    num_turns = result.non_tensor_batch["__num_turns__"]
    mmi = result.non_tensor_batch.get("multi_modal_inputs")
    responses = result.batch["responses"]
    response_mask = result.batch["response_mask"]
    attention_mask = result.batch["attention_mask"]
    input_ids = result.batch["input_ids"]
    position_ids = result.batch["position_ids"]
    B, T = input_ids.shape
    R = responses.size(1)

    # A tool turn shows up as num_turns > 2: [user, assistant] is 2, each tool round
    # adds [tool, assistant].
    tool_rows = [i for i in range(B) if int(num_turns[i]) > 2]
    report: dict = {
        "n_rollouts": int(B),
        "num_turns": [int(x) for x in num_turns],
        "n_tool_rows": len(tool_rows),
        "timing": {"agent_loop_manager_init_s": round(t_init, 1), "generate_sequences_s": round(t_gen, 1)},
    }

    # ----------------------------------------------------------------- A1
    a1_ok = position_ids.dim() == 3 and tuple(position_ids.shape) == (B, 4, T)
    detail = f"position_ids {tuple(position_ids.shape)}; expected ({B}, 4, {T})"
    if a1_ok:
        # Four rows that are all the same ramp is what the 1-D fall-through looks like
        # after someone "fixed" the shape without fixing the content.
        distinct = 0
        for i in range(B):
            if not torch.equal(position_ids[i, 1], position_ids[i, 2]):
                distinct += 1
        a1_ok = distinct > 0
        detail += f"; {distinct}/{B} rows have t!=h (MRoPE actually varies)"

    marks: dict[str, int] = {}
    for p in sorted(mrope_dir.glob("mrope.*.json")):
        try:
            for k, v in json.loads(p.read_text())["counts"].items():
                marks[k] = marks.get(k, 0) + int(v)
        except Exception:
            pass
    report["mrope_markers"] = marks
    detail += f"; markers={marks}"
    if marks:
        if marks.get("qwen2_vl"):
            a1_ok = False
            detail += "  <-- qwen2_vl.get_rope_index RAN; that is the Qwen2.5-VL MRoPE on a Qwen3-VL model"
        if not marks.get("qwen3_vl"):
            a1_ok = False
            detail += "  <-- qwen3_vl.get_rope_index never ran"
    else:
        detail += "  (no marker files; grep the log for '[DEEPEYES-MROPE]' instead)"
    report["A1_position_ids"] = {"pass": bool(a1_ok), "detail": detail}

    # ----------------------------------------------------------------- A2
    zero_fracs, span_fails, spans_checked = [], 0, 0
    for i in tool_rows:
        valid = attention_mask[i][-R:].bool()
        n_valid = int(valid.sum())
        if n_valid == 0:
            continue
        zeros = int((response_mask[i][:n_valid] == 0).sum())
        zero_fracs.append(zeros / n_valid)
        ids = responses[i][:n_valid].tolist()
        m = response_mask[i][:n_valid].tolist()
        depth, start = 0, None
        for j, t in enumerate(ids):
            if t == vs:
                start = j
            elif t == ve and start is not None:
                spans_checked += 1
                if any(x != 0 for x in m[start : j + 1]):
                    span_fails += 1
                start = None
    mean_zero = sum(zero_fracs) / len(zero_fracs) if zero_fracs else 0.0
    a2_ok = bool(tool_rows) and mean_zero > 0.0 and span_fails == 0 and spans_checked > 0
    report["A2_response_mask"] = {
        "pass": a2_ok,
        "detail": (
            f"{len(tool_rows)}/{B} rollouts called the tool; mean masked fraction of "
            f"response_mask on those rows = {mean_zero:.3f}; vision spans inside the "
            f"response checked = {spans_checked}, not fully masked = {span_fails}"
        ),
        "zero_fracs": [round(z, 4) for z in zero_fracs],
        "vision_spans_checked": spans_checked,
        "vision_spans_leaking": span_fails,
    }

    # ----------------------------------------------------------------- A3
    grids, missing = [], 0
    for i in tool_rows:
        d = mmi[i] if mmi is not None else {}
        if "image_grid_thw" not in d or "pixel_values" not in d:
            missing += 1
            continue
        g = d["image_grid_thw"]
        n_pads = int((input_ids[i] == pad).sum())
        n_tokens = int((g[:, 0] * g[:, 1] * g[:, 2]).sum()) // 4  # merge_size 2 -> /4
        grids.append({"rows": int(g.shape[0]), "grid": g.tolist(), "image_pads": n_pads, "grid_tokens": n_tokens})
    a3_ok = bool(tool_rows) and missing == 0 and all(g["rows"] >= 2 for g in grids)
    if grids:
        a3_ok = a3_ok and all(g["image_pads"] == g["grid_tokens"] for g in grids)
    report["A3_image_grid_thw"] = {
        "pass": bool(a3_ok),
        "detail": (
            f"{len(grids)} tool rows carry pixel_values+image_grid_thw ({missing} missing); "
            f"grid rows per row = {[g['rows'] for g in grids]}; "
            f"image_pad count == grid token count on "
            f"{sum(1 for g in grids if g['image_pads'] == g['grid_tokens'])}/{len(grids)}"
        ),
        "grids": grids[:8],
    }

    # A few decoded rollouts, with and without the observation, so the coordinate frame
    # and the tool call are eyeballable from the log. P3 and P9 both want this.
    samples = []
    for i in (tool_rows or list(range(min(2, B))))[:3]:
        valid = attention_mask[i][-R:].bool()
        n_valid = int(valid.sum())
        with_obs = tokenizer.decode(responses[i][:n_valid])
        without = tokenizer.decode(responses[i][:n_valid][response_mask[i][:n_valid].bool()])
        samples.append({"row": int(i), "num_turns": int(num_turns[i]), "with_obs": with_obs[:4000],
                        "without_obs": without[:4000]})
    report["samples"] = samples
    for s in samples:
        print("=" * 74)
        print(f"row {s['row']}  num_turns={s['num_turns']}")
        print("--- response WITH tool observation ---")
        print(s["with_obs"][:2500])
        print("--- response WITHOUT (what the policy loss sees) ---")
        print(s["without_obs"][:2500])
    return report


if __name__ == "__main__":
    sys.exit(main())
