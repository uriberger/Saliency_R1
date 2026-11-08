#!/usr/bin/env python
"""P2's three assertions, on the real agent-loop code, with no GPU.

WHY THIS EXISTS. `oci-nrt-cs-001` currently has ZERO GPU nodes -- every GPU partition
reports `Nodes=(null) TotalNodes=0` (confirmed from `scontrol show partition` and
independently through the slurm broker), so deepeyes_tool_smoke.py cannot run here
today. That is an infrastructure fact, not a verdict on the port, and it would be a
poor reason to leave assertions A1/A2/A3 unanswered.

WHAT A GPU IS ACTUALLY FOR IN STAGE B. One thing: turning a prompt into tokens. Every
part the three assertions are ABOUT -- tool dispatch, image handling, chat templating,
the observation mask, `get_rope_index`, `multi_modal_inputs` -- is CPU code in
`verl/experimental/agent_loop/`. So replace vLLM with a stub that returns scripted
token ids and run the rest FOR REAL:

    ToolAgentLoop.run                     real
    ImageZoomInTool.create / execute      real
    processor.apply_chat_template         real
    AgentLoopWorkerBase._agent_loop_postprocess / _postprocess   real
    the model's own generation            SCRIPTED

WHAT THIS DOES AND DOES NOT ESTABLISH
  does      A1, A2, A3 against the production code path, on Qwen3-VL's real processor.
  does      that the fixed agent_loop.py dispatches to qwen3_vl.get_rope_index.
  does NOT  that Qwen3-VL-8B emits a parseable <tool_call> of its own accord -- that is
            P4's measurement and stage B's remaining job.
  does NOT  FSDP, vLLM, memory, or the step time. Those need the hardware.

Needs ray, which does not come up on the vscode login node but does on a CPU compute
node. Send it to cpu_short:

    bash launch_deepeyes_agentloop_cpu_job.sh --name deepeyes-p2-cpu
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from deepeyes_tool_smoke import assess, build_tool_config, env_vars  # noqa: E402

# The scripted assistant turns. Turn 1 must parse under verl's `hermes` ToolParser --
# which is a regex over <tool_call>...</tool_call> containing {"name":..,"arguments":..}
# -- or the loop terminates after one turn and there is no tool turn to assert about.
TOOL_CALL_TURN = (
    "I need a closer look at the labelled region.\n"
    '<tool_call>\n{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [40, 30, 300, 240], '
    '"label": "angle labels"}}\n</tool_call>'
)
ANSWER_TURN = "Now I can read the labels. The answer is \\boxed{12}."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--max-prompt-length", type=int, default=4096)
    ap.add_argument("--max-response-length", type=int, default=2048)
    args = ap.parse_args()

    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    mrope_dir = out / "mrope"
    mrope_dir.mkdir(parents=True, exist_ok=True)
    os.environ["DEEPEYES_MROPE_DIR"] = str(mrope_dir)
    os.environ.setdefault("DEEPEYES_ASSERT_MROPE", "1")

    import numpy as np
    import ray
    from hydra import compose, initialize_config_dir

    import verl
    from verl.experimental.agent_loop.agent_loop import AgentLoopWorkerBase
    from verl.protocol import DataProto
    from verl.utils import hf_processor, hf_tokenizer
    from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
    from verl.workers.rollout.replica import TokenOutput

    verl_dir = Path(verl.__file__).parent
    print(f"verl: {verl_dir}", flush=True)

    tool_config_path = build_tool_config(out / "image_zoom_in_tool_config.yaml")

    with initialize_config_dir(config_dir=str(verl_dir / "trainer/config"), version_base=None):
        config = compose(config_name="ppo_trainer")

    c = config.actor_rollout_ref
    c.model.path = args.model
    c.rollout.name = "vllm"
    c.rollout.mode = "async"
    c.rollout.prompt_length = args.max_prompt_length
    c.rollout.response_length = args.max_response_length
    c.rollout.n = 1
    c.rollout.agent.default_agent_loop = "tool_agent"
    c.rollout.multi_turn.enable = True
    c.rollout.multi_turn.tool_config_path = str(tool_config_path)
    c.rollout.multi_turn.max_parallel_calls = 1
    c.rollout.multi_turn.max_user_turns = 2
    c.rollout.multi_turn.max_assistant_turns = 3
    c.rollout.multi_turn.tokenization_sanity_check_mode = "ignore_strippable"
    config.data.max_prompt_length = args.max_prompt_length
    config.data.max_response_length = args.max_response_length
    config.data.image_key = "images"
    config.data.return_raw_chat = True
    config.data.return_multi_modal_inputs = False
    config.trainer.n_gpus_per_node = 1
    config.trainer.nnodes = 1
    config.reward_model.enable = False

    # ray BEFORE the dataset: data.tool_config_path follows the rollout's, so
    # RLHFDataset constructs ImageZoomInTool, which makes a ray actor. See
    # deepeyes_tool_smoke.py for the full note.
    ray.init(runtime_env={"env_vars": env_vars()}, ignore_reinit_error=True, num_cpus=16)

    tokenizer = hf_tokenizer(args.model)
    processor = hf_processor(args.model, use_fast=True)
    print(f"processor: {type(processor).__name__} / {type(processor.image_processor).__name__}", flush=True)

    dataset = RLHFDataset(data_files=[args.data], tokenizer=tokenizer, config=config.data, processor=processor)
    rows = [dataset[i] for i in range(min(args.rows, len(dataset)))]
    batch = DataProto.from_single_dict(collate_fn(rows))
    reward_keys = {"data_source", "reward_model", "extra_info", "uid"} & set(batch.non_tensor_batch.keys())
    gen_batch = batch.pop(
        batch_keys=["input_ids", "attention_mask", "position_ids"],
        non_tensor_batch_keys=list(set(batch.non_tensor_batch.keys()) - reward_keys),
    )
    gen_batch.non_tensor_batch.update(batch.non_tensor_batch)
    gen_batch.meta_info = {
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "recompute_log_prob": False,
        "do_sample": True,
        "validate": False,
        "global_steps": 0,
    }
    print(f"batch: {len(gen_batch)} rows", flush=True)

    class StubServerManager:
        """vLLM, replaced by a script.

        Two turns per trajectory, keyed on whether the prompt already contains a
        <tool_response> block: turn 1 asks for the zoom, turn 2 answers. Keying on the
        prompt rather than on a counter is what makes it correct under the loop's
        concurrency -- every trajectory runs as its own asyncio task.
        """

        def __init__(self, tok):
            self.tok = tok
            self.calls = 0

        async def generate(self, request_id, *, prompt_ids, sampling_params, image_data=None):
            self.calls += 1
            text = self.tok.decode(prompt_ids)
            turn = ANSWER_TURN if "<tool_response>" in text else TOOL_CALL_TURN
            ids = self.tok.encode(turn, add_special_tokens=False)
            ids = ids + [self.tok.convert_tokens_to_ids("<|im_end|>")]
            return TokenOutput(token_ids=ids, log_probs=None)

    class CpuAgentLoopWorker(AgentLoopWorkerBase):
        def __init__(self, cfg, tok):
            # AgentLoopWorkerBase only builds an AsyncLLMServerManager if one is not
            # already present -- `if not hasattr(self, "server_manager")`. That hook is
            # how a recipe swaps the engine, and it is how we swap in a script.
            self.server_manager = StubServerManager(tok)
            super().__init__(cfg, server_handles=[], reward_router_address=None)

    worker = CpuAgentLoopWorker(config, tokenizer)
    print("worker up; running the agent loop", flush=True)

    gen = worker.generate_sequences(gen_batch)
    result = asyncio.get_event_loop().run_until_complete(gen) if asyncio.iscoroutine(gen) else gen

    print(f"generations called: {worker.server_manager.calls} "
          f"(expect 2 per row = {2 * len(gen_batch)})", flush=True)

    report = assess(result, tokenizer, mrope_dir, 0.0, 0.0)
    report["mode"] = "cpu-replay (generation scripted, everything else real)"
    report["generate_calls"] = worker.server_manager.calls
    (out / "tool_smoke_report.json").write_text(json.dumps(report, indent=2, default=str))

    print()
    print("=" * 74)
    ok = True
    for a in ("A1_position_ids", "A2_response_mask", "A3_image_grid_thw"):
        r = report[a]
        ok = ok and r["pass"]
        print(f"{'PASS' if r['pass'] else 'FAIL'}  {a}: {r['detail']}")
    print("=" * 74)
    print("NOTE: generation was SCRIPTED. This says nothing about whether Qwen3-VL-8B")
    print("      emits a parseable <tool_call> on its own -- that is P4's measurement.")
    print(f"report -> {out / 'tool_smoke_report.json'}")

    ray.shutdown()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(main())
