# DeepEyes on Qwen3-VL — port and training estimate

Assessment written 2026-11-07. The question: what would it cost to implement
DeepEyes (Zheng et al., ICLR 2026, [arXiv:2505.14362](https://arxiv.org/abs/2505.14362))
on Qwen3-VL-8B so the result is a comparable row next to our model?

**Verdict: ~3–5 weeks of engineering and ~1,000–2,500 A100-hours — but do not port
their fork, and run the 2-day gate in §8 first.**

Their released code is frozen on a stack that predates Qwen3-VL by six months.
Upstream verl has since grown a native, tested version of the exact machinery they
hand-rolled, so the port collapses to roughly 500 lines of tool + reward.

Repos cloned for this assessment (scratch, not under version control):

    /home/uberger/scratch/ref/DeepEyes          # Visual-Agent/DeepEyes @ 11d20c6 (2025-11-14)
    /home/uberger/scratch/ref/verl_upstream     # verl-project/verl @ 0.10.0.dev


## 1. What the method is, mechanically

Qwen2.5-VL-7B (and 32B), GRPO, KL coefficient 0, **80 steps**, 256 prompts × 16
rollouts, `max_response_length=20480`, `max_prompt_length=8192`, lr 1e-6, on H100s.

One tool. `image_zoom_in_tool` takes `bbox_2d = [x1,y1,x2,y2]`, crops the **original,
pre-processor-resize** image, and the crop is spliced back into the trajectory as a
fresh user turn. Observation tokens are masked out of the policy loss. The model
decides for itself whether to zoom or answer.

The reward is the whole trick (`verl/utils/reward_score/vl_agent.py:266`):

    0.8·R_acc  +  0.2·R_format  +  1.2·R_tool

    R_tool   = 1 iff (at least one zoom happened AND R_acc > 0.5)
    R_format = -1 on any tag mismatch, else 0
    R_acc    = self-hosted Qwen2.5-72B judge, binary

The conditional gate is the design: a tool bonus that only pays out on a *correct*
answer. The math subset (`compute_score_math`) drops the tool term entirely and uses
`1.2·acc + 0.4·format` with math_verify plus a generative fallback.

Training data: 47k = V* 22k (47%) + ArxivQA/chart 14k (30%) + ThinkLite-VL 11k (23%),
filtered for difficulty (drop 0% and 100% accuracy items), label correctness, and
"solvable via active perception".

**One fidelity wrinkle.** The paper says "up to a maximum of 6 times of active
perceptions"; the released script sets `max_turns=5`, and in their loop that counts
generate rounds, so it is at most 4 tool calls. Pick one and say which.


## 2. What ports for free

Checked against `checkpoint/coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged`
rather than assumed:

Qwen3-VL uses the **identical ChatML scheme and the identical vision token IDs** as
Qwen2.5-VL — `vision_start 151652`, `vision_end 151653`, `image_token 151655`,
`<|im_start|>` / `<|im_end|>`. Therefore all of the following transfer verbatim:

* the hardcoded observation splice `"\n<|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n"`
  (`visual_toolbox_v2.py:103`)
* the `<image>` → `<|vision_start|><|image_pad|><|vision_end|>` rewrite
  (`parallel_env.py:85`)
* the reward's trick of detecting "a zoom happened" by counting
  `<|vision_start|><|image_pad|>` occurrences in the response string
  (`vl_agent.py:199`)

This was the main worry going in. It is a non-issue.


## 3. What is broken

| | DeepEyes pin | Qwen3-VL needs |
|---|---|---|
| verl | 0.2.0.dev0 (~Feb 2025) | — |
| transformers | 4.51.3 | ≥ 4.57 |
| vLLM | 0.8.2 | ≥ 0.11 |
| torch | 2.6.0 | — |

Two specific landmines in their agent code, both in `verl/workers/agent/parallel_env.py`:

* **Line 11** — `from verl.models.transformers.qwen2_vl import get_rope_index`.
  Qwen3-VL's interleaved MRoPE is a different scheme.
* **Line 282** — position-id computation is gated on
  `processor.image_processor.__class__.__name__ == "Qwen2VLImageProcessor"`. On
  Qwen3-VL that is False, so it **silently falls through to 1-D LM position ids**.
  Training looks fine and is wrong. This is the one that would cost a week if the
  fork were ported naively.


## 4. Why to skip their fork entirely

Upstream verl (0.10.0.dev) already ships everything DeepEyes built by hand:

* `verl/models/transformers/qwen3_vl.py` — Qwen3-VL `get_rope_index` (line 99)
* `verl/experimental/agent_loop/tool_agent_loop.py` — multi-turn with
  `ToolResponse(text=, image=, video=)` as a first-class tool return,
  observation-token loss masking (`response_mask`), and a "Continuous Token"
  tokenization path written specifically for multimodal multi-turn
* `tests/experimental/agent_loop/test_multi_modal.py` — an end-to-end test of a VL
  model calling an image-returning tool, asserting `image_grid_thw` survives
* `examples/grpo_trainer/run_qwen3_vl_8b_fsdp.sh` — a ready Qwen3-VL-8B GRPO baseline
* a committed `uv.lock` — which matters, because per the `laser-env-gotchas` memory
  dependency resolution is historically where the weeks go on this kind of job

DeepEyes' actual delta against verl core is small and surgical: an `action_mask`
swapped in for `attention_mask` at ~6 call sites (`ray_trainer.py:159,187`,
`dp_actor.py:272,318`), an `agent:` config block in `ppo_trainer.yaml:121`, and one
hook into the vLLM rollout (`vllm_rollout_spmd.py:279`). Modern verl's `response_mask`
+ AgentLoop subsumes all of it. Forward-porting their 1,847-line plugin would be
rewriting code that already exists and is tested.

**So the deliverable is:** `image_zoom_in_tool` as a verl `BaseTool` (lift the
crop/validate/28px-minimum logic from `visual_toolbox_v2.py:129-165` verbatim), the
three reward functions, a data conversion, and a launcher. Call it ~500 lines.

Note also that ms-swift ships a [documented DeepEyes recipe](https://swift.readthedocs.io/en/latest/Instruction/GRPO/AdvancedResearch/deepeyes.html)
(`examples/train/grpo/plugin/deepeyes/`), but its script targets Qwen2.5-VL-7B and
it is a third framework. Mentioned for completeness, not recommended.


## 5. Work breakdown

| | Days |
|---|---|
| New conda env: verl 0.10.dev + vLLM ≥0.18 + torch 2.9.1. `transformers 5.13.0.dev0` is already in verl's `>=5.5.3,<5.17` range. `uv.lock` helps; flash-attn on A100 is the usual risk | 2–3 |
| Smoke: verl GRPO on Qwen3-VL-8B / geo3k, 8 GPUs — prove the stack before adding anything | 1–2 |
| `image_zoom_in_tool` as a verl `BaseTool`; wire to the un-resized original image | 1–2 |
| Reward port; swap the 72B judge for ours; re-derive "a zoom happened" from verl turn metadata rather than token counting | 2 |
| Data conversion (see §6) | 1–2 |
| Launcher in this repo's shape: SLURM, 2–4 h chunks, autoresume, Ray if multi-node | 2–3 |
| **Debugging the first real rollouts** — format rate, tool-call rate, truncation at 20k tokens, OOM. Where the time actually goes | 3–5 |
| Eval (see §7) | 1–2 |
| **Total** | **~15–23 days** |

**The cost to flag hardest:** this introduces a *second training framework*.
Everything we have — cold start, overlap rewards, saliency hooks, launchers,
autoresume, `vlm_family` — is TRL. Reimplementing the zoom loop inside
`trl_repo/trl/trainer/grpo_trainer_qwen3.py` instead would keep one framework and
costs about the same engineering (3–4 weeks), but carries much higher correctness
risk — silent mrope and loss-mask bugs are precisely the failure mode — and a
reimplementation is weaker as a paper baseline than "we ran the authors' recipe on
current verl."


## 6. Data

`ChenShawn/DeepEyes-Datasets-47k` on HF: **4.84 GB in 3 parquet files**, images
embedded as bytes.

    0.990 GB  data_0.1.2_visual_toolbox_v2.parquet      # V* / visual search, zoom tool on
    2.199 GB  data_v0.8_visual_toolbox_v2.parquet       # charts, zoom tool on
    1.656 GB  data_thinklite_reasoning_acc.parquet      # math reasoning, zoom tool OFF

Already in verl's schema — `data_source`, `prompt`, `images`, `env_name`,
`reward_model.{ground_truth,style}`, `extra_info.{question,answer,index,split}`. The
conversion is mostly re-routing `env_name=visual_toolbox_v2` to modern verl's
agent/tools config, and confirming the prompt format against Qwen3-VL's chat template
and hermes tool-call parsing.


## 7. Eval is nearly free

`eval/eval_vstar.py` and `eval/eval_hrbench.py` in their repo are plain OpenAI-API
clients against a vLLM server, with the bbox/crop loop client-side. **Model-agnostic
— zero changes for Qwen3-VL**, just point `--api_url` at a vLLM 0.11+ server.
`judge_result.py` wants a judge endpoint, which we already have.

`vstar_bench` (craigwu) and `HR-Bench` (DreamMr) are both already in the HF cache.
`mmerealworld_mini` and `pope_mini` are already in our `eval_mini` natural set, so the
overlap with DeepEyes' headline table is partly covered.


## 8. Run this gate first — 2 days, and it may end the discussion

`ChenShawn/DeepEyes-7B` is public. Serve it on vLLM, run their `eval_vstar.py` /
`eval_hrbench.py` plus our `eval_mini` natural set, against Qwen3-VL-8B-Instruct and
against our cold start. That says whether the method's gains survive our benchmarks
before a month is spent.

**The reason to gate: Qwen3-VL already received this treatment.** Its
[tech report](https://arxiv.org/pdf/2511.21631) describes synthesizing ~10k grounding
traces, SFT-ing Qwen2.5-VL-32B into a "think → act → analyze feedback → answer"
visual agent, distilling ~120k multi-turn agentic interactions, then cold-start SFT +
tool-integrated RL with accuracy (Qwen3-32B judge), multi-turn-reasoning
(Qwen2.5-VL-72B judge) and tool-calling rewards. That is DeepEyes, done by the vendor,
at larger scale. The official Qwen3-VL cookbook even ships an `image_zoom_in_tool`.

Headroom does appear to survive, but it is much smaller than the headline:

| | V* | HRBench-4K | HRBench-8K |
|---|---|---|---|
| Qwen2.5-VL-7B (their baseline) | 71.2 | 68.8 | 65.3 |
| DeepEyes-7B | 90.1 | 75.1 | 72.6 |
| Qwen3-VL-8B-Thinking, **no tool** | 77.5 | 72.4 | 68.1 |

So the honest expected result is something like "+10 on V*, ~+2 elsewhere" rather than
the paper's +18.9. And the paper's own framing — "this ability emerges from RL without
cold-start SFT" — is already falsified for this backbone, because Qwen3-VL got exactly
that cold start.


## 9. GPU budget and the cluster blocker

Their README demands **32 GPUs (4×8) minimum** for the 7B plus 4 more for the 72B
judge, and 1,200 GB CPU RAM per node. Both shrink:

* our existing hosted judge removes the 4 judge GPUs outright;
* per [[fewer-gpus-keeps-batch-and-total-steps]], trading ranks for grad-accum
  preserves batch and step count, so 8 GPUs can hold the same 256×16 batch — slower,
  not different.

Scaling from our ~22.5 s/step (of which ~9.5 s is the rank-scaling part, per
[[grpo-step-time-is-mostly-fixed-cost]]) by token throughput — DeepEyes pushes ~50M
tokens/step against our ~0.7M — gives roughly **3–7 A100-hours per step**, so
**~400–1,200 A100-hours** for one 80-step run. Multi-turn rollouts are latency-bound
rather than FLOP-bound (5 sequential generate rounds with stragglers), so take the top
of that band. Budget 2–3 runs — smoke, real, fix — for **~1,000–2,500 A100-hours**,
i.e. 5–13 days of wall clock on 8 GPUs.

These are a model, not a measurement. The first real smoke run replaces them.

**Blocker.** `sinfo -h -N -o "%G" | grep -ci gpu` returns **0** on this login node —
`oci-nrt-cs-001` still has no GPU hardware ([[gpu-partitions-can-be-empty-of-hardware]]).
All of this has to run on `cs-oci-ord-login-01`, which is **A100, not H100** — hence
the 2x applied to the H100-derived figures above. Whether that cluster can hold a
multi-node Ray job is the one input not visible from here.


## Sources

* [arXiv:2505.14362](https://arxiv.org/abs/2505.14362) — the paper (ICLR 2026)
* [Visual-Agent/DeepEyes](https://github.com/Visual-Agent/DeepEyes) — their code
* [ChenShawn/DeepEyes-7B](https://huggingface.co/ChenShawn/DeepEyes-7B) — the checkpoint
* [ChenShawn/DeepEyes-Datasets-47k](https://huggingface.co/datasets/ChenShawn/DeepEyes-Datasets-47k) — the data
* [verl-project/verl](https://github.com/verl-project/verl) — the framework to build on
* [ms-swift DeepEyes recipe](https://swift.readthedocs.io/en/latest/Instruction/GRPO/AdvancedResearch/deepeyes.html) — Qwen2.5-VL only
* [Qwen3-VL Technical Report](https://arxiv.org/pdf/2511.21631) — §8's caveat
