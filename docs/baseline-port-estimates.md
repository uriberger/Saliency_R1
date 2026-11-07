# Porting external baselines onto Qwen3-VL — estimates

One section per external paper we have costed as a comparable row next to our model:
same backbone (Qwen3-VL-8B-Instruct), same eval protocol, their method. Each section
is a decision document, not a plan of record — the numbers in them are models, and the
first real smoke run replaces the numbers.

The first three are tool-using visual agents and share a shape. Monet is the odd one:
it has no tool, it edits the forward pass instead, and it is the only one here whose
cost is dominated by SFT rather than by rollouts.

| baseline | verdict | gate before committing |
|---|---|---|
| [DeepEyes](#deepeyes-on-qwen3-vl--port-and-training-estimate) — zoom tool, 80 steps, no cold start | ~3–5 weeks, ~1,000–2,500 A100-h | 2 days: serve `ChenShawn/DeepEyes-7B`, score it on our suite |
| [Mini-o3](#mini-o3-on-qwen3-vl--port-and-training-estimate) — crop tool, deep multi-turn, over-turn masking | ~3–4 weeks, ~2,000–3,000 A100-h | 3–5 days: build the agentic eval wrapper, score their released `Mini-o3-7B-v1` |
| [ERGO](#ergo-on-qwen3-vl--port-and-training-estimate) — two fixed turns, efficiency-oriented, crop verified by a VLM | ~2.5–3.5 weeks, ~400–1,000 A100-h | **1 day**: does Qwen3-VL even have a low-resolution penalty to recover? |
| [Monet](#monet-on-qwen3-vl--port-and-training-estimate) — no tool, latent visual embeddings, 3-stage SFT + VLPO | **~6–9 weeks, ~1,500–3,000 A100-h** — the most expensive | 2–3 days: serve `NOVAglow646/Monet-7B` on their own vLLM pin, score it on our suite |

### What every one of these has in common

Three findings recur, and they are why these estimates land where they do rather than
at "six months":

1. **Every released fork is frozen pre-Qwen3-VL.** DeepEyes pins verl 0.2.0.dev0 /
   transformers 4.51.3 / vLLM 0.8.2; Mini-o3 pins verl 0.2.0.dev / transformers 4.52.0 /
   vLLM 0.9.2; Monet pins two stacks, transformers 4.54.0 / vLLM 0.10.0 for SFT and
   transformers 4.51.3 / vLLM 0.8.5 for RL. None of them can *load* Qwen3-VL — it needs
   transformers ≥ 4.57 and vLLM ≥ 0.11. Forward-porting the fork is always the expensive
   path. ERGO is the exception in the other direction: it released no training code at
   all, so there is no fork to port and the rollout loop is ours to write from the paper.
2. **Qwen3-VL is token-compatible with Qwen2.5-VL.** Identical ChatML scheme and
   identical vision token IDs (`vision_start 151652`, `vision_end 151653`,
   `image_token 151655`, `im_end 151645`) — verified against the local
   Qwen3-VL-8B-Instruct snapshot. Every hardcoded splice, token-count heuristic and
   `<image>` rewrite in these repos transfers verbatim. What does *not* transfer is
   geometry: Qwen3-VL is patch-16 / merge-2 (`preprocessor_config.json`: `patch_size
   16`, `merge_size 2`), so one visual token covers 1,024 px² against Qwen2.5-VL's 784.
   Any `max_pixels`-shaped hyperparameter is off by ~23% in token terms and has to be
   re-tuned. This lands hardest on ERGO, whose every configuration and every table
   column is written as a token budget `N×28×28`.
   **And one more that does not transfer: grounding coordinates.** Qwen2.5-VL emits
   absolute pixels in the post-`smart_resize` frame; Qwen3-VL emits a 0–1000 relative
   grid against the original image. Any repo that rescales box output by
   `original/resized` — ERGO's `map_box` does exactly this — crops the wrong place on
   Qwen3-VL while continuing to look healthy. Mini-o3 is already relative and is safe.
3. **The target has already had this treatment.** Qwen3-VL's own tech report describes
   cold-start SFT on distilled multi-turn agentic traces plus tool-integrated RL — i.e.
   the vendor ran this class of recipe at larger scale before shipping the backbone. Each
   section's gate exists to find out how much headroom survives that, cheaply, first.

**`laser_repo/` is the right starting point, and it is already here.** It is verl
`0.7.0.dev`, Qwen3-VL-aware end to end, running in the `laser` conda env
(see [laser-fork-qwen3.md](laser-fork-qwen3.md)), and it ships
`verl/tools/image_zoom_in_tool.py` (392 lines) plus the multimodal
`experimental/agent_loop/tool_agent_loop.py` that already handles tools returning images
and already emits `num_turns`. Upstream verl at that same commit (`55f651c9`) also carries
`recipe/deepeyes/` — the crop-tool GRPO recipe, configured for 1 node × 8 GPUs, our exact
job shape.

This **amends DeepEyes §4**, which recommends building on verl `0.10.0.dev`: the agent
loop and Qwen3-VL support are indeed at HEAD, but `image_zoom_in_tool.py` was deleted
from verl mainline on 2026-05-09 (`5a506cc5`, "remove … tool examples") and `recipe/`
moved out to [verl-recipe](https://github.com/verl-project/verl-recipe) on 2026-01-06
(`2bb42bae`). At HEAD the ready-made zoom tool is gone and the deepeyes recipe's own
`REQUIRED_VERL.txt` pins `bcb63864` (2026-04-17), before the deletion. Start from
`laser_repo` or pin a commit — do not `git clone` verl and expect the tool to be there.

---

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


---

# Mini-o3 on Qwen3-VL — port and training estimate

Assessment written 2026-11-07. The question: what would it cost to implement
Mini-o3 (Lai et al., ICLR 2026, [arXiv:2509.07969](https://arxiv.org/abs/2509.07969))
on Qwen3-VL-8B so the result is a comparable row next to our model?

**Verdict: ~3–4 weeks of engineering and ~3,000–6,000 A100-hours — build the agentic
eval wrapper first (§8), and do not port their fork.**

Same shape of answer as DeepEyes, with one important difference: Mini-o3's cold-start
SFT is **not optional** (their own ablation: −19.0 without it), so this is a two-stage
job, not one. The RL stage is also ~3–5× the DeepEyes bill — 200 steps against 80, and
16 rollouts against 16 but over six turns with up to twelve images in context.

Repos cloned for this assessment (scratch, not under version control):

    /home/uberger/scratch/third_party/Mini-o3        # Mini-o3/Mini-o3 @ 2c5a0de
    /home/uberger/scratch/third_party/verl-upstream  # verl-project/verl @ 0.10.0.dev
    /home/uberger/scratch/third_party/verl-recipe    # verl-project/verl-recipe (recipes after the 2026-01 move)


## 1. What the method is, mechanically

Qwen2.5-VL-7B-Instruct, two stages. **Cold-start SFT**: ~7.3k trajectories collected by
in-context-prompting a VLM with six hand-written exemplars on VisualProbe, keeping only
runs that ended correct; LLaMA-Factory, full FT, vision tower frozen, `cutoff_len 32768`,
3 epochs, lr 1e-5. **RL**: GRPO with DAPO clipping (0.2/0.3), lr 1e-6, no KL, no entropy,
256 prompts × 16 rollouts, mini-batch 32, 8192 prompt + 8192 response, rejection
sampling on, 8 nodes × 8 GPUs, ~150–200 steps.

One tool, and it is strictly richer than the DeepEyes zoom:

    <grounding>{"bbox_2d": [x0,y0,x1,y1], "source": "original_image" | "observation_i"}</grounding>

Coordinates are **relative** ([0,1], `use_relative_coordinates=True`), and `source` lets
the model crop a *previous crop*, not only the original image. The system prompt is
`verl/trainer/constants.py:TOOL_CROP_SYSTEM_PROMPT`; the observation splice is
`TOOL_CALL_CROP_MULTI_TRUN_PROMPT`; the crop/resize logic is
`verl/workers/rollout/vllm_rollout/function_tools.py:crop_image` (≤4× upscale, floor 28 px).

The three claimed ingredients, with their ablation cost on VisualProbe-hard:

| ingredient | ablation |
|---|---|
| cold-start SFT on diverse reasoning patterns | **−19.0** |
| **over-turn masking** | **−12.2** |
| hard RL data (VisualProbe) | −8.6 |
| `max_pixels=2M` (12M → 36.1% and 1.0 avg turns; 0.5M → 36.4%) | −12 |

**Over-turn masking is the actual contribution.** A rollout that hits the turn cap or
the token cap gets `M_i = 0`; advantages become `A'_i = M_i · A_i` and the GRPO objective
normalises by `Σ M_i` rather than `G`. Over-turn rollouts are neither rewarded nor
penalised — they are ignored. Without it the policy collapses into answering early;
with it, a model trained at 6 turns scales to tens of turns at test time and keeps
gaining accuracy. In their code this is `exceed_mask` → `compute_advantage(…,
exceed_mask)` → `rollout_ignore_mask` in `core_algos`. Do **not** approximate it with
DAPO overlong-filtering; the normaliser is the point.

**One fidelity wrinkle.** Training caps turns at 6 and images at 12; validation uses 12
turns. Headline numbers are Avg@32 with `val_do_sample=True`. A greedy Avg@1 number is
not comparable to their table.


## 2. What ports for free

Everything in the shared section above, plus:

* **The cold-start stage is a config change we have already made once.**
  `train/cold_start/laser_qwen3_vl_8b_revisual/train.yaml` is exactly this — somebody
  else's Qwen2.5-VL LLaMA-Factory config retargeted at Qwen3-VL-8B at `cutoff_len:
  32768`. Theirs (`sft_configs/qwen2.5-vl.yaml`) is the same shape: full FT,
  `freeze_vision_tower: true`, ZeRO-3-offload, 3 epochs. Swap `template: qwen2_vl` →
  `qwen3_vl_nothink` and the model path. `check_vision_tower_trained.py` already exists
  to confirm the freeze actually took on Qwen3-VL's deepstack visual tower.
* **The data is small and public** — see §6.
* **The judge.** Their reward wants `API_KEY` / `END_POINT`; our gateway plus
  `llm_judge.py` replaces that block outright.
* **Qwen3-VL plumbing**: `get_rope_index` (interleaved MRoPE), deepstack,
  flops counter, vLLM weight sync — all in `laser_repo` and exercised by the LASER work.


## 3. What is broken

| | Mini-o3 pin | Qwen3-VL needs |
|---|---|---|
| verl | 0.2.0.dev, forked at upstream `03e0efaa` (2025-03-04) | — |
| transformers | 4.52.0 | ≥ 4.57 |
| vLLM | 0.9.2 | ≥ 0.11 |
| torch | 2.7.0 | — |

The fork point is not guesswork: blob-matching their squashed tree against upstream verl
history puts 176 of their 224 `verl/` files byte-identical to `03e0efaa`. Their delta is
**48 files, +9,032/−285**, and it is concentrated in precisely the files verl has
rewritten since — `vllm_rollout_spmd.py` (+1,336), a new `vllm_async_engine.py` (1,037),
`ray_trainer.py` (+718), `fsdp_workers.py` (+292), `core_algos.py` (+171). A literal
rebase across twenty months of verl churn in those five files is the expensive path.

`verl/models/ttvfm_hf.py` (1,983 lines) is a vendored copy of Qwen2-VL modeling, reached
only through `models/registry.py`. It is dead weight for this port; do not carry it.


## 4. Why to skip their fork entirely

Their `vllm_rollout_spmd.py` hand-rolls what `tool_agent_loop.py` in `laser_repo` already
does: multi-turn generation, tool responses carrying images, observation-token loss
masking, and `num_turns` on the output. Their image-pad expansion hack — replace
`<|image_pad|>` with a magic token, repeat it `grid_thw.prod() // merge_length` times,
swap it back, zero the response mask over the span (`vllm_rollout_spmd.py:545-563`) — is
exactly the problem verl's multimodal tokenization path was written to solve.

So the deliverable is a set of deltas on top of the verl we already run, roughly
**800–1,200 lines**:

| | lines | notes |
|---|---|---|
| over-turn masking | ~150 | ignore-mask from `num_turns >= cap` ∪ length-exceeded, into `compute_grpo_outcome_advantage` **and the normaliser** |
| `<grounding>` tool parser | ~100 | their format is not OpenAI function-call format, which `tool_agent_loop` expects. A parser keeps their released cold-start data verbatim; rewriting the data instead is the lossy option |
| zoom-tool extensions | ~100 | `source: "observation_i"` (crop a crop) and relative [0,1] coords. `ImageZoomInTool` does neither |
| reward | ~250 | port `general_qa_tool_mc.py`; judge-extracted answer, MC + free-form, `extract_answer_tags="strict"` |
| data conversion | ~200 | their parquets → verl parquet with `tools_kwargs`, `min_pixels=40000`, `max_pixels` per §9. `deepeyes47k_preprocess.py` is the template |
| launcher | ~150 | our partitions cap at 2/4/8 h; verl `save_freq` + resume needs the wrapper our TRL launchers have |
| agentic eval wrapper | ~300 | see §7 |


## 5. Work breakdown

| | Days |
|---|---|
| Env + data: `laser` env or a sibling, download and convert all five datasets, extract images | 2–3 |
| Cold-start SFT on Qwen3-VL-8B + the usual gate. Cheap, and we have the config | 1–2 |
| RL plumbing (the table in §4, rows 1–6) | 5–7 |
| **Stabilising the first ~50 steps** — format rate, tool-call rate, truncation at 8k, 12-image OOM. Where the time actually goes; `laser-coldstart-format-traps.md` is the checklist | 3–5 |
| Eval wrapper + scoring both models | 3–5 |
| **Total** | **~14–22 days**, plus ~15 days of training wall clock, overlapped |

**The cost to flag hardest** is the same one DeepEyes §5 raises — a second training
framework — except that here it is already half-paid: `laser_repo` plus the `laser` env
exist and ran. The incremental cost is the deltas, not the framework.


## 6. Data

All public, all small, images included:

| | rows |
|---|---|
| `Mini-o3/Mini-o3-Coldstart-Dataset` | 7.27k trajectories |
| `Mini-o3/VisualProbe_train` | 2.9k |
| `Mini-o3/DeepEyes_train_4K` | 4k (a subset of `ChenShawn/DeepEyes-Datasets-47k`, which §6 of the DeepEyes section already covers) |
| `Mini-o3/VisualProbe_{Easy,Medium,Hard}` | 141 / 268 / 106 — val |

`scripts/preprocess_coldstart.py` extracts the images and emits LLaMA-Factory sharegpt
JSON. That script is usable as-is; the RL-side conversion to verl parquet is ours to
write.


## 7. Eval is the expensive part

This is where Mini-o3 differs most from DeepEyes, whose eval scripts were plain
model-agnostic API clients. Mini-o3 evaluates *inside* verl — you append
`rollout.val_n=32 val_do_sample=True trainer.val_only=True` to the training command — so
there is no standalone harness to point at a vLLM server.

Meanwhile our side (`eval_mini`, lmms-eval) scores a **single-turn** `<think>`/`<answer>`
model. A crop-tool agent needs the tool loop at inference. So a comparable row requires
an agentic wrapper on our side either way; `lmms_eval_plugin/qwen3_vl_sinkshift.py` and
the VGA wrapper in `vlm_reasoning` are the precedents. Budget 3–5 days, and build it
first (§8).

`hrbench4k_mini` / `hrbench8k_mini` are already in our mini-suite and `vstar_bench` is in
the HF cache, so the overlap with their headline table is good.


## 8. Run this gate first — 3–5 days, and it is work you need anyway

`Mini-o3/Mini-o3-7B-v1` and `Mini-o3-7B-SFT` are public. Build the agentic eval wrapper,
serve their released checkpoint, and score it on our mini-suite against Qwen3-VL-8B-Instruct
and our cold start. The backbone is wrong (Qwen2.5-VL-7B), so it does not settle the
comparable-row question — but it sizes the target for **zero training cost**, and the
wrapper is on the critical path regardless.

Their headline, for reference (Avg@32):

| | VisualProbe hard / med / easy | V* | HR-4K | HR-8K | MME-RW |
|---|---|---|---|---|---|
| Qwen2.5-VL-Instruct | 23.9 / 26.0 / 39.1 | 75.5 | 68.2 | 62.7 | 57.3 |
| DeepEyes | 35.1 / 29.8 / 60.1 | 83.3 | 73.2 | 69.5 | 64.0 |
| Mini-o3 | 48.0 / 50.4 / 67.0 | 88.2 | 77.5 | 73.3 | 65.5 |

The DeepEyes §8 caveat applies in full and then some: Qwen3-VL-8B-Thinking already scores
77.5 / 72.4 / 68.1 on V* / HR-4K / HR-8K with no tool at all, because the vendor cold-started
exactly this behaviour into the backbone.


## 9. GPU budget, and the comparability trap

Their run: 64 GPUs × ~72 h ≈ **4,600 GPU-hours** (H800-class), 256 × 16 = 4,096
trajectories per step at ~5.6 turns, ~150–200 steps ⇒ ~700k trajectories, ~23–27 GPU-s
each. Qwen3-VL-8B is ~15% more parameters but ~23% fewer visual tokens at equal pixels —
call it a wash. A100 costs ~2× H100/H800, same factor as DeepEyes §9.

| | trajectories | H-class GPU-h | **A100-h** | wall, 1 node × 8 |
|---|---|---|---|---|
| faithful (their config) | ~700k | ~4,300 | ~8,600 | ~45 days — not viable single-node |
| **scaled: n=8, batch 128, 200 steps** | ~205k | ~1,400 | **~2,800** | **~15 days** (~4 days on four nodes) |
| cold-start SFT | 7.3k × 3 ep | ~10–20 | ~20–40 | ~4 h |
| eval, Avg@32 agentic, both suites | — | — | ~30–60/pass | — |

Budget smoke + one real run + one redo: **~3,000–6,000 A100-hours**. Multi-node Ray on
SLURM is the real unlock (four nodes turns 15 days into 4) and none of our launchers do
it — add 1–2 days if we want it. The cluster blocker in DeepEyes §9 applies unchanged:
`sinfo` still reports zero GPU nodes on `oci-nrt-cs-001`, so this runs on the A100
cluster.

**The comparability trap, which has to go in the caption.** Our runs are LoRA GRPO,
8 rollouts × ≤1024 tokens, single-turn — ~190k completions over 3,990 steps. Mini-o3
*scaled down* is still full-parameter, 8 rollouts × 8k tokens × 6 turns with up to twelve
images: roughly **10× our RL token budget**. And it attacks precisely the visual-search
half of the benchmark table where the attention rewards win and the symbolic half is
where they cost ([[benchmark-table-splits-search-vs-symbolic]]). A reviewer will read an
unqualified loss there as the saliency reward losing, when part of it is spend. Either
budget-match deliberately or say so.


## Sources

* [arXiv:2509.07969](https://arxiv.org/abs/2509.07969) — the paper (ICLR 2026)
* [Mini-o3/Mini-o3](https://github.com/Mini-o3/Mini-o3) — their code
* [mini-o3.github.io](https://mini-o3.github.io/) — project page
* [Mini-o3 models](https://huggingface.co/Mini-o3/models) / [datasets](https://huggingface.co/Mini-o3/datasets)
* [verl-project/verl](https://github.com/verl-project/verl) and [verl-recipe](https://github.com/verl-project/verl-recipe) — the framework, and where `recipe/deepeyes` went


---

# ERGO on Qwen3-VL — port and training estimate

Assessment written 2026-11-07. The question: what would it cost to implement ERGO
(Lee, Shin et al., ICLR 2026, [arXiv:2509.21991](https://arxiv.org/abs/2509.21991))
on Qwen3-VL-8B so the result is a comparable row next to our model?

**Verdict: ~14–18 person-days and ~400–1,000 A100-hours — the cheapest of the three —
but run the 1-day gate in §8 FIRST, because ERGO's entire headroom is the
low-resolution penalty, and Qwen3-VL may not have one.**

Two differences from the sections above set this one apart. First, **they released no
training code**, so there is no fork to port and no stack to forward-port — the method
is reimplemented from the paper either way. Second, **ERGO's rollout is two fixed
turns, not an agent loop**, which means it is the one baseline here that fits our
existing TRL stack without the correctness risk that sends the others to verl.

Repos cloned for this assessment (scratch, not under version control):

    /tmp/ERGO                 # nota-github/ERGO @ main -- 627 lines, inference and eval only
    /tmp/ergo_paper/ergo.txt  # arXiv:2509.21991v2, HTML -> text


## 1. What the method is, mechanically

Qwen2.5-VL-7B-Instruct, GRPO, **250 steps**, global batch 128, 8 rollouts per example
(the appendix prose says 16 — see §10), lr 1e-6, AdamW, 4×H100, **~150 GPU-hours**.
No cold start: the authors picked Qwen2.5-VL explicitly because its referring-detection
ability makes grounding work without one.

A rollout is **two fixed turns**:

1. The image is downsampled to a token budget (`N×28×28`, `N ∈ {640, 1280, 16384}`) and
   the policy emits `<think>…</think><zoom>[x1, y1, x2, y2]</zoom>`.
2. That box is cropped from the **original, full-resolution** image, `smart_resize`d
   back under the same budget, and appended as a fresh user turn. The policy answers
   from `[downsampled image, turn-1 output, crop, question]` as
   `<think>…</think><answer>…</answer>`.

The prompts are released verbatim in `src/ergo/const.py` and the crop math in
`src/ergo/utils.py` (`crop_with_zoom`, `map_box`, `parse_zoom_bbox_from_text`).

The reward is four binary terms:

    R = r_TCE + r_acc + r_format
    r_TCE    = 1.0·r_region + 0.5·r_box

    r_region = 1 iff a FROZEN reward model, shown ONLY the crop and the question,
               answers correctly.                        <- the contribution
    r_box    = 1 iff Area(crop) / Area(original) <= 0.6
    r_acc    = 1 iff the policy's own turn-2 answer is correct
    r_format = 1 iff BOTH turns match their tag structure

Everything scores 0 if turn 1 produces no parseable box.

The one real idea is `r_region`'s **withholding of the original image from the reward
model**. DeepEyes and PixelReasoner show the RM both the full image and the crop, so the
RM can answer from the full image and the crop is never forced to be self-contained.
ERGO shows it the crop alone, so the policy is pushed to crop regions that *carry the
answer* — which under downsampling means cropping context it can only infer (the straw
near the coffee cup), not just the target it can see. That is also why the method is
framed as "reasoning-driven perception" rather than better grounding.

`r_box`'s γ = 0.6 comes from the GT-box area distribution in TreeVGR / VisCoT / V* / VGR
(their Fig. 3). We can re-derive it on our own training set in an hour —
`build_grpo_sets.py` already carries `_boxes_abs` per row and calls `union_bbox` on them.


## 2. What they released, and what they did not

`nota-github/ERGO` is **627 lines of Python, inference and evaluation only**:

    src/ergo/const.py     10    the three prompts
    src/ergo/infer.py    169    single-image HF demo of the two-stage loop
    src/ergo/eval.py     283    async OpenAI-API client against a vLLM server
    src/ergo/utils.py    165    smart_resize, box parse, crop, box -> original mapping

**No training code, no reward implementation, no data pipeline, no config.** The repo
credits [VLM-R1](https://github.com/om-ai-lab/VLM-R1) as the RL codebase, but VLM-R1 is
92 files of single-turn open-r1-multimodal GRPO (rec / gui / ovd tasks) with no
multi-turn path at all — so ERGO's two-turn rollout was custom and is unpublished. The
repo has one issue, closed, about `uv`. There is nothing to lift and nobody else has
reimplemented it.

What *is* released and genuinely useful: the exact prompts, the exact crop/resize math,
the exact eval protocol, every hyperparameter in their Table 7, and the public
**`nota-ai/ERGO-7B`** checkpoint — so the method can be *measured* in half a day even
though it cannot be trained from their code.


## 3. What is broken

Nothing dependency-shaped, for once — there is no pinned stack. Three things break
anyway, and **all three are silent**:

* **Coordinates changed scheme.** Per the shared section above, Qwen3-VL emits 0–1000
  relative coordinates against the original image. `utils.py` carries
  `width_scale = width / resize_w` and a `map_box()` built for Qwen2.5-VL's
  absolute-in-resized-frame convention. Replace it with `x * w / 1000` and say so in
  `QUESTION_TEMPLATE1`. Get it wrong and every crop lands in the wrong place, the
  region reward collapses into the box-area reward, and training looks fine.
* **The `28` is a `32`.** Qwen3-VL is patch-16 / merge-2, so `IMAGE_FACTOR = 28` in
  `infer.py` / `eval.py` / `utils.py` must become 32, and every `N×28×28` budget must be
  restated as `N×32×32` to mean the same *token* count. Copy their numbers literally and
  `1280×28×28` buys 980 visual tokens, not 1,280 — a 23% different operating point on
  every row of an efficiency table.
* **Format is not free.** ERGO needs `<think>` in both turns, and
  `build_coldstart_data.py:53` records Qwen3-VL-8B-Instruct at **format 0.000 under every
  prompt variant tried** — which is why our cold start exists. If turn 1 is never
  well-formed, every reward in the group is 0, the advantage is 0, and GRPO has no
  gradient at all. Start from
  `checkpoint/coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged`, which already
  emits `<think>…</think><answer>…</answer>` and is the right starting point for
  comparability anyway. Whether it will also emit `<zoom>` is Gate B in §8.

Two further facts about our side, not theirs:

* **Our training images are capped at 512 px.** `build_grpo_sets.py:109` and
  `trl/grpo_vlm_qwen3.py:122` both set `MAX_IMAGE_SIDE = 512`. Training ERGO on 512 px
  images trains a no-op — there is no high-resolution detail for the crop to recover.
  The set must be rebuilt at native resolution and the trainer cap raised, which is what
  drives the step cost in §9.
* **Stay on vLLM 0.11.0.** vllm-project/vllm#29595 reports corrupted Qwen3-VL grounding
  output above 0.11.0, and `saliency_r1_qwen3_vllm` is on exactly 0.11.0.


## 4. Two routes, and ERGO is the one that can stay in TRL

### Route A — our TRL trainer (recommended if ERGO is built alone)

The key observation: **ERGO's two turns can be two independent training rows with the
same advantage.**

    row 1:  prompt = [sys, user(low-res image, Q1)]                     completion = o_region
    row 2:  prompt = [sys, user(low-res image, Q1),
                      assistant(o_region), user(crop, Q2)]              completion = o_acc

Both carry `Â_g`, computed once per rollout from the four rewards. The gradient is
identical to the concatenated-trajectory form, because turn 1's text is a *prompt* for
turn 2 and carries no gradient there. So there is **no observation splicing, no
observation-token loss mask, and no hand-rolled position ids** — each row is a normal
two-image conversation that the Qwen3-VL processor builds itself and the model MRoPEs
itself. That is precisely the bug class behind DeepEyes §3's `parallel_env.py:282`, and
here it cannot arise. It is the reason this baseline, unlike the other two, does not
need a second training framework.

What does need building in `trl/grpo_trainer_qwen3.py`:

* **Two images per row.** The trainer is single-image throughout
  (`_signature_columns = ["prompt", "image"]`, line 1179) and `split_pixel_values_by_grid`
  assumes one `image_grid_thw` row per sample. Generalising to a per-row image count is
  contained, but it touches the ZeRO-3 path.
* **Multi-image in the vLLM server.** `trl/scripts/vllm_serve.py:572-578` takes one
  base64 image per prompt and builds `multi_modal_data={"image": img}`. We own that
  file; it needs `list[list[str]]`.
* **A two-phase rollout** in `_generate_and_score_completions`: generate turn 1 for all
  rollouts, crop in PIL on the main process, build turn-2 prompts, generate again at
  `n=1`, emit 2× rows.
* **Advantage broadcast**: rewards per rollout, each advantage repeated twice. One
  wrinkle to state in the launcher — a sequence-mean `loss_type` weights the two turns
  equally rather than by length.
* **Length caps.** `--max_prompt_length 2048` cannot hold turn 2 at `N=1280`
  (1280 + 1280 visual tokens + text + turn-1 output ≈ 3,300). Both caps go up, with the
  memory consequence in §9.

### Route B — verl, i.e. `laser_repo` (recommended if DeepEyes or Mini-o3 is also built)

If either section above is executed, ERGO is **incremental, not a second project**: same
framework, same data, same backbone, and ERGO's zoom is `image_zoom_in_tool` pinned to
`max_turns=2`. Per the shared section, `laser_repo` already carries that tool and the
multimodal `tool_agent_loop.py`. The delta is the four reward functions, the vision
reward-model client, and a two-turn cap — **4–7 days on top**.

Standing up verl *only* for ERGO is not worth it: §5's 14–18 days would become ~20–25,
to buy insurance against a bug class Route A does not have.

**Decision rule: if DeepEyes or Mini-o3 is on the roadmap, do verl once and get all
three. If ERGO is the only baseline, keep it in TRL.**


## 5. Work breakdown (Route A)

| | Days |
|---|---|
| **Gate A** — the low-resolution penalty on Qwen3-VL-8B (§8). Stops here if it is small | 1 |
| **Gate B** — `<zoom>` emission and the 0–1000 coordinate check, on Qwen3-VL-8B-Instruct and on our cold start, mirroring `coldstart_gate.py` | 1 |
| Benchmarks: add `vstar_bench`, `hrbench_4k`, `hrbench_8k` (+ `spatialtreebench`) to `eval_mini/benchmarks.py`; baseline our existing arms on them | 1–2 |
| High-resolution training data: raise `MAX_IMAGE_SIDE`, rebuild the V*/visdrone rows; or convert DeepEyes' V* + ArxivQA parquets (§6) | 1–2 |
| Trainer: two images per row (processor, `split_pixel_values_by_grid`) + multi-image `vllm_serve.py` | 2 |
| Trainer: two-phase rollout, PIL crop, 2× rows, advantage broadcast, W&B logging of zoom-parse rate and crop area | 2–3 |
| `trl/rewards/ergo_rewards.py`: region / box / accuracy / format, the vision-RM client, and a `test_ergo_reward_cpu.py` gate | 2 |
| lmms-eval `ergo_qwen3_vl` two-stage chat wrapper (§7) | 1–2 |
| Launcher + **debugging the first real rollouts**: format rate, zoom-parse rate, OOM at 2×1280 visual tokens, prompt truncation | 3–4 |
| **Total** | **~14–18 days** |

**If Gate B fails** — the cold start will not emit `<zoom>` — add **3–5 days** for a
two-turn cold start: ~2–5k SFT rows in ERGO's format, turn-1 boxes taken from the GT
annotations already in `cold_data/grpo_sets`. We have the whole cold-start pipeline, so
that is data work, not new machinery. Note that Mini-o3 §1 pays this cost unconditionally
(−19.0 without it); for ERGO it is contingent, which is part of why this section is
cheaper.


## 6. Data

Two options, answering different questions.

**Paper-faithful**: `ChenShawn/DeepEyes-Datasets-47k`, which DeepEyes §6 already
describes — ERGO reuses it deliberately, "to isolate the contribution of our method from
dataset curation effects". ERGO takes the V* and ArxivQA halves
(`data_0.1.2_visual_toolbox_v2.parquet`, `data_v0.8_visual_toolbox_v2.parquet`) and not
the ThinkLite math half. Reachable from the login node (checked).

**Comparability-faithful**: our own GRPO sets, rebuilt at native resolution. Same rows
our model trained on, so the ERGO arm and our arm differ only in method — the better
choice for a table in our paper. The cost is one rebuild of `build_grpo_sets.py` with
`MAX_IMAGE_SIDE` raised, plus the disk (V*/visdrone at native resolution, tens of GB;
`cold_data` already holds 201 GB).

Note what ERGO's training does **not** need: GT boxes. `r_region` is supervised by the
reward model's answer, not by an annotation. Boxes are needed only to re-derive γ.


## 7. Eval

The task files are already here. In
`/home/uberger/scratch/research/lmms-eval/lmms_eval/tasks`: `vstar_bench`, `hrbench`,
`mme_realworld`, `spatialtreebench` all exist. VisualProbe (Mini-o3's) does not —
optional, and Mini-o3 §6 covers where it lives.

Three things need doing:

* **Put them in the table.** None of V*, HR-Bench or TreeBench is in
  `eval_mini/benchmarks.py` — only `mmerealworld`. Worth a day regardless of ERGO: our
  effect lives in the visual-search half ([[benchmark-table-splits-search-vs-symbolic]])
  and these are the canonical visual-search benchmarks.
* **An `ergo_qwen3_vl` wrapper** running the two-stage loop at inference. Their
  `eval.py` is the reference but is a standalone async OpenAI client, not an lmms-eval
  model. Cheaper than Mini-o3 §7's wrapper, because the loop is fixed at two turns with
  no tool-call parsing; `thyme.py` in our clone is the closest precedent, and
  `qwen3_vl_vga.py` / `qwen3_vl_sinkshift.py` are the house style.
* **A pixel-constraint sweep**, which is configuration rather than code:
  `launch_lmms_eval_job.sh --max-pixels` exists and already suffixes the output
  directory when set explicitly, exactly so sweeps do not collide. Two cautions. The
  default in `_chat_base.py:40` is `IMAGE_KWARGS = {"max_pixels": 512*512}` — **256
  visual tokens at factor 32**, far below even ERGO's smallest setting, so nothing here
  is at ERGO's operating point today. And *our* model must be re-scored at the same caps
  or the comparison is not matched (§10).


## 8. Run this gate first — 1 day, and it is likelier than the other two to end the job

ERGO's headroom *is* the gap its Table 1 reports for Qwen2.5-VL-7B on V*:

| Pixel const. | Qwen2.5-VL-7B on V* |
|---|---|
| 16384×28×28 | 77.0 |
| 1280×28×28 | 64.9 |
| 640×28×28 | 56.5 |
| 640×28×28 **+ GT crop appended** | **77.0** |

The method recovers the 12–20 points that downsampling destroys — and the last row is
the ceiling it is chasing. **If Qwen3-VL-8B does not lose those points, there is nothing
to recover and the arm is pointless.** There is specific reason to doubt it does:
DeepStack fuses multi-level ViT features precisely to hold detail at fewer tokens, and
per DeepEyes §8 the backbone already had tool-integrated RL with an `image_zoom_in_tool`
in its own post-training.

**The gate**: `vstar_bench` + `hrbench_4k` + `hrbench_8k` on Qwen3-VL-8B-Instruct and on
our cold start, at `--max-pixels` ∈ {640, 1280, 16384}×32×32 (note the 32, per §3), plus
a fourth arm with the GT crop appended — a ~40-line task variant, since V* and HR-Bench
both ship boxes. One day, ~10–20 A100-hours, and it reproduces their Table 1 on our
backbone.

Read it as:

* **gap ≥ 10 points** — the method has room; proceed.
* **gap 3–10 points** — proceed only if the claim being made is the efficiency axis
  (tokens, latency), not accuracy.
* **gap < 3 points** — stop. "ERGO's premise does not hold on this backbone" is a
  publishable sentence and costs one day rather than a month.

**A second, cheaper reference while the gate runs.** `nota-ai/ERGO-7B` is public: serve
it and run their `src/ergo/eval.py` against our benchmark set. The backbone is wrong
(Qwen2.5-VL-7B) so it does not settle the comparable-row question, but it sizes the
target for zero training cost and it exercises the two-stage loop before any of ours
exists. Half a day. Same logic as Mini-o3 §8, at a tenth the price.


## 9. GPU budget

Two independent anchors, which agree:

* **The paper**: ~150 H100-hours for one 250-step run ⇒ **~300 A100-hours** at the same
  2× used in DeepEyes §9.
* **Token-volume scaling from our own runs**: a rollout is ~1,950 tokens for turn 1 and
  ~3,320 for turn 2 at `N=1280`, so ~5,300 per rollout against our ~1,360, and 128
  rollouts per step against our 48 — **~10× our token volume per step**. From ~22.5 s/step
  on 8 H100s ([[grpo-step-time-is-mostly-fixed-cost]]) that is roughly 140–450 s/step on
  8 A100s, i.e. **80–250 A100-hours** for 250 steps.

Take **~200–350 A100-hours per run**; budget 2–3 runs (smoke, real, fix) ⇒
**~400–1,000 A100-hours**, plus **~100–200** for the eval sweeps, which are not cheap
here: two forward passes per document × 4 benchmarks × 3 pixel caps × the arms compared.

That is **half the DeepEyes bill and a fifth of Mini-o3's**, and the reason is rollout
shape: DeepEyes pushes `max_response_length=20480` over up to 5 generate rounds and
Mini-o3 six turns with twelve images; ERGO is two rounds of ≤512 tokens.

**The reward model is the one open cost.** The paper uses frozen Qwen2.5-VL-72B-Instruct
— ~145 GB in bf16, so 2 A100-80GBs at best and realistically 4, as a sidecar for the
whole run (3 runs × 4 GPUs × ~8 h ≈ 100 A100-hours). Three options:

1. **Gateway** — `trl/rewards/openai_rewards.py` already calls the NVIDIA gateway and
   its default `azure/openai/gpt-4o-mini` is vision-capable. Zero GPUs, and ~128 image
   calls per step sits well inside the existing `ThreadPoolExecutor` judge path.
   **Unverified**: that the gateway accepts image parts, and that a ~1 MP crop fits its
   payload limit. Ten minutes to check, and it is the first thing to check.
2. **Local Qwen2.5-VL-72B** — paper-faithful, 4 A100s, mirrors `serve_grounding_dino.sh`.
3. **Local Qwen3-VL-32B** — the same-family analogue, 1–2 A100s. Cheapest local option
   and arguably the better control, since the RM then shares the policy's visual
   front-end.

The cluster blocker in DeepEyes §9 applies unchanged: `sinfo` still reports zero GPU
nodes on `oci-nrt-cs-001` ([[gpu-partitions-can-be-empty-of-hardware]]), so this runs on
the A100 cluster — which is where the 2× above comes from. Unlike Mini-o3, nothing here
wants multi-node Ray; 8 GPUs on one node is the whole job.


## 10. Fidelity notes, and the comparability trap

* **Rollouts per sample: 8 or 16?** Table 7 says 8; Appendix B's prose says "Sixteen
  rollouts were sampled per training example". Pick one and say which.
* **"Half of each mini-batch was allocated to longer rollouts to avoid VRAM
  bottlenecks"** (Appendix B) is not reproducible from the paper. Ignore it and say so.
* **γ = 0.6** was derived from *their* data's GT-box areas. Re-derive on ours — an hour
  with `union_bbox` over `cold_data/grpo_sets` — and report both.
* **"No cold start" is a claim about Qwen2.5-VL**, not a property of the method. If Gate
  B forces one on Qwen3-VL, that is a finding worth a sentence, not a failure — and it is
  the same observation DeepEyes §8 makes about *its* "emerges without SFT" framing.
* **The comparability trap**, the ERGO-shaped version of Mini-o3 §9's. ERGO is an
  *efficiency* method: its claim is accuracy at a reduced token budget, and its headline
  is "+4.7 on V* at 23% of the vision tokens". Our model has never been evaluated under a
  pixel cap at all — the default is 512×512, 256 tokens. An honest table reports accuracy
  **at a matched pixel budget** for both arms, plus vision-token counts and latency, or
  it is comparing a method tuned for a budget against one that never saw it.


## Sources

* [arXiv:2509.21991](https://arxiv.org/abs/2509.21991) — the paper (ICLR 2026)
* [nota-github/ERGO](https://github.com/nota-github/ERGO) — inference and eval only, Apache 2.0
* [nota-ai/ERGO-7B](https://huggingface.co/nota-ai/ERGO-7B) — the public checkpoint
* [om-ai-lab/VLM-R1](https://github.com/om-ai-lab/VLM-R1) — the credited RL codebase; single-turn only
* [Qwen3-VL blog](https://qwen.ai/blog?id=qwen3-vl) and `cookbooks/2d_grounding.ipynb` — the 0–1000 coordinate change

---

# Monet on Qwen3-VL — port and training estimate

Assessment written 2026-11-07. The question: what would it cost to implement Monet
(Wang, Shi et al., CVPR 2026, [arXiv:2511.21395](https://arxiv.org/abs/2511.21395)) on
Qwen3-VL-8B so the result is a comparable row next to our model?

**Verdict: ~31–44 person-days (~6–9 weeks) and ~1,500–3,000 A100-hours — the most
expensive of the four — and the published numbers already say Qwen3-VL has closed most
of the gap Monet opens. Run the 2–3 day gate in §8 first.**

Three things set this section apart from the three above. **There is no tool**, so none
of the agent-loop machinery applies and `laser_repo` buys nothing — the RL half is
EasyR1, not verl. **The method edits the forward pass**, a ~950-line delta inside
`Qwen2_5_VLModel.forward`, which is the one thing none of the other three require. And
**the SFT is not optional**: three stages plus two full precompute passes over 125K
samples produce the checkpoint RL consumes, so unlike DeepEyes you cannot start at the
interesting part.

Repos cloned for this assessment (scratch, not under version control):

    /home/uberger/scratch/ref/Monet    # NOVAglow646/Monet @ 08939998 (2026-03-19)
    /home/uberger/scratch/ref/EasyR1   # hiyouga/EasyR1 @ 602a128, for the RL rebase
    /home/uberger/scratch/ref/_ref/    # the exact upstream files their forks diverge from


## 1. What the method is, mechanically

Qwen2.5-VL-7B, **vision tower frozen** (`src/main.py:127`), LLM fully fine-tuned,
8 GPUs, batch 1 × grad-accum 16, DeepSpeed ZeRO-2, lr 1e-5. Three SFT stages, then RL:

1. **Warm-up (stage 1)** — ordinary interleaved image-text CoT SFT on Monet-SFT-125K.
   3,885 steps; the paper calls this 3 epochs, `script_examples/sft_stage1.sh` passes
   `--epochs 4` (§10).
2. **Target latents (stage 2)** — a student emits `latent_size` ∈ {8, 10, 12} continuous
   embeddings after each auxiliary image, behind a custom 4D mask so *only the latents*
   can see that image. Loss = NTP + 2.0 · an all-layer alignment between the observation
   tokens' hidden states with and without the auxiliary image, back-propagated **only
   through the latent vectors** (`torch.autograd.grad` on them, `src/trainer.py:11`).
3. **Latent generation (stage 3)** — reinitialise from stage 1 and train the model to
   produce those same latents with no auxiliary image present. Loss = NTP + 2.0 ·
   all-layer alignment to the stage-2 latents.
4. **VLPO** — GRPO where latents get a surrogate log-prob `−‖z − μ‖² / 2σ²` (σ = 10)
   spliced into `log_probs` at latent positions, so the existing ratio machinery
   optimises them. 3.2K Thyme-RL prompts, 1 epoch, batch 64 × 8 rollouts, temperature
   0.5, KL 0.01, max response 4,096, online difficulty sampling (keep only examples
   whose rollout accuracy falls in (0, 0.6)), Gemini-2.5-Pro as judge.

The latent mechanism itself is two lines of idea: **the last decoder layer's hidden
state is fed back as the next input embedding**, for N steps, triggered by the model
emitting `<abs_vis_token>`. Everything else is the supervision needed to make that
hidden state mean something.

Stages 2 and 3 each require a **full forward pass over all 125K samples first** to dump
their alignment targets to disk (`src/precompute_teacher_reps.py`,
`src/precompute_teacher_latents.py`). That is five GPU jobs, not two.

Reward at RL time is `accuracy + 0.1 · format` with a length penalty and an API-based
repetition penalty. Deliberately **no reward on using latents** — "to prevent
indiscriminate invocation".


## 2. What they released, and how big the delta actually is

Everything: SFT code, RL code, both vLLM patches, the data, and three checkpoints
(stage 1 / stage 3 / final). 26,367 lines of Python. But almost all of it is vendored
copies of upstream files, and the *actual* Monet delta is small. Measured by diffing
against the exact upstream versions they pin:

| file | lines | delta vs upstream | portability |
|---|---|---|---|
| `monet_qwen_model/modeling_qwen2_5_vl_monet.py` | 2,583 | **+954 / −105** vs HF 4.54.0 | **the hard part** |
| `RL/monet_models/transformers/monet_modeling_qwen2_5_vl.py` | 2,070 | +48 / −13 vs HF 4.51.3 — of which **three lines are real**: `inputs_embeds[0, latent_poss] = latents` at `:1858` | trivial |
| `RL/monet_models/vllm/monet_gpu_model_runner.py` | 2,001 | **+234 / −9** vs vLLM 0.8.5 | model-agnostic |
| `inference/vllm/monet_gpu_model_runner.py` | 2,929 | **+87 / −12** vs vLLM 0.10.0 | model-agnostic |
| `src/` — their SFT harness, subclassing `trl.SFTTrainer` | ~2,400 | their own | mostly agnostic |
| `RL/verl` — an EasyR1 fork | ~10,000 | ~100 lines mention monet/latent across 11 files, **plus** ~1,000 lines of `SampleHashServer`, online difficulty sampling, API judge and repetition penalty that upstream EasyR1 does not have | medium |

**The latent machinery is backbone-agnostic.** Both vLLM runners key off two token IDs
read from environment variables and never mention Qwen anywhere. VLPO is ~30 lines in
`RL/verl/workers/actor/dp_actor.py` (`compute_latent_log_probs` at `:135`, the splice at
`:302`) plus `compute_grpo_latent_advantage` at `core_algos.py:258`. The ~260 lines of
alignment-loss helpers (`compute_affine_projectors`, `affine_subspace_alignment_loss`,
`_svd_select_U`) are pure tensor math over hidden states.

What is *not* agnostic is the SFT forward. `Qwen2_5_VLModel.forward` is rewritten from
`:1689` onward into a Python loop that drives `self.language_model(...)` segment by
segment with a KV cache, one extra call per latent token, at batch size 1 — their own
comment at `:1745` reads "we only use batch_size=1 in our training. Larger batch_size
has not been tested."

One piece is dead on arrival and should be deleted rather than ported: the modified
`Qwen2_5_VLAttention.forward` exists solely to serve `collect_emphasize_attn`
(`:948`), an auxiliary attention loss that nothing in the repo ever enables — one of the
paper's own "early unsuccessful attempts". Dropping it means **the attention module is
never touched**, which is what keeps interleaved MRoPE out of the blast radius.


## 3. What ports for free

Checked against the local `checkpoint/coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged`
and against the installed vLLM, not assumed:

* **Token compatibility**, per the shared section above. Monet's text manipulation is
  built on `<|vision_start|><|image_pad|><|vision_end|>` splicing and an
  `<|im_start|>assistant` answer-start pattern, all identical on Qwen3-VL. So
  `replace_latent_placeholder_with_img_pad`, `add_latent_pad_after_auxiliary_img` and
  the two 4D mask builders in `src/utils.py` (`build_4d_attn` at `:487`,
  `build_4d_attn_wo_helper_images` at `:764`) transfer as written.
* **No grounding coordinates anywhere.** The auxiliary images in the SFT data are
  pre-cropped files on disk (`ReFocus/images/1_1.jpg` and friends), not boxes the model
  emits. The 0–1000 coordinate change that breaks ERGO cannot touch this one.
* **The 4D mask passthrough survives transformers 5.x.** `_preprocess_mask_arguments`
  in `masking_utils.py` documents that an already-4D mask "is returned as-is". Their
  dict form `{'full_attention': …}` becomes a bare tensor; that is the whole change.
* **Both vLLM hook sites exist at 0.11.0**, in our own `saliency_r1_qwen3_vllm` env:
  `self.inputs_embeds.gpu[:num_scheduled_tokens].copy_(…)` at `gpu_model_runner.py:1984`
  (where the pending hidden state is written over the next token's embedding) and
  `sample_hidden_states = hidden_states[logits_indices]` at `:2330` (where it is
  captured). vLLM 0.11 even ships a native `prompt_embeds` path writing into the same
  buffer at `:2014`.
* **Current EasyR1 already supports Qwen3-VL** — `verl/models/transformers/qwen3_vl.py`,
  `monkey_patch.py` dispatching on `qwen3_vl` / `qwen3_vl_moe`, and
  `examples/qwen3_vl_4b_geo3k_grpo.sh`. Commit `a354cdc`, "[deps] Allow transformers
  <5.0.0 for Qwen3-VL". Same finding as DeepEyes §4: the Qwen3-VL work is upstream, the
  fork is what is old.
* **The data is self-contained**, see §6.
* **The checkpoint is public and ordinary** — `NOVAglow646/Monet-7B` declares
  `architectures: ["Qwen2_5_VLForConditionalGeneration"]`, `vocab_size: 151670`, no
  `auto_map`, no custom code. That is what makes §8's gate cost days rather than weeks.


## 4. What is broken, and the one silent trap

**Pin both environments at transformers 4.57.6, not 5.x.** Their SFT code targets
4.54.0 and current EasyR1 requires `>=4.54.0,<5.0.0`; Qwen3-VL landed in 4.57.0
(verified: the `qwen3_vl` modeling file is present at tag `v4.57.0`). 4.57.6 is the only
point that satisfies all three. Our working envs are at `transformers 5.13.0.dev0`, so
this is a third pinned env, not a reuse — but porting a hand-modified 2,583-line
modeling file across the 4.x → 5.x boundary is work that buys nothing.

Qwen3-VL-specific breaks in the segmented SFT forward, in descending order of how
expensive they are to find late:

* **DeepStack, and it is silent.** `Qwen3VLTextModel.forward` (installed transformers,
  `modeling_qwen3_vl.py:823`) injects ViT features into the first *k* decoder layers at
  the rows flagged by `visual_pos_masks`, via `_deepstack_process` at `:849`. Monet calls
  the language model **once per segment**, so each call needs `visual_pos_masks` sliced
  to `[s, e)` *and* the matching rows of every `deepstack_visual_embeds[l]` selected
  cumulatively across segments. Get the offsets wrong and the injection lands on the
  wrong tokens; training runs, loss falls, and the model is wrong. This is the Qwen3-VL
  analogue of DeepEyes §3's `parallel_env.py:282`.
* **`position_ids` has four rows, not three** (text, t, h, w). Monet's
  `position_ids[:, b:b+1]` slicing is dimension-agnostic and survives, but
  `get_rope_index` / `compute_3d_position_ids` / `rope_deltas` have different signatures
  and `Qwen3VLTextModel` splits off `text_position_ids = position_ids[0]` itself.
* **`get_image_features` returns a tuple** on Qwen3-VL — `(image_embeds,
  deepstack_image_embeds)`. Monet does `torch.cat(self.get_image_features(...), dim=0)`
  at `:1603`, which will silently concatenate the wrong thing.
* **The `28` is a `32`**, exactly as in ERGO §3, and it is wired deeper here.
  `src/utils.py:299` hardcodes `divisor=28` and every budget in the codebase is written
  `N*28*28` — `2000*28*28` for stages 1 and 3, `1500*28*28` global and `1280*28*28`
  per image for stage 2, `1000*28*28` / `500*28*28` in the latent precompute. Two
  consequences, not one: copying the numbers literally buys ~23% fewer visual tokens
  than they trained on, *and* `divisor=28` emits image dimensions that are not multiples
  of 32, so Qwen3-VL's `smart_resize` re-rounds them and the `new_sizes` the collate
  re-applies at `src/main.py:264` go stale.
* **vLLM async scheduling must be off.** The latent state machine reads
  `valid_sampled_token_ids` synchronously inside `execute_model`; at 0.11 that is behind
  `if not self.use_async_scheduling:` (`gpu_model_runner.py:2139`), with an
  `AsyncGPUModelRunnerOutput` path that returns token ids later. Assert it rather than
  assume it.
* **Stay on vLLM 0.11.0** for the same reason ERGO §3 gives.

**The silent trap.** `README.md:212-213` hardcodes `LATENT_START_ID=151666` and
`LATENT_END_ID=151667`. On Qwen3-VL those IDs are **`</tool_response>`** and
**`<think>`**. Their training code is clean — it derives the IDs from
`tokenizer.add_tokens(...)`, which on Qwen3-VL lands the three new tokens at
151669/151670/151671 (free, below `vocab_size: 151936`, so no embedding resize is
needed). But the eval recipe is a copy-pasted `sitecustomize.py` carrying those two
literals, and anyone who follows the README gets a model that flips into latent decoding
whenever the policy emits `</tool_response>`. **Set both from the tokenizer, never from
the README.**


## 5. Work breakdown

| | Days |
|---|---|
| **The gate** (§8) — serve `Monet-7B` on their own vLLM 0.10.0 pin, score it against Qwen3-VL-8B-Instruct and our cold start. Stops here if the headroom is gone | 2–3 |
| Two conda envs pinned at transformers 4.57.6: SFT (trl, deepspeed ZeRO-2) and RL (EasyR1 + vLLM 0.11). flash-attn on A100 is the usual risk ([[laser-env-gotchas]]) | 3–4 |
| Port the ~950-line SFT modeling delta onto Qwen3-VL: DeepStack slicing, 4-row position ids, the `get_image_features` tuple, the 28→32 restatement; delete the dead attention path | 5–8 |
| **Equivalence tests**, which are the only defence against §4's silent breaks: `latent_mode=False` must reproduce stock Qwen3-VL logits, and the segmented forward must equal one full forward on the same input | 2 |
| Data/template layer: `src/utils.py` mask builders, the new special tokens, a `test_monet_masks_cpu.py` in this repo's shape | 2–3 |
| Precompute passes + **sharded** storage (§6) | 2 |
| RL: graft EasyR1's `models/transformers/qwen3_vl.py` + `monkey_patch.py` into their fork, bump vLLM 0.8.5 → 0.11 and re-apply the 234-line runner patch at the two anchors, swap the Gemini judge for ours | 5–8 |
| Launchers in this repo's shape — SLURM, 4 h chunks, autoresume — for **five** jobs, not one | 3 |
| Eval: either VLMEvalKit + their `sitecustomize`, or port the 87-line inference patch to our vLLM 0.11 and write a `monet_qwen3_vl` lmms-eval wrapper (§7) | 3–4 |
| **Debugging the first real runs** — where the time actually goes | 6–10 |
| **Total** | **~31–44 days** |

**The cost to flag hardest** is not any single line above: it is that this adds *two*
frameworks. Their SFT harness subclasses `trl.SFTTrainer`, so it is at least the same
family as ours, but it is a separate trainer from
`trl_repo/trl/trainer/grpo_trainer_qwen3.py` with its own collate, its own loss and its
own DeepSpeed config. The RL half is EasyR1 — a different framework from the verl that
DeepEyes, Mini-o3 and ERGO Route B all share. **Monet is the one baseline here that
reuses nothing from the other three.** If the roadmap is "build several baselines", it
should be built last or not at all.

**The cheap partial.** Stage 1 alone is an ordinary interleaved-CoT SFT on 125K rows:
no modeling fork, no latents, no precompute, runs on the cold-start pipeline we already
have. ~3–5 days and ~150–350 A100-hours, and it reproduces the paper's "+ vanilla SFT"
row — which already carries two thirds of their V* gain (§8). That isolates how much of
Monet is the dataset and how much is the latent space, for a twentieth of the price, and
it is worth doing even if the full arm is cancelled.


## 6. Data and disk

**The data is the easiest of the four.** `NOVAglow646/Monet-SFT-125K` is **13.5 GB with
images bundled** — Visual-CoT 8.10 GB, Zebra-CoT-count 4.46 GB, Zebra-CoT-visual-search
0.70 GB, CogCoM 0.04 GB, ReFocus 0.02 GB, plus five `train.json` files totalling 0.17 GB.
One `images.zip` per subset, no scavenging across source repos, reachable from the login
node. RL uses `Kwai-Keye/Thyme-RL`, of which they take a 3.2K subset.

The format is a plain conversation list with `<abs_vis_token></abs_vis_token>` marking
where a latent segment goes and `<observation>…</observation>` marking the alignment
targets. Checked on ReFocus: **2.00 auxiliary images and 2.00 latent segments per
sample**, and the observation spans are tiny — mean **1.7 words per sample**, median 1.

**The disk cost is the precompute, and their layout is wrong for lustre.** Both
precompute scripts write one `.pt` per sample (`rep_*.pt`, `latent_*.pt`) holding
all-layer hidden states at the aligned positions. On Qwen3-VL-8B that is 37 layers ×
4,096 × 2 B = **303 KB per aligned token**:

| | tokens/sample | per sample | × 125K |
|---|---|---|---|
| stage-2 targets (observation tokens) | ~5–20 | 1.5–6 MB | **190–760 GB** |
| stage-3 targets (latents: 2 segments × 8–10) | ~16–30 | 5–9 MB | **600 GB–1.1 TB** |

So **~0.8–1.9 TB in ~250,000 files**. Lustre has 17 TB free, so the bytes are fine; a
quarter-million files in two directories is not. Shard them (one file per 256 samples,
indexed) before this runs — it is an hour's work beforehand and a day's recovery after.
The scripts already carry `--resume` and a `_scan_existing_reps` glob, which is itself a
hint about how long these passes take.


## 7. Eval

Three of the paper's five benchmarks are **already in the HF cache**:
`craigwu___vstar_bench`, `DreamMr___hr-bench`, `yifanzhang114___mme-real_world-lmms-eval`.
VisualPuzzles is not and would need downloading; it is small.

The gap is the harness. They evaluate through **VLMEvalKit** with a `sitecustomize.py`
that swaps `vllm.v1.worker.gpu_model_runner` for theirs, plus an API judge replacing
exact matching. Two routes:

* **Their way** — a throwaway VLMEvalKit + vLLM 0.10.0 env. Zero porting, correct for
  the gate in §8, wrong as a permanent arm because none of our benchmark table lives
  there.
* **Our way** — port the 87-line inference patch to vLLM 0.11 (§3 confirms the anchors)
  and write a `monet_qwen3_vl` lmms-eval model, in the house style of
  `qwen3_vl_vga.py` / `qwen3_vl_sinkshift.py`. The wrapper is genuinely simple, because
  unlike Mini-o3 §7 and ERGO §7 there is **no loop to drive**: latent decoding happens
  inside vLLM, so the wrapper is a normal single-turn chat model that happens to need a
  patched runner and two env vars.

Their system prompt is load-bearing and is published: *"You are a helpful multimodal
assistant. You are required to answer the question based on the image provided. Put your
final answer in \\boxed{}."* Use it verbatim or the numbers are not theirs.

One output-handling detail: latent tokens are not decodable text, so any logging path
has to detect `<abs_vis_token>` and substitute a `<latent>` placeholder before the
string reaches a judge or a W&B table.


## 8. Run this gate first — 2–3 days, and it is likely to end the job

`NOVAglow646/Monet-7B` is public and, per §3, is a stock Qwen2.5-VL checkpoint. Serve it
in a throwaway vLLM **0.10.0** env with their `monet_gpu_model_runner.py` and
`LATENT_START_ID=151666` — their exact recipe, zero porting — and score V*, HR-Bench and
MME-RealWorld against Qwen3-VL-8B-Instruct, our cold start, and our `eval_mini` natural
set.

The reason to gate, from their Table 3 and Table 5 against the Qwen3-VL figures DeepEyes
§8 already collected:

| | V* | HRBench-4K | HRBench-8K | VisualPuzzles |
|---|---|---|---|---|
| Qwen2.5-VL-7B (their baseline) | 76.44 | 68.00 | 55.75 | 32.71 |
| + vanilla SFT on their 125K | 81.68 | 68.38 | 58.50 | 33.99 |
| + vanilla SFT + GRPO | 78.53 | 70.00 | 56.75 | 30.99 |
| Monet-SFT (latents, no VLPO) | 82.20 | 68.50 | 66.00 | 30.48 |
| **Monet-7B** | **83.25** | **71.00** | **68.00** | **35.02** |
| Qwen3-VL-8B-Thinking, **nothing added** | 77.5 | 72.4 | 68.1 | — |

**Qwen3-VL-8B already beats Monet-7B on HRBench-4K and ties it on HRBench-8K.** The only
clear headroom is V* (~+5.75) and the out-of-distribution VisualPuzzles column. Those
Qwen3-VL numbers are not ours — they come from DeepEyes §8 and a community benchmark
table, which is precisely what the gate converts into a measurement.

Read it as:

* **Monet-7B clearly above Qwen3-VL on ≥ 3 of 4** — proceed to §5.
* **Monet-7B above on V* only** — do the stage-1 partial from §5 and report it as a data
  result, not a method result.
* **Qwen3-VL at or above on ≥ 3 of 4** — stop. Same sentence DeepEyes §8 earns, one week
  instead of two months.

**Read the ablation column while the gate runs**, because it reframes what is being
bought. On V*, the entire latent apparatus is worth **+0.52** over a plain SFT on the
same data (81.68 → 82.20), and VLPO a further +1.05. Where latents genuinely pay is
HRBench-8K: 58.50 → 66.00 → 68.00, i.e. **+9.5 from the latent machinery**. That single
column is the whole case for this arm — and it is also the column Qwen3-VL already
matches at 68.1 with nothing added.


## 9. GPU budget

No hardware or hour count appears anywhere in the paper; the only resource statement is
"to avoid OOM, we use a batch size of 1". So this is modelled rather than scaled from
theirs, from two anchors: a FLOP count at ~110 TFLOP/s effective on an A100 with
gradient checkpointing, and our own step times for the rollout half
([[grpo-step-time-is-mostly-fixed-cost]]).

A sample is ~1,000 tokens: the global budget caps all images in a sample at 2,000
patches, i.e. ~500 visual tokens after the 2×2 merge, plus text.

| stage | volume | A100-h |
|---|---|---|
| 1 — warm-up | 500K sample-passes, full FT of the 8B LLM | 150–350 |
| precompute A — teacher reps | 125K forward-only | 20–40 |
| 2 — target latents | 1,000 steps × 128, plus the per-latent-token forward *and backward* | 100–250 |
| precompute B — target latents | 125K through the segmented latent forward | 60–120 |
| 3 — latent generation | 1,000 steps × 128, same loop | 100–250 |
| **one clean SFT pass** | | **~450–1,000** |
| VLPO, per run | ~50 optimizer steps | 35–100 |

**Where the unusual cost is.** Stages 2 and 3 run ~16–30 extra single-token
`language_model()` calls per sample, each carrying its own autograd graph, at batch size
1. Those are launch-latency bound rather than FLOP bound — roughly +0.3–0.9 s per
sample, which is ~20–40% of the stage cost and is pure overhead that no amount of GPU
buys back. It is also why batch size 1 is in the paper at all.

**RL is comparatively cheap.** 3.2K prompts over 1 epoch at `rollout_batch_size=64` is
~50 optimizer steps. But `online_accum_size=256` means 256 prompts are rolled out to
keep 64, so each step generates ~2,048 completions of up to 4,096 tokens — ~8M tokens
against our ~0.7M per step, and latent decoding serialises on top. Call it 5–15 min/step
on 8 A100s, 4–12 h per run.

Budget **two SFT passes** (the first will be wrong) and **three RL runs** (smoke, real,
fix): **~1,100–2,400 A100-hours**, plus eval, so **~1,500–3,000** all in. That is the
largest of the four sections, and the reason is that it is the only one paying for SFT.

Per [[fewer-gpus-keeps-batch-and-total-steps]], the 8-GPU shape is not a requirement —
trading ranks for grad-accum preserves batch and step count. The cluster blocker in
DeepEyes §9 applies unchanged: `oci-nrt-cs-001` still has no GPU hardware
([[gpu-partitions-can-be-empty-of-hardware]]), so all of this runs on
`cs-oci-ord-login-01`. Nothing here wants multi-node Ray.


## 10. Fidelity notes, and the comparability trap

* **Stage 1 is 3 epochs or 4.** Table 7 says 3 epochs / 3,885 steps;
  `script_examples/sft_stage1.sh` passes `--epochs 4`. 125K / 128 × 4 = 3,908, so the
  step count matches the script and the epoch count does not. Pick the step count and
  say which.
* **Stages 2 and 3 are 1 epoch or 2**, same disagreement: Table 7 says ~1,000 steps
  ("~1 epoch"), the scripts pass `--epochs 2`.
* **`latent_size` is {8, 10, 12} in the SFT table, 8 in the scripts, 10 at RL and 10 at
  inference.** It is a genuine free parameter and the paper never settles it; state the
  value used.
* **Their two tables disagree about their own model.** Monet-7B's MME-RealWorld is 58.34
  in Table 3 and 55.50 in Table 5. The ablation column is therefore not strictly
  comparable to the main one, which matters because §8's argument leans on the ablation.
* **The judge is Gemini-2.5-Pro over an external API**, both for accuracy during RL and
  as a "supplementary judge" at eval. Ours is self-hosted. Swapping it is correct and
  cheap, but it is a deviation that belongs in the caption, not a footnote — their
  VisualPuzzles and MME-RealWorld numbers were produced by a different grader than ours
  would be.
* **"No cold start" is not claimed here, and that is a comparability gift.** Monet's
  stage 1 *is* a cold start in their own format, so the Qwen3-VL format problem recorded
  at `build_coldstart_data.py:53` is solved by the method rather than around it. The flip
  side is that a Monet arm and our arm would then have **different cold starts** — theirs
  on Monet-SFT-125K, ours on LLaVA-CoT + Mulberry. An honest table says so, or adds a
  "+vanilla SFT on Monet-SFT-125K" control, which is the §5 partial and another reason to
  do it.
* **The comparability trap, in the Monet-shaped version.** Their claim is not accuracy
  at a budget (ERGO) or depth of search (Mini-o3) — it is that *continuous embeddings
  can carry visual thought*. The benchmark table is weak evidence for that and §8 shows
  it has largely evaporated on this backbone. If this arm is built, the interesting
  comparison is not V*: it is whether our attention-based story and their
  latent-embedding story are measuring the same thing, which is an analysis, not a row.
  Worth deciding **before** six weeks are spent, not after.


## Sources

* [arXiv:2511.21395](https://arxiv.org/abs/2511.21395) — the paper (CVPR 2026)
* [NOVAglow646/Monet](https://github.com/NOVAglow646/Monet) — SFT, RL, both vLLM patches
* [NOVAglow646/Monet-7B](https://huggingface.co/NOVAglow646/Monet-7B) — the public checkpoint; stage 1 and stage 3 are also released
* [NOVAglow646/Monet-SFT-125K](https://huggingface.co/datasets/NOVAglow646/Monet-SFT-125K) — 13.5 GB, images included
* [Kwai-Keye/Thyme-RL](https://huggingface.co/datasets/Kwai-Keye/Thyme-RL) — the RL data
* [hiyouga/EasyR1](https://github.com/hiyouga/EasyR1) — the RL framework; already Qwen3-VL-aware upstream
* [UMass-Embodied-AGI/Mirage](https://github.com/UMass-Embodied-AGI/Mirage) — the latent-reasoning predecessor they credit
