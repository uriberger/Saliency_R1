# Porting external visual-agent baselines onto Qwen3-VL — estimates

One section per external paper we have costed as a comparable row next to our model:
same backbone (Qwen3-VL-8B-Instruct), same eval protocol, their method. Each section
is a decision document, not a plan of record — the numbers in them are models, and the
first real smoke run replaces the numbers.

| baseline | verdict | gate before committing |
|---|---|---|
| [DeepEyes](#deepeyes-on-qwen3-vl--port-and-training-estimate) — zoom tool, 80 steps, no cold start | ~3–5 weeks, ~1,000–2,500 A100-h | 2 days: serve `ChenShawn/DeepEyes-7B`, score it on our suite |
| [Mini-o3](#mini-o3-on-qwen3-vl--port-and-training-estimate) — crop tool, deep multi-turn, over-turn masking | ~3–4 weeks, ~2,000–3,000 A100-h | 3–5 days: build the agentic eval wrapper, score their released `Mini-o3-7B-v1` |

### What every one of these has in common

Three findings recur, and they are why these estimates land where they do rather than
at "six months":

1. **Every released fork is frozen pre-Qwen3-VL.** DeepEyes pins verl 0.2.0.dev0 /
   transformers 4.51.3 / vLLM 0.8.2; Mini-o3 pins verl 0.2.0.dev / transformers 4.52.0 /
   vLLM 0.9.2. Neither stack can *load* Qwen3-VL — it needs transformers ≥ 4.57 and
   vLLM ≥ 0.11. Forward-porting the fork is always the expensive path.
2. **Qwen3-VL is token-compatible with Qwen2.5-VL.** Identical ChatML scheme and
   identical vision token IDs (`vision_start 151652`, `vision_end 151653`,
   `image_token 151655`, `im_end 151645`) — verified against the local
   Qwen3-VL-8B-Instruct snapshot. Every hardcoded splice, token-count heuristic and
   `<image>` rewrite in these repos transfers verbatim. What does *not* transfer is
   geometry: Qwen3-VL is patch-16 / merge-2, so one visual token covers 1,024 px²
   against Qwen2.5-VL's 784. Any `max_pixels`-shaped hyperparameter is off by ~23% in
   token terms and has to be re-tuned.
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
