# DeepEyes on Qwen3-VL — agent prompts for build and training

Companion to [baseline-port-estimates.md](baseline-port-estimates.md) § *DeepEyes on
Qwen3-VL*. That document is the decision; this one is the execution.

> ## Status as of 2026-11-07 — read before running any prompt below
>
> The build ran. **P0, P1, P2, P3, P4, P5, P6 and P8 were executed; P7 and P9 are held.**
>
> **P1 (the headroom gate) returned NO-GO.** Three independent lines put the method's
> gain on Qwen3-VL at ~+5 on V\*, not +18.9: the Qwen3-VL tech report §5.6 states the
> absolute improvement from adding tools is *"consistently ~5 points across V\*"* within
> the family; the +18.9 is measured from a Qwen2.5-VL-7B baseline our backbone has
> already passed (Qwen3-VL-8B-Instruct measured at 74.0 on V\*); and ZoomEye gets
> 90.6 training-free against DeepEyes' 90.1 in DeepEyes' *own* Table 1. See
> [deepeyes-gate.md](deepeyes-gate.md), whose §8 pre-registers what would reopen it.
> **Do not start the 80-step run off the back of this document alone.**
>
> **P2 (the engineering gate) passed**, on real code without a GPU, and the stack is
> sound. The blocker is hardware, not the port: `oci-nrt-cs-001` has **zero GPU nodes**,
> so everything GPU-bound runs on `cs-oci-ord-login-01` — a separate filesystem, so
> branches travel through `origin`. See `DEEPEYES_GATE_RUN_PROMPT.md`.
>
> **Five claims in this document were wrong and are corrected inline below**, each marked
> at the point of use: the `R_format = -1` penalty (Prompt 4, Prompt 9, §4), the
> transferability of the vision-token counting trick (Prompt 5, §5), §5's assurance
> that identical token IDs make the processor discriminator safe, and — the one that
> changes what gets trained — **the recommendation to start from the cold start, which
> measurement inverted** (Prompt 4, Prompt 9). All of them were the silent-when-wrong
> kind. Prefer the corrections to the original text.
>
> **Three results are now measured rather than assumed.** In Prompt 4: the coordinate
> frame is **`rel1000`**, by a margin that is not close, which means DeepEyes' own tool
> is wrong on every call for this backbone; and the start point is
> **`Qwen/Qwen3-VL-8B-Instruct`**, not the cold start. In Prompt 6, the one that bears
> hardest on whether to run at all: across **all 36,021 tool rows of their released
> corpus the zoom's downscale is a median 1.000** — every image already fits under
> Qwen3-VL's 16.8 MP cap, and under the 12.8 MP cap of the `qwen-vl-utils` *they*
> pinned. **A zoom recovers no pixel the model did not already have, and that was true
> for them too.** Read it with P1's gate: the headroom is small, and on the training
> distribution the stated mechanism is not operating.

**Do not port their fork, and do not clone verl.** Their release is frozen on verl
0.2.0.dev0 / transformers 4.51.3 / vLLM 0.8.2, none of which can load Qwen3-VL. And
verl at HEAD is the wrong target too: `image_zoom_in_tool.py` was deleted from
mainline on 2026-05-09 (`5a506cc5`) and `recipe/` moved out to `verl-recipe` on
2026-01-06 (`2bb42bae`). The right starting point is already on disk — **`laser_fork/`**,
verl `0.7.0.dev0` on branch `qwen3-vl`, Qwen3-VL-aware end to end, running in the
`laser` conda env, carrying both `verl/tools/image_zoom_in_tool.py` and the
multimodal `verl/experimental/agent_loop/tool_agent_loop.py`. All verified present.

With that as the base, DeepEyes' actual deliverable collapses to roughly **500 lines**
of tool + reward + data conversion.

**Budget.** 80 steps — which is the paper's *full* run, not a scaled-down one. This
is the only baseline where what fits the clock and what the authors did are the same
thing. 30–71 h on 8×A100, 4–9 chunked jobs. Eval is nearly free.

---

## 1. How to use this file

Each `### Prompt N` section **is** the prompt — paste the whole section, heading
included, into a fresh agent. Prepend §2 to every one of them. Prompts at the same
depth in §3 are independent; launch them in one message so they run concurrently.

---

## 2. Shared preamble — prepend this to every prompt below

> You are working in `/lustre/fs1/portfolios/nvr/projects/nvr_israel_rlop/users/uberger/research/saliency_r1`.
> Read `CLAUDE.md` first and obey it exactly.
>
> - The directory you start in is the **central tree** and is **read-only**. Read,
>   grep and launch jobs there; never edit a file and never run a state-changing git
>   command in it.
> - Open your own worktree for all edits: `./worktree.sh new <branch>`, then work
>   entirely inside `.worktrees/<branch-with-dashes>/`. Commit early and often.
> - The user's shell is **fish**. Write every command and snippet in fish syntax.
> - `checkpoint/`, `outputs/`, `cold_data/`, `trl_repo/`, `laser_repo/`, `laser_fork/`
>   are **symlinks shared with every other session and every running job**. Write run
>   outputs to a per-branch subdirectory. **Never run `patch_laser_qwen3.sh`** — it
>   re-patches a shared clone for every session.
> - Never `git add -A`. Stage explicit paths.
> - Do not score test benchmarks while a training run is in flight; dispatch
>   `watch_bench_evals.sh` afterwards.
> - When your task is finished, report the branch path and stop. Do **not** merge to
>   `main` — the user confirms that.
>
> The verl you are building on, verified:
>
> | path | what it is |
> |---|---|
> | `laser_fork/` | verl `0.7.0.dev0`, branch `qwen3-vl`, the copy that executes. **Shared.** |
> | `laser_repo/` | LASER's released fork, read-only reference. Do not execute from it. |
> | `laser_fork/verl/tools/image_zoom_in_tool.py` | `ImageZoomInTool`, already a verl `BaseTool`, takes `bbox_2d` |
> | `laser_fork/verl/experimental/agent_loop/tool_agent_loop.py` | multi-turn loop; `ToolResponse(text=, image=, video=)`, observation-token loss masking via `response_mask`, emits `num_turns` |
> | `laser_fork/verl/models/transformers/qwen3_vl.py` | Qwen3-VL `get_rope_index` (interleaved MRoPE), `_get_input_embeds`, three backend forwards |
> | conda env `laser` | torch 2.8.0+cu128, transformers 4.57.6, **vLLM 0.11.0**, ray 2.50.0, verl editable |
> | `docs/laser-fork-qwen3.md` | how this fork was made Qwen3-VL-native; read §1 before assuming anything is Qwen2.5-shaped |
>
> There is **no `recipe/` directory** in either clone — do not go looking for
> `recipe/deepeyes`.
>
> Our side:
>
> | path | what it is |
> |---|---|
> | `checkpoint/coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged` | the cold start every arm starts from |
> | `llm_judge.py`, `answer_grading.py`, `trl/rewards/openai_rewards.py` | our hosted judge; replaces their self-hosted Qwen2.5-72B outright |
> | `eval_mini/benchmarks.py` | the 13-benchmark mini suite table |
> | `cluster_env.sh` | `sr1_pick_partition` / `sr1_find_submit_job` |
> | `launch_laser_job.sh` + `launch_laser.sh` | the `_job.sh` (submit) / `_run.sh` (execute) launcher split to copy |
> | `build_laser_data.py` | the precedent for building a verl parquet from this repo |

---

## 3. Order of work

```
day 1-2   P0 env ─────────┐
          P1 Gate ────────┼── in parallel
          P4 coord/format ┘
                          │
day 2     P2 verl smoke ──┤   (depends on P0; prove the stack before adding anything)
                          │
day 3-4                   ├─ P3 zoom tool ────┐
                          ├─ P5 rewards ──────┤
                          ├─ P6 data ─────────┤
                          └─ P8 eval ─────────┤
                                              │
day 4-5   P7 launcher ────────────────────────┤
day 5-8   P9 smoke + 80-step run ─────────────┘
```

**P2 is the real gate on the engineering.** If plain verl GRPO will not run on
Qwen3-VL-8B in the `laser` env, nothing after it matters and the arm should be
abandoned the same day rather than debugged alongside new code.

---

### Prompt 0 — a private verl clone and environment for DeepEyes

Branch: `feat/deepeyes-env`

`laser_fork/` is a **shared symlink**: it is what the LASER work executes and
`patch_laser_qwen3.sh` re-patches it for every session. DeepEyes needs its own tool,
its own reward and its own config block, so give it its own clone.

1. Create `deepeyes_fork/` as a clone of `laser_fork` at its current branch
   (`qwen3-vl`, head `bfd3b30`). Read `patch_laser_qwen3.sh` for the structure of how
   our tracked sources get into a fork — copy the pattern, do not run the script.
2. Add `deepeyes_fork` to `.worktree-links` with a house-style comment saying what it
   is and that it is shared.
3. Create a conda env named `deepeyes` by cloning `laser`
   (`conda create -n deepeyes --clone laser`), then `pip install -e deepeyes_fork`
   into it.

**One thing you will see and must not "fix".** In the `laser` env, `pip list` reports
verl's editable location as `.worktrees/feat-laser-env/laser_fork` — a worktree that
no longer exists. `python -c "import verl, os; print(os.path.dirname(verl.__file__))"`
nevertheless resolves correctly to the central `laser_fork/verl`. Reinstalling verl
in the `laser` env to tidy the metadata risks breaking the LASER arm; leave it alone
and only make sure your *new* `deepeyes` env points at `deepeyes_fork`.

Verify and report: in `deepeyes`, `import verl` resolves under `deepeyes_fork`; in
`laser`, it still resolves under `laser_fork`; and `git -C laser_fork status --short`
is clean. Versions that must not move: transformers **4.57.6** (it is what ships
Qwen3-VL and what verl is pinned against), vLLM **0.11.0** (vllm#29595 reports
corrupted Qwen3-VL grounding output above it, and grounding output is what this
method crops on), torch 2.8.0+cu128.

---

### Prompt 1 — the gate: score their released checkpoint on our benchmarks

Branch: `probe/deepeyes-gate`

`ChenShawn/DeepEyes-7B` is public. Serve it on vLLM and run their `eval/eval_vstar.py`
and `eval/eval_hrbench.py` — both are plain OpenAI-API clients against a vLLM server
with the bbox/crop loop client-side, so they are **model-agnostic and need zero
changes**; just point `--api_url` at the server. `judge_result.py` wants a judge
endpoint, which we already have.

Score it alongside Qwen3-VL-8B-Instruct, our cold start, and our `eval_mini` natural
set. `vstar_bench` (craigwu) and `HR-Bench` (DreamMr) are already in the HF cache.

**Why this gate exists.** Qwen3-VL already received this treatment: its tech report
describes synthesizing ~10k grounding traces, SFT-ing Qwen2.5-VL-32B into a
"think → act → analyze feedback → answer" visual agent, distilling ~120k multi-turn
agentic interactions, then cold-start SFT plus tool-integrated RL with accuracy,
multi-turn-reasoning and tool-calling rewards. That is DeepEyes, done by the vendor,
at larger scale — and the official Qwen3-VL cookbook ships an `image_zoom_in_tool`.

Headroom appears to survive, but much smaller than the headline:

| | V* | HRBench-4K | HRBench-8K |
|---|---|---|---|
| Qwen2.5-VL-7B (their baseline) | 71.2 | 68.8 | 65.3 |
| DeepEyes-7B | 90.1 | 75.1 | 72.6 |
| Qwen3-VL-8B-Thinking, **no tool** | 77.5 | 72.4 | 68.1 |

So the honest expected result is "+10 on V*, ~+2 elsewhere", not +18.9. Report the
measured table and say plainly whether the gap is worth the run. Note for the
writeup: the paper's framing — "this ability emerges from RL without cold-start SFT"
— is already falsified for this backbone, because Qwen3-VL got exactly that cold
start. That is a sentence worth having either way.

---

### Prompt 2 — prove the stack before adding anything

Branch: `probe/deepeyes-verl-smoke`

Depends on P0. Run **plain verl GRPO on Qwen3-VL-8B** in the `deepeyes` env, on a
small standard task (geo3k is the usual choice), 8 GPUs, a handful of steps. No tool,
no custom reward, nothing of ours.

This exists because every hour spent debugging a new tool on top of an unproven stack
is an hour spent not knowing which half is broken. `docs/laser-fork-qwen3.md` §1
records that this fork is already Qwen3-VL-aware end to end, so this should work; the
point is to find out in a day rather than discover it inside a 20-hour run.

Then, in the same job, run **one multi-turn smoke with the existing
`ImageZoomInTool`** and assert three things explicitly, because all three are silent
when wrong:

1. **Position ids are Qwen3-VL's.** The bug this guards against is DeepEyes' own
   `verl/workers/agent/parallel_env.py:282`, which gates position-id computation on
   `processor.image_processor.__class__.__name__ == "Qwen2VLImageProcessor"` — False
   on Qwen3-VL, so it falls through to 1-D LM position ids and **training looks fine
   and is wrong**. We are not porting that file, but assert that `get_rope_index` from
   `verl/models/transformers/qwen3_vl.py` is what actually ran, and that
   `position_ids` has the 4-row interleaved-MRoPE shape rather than 1-D.
2. **Observation tokens are masked out of the policy loss.** Print the fraction of
   `response_mask` that is zero on a turn carrying a tool image; it must be non-zero
   and must line up with the `<|vision_start|><|image_pad|>…<|vision_end|>` span.
3. **`image_grid_thw` survives the tool turn.** Upstream verl has an end-to-end test
   for exactly this (`tests/experimental/agent_loop/test_multi_modal.py`); adapt it
   rather than writing one from scratch.

Report pass/fail per assertion plus the step time. If any fail, stop and report —
do not work around them.

---

### Prompt 3 — the zoom tool, and the two numbers in it that are wrong for Qwen3-VL

Branch: `feat/deepeyes-zoom-tool`

Work in `deepeyes_fork/`. `verl/tools/image_zoom_in_tool.py` already exists and is
already a verl `BaseTool` with the crop/validate logic — this prompt is adaptation,
not a port. Read it fully first (`ImageZoomInTool` at :115, `_maybe_resize_bbox` at
:205, `execute` at :345).

Three changes, in descending order of how expensive they are to find late:

1. **The coordinate frame.** The tool's `execute` reads `bbox_2d` and treats it as
   absolute pixels. Qwen2.5-VL emits absolute pixels in the post-`smart_resize` frame;
   **Qwen3-VL emits a 0–1000 relative grid against the original image.** Use whichever
   frame P4 measured, convert explicitly at the tool boundary, and **log the raw box
   and the converted box for the first N calls** so the frame is visible in the run
   rather than assumed. Getting this wrong crops the wrong region on every call while
   the loss curve stays healthy.
2. **`MIN_DIMENSION = 28` is a 32.** Qwen3-VL is patch-16 / merge-2, so one visual
   token covers 32×32 px against Qwen2.5-VL's 28×28. The 28 px floor is a one-token
   floor in their geometry and is now sub-token in ours. Restate it, and restate any
   `N×28×28` budget as `N×32×32` to mean the same token count.
3. **Crop the original, pre-processor-resize image.** This is DeepEyes' actual
   requirement and the reason the method recovers detail at all. Wire the tool to the
   un-resized original and assert it — a crop taken from an already-downsampled image
   recovers nothing and the whole arm becomes a no-op.

**Fidelity wrinkle to settle and state.** The paper says "up to a maximum of 6 times
of active perceptions"; their released script sets `max_turns=5`, and in their loop
that counts generate rounds, so it is at most 4 tool calls. Pick one, put it in the
launcher header, and say which in the writeup.

Ship a CPU test covering: coordinate conversion in both frames, the 32 px floor, an
out-of-bounds box, a degenerate box, and that the crop source is the original image.

---

### Prompt 4 — the coordinate frame and the format rate, measured

Branch: `probe/deepeyes-coord-format`

Independent of everything; run it on day 1 alongside P1. Two measurements that
P3 and P5 both depend on.

**Coordinates.** Over ~200 V*/HR-Bench rows, prompt Qwen3-VL-8B-Instruct and our cold
start for a zoom box in DeepEyes' format, and for each emitted box compute IoU
against the ground truth under the 0–1000-relative reading and under the
absolute-pixel reading. Report which interpretation wins and by what margin. Do not
assume — Qwen3-VL's cookbook says 0–1000 relative, but what the model does under
*their* prompt is the thing that matters.

**Format.** `build_coldstart_data.py:52` records Qwen3-VL-8B-Instruct at **format
0.000 under every prompt variant tried**. Measure the tag-match rate under their prompt
on both models.

> **Corrected 2026-11-07, after P5 transcribed their `vl_agent.py` and tested it
> differentially against their own file.** This paragraph used to read "`R_format` is
> **−1 on any tag mismatch**, so a model that cannot hit the tag structure is actively
> penalised on every rollout". **That is wrong.** Their check is a *count comparison*
> per tag clause, so −1 fires on an genuine *imbalance* (two `<think>` against one
> `</think>`), but a completion with **no tags at all** scores 0 == 0 on every clause
> and gets **0.0, not −1.0**. A model at format 0.000 is not being penalised.
>
> The real argument for starting from the cold start is the **answer parse**, and it
> bites through accuracy rather than format: with no tags, their
> `split("<answer>")[-1]` returns the *whole completion*, which then trips their
> ≥1000-character anti-hack guard — so a tagless model loses **both** `R_acc` and
> `R_format`. Same recommendation, different and stronger mechanism. Both behaviours
> are pinned in `test_deepeyes_rewards_cpu.py`.

This is the argument for starting DeepEyes from
`checkpoint/coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged` rather than from
the Instruct model, even though the paper's claim is "no cold start needed". Report
the two format rates; if the Instruct model is near 0 and the cold start is high,
recommend the cold start and note that "no cold start" was a claim about Qwen2.5-VL,
not a property of the method.

~1 h on one GPU. `coldstart_gate.py` is the precedent for the shape of this script.

> **Measured 2026-11-07 — the paragraph above predicted the wrong direction and the
> recommendation inverts.** 96 completions per model, their prompt, temperature 1.0:
>
> | model | `R_format` ok | zooms their env actually runs | episode stalls |
> |---|---|---|---|
> | **Qwen3-VL-8B-Instruct** | **100.0%** | **95.8%** | 0.8% |
> | cold start | 75.0% | 42.7% | 15.6% |
>
> **Start DeepEyes from `Qwen/Qwen3-VL-8B-Instruct`, not from the cold start.** The
> Instruct model is already a tool-calling model. The cold start's two epochs of
> `<think>chain</think>` with the answer in the same turn are two epochs of pressure
> *away* from the deferred answer this protocol needs. `build_coldstart_data.py:52`'s
> "format 0.000" is about **LASER's four-tag format** and does not transfer to
> DeepEyes' prompt — the premise of the paragraph above.
>
> So the paper's "no cold start needed" is, on this backbone, *correct for a reason the
> paper did not have*: Qwen3-VL shipped with the agentic cold start already in it.
>
> Measured on `cpu_short` (no GPU hardware), so both arms carry a visual-token cap and
> `max_new_tokens=320` against their 20,480; the ≥1000-char clause is therefore a
> **floor** and the format column is the weaker of the two results. The coordinate
> result below does not depend on any of that.

> **Coordinate frame, measured the same day: `rel1000` wins, and it is not close.** 158
> emitted boxes against V*'s ground truth (HR-Bench carries no boxes, so V* alone):
>
> | reading | mean IoU | touches target | mostly off-image |
> |---|---|---|---|
> | **`rel1000`** (`x·W/1000`) | **0.260** | **77.2%** | 0.0% |
> | `abs_orig` (what DeepEyes' tool assumes) | 0.000 | 3.8% | 0.0% |
> | `abs_resized` (Qwen2.5-VL's frame) | 0.003 | 8.2% | 31.0% |
>
> The invariance test settles it: halving the resolution the model sees moved the
> emitted numbers by **1.03–1.04×** — invariant, therefore relative — while the
> resized-frame reading moved 0.488×. **DeepEyes' `visual_toolbox_v2` reads `bbox_2d`
> as absolute pixels and is therefore wrong on every call for this backbone.**

---

### Prompt 5 — the three reward functions

Branch: `feat/deepeyes-rewards`

Port from `verl/utils/reward_score/vl_agent.py` in their repo (clone to scratch; it
is reference only). The composite is the whole trick:

```
0.8*R_acc + 0.2*R_format + 1.2*R_tool

R_tool   = 1 iff (at least one zoom happened AND R_acc > 0.5)
R_format = -1 on any tag mismatch, else 0      # a COUNT mismatch -- zero tags scores
                                               # 0.0, not -1.0; see Prompt 4
R_acc    = binary, from a judge
```

The conditional gate is the design: a tool bonus that **only pays out on a correct
answer**, so the policy cannot farm it by zooming indiscriminately. Preserve that
exactly.

The math subset (`compute_score_math`) drops the tool term entirely and uses
`1.2*acc + 0.4*format` with math_verify plus a generative fallback. Implement both
and route on `data_source` the way their code does.

Two substitutions:

- **The judge.** They self-host Qwen2.5-72B. Use ours — `llm_judge.py` /
  `trl/rewards/openai_rewards.py` against the NVIDIA gateway. This removes 4 GPUs
  from the job outright. Note the swap in the writeup; a different grader is a
  caption-level deviation.
- **"A zoom happened."** Their code detects this by counting
  `<|vision_start|><|image_pad|>` occurrences in the response string
  (`vl_agent.py:199`). **That trick is dead in this fork — it does not merely transfer
  badly, it silently returns 0 forever.** Their `naive.py:71` decodes with no skip;
  ours (`naive.py:79`) passes `skip_special_tokens=True`, and those ids are
  `special=True` in the cold start's `tokenizer.json`. Both sides of the comparison are
  0 on every rollout, so `R_tool` is identically 0 and the arm quietly degrades to plain
  GRPO. (Corrected 2026-11-07; this entry previously said "that trick does transfer —
  Qwen3-VL has identical vision token IDs". The token IDs *are* identical — `vision_start
  151652`, `vision_end 151653`, `image_token 151655` — which is exactly why the failure
  is invisible: nothing about the IDs is wrong, the tokens are gone before the count.)

  So **re-derive it from verl's `num_turns` metadata**, which was already this entry's
  recommendation and is now mandatory rather than preferred. `tool_agent_loop.py:180`
  emits `num_turns` = user+assistant+1, so no-tool is 2 and each zoom adds 2:
  `rounds = (num_turns - 1) // 2`. It cannot be fooled by a model that types the literal
  token text, and it is the thing you actually want to log.

Ship a CPU test asserting: the gate (zoom + wrong answer scores no tool bonus; no
zoom + right answer scores no tool bonus; both scores 1.2), the −1 format penalty,
and the math route.

---

### Prompt 6 — data conversion

Branch: `feat/deepeyes-data`

`ChenShawn/DeepEyes-Datasets-47k` on HF, **4.84 GB in 3 parquet files**, images
embedded as bytes, reachable from the login node:

```
0.990 GB  data_0.1.2_visual_toolbox_v2.parquet      # V* / visual search, zoom tool on
2.199 GB  data_v0.8_visual_toolbox_v2.parquet       # charts, zoom tool on
1.656 GB  data_thinklite_reasoning_acc.parquet      # math reasoning, zoom tool OFF
```

47k rows = V* 22k (47%) + ArxivQA/chart 14k (30%) + ThinkLite-VL 11k (23%), already
filtered by the authors for difficulty (0% and 100% accuracy items dropped), label
correctness, and solvability by active perception.

The good news: it is **already in verl's schema** — `data_source`, `prompt`,
`images`, `env_name`, `reward_model.{ground_truth,style}`,
`extra_info.{question,answer,index,split}`. The conversion is mostly re-routing
`env_name=visual_toolbox_v2` to modern verl's agent/tools config block, and
confirming the prompt format against Qwen3-VL's chat template and hermes tool-call
parsing. `build_laser_data.py` is the in-repo precedent for producing a verl parquet.

Check and report two things that are easy to get silently wrong:

1. **The ThinkLite split must have the tool disabled**, as in their config. If it
   gets the tool, the math reward's composite is wrong for a quarter of the corpus.
2. **The images must stay at native resolution.** Our own corpora are capped at 512 px
   (`build_grpo_sets.py:109`), and anything that applies that cap here removes the
   detail the zoom exists to recover.

Send the conversion to `cpu_short` (50 GPU-free nodes, instant) rather than queueing
behind a GPU allocation.

> ### Measured 2026-11-07 — the zoom has no headroom on their own training corpus
>
> Requirement 2 above is satisfied, and it does not help. Measured on **all 36,021 tool
> rows**, not a sample:
>
> | split | downscale median | max | rows at or below 1.0 |
> |---|---|---|---|
> | vstar | **1.000** | 1.066 | 12,530 / 22,362 |
> | chart | **1.000** | 1.042 | 7,070 / 13,659 |
>
> Qwen3-VL's cap is 16,384 × 32² = **16.8 MP**; the largest image in the whole corpus is
> **4.06 MP**. `smart_resize` therefore never shrinks anything here — it only rounds to
> the 32 px grid, which is the entire 0.975–1.066 spread. **A zoom recovers no pixel the
> model did not already have.**
>
> Three checks rule out this being our doing:
>
> 1. The output image bytes hash identical to the release, so nothing in our conversion
>    capped them.
> 2. DeepEyes pin `qwen-vl-utils==0.0.11`, whose cap is **12.8 MP** — also far above
>    4.06 MP. **They trained at downscale ~1.0 too.**
> 3. The only lever is lowering the *view* (`view_max_tokens`, or the dataloader's
>    `max_pixels`), which is a deviation from DeepEyes and belongs in the launcher as a
>    stated decision, not a silent one.
>
> This is a claim about the **training** corpus, and it sits oddly against the
> **evaluation**: HR-Bench 8K images do exceed the cap, so the policy is trained where
> zooming cannot recover detail and scored where it can. Whatever the method buys on
> their training distribution, it is not resolution.
>
> **P9 must read the tool-call rate against this.** The build doc's Prompt 9 says a rate
> near 0 means `R_tool` never fires and the run is plain GRPO. On this corpus a low rate
> may instead be the policy being *correct* — there is nothing to zoom into. The two
> readings are indistinguishable without this table.

---

### Prompt 7 — the launcher

Branch: `feat/deepeyes-launcher`

Copy the `_job.sh` / `_run.sh` split from `launch_laser_job.sh` + `launch_laser.sh`.

Requirements:

- `source cluster_env.sh` and resolve `PARTITION` with `sr1_pick_partition`
  **after** argument parsing — which partitions can hold the job depends on how long
  it asks for.
- Partitions cap at 2/4/8 h; the run is 30–71 h on 8 A100s, so **chunked autoresume
  is mandatory** — 4–9 jobs with verl's `save_freq` and resume-from-latest. Each
  restart pays model load plus vLLM engine init; use the longest chunk the partition
  allows.
- Their README demands 32 GPUs (4×8) plus 4 for the judge, and 1,200 GB CPU RAM per
  node. Both shrink: our hosted judge removes the judge GPUs outright, and trading
  ranks for grad-accum preserves batch and step count, so 8 GPUs hold the same 256×16
  batch — slower, not different. Encode that in the config and say so in the header.
- Outputs to a per-branch subdirectory under the shared `outputs/` and `checkpoint/`.
- If a multi-node Ray shape is attempted, note that none of our launchers do it today
  and that it is new unvalidated infrastructure — single node is the supported path.

Header in the house style of `launch_grpo_qwen3_overlap_job.sh`: what is not
optional, what the first step should print, what a mismatch there means.

---

### Prompt 8 — eval

Branch: `feat/deepeyes-eval`

The cheapest eval of the four baselines, and it can be built before the model exists.

- **Their scripts need zero changes.** `eval/eval_vstar.py` and `eval/eval_hrbench.py`
  are plain OpenAI-API clients against a vLLM server with the bbox/crop loop
  client-side. Point `--api_url` at a vLLM 0.11 server. `judge_result.py` wants a
  judge endpoint, which we have. This is the path for reproducing *their* table.
- **For our table**, `eval_mini/benchmarks.py` carries 13 benchmarks and none of V*
  or HR-Bench is among them — only `mmerealworld` overlaps DeepEyes' headline.
  `mmerealworld_mini` and `pope_mini` are already in our natural set. Add
  `vstar_bench`, `hrbench_4k` and `hrbench_8k`; read the module docstring first and
  set `scale` from a real results.json, not from the yaml. The task files already
  exist under `/home/uberger/scratch/research/lmms-eval/lmms_eval/tasks/`.

If the ERGO arm is also being built, **this overlaps its Prompt 4 exactly** — do it
once, in one branch, and have both arms use it.

Acceptance: both of their scripts run end to end against a served Qwen3-VL-8B and
produce a number; the three new benchmarks reduce through `bench_eval.py` without a
KeyError.

---

### Prompt 9 — smoke, then the 80-step run

Branch: `run/deepeyes-smoke`

**Smoke first**, ~5 steps with full logging. Read these before scaling, in order:

1. **tool-call rate** — the fraction of rollouts that zoom at least once. If it is
   near 0, `R_tool` never fires and the run is plain GRPO with a format penalty.
   If it is 1.0 on every rollout, check that the conditional gate is actually gating.
2. **format rate** — a low format rate is not an active penalty (see the correction in
   Prompt 4: a tagless completion scores `R_format` 0.0, not −1). It bites through
   **accuracy** instead: no `<answer>` tag means the whole completion is parsed as the
   answer, which trips their ≥1000-character anti-hack guard and zeroes `R_acc` — and
   `R_tool`, which gates on `R_acc > 0.5`, then never fires either. So watch `R_acc` and
   the tool-call rate together with the format rate; a format collapse shows up as all
   three going flat at once. If P4's cold-start recommendation was ignored, this is
   where it shows.
3. **the logged raw-vs-converted boxes** from P3 — eyeball ten of them against the
   image. This is the only cheap check on the coordinate frame.
4. **truncation at `max_response_length=20480`** and OOM. Both appear in the first few
   steps or not at all.

Then the real run. Their config: GRPO, **KL coefficient 0**, **80 steps**, 256 prompts
× 16 rollouts, `max_response_length=20480`, `max_prompt_length=8192`, lr 1e-6. ~~Start
from the cold start unless P4 said otherwise.~~ **P4 said otherwise: start from
`Qwen/Qwen3-VL-8B-Instruct`** — 100.0% on their format against the cold start's 75.0%,
and 95.8% of its tool calls survive their env against 42.7%. See the measured block in
Prompt 4.

Expect **30–71 h on 8 A100s** (22–53 min/step), 4–9 chunked jobs. Multi-turn rollouts
are latency-bound rather than FLOP-bound — 5 sequential generate rounds with
stragglers — so plan for the top of the band. Do not score test benchmarks while it
runs; dispatch `watch_bench_evals.sh` afterwards.

**Note for the writeup.** `baseline-port-estimates.md` §9 states both "3–7 A100-hours
per step" and "~400–1,200 A100-hours for one 80-step run"; 7 × 80 = 560, so the two
figures disagree by ~2×. Record the *measured* per-step time from the smoke run and
correct the estimate rather than carrying both forward.

---

## 4. The traps, in one place

| trap | where it bites | what it looks like when it bites |
|---|---|---|
| porting their fork, or cloning verl at HEAD | P0 | weeks on a transformers/vLLM forward-port; or a HEAD with the zoom tool deleted |
| `parallel_env.py:282` falling through to 1-D position ids | P2 | **training looks fine and is wrong** — the one that costs a week if found late |
| 0–1000 relative coordinates | P3, P4 | every crop lands in the wrong place; the loss curve stays healthy |
| `MIN_DIMENSION = 28` on a 32 px grid | P3 | a sub-token floor; degenerate crops accepted |
| cropping the resized image instead of the original | P3 | the method becomes a no-op; nothing to recover |
| format near 0 on a tagless model | P4, P9 | **not** a −1 penalty (that was this table's error until 2026-11-07) — the whole completion parses as the answer, trips their ≥1000-char guard, and zeroes `R_acc`; `R_tool` gates on `R_acc` so it never fires either |
| `skip_special_tokens=True` eating the zoom detector | P5 | `R_tool` identically 0, arm silently degrades to plain GRPO with no error anywhere |
| the tool enabled on the ThinkLite math split | P6 | wrong composite on 23% of the corpus |
| editing `laser_fork/` instead of `deepeyes_fork/` | P0 | the LASER arm and any in-flight job change underneath it |
| vLLM above 0.11.0 | P0 | corrupted Qwen3-VL grounding output (vllm#29595) — i.e. corrupted zoom boxes |
| `max_turns`: 5 or 6 | P3 | an unstated fidelity difference from the paper — **settled: 5, meaning 4 tool calls**, following their code over the paper |
| ~~no `process_raw_image` in the fork~~ **resolved — not a trap** | P3, P6 | Raised because DeepEyes keeps a second un-resized copy and our `vision_utils.py` has only `process_image`. It does not apply: DeepEyes is on verl **0.2.0.dev0**, where tool state came from a dataloader column; verl **0.7** routes it through `extra_info.tools_kwargs`, which `RLHFDataset` passes through **untouched** (`rl_dataset.py:445,451`). Proved on a real 1000×700 row — prompt image 992×704, tool holds 1000×700. No fork change needed. The real cost is that bytes appear twice (4.84 → ~9.7 GB), avoidable by putting a *path* in `create_kwargs["image"]` |
| a correctly-wired tool on a corpus that was capped | P6 | the zoom **upscales** and recovers nothing — indistinguishable from correct wiring without checking the downscale distribution. `python deepeyes/zoom_tool_config.py verify <parquet>` reports it; on geo3k it reads median 0.999, 9/16 rows ≤ 1.0 |
| a turn carrying both `<tool_call>` and a closed `<answer>` | P5, P9 | their `VisualToolBoxV2.execute` checks `extract_answer` **first** and returns `done=True`, so the zoom is discarded — 26.0% of cold-start turns, 0.0% of Instruct. `tool_call_rate` overstates real tool use by more than half on some policies |

## 5. What does *not* need worrying about

Verified against `checkpoint/coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged`
rather than assumed. Qwen3-VL uses the **identical ChatML scheme and identical vision
token IDs** as Qwen2.5-VL — `vision_start 151652`, `vision_end 151653`,
`image_token 151655`, `<|im_start|>` / `<|im_end|>`. So the hardcoded observation
splice and the `<image>` → `<|vision_start|><|image_pad|><|vision_end|>` rewrite
transfer verbatim. This was the main worry going in; it is a non-issue.

**Two exceptions found on 2026-11-07, both of which this section originally got wrong:**

* **The token-counting tricks do _not_ transfer.** This section used to say they did.
  Identical token IDs are not sufficient, because our `naive.py:79` decodes with
  `skip_special_tokens=True` and strips them before any count happens — see the
  correction in Prompt 5.
* **Identical naming makes verl's processor discriminator lie.** Qwen3-VL's
  `processor_config.json` declares `image_processor_type: "Qwen2VLImageProcessor"`, so
  verl's recurring `"Qwen2VLImageProcessor" in <class name>` gate is **True for
  Qwen3-VL** and cannot be used to tell the two apart. `rl_dataset.py:380` knows this and
  branches on `Qwen3VLProcessor`; `agent_loop.py:504` — the multi-turn path every tool
  rollout goes through — did not, and imported `qwen2_vl.get_rope_index` unconditionally.
  P2 fixed it and added a tripwire. This is the same failure shape as DeepEyes' own
  `parallel_env.py:282` in §4, arriving from the opposite direction.

What does *not* transfer is geometry (patch-16 / merge-2, hence the 28→32
above) and grounding coordinates (0–1000 relative, hence P4).

## Sources

* [arXiv:2505.14362](https://arxiv.org/abs/2505.14362) — the paper (ICLR 2026)
* [Visual-Agent/DeepEyes](https://github.com/Visual-Agent/DeepEyes) — their code, reference only
* [ChenShawn/DeepEyes-7B](https://huggingface.co/ChenShawn/DeepEyes-7B) — the checkpoint for P1
* [ChenShawn/DeepEyes-Datasets-47k](https://huggingface.co/datasets/ChenShawn/DeepEyes-Datasets-47k) — the data for P6
* [Qwen3-VL Technical Report](https://arxiv.org/pdf/2511.21631) — why P1's gate exists
* [baseline-port-estimates.md](baseline-port-estimates.md) — the estimate this executes
* [laser-fork-qwen3.md](laser-fork-qwen3.md) — how `laser_fork` became Qwen3-VL-native
