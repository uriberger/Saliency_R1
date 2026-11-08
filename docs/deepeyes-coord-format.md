# DeepEyes on Qwen3-VL: the coordinate frame and the format rate — 2026-11-07

[deepeyes-build-prompts.md](deepeyes-build-prompts.md) Prompt 4. Two measurements that
Prompt 3 (the zoom tool) and Prompt 5 (the rewards) both read, taken under **DeepEyes' own
prompt**, character for character, on **Qwen3-VL-8B-Instruct** and on
`checkpoint/coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged`.

Both exist because neither announces itself when it is wrong. A misread coordinate frame
crops the wrong region on every call while the loss curve stays healthy. A format rate
near zero means `R_format = -1` on nearly every rollout while `R_tool`, which pays out
only on a correct answer, never fires at all.

**The headline, in one sentence each.**

> *(filled from the run — see §4 and §5)*

Code: `deepeyes_coord_format_probe.py`, `test_deepeyes_coord_format_cpu.py`,
`launch_deepeyes_coord_format_job.sh` (GPU) / `launch_deepeyes_coord_format_cpu_job.sh`
(the fallback actually used). Branch `probe/deepeyes-coord-format`.

---

## 1. What was asked, and under which prompt

DeepEyes' system prompt and user suffix are embedded in the probe verbatim, including the
two faults in them: `"required":["bbox"]` names a parameter that does not exist (it is
`bbox_2d`), and three lines end in two spaces. The probe refuses to start unless the
embedded strings are character-identical to
`eval/eval_vstar.py:instruction_prompt_system` / `USER_PROMPT_V2` **and** to
`verl/workers/agent/envs/mm_process_engine/prompt.py:PROMPT.SYSTEM_PROMPT_V2` /
`USER_PROMPT_V2` in a local clone — their eval script and their training environment use
the same pair, and if they ever stop agreeing, which one is being measured becomes a
choice that has to be made out loud.

The rendered user turn is theirs too:

```
Question: {question}
Options:
A. ...
B. ...

<newline>Think first, call **image_zoom_in_tool** if needed, then answer. Format strictly as:
  <think>...</think>  <tool_call>...</tool_call> (if tools needed)  <answer>...</answer>
```

Only the **first assistant turn** is generated. The coordinate question is entirely a
turn-1 question, and in a real rollout the vision-token clause of their format rule is
satisfied by construction, because the tool splice writes those tokens, not the model.

Sampled at temperature 1.0, which is what GRPO rolls out at. Greedy overstates format
compliance.

## 2. The corpus, and why HR-Bench is not in the coordinate half

**V\*** (`craigwu/vstar_bench`) carries ground-truth boxes: each item's `.json` has
`target_object` and `bbox` as `[x, y, w, h]` in **original image pixels**. 191 items over
the two scored splits — `direct_attributes` 115, `relative_position` 76. That is the
~200 rows Prompt 4 asked for, and it is the whole of the coordinate half.

**HR-Bench does not carry boxes.** `hr_bench_4k.parquet` and `hr_bench_8k.parquet` are
800 rows each with columns `index, answer, question, A, B, C, D, category,
cycle_category, image` and nothing else. Saying so is the point: the alternative was to
invent a label, and an invented label would have produced a confident number for the one
question Prompt 3 is not allowed to get wrong.

`relative_position` items name **two** target objects and carry two boxes. Each emitted
box is scored against the better of them.

## 3. Three readings, and the arm that separates the last two

Every emitted `bbox_2d` is mapped into original-image pixels three ways and scored against
the ground truth:

| reading | conversion | whose convention |
|---|---|---|
| `rel1000` | `x·W/1000, y·H/1000` | Qwen3-VL's documented grounding frame |
| `abs_orig` | `x, y` | what DeepEyes' `visual_toolbox_v2` assumes |
| `abs_resized` | `x·W/rW, y·H/rH` | Qwen2.5-VL's post-`smart_resize` frame |

`rW, rH` are read back out of `image_grid_thw` rather than recomputed, so they are the
frame the model actually saw.

**Arm A alone cannot separate the last two.** Qwen3-VL's processor config is
`size = {shortest_edge: 65536, longest_edge: 16777216}` — max_pixels 16.7 MP — and
neither DeepEyes' training script nor `laser_fork` sets `max_pixels` anywhere, so verl
hands the model a V\* photo at native resolution. Nothing is downscaled, `abs_resized`
and `abs_orig` coincide to within the 32 px rounding, and a tie between them reads
exactly like a result.

So the probe runs the same prompt at **two resolutions** and turns the question into an
invariance test:

* a 0–1000 relative grid does not move when the processor's resolution does;
* original-image pixels do not move either — but then `abs_orig`, not `rel1000`, is the
  reading with the high IoU in **both** arms;
* post-resize absolute pixels shrink by the resize factor.

One gate enforces that the two arms really differ, because the override that creates them
is a silent no-op if the processor ignores it: the fast image processor only honours
`min_pixels`/`max_pixels` when **both** are passed, and the first version of this probe
passed only `max_pixels` and changed nothing at all.

## 4. Result — the coordinate frame

> *(filled from the run)*

## 5. Result — the format rate

> *(filled from the run)*

## 6. What this means for Prompts 3, 5 and 9

> *(filled from the run)*

## 7. How it was run, and the compromise in it

> *(filled from the run)*

## Sources

* [deepeyes-build-prompts.md](deepeyes-build-prompts.md) — Prompt 4, which this executes
* [baseline-port-estimates.md](baseline-port-estimates.md) § *DeepEyes on Qwen3-VL*
* [Visual-Agent/DeepEyes](https://github.com/Visual-Agent/DeepEyes) — the prompt, the tool
  and the reward, reference only
* [arXiv:2505.14362](https://arxiv.org/abs/2505.14362) — the paper
