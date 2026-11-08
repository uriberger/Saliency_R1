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

## 6. The third thing, which Prompt 4 did not ask for

Their **env** and their **reward** read the same completion differently, and the
difference decides whether `R_tool` can fire at all.

`verl/workers/agent/envs/mm_process_engine/visual_toolbox_v2.py:VisualToolBoxV2.execute`
opens with:

```python
answer = self.extract_answer(action_string)      # re.findall(r'<answer>(.*?)</answer>')
if answer:
    return "", 0.0, True, {}                     # done. the tool call is never read.
action = self.extract_action(action_string)
```

So a turn that carries **both** a `<tool_call>` and a closed `<answer>` ends the episode
without zooming. No image is spliced into the response, so
`count("<|vision_start|><|image_pad|>")` stays 0, so
`tool_reward = 1.0 if count_vision_1 > 0 and acc_reward > 0.5` is 0 — however correct the
answer is. The zoom is discarded silently.

That is not hypothetical under this prompt: both models answer in the turn they call the
tool. §5's `ZOOMS` column is the tool-call rate **after** this filter, and it is the one
to quote.

The two readers also disagree about an *unclosed* `<answer>`. `extract_answer` needs both
tags and ignores it, so the env proceeds to the tool call and the zoom happens.
`compute_score` counts `<answer>` against `</answer>` and marks the imbalance, so the same
completion takes `R_format = -1`. A turn ending `<answer> B` with no close therefore zooms
**and** is penalised.

## 7. What this means for Prompts 3, 5 and 9

> *(filled from the run)*

## 8. How it was run, and the compromise in it

**There were no GPUs.** On 2026-11-07 `oci-nrt-cs-001` had no GPU hardware at all — every
GPU partition reported `TotalNodes=0`, `Nodes=(null)` and no node advertised a `gpu`
GRES, a state it had been in since 2026-10-30. A GPU job submitted into that sits at
`Reason=PartitionConfig` indefinitely and looks exactly like ordinary queueing. The
one-line test:

```fish
sinfo -h -N -o "%G" | grep -ci gpu     # 0 means there is nothing to queue behind
```

So the probe ran on `cpu_short` — 8 nodes, 96 cores each, one worker per node, ~4 h of
wall clock for what a single A100 would have done in under an hour.
`launch_deepeyes_coord_format_job.sh` is the GPU launcher and is the one to use the
moment GPUs return; it writes the same JSONL into the same directory and
`--stage report` reduces either.

Two compromises follow from that, and both are visible in the numbers:

1. **Both arms carry a pixel cap** — arm A at 1,280 visual tokens, arm B at 320 — where
   the GPU design gives arm A the processor's own config. One row at V\*'s native
   resolution (3,290 visual tokens, 3.3k-token prompt) did not finish in 20 minutes on 32
   cores. The three readings stay separable because the two caps differ by 2.0x linear,
   which is all the coordinate question needs. What is lost is that arm A is no longer
   the resolution verl will train at. Nothing in §4 depends on it: the frame the model
   emits in was identical at both resolutions, which is the whole content of the
   invariance test.
2. **`max_new_tokens = 320`**, against their config's 20,480. Truncation is reported as
   its own column, and `format_ok` is given over the untruncated subset as well.

The 2.0x arm separation is asserted, not assumed. The override that creates it is a
silent no-op if the processor ignores it — the fast image processor only honours
`min_pixels`/`max_pixels` when **both** are passed, and the first version of this probe
passed only `max_pixels` and changed nothing. A gate refuses to run below 1.2x, and every
worker log carries the line:

```
arm gate: original (2250, 1500) -> arm A (928, 1376) (1247 vision tokens)
                                vs arm B (448, 672) (294 tokens)  (2.048x linear)
```

`test_deepeyes_coord_format_cpu.py` covers the scoring half on CPU with no model: the
three conversions, their format rule case by case, the tool-call parser, the env's
answer-first behaviour, and — the one that makes the headline trustworthy — a box planted
in each of the three frames in turn, which the probe must read back as that frame.

## Sources

* [deepeyes-build-prompts.md](deepeyes-build-prompts.md) — Prompt 4, which this executes
* [baseline-port-estimates.md](baseline-port-estimates.md) § *DeepEyes on Qwen3-VL*
* [Visual-Agent/DeepEyes](https://github.com/Visual-Agent/DeepEyes) — the prompt, the tool
  and the reward, reference only
* [arXiv:2505.14362](https://arxiv.org/abs/2505.14362) — the paper
