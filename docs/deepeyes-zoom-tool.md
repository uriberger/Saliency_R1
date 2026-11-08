# The DeepEyes zoom tool on Qwen3-VL

Execution notes for [deepeyes-build-prompts.md](deepeyes-build-prompts.md) **Prompt 3**.
What changed, what was decided, and what the later prompts have to do differently
because of it (§6 — three of those will look like breakage if nobody says so first).

| file | what it is |
|---|---|
| `deepeyes/image_zoom_in_tool.py` | the tool. Tracked source; the copy that executes is in `deepeyes_fork/` |
| `deepeyes/zoom_tool_config.py` | the tool YAML, the `create_kwargs` builder, and the turn budget |
| `patch_deepeyes_zoom_tool.sh` | installs it into `deepeyes_fork/`; `--check` reports |
| `test_deepeyes_zoom_tool_cpu.py` | 43 CPU checks against the **installed** copy |
| `deepeyes_zoom_agentloop_cpu.py` | the same tool inside the real `ToolAgentLoop`, via P2's harness |
| `launch_deepeyes_zoom_agentloop_cpu_job.sh` | sends both to `cpu_short` |

`verl/tools/image_zoom_in_tool.py` was already a verl `BaseTool` with working
crop/validate logic, so this was adaptation rather than a port. Three numbers in it
were Qwen2.5-VL's.

---

## 1. The coordinate frame

`execute` read `bbox_2d` and cropped with it as absolute pixels. That is correct for
Qwen2.5-VL, which emits absolute pixels in the **post-`smart_resize`** frame.
Qwen3-VL's cookbook documents a **0–1000 relative grid against the original image**.
On a 1600×1200 photo those two readings are about a factor of two apart in each linear
dimension, and the failure is completely silent: a crop of the wrong region is a valid
image, the vision tokens are well formed, the loss falls.

### It is a switch, not a constant

P4 (`deepeyes_coord_format_probe.py`) measures which frame Qwen3-VL actually emits
under DeepEyes' prompt. **It has not reported** — it needs a GPU and this cluster has
none; its CPU fallback was still running at the time of writing. So the frame is not
hardcoded. It is one config key with three readings and **no default**:

```yaml
bbox_frame: rel1000          # (x/1000)*W, (y/1000)*H     Qwen3-VL cookbook
bbox_frame: abs_original     # x, y                       DeepEyes' assumption
bbox_frame: abs_smartresize  # (x/rW)*W,   (y/rH)*H       Qwen2.5-VL's frame
```

Unset or unrecognised raises at construction with the reason. P4's own spellings
(`abs_orig`, `abs_resized`) are accepted as aliases, so its verdict can be pasted in
without a translation step — a translation step is where a frame gets flipped.

**When P4 reports, the change is one line in the tool config.** Nothing else moves.

### Which one I would bet on, and why: `rel1000`

Not baked in, but worth recording so the bet can be scored:

1. The vendor post-trained this convention in. The Qwen3-VL tech report describes
   synthesising grounding traces and SFT-ing a "think → act → analyse → answer" visual
   agent, and the cookbook that ships with it states 0–1000 relative. Qwen2.5-VL's
   absolute-pixel frame was the odd one out between Qwen2-VL and Qwen3-VL, not a
   family property.
2. The DeepEyes arm starts from
   `checkpoint/coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged`, whose SFT
   corpus is not a grounding corpus. The base model's grounding prior survives it.
3. The one argument the other way is DeepEyes' own system prompt, which carries the
   literal example `{"bbox_2d": [10, 20, 100, 200]}`. Those numbers read as absolute
   pixels, and a model that copies the example's magnitude would look like
   `abs_original`. A single in-context example rarely overrides a post-trained output
   convention — but this is exactly the kind of thing that makes measuring it the
   right call rather than reading the docs.
4. `abs_original` and `abs_smartresize` are nearly the same frame at Qwen3-VL's
   default `max_pixels` of 16,777,216, because no real downscale happens. P4's arm B
   exists to separate them and that is the only thing that can.

### The frame is visible in the run, not assumed in a file

The first `log_first_n_calls` calls per process print and append JSONL:

```
[DEEPEYES-ZOOM] call=0 frame=rel1000 raw=[200, 150, 800, 700]
  -> converted=[50.0, 38.7, 200.0, 180.6] -> cropped=[50.0, 38.7, 200.0, 180.6]
  original=[250, 258] view=[256, 256] crop=[150, 142] status=ok
  plausible=['abs_original', 'rel1000']
```

`plausible` is a free per-call diagnostic: which frames the raw numbers *could* have
come from, by extent alone. It decides nothing on one call. Aggregated over a few
hundred, it is the cheap version of P4 and is available from any run: a model emitting
a 0–1000 grid is almost never plausible as `abs_original` on a 2000 px photo, and a
model emitting absolute pixels is almost never plausible as `rel1000`.

`log_crops: N` additionally dumps the crop and the source with the box drawn, which is
what P9 step 3 ("eyeball ten of them against the image") actually needs.

---

## 2. `MIN_DIMENSION = 28` is a 32

Qwen2.5-VL is patch-14 / merge-2: one visual token covers 28×28 px, and their 28 px
floor is a **one-token floor**. Qwen3-VL is patch-16 / merge-2 — 32×32 px per token —
so 28 is *sub-token* here and the floor stops meaning what it meant. It is now derived
as `patch_size * merge_size`, and a configured `min_dimension` below one token is
refused rather than accepted.

On *"restate any `N×28×28` budget as `N×32×32`"*: **there is no such budget in this
file**, and DeepEyes does not cap the crop either — the only 28 was the floor. The
knobs exist anyway (`max_crop_tokens` / `min_crop_tokens`), denominated in **tokens**
rather than pixels so the question cannot come back, and both default to `None`, which
is DeepEyes' behaviour.

### The same mistake, one layer up, is live

verl's dataloader has its own copy of this number and it is still 14:

```
verl/utils/dataset/rl_dataset.py:108      self.image_patch_size = config.get("image_patch_size", 14)
verl/trainer/config/data/legacy_data.yaml:71   image_patch_size: 14
```

Nothing derives it from the processor. So on Qwen3-VL the **prompt** image is
smart-resized onto a 28 px grid unless the launcher says otherwise.

> **P7 must set `data.image_patch_size=16`.**

The tool checks this indirectly: it recomputes the view the model was shown and fails
the run if the data's stated `view_size` disagrees with it.

---

## 3. Crop the original, and assert it

This is the whole method. DeepEyes' own tool
(`verl/workers/agent/envs/mm_process_engine/visual_toolbox_v2.py:reset`) crops
`origin_multi_modal_data`, which their dataloader builds with `process_raw_image` — a
bare `Image.open(...).convert("RGB")`, no `smart_resize` — held alongside the
`process_image` copy the model is shown. A crop taken from the already-downsampled
copy recovers **no detail**, produces a plausible image, and turns the arm into
expensive plain GRPO.

verl's version called `qwen_vl_utils.fetch_image` inside `create`, which ends in
`image.resize(...)`. This one never resizes on load, and refuses an image spec that
carries resize instructions (`max_pixels`, `resized_height`, …).

### The assertion, and why it needs the data's help

"Is this image the original?" is **not answerable from the pixels**: an already
smart-resized image looks exactly like an original that happened to need no resize. So
the data builder states the raw size and the tool checks it, fatally:

```python
create_kwargs = {
    "image":         <original bytes / path / PIL, un-resized>,
    "original_size": [W, H],    # read from the RAW bytes at build time
    "view_size":     [vW, vH],  # what the dataloader showed the model
}
```

`deepeyes/zoom_tool_config.py::build_create_kwargs` produces this, deriving the two
sizes by two different computations from the raw image — so a conversion that reuses
one already-processed variable for both cannot pass. `crop_source_check: strict` is
the default and `warn`/`off` exist for smoke data only.

`min_expected_downscale` is the corpus-level version of the same question: set it and
a corpus whose images are not meaningfully bigger than what the model already sees
fails the run instead of quietly making the tool decoration. It is unset by default.

---

## 4. Fidelity: `max_turns` is **5**, meaning **4 tool calls**

The paper says *"up to a maximum of 6 times of active perceptions"*. Their released
training script sets `agent.max_turns=5`:

```
examples/agent/final_merged_v1v8_thinklite.sh:56   actor_rollout_ref.rollout.agent.max_turns=5
```

and their loop counts **generate rounds**, skipping the tool on the last one:

```
verl/workers/agent/parallel_env.py:181   for step in range(config.agent.max_turns):
verl/workers/agent/parallel_env.py:226   if done or step == config.agent.max_turns - 1:
                                             active_mask[idx] = False     # no tool call
```

So `max_turns=5` is 5 generate rounds and **at most 4 tool calls**, not 6 perceptions.

**We follow the code, not the sentence.** It is what the released checkpoint was
trained under, and P1 scores that checkpoint, so both arms of the comparison sit on
the same budget. Recorded once, in `deepeyes/zoom_tool_config.py`:

```python
MAX_ASSISTANT_TURNS = 5   # actor_rollout_ref.rollout.multi_turn.max_assistant_turns
MAX_USER_TURNS      = 4   # actor_rollout_ref.rollout.multi_turn.max_user_turns
MAX_TOOL_CALLS      = 4
```

Modern verl spells the budget with two keys instead of one
(`verl/experimental/agent_loop/tool_agent_loop.py:239,241`); `max_assistant_turns`
alone reproduces their loop exactly, because the check runs after the generate and
before the tool call, same as theirs. `max_user_turns` is the implied bound, stated
rather than inferred.

**For the writeup:** this is a stated deviation from the paper's prose and not from
their code. If the paper's 6 is wanted instead, it is `max_assistant_turns=7` /
`max_user_turns=6` — and say so, because it is not what their checkpoint saw.

---

## 5. Results

### Unit, on the login node, no GPU and no ray

```fish
conda activate deepeyes
bash patch_deepeyes_zoom_tool.sh
python test_deepeyes_zoom_tool_cpu.py          # 43/43
```

It imports `verl.tools.image_zoom_in_tool` — the copy inside `deepeyes_fork/` that a
training run executes — and refuses to run if that copy has drifted from the tracked
source. A test that imported the tracked file directly would pass on a fork nobody
patched.

Covered: conversion in all three frames mapped onto one region of one image; the same
numbers under the wrong frame cropping a different region with no error; the 32 px
floor and the refusal of a 28; out-of-bounds (clamped when overlapping, `-0.05` when
not); degenerate (zero area, inverted, aspect > 100, malformed `bbox_2d`); the crop
source (no resize on load, the resized copy rejected, a missing `original_size`
rejected, resize instructions rejected, and the no-op demonstrated with the check
off); the YAML path verl actually uses; and the raw-vs-converted log.

### Integration, through the real `ToolAgentLoop`, on `cpu_short`

Job **7742599**, 4 rollouts of geo3k, wrapping P2's `deepeyes_agentloop_cpu.py`:

```
41/41 unit checks
A1 PASS  position_ids (4, 4, 6144), t!=h on 4/4, markers {'qwen3_vl': 1}
A2 PASS  0.610 of the response masked; 4 vision spans, 0 leaking
A3 PASS  2 grid rows per row, image_pad count == grid token count on 4/4
A4 PASS  4 tool calls under frame rel1000, all ok
```

Worth reading off the log: geo3k's images are 250×258 and the view is 256×256, so the
"zoom" **upscales** — `downscale` 0.99. geo3k is a stack proof, not a zoom corpus, and
`min_expected_downscale` would reject it. That is the knob working.

---

## 6. What this changes for the other prompts

Three of these will look like breakage if nobody says so.

1. **P2.** `deepeyes_tool_smoke.py::build_tool_config` predates `bbox_frame`, so
   re-running P2's harness against the patched fork now **fails at tool
   construction** with the explanatory error. That is the no-silent-default working.
   The fix is one keyword; after both branches land, P2's helper should just call
   `deepeyes/zoom_tool_config.py::build_tool_config`.
2. **P6.** `extra_info.tools_kwargs[...]["create_kwargs"]` must now carry
   `original_size` and `view_size` alongside `image`. Use `build_tools_kwargs`; it is
   two lines and it is right by construction. `build_deepeyes_smoke_data.py` passes
   only `image` and needs the same change.
3. **P5.** The tool returns `-0.05` on a malformed or unusable box, kept from verl's
   version. `ToolAgentLoop` collects those into `output.extra_fields["tool_rewards"]`
   (`tool_agent_loop.py:184`) and does **not** add them to the reward — it exposes
   them. DeepEyes' composite `0.8*R_acc + 0.2*R_format + 1.2*R_tool` has no such term,
   so leave it out of the sum and log it: a rising rate of `-0.05` is the symptom of a
   wrong `bbox_frame` that P9 would otherwise have to eyeball for.
4. **P7.** Set `data.image_patch_size=16`, `multi_turn.max_assistant_turns=5`,
   `multi_turn.max_user_turns=4`, and a `bbox_frame` in the tool config. Point
   `log_dir` at the run's output directory and set `log_crops` to ~10 for the smoke.
5. **P9.** The raw-vs-converted lines and the `plausible=` field are step 3 of the
   smoke checklist. If P4 still has not reported by then, aggregate `plausible_frames`
   over the smoke's tool calls — it will not be as good as P4's IoU, but it is
   decisive against a frame that is grossly wrong.

## Sources

* [arXiv:2505.14362](https://arxiv.org/abs/2505.14362) — the paper
* [Visual-Agent/DeepEyes](https://github.com/Visual-Agent/DeepEyes) — their code, reference only; cloned to `/home/uberger/scratch/research/DeepEyes`
* [deepeyes-build-prompts.md](deepeyes-build-prompts.md) §3 Prompt 3, §4 traps 3–5 and 9
* [laser-fork-qwen3.md](laser-fork-qwen3.md) — how the verl under this became Qwen3-VL-native
