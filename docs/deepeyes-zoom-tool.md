# The DeepEyes zoom tool on Qwen3-VL

Execution notes for [deepeyes-build-prompts.md](deepeyes-build-prompts.md) **Prompt 3**.
What changed, what was decided, and what the later prompts have to do differently
because of it (§6 — three of those will look like breakage if nobody says so first).

| file | what it is |
|---|---|
| `deepeyes/image_zoom_in_tool.py` | the tool. Tracked source; the copy that executes is in `deepeyes_fork/` |
| `deepeyes/zoom_tool_config.py` | the tool YAML, the `create_kwargs` builder, and the turn budget |
| `patch_deepeyes_zoom_tool.sh` | installs it into `deepeyes_fork/`; `--check` reports |
| `test_deepeyes_zoom_tool_cpu.py` | 48 CPU checks against the **installed** copy |
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

### It is a switch, not a constant — and P4 has now set it

The frame is one config key with three readings and **no default**:

```yaml
bbox_frame: rel1000          # (x/1000)*W, (y/1000)*H     <-- MEASURED CORRECT
bbox_frame: abs_original     # x, y                       DeepEyes' assumption
bbox_frame: abs_smartresize  # (x/rW)*W,   (y/rH)*H       Qwen2.5-VL's frame
```

Unset or unrecognised raises at construction with the reason. P4's own spellings
(`abs_orig`, `abs_resized`) are accepted as aliases, so its verdict can be pasted in
without a translation step — a translation step is where a frame gets flipped.

**P4 reported: `rel1000`, decisively.** Over 158 emitted boxes against V*'s ground
truth:

| reading | mean IoU | touches target |
|---|---|---|
| **`rel1000`** | **0.260** | **77.2%** |
| `abs_orig` | 0.000 | 3.8% |
| `abs_resized` | 0.003 | 8.2% |

and the invariance arm settles it independently: halving the resolution the model sees
moved the emitted numbers by **1.03–1.04×** — i.e. invariant, therefore relative —
while the resized-frame reading moved 0.488×.

So **DeepEyes' `visual_toolbox_v2`, which reads `bbox_2d` as absolute pixels, is wrong
on every call for this backbone.** That is the single most valuable thing the
switchable design bought: the number is now set by measurement rather than by whichever
of the two plausible readings got typed first, and setting it was one line. Everything
in §5 below was already passing under `rel1000`; nothing moved.

Keep the key required. The default stays absent on purpose — the next backbone will
have its own answer, and a default is how a measured fact becomes an inherited
assumption.

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

### 3.1 Where the original comes from, given this fork has no `process_raw_image`

P4 raised this and it is the right question: DeepEyes keeps a second un-resized copy
in the **dataloader** — `process_raw_image` beside `process_image`, carried as an
`origin_multi_modal_data` column, handed to the tool at `parallel_env.py:479` — and
our fork's `vision_utils.py` has only `process_image`. If nothing upstream held
un-resized pixels, the assertion in §3 would not be a guard, it would be a tripwire
that fires on every call.

**It is a guard. The second copy exists; verl 0.7 just keeps it somewhere else.**
`RLHFDataset` passes the row's own `extra_info.tools_kwargs` through **untouched** —
`rl_dataset.py:445` reads it out of `extra_info`, `:451` assigns it, and
`process_image` never goes near it. The traced path for one row:

```
parquet row
  .images[0] ──────────────── process_image() ─ smart_resize ─→ multi_modal_data["image"]
                                                                 THE MODEL'S VIEW
  .extra_info.tools_kwargs.image_zoom_in_tool.create_kwargs.image
            (a SECOND copy of the same raw bytes, written by build_create_kwargs)
    → rl_dataset.py:451        row_dict["tools_kwargs"] = tools_kwargs
    → collate_fn               batch.non_tensor_batch["tools_kwargs"]
    → agent_loop.py:380        kwargs = {k: v[i] for k, v in non_tensor_batch.items()}
    → tool_agent_loop.py:125   tools_kwargs = kwargs.get("tools_kwargs", {})
    → tool_agent_loop.py:443   tool.create(create_kwargs=...)
    → load_image_unresized()   NO RESIZE ANYWHERE ON THIS PATH
                                                                 THE CROP SOURCE
```

So `process_raw_image` is **not** needed and nothing in `deepeyes_fork/` had to change
for it. Adding it would mean touching `vision_utils.py`, `rl_dataset.py`, `collate_fn`,
the agent-loop batch plumbing and `ToolAgentLoop.run` — five files in a shared fork —
to re-derive per batch what two lines in the data builder already carry.

**Proved on real code, not read off it.** `test_deepeyes_zoom_tool_cpu.py` CASE 9 runs
the real `RLHFDataset` with the real Qwen3-VL processor on a 1000×700 row:

```
images[0] (1000, 700) -> multi_modal_data (992, 704)     the prompt image IS resized
create_kwargs survives with original_size [1000, 700]    tools_kwargs is NOT
tool holds (1000, 700); model saw (992, 704)             two different images
```

and the agent-loop job's **A5** closes the same loop at the far end: decode the raw
`images` column independently and check the sizes the tool logged are those.

**The trade, stated.** The image bytes appear twice in the parquet, so the 47k corpus
goes from 4.84 GB to roughly 9.7 GB. That is P6's cost to pay, and it is cheaper than
five edits to a shared fork. If it ever becomes a problem the fix is to put a
**path** in `create_kwargs["image"]` rather than a data URI — the loader already
accepts paths, `file://` and `http(s)://` — not to re-plumb the dataloader.

**The one-command check for the real corpus**, which is what P6 should run before the
real run rather than discovering this at step 40:

```fish
python deepeyes/zoom_tool_config.py verify <parquet> --n 50
```

It checks three things per row against a *third* independent witness, the raw `images`
column: that `create_kwargs.image` decodes — through the tool's own decoder — to the
raw size, that the declared `original_size` matches both, and that the declared
`view_size` is what the dataloader will really show at `image_patch_size=16`. It also
reports the downscale distribution, which answers the question the assertions cannot:
not *is this the original* but *is the original big enough for a zoom to recover
anything*. A corpus at downscale 1.0 makes the tool decoration however correctly it is
wired.

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

### 4 is a ceiling on *executed* calls, and it is not the number emitted

P4 found that `VisualToolBoxV2.execute` checks `extract_answer` **first** and returns
`done=True` if it finds a closed `<answer>`:

```python
answer = self.extract_answer(action_string)
if answer:
    return "", 0.0, True, {}        # episode over; any <tool_call> in the same turn
action = self.extract_action(...)   # is never reached
```

So a turn carrying *both* a `<tool_call>` and a closed `<answer>` ends the episode and
the zoom is **discarded**. P4 measured that at **26.0% of cold-start turns and 0.0% of
Instruct turns** — i.e. it is a property of the policy we are actually starting from,
not a rare edge case.

Three consequences, none of which change code here:

1. **"4 tool calls" is a ceiling on executions, not on emissions.** On the cold start,
   expect the mean executed count to sit well below any count of emitted
   `<tool_call>` blocks. Do not read a gap between the two as a bug.
2. **verl behaves differently here, and more sensibly.** `ToolAgentLoop` has no
   answer-first short-circuit: it runs the tool parser over the response and dispatches
   whatever tool calls it finds (`tool_agent_loop.py:245,257`). So on our stack the
   zoom in such a turn *is* executed and the episode continues. That is a deviation
   from DeepEyes in our favour, and it should be stated rather than discovered — it
   makes our `R_tool` fire on turns where theirs would not.
3. **P5 should count executions, not emissions.** The build doc already says to derive
   "a zoom happened" from verl's `num_turns` rather than from counting
   `<|vision_start|><|image_pad|>` in the response string. This is a second reason:
   `num_turns` counts what the loop actually did. For the tool-call *rate* in the
   smoke, log both and report the gap.

---

## 5. Results

### Unit, on the login node, no GPU and no ray

```fish
conda activate deepeyes
bash patch_deepeyes_zoom_tool.sh
python test_deepeyes_zoom_tool_cpu.py          # 48/48
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
off); the YAML path verl actually uses; the raw-vs-converted log; and **the supply
path through the real `RLHFDataset` and the real Qwen3-VL processor** (§3.1), including
`verify_zoom_corpus` passing on a correct corpus and failing on one built from the
resized copy.

### Integration, through the real `ToolAgentLoop`, on `cpu_short`

Job **7742614**, 4 rollouts of geo3k, wrapping P2's `deepeyes_agentloop_cpu.py`:

```
48/48 unit checks
A1 PASS  position_ids (4, 4, 6144), t!=h on 4/4, markers {'qwen3_vl': 1}
A2 PASS  0.610 of the response masked; 4 vision spans, 0 leaking
A3 PASS  2 grid rows per row, image_pad count == grid token count on 4/4
A4 PASS  4 tool calls under frame rel1000, all ok
A5 PASS  the tool held the RAW parquet image, not the one the model saw
     tool held       [(207,182), (250,258), (263,280), (353,310)]
     raw images[0]   [(206,215), (207,182), ..., (543,250)]   <- all four are in here
     model was shown [(192,192), (256,256), (256,288), (352,320)]
     verify_zoom_corpus: ok=True downscale={min 0.952, median 0.999, max 1.016}
     WARNING 9/16 rows have the original no bigger than the view.
```

Two things worth reading off that. The four sizes the tool held are raw sizes from the
parquet and **none of them is a size the model was shown** — §3.1, demonstrated rather
than argued. And geo3k's median downscale is 0.999: on 9 of 16 rows the "zoom"
*upscales*. geo3k is a stack proof, not a zoom corpus; `min_expected_downscale` would
reject it outright and the corpus warning fires without being asked. That is the knob
working, and it is exactly the check P6 needs to run on the real 47k.

---

## 6. What this changes for the other prompts

Three of these will look like breakage if nobody says so.

1. **P2.** `deepeyes_tool_smoke.py::build_tool_config` predates `bbox_frame`, so
   re-running P2's harness against the patched fork now **fails at tool
   construction** with the explanatory error. That is the no-silent-default working.
   The fix is one keyword; after both branches land, P2's helper should just call
   `deepeyes/zoom_tool_config.py::build_tool_config`.
2. **P6.** `extra_info.tools_kwargs[...]["create_kwargs"]` must now carry
   `original_size` and `view_size` alongside `image`, built from the **raw** bytes.
   Use `build_tools_kwargs`; it is two lines and it is right by construction.
   `build_deepeyes_smoke_data.py` passes only `image` and needs the same change. Then
   run `python deepeyes/zoom_tool_config.py verify <parquet> --n 50` on the output and
   read the `downscale` block — see §3.1. This is also where the corpus doubles in
   size (≈4.84 GB → ≈9.7 GB); if that is a problem, store a path rather than a data
   URI, not a second dataloader column.
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
