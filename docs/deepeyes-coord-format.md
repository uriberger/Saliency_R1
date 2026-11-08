# DeepEyes on Qwen3-VL: the coordinate frame and the format rate — 2026-11-07

[deepeyes-build-prompts.md](deepeyes-build-prompts.md) Prompt 4. Two measurements that
Prompt 3 (the zoom tool) and Prompt 5 (the rewards) both read, taken under **DeepEyes' own
prompt**, character for character, on **Qwen3-VL-8B-Instruct** and on
`checkpoint/coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged`.

Both exist because neither announces itself when it is wrong. A misread coordinate frame
crops the wrong region on every call while the loss curve stays healthy. A model that
cannot hit their tag structure never zooms, so `R_tool` — **1.2 of a maximum 2.0**
(`0.8·R_acc + 0.2·R_format + 1.2·R_tool`, with `R_format` capped at 0), and the whole
design of the method — never fires at all.

**One correction to the brief before any number.** Prompt 4 states that `R_format` is
"−1 on any tag mismatch", so a model at format 0.000 is *actively penalised on every
rollout*. Their code does not do that. Every clause of `vl_agent.compute_score`'s
`is_format_error` is a **count comparison**, so a completion with no tags at all balances
0 == 0 and scores `R_format = 0.0`. It is not penalised; it is invisible. The −1 is for
opening a tag and failing to close it.

What a tagless model loses is its **accuracy**, and §5 measures that instead. This was
found independently here and in Prompt 5 (`feat/deepeyes-rewards`), which transcribed
their reward and tested it differentially against their own file.

**The headline, in one sentence each.**

> **Coordinates.** Qwen3-VL emits `bbox_2d` as a **0–1000 relative grid against the
> original image** — mean IoU **0.260** against V\*'s boxes under that reading versus
> **0.000** for absolute original pixels and **0.003** for post-`smart_resize` pixels,
> over 158 boxes from both models; 77.2% of boxes touch the target under `rel1000`
> against 3.8% under `abs_orig`, and the emitted numbers do not move when the
> processor's resolution is halved (1.03–1.04× while the resized frame went to 0.488×).
> **DeepEyes' tool reads `bbox_2d` as absolute pixels and is therefore wrong on every
> call.**

> **Format.** Neither model writes `<think>` at all (0.0% on 192 completions), but
> Qwen3-VL-8B-Instruct hits their tool protocol cleanly and the cold start does not:
>
> | | `R_format` ok | zooms their env will run | episode stalls at turn 1 |
> |---|---|---|---|
> | **Qwen3-VL-8B-Instruct** | **100.0%** | **95.8%** | **0.8%** |
> | **cold start** (`...sft_epoch2_lr5e5_merged`) | 75.0% | 42.7% | 15.6% |
>
> **Start DeepEyes from the Instruct model, not from the cold start** — the opposite of
> what Prompt 4 anticipated. §5 and §8.

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

**`rel1000`, by a margin that leaves nothing to interpret.** 158 emitted boxes, both
models, both arms pooled:

| reading | mean IoU | median IoU | IoU ≥ 0.5 | contains the target | touches it at all | mostly off the image |
|---|---|---|---|---|---|---|
| **`rel1000`** | **0.260** | **0.145** | **24.7%** | **60.1%** | **77.2%** | 0.0% |
| `abs_orig` | 0.000 | 0.000 | 0.0% | 1.9% | 3.8% | 0.0% |
| `abs_resized` | 0.003 | 0.000 | 0.0% | 5.7% | 8.2% | 31.0% |

Per model, `rel1000` mean IoU is 0.288 for the Instruct model (92 boxes) and 0.220 for
the cold start (66). `abs_orig` is **0.000 for both**, to three decimals — a V\* photo is
1,500–2,330 px wide, a 0–1000 number read as a pixel lands in the upper-left corner of
it, and the target is almost never there.

Three independent signals, all agreeing:

* **IoU.** 0.260 against 0.000. There is no threshold at which the ranking changes.
* **Containment.** 60.1% of `rel1000` crops swallow the whole target against 1.9% of
  `abs_orig` crops. This matters because IoU is a harsh metric for a zoom — the crops are
  small (median 0.4–0.6% of the image) and a generous crop that catches the object is a
  success scoring IoU 0.2. The two metrics pick the same winner anyway.
* **Off-image, which needs no ground truth.** `abs_resized` pushes 80% of arm-B boxes
  mostly outside the picture (multiplying by `W/rW = 3.35`). `rel1000` puts 0% outside,
  in either arm, for either model.

**The invariance test settles the two absolute readings.** Halving the resolution the
model sees barely moves the numbers it writes:

| model | median raw max coordinate, arm A → B | resized/original width, arm A → B |
|---|---|---|
| cold start | 714 → 746 (**1.044×**) | 0.612 → 0.299 (0.488×) |
| Instruct | 732 → 754 (**1.031×**) | 0.612 → 0.299 (0.488×) |

If the model were emitting post-`smart_resize` pixels, the left column would have tracked
the right one. It did not. Combined with `abs_orig` scoring 0.000, the frame is a
resolution-independent **0–1000 grid against the original image** — exactly what
Qwen3-VL's cookbook says, and the reason this had to be measured rather than assumed: it
is *not* what Qwen2.5-VL does, and DeepEyes' tool was written for Qwen2.5-VL.

For orientation, `rel1000`'s absolute level is modest: median IoU 0.145, and only a
quarter of boxes clear 0.5. The models point at roughly the right place three times in
four and frame it loosely. That is the headroom DeepEyes exists to train, and it is not a
reason to doubt the frame — a wrong frame does not produce 77% partial hits and 0%
off-image boxes.

## 5. Result — the format rate

192 completions, 96 per model, temperature 1.0, first assistant turn.

| model / arm | `R_format` ok | one `<think>` pair | `<think>` at all | tool call | **zooms their env runs** | zoom discarded | truncated |
|---|---|---|---|---|---|---|---|
| cold start / arm A | 75.0% | 0.0% | **0.0%** | 62.5% | **40.6%** | 21.9% | 14.1% |
| cold start / arm B | 75.0% | 0.0% | **0.0%** | 81.2% | **46.9%** | 34.4% | 6.2% |
| Instruct / arm A | **100.0%** | 0.0% | **0.0%** | 96.9% | **96.9%** | 0.0% | 0.0% |
| Instruct / arm B | **100.0%** | 0.0% | **0.0%** | 93.8% | **93.8%** | 0.0% | 0.0% |

Pooled over arms: Instruct 100.0% / 95.8% zooms / 0.0% discarded; cold start 75.0% /
42.7% zooms / 26.0% discarded. Over the completions that were not truncated, the cold
start's `R_format` rate is 80.0% (arm A, n=55) and 76.7% (arm B, n=30).

**Neither model writes `<think>`.** Not once in 192 completions did either model produce
the `<think>…</think>` block the prompt demands: **178 of 192 begin directly with
`<tool_call>`** (87/96 cold start, 91/96 Instruct), and the string `<think>` appears at
all in exactly 2, both times *after* the tool call and neither time closed. The
`tagstruct` column is 0.0% everywhere for that reason alone. Their format rule does not
notice, because 0 `<think>` against 0 `</think>` balances. If any part of this method is
supposed to depend on a visible reasoning chain before the zoom, it is not there to
depend on.

**What the judge is handed, and the one column the recommendation rests on:**

| model / arm | **episode stalls** | no `<answer>` in turn 1 | median answer chars | acc forced to 0 |
|---|---|---|---|---|
| cold start / arm A | **15.6%** | 42.2% | 55 | 9.4% |
| cold start / arm B | **15.6%** | 43.8% | 58 | 3.1% |
| Instruct / arm A | **1.6%** | 98.4% | 149 | 0.0% |
| Instruct / arm B | **0.0%** | 93.8% | 150 | 0.0% |

*Stall* means **neither** a closed `<answer>` **nor** a parseable tool call.
`VisualToolBoxV2.execute` returns `done=True` on both misses, so the episode ends at turn
1 with no zoom and the judge holding the raw chain as the answer. That is the
answer-parse failure, and it is 15.6% of the cold start's rollouts against 0.8% of the
Instruct model's.

The `no <answer>` column must **not** be read as a failure: a model that calls the tool
and defers its answer to turn 2 is doing exactly the right thing and scores ~100% there.
That is why the Instruct model's 98.4% sits beside a 1.6% stall rate.

`acc forced to 0` (the ≥1,000-character clause) is a **floor, not an estimate**. Only
turn 1 was generated and the budget was 320 tokens against their config's 20,480, which
truncates the very chains that would trip the clause. The cold start reaches 9.4% even so;
the Instruct model never does, because its turn-1 output is a 150-character tool call.

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
without zooming. No image is spliced into the response, so no turn is added, so `R_tool`
is 0 however correct the answer is. The zoom is discarded silently.

(Their own detector for "a zoom happened" counts
`<|vision_start|><|image_pad|>` occurrences in the response string, and Prompt 5 found
that detector is **dead in our fork** — `naive.py:79` decodes with
`skip_special_tokens=True` and those ids are `special=True`, so the count is 0 forever.
P5 re-derives `R_tool` from verl's `num_turns`. The argument here is about turns, not
about that count, and survives the substitution: a discarded tool call adds no turn
either.)

That is not hypothetical under this prompt, but it is **specific to the cold start**:
26.0% of its turns carry both, against 0.0% of the Instruct model's. §5's `ZOOMS` column
is the tool-call rate **after** this filter, and it is the one to quote.

The two readers also disagree about an *unclosed* `<answer>`. `extract_answer` needs both
tags and ignores it, so the env proceeds to the tool call and the zoom happens.
`compute_score` counts `<answer>` against `</answer>` and marks the imbalance, so the same
completion takes `R_format = -1`. A turn ending `<answer> B` with no close therefore zooms
**and** is penalised.

## 7. What this means for Prompts 3, 5 and 9

**P3, the zoom tool — convert at the boundary, and crop the un-resized original.**
`bbox_2d` is a 0–1000 grid against the **original** image, so the conversion is
`x·W/1000, y·H/1000` with `W, H` the original size — not the post-`smart_resize` size,
and not the identity the tool currently assumes. Two consequences that are easy to miss:

* The frame is relative to the image the TOOL holds, not the one the model saw, so the
  conversion needs the original's dimensions. DeepEyes gets those from
  `process_raw_image` in `verl/utils/dataset/rl_dataset.py:170`, which keeps a second,
  un-resized copy alongside the `fetch_image`-resized one the processor consumes.
  **`laser_fork` has no `process_raw_image`** — `verl/utils/dataset/vision_utils.py`
  carries only `process_image`, which calls `qwen_vl_utils.fetch_image` and returns the
  resized picture. P3 has to add the second copy; without it there is no original to
  crop and no original size to divide by.
* For a V\* photo the two happen to be close — `IMAGE_MAX_TOKEN_NUM = 16384` gives
  `fetch_image` a 12.8–16.8 MP ceiling and V\* tops out near 3.5 MP, so nothing is
  downscaled — but that is an accident of this corpus. The 47k training set contains
  charts and screenshots, and the gap is real there.

The `log the raw box and the converted box for the first N calls` instruction in P3 stays
exactly as written. This document says what the frame was on 2026-11-07 under this
prompt; it does not make the run self-verifying.

**P5, the rewards — `R_tool` is gated by the env, not by the tool-call rate.** Keep
counting zooms from `num_turns` as P5 already does, and note that on this prompt the two
diverge badly for the cold start: 68.8% of its turns carry a parseable tool call and only
42.7% produce a zoom, because 26.0% are thrown away by an `<answer>` in the same turn
(§6). Reporting `tool_call_rate` as the tool-use rate would overstate it by more than
half. On the Instruct model the two coincide (95.8% / 95.8%), which is exactly why the
discrepancy would go unnoticed if only that arm were smoke-tested.

**P9, the smoke — the three numbers to read, with what they should be.**

1. *tool-call rate*: from the Instruct model, expect it high from step 0 (95.8% here) —
   this model does not need to learn to call the tool.
2. *format rate*: expect 100% on the Instruct model and **no `-1` penalties at all**,
   since the −1 requires an unclosed tag. A format rate near 0 in the smoke therefore
   means something is broken, not that the model is bad at tags.
3. *the raw-vs-converted boxes*: if every converted box is confined to the upper-left of
   the picture and no coordinate ever exceeds 1000 px, the 0–1000 conversion is missing.
   That is the whole visual signature of the bug, and it is what `abs_orig`'s mean IoU of
   0.000 looks like from inside a run.

**And one thing P9 should expect not to see.** No `<think>` block, from either model, on
any rollout. The prompt asks for one and gets none.

## 8. The recommendation, and why it inverts Prompt 4's

Prompt 4 expected the Instruct model near format 0.000 and the cold start high, and asked
for a recommendation to start from the cold start on that basis. **The measurement says
the opposite**, and for a reason specific to what each checkpoint was trained to do:

* **Qwen3-VL-8B-Instruct is already a tool-calling model.** It emits a well-formed
  `<tool_call>` with a `bbox_2d` 95.8% of the time, closes every tag it opens, and
  defers its answer to the next turn as the protocol requires. 0.0% stalls, 0.0%
  discards, 100.0% `R_format`.
* **The cold start was trained out of it.** Saliency-R1's cold start teaches
  `<think>chain</think> answer` — answer in the same turn, no tools — and that is what it
  does here: it calls the tool and answers anyway in 26.0% of turns (killing the zoom),
  or writes a bare untagged chain and stalls in 15.6%. Its 2 epochs of SFT are 2 epochs
  of pressure *away* from the behaviour DeepEyes needs.

The `build_coldstart_data.py:52` note that Qwen3-VL-8B-Instruct "scores format 0.000
under every prompt variant tried" is about **LASER's four-tag
`<think></think><answer></answer>` format**, which is a different protocol with a
different reward. It does not transfer, and it is not evidence about this one.

So: **run DeepEyes from `Qwen/Qwen3-VL-8B-Instruct`.** The paper's "no cold start needed"
claim happens to hold on this backbone — but note in the writeup that it holds for a
reason the paper did not have, namely that Qwen3-VL already received agentic
tool-calling SFT from the vendor (its tech report describes ~120k distilled multi-turn
agentic interactions plus tool-integrated RL), which is the same treatment DeepEyes
applies. It is not that no cold start is needed; it is that one was already done.

If the arm is run from the cold start anyway — for comparability with the other
baselines, which is a legitimate reason — expect roughly **half the rollouts to produce
no zoom at all**, and budget for `R_tool` to be sparse for the first part of the run.

## 9. How it was run, and the compromise in it

**There were no GPUs.** On 2026-11-07 `oci-nrt-cs-001` had no GPU hardware at all — every
GPU partition reported `TotalNodes=0`, `Nodes=(null)` and no node advertised a `gpu`
GRES, a state it had been in since 2026-10-30. A GPU job submitted into that sits at
`Reason=PartitionConfig` indefinitely and looks exactly like ordinary queueing. The
one-line test:

```fish
sinfo -h -N -o "%G" | grep -ci gpu     # 0 means there is nothing to queue behind
```

So the probe ran on `cpu_short` — 8 nodes, 96 cores each, one worker per node. 192
completions in **66 minutes** of wall clock (job array `7742590`, 19–66 min per task),
at ~150–190 s a row for arm A and ~35 s for arm B, against the hour on one GPU Prompt 4
budgeted. `launch_deepeyes_coord_format_job.sh` is the GPU launcher and is the one to use
the moment GPUs return; it writes the same JSONL into the same directory and
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

### Reproducing

```fish
# the scoring half, no GPU, no model, no network — seconds
python test_deepeyes_coord_format_cpu.py

# the real thing, once GPUs exist again
bash launch_deepeyes_coord_format_job.sh --name de-coordfmt --gpus 8 --duration 2 \
    --out-dir outputs/deepeyes_coord_format/probe-deepeyes-coord-format

# what was actually run on 2026-11-07
bash launch_deepeyes_coord_format_cpu_job.sh --name de-coordfmt-cpu \
    --out-dir outputs/deepeyes_coord_format/probe-deepeyes-coord-format

# re-reduce whatever is on disk
python deepeyes_coord_format_probe.py --stage report \
    --out-dir outputs/deepeyes_coord_format/probe-deepeyes-coord-format
```

Outputs: `report.txt`, `summary.json`, `scored.jsonl` (one row per completion, with the
three conversions and every flag) and `raw/*.jsonl` (the completions themselves) under
`outputs/deepeyes_coord_format/probe-deepeyes-coord-format/`.

## Sources

* [deepeyes-build-prompts.md](deepeyes-build-prompts.md) — Prompt 4, which this executes
* [baseline-port-estimates.md](baseline-port-estimates.md) § *DeepEyes on Qwen3-VL*
* [Visual-Agent/DeepEyes](https://github.com/Visual-Agent/DeepEyes) — the prompt, the tool
  and the reward, reference only
* [arXiv:2505.14362](https://arxiv.org/abs/2505.14362) — the paper
