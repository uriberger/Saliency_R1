# ERGO Gate B — will the cold start emit `<zoom>`, and in which coordinate frame?

Companion to [ergo-build-prompts.md](ergo-build-prompts.md) § *Prompt 2*, and one of the
two gates that stand in front of the whole ERGO arm. Sections 1–6 are the design and the
instrument, written before the run; **section 7 is the result**, and section 8 is what to
do with it.

ERGO ([arXiv:2509.21991](https://arxiv.org/abs/2509.21991), ICLR 2026,
[nota-github/ERGO](https://github.com/nota-github/ERGO), Apache 2.0) is two turns:

```
turn 1   <think> … </think><zoom>[x1, y1, x2, y2]</zoom>
         -> the box is cropped out of the ORIGINAL image and enlarged
turn 2   <think> … </think><answer> … </answer>      conditioned on BOTH pictures
```

`r_format` is 1 only if **both** turns match, and ERGO's rule is that a turn-1 completion
with no parseable box scores **zero on every term**. Two things can therefore be wrong
without anything failing.

---

## 1. Failure one: the format is never produced

`build_coldstart_data.py:52` records Qwen3-VL-8B-Instruct at **format 0.000 under every
`<think>` prompt variant tried** — that is the entire reason our cold start exists. ERGO
asks for strictly more than that: `<think>` in *both* turns, plus a `<zoom>` tag in turn 1
that our cold-start corpus has never contained.

If turn 1 is malformed on most rollouts, every reward in the GRPO group is 0, the group
advantage is **exactly** 0, and there is no gradient at all. Nothing raises. The loss
curve looks like a loss curve. This is the same shape of failure `coldstart_gate.py`
exists for, one layer out.

## 2. Failure two: the box is read in the wrong frame

Qwen2.5-VL — the backbone ERGO was built on — emits absolute pixels in the
**post-`smart_resize`** frame. That is why `src/ergo/utils.py:map_box` rescales a box by
`original / resized`:

```python
left  = max(left  * width_scale, 0)          # width_scale = width / resize_w
right = min(right * width_scale, ori_width)
```

Qwen3-VL is documented to emit a **0–1000 relative grid against the original image**
instead. Port `map_box` unchanged and every crop lands somewhere else in the picture —
while the pipeline stays healthy:

- `r_box` (crop area ≤ γ) still fires, because a misplaced box is still a small box;
- `r_acc` degrades a little, which looks like an ordinary hard task;
- `r_region` — the frozen reward model shown **only the crop** — stops firing, which is
  indistinguishable from "the reward model is strict".

The policy then optimises box **area**, which is `r_box`, which is the half of `r_TCE`
that is not the contribution. The paper's one idea is silently removed and the training
curve does not say so. It is the first row of the build document's trap table for a
reason.

## 3. Why V\*, and why only V\*

The frame question can only be answered against ground-truth boxes, and of ERGO's three
benchmarks only one has them.

| benchmark | boxes? | |
|---|---|---|
| **V\*** (`craigwu/vstar_bench`) | **yes** | one `.json` per image: `target_object`, `bbox` as `[x, y, w, h]` in **original pixels**. 191 questions, 115 `direct_attributes` + 76 `relative_position`; 137 single-target, 54 two-target (unioned). Median image 2250×1500 |
| HR-Bench (`DreamMr/HR-Bench`) | no | the parquet is `index, answer, question, A, B, C, D, category, cycle_category, image` and nothing else |
| MME-RealWorld-lite | no | — |

Note the mirror trap: `lmms-lab-encoder/vstar-bench`, which the lmms-eval task uses, ships
the images and `test_questions.jsonl` **without** the per-image annotation files. The gate
pulls `craigwu/vstar_bench`, which is the original and carries them.

191 rows is the "~200" the build document asks for, and they all carry boxes, so the parse
rate, the format rates and the frame verdict are all measured on the same set.

## 4. The instrument

`ergo_zoom_gate.py`, in the shape of `coldstart_gate.py` — a cheap check on the two
mechanisms a reward depends on, not a quality metric. One GPU, ~1–2 h for both models.

```fish
bash launch_ergo_zoom_gate_job.sh \
    --model checkpoint/coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged \
    --reference Qwen/Qwen3-VL-8B-Instruct
```

**Two passes, on purpose.** Turn 1 is generated for every row first, the frame is decided
from the **aggregate**, and only then is turn 2 generated with the crop taken under the
winner. Choosing the frame per row from that row's own IoU would be reading the answer
key, and the trainer cannot do it either — it has one global convention or it has nothing.

**ERGO's prompts verbatim**, vendored into the gate and diffed back against the clone by
`_check_ergo_sources` whenever the clone is present, the way `coldstart_gate` diffs its
regex against the trainer. That includes their typo: `QUESTION_TEMPLATE2` asks for
`<think> ... <think>`, where the second tag is not a closing tag. Fixing it would measure
a prompt ERGO-7B was never trained under, so it is reproduced exactly and the turn-2 rate
is reported under **both** the strict reading and the lenient one their instruction would
actually accept.

**Temperature 1.0, not greedy**, because that is what GRPO generates at and greedy
overstates format compliance. Same reasoning as `coldstart_gate.py`.

**`--reference` is not optional.** The cold start's parse rate alone cannot separate "this
recipe produces no boxes" from "this *checkpoint* produces no boxes" — the Qwen3-VL tech
report describes tool-integrated RL with an `image_zoom_in_tool` in the backbone's own
post-training, so the ability may be there and may have been cold-started out. Those two
have completely different fixes.

### The 28 that is a 32

Every budget in ERGO's code and tables is written `N × 28 × 28` — Qwen2.5-VL's patch-14 /
merge-2 token. Qwen3-VL is **patch-16, merge-2: one visual token covers 32 × 32 px**
(`vlm_family.Qwen3VL.patch_px`, independently). A literal copy of `1280 × 28 × 28` buys
**973** visual tokens here where the table says 1280 — 23% short, i.e. every row of an
efficiency table at the wrong operating point. The gate uses 32 and pins the discrepancy
as a test.

The gate also asserts that the processor did not re-resize the image under it, by reading
`image_grid_thw` back and comparing against its own `smart_resize` output
(`processor_resized_under_us`, which must be 0). If it is not 0, the "resized" frame is not
the frame the model saw and every number in the frame table is measuring something else.

## 5. What it reports

**Turn 1** — `<zoom>` tag present; box parses under **ERGO's own**
`parse_zoom_bbox_from_text` (integers only, inside the tags); box parses **leniently**
(floats, outside the tag, several boxes); `<think>` exactly once; the joint turn-1 format.
The strict rate is the gate — it is the function the reward would call. The gap between
strict and lenient is the diagnosis: "wrote floats" and "wrote no box" are the same number
on the strict reading and have different fixes.

**Turn 2** — strict `<think>…</think><answer>…</answer>`, lenient, and `<answer>` tags at
all. Plus **both turns well-formed**, which is `r_format` and the floor under every other
reward.

**The frame**, three ways, deliberately kept apart because they can disagree:

| | what it is | why it is here |
|---|---|---|
| **structural** | how many emitted coordinates exceed 1000, and how many exceed the resized side | needs no ground truth and no model competence. A 0–1000 grid *cannot* emit a coordinate above 1000. Survives a checkpoint that cannot find the object at all |
| **IoU** | mean/median against the GT box under each reading | what the build document asks for, and what the reward experiences |
| **centre error** | ‖predicted centre − GT centre‖ / image diagonal | the only one **not** confounded by crop size. V\*'s targets are a fraction of a percent of the picture, so a crop containing the whole answer still scores IoU ≈ 0 — the CPU test pins exactly that case. A frame read at the wrong scale pulls every box toward the top-left, and this sees it where IoU reads it as "a smaller box that missed" |

Reported alongside: **GT coverage** (the fraction of the GT box inside the crop — what
`r_region` actually depends on, since the reward model sees only the crop), the fraction
of crops that contain the GT box entirely, and the centre-hit rate. The winner is taken on
IoU as asked; if centre-hit or centre-error picks a different frame, the gate prints
`DISAGREEMENT` rather than letting the IoU column read as settled.

**Crop area** — median and mean as a fraction of the original, and the fraction at or
under **0.6**, so ERGO's γ can be checked against what this policy emits rather than
against the TreeVGR/VisCoT/V\*/VGR distribution it was derived from.

Everything is written to `outputs/ergo_gate_zoom/gate.json`, and every turn-1/turn-2
completion with its box and GT box to `completions.jsonl`, so the frame can be re-decided
offline without regenerating.

## 6. The thresholds, and what failing them costs

| | gate | if it fails |
|---|---|---|
| zoom-parse rate (strict), on the **cold start** | ≥ 0.60 | **a two-turn ERGO cold start is needed** |
| turn-2 format (strict) | ≥ 0.60 | more cold-start pressure; cheaper, our corpus already suppresses `<answer>` tags |

A two-turn cold start means ~2–5k SFT rows in ERGO's own format, with turn-1 boxes taken
from the GT annotations already in `cold_data/grpo_sets` (`_boxes_abs` per row,
`union_bbox` at `build_grpo_sets.py:133`). **That is +3–5 days and it is the single thing
that blows this arm's budget.** It is to be said out loud before it is started, not
started and reported afterwards.

---

## 7. Result

**Not yet measured — the gate is built, tested and blocked on hardware.** See section 8.

What exists and is verified:

- `ergo_zoom_gate.py` — the instrument. 50 CPU tests in `test_ergo_zoom_gate_cpu.py`, all
  passing, covering ERGO's parser verbatim, the two turns' format checks, the three frame
  mappings, the IoU/coverage confound, the "a wrong frame is a valid rectangle" case, the
  undecided-with-no-boxes case, the 28-vs-32 budget, and the prompt diff against the
  clone.
- `launch_ergo_zoom_gate_job.sh` — one GPU, ~2 h, `sbatch` or `submit_job`.
- `craigwu/vstar_bench` fetched into `HF_HOME`: 191 rows, 191 ground-truth boxes, 0
  missing annotations.
- The geometry, verified on the real models at the real budget: the pre-resize and the
  processor agree exactly, `processor_resized_under_us = 0`. See the table below.

## 8. The blocker: this cluster has no GPUs

As of 2026-11-07, `oci-nrt-cs-001` has **no GPU hardware at all**:

```
$ sinfo -h -N -o "%G" | grep -ci gpu
0
$ scontrol show partition batch_short
   ...  Nodes=(null)  TotalNodes=0
```

Every GPU partition — `batch_block1`, `batch_singlenode`, `batch_short`, `batch_long`,
`batch_large`, `backfill` — reports `TotalNodes=0`, and 32 jobs cluster-wide have been
pending at `Reason=PartitionConfig` since 2026-10-30. `submit_job` does not resolve from
`oci-nrt-cs-001-vscode-01` at all; plain `sbatch` rejects a 1-GPU request with **"More
processors requested than permitted"** at *any* `--cpus-per-task`, including none, which
reads like a QoS cap and is not one.

The launcher now runs that one-second check before submitting and refuses with the move
instead of parking a job forever. **To get section 7's numbers:**

```fish
ssh cs-oci-ord-login-01
cd /home/uberger/scratch/research/saliency_r1/.worktrees/probe-ergo-gate-zoom
bash launch_ergo_zoom_gate_job.sh
```

Nothing needs copying — the second cluster checks this repo out at the same absolute
paths, and `checkpoint/` and `HF_HOME` resolve on both. ~2 h on one card for both models.

The same wall stands in front of Gate A (P1), which wants 8 GPUs for a benchmark sweep.

### What was validated without a GPU

The whole path — both `generate()` calls, both passes, the frame table, the verdict, the
JSON and the JSONL — was run on `cpu_short` against the **real** cold start and the
**real** Instruct model (`--gpus 0` routes there and loads in float32, because torch's CPU
bf16 is emulated on those nodes). Two jobs:

| | budget | what it proved |
|---|---|---|
| `outputs/ergo_gate_zoom/cpu_tiny` | `--vision-tokens 64 --n 2 --max-new-tokens 24` | the script completes end to end on both models, writes both files, and takes the no-boxes branch correctly: `winner_note` replaces the verdict instead of a frame being named from nothing |
| `outputs/ergo_gate_zoom/cpu_smoke` | **the production geometry**, `--vision-tokens 1280 --n 2 --max-new-tokens 48` | **`processor_resized_under_us: 0`** — the pre-resize and the processor agree exactly at the budget the real run uses. 2000×1500 → 1312×992 = 1271 tokens; 2250×1500 → 1408×928 = 1276 |

The `cpu_tiny` job is also where the processor-floor warning came from: 64 tokens is 65536
px, which is exactly `size.shortest_edge`, so `smart_resize`'s rounding landed just under
it and the processor upscaled 1 of the 2 images back. The assertion caught it, and the
budget is now checked against both of the processor's bounds at load time.

**Nothing about format or boxes can be read off these two jobs.** 24 and 48 new tokens
truncate every completion mid-sentence, `n` is 2, and the sampling is at temperature 1.0.
The `FAIL` they print is the truncation, not the checkpoint.

## Sources

* [arXiv:2509.21991](https://arxiv.org/abs/2509.21991) — ERGO (ICLR 2026)
* [nota-github/ERGO](https://github.com/nota-github/ERGO) — 627 lines, inference and eval
  only, Apache 2.0. Cloned to `/home/uberger/scratch/research/ERGO`
* [ergo-build-prompts.md](ergo-build-prompts.md) § Prompt 2 — the brief this executes
* [baseline-port-estimates.md](baseline-port-estimates.md) § *ERGO on Qwen3-VL* — the
  estimate the gate can cancel
