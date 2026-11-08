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
mechanisms a reward depends on, not a quality metric. One GPU — and **~6 h** for both
models at the full 191 rows, measured, against the ~1–2 h originally estimated here:
191 rows × 2 turns × 2 models is 764 generations and an A100-80GB does 2.2/min at this
geometry. Nothing is written until the second model finishes, so the allocation has to
cover the whole thing.

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

### VERDICT: FAIL — the cold start parses a `<zoom>` box on 4.2% of rollouts

> **Under ERGO's own parser, a two-turn cold start is needed before any GRPO run is
> worth starting. That is +3–5 days and it is the single thing that blows this arm's
> budget.** It has **not** been started.
>
> **But the cold start produces the box and withholds only the tag.** Scored for a
> bracketed box after `</think>` instead of inside `<zoom>` tags, the same rollouts
> pass at **83.3%**. A parser accommodation — a deliberate, documented deviation from
> ERGO — unblocks turn 1 for free. Turn 2's 0.0% `<answer>` rate does not go away
> either way. See *The number that changes the decision*.

Measured 2026-11-08 on `cs-oci-ord`, one A100-80GB inside interactive allocation
`38208034`, temperature 1.0, 1280 visual tokens, `coord hint: 'none'`, and
`processor_resized_under_us = 0` on both models — so the frame table below is reading
the pixels the model actually saw.

**Read the n first: this is n = 24, not 191.** See *Why n = 24* below. It is a
complete run of the gate — both turns, both models, the frame table, `gate_sub24.json`
and `completions_sub24.jsonl` — over a stratified quarter of the benchmark.

| turn 1 | cold start | base Instruct | gate |
|---|---|---|---|
| `<zoom>` tag present | 8.3% | **100.0%** | |
| **box parses, ERGO's own parser** | **4.2%** | **95.8%** | ≥ 60% |
| box parses leniently | 79.2% | 95.8% | |
| `<think>…</think>` exactly once | **66.7%** | **0.0%** | |
| **joint turn-1 format** | **0.0%** | **0.0%** | |

| turn 2 | cold start | base Instruct | gate |
|---|---|---|---|
| strict `<think>…</think><answer>…</answer>` | 0.0% | 0.0% | ≥ 60% |
| lenient (one `<answer>` pair) | 0.0% | 0.0% | |
| `<answer>` tags at all | 0.0% | **100.0%** | |
| **both turns well-formed = `r_format`** | **0.0%** | **0.0%** | |
| letter accuracy (context only) | 58.3% | 79.2% | |

**The two models have exactly one half of the format each, and neither has both.**
This is the thing `--reference` was made non-optional for, and it pays for itself
here. The cold start writes `<think>` (66.7%) and almost never a box (4.2%). Vanilla
Instruct writes a parseable box on **95.8%** of rollouts and never once writes
`<think>` (0.0%). So:

- **The box ability is in the backbone, and our cold start removed it.** That matches
  the Qwen3-VL tech report's `image_zoom_in_tool` RL, and it is a completely different
  problem from "this recipe cannot produce boxes" — which 4.2% alone could not have
  distinguished. The two-turn cold start has to *add* `<zoom>` to a corpus that
  already teaches `<think>` without destroying what the backbone came with.
- **`r_format` is 0.0% on both models**, so on today's checkpoints every reward in a
  GRPO group is 0, the group advantage is exactly 0, and the run trains nothing while
  the loss curve looks ordinary. This is the failure the gate exists to find, and it
  is present.
- The 4.2%-strict vs 79.2%-lenient gap is the diagnosis, not noise: the cold start
  *does* emit four numbers on most rows, just not inside `<zoom>` tags. The fix is
  format supervision, not a capability transplant.

### The 512-token cap was hiding half the signal — but not the failure

The run above gave **both** models `--max-new-tokens 512`. That is right for vanilla
Instruct (median turn-1 88 tokens, never truncates) and wrong for the cold start,
which `ergo_gate_lowres` deliberately gives **4096** because the r1 system prompt
produces long chains. 5/24 of the cold start's turn-1 completions came back exactly
at the cap and 8/24 had no `</think>` at all. Re-measured on the same 24 rows with
only the cap changed:

| cold start, turn 1 | cap 512 | cap 2048 |
|---|---|---|
| `<think>…</think>` exactly once | 66.7% | **87.5%** |
| emits a box *somewhere* | 79.2% | **91.7%** |
| `<zoom>` tag present | 8.3% | 16.7% |
| **box parses, ERGO's strict parser** | **4.2%** | **4.2%** |
| turn-2 `<answer>` tags at all | 0.0% | **0.0%** |

**The cap was suppressing the think block and the box. It was not causing the
failure.** Strict parse is identical at 4.2%, and turn-2 `<answer>` is exactly 0.0%
at both caps — completely insensitive to the generation budget. So the two headline
numbers in this document stand, and the diagnosis is now sharper rather than
different: the cold start **produces the box and withholds the tag.**

### The number that changes the decision

ERGO's parser requires `<zoom>[x1,y1,x2,y2]</zoom>`. Score instead for a bracketed
four-number list **after the closing `</think>`** — the shape the cold start actually
emits, e.g. `… The answer should be latex. </think> [329, 84, 539, 225]`:

| parser | cap 512 | cap 2048 | gate |
|---|---|---|---|
| ERGO strict, `<zoom>` tags | 4.2% | 4.2% | 60% |
| **bracketed box after `</think>`** | 50.0% | **83.3%** | 60% |
| four numbers anywhere | 79.2% | 91.7% | |

**At the correct generation cap a tail parser clears the gate** — 20/24, whose 95%
Wilson interval is roughly 64–93%, with the lower bound still above 60. That is the
difference between a port decision and a +3–5 day cold start, so it is worth stating
precisely what it does and does not buy.

What it buys: a parseable, croppable box on most rollouts, so `r_format` is not
structurally 0 and GRPO has a gradient to work with.

What it does not buy: **box quality**. Under `rel1000` the cold start's boxes score
mean IoU **0.056** and centre-hit **0.50**, against the base model's **0.206** and
**0.652**. (Note the IoU/centre divergence between the two caps — 0.18→0.06 IoU while
centre-hit rises 0.40→0.50 — which is the crop-size confound §5 warns about, not a
regression: the longer-cap boxes are larger, so they cover the target more often and
overlap it less.) And it does nothing at all for turn 2, which is 0.0% at both caps.

**This is a deliberate deviation from ERGO and must be taken as one.** Their
`r_format` is what teaches the policy the tag; loosening the parser means the tag is
never learned. That is defensible — the reward exists so the trainer can crop, not to
enforce angle brackets — but it is a change to the method, it belongs in the writeup,
and it should not be made silently inside a regex.

### An operational constraint found the hard way

**Gate B OOMs on an 80 GB card at `--max-new-tokens 4096 --batch-size 4`**, in the
**turn-2** loop (`ergo_zoom_gate.py:606`), after turn 1 has completed — and since the
gate writes only at the end, that is the whole run lost. Turn 2 is the expensive
turn: its prompt carries *both* images (~2560 visual tokens against turn 1's ~1280)
plus the entire turn-1 completion, so the sequence reaches ~7800,
`sdpa_attention_forward` falls back to the math kernel, and it materialises a
`batch × heads × seq × seq` tensor — 15.65 GiB in one allocation.

`--max-new-tokens 2048 --batch-size 2` with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` peaks at ~25 GB and completes.
**The full-191 rerun must use those settings**, and the ~6 h estimate in §4 assumes
batch 4 — at batch 2 budget proportionally more.

### The coordinate frame: `rel1000`, decided on the base model

The cold start parsed **1** box, so its frame table is undecided and printed
`DISAGREEMENT`, as designed. The base model parsed **23**, and they are unambiguous:

| frame | n | mean IoU | GT coverage | centre-hit | centre-err |
|---|---|---|---|---|---|
| **rel1000** | 23 | **0.206** | **0.617** | **0.652** | **0.110** |
| resized | 23 | 0.0102 | 0.194 | 0.174 | 0.161 |
| original | 23 | 0.0000 | 0.000 | 0.000 | 0.258 |

**`rel1000` wins by +0.196 mean IoU and all three statistics agree** — no
`DISAGREEMENT` warning, which is the condition section 5 set for taking the frame
seriously. The structural evidence agrees too: 0 of 23 boxes exceed 1000, median max
coordinate 775, largest 999 — a distribution pressed right up against a 0–1000 grid.

**So ERGO's `map_box` must not be ported unchanged.** Qwen3-VL emits the 0–1000
relative grid its documentation describes; Qwen2.5-VL's post-`smart_resize` pixel
reading scores 20× worse on IoU and misses the target centre 83% of the time. This
was the first row of the build document's trap table and it is now measured rather
than assumed.

**ERGO's γ = 0.6 is non-binding on this backbone.** Median crop area is **1.14%** of
the image on the base model and **0.02%** on the cold start; **100%** of crops are
under γ on both. `r_box` would fire on every rollout and carry no signal at all. It
is not harmful, but it is not doing the work the paper's ablation gives it credit
for, and the TCE reward is effectively `r_region` alone here.

### Why n = 24

The plan budgeted ~2 h for 191 rows × 2 turns × 2 models. Measured throughput on an
A100-80GB at the production geometry was **2.2 generations/min**, which puts the full
764-generation gate at **~5.8 h** — and `ergo_zoom_gate.py` writes nothing until both
models finish, so a full run inside a 75-minute window returns *zero*, not a partial.
The full run was started, got to `turn1 ...40/191 parsed 3`, and was stopped.

The subset is **14 `direct_attributes` + 10 `relative_position`** — proportional to
V\*'s real 115/76 split — and evenly strided inside each category.
`load_vstar`'s own `--n` is `rows[:n]`, which on a category-ordered `test_questions.jsonl`
would have produced an all-`direct_attributes` sample; the stratified file lives at
`outputs/ergo_gate_zoom/vstar_sub24/` and the runner is
`outputs/logs/ergo_gate_zoom.sub24.sh`.

**What n = 24 does and does not support.** The decision this gate drives is one
threshold, 60%, and 4.2% is nowhere near it: a 1/24 rate has a 95% Wilson interval of
roughly 0.7–21%, entirely below the gate. The aborted full run independently saw
3/40 = 7.5% strict on a different, larger slice. The two agree, and the verdict does
not depend on which is right. By the same token the base model's 95.8% and the
`rel1000` margin are not close calls either.

What n = 24 **cannot** carry: the frame verdict rests on 23 boxes from one model, the
cold start's frame is simply undecided, and the crop-area medians are indicative.
Rerun the full 191 — it is one GPU for ~6 h, and it now has a measured cost rather
than an estimate — before any of those get quoted as settled.

### Also verified

- `ergo_zoom_gate.py` — 50 CPU tests in `test_ergo_zoom_gate_cpu.py`, all passing.
- `craigwu/vstar_bench` fetched into `cs-oci-ord`'s `HF_HOME`: 191 rows, 191
  ground-truth boxes, 0 missing annotations. It was **absent** on that cluster and
  had to be staged from the login node; the lmms-eval mirror that *was* present
  carries no boxes and would not have worked.
- The geometry on the real models at the real budget: `processor_resized_under_us = 0`.

## 8. Where it ran, and the cluster that could not run it

Section 7 was produced on **`cs-oci-ord`**, from an interactive allocation rather
than `submit_job`. `--dry-run` prints the runner the batch path would have submitted;
`outputs/logs/ergo_gate_zoom.sub24.sh` is that runner pinned to one card with
`CUDA_VISIBLE_DEVICES=0` and pointed at the stratified subset:

```fish
srun --jobid=$JOBID --overlap --ntasks=1 bash outputs/logs/ergo_gate_zoom.sub24.sh
```

**To redo it at the full 191 rows, budget ~6 h on one GPU**, not the ~2 h this
document estimated before the run, and drop `--vstar-root` to get the whole
benchmark. Gate B does **not** resume and writes nothing until both models finish, so
a short allocation returns nothing at all — give it the whole window or cut `n`.

### The cluster that could not run it

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
instead of parking a job forever. The same wall stood in front of Gate A (P1); both gates
were moved to `cs-oci-ord` and both ran there on 2026-11-08.

One correction to what this section used to promise. *"Nothing needs copying — the second
cluster checks this repo out at the same absolute paths"* is true of the **repo** and of
`checkpoint/`, and false of `HF_HOME`. The two clusters' `$HOME/scratch` are different
Lustre mounts behind identical paths, and `craigwu/vstar_bench` was **not** staged on
`cs-oci-ord` — only the lmms-eval mirror, which ships no boxes. Under `HF_HUB_OFFLINE=1`
that is a `SystemExit` on the first row, inside the allocation. Stage it from the login
node before allocating:

```fish
set -x HF_HOME /home/uberger/scratch/cache/hf_cache
python -c "from huggingface_hub import snapshot_download as s; print(s('craigwu/vstar_bench', repo_type='dataset'))"
```

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
