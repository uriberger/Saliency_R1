# Gate A — does Qwen3-VL even have a low-resolution penalty?

Companion to [baseline-port-estimates.md](baseline-port-estimates.md) § *ERGO on
Qwen3-VL* and to [ergo-build-prompts.md](ergo-build-prompts.md) § *Prompt 1*. This
is the blocking gate: nothing in P5–P11 starts until it returns a number.

**Status (2026-11-07): the accuracy grid has NOT run. There are no GPUs.** Every
GPU partition on `oci-nrt-cs-001` reports `TotalNodes=0` and no node anywhere
advertises a `gpu` GRES, so the 8-GPU job cannot be scheduled from here — see
[*How to finish it*](#how-to-finish-it). What the gate *can* answer without a GPU,
it does, and that turns out to be a good deal more than nothing.

---

## The question

ERGO's headroom is the accuracy that downsampling destroys and a crop gives back.
Their Table 1, Qwen2.5-VL-7B on V*:

| visual tokens | 16384 | 1280 | 640 | 640 + GT crop |
|---|---|---|---|---|
| V* accuracy | 77.0 | 64.9 | 56.5 | 77.0 |

If Qwen3-VL-8B does not lose those points there is nothing for the method to
recover, and no amount of making it work produces a number it cannot reach.

There was specific reason to doubt it does. DeepStack fuses multi-level ViT
features precisely so detail survives a smaller token budget, and per the Qwen3-VL
tech report the backbone already saw tool-integrated RL with an `image_zoom_in_tool`
in its own post-training. Either would show up as a flat resolution curve.

**The decision rule, agreed in advance:**

- gap ≥ 10 points between 16384 and 640 — proceed.
- gap 3–10 — proceed only if the claim is the efficiency axis, not accuracy.
- gap < 3 — stop. *"ERGO's premise does not hold on this backbone"* is a
  publishable sentence that cost one day.

---

## The factor is 32, not 28

Qwen3-VL is patch-16 with a 2×2 spatial merge, so **one visual token covers 32×32
pixels**, not Qwen2.5-VL's 28×28. A budget copied literally from ERGO as `N×28×28`
buys 23% fewer tokens than the label says, in the direction that makes the penalty
look larger. Every cap here is `N×32×32`.

This is not an inference from the config. `qwen_vl_utils.fetch_image` computes
`patch_factor = image_patch_size * SPATIAL_MERGE_SIZE`, and both lmms-eval wrappers
pass `image_patch_size=16`, so 32 is the factor the resize actually uses.

### What the caps really buy

`ergo_gate_lowres.py --token-census`, measured through `smart_resize` itself:

| benchmark | cap 640 | cap 1280 | cap 16384 | native size |
|---|---|---|---|---|
| V* | 600 | 1247 | **3290** | ~2000×1500 (3 MP) |
| HR-Bench 4K | 625 | 1225 | 15876 | 4032×4032 |
| HR-Bench 8K | 620 | 1232 | 16274 | 7680×4992 |

Two things to carry into the table when it exists:

- **The 16384 arm on V\* is 3.3k tokens, not 16k.** An image is never *enlarged* to
  meet a cap, and V*'s photos are 3 MP. The top arm is "native resolution", and
  calling it a 16384-token arm would overstate the top of the curve. ERGO's own
  16384 row has exactly the same property, which is part of why it reads 77.0 and
  not 95.
- The caps land a little under their label because `floor_by_factor` rounds down to
  whole 32-px steps. 600 of a requested 640 is the budget working, not failing.

---

## `max_pixels` does nothing in the chat wrapper

This is the finding that had to come first, and it is not confined to this probe.

`--model qwen3_vl` resolves to the **chat** wrapper,
`lmms_eval/models/chat/qwen3_vl.py`. That wrapper:

1. calls `ChatMessages.to_hf_messages(video_kwargs=video_kwargs)` — with **no
   `image_kwargs`** — so no per-image budget ever reaches `qwen_vl_utils`, and
   `fetch_image` falls back to its own defaults of 4…16384 tokens; and
2. passes `do_resize=False` to the processor, so the processor's own `max_pixels`
   (set in `AutoProcessor.from_pretrained`) never applies either.

Measured on a 2000×1500 image with `max_pixels` set to 640 tokens:

| path | visual tokens produced |
|---|---|
| chat (`qwen3_vl`) | **2914** — native, capped only by qwen_vl_utils' 16384 |
| simple (`qwen3_vl --force_simple`) | **609** — the cap applied |
| processor alone, `do_resize=True` | 638 |

So the chat wrapper has no resolution knob at all, and `--force_simple` is the only
way to have a resolution axis. Every unit in this probe passes it; a CPU test
asserts that it is never dropped, because without it every arm is the same arm and
the gap reads exactly 0.0 — which is also the "stop the port" answer.

**The consequence outside this probe.** `--max-pixels` has been passed through
`launch_lmms_eval_job.sh` in chat mode for a long time, including the 3211264
default that script applies to `mmerealworld`, `hrbench4k`, `hrbench8k` and
`vstar_bench`. In chat mode **it did nothing**. Those runs were not at 3136 tokens;
they were at native resolution capped at 16384. That is not necessarily wrong — it
is more pixels, not fewer — but the `_px3211264` suffix on those result directories
means something it does not mean, and any conclusion of the form "we checked that
resolution was not the problem" needs re-reading. The note in
[the baseline estimates](baseline-port-estimates.md) that `_chat_base.py:40`'s
`IMAGE_KWARGS = {"max_pixels": 512*512}` caps us at 256 tokens is also wrong for
this wrapper: that constant belongs to `ChatMixin`, which the omni wrappers use and
`qwen3_vl_chat` does not.

---

## What the gate can say without a GPU

The accuracy table answers *whether* the penalty exists. Arithmetic answers whether
it *can*, and that half needs no hardware: a visual token is 32×32 pixels of the
**resized** image, so scaling V*'s ground-truth box by the same factor the image was
scaled by says how much of the answer survives each cap.

All 191 V* documents, all three caps (`--token-census` writes `target_census.json`):

| arm | median target size | p10 | p90 | share under **one** token |
|---|---|---|---|---|
| 640 tokens | **0.65 tokens** | 0.08 | 14.4 | **58.6%** |
| 1280 tokens | 1.35 tokens | 0.18 | 29.6 | 42.9% |
| 16384 tokens (= native, ~3.3k) | 3.62 tokens | 0.47 | 80.8 | 24.6% |
| 640 + GT crop | **272 tokens** | 264 | 286 | 0.0% |

Read it as three statements:

1. **The axis is real on this benchmark.** The answer-bearing region shrinks 5.6×
   in token area between the top arm and the bottom one, and at 640 tokens the
   median V* target does not fill a single visual token. Whatever Qwen3-VL's
   accuracy does, it is not being asked an easier question at 640.
2. **The top arm is not generous either.** Even at native resolution the median
   target is 3.6 tokens and a quarter of the items are still sub-token. This is why
   V* is hard for everyone, and it caps how high the 16384 row can go.
3. **The oracle arm has enormous room.** The crop puts 272 tokens on the region —
   400× the 640-token arm's 0.65. If accuracy does *not* move between those two,
   the explanation is not resolution, and that would be a much more interesting
   result than a confirmation.

A caveat that belongs with the number: `min_pixels` is 256 tokens (the wrapper's own
default, passed explicitly), so the crop is *upsampled* to ~272 tokens from a region
that is often only a few thousand pixels. Those tokens carry no information the
crop did not have — but ERGO's `crop_with_zoom` re-encodes its crop through
`smart_resize` the same way, so the oracle arm is not quietly stronger or weaker
than the method it stands in for.

---

## Design notes that will matter when it runs

**V\*'s boxes are not in either HF release.** `lmms-lab-encoder/vstar-bench` and
`craigwu/vstar_bench` both expose only `(image, text, category, question_id, label)`.
The boxes live in the per-image annotation JSONs of the craigwu repo
(`direct_attributes/sa_4690.json` and 237 others) as absolute `[x, y, w, h]` in
original-image pixels. They are fetched once from a host with network and committed
as `assets/vstar_bench_boxes.json`: all 191 documents covered, 137 with one box and
54 (the `relative_position` half, which asks about two objects) with two. Validated
against the images the task actually loads — the encoder release preserves the
original resolution, so the coordinates apply unchanged and no box falls outside its
image.

**HR-Bench gets no crop arm.** It ships no boxes at all — its columns are
`(index, question, answer, A–D, category, cycle_category, image)` and its paper
annotates none — so there is nothing to crop to. The prompt's assumption that "V*
and HR-Bench both ship boxes" holds for V* only.

**HR-Bench's scorer had to be replaced.** Upstream `hrbench_process_results` hands
anything it cannot read locally to a GPT judge. On a compute node that is fatal
twice: there is no route to the API, so each unreadable item burns ten 30-second
POSTs, and when they all fail `get_chat_response` returns *without* writing
`gpt_prediction`, so the caller raises `KeyError` and takes the whole task's
aggregation with it. The generated `hrbench_gate.py` keeps upstream's local matcher
(`can_infer` — the thing that already decides every item the judge never sees, so
the rules do not change), replaces the judge with a letter regex over the
post-`</think>` segment, and reports **`parse_fail`** as its own metric. That column
is the point: an arm that looks worse because its chains stopped ending in a letter
is a grading accident, not a resolution effect, and we have shipped two of those
already (LogicVista, MathVision).

**No `--use_cache`, ever.** lmms-eval's response cache keys on
`(context, gen_kwargs, {task, split, doc_id})`. The pixel budget is in none of them,
so a cache shared across caps would serve the 640-token answers back for the 16384
arm and the gap would read exactly 0.0. A CPU test asserts the flag is absent.

**`LMMS_LOCAL_CACHE=/nonexistent`.** lmms-eval reroutes the `datasets` cache to
node-local scratch whenever `HF_HOME` is on a remote filesystem, which ours is. The
rerouted directory is empty, and with `HF_HUB_OFFLINE=1` every task then dies on a
`ConnectionError` before the model is built. `/nonexistent` is the documented way to
say "write straight to `HF_HOME`"; `launch_lmms_eval_job.sh` exports the same value
for the same reason.

**The two models are not compared with each other.** Vanilla
Qwen3-VL-8B-Instruct runs under the stock assistant prompt with a 512-token cap; the
cold start runs under the r1 system prompt with 4096, which is how every other
benchmark row we report is set up. The gate is read **down each model's own column**
— the quantity of interest is the 16384-minus-640 gap, not the absolute level. See
[[plain-instruct-rows-are-not-comparable]].

**HR-Bench is subsampled to 300 per split**, seeded and prefix-stable
(`select(range(300))` is a prefix of `select(range(800))`, so raising it later adds
documents and moves none). A plain `--limit` would not do: HR-Bench stores its two
categories in runs, and the first 400 rows of each split are 320 `single` to 80
`cross`. V* runs whole at 191.

---

## How to finish it

The hardware is the only thing missing. `oci-nrt-cs-001` lost every GPU node around
2026-10-27; each GPU partition still reports `State=UP`, so it will *accept* an
8-GPU job and park it at `Reason=PartitionConfig` — which reads exactly like
ordinary queueing and is normally something to poll rather than resubmit. It is not
queueing. The one-second test:

```fish
sinfo -h -N -o "%G" | grep -ci gpu      # 0 means there is no GPU hardware to wait for
```

`launch_ergo_gate_lowres_job.sh` now runs that test itself and **refuses to submit**
rather than queueing into a partition with no nodes.

Run it from `cs-oci-ord-login-01` ([[second-a100-cluster]]), which checks the repo
out at the same absolute paths:

```fish
cd /home/uberger/scratch/research/saliency_r1
git fetch origin; and git checkout probe/ergo-gate-lowres
bash launch_ergo_gate_lowres_job.sh --name ergo-gate-lowres \
    --out-dir outputs/ergo_gate_lowres --gpus 8 --duration 4
```

Then, once it is done:

```fish
python ergo_gate_lowres.py --report --out-dir outputs/ergo_gate_lowres \
    --markdown outputs/ergo_gate_lowres/report.md
```

and fold that table into the [Result](#result) section below. Point `--markdown` at
the output directory, not at this file — it writes a bare table and would discard
everything above.

**14 units, 8 GPUs, ~4 h.** One `accelerate launch` each, data-parallel over
documents. It **resumes**: a unit with a harvested json is skipped, so a job that
dies at hour 3 is resubmitted rather than restarted, and it exits 17 while units are
owed and 0 when the grid is complete. `--job-minutes` stops it *starting* a unit it
cannot finish. Two consecutive failures abort the run, because every unit is the
same command with two substitutions and two in a row is a config fault, not a flake.

Order is deliberate: **every V\* arm runs before any HR-Bench arm.** V* is 191
documents against HR-Bench's 2×300 and it is the benchmark ERGO's Table 1 is quoted
from, so a short allocation still returns the numbers the gate is decided on.

Everything else is already verified on CPU: `test_ergo_gate_lowres_cpu.py` is 24
tests, all passing, and the task configs load through lmms-eval's own YAML
constructor with all four tasks registering and building documents (V* + GT crop
correctly emits two images, 2000×1500 and 155×157 on item 0).

---

## Result

**Pending.** No accuracy number yet — see above.

What is already settled, and does not depend on the run:

- The caps are `N×32×32`, and the 16384 arm on V* is really ~3.3k tokens.
- `max_pixels` is inert in the chat wrapper; `--force_simple` is mandatory, and
  every `_px…` result directory we already have is mislabelled.
- At 640 tokens the median V* target is 0.65 visual tokens and 58.6% of items are
  sub-token; the GT crop puts 272 tokens on the same region. The axis is physically
  real on this benchmark whatever the accuracy turns out to do.
