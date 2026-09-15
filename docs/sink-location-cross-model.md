# Does the ring survive a different model? Porting the sink-location experiment to InternVL-3.5 and LLaVA

**Handoff for a new session.** Written 2026-09-14.

## The claim under test

On Qwen3-VL-8B we find that attention inside the picture concentrates on the **one-patch
outer ring**, at 2.6–3.1x the interior's per-patch density, carrying about half of the
image's attention from 30% of the patches, with 76–85% of map peaks on it — and with a
specific peak on the **upper-left patch**. It holds in all 12 image types, in the base
model as well as both RL-trained ones, so it is neither a background effect nor something
our training produced.

The permutation arm already told us where it comes from: when the visual tokens are
**shuffled after the vision encoder emits them**, the attention follows the upper-left
patch to its new sequence position. Whatever attracts attention is written into the token
by the encoder; it is not the LLM's positional slot.

**The open question is whether any of this is a property of Qwen3-VL or of VLMs.** The
literature gives real reason to doubt it generalises — MCA-LLaVA and VisPruner report the
*opposite* positional bias for LLaVA-family models (attention to *later* raster tokens,
i.e. the bottom of the image, via RoPE decay), Darcet et al. find high-norm artifacts in
DINOv2/OpenCLIP/DeiT but not DINOv1, and Bi et al. (CVPR 2025) find visual heads correlate
within a model family and are "distinctly different" across families. See
`docs/where-attention-goes.md` for the full survey. A null result here is a publishable
scoping statement, not a failure.

---

## Read these first, in this order

| file | why |
|---|---|
| `docs/sink-location-by-image-type.md` | the design of the existing experiment |
| `sink_location.py` (957 lines) | the measurement: geometry, reducer, attention impl, image transforms, the two model-level interventions |
| `sink_location_probe.py` (2101 lines) | the harness: `--stage corpus\|selftest\|scan\|arms\|report\|monitor` |
| `docs/peak-location-results.md` | the Qwen3-VL numbers you are trying to reproduce elsewhere |
| `docs/where-attention-goes.md` | what the literature claims, and what it does not |

## Do not rebuild any of this

Everything the experiment needs already exists and is gated by a selftest. In particular
**both things this port must include are already implemented for Qwen3-VL**:

- **Uniform grey / noise / blank images** — `sink_location.py` `_fill()` handles
  `grey`/`white`/`black`/`noise`, and `ARMS` already contains `pad_grey_1`,
  `pad_white_1`, `pad_noise_1`, `pad_grey_2` (A3), `canvas0/4/8` (A4, blank everywhere
  with the object moved) and `donut` (A5, blank centre, content only on the ring).
- **Post-encoder token permutation** — `PatchPermute` (`sink_location.py:694`), with
  `mode="shuffle" | "roll" | "identity"`, reached through the `permute` and
  `permute_identity` entries in `SPECIAL_ARMS`.

Your job is to make them run on two more models, not to write them again.

---

## The model-specific surface to abstract

These are the only places `sink_location.py` knows it is talking to Qwen3-VL. Turn them
into a small adapter (one class per family, selected off `config.model_type`) rather than
branching inline:

| what | where | Qwen3-VL | InternVL-3.5 | LLaVA-1.5 |
|---|---|---|---|---|
| decoder attention module | `sink_location.py:598,611` | `Qwen3VLTextAttention` | text tower is **Qwen3** — confirm the class name on the loaded object, it differs between the remote-code and native paths | `LlamaAttention` |
| vision module, for the permute hook | `:717` | `Qwen3VLVisionModel` / `…VisionTransformerPretrainedModel` | `InternVisionModel` | `CLIPVisionModel` |
| locating the image token run | `locate_image_runs`, `:358` | image token id + `image_grid_thw` | `IMG_CONTEXT` token id; **no `image_grid_thw`** — the grid must be derived from the tiling | fixed 576-token block, single grid |
| patch grid (gh, gw) | same | from `image_grid_thw` / `spatial_merge_size=2` | 448/14 = 32x32, pixel-shuffled by `downsample_ratio 0.5` → **16x16 = 256 tokens per tile** | 336/14 = **24x24 = 576** |
| extra features to permute | `:740` | `deepstack_features` at ViT layers 8/16/24 must be permuted **too** | none | none |

`deepstack_features` is the one piece with no analogue elsewhere: Qwen3-VL injects vision
features into the LLM at three depths, and the permute arm is only honest if all three are
permuted with the same index. The other two models have a single projector, which makes
their permute arm simpler — and makes any difference in the result interpretable.

### The trap that will ruin this run: InternVL tiling

InternVL-3.5 has `max_dynamic_patch: 12`. An image becomes up to 12 tiles of 448px **plus a
thumbnail**, each tile its own 16x16 grid. "The outer ring" is then ambiguous — the ring of
a tile, or the ring of the picture? A tile boundary is an interior edge of the image but a
border of the tile, and conflating them will produce a confident, meaningless number.

**Run the primary comparison with tiling forced off** (`max_dynamic_patch=1`, or
`max_num=1` in their preprocessing), so there is exactly one 16x16 grid and the geometry
matches Qwen3-VL's. Then run the tiled configuration as a **separate, clearly labelled
arm** — if the ring appears on every tile's own border, that is a *stronger* result than
ours and worth its own section, because it would show the effect tracks encoder input
boundaries rather than the picture.

### LLaVA: pick 1.5, not NeXT

LLaVA-1.5-7B is the clean case: fixed 336px, one 24x24 grid, frozen CLIP encoder, no
tiling. LLaVA-NeXT/1.6 uses anyres tiling and has the same ambiguity as InternVL. Use
`llava-hf/llava-1.5-7b-hf`, which loads as `LlavaForConditionalGeneration` in the
installed transformers (5.13.0.dev0) with no remote code.

**LLaVA-1.5 is also the best falsification target.** It has a *frozen* CLIP encoder at its
native training resolution, so it barely interpolates position embeddings — and it is the
family for which the literature reports a bottom-of-image bias. If the ring is going to
fail anywhere, it is here, and that is exactly why it must be in the run.

### InternVL-3.5-8B is the best-controlled comparison

Its text tower is **Qwen3, 36 layers, 32 heads** — the same shape as Qwen3-VL-8B's. So
Qwen3-VL vs InternVL-3.5-8B holds the language model nearly fixed and swaps the vision
encoder and connector. If the ring survives that swap it is not about the LLM; if it dies,
the encoder is implicated directly. Say so in the report — it is a better argument than
either model alone.

---

## What must be preserved from the Qwen3-VL implementation

Four things in the existing selftest are not optional, and a port that drops them produces
numbers that look fine and are wrong:

1. **The attention implementation must reproduce stock SDPA**, in the logits and token for
   token under greedy decoding. It edits nothing; that has to be measured, not asserted.
2. **The coordinate frame check.** A picture with one bright patch at a known position,
   put through every transform, must land where `patch_correspondence` says
   (`sink_location.py:936`). An off-by-one here answers the content-versus-position
   question confidently and backwards. Each new model needs its own frame check — the
   patch ordering after InternVL's pixel shuffle is where I would expect this to break.
3. **The permutation must be a permutation**: the multiset of patch embeddings is
   unchanged, and `mode="identity"` reproduces the baseline exactly.
4. **The causal column correction.** In a causal LLM the first image token is visible to
   every later query and the last to one, which is a top-left-heavy gradient *of exactly
   the shape under test*. It must be divided out before any upper-left claim is made. This
   matters more here than anywhere else in the experiment.

Report S1 (peak), S2 (mass) and S3 (sink: large **and** query-invariant) separately, as the
Qwen3-VL run does. S3 can come back empty while S1 and S2 are strong, and that distinction
is the paper's honesty.

Also report the **attention budget** first, as `--stage survey` does for sink_shift: if the
whole picture receives ~1% of an attention row, the ring result is a claim about the
distribution *within* the picture and must be worded that way.

---

## One arm worth adding

The existing permute arm shuffles tokens **after** the encoder and shows the attractor
travels with the token. That rules out the *LLM's* positional slot — but not the *encoder's*
own position embeddings, which could have stamped the token on the way through.

Add **A10: permute the image patches at the encoder input** (shuffle 14x14 or 16x16 pixel
blocks before the ViT, then unshuffle the resulting grid for scoring). If the attractor
then appears at whatever token now occupies the top-left *of the ViT grid*, the encoder's
position embeddings are writing it. If it follows the original content instead, it is
content-driven. This is cheap, it sharpens the mechanism claim from "made in the encoder"
to "made by the encoder's position embeddings", and it is the natural companion to the arm
you already have.

---

## Suggested plan

1. **Worktree.** `./worktree.sh new feat/sink-location-cross-model`. Write outputs to a
   per-branch subdirectory — `outputs/` is a shared symlink.
2. **Adapter + selftest on Qwen3-VL first.** Refactor to the adapter, then re-run the
   existing Qwen3-VL selftest and a short scan and confirm the published numbers are
   reproduced **bit for bit**. A refactor that silently changes the baseline invalidates
   the comparison you are about to make.
3. **LLaVA-1.5-7B.** Simplest geometry, so it debugs the adapter. Needs a download — see
   Environment.
4. **InternVL-3.5-8B**, `max_dynamic_patch=1`. Then the tiled arm separately.
5. **Report** per model: budget, S1/S2/S3, the 12 image types, the blank/noise arms, the
   permute arms, and A10 if implemented.

## What each outcome means

| result | reading |
|---|---|
| ring + upper-left peak in all three | a VLM-wide property; the strongest version of the paper, and the one that needs A10 to explain it |
| holds in InternVL, fails in LLaVA-1.5 | ties it to native-resolution / interpolated-PE encoders; scope the claim to those, and cite the LLaVA bottom-bias literature as the contrast |
| fails in both | a Qwen3-VL property. Still publishable — and it makes the reward-design conclusion *more* interesting, not less, because it says the thing our reward was fighting is architecture-specific |
| ring without the upper-left peak, or vice versa | two mechanisms, not one. Split the claim in the paper |

Whatever comes back, the scan is cheap and the answer is needed before the paper claims
anything beyond Qwen3-VL.

---

## Environment

- **InternVL-3.5 is already cached** — `models--OpenGVLab--InternVL3_5-8B-Instruct` under
  `/home/uberger/scratch/cache/hf_cache/hub` (1B/2B/4B/14B/38B are there too). Note the
  cached snapshot is the **remote-code** repo (`architectures: ["InternVLChatModel"]`, with
  an `auto_map`), so it needs `trust_remote_code=True` and its module class names come from
  the repo, not from transformers. The installed transformers also has a native
  `InternVLForConditionalGeneration`; if you prefer that path you need the `-HF` repo, which
  is **not** cached. Decide once and record which you used — the attention class names
  differ between them.
- **LLaVA-1.5 is not cached.** `HF_HUB_OFFLINE=1` is the default in every launcher here, so
  fetch it once with the variable unset, then restore it. The network reaches PyPI and the
  Hub from the login node.
- Env `saliency_r1_qwen3_vllm`, transformers `5.13.0.dev0`, which has
  `LlavaForConditionalGeneration` natively.
- Submit with `launch_fig1_step_referent_job.sh` as the template for a small job (1 GPU,
  `--duration 1` to reach `batch_short`) or the existing sink-location launcher for the
  full scan. `Reason=QOSGrpGRES` while queueing is churn — poll, do not resubmit.
- Do not run DINO or any heavy CPU work on the login node; it is heavily contended and a
  188-pair detector pass did not finish in 45 minutes there, against ~55 s on one GPU.

---

# The port, as built — 2026-09-14

Everything above is the plan. This section is what exists, and the decisions that were
made where the plan left a choice open. The results are in the section after it.

## The seam: `vlm_family.py`

One class per family, selected off `config.model_type`, holding the six things
`sink_location.py` used to know about Qwen3-VL:

| | Qwen3-VL-8B | InternVL3.5-8B | LLaVA-1.5-7B |
|---|---|---|---|
| decoder attention | `Qwen3VLTextAttention` | `Qwen3Attention` | `LlamaAttention` |
| rows the LLM consumes | `Qwen3VLVisionModel`'s `pooler_output` | `InternVLMultiModalProjector` | `LlavaMultiModalProjector` |
| image token | 151655 | 151671 | 32000 |
| delimiters | `<\|vision_start\|>` / `<\|vision_end\|>` | `<img>` / `</img>` | **none** |
| grid | per picture, from `image_grid_thw` | fixed 16x16 | fixed 24x24 |
| one grid cell, encoder pixels | 32 | 28 | 14 |
| extra injection points | DeepStack at ViT 8/16/24 | none | none |
| the grid covers | the whole picture | the whole picture | **a centre crop** |

`install(model)` picks the family; the module-level Qwen3-VL ids stay only so callers
that predate the seam still resolve.

### The view box, and why it is the dangerous one

LLaVA-1.5's processor resizes the short side to 336 and **centre-crops** to 336x336, so
its 24x24 grid covers a centred square of the picture and not the picture. `view_box`
is that square in normalised picture coordinates, and `patch_correspondence` now walks
grid → picture → (inverse transform) → baseline picture → baseline grid rather than
assuming the two frames are the same one. Both views default to the whole picture, which
is what Qwen3-VL and InternVL do and what every published number was computed under.

The centre crop is kept because it is how LLaVA-1.5 is normally run. The consequence is
stated wherever it matters: on that model **"the ring" is the border of what the encoder
saw**, which is a centre crop of the picture, and the report prints that line itself
whenever a run's view boxes are not the whole picture.

Validated on CPU against the real processors, before a single GPU-second was spent: a
marker painted at cell (r, c) of the grid, through the processor, lands at cell (r, c) of
the 336x336 / 448x448 tensor. **48/48 on each**, over six picture sizes and eight cells.

### Two traps this found

- **`InternVLProcessor` tiles even though `image_processor.crop_to_patches` is `False`.**
  The processor carries its own default and overrides the attribute, so a 500x300 picture
  came back as seven 448px tiles. The analysis assumes one grid, so every column statistic
  would have described the top-left corner of the picture while claiming to describe the
  picture — and nothing downstream would have looked wrong. `Family.proc_defaults` pins it,
  and the tiled configuration is now reachable only by asking for it.
- **LLaVA-1.5 fails "the scan reproduces stock SDPA (greedy tokens)"** at token 10 of 16.
  The scan and stock *eager* both take an explicit softmax where the fused kernel does not,
  and on fp16 weights read in bf16 that drift flips a near-tie. The reference is now eager
  — the path the scan is a copy of — and the SDPA agreement is printed rather than
  asserted. Against eager: **609/609 tokens**, and `|scan − eager|` is below one bf16 step
  of the logits' own magnitude.

## Which checkpoints

- **Qwen3-VL**: `Qwen/Qwen3-VL-8B-Instruct`, the released base model, because the other two
  are released models too. §16's cold start and §17's base run are the prior reference.
- **InternVL**: `OpenGVLab/InternVL3_5-8B-HF`, the **native** `InternVLForConditionalGeneration`
  path — no remote code, a real processor, and the tiling switch is a processor argument.
  `InternVL3_5-8B-Instruct-HF` does not exist on the Hub; the cached `-Instruct` snapshot is
  the remote-code repo written against transformers 4.51 and this environment runs
  5.13.0.dev0. The architecture — Qwen3, 36 layers, 32 heads, plus InternViT — is identical
  either way, and that architecture is what the comparison is about. The checkpoint is the
  flagship (post-RL) rather than the SFT-only one; recorded here rather than assumed away.
- **LLaVA-1.5**: `llava-hf/llava-1.5-7b-hf`, fetched once with `HF_HUB_OFFLINE` unset.

## The prompt

Each model gets its own chat template. Putting Qwen3-VL's `<think>` system prompt in front
of LLaVA-1.5 would measure an off-distribution model. `--system-prompt` controls it:
`auto` gives Qwen3-VL the project's trainer prompt so §16–17 reproduce, and **every
cross-model table is run under `none`**, which puts all three on the same footing. The
`prompt_swap` arm prices what remains.

## A10 — permute the pixels *before* the encoder

The new arm the plan asked for. A9 shuffles the rows the encoder **emitted** and shows the
attractor travels with the token, which rules out the language model's positional slot but
not the **encoder's own position embeddings** — those could have stamped the token on the
way through. A10 shuffles the pixel blocks that will become grid cells, before the vision
tower runs. If the attractor appears at whatever content now occupies the top-left *of the
ViT grid*, the encoder's position embedding is writing it; if it follows the original
content, it is content-driven.

The blocks are cut at the **encoder's own** pixel size (32 / 28 / 14), so the processor's
resize is the identity and the shuffle is lossless. That crop-and-resize is still not free
on the source side, so A10's baseline is **`permute_pixels_identity`** — the same resize
with the identity permutation — not `identity`. `ARM_BASELINE` is what carries that, and
the selftest checks the blocks are a permutation, that the pixels are the same multiset,
and that the grid the blocks were cut on is the grid the processor then chose.

## The gate: the refactor did not move the baseline

`--stage verify --against DIR` compares two scan directories unit by unit. Against
`outputs/sink_location/coldstart`, on 72 pictures:

```
grids identical on 72/72 units
largest |difference|:  0.000e+00 on all 26 statistics
peak patch identical in 82944/82944 cells (1.000000)
largest |difference| in the layer-mean maps: 0.000e+00
```

Bit for bit. A refactor that silently changed what the Qwen3-VL scan measures would have
invalidated the comparison the port exists to make, and it would have done so invisibly.

## How to run it

```fish
# one output directory per model -- the report refuses to pool two geometries
bash launch_sink_location_job.sh --name slx-qwen-base --stage selftest,scan,arms --gpus 8 \
    --duration 1 --out-dir outputs/sink_location/xmodel/qwen3vl_base \
    --model Qwen/Qwen3-VL-8B-Instruct -- --system-prompt none
bash launch_sink_location_job.sh --name slx-llava --stage selftest,scan,arms --gpus 8 \
    --duration 1 --out-dir outputs/sink_location/xmodel/llava15 \
    --model llava-hf/llava-1.5-7b-hf -- --system-prompt none
bash launch_sink_location_job.sh --name slx-internvl --stage selftest,scan,arms --gpus 8 \
    --duration 1 --out-dir outputs/sink_location/xmodel/internvl35 \
    --model OpenGVLab/InternVL3_5-8B-HF -- --system-prompt none

# InternVL's tiled configuration, as its own labelled arm
bash launch_sink_location_job.sh --name slx-internvl-tiled --stage arms --gpus 8 \
    --duration 1 --out-dir outputs/sink_location/xmodel/internvl35 \
    --model OpenGVLab/InternVL3_5-8B-HF -- --system-prompt none --arms identity,tiled

# the table
python sink_location_probe.py --stage crossmodel --out-dir outputs/sink_location/xmodel \
    --dirs outputs/sink_location/xmodel/qwen3vl_base,outputs/sink_location/xmodel/internvl35,outputs/sink_location/xmodel/llava15
```

---

# The result — 2026-09-15

**The same 1,800 pictures through three models, 150 per type, twelve types, all layers ×
all heads, every model on its own chat template with no system prompt.** Selftest passed
on each. Full output in `outputs/sink_location/xmodel/{qwen3vl_base,internvl35,llava15}/report.txt`
and the side-by-side in `outputs/sink_location/xmodel/crossmodel.txt`.

Three sentences. **The ring survives the encoder swap and half-survives the family swap**
— averaged over every head, 1.54 on Qwen3-VL, 1.67 on InternVL-3.5 and 1.29 on LLaVA-1.5,
where on LLaVA-1.5 it is one edge rather than a ring. **The upper-left peak does not
survive at all**: it is 13.1× on Qwen3-VL, 5.5× on InternVL, and on LLaVA-1.5 it has moved
to the opposite corner —
bottom-right at 13.0×, with the bottom row at 2.43 against a top row at 1.02. **And the
mark is held in different places**: shuffling the encoder's output rows moves it on
Qwen3-VL and InternVL and leaves it where it was on LLaVA-1.5.

So the answer to "is it Qwen3-VL or is it VLMs" is neither. It is a **raster-order effect
in all three, pointing in opposite directions**, and the paper's sentence has to be split.

## 1. The budget, first

| model | grid | the grid covers | the picture's share of an attention row |
|---|---|---|---|
| Qwen3-VL-8B-Instruct | 16x16 modal, per picture | the whole picture | 0.1096 |
| InternVL3.5-8B | 16x16 fixed | the whole picture | 0.1472 |
| LLaVA-1.5-7B | 24x24 fixed | **a centre crop** (569 distinct boxes) | 0.1348 |

Comparable across the three, and in all three the real sink is outside the picture. Every
number below is about how each model divides up its ~12%.

## 2. Where the picture's attention goes — all heads, pooled over twelve types

| model | ring | depth1 | middle | top | bottom | left | right | **top-left** | **bottom-right** |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-VL | 1.54 | 0.81 | 0.69 | **2.44** | 1.29 | **2.18** | 1.59 | **13.13** | 3.43 |
| InternVL-3.5 | **1.67** | 0.91 | 0.73 | **2.51** | 1.64 | 1.57 | 1.78 | 5.52 | 4.30 |
| LLaVA-1.5 | 1.29 | 1.03 | 0.92 | 1.02 | **2.43** | 0.98 | 1.27 | 2.10 | **12.96** |

Read the first and last columns together. Qwen3-VL and InternVL lean on the **early** half
of the border — top and left, and a single blazing top-left patch. LLaVA-1.5 leans on the
**late** half: bottom 2.43 against top 1.02, and 12.96 on the last patch. **It is the same
raster-order signature with the sign flipped**, which is what MCA-LLaVA and VisPruner
report for that family and what `docs/where-attention-goes.md` said to expect.

**LLaVA-1.5 does not have a ring; it has a bottom edge.** Its four sides are 1.02 / 2.43 /
0.98 / 1.27 and its interior is flat (depth1 1.03, middle 0.92). The 1.29 in the ring
column is the bottom row carrying three edges that do nothing. Calling that "the outer
ring" would be true arithmetic and a false picture.

The number that settles it is the **border with each model's hottest edge taken out** — a
side is a subset of the ring, so removing it is exact in both numerator and denominator:

| model | ring | interior | hottest edge | ring **minus** that edge |
|---|---|---|---|---|
| Qwen3-VL | 1.54 | 0.75 | top (2.44) | **1.18** |
| InternVL-3.5 | 1.67 | 0.79 | top (2.51) | **1.37** |
| LLaVA-1.5 | 1.29 | 0.94 | bottom (2.43) | **0.89** |

On Qwen3-VL and InternVL the border survives losing its best edge and the interior is
genuinely depleted: those two have a ring, with a gradient running across it. On LLaVA-1.5
the rest of the border falls **below chance** and the interior is not depleted at all: it
has one lit edge and no ring. So the honest decomposition is that **the raster-order effect
is in all three and the ring is in two of three** — LLaVA-1.5's hot row is a border row only
because the last token of a raster scan happens to land in a corner.

### The profile that makes it unambiguous

Enrichment per grid **row** and per grid **column**, at each model's modal grid. 1.00 is
that row's or column's fair share. A 2-D border effect has to be a **U on both axes**; a
raster-position effect is a **ramp on the row axis and nothing on the column axis**.

```
qwen3_vl  16x16, n=245
  row: 3.25 0.71 1.46 0.60 1.76 0.53 0.92 0.49 0.54 0.87 0.46 1.21 0.47 0.97 0.61 1.15
  col: 1.95 1.27 1.10 0.69 1.17 0.64 0.96 0.60 0.85 0.75 0.64 1.06 0.71 1.13 0.76 1.71

internvl  16x16, n=1800
  row: 2.37 1.13 1.01 0.79 0.79 0.78 0.80 0.92 0.73 0.75 0.73 0.85 0.82 0.95 1.10 1.48
  col: 1.43 0.89 0.82 0.79 0.95 0.92 0.97 0.88 0.88 0.94 1.08 0.93 0.92 0.94 0.96 1.69

llava     24x24, n=1800
  row: 1.06 0.96 0.71 0.56 0.57 0.56 0.67 0.74 0.80 0.77 0.78 0.78 0.84 0.81 0.93 1.07
       1.26 1.20 1.20 1.42 1.23 1.47 1.43 2.19
  col: 1.00 1.08 0.87 0.88 0.97 1.02 1.07 1.10 1.04 1.41 0.99 0.89 0.91 1.07 1.10 0.93
       0.98 0.98 1.04 0.88 0.81 0.89 0.94 1.15
```

Qwen3-VL and InternVL are U-shaped on **both** axes — first and last column 1.95/1.71 and
1.43/1.69, first and last row above an interior that sags to ~0.8. That is a border.

LLaVA-1.5 is **flat on the column axis** (1.00 at the left, 1.15 at the right, no U
anywhere) and its row axis is a **monotonic ramp** rather than a U: the top row is at
chance (1.06), the middle sags to 0.56, and the second half climbs without interruption to
2.19. Being at an edge buys nothing in this model. Attention grows with token index and the
ramp ends at the bottom row because that is where a raster scan ends. It also means
"bottom-right" overstates the result: the single corner patch is extreme (13.0x) but there
is no right-column preference at all.

Loose thread, not chased: Qwen3-VL's row profile alternates, even rows consistently above
odd ones (3.25, 0.71, 1.46, 0.60, 1.76, 0.53 ...). It may be an artefact of the 2x2 patch
merge. n=245 at that one grid shape.

Per type, the ring clears 1.5 in **9 of 12** types on InternVL, **8 of 12** on Qwen3-VL and
**2 of 12** on LLaVA-1.5 — where it also drops *below 1.0* on maths figures (0.92) and
board puzzles (0.97).

One caution about that sentence. §16's pre-registered 1.5 was a threshold on `E_ring` **at
the dev-selected cells**, where the cold start read 2.1–3.6 and cleared it twelve times out
of twelve. Every number in this section is the **all-head** average instead, because a
cross-model table must not be allowed to pick each model's most border-leaning cells and
then report that all three lean on the border. That makes it a stricter test than the
original, applied identically to all three, and it is the comparison between the three
columns that carries the argument — not the distance from a threshold that was fixed for a
different statistic.

## 3. Still not a sink — in any of the three

| model | peak ÷ uniform, at the strongest cell | peak CV across queries | verdict |
|---|---|---|---|
| Qwen3-VL | 30.9 | 0.85 | a peak that moves |
| InternVL-3.5 | 56.1 | 0.71 | a peak that moves |
| LLaVA-1.5 | 66.4 | 0.54 | a peak that moves |

Pre-registered: ≥10× uniform **and** CV ≤ 0.5. All three clear the magnitude leg by a wide
margin and **all three fail the invariance leg**. §16.4's conclusion about the *word*
generalises: what sits inside the picture is a peak, not a sink, in every family tested.

## 4. Not registers either — in any of the three

The border patches arrive from the vision tower with a **smaller** norm than the interior
in all three — 0.895 / 0.942 / 0.921 — before a single text token exists, and their keys
are not bigger (‖k‖ ring/interior 1.015 / 1.033 / 1.008). What differs is alignment with
where the queries look: **+0.550 / +0.698 / +0.278**. The border's keys are not large; they
are *aimed*, and they are aimed hardest in InternVL and least in LLaVA — the same ordering
as the ring itself.

## 5. Background loses twice and draws once

`E_blank_interior − E_ring`, on pictures whose blank *interior* covers ≥15% of the grid:

- **Qwen3-VL: negative in 11 of 11** types with data (−0.32 to −0.68), every CI excluding 0.
- **InternVL: negative in 10 of 12**, as far as −1.54 on maths figures.
- **LLaVA-1.5: negative in 7 of 12, and POSITIVE in three** — maths figures +0.082,
  abstract puzzles +0.122, board puzzles +0.115. On the model with no ring, a big interior
  blank is attended at least as much as the border.

H2 is refuted on the two models that have a ring and is not refuted on the one that does
not. That is consistent rather than convenient: there is no border effect on LLaVA-1.5 for
a background account to lose to.

## 6. The arms — and this is where the three models come apart

Paired against each picture's own baseline, at each model's own dev-selected cells.

| arm | Qwen3-VL | InternVL-3.5 | LLaVA-1.5 |
|---|---|---|---|
| `rot180` ΔE_ring | −0.023 | +0.033 | +0.034 |
| `rot180` follow slot | 1.000 | 0.829 | 0.902 |
| `rot180` follow content | 0.000 | 0.127 | 0.000 |
| **`permute`** (A9) ΔE_ring | **−1.687** | **−1.113** | **−0.194** |
| **`permute` follow content** | **0.998** | **0.652** | **0.015** |
| **`permute` follow slot** | **0.000** | **0.010** | **0.642** |
| `permute_identity` (control) | +0.000 / 1.000 / 1.000 | +0.000 / 1.000 / 1.000 | +0.000 / 1.000 / 1.000 |
| **`permute_pixels`** (A10) ΔE_ring | −0.130 | +0.059 | +0.002 |
| **`permute_pixels` follow content** | **0.000** | **0.002** | **0.000** |
| **`permute_pixels` follow slot** | **0.998** | **0.727** | **0.900** |

**`rot180` kills the content account everywhere.** Turn the picture upside down and the
peak does not move a patch in any of the three. The sky confound is dead in three families,
not one.

**A10 rules out content everywhere, and it is the new arm.** Shuffle the pixel blocks that
will become grid cells *before* the vision tower runs, and the peak follows the grid
position (0.998 / 0.727 / 0.900) and never the content that moved (0.000 / 0.002 / 0.000).
Whatever attracts attention is attached to a *place in the encoder's grid*, not to what is
drawn there — in Qwen3-VL, in InternVL and in LLaVA-1.5 alike.

**A9 is what says WHERE that place is recorded, and the three models disagree.** A10 cannot
settle it on its own: the ViT's grid index and the language model's slot index are the same
number, so an arm that holds both fixed cannot separate them. A9 moves one and not the
other —

- **Qwen3-VL: the mark is in the embedding.** The peak leaves its slot completely (0.000)
  and follows the vector it was sitting on (0.998). Reproduces §16.7 (−1.644 / 0.981 /
  0.000) on a different checkpoint and a different prompt.
- **InternVL-3.5: the same, weaker.** Follows the vector 0.652 of the time, leaves the slot
  0.990 of the time.
- **LLaVA-1.5: the mark is in the slot.** Follows the vector 0.015 of the time and *stays
  where it was* 0.642 of the time. Shuffling this model's patch embeddings costs its ring
  0.194, against Qwen3-VL's 1.687.

So: in Qwen3-VL and InternVL-3.5 the vision tower stamps a direction into the patch vector
and the language model attends to the stamp — H5, now evidenced on two encoders instead of
one, and with A10 narrowing "made in the encoder" to "made by the encoder's own position".
In LLaVA-1.5 the attractor is held by the language model's position, which is the
RoPE-decay account the LLaVA literature gives, and which A9 can distinguish and nothing
else here can.

## 7. InternVL's tiling — the ring is the ENCODER'S border, not the picture's

The doc called this the trap and said a tiled arm would be a stronger result if the ring
appeared on every tile's own border. It does. 480 pictures, tiling on, scored **per tile**
(3,048 tiles; 13 tiles for 127 of the pictures, 1 for 119 small ones):

| what was scored | n | ring | top | bottom | top-left | bottom-right |
|---|---|---|---|---|---|---|
| one 16x16 grid over the whole picture | 480 | 1.63 | 2.40 | 1.64 | 5.56 | 4.26 |
| every tile, on its own 16x16 grid | 3,048 | 1.38 | 2.07 | 1.19 | 3.81 | 2.79 |
| tile 0 (also the picture's corner) | 480 | **2.07** | **4.80** | 0.83 | 5.56 | 2.41 |
| every tile but the last | 2,568 | 1.35 | 2.10 | 1.08 | 3.29 | 2.38 |
| the last tile (the thumbnail) | 480 | 1.53 | 1.87 | 1.78 | 6.62 | 5.00 |

**A tile's own border is enriched even when it is an interior edge of the picture** (1.35 on
the non-thumbnail tiles) and every tile has its own lit top-left patch (3.29). The effect
tracks the boundary of what the encoder was handed, not the boundary of the photograph.
That is an independent confirmation of A10 from the other direction: A10 moves the content
within a fixed encoder input, tiling moves the encoder input around fixed content, and both
say the mark belongs to the encoder's grid.

## 8. What this changes

Against the outcomes written down before the run, this is the fourth: **ring without the
upper-left peak — two mechanisms, not one, and the paper must split the claim.**

1. **"Attention concentrates on the outer ring of the patch grid" is not a VLM-wide
   statement.** It holds on Qwen3-VL and InternVL-3.5 — all four edges above chance, the
   interior depleted to 0.75/0.79, and 1.18/1.37 left after the hottest edge is removed.
   It fails on LLaVA-1.5, whose column profile is flat and whose row profile is a
   monotonic ramp rather than a U: being at an edge buys nothing there, and the border
   minus its bottom row is 0.89, below chance. Two of three, not three of three -- and on
   the third what looks like an edge is the end of the raster order.
2. **The raster-order signature IS VLM-wide, and its direction is not.** All three put
   several times their share on one end of the token sequence; Qwen3-VL and InternVL pick
   the first token, LLaVA-1.5 the last. Any claim about "the top row" is a claim about a
   family.
3. **The mechanism differs by family, and A9 is the only arm that shows it.** Encoder stamp
   in two of three, language-model position in the third. A paper that reports the border
   effect without A9 would attribute all three to the same cause.
4. **A10 is worth its cost.** It rules out content in every family, in one cheap arm, and
   with the tiling result it pins the effect to the encoder's input grid.
5. **For this project's reward**: nothing changes about Qwen3-VL — §16 and §17 reproduce
   bit for bit — but the reason `--overlap_rect_frac` was fighting a lit corner is now
   known to be an **encoder-architecture** property rather than a VLM one. A reward built
   on `mean_in` would meet a differently-shaped, and on LLaVA-1.5 a weaker and inverted,
   obstacle.

## 9. Caveats

- **One checkpoint per family.** InternVL3.5-8B is the flagship post-RL release, not the
  SFT-only `-Instruct`; the cross-family contrast is about architecture, but a
  checkpoint-level contribution is not excluded.
- **LLaVA-1.5's grid covers a centre crop**, because that is how the model is normally run.
  Its "ring" is the border of the crop. The bottom-heavy result is not an artefact of that
  — a centre crop is symmetric top-to-bottom — but its per-type numbers are about a
  different set of pixels from the other two models'.
- **Prefill, not generation.** §17.2 measured the prefill readout as ~1.5× the
  generated-token one on Qwen3-VL. The direction and the ordering survived that there; it
  has not been re-measured on the other two, and `--max-new-tokens` is what would.
- **`MAX_IMAGE_SIDE = 512`** still, so LLaVA-1.5 and InternVL both upsample most pictures to
  their native 336/448. The resolution ladder moves the number and does not reach past 512.
- **A10's follow-slot is 0.727 on InternVL**, against 0.990 for its own identity control —
  so roughly a quarter of its pictures do move. The arm is clear in direction and not
  absolute.
