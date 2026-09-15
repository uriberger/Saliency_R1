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
