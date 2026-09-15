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
