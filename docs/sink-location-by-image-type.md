# Is the sink in the outer ring, or is the outer ring just the background? — 2026-09-08

**Answered, on 1,800 pictures across twelve image types, at the cold start.** Full output
in `outputs/sink_location/coldstart/report.txt`. Sections 1–15 below are the design, which
was written before the run and is unchanged except where marked; **section 16 is the
result.** Read section 16 first if you want the answer, and section 1 before believing any
percentage in it.

Three sentences: **the border is enriched in every one of the twelve types, including the
ones where the blank region is in the middle** — so it is not the background. **Turning the
picture upside down does not move it** — so it is not the content either. **Permuting which
patch embedding sits in which grid slot moves it, and the peak follows the embedding** — so
it is not the language model's own position: the mark is stamped into the patch vector by
the vision tower, and the language model attends to the mark.

---

# The design, as written on 2026-09-07

[peak-location-results.md](peak-location-results.md) says, of the cold start on `val_natural`:

> 82% of peaks sit on the outer ring of the patch grid, 55% on the top row, 27% in a
> literal corner.

and [inference-intervention.md](inference-intervention.md) builds a whole intervention on
top of it — `frame` is the source of the edit because "the sink is the frame, which holds
~half the image attention and 76–85% of the map peaks."

That number was measured on **30 natural photographs, one layer, two heads**. The prior
literature says sinks land in the **background**, and in a photograph the border *is* the
background, so the two accounts are indistinguishable on this corpus. This plan separates
them, on image types where border and background come apart.

Everything here is prefill-only: no generation, no Grounding-DINO, no judge. The whole
experiment is **under 8 GPU-hours** — about an hour on one 8-GPU node — which makes it the
cheapest causal question this project has asked.

---

## 1. Scope the claim before measuring it

Three different things get called "the sink", and the existing 82% is only the first.

| | definition | what it is |
|---|---|---|
| **S1** | `argmax` patch of the attention map | what `peak_location_probe.py` measured. The *maximum of a distribution*, which exists whether or not anything is a sink |
| **S2** | share of the row's **image** mass on the ring, over the ring's area share | concentration, not a single patch. Robust to ties and to quantisation |
| **S3** | a **sink proper**: a token whose received attention is ≫ uniform, in most heads, and *does not depend on the query* | the literature's definition. Requires magnitude **and** query-invariance |

S1 and S2 can be strong while S3 is empty — that would mean "within the picture, attention
leans on the border", not "there is an attention sink in the picture". The distinction is
not pedantic here: [inference-intervention.md §3](inference-intervention.md) measures the
image's share of a whole attention row at **0.4%–1.4%** at L22 h28/31. If that holds across
layers, then ~99% of every row is on text and BOS, and the image's "sink" is a ripple on a
1% budget.

**So the first table the report prints is the budget**, per layer and head: mass on BOS, on
the system prompt, on `<|vision_start|>` / `<|vision_end|>`, on the image ring, on the image
interior, on the question. Every later percentage is read against it. If the image ring
carries 0.2% of a row, the honest claim is "*within the image*, the border is favoured",
and the write-up says so in those words.

## 2. What could be true instead

| | hypothesis | mechanism |
|---|---|---|
| **H1** | **sequence position** | image tokens are raster-ordered; the first ones are adjacent to `<|vision_start|>` and are the earliest keys every later query can see. The classic StreamingLLM sink is *positional*, and the top row is the first 10% of the image tokens. The **55% top-row / 27% corner** asymmetry is H1's fingerprint: a 2D border effect has no reason to prefer the top |
| **H2** | **background content** | sinks go where there is nothing to look at. In photographs that is the border (and the sky, which is also the top). The prior literature's account |
| **H3** | **2D border geometry** | the ViT's learned position embeddings (`num_position_embeddings: 2304`, interpolated to each grid) and the LLM's 2D M-RoPE both put border patches at extremal coordinates. Border patches also have fewer neighbours in the vision tower, so their features are built from a truncated context |
| **H4** | **registers / massive activations** | some tokens carry outlier-norm hidden states used as scratch space (Darcet et al.; Sun et al.). Allocated to low-information patches, which correlates with, but is not, the border |
| **H5** | a mixture, in shares that differ by image type | the likely answer. The deliverable is then the **shares**, not a winner |

My prior, from the 55/27 split alone, is **H1 + H3 dominant, H2 a minority share on natural
photos only**. The experiment is built to make me wrong cheaply if that is wrong.

## 3. The measurement

For one image, patch grid `gh x gw`, `N = gh*gw` image tokens, layer `l`, head `h`:

- `p[l,h,i]` = attention received by image token `i`, averaged over a fixed query set, then
  renormalised over the image only. Query set is **the prompt's text tokens after the
  image** (primary), with **the image tokens themselves** and **generated tokens** as two
  secondary readouts. All three come out of one forward.
- **Ring** = Chebyshev depth 0, i.e. `sink_shift.frame_set(gh, gw)`. Area fraction
  `(2gh + 2gw − 4)/(gh·gw)` — **0.23 on 16x16, 0.30 on 10x16, 0.50 on 6x8**. That spread is
  larger than most effects anyone reports, so *nothing* is reported as a raw percentage.
- **Primary endpoints**, per image, then averaged within type:
  - `L_ring = P(S1 ∈ ring) − ring_area_frac`  (lift; the 82% becomes +0.52)
  - `E_ring = ring_mass_share / ring_area_frac`  (enrichment; 1.0 = no effect)
- **Secondary**: radial profile `E(d)` by Chebyshev depth `d` (subsumes any choice of ring
  width); **edge-resolved** enrichment for top / bottom / left / right and the 4 corners
  (H1 lives here); the layer x head map of `E_ring`; S3's query-invariance (CV of `p` across
  queries); agreement among S1/S2/S3.
- **Negative controls on the metric itself**: a random contiguous patch set of the same
  size as the ring, and the depth-2 ring (`ring2_set`). Both must sit near 1.0.

Reusable as-is from `sink_shift.py`: `_locate_images` (image token runs → per-image
`(t, gh, gw)`), `patch_set` / `frame_set` / `ring2_set` / `core_set`, and the registered
attention implementation that already sees real keys and values in prefill and decode. What
is new is a reducer that keeps the per-image-token column statistics at **every** layer and
head, in the shape of `_record_survey`, so no `[heads, q, kv]` tensor outlives its forward.

## 4. The corpus — everything is already on disk, offline

`val_natural` / `val_nonnatural` carry a `dataset` column naming the source, so eleven
strata come for free. Puzzles and the synthetic set come from the benchmark caches
`eval_mini/benchmarks.py` already uses.

| # | type | source | why it is in the design |
|---|---|---|---|
| 1 | photographs, object/relation VQA | `val_natural`: gqa, aokvqa, visual7w, openimages, vsr | the corpus the claim was made on |
| 2 | photographs, dense aerial | `val_natural`: visdrone | no sky, no centred subject — kills the photographer-bias story on its own |
| 3 | charts & tables | `val_nonnatural`: virl_charts + ChartQA | **large interior whitespace**: blank and border come apart without any synthesis |
| 4 | scanned documents | `val_nonnatural`: docvqa | ink fills the frame to the margins; the border is *not* blank |
| 5 | infographics | `val_nonnatural`: infographicsvqa | dense, coloured, edge-to-edge |
| 6 | math & geometry figures | `val_nonnatural`: virl_math_geo + MathVision | line art on white; most of the image is "background" |
| 7 | science diagrams | `val_nonnatural`: virl_science + ScienceQA | labelled diagrams, mixed |
| 8 | abstract reasoning puzzles | VisuLogic (`data.jsonl`, tagged by kind) | Raven-style grids: content is *uniformly tiled*, including the border |
| 9 | algorithmic / board puzzles | AlgoPuzzleVQA | boards fill the frame; the border is a board edge, i.e. informative |
| 10 | synthetic pop-out | SALBench **P3** (3 shards x 1000) | homogeneous distractor field everywhere, one odd item at a random position. **Background is not a place** here — H2 has nowhere to point |
| 11 | exam pages / screenshots | MMMU-Pro standard | mixed text+figure layout |
| 12 | illusions | IllusionVQA soft-localization | optional; low-texture fields |

**n = 150 images per type** (P3 and VisuLogic can go higher for free), one question each,
`MAX_IMAGE_SIDE = 512` and `prepare_image()` exactly as `overlap_probe.py` applies them, so
grids match everything else in this project.

Types 3, 4, 6, 8, 9 and 10 are why this design does not need to trust its own synthetic
images: **they already dissociate blank-from-border.** In a chart or a math figure the
largest blank region is usually *interior*. H2 predicts the sink sits in it. H1/H3 predict
the sink stays on the border while a big empty interior region is ignored. That contrast is
observational, free, and it is the single most informative number in Stage 2.

## 5. Stage 0 — the corpus, on CPU (~1 h, `cpu_short`)

1. Materialise the 12 strata to a manifest with a stable `row_key`.
2. **Grid census**: the `(gh, gw)` each image actually gets, per type, and the ring area
   fraction it implies. If math figures land on 16x16 (0.23) while photographs land on
   10x16 (0.30), then a cross-type comparison of raw ring percentages is meaningless and
   every table must be within-grid-shape as well as pooled. Measuring this *first* is what
   stops that error being discovered in the results.
3. **Per-patch content statistics**, on the same grid: pixel variance, edge energy (Sobel),
   distance to the nearest ink/non-background pixel, "is blank" (a flat-field test), and a
   CLIP patch-feature norm (`openai/clip-vit-base-patch16` is cached). These are the
   covariates §8's regression needs, and the blank-region maps §4 leans on.
4. Synthesise the fixed image sets for Stage 3 (donut, canvas-translation, noise-pad).

## 6. Stage 1 — tie back to the 82%, then license the cheap readout (1 GPU, ~1 h)

Two gates. Neither is optional; both are the lesson `flow_intervene_probe.py` paid for.

**1a. Reproduce.** On the same `val_natural` images, at L22 heads 28/31, from **generated**
chains, recompute `P(peak ∈ ring)`. It must land at 0.82 ± bootstrap. If it does not, the
new code is measuring something else and nothing downstream means anything.

**1b. Validate the proxy.** The whole experiment is prefill-only. So on those same images,
correlate the prefill-query readout with the generated-token readout, per image, at the same
cells. Pre-registered acceptance: **Spearman ≥ 0.8 on `E_ring` and agreement ≥ 0.9 on
`S1 ∈ ring`**. Below that, Stage 2 runs on generated tokens instead and costs 10x more —
which is affordable, but the report must say which readout produced every number.

**1c. Coordinate frame.** A synthetic image with one bright patch at a known pixel location
must produce a content peak at the predicted `(row, col)`, and must still do so after each of
Stage 3's transforms. **This is where this experiment is most likely to go quietly wrong**:
if `rot90`'s pixel→patch mapping is off, the content-vs-position analysis decodes the wrong
frame and produces a confident, meaningless answer.

## 7. Stage 2 — the observational sweep (8 GPUs, ~1 h)

12 types x 150 images x 1 prefill, **all 36 layers x 32 heads**, plus hidden-state norms.
Outputs, per type:

- the attention budget of §1 (where the row's mass actually is);
- `L_ring`, `E_ring` with 95% cluster-bootstrap CIs over images;
- the radial profile `E(d)`, and the edge/corner decomposition;
- the layer x head map: **in what fraction of the 1,152 cells is `E_ring > 1`**, and which
  layers carry it. "82% at one cell" and "82% everywhere" are different claims;
- S3: is there a token that is both large and query-invariant, and where is it;
- **`P(sink ∈ largest interior blank region | that region ≥ 15% of the grid)`**, against its
  own area chance. This is H2's home turf, and types 3/4/6/8/9 are where it has one.

**Head selection is done on a held-out dev split** of 100 images that never enters any
reported estimate. Picking "the sink heads" on the same data that reports their effect is
the easiest way to manufacture a result here.

## 8. Stage 3 — the arms that separate position from content (8 GPUs, ~1 h)

Each arm is the **same images** with exactly one thing changed, paired against its own
baseline by common random numbers. 400 images (pooled across types, stratified).

| arm | change | H1 sequence | H2 background | H3 2D border | H4 registers |
|---|---|---|---|---|---|
| **A1** `rot180`, `rot90`, `hflip` | content moves, ring maps to itself | sink **stays on the top row** | sink **follows the old top content** | ring holds, edges symmetric | follows the low-info content |
| **A2** centre-crop + zoom | the ring is now foreground; the old background is gone | ring keeps it | sink **leaves the ring** | ring keeps it | leaves |
| **A3** uniform pad, 1 and 2 patches, in grey / white / **noise** | the new ring is blank (or noisy) | new ring takes it in every colour | takes it when blank, **not when noise** | new ring, every colour | blank yes, noise no |
| **A4** paste onto a uniform canvas, 9 positions | blank is everywhere, the object moves | ring, never the blank beside the object | blank anywhere, uniformly | ring | blank anywhere |
| **A5** donut / vignette: content on the border, blank centre | the middle is the background | ring | **centre** | ring | centre |
| **A6** two pictures in one prompt | a second image run, later in the sequence | image 1 only (absolute) *or* both top rows (delimiter-relative) — either way informative | both, wherever each is blank | both rings | both |
| **A7** resolution ladder, `MAX_IMAGE_SIDE ∈ {256, 384, 512}` | same picture, three grids | fixed **absolute token index** | fixed **content** | fixed **normalised position** | fixed content |
| **A8** prompt swap: 4 questions, "Describe the image.", empty | query changes, image does not | invariant | invariant | invariant | invariant — a *failure* here refutes S3 outright |
| **A9** **permute the patch embeddings across slots** | content and grid position decoupled by construction, no pixels involved | sink **stays at the same grid slots** | sink **follows the patches it moved with** | stays at the slots | follows |

**A9 is the crown.** Everything else argues from pixels and can be answered "your transform
changed the content in some way you did not model". A9 changes nothing but *which slot holds
which vector*: position ids stay with the slot, the vision tower has already run, and the
model's answer becomes nonsense — which is fine, because the readout is where attention
goes, not whether the answer is right. Its mirror (keep content, permute the M-RoPE image
position ids) is the same test from the other side and is worth building if A9 is ambiguous.

**A1's `rot180` alone settles the sky confound.** GQA and COCO photographs have low-texture
sky at the top; "55% top row" and "55% on the flattest band" are the same observation until
the picture is turned upside down.

## 9. Stage 4 — mechanism: why (8 GPUs, ~2 h)

Only run in full on whichever hypothesis survives Stage 3; all four are cheap.

- **M1 — massive activations.** `‖h_i‖₂` per image token at every LLM layer, at the vision
  tower's output, and at the three DeepStack injection points (vision layers 8/16/24). If
  ring tokens are norm outliers *at the vision tower's output* — before a single text token
  exists — the sink is set by the encoder and the LLM inherits it. Which dimensions carry
  the norm, and whether they are the same few for every image, decides H4.
- **M2 — key-norm vs alignment.** In the top sink heads, split each image token's mean logit
  into `‖k_i‖` and its alignment with the mean query direction. A pure key-norm effect is
  the register story; alignment with a query-constant direction is the sink-direction story.
  These have different fixes.
- **M3 — vision tower alone.** Locate ring bias in the ViT's own attention, on the same
  grids. If it is already there, §8's arms are describing something inherited.
- **M4 — the regression that answers the question as asked.** Per patch:
  `sink ~ ring + depth + top_row + is_blank + pixel_var + edge_energy + clip_norm + type`,
  fit within type and pooled with type interactions. **The reported quantity is the partial
  effect of `ring` after the content covariates, and the share of deviance each block
  explains.** That is the number that says "outer ring" and "background" in the same units.
- **M5 — position ablations** (optional): interpolate away the ViT's learned position
  embeddings, or flatten the M-RoPE h/w sections for image tokens, and re-read the attention
  statistics. Destroys the model's output; the readout is attention only.

## 10. Stage 5 — is it Qwen3-VL's, or is it VLMs' (8 GPUs, ~3 h)

Stage 2 only, on cached models: **Qwen3-VL 2B / 4B / 8B / 32B** (does the sink layer move
with depth?), **Qwen2.5-VL-7B** (different ViT, windowed attention), **InternVL3.5-8B**
(different family and tiling scheme), and this project's own `coldstart` and
`mean_in`-8k checkpoints (did the reward move it?). A claim that survives three families is
a claim about VLMs; one that does not is a claim about one checkpoint, which is still worth
writing down accurately.

## 11. Statistics, fixed before any run

- Unit of analysis is the **image**. 10,000-resample cluster bootstrap within type; arm
  contrasts paired by common random numbers, as `peak_location_probe.py` already does.
- **Primary**: `E_ring` and `L_ring` per type. Holm across the 12 types, two endpoints.
- **Effect-size thresholds, pre-registered**: `E_ring ≥ 1.5` = "the sink concentrates on the
  ring"; `1.2–1.5` = weak; `< 1.2` with a CI excluding 1.5 = **the claim fails for that
  type**. Sign-only significance is not enough at n=150 per cell.
- Every table is reported **within grid-shape bin** as well as pooled.
- Cells whose image mass is below the §1 budget floor are printed but not interpreted, the
  way `--min-format` gates the intervention's cells.

## 12. What would falsify what

- **`E_ring` high in every one of the 12 types, and A2/A5/A9 keep the sink on the ring** →
  the claim holds and is *positional*. Rewrite the paper's sentence: not "sinks are in the
  background", but "sinks are on the grid border, and in photographs the border happens to
  be the background". `frame` stays the right source set for the intervention.
- **`E_ring` high on photographs, near 1 on charts / math / documents / P3, and the sink
  sits in interior blank regions there** → the border was a proxy for background all along.
  The claim as written is wrong outside natural imagery, and the intervention's `frame`
  source is mis-specified for the non-natural half of the benchmark — which is exactly the
  half where [next-reward-experiments.md](next-reward-experiments.md) found the box-aware
  channel carrying the transfer.
- **A1 `rot180` moves the top-row mass to the bottom** → content, not raster order. H1 dies
  and the 55% was sky.
- **A1 keeps the top row, A9 keeps the slots, A7 pins the absolute token index** → it is
  sequence position. The mechanism is the same one StreamingLLM describes, and "outer ring"
  is a 2D restatement of "first tokens".
- **S3 finds no query-invariant, large-magnitude token inside the image** → there is no
  attention sink in the picture at all; there is a *peak*, and the word "sink" should be
  dropped from the claim. The paper's sentence needs rewriting even if every ring number is
  high.
- **The §1 budget shows the image ring holds <0.5% of a row** → true but small. Say both.

## 13. Files, and the trap

| file | what |
|---|---|
| `sink_location_probe.py` | `--stage selftest / corpus / scan / arms / mechanism / report` |
| `launch_sink_location.sh` | shards `scan` and `arms` over a node's GPUs; **selftest gates the run**, as `launch_sink_shift.sh` does |
| `test_sink_location_cpu.py` | patch sets agree with `sink_shift.patch_set` on every grid 4x4..21x25; ring/edge/radial masks sum to the grid; **every transform's pixel→patch mapping**, checked by a single bright patch at a known place |
| `docs/sink-location-by-image-type.md` | this file, then the results |

Selftests, all CPU except the first:

1. the probe's attention implementation reproduces the fused kernel's output to `1e-4` when
   it edits nothing — the `α = 0` identity `sink_shift` already insists on;
2. the ring's measured area fraction equals `(2gh+2gw−4)/(gh·gw)` on every grid seen;
3. the two negative-control sets (random contiguous, depth-2 ring) come back at `1.00 ± ε`
   on shuffled maps;
4. `rot90 ∘ rot90 ∘ rot90 ∘ rot90` is the identity **through the grid decoder**, not just on
   the pixels;
5. A9's permutation is a permutation: the multiset of patch embeddings is unchanged, and the
   inverse permutation restores the baseline attention bit-for-bit.

## 14. Cost

| stage | where | time |
|---|---|---|
| 0 corpus + content stats | `cpu_short`, no GPU | ~1 h |
| 1 tie-back + proxy validation | 1 GPU | ~1 h |
| 2 observational sweep, 12 types | 8 GPUs | ~1 h |
| 3 the nine arms | 8 GPUs | ~1 h |
| 4 mechanism | 8 GPUs | ~2 h |
| 5 cross-model | 8 GPUs | ~3 h |

Under 8 GPU-hours of real work; one node-hour wall clock for stages 2–3, which are the ones
that answer the question. Stages 4 and 5 are only worth buying once 2 and 3 have an answer.

## 15. Caveats, stated in advance

- **`MAX_IMAGE_SIDE = 512` gives grids of at most 16 patches on the long side**, so the ring
  is 23%–50% of the picture. This is a *low-resolution* study of a border effect, and a
  border one patch wide is 1/10th of the image. At the resolutions a deployed model uses,
  the same ring is a much thinner band and the effect may be a different size. The
  resolution ladder (A7) is the only handle this design has on that, and it does not reach
  past 512.
- **One question per image.** A8 tests query-invariance on a subset; the main sweep does not.
- **The 12 "types" are 12 corpora**, and a corpus differs from another in more than its
  imagery — question style, aspect ratio, JPEG history. The within-image arms of Stage 3 are
  what carry the causal weight; the type sweep establishes *where* the phenomenon is, not
  why.
- **P3 has no published target coordinates** in the cached parquet. Its value here does not
  depend on them — a homogeneous field leaves H2 with nowhere to point — but recovering the
  odd item's position by simple detection would add a real content control, and that has not
  been written.
- **Prefill queries are not what the reward saw.** The training signal was computed on
  generated tokens. Stage 1b is what licenses the substitution, and if it fails the cost
  goes up tenfold rather than the conclusion changing.

---

# 16. The result — 2026-09-08

Cold start (`coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged`), 1,800 pictures,
150 per type, 12 types. 457 dev / 1,343 test. All 36 layers × 32 heads. Headline cells are
the 16 with the largest ring enrichment **on the dev split**, reported on the test split.
Selftest passed on the real corpus, including every transform's pixel→patch mapping at
every picture size in it. `outputs/sink_location/coldstart/report.txt`.

Total cost: **under 15 GPU-minutes.** The scan is 74 s per shard on 8 GPUs; the arms are
about 4 minutes. No generation, no Grounding-DINO, no judge.

## 16.1 The budget — read this before any percentage

| span | share of an attention row, over all 1,152 cells |
|---|---|
| first token | 0.470 |
| everything before the picture | 0.609 |
| `<\|vision_start\|>` | 0.0044 |
| **the picture** | **0.087** |
| `<\|vision_end\|>` | 0.015 |
| the question and the assistant header | 0.285 |

The picture receives **8.7% of a row** — far more than the 0.4–1.4% that
[inference-intervention.md §3](inference-intervention.md) measured at L22 h28/31, which is
a statement about those two heads and not about the model. But the first token alone takes
47%, so the real sink is where the sink literature says it is, outside the image. Every
number below is about **how the model divides up its 8.7%**.

## 16.2 The ring, per image type — the claim holds everywhere

`E_ring` = the border's share of the picture's attention ÷ the border's share of the
patches. 1.0 is no effect. `L_ring` = P(peak on the border) − that same chance level.
CIs are a 10,000-resample bootstrap over pictures; p is Holm-adjusted across the twelve.

| type | n | modal grid | ring area | E_ring | 95% CI | L_ring |
|---|---|---|---|---|---|---|
| synthetic pop-out (P3) | 107 | 16x16 | 0.234 | **3.58** | [3.55, 3.60] | +0.764 |
| puzzle, board (AlgoPuzzleVQA) | 110 | 16x16 | 0.262 | 3.10 | [3.04, 3.16] | +0.738 |
| photographs (GQA/A-OKVQA/…) | 114 | 12x16 | 0.283 | 2.81 | [2.77, 2.84] | +0.707 |
| photographs, aerial (VisDrone) | 114 | 9x16 | 0.300 | 2.74 | [2.70, 2.78] | +0.699 |
| scanned documents | 119 | 16x12 | 0.278 | 2.62 | [2.58, 2.66] | +0.710 |
| illusions | 110 | 8x16 | 0.344 | 2.58 | [2.57, 2.59] | +0.656 |
| charts and tables | 113 | 14x14 | 0.348 | 2.52 | [2.42, 2.62] | +0.650 |
| puzzle, abstract (VisuLogic) | 117 | 7x16 | 0.347 | 2.49 | [2.41, 2.57] | +0.652 |
| exam pages (MMMU-Pro) | 109 | 10x16 | 0.360 | 2.48 | [2.37, 2.58] | +0.639 |
| math & geometry figures | 108 | 8x9 | 0.400 | 2.20 | [2.12, 2.29] | +0.600 |
| infographics | 117 | 16x11 | 0.376 | 2.11 | [2.03, 2.19] | +0.604 |
| science diagrams | 105 | 7x10 | 0.410 | 2.10 | [2.02, 2.19] | +0.585 |

**Twelve of twelve clear the pre-registered 1.5, p < 0.002 after Holm.** The spread is
real but small: 2.1 to 3.6, and it does not line up with "how much background is at the
edge". The *highest* enrichment is on P3, the synthetic pop-out field where the picture is
a homogeneous array of distractors and "background" is not a place. The *lowest* are the
science diagrams and math figures, which are the types with the most white space — the
opposite of what a background account predicts.

It is not a cell either: **99.7% of the 1,147 cells that clear the image-mass floor have
E_ring > 1**, 40% above 1.5. The rewarded pair sits at E_ring 2.33 (h28) and 1.84 (h31).

## 16.3 The shape — and this is where the answer starts

Enrichment by set, pooled over types (each column is that set's share ÷ its area share;
`first` and `last` are single patches priced against a flat map's 1/N):

| ring | depth 1 | deep | top | bottom | left | right | corner (mean of 4) | **first patch** | last patch |
|---|---|---|---|---|---|---|---|---|---|
| 2.61 | 0.32 | 0.24 | 7.48 | 0.75 | 7.29 | 1.59 | 21.6 | **72.5** | 3.4 |

(Pooled over pictures, not over type means — the two differ in the second decimal and the
picture-weighted form is what every other number here uses.)

Two things a 2-D border effect cannot explain:

- **top 7.5× and left 7.3×, against bottom 0.75× and right 1.59×.** The border is not
  enriched; its *early* half is. The bottom row is *below* chance.
- **the single top-left patch is at 72×**, more than three times the four-corner average
  and twenty-three times the last patch. Per type it runs from 46.6 (math figures) to
  118.4 (P3).

That is the raster-order signature, and it made H1 (sequence position) the leading account
right up until the permutation arm.

## 16.4 Is it a sink at all — no

| | at the ring cells | at the strongest cell anywhere |
|---|---|---|
| peak ÷ uniform | 4.9 | 33.4 |
| peak CV across queries | 2.55 | 1.39 |
| normalised entropy | 0.50 | 0.52 |

Pre-registered: a sink needs ≥10× uniform **and** CV ≤ 0.5. Nothing meets both. At the
strongest cell in the whole model — which is L2 h17 on 524 of 1,343 pictures — the top
column is 33× uniform and still moves with the query (CV 1.39).

**So "attention sinks are located in the outer ring" is the wrong sentence even though
every number in §16.2 is high.** What sits on the border is a *peak*, and a peak whose
location depends on the query is not what the sink literature describes. The accurate
sentence is: *within the picture, attention concentrates on the border, especially its
first row and column.*

## 16.5 Background, tested four ways — it loses all four

1. **Interior blank regions.** On pictures whose blank *interior* region covers ≥15% of the
   grid (n = 6–99 per type), `E_blank_interior` is 0.79–1.19 while `E_ring` on the same
   pictures is 1.23–1.73. The gap is negative in **all eleven** types with data, from
   −0.30 to −0.58, every CI excluding 0. Where the blank is in the middle, the attention
   does not go there.
2. **Zoom.** Cropping to the central 60% and rescaling makes the border foreground.
   `dE_ring = +0.020` [+0.008, +0.032] — it goes *up*.
3. **Padding.** A grey pad costs 0.032, a white pad 0.001, a **noisy** pad 0.118. Blankness
   is worth something at the margin, and it is an order of magnitude smaller than the
   effect itself.
4. **The regression.** With blankness, edge energy, pixel variance and radial depth in the
   model, `b_ring` stays **positive in 12 of 12 types**, mean +0.49 (0.14 documents to 0.83
   P3). Dropping `ring` costs R² in every type.

## 16.6 Not registers either

The vision tower's border patches arrive with **smaller** norms than the interior:
ring/interior = **0.895** [0.889, 0.902], before a single text token exists. The LLM
hidden-state ratio never exceeds 1.01 at any layer. And the border's keys are not bigger:
‖k‖ ring/interior = **1.012**. So this is not a massive-activation or register story.

The logit gap is **alignment**: `align_ring − align_interior = +0.542` [+0.537, +0.547]
against a key-norm ratio of 1.012. The border's keys point where the queries look. They
are not large; they are *aimed*.

## 16.7 The arms — position beats content, then the permutation moves the goalposts

Paired against each picture's own baseline, at the dev-selected cells. `follow content` is
the share of pictures whose peak patch shows the baseline's peak patch; `follow slot` is
the share whose peak sits at the same grid position.

| arm | ΔE_ring | ΔE_top | follow content | follow slot |
|---|---|---|---|---|
| `rot180` | −0.008 [−0.017, +0.000] | −0.055 | 0.000 | **1.000** |
| `rot90` | +0.007 [−0.002, +0.015] | +1.662 | 0.000 | 0.989 |
| `hflip` | −0.013 | −0.081 | 0.000 | 0.998 |
| `zoom60` | +0.020 | +0.105 | 0.000 | 0.998 |
| `canvas4` (object shrunk onto a blank field) | −0.210 | −0.393 | 0.000 | 0.996 |
| `pad_white_1` | −0.001 | +0.062 | 0.000 | 0.998 |
| `pad_noise_1` | −0.118 | −0.242 | 0.000 | 0.991 |
| `two_images` | −0.054 | −0.204 | — | 0.994 |
| `prompt_swap` | +0.074 | +0.333 | — | 0.996 |
| **`permute`** | **−1.644** [−1.708, −1.581] | **−5.998** | **0.981** | **0.000** |
| `permute_identity` (control) | +0.000 | +0.000 | 1.000 | 1.000 |

**`rot180` is the sky confound's obituary.** Turn a photograph upside down and the peak
does not move a patch (`follow slot` 1.000, ΔE_top −0.055). The 7.4× top row is not the
sky, not the horizon and not the photographer.

**`permute` is the result.** It touches no pixel: the vision tower runs on the unmodified
picture and its output rows are shuffled, so position ids, the grid, the prompt and the
token count are all identical and only *which slot holds which vector* changes. The peak
leaves its slot completely (0.000) and **follows the embedding it was sitting on (0.981)**.

`permute_identity` returns +0.000 exactly on every column, which is what says the machinery
is not inventing this.

## 16.8 The hypothesis

Scored mechanically against the predictions written down beforehand:

| | | |
|---|---|---|
| H1 sequence position | 4/5 | dies on the permutation: if the LLM's slot held the mark, shuffling embeddings would not move it |
| H2 background content | **0/4** | every leg fails, including its own home turf |
| H3 2-D border geometry | 2/3 | right that it is positional, cannot explain top ≫ bottom |
| H4 registers / massive activations | **0/2** | the border arrives *smaller*, and its keys are not bigger |
| **H5 the vision tower's positional signature** | **5/5** | post hoc — see the caveat |

**H5, stated so it can be killed.** The vision tower stamps its border patches — most
strongly the first one — with a direction in feature space. The stamp is not a big norm
(§16.6), it rides *inside the patch embedding*, and the language model's queries are
aligned with it (§16.6). That is why rotating the picture does nothing (the ViT re-stamps
the new border, which is the same border), why permuting the embeddings moves the peak
with the vector (the stamp travels with it), why no pixel statistic predicts it (§16.5),
and why it is strongest at the top-left (the ViT's own position embedding is most extremal
there, and it is also the first token of the run).

**H5 was not pre-registered.** It is what H1 and H3 turn into once the permutation arm is
read, so this run generated it and cannot also confirm it. Its five predictions are written
in the probe's `HYPOTHESES` table in falsifiable form so another model, or another
resolution, can fail them.

## 16.9 What this changes for the rest of the project

- The sentence in [peak-location-results.md](peak-location-results.md) generalises across
  imagery — but the word **"sink" does not survive §16.4**, and "background" is refuted
  outright rather than merely unsupported.
- **`frame` remains the right source set** for `sink_shift.py`'s edit: the border really
  does hold 2.6× its share in every image type, so the intervention is aimed at something
  real. What it is draining is a *vision-encoder positional stamp*, not a background prior.
- The **8.7% figure** in §16.1 says the broad arm has ~20x more to move than the two
  rewarded heads suggested, which strengthens the case for running `--scope all` in
  [inference-intervention.md §3](inference-intervention.md) before reading any null.
- **`prompt_swap` at +0.074** is the closest thing here to an intervention that *increases*
  ring enrichment, and it is a reminder the effect has a query-dependent component.

## 16.10 Caveats on the result

- **One model.** The cold start, on Qwen3-VL-8B. Stage 5 (2B/4B/32B, Qwen2.5-VL,
  InternVL3.5) is built and not run; H5 is a claim about vision encoders and is currently
  evidenced on one.
- **`MAX_IMAGE_SIDE = 512`**, so the grids are at most 16 patches on the long side and the
  ring is 23%–41% of the picture. The resolution ladder is the only handle on this and it
  does not reach past 512 — and it moves the number (`res256` costs 0.656), so resolution
  is not neutral.
- **Prefill, not generation.** `--tie-back` implements the check that licenses this and was
  not run; the reported cells' agreement with the generated-token readout is unmeasured.
- **The types are corpora.** Within-picture arms carry the causal weight; the type sweep
  says where the phenomenon is, not why.
- Six types are topped up from `set_a`/`set_b`, which the cold start never trained on. Any
  GRPO-trained checkpoint must be run with `--val-only`.
- The permutation destroys the model's answer. That is fine — the readout is where
  attention goes, not whether the answer is right — but it means `permute` says nothing
  about behaviour.

---

# 17. Base Qwen3-VL-8B, and the tokens the model wrote — 2026-09-08

Two changes: the **vanilla `Qwen/Qwen3-VL-8B-Instruct`** instead of the cold start, and the
attention from **generated** tokens as well as from the question's. 360 pictures, 30 per
type, one greedy answer each capped at 256 tokens, then one teacher-forced forward over
prompt ++ answer. That is the same construction `grpo_trainer_qwen3.py` uses to compute the
reward (`overlap_layer=22`, `overlap_heads=(28, 31)`, mean over heads — verified), so the
`generated` column is the reward's own view rather than a proxy for it. Selftest passed.
`outputs/sink_location/base_qwen3vl/report.txt`.

Pooled over the twelve types. Enrichment: the location's share of the picture's attention ÷
its share of the patches; `topleft`/`botright` are single patches against 1/N.

| tokens asking | heads | ring | one in | middle | top | bottom | left | right | top-left | bottom-right |
|---|---|---|---|---|---|---|---|---|---|---|
| question | all 1,152 | 1.53 | 0.81 | 0.71 | 2.48 | 1.26 | 2.18 | 1.47 | 12.9 | 3.14 |
| question | L22 h28,31 | 2.02 | 0.62 | 0.43 | 4.65 | 0.86 | 4.17 | 1.32 | 36.9 | 1.11 |
| **generated** | all 1,152 | **1.29** | 0.90 | 0.85 | 1.86 | 1.10 | 1.65 | 1.20 | 8.0 | 2.09 |
| **generated** | **L22 h28,31** | **1.61** | 0.83 | 0.61 | 2.68 | 1.02 | 2.09 | 1.74 | 14.7 | 1.34 |
| both | all 1,152 | 1.39 | 0.86 | 0.79 | 2.11 | 1.17 | 1.86 | 1.33 | 9.9 | 2.63 |
| both | L22 h28,31 | 1.73 | 0.76 | 0.56 | 3.29 | 0.96 | 2.76 | 1.57 | 21.6 | 1.21 |

## 17.1 The base model and the cold start are the same model here

Query tokens, the only column both runs have:

| | ring, all heads | ring, L22 h28,31 | top-left, all heads | top-left, L22 h28,31 |
|---|---|---|---|---|
| base Qwen3-VL-8B (n=30/type) | 1.53 | 2.02 | 12.9 | 36.9 |
| cold start (n=150/type) | 1.51 | 1.97 | 12.2 | 28.6 |

**The SFT did not create the border bias.** It is in the released model, and §16's whole
argument — twelve types, the rotation, the permutation — is about the base model too.

## 17.2 The effect is smaller on the tokens the model writes

Ring enrichment drops from 1.53 to **1.29** over all heads, and from 2.02 to **1.61** at the
trained pair. The top-left patch drops from 12.9 to 8.0, and from 36.9 to **14.7**. Roughly
two thirds the size, in both head sets and in every type.

Two consequences.

- **Every prefill-only number in §16 is an upper bound on what the reward saw**, not an
  estimate of it. The direction and the ordering hold, the magnitudes do not. §16.10's
  caveat about the missing tie-back is now settled in the direction that costs us: the
  proxy is biased upward by about a third.
- It **does not vanish**. Even the reward's exact readout — the two trained heads, the
  generated tokens, the teacher-forced pass — puts 1.61× on the border and 14.7× on the
  single top-left patch. What `--overlap_rect_frac` was scoring against really is a map
  with a lit corner in it.

## 17.3 The trained pair is not a typical pair

L22 h28,31 leans on the border **harder than the model's average head does**, on every
token set: 2.02 vs 1.53 on the question's tokens, 1.61 vs 1.29 on the generated ones, and
the top-left patch at 36.9 vs 12.9. Its interior is correspondingly starved — `middle` 0.43
against the model-wide 0.71.

So the pair the reward reads is a border-biased pair, chosen (for other reasons) from the
more border-biased end of the distribution. That is worth knowing whenever `mean_in` is
described as measuring "where the model looks": it measures where two unusually
edge-leaning heads look.

## 17.4 Still not a sink, more so

At the strongest cell anywhere in the base model the top column takes **51×** uniform
(cold start: 33×) but its across-query CV is **1.21** — still far above the pre-registered
0.5. The strongest cell is L0 h27 on 103 of 260 pictures. The picture's share of a row is
0.089, essentially identical to the cold start's 0.087.

## 17.5 Caveats specific to this run

- **30 pictures per type**, against 150 in §16. Enough for these effect sizes; the CIs are
  wider and per-type orderings should not be read closely.
- **Answers were capped at 256 tokens and the median hit the cap**, so `generated` means
  the first 256 tokens of an answer that was still going. The base model was given the
  cold start's `<think>` system prompt, which it was never trained to follow.
- The arms (rotation, permutation, zoom) were **not** re-run on the base model. §16's
  causal claims rest on the cold start; §17.1 is the reason to expect they carry over, not
  evidence that they do.


---

# 18. The interactive page — 2026-09-09

`docs/sink-location.html` is the whole result as one self-contained page: nine figures, a
table view behind every one of them, hover tooltips, and a selected dark mode. No CDN, no
fonts to fetch, no build step — it opens from a file:// URL on a machine with no network,
which is what the eval nodes and a laptop on a plane have in common.

```fish
python sink_location_html.py                       # -> docs/sink-location.html
node assets/check_sink_location_html.js docs/sink-location.html
```

It **reads the same npz and JSONL the report reads** and calls the report's own functions
with stdout swallowed, so no number on it is retyped and a figure cannot drift away from
§16–17. Regenerate it after any new run rather than editing the HTML.

The layout check exists because there is no browser on this cluster. It runs the page's own
drawing code against a DOM stub and measures what came out — every coordinate finite, every
mark inside its own viewBox, every label with room for its text, every figure non-empty.
That is most of what looking at a screenshot would have caught, and it caught two real
things: figure 3 was indexing location names the report exposes under different keys and
was emitting `NaN` for all six bars, and the negative bars were positive bars pushed left by
a transform, which put the rounded end at the baseline instead of at the data end.

The palette is the documented default, validated rather than eyeballed:
`node scripts/validate_palette.js "#2a78d6,#eb6834,#1baf7a" --mode light` and the dark
steps `"#3987e5,#d95926,#199e70" --mode dark` both pass every gate. Light-mode aqua sits
below 3:1 on the surface, so the relief rule applies — hence the direct labels on every bar
and a table view on every figure.
