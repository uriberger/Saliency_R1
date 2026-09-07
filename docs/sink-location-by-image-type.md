# Is the sink in the outer ring, or is the outer ring just the background? — plan, 2026-09-07

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
