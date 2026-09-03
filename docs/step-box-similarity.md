# Do the steps of one reasoning chain get different boxes out of DINO?

Tested 2026-09-03 on data already on disk. Tool: `step_box_similarity.py`.
Report: `outputs/step_box_similarity/mean_in/report.txt` (+ `report.json`).

## The question

The overlap reward runs Grounding-DINO once per observe step, on that step's own
sentence, and scores that step's attention map against the boxes that come back. That
only pays for itself if the boxes change from step to step. The claim under test is that
they mostly do not.

Nothing new had to be run. `overlap_probe.py --store-maps` already wrote, for every step
of every completion it generated, the DINO boxes (`boxes_kept`), the rasterised union
mask the reward scored (`mask_q`) and the attention map itself (`map_q`). Fourteen
model-by-run cells were re-read: cold start and eleven checkpoints on `val_natural`, plus
cold start and one checkpoint on `set_a`.

## Words used below

- **IoU** — overlap of two steps' union masks: patches in both, divided by patches in
  either. 1.0 means the same patches, 0.0 means no shared patch.
- **chance IoU** — what two *unrelated* masks of exactly those two sizes would score.
  Needed because these unions are enormous (median step covers 58% of the patch grid) and
  two big blobs overlap heavily no matter what is in them.
- **best IoU** — what those two sizes score when the smaller sits entirely inside the
  larger, i.e. as identical as masks of those sizes can be.
- **closeness** — `(IoU - chance) / (best - chance)`. 0 = unrelated once you know the
  sizes, 1 = as identical as the sizes permit.

Three pairings, each weighted by the same completions:

- **within** — two steps of the same chain. The hypothesis.
- **same-image** — two steps of two *different* chains about the same picture. Different
  sentences, same image.
- **diff-image** — a step's boxes drawn on another picture's grid. The floor.

## The answer

**Yes in the way that matters, and the reason is sharper than "the boxes look alike": a
step's own sentence buys nothing over any other sentence about the same picture.**

Cold start on `val_natural` — 30 images, 205 chains, 833 steps, 1651 within-chain pairs:

| pairing | IoU | chance | best | closeness | IoU ≥ 0.9 | bit-identical |
|---|---|---|---|---|---|---|
| within | 0.635 | 0.383 | 0.727 | **0.721** | 16.2% | 6.8% |
| same-image | 0.637 | 0.387 | 0.727 | **0.704** | 18.7% | 8.0% |
| diff-image | 0.449 | 0.386 | 0.700 | 0.235 | 0.4% | 0.0% |

Two steps of one chain land 72% of the way from "unrelated" to "as identical as their
sizes allow". Two steps of two different chains land at 70%. **The gap is nothing.** It
holds across all fourteen cells, and in ten of them the within-chain number is *lower*
than the same-image number:

```
within minus same-image, IoU, per cell:
 -0.003 -0.017 -0.013 +0.018 -0.022 -0.007 +0.017 -0.020 -0.167 -0.080 -0.007 -0.005 +0.023 +0.013
```

The floor is real, so the measure is not saturated: put the same boxes on a different
picture and closeness drops to 0.235. The mask is a property of the **image**. Which
sentence produced it is close to irrelevant.

The same holds before rasterisation, on the raw box lists (mean best-match IoU per box,
share of boxes with a ≥0.9 partner in the other step):

| | within | same-image | diff-image |
|---|---|---|---|
| best-match IoU | 0.480 | 0.485 | 0.185 |
| share with a ≥0.9 partner | 29.0% | 29.9% | 0.1% |

So this is not the patch grid blurring two different box sets into one mask. The boxes
themselves repeat.

### Does the sentence do anything at all?

A little. Splitting the within-chain pairs by how many content words the two steps share
(cold start, `val_natural`):

| shared content words | pairs | IoU | closeness |
|---|---|---|---|
| none | 646 | 0.564 | 0.606 |
| 0–10% | 387 | 0.639 | 0.739 |
| 10–25% | 417 | 0.677 | 0.788 |
| >25% | 201 | 0.764 | 0.914 |

Two steps with **no word in common** still reach closeness 0.606 — against 0.235 for a
different image. The words move the boxes; the image decides them.

### On the literal reading

"Almost identical" in the strict sense is a minority of chains, and the report says so:

- every step pair at IoU ≥ 0.9: **5.9%** of chains
- every step's mask bit-for-bit identical: **2.0%** of chains
- chain's mean IoU ≥ 0.9 / ≥ 0.75: **9.8%** / **30.7%** of chains
- median chain has 4 steps, mean IoU over its own pairs **0.634** [0.576, 0.688]

So the raw masks are similar rather than duplicated. The damaging version of the claim is
the controlled one above, not this one.

## What it costs the reward

Rescoring every step's attention map against every other step's mask, cold start,
`mean_in`, 833 steps:

- a step scores higher on its **own** mask than on another step's **52.6%** of the time.
  A coin flip is 50%.
- the own-mask advantage is **0.097** of the spread of scores across the chain's steps.
- rebuild the chain's reward as if DINO had been run **once per chain** (first step's
  boxes for every step): within-group rank correlation with the real reward **0.621**,
  and the group's best completion is still the best **53.6%** of the time.
- run it **once per image**, on a *different chain's* sentence: correlation **0.506**.

With `auroc` instead of `mean_in` the own-mask rate is **0.532** — which reproduces the
**0.534** `ID accuracy` in `docs/attention-intervention-plan.md`, measured independently
here, and is the check that this machinery agrees with what was already known. `auroc`
reacts more to which mask you use than `mean_in` does (once-per-chain correlation 0.330
vs 0.621): it is rank-based, so it feels the mask's *shape*, while `mean_in` mostly feels
its *size*.

## Caveats

- **Metric fidelity.** The stored maps are quantised to 1/255 of their own peak. Every
  step's metric was recomputed from the stored map and compared with the value the probe
  wrote. Largest disagreement: `mean_in` **0.0009**, `mean_in_v2` **0.027**, `auroc`
  **0.055**. `mean_in` is exact for this purpose; the `auroc` swap numbers carry real
  quantisation noise and, because noise pulls a correlation toward zero, its
  once-per-chain 0.330 is if anything understated.
- **Steps DINO could not ground are absent**, and so are steps whose union swallowed the
  whole grid — `_union_mask` returns `None` for both, the reward skipped them, and the
  probe stored no mask. Dropping the whole-grid ones works *against* the hypothesis:
  they are the most duplicated masks there are.
- **`max_union_area` was off** in every run analysed (`max_box_area` = 0.5). A union cap
  would remove the largest masks, which are the ones that overlap most by construction.
  The three pairings would all fall; whether the within-vs-same-image gap opens up is not
  answered here.
- **The same-image control is not the strongest one available.** Both texts are real
  sentences written by the same model about the same picture, and those are alike to
  begin with. `dino_text_sensitivity.py` closes that hole with texts that are
  deliberately wrong, scrambled or empty; it needs a GPU because it re-runs DINO.

## What follows

The per-step structure of the overlap reward is close to decorative. Grounding once per
image and reusing the mask would give an almost identical training signal for a fraction
of the DINO cost — which is a way of saying the reward was never scoring *this step's*
objects.

This is the missing piece under `docs/HANDOFF.md` result 3. That result showed the
attention cannot tell you which step it was written for (ID accuracy 0.534 against a
0.953 oracle). It was read as a fact about attention. Half of it is a fact about the
targets: the per-step masks are barely distinguishable from each other, so there was
little for the attention to be faithful *to*. The 0.953 oracle does not contradict this —
it scores each mask against itself, which stays maximal however alike the masks are.

It also bears on open question 4 in the handoff, the supervised route. An `L_attn`
contrastive across a chain's own steps needs the steps' targets to differ. At closeness
0.72 they mostly do not, so that loss would be near-degenerate on these masks as they
stand. Either the targets get sharper (a real union cap, phrase-level grounding rather
than whole-sentence grounding) or the contrast has to be drawn somewhere other than
between steps of one chain.

## Reproduce

```fish
set PY /home/uberger/scratch/miniconda3/envs/saliency_r1_qwen3_vllm/bin/python
$PY step_box_similarity.py \
    outputs/overlap_probe/20260809-021810-crossrun-val_natural-plus/probe_merged.json \
    outputs/overlap_probe/grad_spread/probe_merged.json \
    outputs/overlap_probe/20260805-011323-mean_in_v2_cp_1700/probe_merged.json \
    --verify --out-dir outputs/step_box_similarity/mean_in
```

CPU, about 2.5 minutes, no GPU and no DINO call. `--metric auroc` switches the mask-swap
section; `--models` restricts which checkpoints are read.
