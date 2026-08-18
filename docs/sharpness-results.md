# Sharpness vs grounding: is a concentrated map better than a well-placed one?

The hypothesis: the model may do better whenever its saliency map is *concentrated* --
on anything -- rather than when it is concentrated on the objects the step names. If so,
every grounding measurement in this repo has been reading concentration through a
box-shaped window, and the DINO boxes were never the operative variable.

Run 2026-08-18. Same 1,157 completions / 3,471 observe steps as every other probe on
this corpus (`outputs/intervene_probe/coldstart_setA_v2`, cold-start model, set_a,
accuracy 0.461), so every number here is directly comparable with
[probe-results.md](probe-results.md). Output in `outputs/sharpness/`.

**Harness validation.** The re-scan reproduces every published number on this corpus to
four decimals: L0H19 auroc −0.155, L1H4 mean_in_v2 +0.146, L22H28 v2 +0.046, inc34 auroc
+0.124 (held out +0.138), gxi auroc −0.097, glimpse auroc −0.064. The sharpness columns
are computed on the same maps in the same pass, so nothing but the added statistics
differs.

---

## The three blocks

| block | question | boxes needed |
|---|---|---|
| **DINO** | does the map sit on the objects the step names (`mean_in_v2`, `auroc`) | yes |
| **SHARP** | how peaked is the map (`cv`, `gini`, `nent`, `top1`, `top5`, `top20`, `sconc`) | **no** |
| **MASS** | how much total weight does the map put on the image at all | **no** |

Six of the seven SHARP columns are invariant under an arbitrary permutation of the
patches, so they are mathematically incapable of encoding a location. `sconc` is the
deliberate exception -- compact-somewhere, still never asking where. All seven are
oriented so that higher = sharper, and all are computed on the L1-normalised map, so
they are orthogonal to MASS by construction.

Each block is corrected with its own **max-|r| permutation threshold** (shuffle the
completion labels, take the largest |r| anywhere in the block, read the 95th
percentile). Bonferroni would badly over-correct 1,152 neighbouring attention heads;
this is exact whatever their mutual dependence, and it is what makes 8,064 sharpness
tests comparable with 2,304 DINO ones.

---

## 1. The raw correlations are large, and they are corpus composition

Ranked on odd `row_index`, re-scored on even, no covariates held:

| family | DINO | SHARP | MASS |
|---|---|---|---|
| heads | L0H19/auroc **−0.113** | L10H8/gini **−0.322** | L1H24 **−0.360** |
| rollout_wnorm | inc22/auroc +0.067 | inc19/nent **+0.302** | inc19 **−0.304** |
| grad | gxi/auroc **−0.103** | +0.058 | +0.046 |
| glimpse | −0.075 | −0.029 | −0.035 |

Sharpness beats grounding by 2.8x in the head family and by 4.5x in the rollout. It
would be a striking result if any of it survived a control, and almost none of it does.

**The carrier is which source corpus the question came from.** Accuracy on set_a runs
from 0.855 (aokvqa) to 0.054 (visual7w) across the six sources, which is a far larger
spread than any map statistic produces within one:

```
r(correct, ds:visual7w) −0.314    r(correct, ds:openimages) −0.287
r(correct, ds:gqa)      −0.204    r(correct, nsteps)         −0.196
r(correct, ntok)        +0.109    r(correct, log_npatch)     +0.019
```

Holding log(patch count), step token count, step count, DINO union area and dataset
dummies fixed collapses all three blocks -- inc19/nent +0.318 -> +0.120, L10H8/gini
−0.339 -> −0.056, L1H24 mass −0.399 -> −0.029 -- and in the head family it *flips the
sign*. The eye-catching −0.34 is not a fact about saliency maps.

---

## 2. Controlled, with a clean held-out split: grounding survives nowhere

`sharpness_report.py --residualize`: X and the label are residualised on the covariate
set first, so the permutation threshold and the odd/even split apply to the **partial**
correlation. Columns are then ranked on the odd half and re-scored on the even half
(n=573), where only one column per block is tested and the plain single-test threshold
|r| >= 0.082 applies.

| family | DINO | SHARP | MASS |
|---|---|---|---|
| heads | L16H11/auroc −0.028 | **L2H11/cv +0.177** | **L1H28 +0.163** |
| rollout_wnorm | inc17/auroc +0.062 | **inc19/nent +0.097** | **inc1 +0.089** |
| grad | gnorm/v2 −0.069 | gxi_ds/gini −0.004 | gnorm +0.059 |
| glimpse | glimpse/auroc −0.030 | glimpse/cv +0.021 | **glimpse +0.118** |

**DINO survives in 0 of 4 families. SHARP in 2. MASS in 3.** In the head family the
DINO block's best of 2,304 tests reaches p_fw 0.87 -- not a weak effect, no effect. Two
box-free statistics beat it, and one of them does not measure concentration at all.

`inc34`, the one column [probe-results.md](probe-results.md) §3 found surviving
multiplicity correction, is included in this: its raw +0.124 is +0.109 partialled, and
the odd-half pick from its own family lands at +0.062 on the even half. The earlier
result selected and scored on the same 1,157 cases.

---

## 3. But sharpness is narrow, and the unselected columns are null

The head family gives 1,152 chances. Selection is the whole risk, so what the family
does *as a whole* matters as much as its winner. Residualised, on `cv`:

```
                over thr95   median r    positive    per-layer mean
SHARP (cv)         2/1152     −0.015     456/1152    ~0.00 at L0-L12, −0.02..−0.05 at L13-L35
MASS              35/1152     +0.038     846/1152
DINO               0/2304     −0.011
```

So the simplest reading of the hypothesis -- *the model's attention is sharper when it
is right* -- is *false* on this corpus. The average head shows nothing, the median is
slightly negative, and every deep layer leans the wrong way. The unselected reference
columns agree: the mean over all 1,152 heads reads −0.044 to −0.054 on all seven
metrics, and the rollout's pre-registered `L35` and `inc35` readouts are inside noise.

What holds up is narrower and honest about it: **one head, L2H11, whose attention
concentration predicts correctness at +0.177 on data it was not chosen from.** The
20 heads ranked highest on the odd half keep their sign 18/20 times on the even half
(binomial p ~ 2e-4) while shrinking to 43% of their magnitude -- real structure,
heavily over-estimated by selection.

L2H11 is also not uniform across the corpus:

| L2H11/cv, residualised | aokvqa | gqa | openimages | visual7w | vsr |
|---|---|---|---|---|---|
| n | 400 | 392 | 128 | 149 | 73 |
| accuracy | 0.855 | 0.319 | 0.055 | 0.054 | 0.630 |
| r | −0.005 | **+0.322** | +0.104 | **−0.195** | **+0.540** |

Positive and large in the two corpora with usable accuracy variance, ~0 at the aokvqa
ceiling, negative in the near-floor visual7w. That pattern is consistent with a real
effect attenuated by range restriction, and equally consistent with corpus-specific
image statistics. Nothing here separates the two.

---

## 4. The cleanest result in the study is not about sharpness at all

**GLIMPSE has exactly one column.** No head to choose, no layer to choose, no metric
family -- so its MASS number carries no selection whatsoever, and it is **+0.118 on the
held-out half** against a 0.082 threshold. Its DINO auroc on the same completions is
−0.030 and its sharpness +0.021.

The same variable is the broadest effect in the head family (846/1,152 heads positive,
35 over the family-wise threshold) and survives in the rollout (+0.089). Across four
map families computed by four unrelated methods, the quantity that predicts being right
is **how much total weight the map puts on the image** -- not where the weight sits, and
not how peaked it is.

That is not a new variable here: [probe-results.md](probe-results.md) §3 already used
image mass as a covariate precisely because it was "the strongest correlate of
correctness anyone has measured here (+0.22..+0.29)". What is new is that it now stands
in a controlled comparison against both grounding and concentration, and wins.

---

## What this does and does not license

- It does **not** show that making maps sharper, or heavier on the image, makes the
  model right. Every number here is an association measured on the model's own chains.
  The one causal test run on this corpus -- forcing an observe step's attention onto the
  objects it names -- was a tight null ([probe-results.md](probe-results.md) §1, §4),
  and nothing here revisits that.
- It does show that **the box-derived grounding metrics have no controlled, held-out
  association with correctness on this corpus**, in any of the four map families, and
  that two box-free quantities do. A grounding reward built on these metrics is
  optimising a variable that does not predict the outcome it is meant to improve.
- The confirmation is still internal: an odd/even split of the same 1,157 cases. The
  image-disjoint `val_natural` draw has never been run for any of this. Sharpness needs
  no boxes, so it could be scored there without Grounding-DINO at all -- which makes it
  the cheapest confirmation available and the obvious next step.
- `visdrone` (n=15) is below the 30-completion floor and is dropped from every
  per-corpus table.

## Reproducing

```fish
bash launch_sharpness.sh --gpus 8            # ~40 min on 8 GPUs; glimpse is 28 of it
python sharpness_report.py --scan heads=outputs/sharpness/heads ... [--residualize]
```

Read `--residualize` first. Without it the report shows raw correlations, which on this
corpus are mostly a readout of which dataset the question came from.
