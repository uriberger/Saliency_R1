# Where `mean_in`'s denominator sits, and how flat the map got — 2026-09-03

`mean_in = mean_U(m) / max(m)`. Everything measured so far has been about the
numerator. This page is about `max(m)` — the single patch the whole score is divided
by — and about the variance of the map around it, for the cold start and for the runs
trained on `mean_in` and on `mean_in_v2`.

Nothing new was computed on a GPU. `overlap_probe.py --store-maps` already persists,
per observe step, the map quantised to its own peak (`map_q`), the absolute peak
(`map_max`), the DINO union raster (`mask_q`) and the grid, so a patch's attention
weight is exactly `q/255 * map_max`. `peak_location_probe.py` reads those bytes back.

    python peak_location_probe.py --probe outputs/overlap_probe/20260809-021810-crossrun-val_natural-plus --selftest

Corpus: `val_natural`, 30 images x 8 generations per model, attention map at L22 heads
28/31 (token_reduction mean) — the map the reward was trained on. 456–3,336 grounded
observe steps per model. Full output in
`outputs/peak_location/crossrun_val_natural.txt`. Every CI is a 2,000-sample bootstrap
over **images** (the cluster), paired across models by common random numbers.
`--selftest` re-derives the stored `mean_in_raw` from the decode: median error 9.3e-05
against a value of ~0.038, i.e. pure uint8 rounding.

## 1. At the cold start the peak is OUTSIDE the box, far below chance

| | |
|---|---|
| P(argmax patch inside the DINO union) | **0.243** [0.174, 0.315] |
| chance = mean union area | 0.565 |
| lift | **−0.323** [−0.395, −0.251] |

Not a box-size artefact: within every union-area bin the peak is below chance, and by
the largest margin in the middle of the range (union 0.4–0.6: 0.19 observed vs 0.50
chance, 339 steps). The whole head of the distribution is out there too — the top 5% of
patches are inside 0.377 of the time against the same 0.565 chance.

**The peak is an edge sink, not an object.** 82% of peaks sit on the outer ring of the
patch grid, 55% on the top row, 27% in a literal corner. DINO boxes reach the ring far
less than the interior — the union covers 37% of ring patches against 64% of interior
ones — so "the peak is outside the union" is in large part "the peak is on a border
patch a box never covers". It is not *only* that: conditioning on the ring the peak is
still inside only 0.182 of the time against a 0.374 chance, and conditioning on the
interior 0.519 against 0.641. Anti-grounded in both halves, worse on the ring.

This is what `max(m)` is: a near-constant of the model, pinned to the frame edge, not a
property of the step being scored. `mean_in` is therefore closer to "in-box mass ÷ a
sink" than to "fraction of the map's peak that is on the object".

## 2–3. Both metrics move the peak inside — but only when the run collapses

Change in P(peak inside), image-paired against the cold start:

| run | steps/comp | P(peak in) | Δ vs cold start | Δ lift | benchmark |
|---|---|---|---|---|---|
| cold start | 3.7 | 0.243 | — | — | — |
| `mean_in` set_a cp1000 | 3.4 | 0.233 | −0.010 [−0.054, 0.037] | −0.015 | — |
| `mean_in` set_a cp2000 | 13.2 | **0.669** | **+0.426** [0.335, 0.514] | +0.418 | −11.5% |
| `mean_in` 8k (final) | 2.1 | 0.299 | +0.057 [−0.003, 0.121] | +0.017 | **+3.6%** |
| `mean_in_v2` set_a cp1000 | 3.8 | 0.285 | +0.043 [0.002, 0.087] | +0.030 | — |
| `mean_in_v2` set_a cp1500 | 5.6 | 0.411 | +0.168 [0.119, 0.220] | +0.153 | — |
| `mean_in_v2` set_a cp1700 | 14.1 | 0.595 | +0.352 [0.239, 0.468] | +0.379 | greedy val 0.55→0.17 |
| `auroc` set_a cp2500 | 1.1 | 0.711 | +0.469 [0.307, 0.619] | +0.534 | −8.5% |

Read the middle column against the last one and the pattern is not "metric X moves the
peak". It is **the peak moves inside exactly when the policy collapses into
background-phrase spam**, under all three metrics including `auroc`, and it barely
moves in every checkpoint that is still healthy — `mean_in` cp1000 (−0.010),
`mean_in` 8k (+0.057, CI touching 0), `mean_in_v2` cp1000 (+0.043).

The mechanism is visible in the border columns. For `mean_in` cp2000 the *total* union
area is unchanged (0.573 vs 0.565) but its **placement** moved: union coverage of the
ring 0.374 → 0.527 while coverage of the interior *fell* 0.641 → 0.592. The collapsed
policy writes sentences whose DINO boxes reach the frame edge, and the edge is where
the sink already was. `mean_in_v2` cp1700 does the same thing (ring 0.496, interior
0.557). So at matched training pressure the two metrics answer question 2 and question
3 the same way, and the answer is about the text, not the attention.

## 4. Flatness: both flatten the map, and `mean_in` flattens it more per step of damage

Scale-free (cv = sd/mean, so attending to the image harder cannot move it), as a ratio
to the cold start, image-paired. Below 1 = flatter.

| run | cv overall | cv inside | cv outside |
|---|---|---|---|
| `mean_in` set_a cp1000 | 0.966 [0.938, 0.995] | 0.980 [0.945, 1.022] | 0.968 [0.943, 0.996] |
| `mean_in` set_a cp2000 | 0.851 [0.801, 0.902] | 0.885 [0.834, 0.938] | **0.756** [0.689, 0.824] |
| `mean_in` 8k (final) | **0.859** [0.811, 0.909] | 0.891 [0.848, 0.939] | 0.867 [0.814, 0.918] |
| `mean_in_v2` set_a cp1000 | 0.994 [0.974, 1.014] | 1.006 [0.980, 1.033] | 0.983 [0.958, 1.005] |
| `mean_in_v2` set_a cp1500 | 0.943 [0.922, 0.963] | 0.958 [0.925, 0.991] | 0.900 [0.867, 0.935] |
| `mean_in_v2` set_a cp1700 | 0.878 [0.835, 0.921] | 0.926 [0.868, 0.984] | 0.843 [0.771, 0.922] |
| `auroc` set_a cp1000/2000/2500 | 1.02 / 1.04 / 1.01 | 1.01 / 1.06 / 1.10 | 1.05 / 1.09 / 0.96 |

Three things it says:

- **Every `mean_in` and `mean_in_v2` checkpoint flattens; every `auroc` checkpoint does
  not** (and cp2500 sharpens *inside* the box, 1.095). The flattening is a property of
  the two `/max` and `/mean` metrics, not of GRPO.
- **Outside flattens more than inside** in every case. The reward has no term for the
  outside; that is the box-blind channel, below.
- **`mean_in` gets there sooner.** At the matched step 1000 — the weights 0.4 and 0.033
  were calibrated to equal within-group reward sd — `mean_in` is at 0.966 overall and
  `mean_in_v2` at 0.994. The healthy 8k `mean_in` run reaches 0.859 while barely moving
  the peak or the step count.

In **absolute** attention units the answer is the opposite and it is not a
contradiction: sd goes *up* almost everywhere (`mean_in` 8k sd_all x1.96, cp2000
x3.15), because these runs put much more softmax mass on the image (0.00394 → 0.00894
for 8k, → 0.01376 for cp2000) and a bigger map has bigger spread. The shape flattens
while the scale grows. The one place absolute spread falls is `mean_in_v2` cp1700
outside the box (x0.838 [0.732, 0.974]) — that run raises inside-box spread (x1.83) and
suppresses outside-box spread, which is exactly what a mean-normalised metric pays for.

## What ties it together: `mean_in = mean_in_v2 × flatness`, exactly

Step by step, `mean_U(m)/max(m) = [mean_U(m)/mean(m)] · [mean(m)/max(m)]`. The second
factor is `flatness`, the mask-free statistic from
[next-reward-experiments.md](next-reward-experiments.md) — it never sees a box. **The
two metrics differ by exactly that one box-blind factor**, and in logs the split is
additive:

| run | mean_in | mean_in_v2 | flatness | Δlog mean_in | = Δlog v2 | + Δlog flat | flat share |
|---|---|---|---|---|---|---|---|
| cold start | 0.0381 | 0.736 | 0.0502 | — | — | — | — |
| `mean_in` set_a cp1000 | 0.0398 | 0.747 | 0.0520 | 0.064 | 0.023 | 0.041 | 0.64 |
| `mean_in` set_a cp2000 | 0.0699 | 1.188 | 0.0610 | 0.635 | 0.456 | 0.179 | 0.28 |
| **`mean_in` 8k (final)** | 0.0490 | 0.812 | 0.0594 | 0.299 | 0.118 | **0.181** | **0.61** |
| `mean_in_v2` set_a cp1000 | 0.0406 | 0.772 | 0.0505 | 0.052 | 0.050 | 0.003 | 0.06 |
| `mean_in_v2` set_a cp1700 | 0.0649 | 1.126 | 0.0582 | 0.548 | 0.421 | 0.126 | 0.23 |
| `auroc` set_a cp2500 | 0.0525 | 1.173 | 0.0462 | 0.410 | 0.475 | −0.064 | −0.16 |

The 8k `mean_in` run — the only run in this table that *helped* the benchmark — took
**61% of its reward gain through the box-blind factor**: `flatness` 0.0502 → 0.0594,
+18%, independently reproducing the number in
[next-reward-experiments.md](next-reward-experiments.md). It moved the peak hardly at
all and it moved the box-aware factor by 0.118. The runs that took most of their gain
through the box-aware factor (`mean_in` cp2000 at 0.456, `mean_in_v2` cp1700 at 0.421,
`auroc` cp2500 at 0.475) are the three that collapsed.

`mean_in_v2` is `mean_in` with that channel removed by construction: its numerator and
denominator are means of the same map, so it is invariant to the map's shape in a way
`mean_in` is not, and at step 1000 it had moved `flatness` by 0.003 against `mean_in`'s
0.041. On this corpus, the difference between the two metrics *is* the flattening
channel, and the flattening channel is where the run that worked spent its gradient.

That is a mechanism, not a proof: nothing here shows flattening *causes* the benchmark
gain. It shows the healthy run got its reward from the box-blind half of `mean_in` and
the metric that deletes that half did not gain. `--maskfree flatness` and
`--maskfree mass` are the experiment that can settle it.

## Caveats

- **30 images.** Every model saw the same 30 `val_natural` prompts, and the CIs are
  clustered on them, but 30 clusters is 30 clusters.
- **Each model generated its own text**, so the steps and the boxes differ between
  models. The union-area bins control for box *size*, nothing controls for box
  *content*, and §2–3 argues the peak result is driven by exactly that. A fixed-text
  control — teacher-force one policy's chains through all three models and re-read the
  attention — would separate "the attention moved" from "the sentences moved". It has
  not been run.
- **Steps per completion vary 2.1x–14.1x** across models, so per-step means re-weight
  the corpus toward the chattiest policy. The per-completion table at the bottom of the
  report reproduces every sign and every ordering.
- **The matched 8k pair is incomplete.** `mean_in` 8k is in this probe;
  `wov0.033_..._saliency_r1_8k_mean_in_v2` (trained, benched, 3,990 steps) was never
  probed, so the only head-to-head at equal pressure is on set_a, where *both* runs
  collapse. One 8-GPU probe run (~10 min, the cost of
  `outputs/overlap_probe/20260809-020113-newmodel`) would close it.
