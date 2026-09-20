# Figure-1 search: 797 (model, picture) chains over fig1b-hrbench, fig1b-math, fig1b-natural

ours = `ours`; maps = glimpse, direct; tight referent = boxes within 80% of the step's best Grounding-DINO score, each under 35% of the image; the reward referent is the union the overlap reward would have scored (threshold 0.1, per-box cap 0.5).

## 1. Per step, inside the step's own referent

| model | map | referent | steps | median v2 | median AUROC | AUROC > 0.5 | peak inside | median border mass |
|---|---|---|---|---|---|---|---|---|
| ours | glimpse | tight | 2071 | 1.13 | 0.578 | 63% | 27% | 32% |
| ours | glimpse | reward | 2073 | 1.04 | 0.509 | 53% | 48% | 32% |
| ours | direct | tight | 2071 | 0.56 | 0.387 | 25% | 12% | 51% |
| ours | direct | reward | 2073 | 0.66 | 0.331 | 15% | 27% | 51% |
| coldstart | glimpse | tight | 2827 | 1.06 | 0.548 | 59% | 28% | 31% |
| coldstart | glimpse | reward | 2839 | 1.00 | 0.493 | 48% | 51% | 31% |
| coldstart | direct | tight | 2827 | 0.48 | 0.364 | 20% | 11% | 52% |
| coldstart | direct | reward | 2839 | 0.62 | 0.331 | 12% | 25% | 52% |

Chance is 1.00 for v2 and 0.500 for AUROC. Ours against base, on the tight referent (Mann-Whitney over steps):

- `glimpse`: P(a random ours step beats a random coldstart step) = 0.535, p = 3.5e-05
- `direct`: P(a random ours step beats a random coldstart step) = 0.542, p = 4e-07

## 1b. Per benchmark, and per whole chain

`clean` is the figure's actual claim: every grounded step of that chain above chance inside its own referent, over at least 3 scored steps and 2 disjoint places. It is harder for a model that writes SHORT chains, so read it next to the median-steps column rather than on its own.

| benchmark | model | chains | scored steps | median AUROC | AUROC > 0.5 | median steps | clean |
|---|---|---|---|---|---|---|---|
| algopuzzlevqa_mini | ours | 100 | 581 | 0.503 | 51% | 5 | 5 |
| algopuzzlevqa_mini | coldstart | 100 | 695 | 0.451 | 39% | 5 | 5 |
| hrbench4k_mini | ours | 92 | 161 | 0.781 | 86% | 1 | 3 |
| hrbench4k_mini | coldstart | 97 | 244 | 0.723 | 84% | 2 | 12 |
| hrbench8k_mini | ours | 83 | 115 | 0.785 | 90% | 1 | 3 |
| hrbench8k_mini | coldstart | 98 | 216 | 0.752 | 80% | 2 | 4 |
| mathvision_mini | ours | 98 | 432 | 0.566 | 62% | 3 | 5 |
| mathvision_mini | coldstart | 99 | 484 | 0.558 | 63% | 4 | 6 |
| omnispatial_mini | ours | 91 | 255 | 0.568 | 62% | 2 | 2 |
| omnispatial_mini | coldstart | 99 | 398 | 0.535 | 57% | 3 | 6 |
| p3_mini | ours | 80 | 133 | 0.629 | 71% | 1 | 2 |
| p3_mini | coldstart | 79 | 140 | 0.576 | 65% | 2 | 3 |
| pope_mini | ours | 90 | 130 | 0.586 | 72% | 1 | 0 |
| pope_mini | coldstart | 98 | 261 | 0.582 | 65% | 3 | 7 |
| wemath_mini | ours | 89 | 264 | 0.563 | 63% | 2 | 6 |
| wemath_mini | coldstart | 97 | 389 | 0.529 | 55% | 3 | 4 |

## 2. Crossover rate over disjoint within-chain step pairs

A pair qualifies when the two tight referents have grid IoU <= 0.05 and each covers <= 25% of the patch grid. It *moves the right way* when min over the two steps of (v2 in its own region - v2 in the other) is positive: one step firing everywhere cannot carry the pair.

| map | model | pairs | move the right way | rate |
|---|---|---|---|---|
| glimpse | ours | 563 | 218 | 39% |
| glimpse | coldstart | 701 | 289 | 41% |
| direct | ours | 563 | 59 | 10% |
| direct | coldstart | 701 | 82 | 12% |

- `glimpse`: ours 39% vs coldstart 41%, p = 0.37
- `direct`: ours 10% vs coldstart 12%, p = 0.49

## 3. Answers on the same pictures

One row per (model, picture), graded on the scan's own greedy completion -- not the benchmark's, which runs at a longer token budget and full image resolution. A chain with no end-of-turn token hit the cap and is counted as truncated rather than wrong.

| model | pictures | soft-correct | strict-correct | hit the token cap | `<think>` format |
|---|---|---|---|---|---|
| ours | 750 | 57% | 54% | 24% | 76% |
| coldstart | 794 | 53% | 52% | 27% | 73% |

- vs `coldstart` over 747 shared pictures: ours only 75 (47 with the baseline's chain complete), coldstart only 38.

