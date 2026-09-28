# Figure 1: a chain that looks at a different object at every step

The ask: one picture, one question that cannot be answered from a single place, a chain
whose steps ground to **non-overlapping** Grounding-DINO boxes, our attention landing on
each box at the step that names it, and base Qwen3-VL-8B-Instruct doing worse on both the
answer and the maps.

Three of those four hold. The one that does not is the one the reward was built on.

Companion pages: [saliency-viz-howto](saliency-viz-howto.md) (how the maps are drawn),
[saliency-maps](saliency-maps.md) (what the six maps *are*),
[probe-results](probe-results.md) (what they score). The earlier, different search --
ours against `saliency-r1` on a *shared* referent -- is `fig1_step_referent.py`, and it
came back empty; this page is not that one.

---

## 0. The short version

| the ask | answer |
|---|---|
| a question needing several places, with disjoint per-step boxes | **yes**, 266 such step pairs over 805 chains |
| our attention follows them, step by step | **yes, on GLIMPSE**: 73% of pairs move the right way, against base's 59% (p = 1.3e-5) |
| ...on the two rewarded heads (`direct`, L22 h28/31) | **no.** 17% of pairs, and the median step scores AUROC **0.43** inside its own referent -- below chance |
| base answers worse | **not in general.** Over 667 shared pictures we win 48 and lose 63. It is true on the panel below, and that is an instance, not a rate |

The panel to use is `outputs/fig1-multistep/panel-motorcycle/` (all four hold) or
`panel-clevr/` (the clearest picture of the behaviour, on a question where base answers
correctly too). Both are drawn with **GLIMPSE**, which is the only map here that is above
chance at all.

---

## 1. Why the per-step box had to be redefined

At the reward's own detector settings -- Grounding-DINO over the step's **whole sentence**
at `box_threshold` 0.1, per-box area cap 0.5 -- a step's referent is a median **52% of the
patch grid** built out of ~14 boxes. In the 20-sample `sviz-3models` run, **0 of 59**
within-chain step pairs came out below IoU 0.1. There is nothing to find at those
settings: "the step's referent" is not a place, it is most of the picture.

So each step carries two referents and both are reported:

| | what it is | what it is for |
|---|---|---|
| `reward` | every box above threshold, unioned, per-box cap applied | the training instrument. Quoted so a caption can say what the reward saw |
| `tight` | boxes within 80% of that step's **best** DINO score, each under 35% of the image | a region that is one object. A **figure** definition, chosen to be drawable |

The margin rather than the single top box is deliberate: the detector routinely returns an
object and a part of it ("the man", "his jacket") at nearly the same score, and keeping
only the first makes the region an arbitrary half of the same thing.

Under `tight`, 4,183 of 4,300 steps ground to something, the median referent is a few per
cent of the grid, and disjoint pairs are common. **None of this changes the reward** --
nothing here is retrained or rescored; it changes what the picture is drawn around.

## 2. The test: a crossover, not an overlap

An overlap number cannot say "it moved". For two steps `i, j` whose tight referents
`R_i, R_j` are disjoint:

```
margin = min( v2(map_i, R_i) - v2(map_i, R_j),
              v2(map_j, R_j) - v2(map_j, R_i) )
```

`v2` is `overlap_rewards._mean_in_v2`, chance = 1.0. The **min** is what makes it a
statement about both steps: a map that fires everywhere scores high in its own region and
in the other one, and contributes 0.

Every other model is scored on the *same two regions*, maximised over **all** of its own
step pairs -- the best showing it can make on that picture, rather than whichever of its
steps happens to share an index.

## 3. What came out, over 805 chains

Two held-out validation sets (`val_natural` + `val_e_natural`, 512 pictures, both
question-id-disjoint from the `saliency_r1_8k` the checkpoint trained on) and the 300
documents of the natural mini-benchmarks (`mmstar_mini`, `realworldqa_mini`,
`mmerealworld_mini`), scanned with `--methods glimpse,direct` for
`overlap-8k` (= `overlap__wov0.4_2head_trmean`) and for base Qwen3-VL-8B-Instruct.

### Per step, inside the step's own referent

| model | map | referent | steps | median v2 | median AUROC | AUROC > 0.5 | peak inside | median border mass |
|---|---|---|---|---|---|---|---|---|
| ours | glimpse | tight | 1264 | 1.54 | 0.681 | 82% | 44% | 30% |
| ours | glimpse | reward | 1232 | 1.11 | 0.577 | 69% | 67% | 30% |
| ours | direct | tight | 1264 | **0.61** | **0.429** | 32% | 13% | 47% |
| ours | direct | reward | 1232 | 0.79 | 0.405 | 22% | 34% | 47% |
| base | glimpse | tight | 2919 | 1.46 | 0.662 | 79% | 34% | 30% |
| base | glimpse | reward | 2914 | 1.13 | 0.579 | 69% | 55% | 30% |
| base | direct | tight | 2919 | 0.34 | 0.392 | 24% | 8% | 46% |
| base | direct | reward | 2914 | 0.60 | 0.378 | 17% | 23% | 46% |

### Crossover rate

| map | model | pairs | move the right way | rate | |
|---|---|---|---|---|---|
| glimpse | ours | 266 | 193 | **73%** | ours vs base p = 1.3e-5 |
| glimpse | base | 2609 | 1534 | 59% | |
| direct | ours | 266 | 46 | **17%** | p = 0.065 |
| direct | base | 2609 | 345 | 13% | |

### Answers, on the scan's own completions

| model | pictures | soft-correct | strict-correct | hit the token cap | `<think>` format |
|---|---|---|---|---|---|
| ours | 749 | 53% | 52% | 2% | 98% |
| base | 723 | 52% | 27% | 5% | 0% |

Over 667 shared pictures ours is right and base wrong 48 times; base is right and ours
wrong **63** times. The strict column is not a result: it is the trainer's exact-match
rule meeting a model that answers in prose.

These are the scan's answers at `--max-new-tokens 768` and `prepare_image`'s 512 px cap,
**not** the benchmark's, which runs at 4,096 tokens and full resolution. The difference is
not academic: on `mmstar_mini` doc 7 base runs past the cap here and never emits an option
letter, while `outputs/bench_baselines/qwen3-vl-8b-instruct` has it scoring 1.0. Any
"the baseline got it wrong" claim has to be checked against a complete chain first, which
is why the report counts truncation separately.

## 4. The two rewarded heads do not do this

`direct` is the map the overlap reward paid for: L22, heads 28 and 31, trimmed mean. Inside
the step's own tight referent it sits at **AUROC 0.429** with **47% of its mass on the
outer one-patch ring**, and the crossover works on **17%** of pairs -- i.e. 83% of disjoint
step pairs move the *wrong* way on it.

That is not a surprise and it is not new here: it is
[trained-heads-lean-on-the-border] and the anti-localisation
`fig1_step_referent.py` measured against the answer box, now measured per step against the
step's own object, on 4,183 steps instead of 188. Training did move the number -- ours
beats base on `direct` by more than it does on `glimpse` (Mann-Whitney AUC 0.568,
p = 2.7e-12, against 0.524, p = 0.013) -- but it moved it from *far* below chance to
*below* chance.

**So a Figure 1 cannot be drawn on the two heads.** It can be drawn on GLIMPSE, which the
reward never touched.

## 5. The panels

Each directory holds `panel.png` (labelled), `panel.html` (the chains and the numbers) and
`parts/*.png` (the same overlays, uncaptioned, for LaTeX).

**`panel-motorcycle/`** -- `val_e_natural` row 167, openimages. *"What the walking man
ride?"*, gold `motorcycle`. Our step 1 grounds to the motorcycle (18% of the grid), step 3
to the man on the left (7%), IoU 0.00. GLIMPSE: 2.30 in its own region against 0.24 in the
other, then 3.32 against 1.05 -- margin **+2.06**, where base's best pair of its own
manages +0.55. Ours answers `Motorcycle`; base answers *"The walking man is not riding
anything."*

This is the only picture in 805 where all four conditions hold at once. The caveat to
carry: base's answer is arguably the better description of the scene -- the man walking is
holding a helmet and a woman is on the bike -- and it is marked wrong because the
dataset's gold answers the question's presupposition. A reviewer can make that objection.

**`panel-clevr/`** -- `mmstar_mini` doc 7, a CLEVR counting question (*"Subtract all large
yellow matte cubes. Subtract all metal things. How many objects are left?"*). Our chain
enumerates the four objects one per step, and GLIMPSE lands on each in turn:

| step | region | v2 in it | AUROC | referent area |
|---|---|---|---|---|
| 2 "Small brown metallic cylinder" | brown cylinder | 4.79 | 0.99 | 1.3% |
| 3 "Small red metallic sphere" | red sphere | **11.87** | **1.00** | 1.3% |
| 4 "Small green matte sphere" | green sphere | 3.56 | 0.95 | 2.5% |
| 5 "Large blue matte cube" | blue cube | 4.10 | 0.92 | 7.5% |

It is the best *picture* of the behaviour by a distance. It is **not** a win over base,
and it is worth being precise about why, because the obvious caption is wrong.

Base writes 16 observe steps on this picture, seven of them `-` bullets, and **every one
of the seven peaks on the object its own sentence names** (`figure-clevr-base-alpha/`):

| base step | glimpse v2 in its own referent | AUROC |
|---|---|---|
| 2 "- There is a cyan cube, but it is not yellow." | 3.78 | 0.97 |
| 4 "- The gold cylinder appears to be made of metal" | 2.20 (red sphere 2.06 -- it splits) | 0.90 |
| 5 "- The red sphere also appears to be made of metal" | **6.71** | **1.00** |
| 6 "- The green sphere appears to be matte" | 3.77 | 0.94 |
| 7 "- The cyan cube appears to be shiny (metallic)." | 2.93 | 0.89 |

So the baseline's chain is grounded here too. What it gets wrong is the **judgement it
makes while looking at the right thing**: the cube is matte, base's step 7 lands squarely
on it and calls it *shiny (metallic)*, subtracts three objects instead of two, arrives at
"1", finds 1 is not an option, and loops -- "But 1 is not an option" six times, two
four-sentence blocks repeated verbatim -- until `--max-new-tokens 768` cuts it off with no
option letter. At the benchmark's own 4,096 tokens it recovers and scores 1.0
(`outputs/bench_baselines/qwen3-vl-8b-instruct`).

Caption it as *enumerating versus rambling*, or as *where you look is not what you
conclude*. Not as *grounded versus ungrounded*, which this picture does not show.

Redrawn 2026-09-22 with `--upsample map` and `--overlay-mode alpha`. The four objects are
now visible under the heat, where the old 0.5 blend left a grey wash, and the muddy
purples the RGB resize invented are gone. Nothing else moved: every v2 in the table above
and every step the script picked for base is the same number it was, because none of the
three knobs reaches the scoring path. No blur -- see "Making it legible" for why 8x10 is
the wrong grid for one.

**`panel-dogbed/`** -- `val_natural` row 234, *"Where is on the dog bed?"*, gold `cat`.
The widest margin gap on an unambiguous question (+1.24 against base's **-0.64**), but
both models answer correctly and region B is scene furniture rather than something the
question needs.

## 5b. Eight more benchmarks, against the cold start

A second search, 2026-09-20, on the eight benchmarks asked for -- AlgoPuzzleVQA, POPE,
HR-Bench 4K and 8K, OmniSpatial, SalBench P3, MathVision, WeMath -- 100 documents each,
`overlap-8k` against the **cold-start SFT it was trained from** rather than against
vanilla Qwen3-VL. That is the ablation that isolates the reward: same weights, same
prompt format, with and without the overlap GRPO. Numbers in
[fig1-benchmarks-numbers.md](fig1-benchmarks-numbers.md); the scans are
`outputs/saliency_viz/fig1b-{natural,math,hrbench}` and the search
`outputs/fig1-multistep/bench_b.json`.

HR-Bench is scanned at `--max-image-side 1024` rather than the training 512, because a
benchmark whose question is "what is written above that doorway in a 4K frame" is not
being asked at 512 px. GLIMPSE's cost grows with the square of the token count, so that
arm also runs `--glimpse-layer-frac 0.6 --max-steps 6`.

**Per step, inside its own referent, `glimpse`:** ours median AUROC **0.578** against the
cold start's 0.548, 63% vs 59% of steps above chance, over 2,071 and 2,827 steps
(Mann-Whitney p = 3.5e-5). Ours is ahead on 7 of the 8 benchmarks; MathVision is the tie.
Two of them are much better than that average and two are at chance:

| benchmark | ours median AUROC | cold start | ours above chance |
|---|---|---|---|
| hrbench8k | **0.785** | 0.752 | 90% |
| hrbench4k | **0.781** | 0.723 | 86% |
| p3 (SalBench) | 0.629 | 0.576 | 71% |
| pope | 0.586 | 0.582 | 72% |
| omnispatial | 0.568 | 0.535 | 62% |
| mathvision | 0.566 | 0.558 | 62% |
| wemath | 0.563 | 0.529 | 63% |
| algopuzzlevqa | **0.503** | 0.451 | 51% |

HR-Bench is where the reward shows up most clearly, which is the one place it should:
high-resolution photographs where the answer is a small object in a large scene. On
AlgoPuzzleVQA our model is at chance and the cold start is *below* it -- these are
rendered puzzle boards, and Grounding-DINO has no referent to find in "the box is in the
center at (3,3)", so neither model's number there means much.

**Answers.** Unlike the vanilla comparison, ours is ahead: 57% vs 53% soft-correct, and
over 747 shared pictures ours is right where the cold start is wrong 75 times against 38
the other way.

**Whole chains.** Requiring every grounded step of a chain to be above chance, over at
least 2 steps and 2 disjoint places: **50 of 723** pictures for ours; on 19 of those the
cold start has at least one below-chance step; on exactly **one** it also answers wrong.

**`outputs/fig1-multistep/figB-hrb-count/`** is that one, and it is the cleanest example
this whole investigation has produced. It is drawn the way every other heatmap in this
repo is -- jet blended at 0.5, so everything unattended goes blue -- with the input
picture first in each model's block. `figB-hrb-count-alpha/` is the same figure with
`--overlay-mode alpha`, which keeps the photograph's own colours where nothing fires;
useful for checking *what* is under a hotspot, wrong for a figure that has to be read
next to the rest of the paper.

HR-Bench 4K, *"How many people are there in the image?"*, gold **C. Two**. A tram
interior: a man in a wide-brimmed hat fills the frame, and a second person is barely
visible through the window behind him.

| | step | referent | AUROC | v2 | area |
|---|---|---|---|---|---|
| **ours** | 0 "a person wearing a wide-brimmed hat and a tan jacket" | the man | 0.684 | 2.45 | 8.2% |
| **ours** | 1 "another person partially visible in the background, sitting near a window" | the second person | 0.703 | 3.63 | 4.4% |
| cold start | 0 "there's a person sitting inside a vehicle" | a blob | 0.559 | 1.17 | **33.4%** |
| cold start | 1 "There are no other people visible in the image." | — | **0.437** | 0.89 | 37.8% |

Ours answers `C. Two`. The cold start answers `D. One`. Its second step asserts there is
nobody else while its attention never leaves the foreground, and its referents cover a
third of the picture each where ours are 4-8%.

### Making it legible

A 32x32 glimpse map painted over a 1024px photograph is speckle: the regions are there,
but they are scattered single patches and the eye cannot assemble them.
`figB-hrb-count-smooth/` is the same figure with `--smooth 1.0`, a Gaussian on the patch
grid with sigma in patches, and it is the one to put in the paper -- the hat becomes one
blob, the second person becomes one blob, and the cold start's diffuse wash over the
foreground becomes visibly diffuse rather than merely busy. Above about 1.5 the regions
bleed into each other and the quiet background stops being quiet.

`--upsample map`, now the default, is the other half. Colouring the 32x32 grid and then
resizing the *RGB* -- which `saliency_viz.py` still does -- interpolates along a straight
line between two colours of a ramp that is not straight, so a red patch beside a blue one
yields muddy purples that jet does not contain and that read as a mid value which is not
there. Interpolating the scalar field and colouring afterwards costs nothing and removes
them. `--upsample rgb` reproduces the old figures.

**Sigma is in patches, and the patch grid is not the same size twice.** `--smooth 1.0` is
3.1% of the width on the 32x32 grid these benchmark panels have (1024px images), and 10.0%
on the 8x10 grid a 320x240 picture gets -- three times the blur, on a grid where a single
referent is already one patch out of eighty. There is no speckle to merge at that size:
the bicubic in `--upsample map` is the whole of the fix, and `--smooth 0.4` on
`panel-clevr/` is not distinguishable from `--smooth 0` while 1.0 melts four objects into
one blob apiece. Read the grid off `maps.npz` before carrying a sigma from one figure to
another; do not treat 1.0 as a default.

`--overlay-mode alpha` is the third knob and the one a small, low-contrast picture needs.
The default `blend` tints every pixel by `--alpha`, which on CLEVR's grey background hides
the four objects the figure is entirely about; `alpha` paints the heat in proportion to
the map, so the quiet parts stay the photograph and the object under the hot blob is still
identifiable. `figure-clevr-2row-alpha/` and `figB-hrb-count-smooth-alpha/` are the same
comparison on the other two figures.

**The blur does not belong in a number.** It is a Gaussian smoother applied to the very
statistic being scored, and it flatters us in both directions -- on the four panels above
AUROC goes 0.684 -> 0.858, 0.703 -> 0.909, 0.559 -> 0.668 and 0.437 -> 0.379 at sigma 1.0.
That is a real signal-to-noise statement (a region's attention is better estimated by a
local average than by one patch) and it is not the statement the table makes. Every AUROC,
`mean_in` and v2 in this document is `fig1_multistep.py` on the raw grid; the table above
stays as it is, and the smoothed picture is a rendering of it, not a second measurement.

Two more worth keeping, both with the cold start failing on attention but matching on the
answer: `figB-wemath/` (WeMath, a parallelogram diagram -- five consecutive steps at
AUROC 0.85-0.92, each on the dimension label its sentence names: 30 cm, then 14 cm, then
the 20 cm base; the cold start manages 3 of 4 with a low of 0.48) and `figB-hrb-flag/`
(HR-Bench 4K, the American flag located at AUROC 0.95 on a region that is **0.6%** of the
grid). `figB-hrb8k-chart/` is a fourth, and it is the one to skip: a 30-bar chart at
AUROC 0.81-0.86 on 1-2% regions is a real result and an unreadable picture.

All four are `--model ours --model coldstart` on one sheet, so the two chains sit above
each other on the same picture. They are laid out per model rather than in a shared grid
because the chains are different lengths and a grid would imply step k of one is step k
of the other.

**What did not reproduce.** The crossover rate, which separated ours from vanilla on
natural validation images (73% vs 59%), is flat here: 39% vs 41%, p = 0.37. These eight
benchmarks are mostly diagrams, puzzle boards and synthetic fields, where the per-step
referent is not a place in the sense the crossover test needs. The per-step AUROC, which
does not depend on two referents being disjoint, still separates them.

And `direct` is unchanged: median AUROC 0.387 / 0.364, 51% of its mass on the border ring.

## 5c. Playing one chain

`fig1_steps_video.py` animates what §5's panels lay out side by side: frame 0 is the
picture and the question, and each frame after it holds one step's attention over the
picture while that step's own sentence lights up in the chain on the right. It renders
through `fig1_steps_figure.overlay` rather than through a copy of it, so a frame is the
same image as that figure's panel and `--smooth` means the same thing.

The ask it answers -- *a chain of at least three steps that correctly attends to a
different place each time* -- is a filter on the `chains` block the search already writes:
`n_scored >= 3`, `n_regions >= 3` (the referents are clustered at IoU 0.05, so restating
one object three times counts once), every step above chance, and the answer right.
**Ten chains in 1,980 pass it**, all from `bench_c.json` and `bench_d.json`:

| out | sample | steps, raw-grid AUROC | what moves |
|---|---|---|---|
| `video-clevr-vehicles/` | mmstar_mini 69, `fig1b-realworld` | 0.87 / 0.89 / 0.73 / 0.86 / 0.74 / 0.85 | six named vehicles, one per step, on a plain background and with nothing drawn into the picture |
| `video-attic/` | cv_bench_mini 167, `fig1d-search` | 0.80 / 0.84 / 0.89 | chair (right) -> table (centre) -> bookcase (left), the widest spatial spread of the three |
| `video-shelf/` | cv_bench_mini 149, `fig1d-search` | 0.83 / 0.93 / 0.85 | top shelf -> middle shelf -> the desk below it, a vertical sweep |

Two caveats a caption has to carry. **CV-Bench draws its red/blue/green boxes into the
pixels**, and both `video-attic` and `video-shelf` are questions about those boxes, so a
reviewer can say the attention is landing on a salient painted rectangle rather than on
the object -- `video-clevr-vehicles` is the one with no such confound, at the cost of
being a rendered scene. And the **blur is cosmetic**: every AUROC in the table is
`fig1_multistep.py` on the raw grid, as everywhere else on this page.

All three use `--overlay-mode alpha`, for the reason "Making it legible" gives: `blend`
at 0.5 greys out the attic's furniture and the CLEVR background, and the frame has to
show *what* is under the hot blob. Sigma follows the grid, not a default -- 1.0 on the
attic's 24x32, 0.6 on the other two, which is ~3% of the width in all three.

### ...against a baseline that looks in the wrong place and gets it wrong

`--model` repeats, and then each model gets a row and the rows advance together on one
picture. The search for a picture worth putting in those two rows -- **ours right, every
step above chance; the other model wrong, with at least one step below chance** -- has to
drop any chain that hit `--max-new-tokens`, because a cut-off chain has no answer and
"it got it wrong" would be a statement about the budget. That filter matters: the
highest-contrast candidate before it (SalBench P3 357, ours at AUROC 0.97-1.00 against
the cold start's 0.405) is **ours** truncating at 12 steps and being scored correct
because the gold word appears somewhere in the ramble.

After it, against the **cold start**, four pictures survive across every search:

| out | picture | ours | the cold start |
|---|---|---|---|
| `video-cmp-soccer/` | mmstar 54, *"how many soccer players are on the field?"*, gold **C. 4** | 0.625 foreground / 0.636 on the two background players; answers **C** | step 1 at **0.426** over 32% of the frame -- "there are three soccer players visible"; answers **D** |
| `video-cmp-count/` | hrbench4k 29, *"how many people?"*, gold **C. Two** | 0.684 on the man, 0.703 on the second person through the window; answers **C. Two** | step 2 at **0.437** over 37.8% -- "There are no other people visible in the image"; answers **D. One** |
| — | mathvision 59, plums and apples, gold **3** | 0.745 / 0.509; answers **3** | reads *five* plums off the right pan at **0.353**; answers **4** |
| `video-cmp-island/` | hrbench8k 85, *"what is in the middle of the water?"*, gold **C. a tree** | 0.513 / 0.654 / 0.617 on 3-4% referents; answers **C** | "a structure that appears to be a gazebo" at 0.567 over 19%, then **0.414** over 23.5%; answers **D** |

`video-cmp-soccer` is the better of the three rendered: the cold start's second step lands
on the goalkeeper's chest in the foreground while its sentence enumerates the background,
and both background players are visible to a reader.

### The same thing at more than two steps

Asked for the version of this with a **longer chain** -- ours writing more than two
grounded steps -- the pool is one picture, and it is `video-cmp-island/`. Nothing else in
any search clears `n_scored >= 3`, all above chance, answer right, other model wrong and
below chance somewhere, both chains complete. **Against vanilla Qwen3-VL it is zero**, at
any length.

That is a fact about chain length rather than about attention, and it is the shortening
§ 5b's write-up reports from the other side. Over 1,980 of our chains the median is **2**
scored steps and 29% reach three, against 49% of the cold start's 2,139. The funnel:
583 of ours have >= 3 scored steps, 199 of those have every step above chance, 90 of
those also answer correctly, and one of *those* has a cold start that is both wrong and
ungrounded.

`video-cmp-island/` is rendered in `side-by-side/` (`--layout columns`, 1506x978) and
`stacked/`. The picture is good -- ours holds a tight blob on the tree on the island at
all three steps while the cold start spreads from the island across the right bank and
the building, inventing a gazebo and then describing its surroundings -- but two things
make it weaker than the counting panel, and a caption has to survive both:

- **Our three steps are the option strings verbatim** ("A small island with a statue",
  "...a bench", "...a tree"). That is the model enumerating the choices, not observing
  the picture, and it is the same shape as the panels in § 5 where the chain text is
  recitation rather than description.
- **All three land on the same island** (`n_regions` 1). This is *stayed tight on the
  right thing*, not *moved between places*; the crossover claim of § 2 is not in it.

Three near-misses, each failing for a reason worth knowing, because two of them look like
hits in a table. **OmniSpatial 6** (net folding): the cold start writes six steps at
median 0.457 and answers wrong, but every referent is 36-58% of the grid and ours has a
step below chance -- neither model is grounded on a rendered net. **MMStar 17** (a wheelie,
*"what will happen next?"*): ours writes five steps at 0.83/0.84/0.75/0.79/0.57 and the
cold start answers D, but the cold start's map is **as good as ours** (median 0.803) --
this is § 5's *where you look is not what you conclude*, not a grounding failure.
**HR-Bench 4K 23**: the cold start is wrong with a *better* map than ours, 0.882 against
0.532. An anti-example, and the one to check a filter against.

**`video-cmp-count/` is the paper's Figure 1, animated**, in
`portrait/` (`--layout rows`, 1286x1416) and `landscape/` (`--layout columns`,
2066x1546). It matches the figure rather than the scan: the question loses its option
list, the four step captions are the figure's paraphrases, the answers read `One` and
`Two` instead of `D. One` and `C. Two`, and a red cross and a green tick sit beside
them. **Each of those is a `--label` / `--step-text` / `--answer` / `--gold` override,
and every one is written to `<out>/chain.json` next to the text it replaced** -- the map
under each step is untouched and is still that step's.

Both are set at `--font-size 42`, and raising it is not free: the chain's height grows
with the type size while the picture's does not, so a bigger font walks **both** layouts
towards square. Raising it from 30 to 42 needed `--text-width` to come *down* (600, not
580 -> 760) to keep `rows` portrait at 0.91, and `--image-width` to go *up* (1000) to
keep `columns` at 1.34 rather than 1.18. In `rows` the type size is also a ceiling, not a
request: the chain has to fit beside the picture, so past a point `Canvas` silently
shrinks it back. It does not in `columns`, where the chain hangs below.

One thing to carry from it: the figure labels the top row **Vanilla**, and the row is the
**cold start**. `outputs/saliency_viz/fig1b-hrbench/` only ever scanned `ours` and
`coldstart`, the step sentences under it are the cold start's verbatim, and § 5b's
Baselines list vanilla Qwen3-VL-8B-Instruct and the cold-start checkpoint as two
different models. The video reproduces the paper's label; the paper's label is wrong.

**Against vanilla Qwen3-VL there is no such example.** The same filter over `all.json`
returns four pictures and all four are grading, not attention: base answers *"Cabinets"*
to gold `cabinet`, or in prose that the extractor cannot read. That is the same thing
§3 reports at scale -- over 667 shared pictures ours wins 48 and loses 63 -- and it is
why §5b's comparison is against the cold start.

## 6. Reproducing it

```fish
# 1. scan. ~35 min per model per 256 pictures on 8 GPUs; the models run sequentially
bash launch_saliency_viz_job.sh --name fig1ms-valnat --duration 2 --n-samples 256 \
    --model ours=checkpoint/coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged+checkpoint/grpo-coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged-overlap__wov0.4_2head_trmean \
    --model base=Qwen/Qwen3-VL-8B-Instruct \
    -- --methods glimpse,direct --dataset (pwd)/cold_data/grpo_sets/val_natural \
       --split all --max-new-tokens 768

# the benchmark half first needs its 300 documents in the grpo_sets shape
python build_bench_candidates.py --out outputs/fig1-multistep/bench_mini_natural

# 2. search. One GPU, one batched Grounding-DINO pass over every step sentence
bash launch_fig1_multistep_job.sh --name fig1ms-all --duration 1 \
    --run-dir outputs/saliency_viz/fig1ms-valnat \
    --run-dir outputs/saliency_viz/fig1ms-valenat \
    --run-dir outputs/saliency_viz/fig1ms-bench \
    --out outputs/fig1-multistep/all.json \
    -- --model ours=ours --model base=base --ours ours --maps glimpse,direct

# 3. the numbers, and the pictures. Both CPU
python fig1_report.py --json outputs/fig1-multistep/all.json
python fig1_panel.py --json outputs/fig1-multistep/all.json \
    --sample sample_167_row000167 --out outputs/fig1-multistep/panel-motorcycle
# the CLEVR panel. --overlay-mode alpha because its objects are small and low-contrast;
# no --smooth, because its grid is 8x10 and there is nothing to merge ("Making it legible")
python fig1_panel.py --json outputs/fig1-multistep/all.json \
    --sample sample_007_row000007 --chain 2,3,4,5 --overlay-mode alpha \
    --out outputs/fig1-multistep/panel-clevr

# the counting panel, smoothed. Drop --smooth to get figB-hrb-count/ back
python fig1_steps_figure.py --run-dir outputs/saliency_viz/fig1b-hrbench \
    --model ours --model coldstart --sample sample_029_row000029 \
    --map glimpse --boxes outputs/fig1-multistep/bench_b.json \
    --cols 3 --scale 0.8 --smooth 1.0 \
    --question "HR-Bench 4K:  How many people are there in the image?   A. Three   B. Four   C. Two   D. One    (gold: C)" \
    --out outputs/fig1-multistep/figB-hrb-count-smooth
```

```fish
# the animation. CPU, seconds; --question replaces the benchmark's own scaffolding
# ("Answer with the option letter only.", the empty "nan" choices) in the header only
python fig1_steps_video.py --run-dir outputs/saliency_viz/fig1d-search --model ours \
    --sample sample_167_row000167 --smooth 1.0 --overlay-mode alpha --alpha 0.8 \
    --question "Estimate the real-world distances between objects in this image. Which object is closer to the chair (red box), the bookcase (blue box) or the table (green box)?   (A) bookcase   (B) table" \
    --out outputs/fig1-multistep/video-attic

# two models, one picture, one row each. Two rows want a smaller --image-width
python fig1_steps_video.py --run-dir outputs/saliency_viz/fig1b-realworld \
    --model ours --model coldstart --sample sample_054_row000054 \
    --smooth 0.6 --overlay-mode alpha --alpha 0.8 --image-width 560 --text-width 520 \
    --question "MMStar:  Based on the image, how many soccer players are on the field?    A. 1    B. 2    C. 4    D. 3" \
    --out outputs/fig1-multistep/video-cmp-soccer

# the paper's Figure 1. --layout columns instead for the landscape cut; the captions,
# the names and the answers are the figure's, and chain.json records what each replaced
python fig1_steps_video.py --run-dir outputs/saliency_viz/fig1b-hrbench \
    --sample sample_029_row000029 --model coldstart --model ours \
    --label "coldstart=Vanilla" --label "ours=Self-Saliency (ours)" \
    --step-text "coldstart:0=A person wearing a brown jacket and a hat." \
    --step-text "coldstart:1=There are no other people in the image." \
    --step-text "ours:0=A person wearing a hat and a tan jacket." \
    --step-text "ours:1=Second person partially seen in the background." \
    --answer "coldstart=One" --answer "ours=Two" --gold Two \
    --question "How many people are there in the image?" \
    --smooth 1.0 --overlay-mode alpha --alpha 0.8 --font-size 42 --map-label "" \
    --layout rows --image-width 620 --text-width 600 \
    --out outputs/fig1-multistep/video-cmp-count/portrait
# ...and `--layout columns --image-width 1000 --gif-scale 0.4` for the landscape cut

# the three-step one. No --step-text: its captions are the model's own, and they are
# the option strings, which is the honest thing to show
python fig1_steps_video.py --run-dir outputs/saliency_viz/fig1e-r01 \
    --sample sample_085_row000085 --model coldstart --model ours \
    --label "coldstart=Vanilla" --label "ours=Self-Saliency (ours)" \
    --answer "coldstart=a gazebo" --answer "ours=a tree" --gold "a tree" \
    --question "What's located in the middle of the water?    A. a statue    B. a bench    C. a tree    D. a gazebo" \
    --smooth 1.0 --overlay-mode alpha --alpha 0.8 --font-size 26 --map-label "" \
    --layout columns --image-width 720 \
    --out outputs/fig1-multistep/video-cmp-island/side-by-side
```

`--chain` is the N-object mode and `--rank`/`--sample` the two-region one; the second
needs exactly two regions because the crossover claim is about two regions swapping.

`test_fig1_multistep_cpu.py` gates the parts that can be wrong silently: which boxes
become the tight referent, the sign of the margin, and -- the one that actually bit -- how
an answer is extracted and graded. Base signs off with *"This is my answer."* on its own
line after the conclusion, and answers multiple choice with a paragraph followed by a bare
letter; the first version of the grader scored a correct baseline wrong on both.

[trained-heads-lean-on-the-border]: sink-location-by-image-type.md
