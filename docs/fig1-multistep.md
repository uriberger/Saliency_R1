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

**`outputs/fig1-multistep/figB-hrb-count-alpha/`** is that one, and it is the cleanest
example this whole investigation has produced.

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

Two more worth keeping, both with the cold start failing on attention but matching on the
answer: `figB-wemath/` (WeMath, a parallelogram diagram -- five consecutive steps at
AUROC 0.85-0.92, each on the dimension label its sentence names: 30 cm, then 14 cm, then
the 20 cm base; the cold start manages 3 of 4 with a low of 0.48) and
`figB-hrb-flag-ours/` (HR-Bench 4K, the American flag located at AUROC 0.95 on a region
that is **0.6%** of the grid).

**What did not reproduce.** The crossover rate, which separated ours from vanilla on
natural validation images (73% vs 59%), is flat here: 39% vs 41%, p = 0.37. These eight
benchmarks are mostly diagrams, puzzle boards and synthetic fields, where the per-step
referent is not a place in the sense the crossover test needs. The per-step AUROC, which
does not depend on two referents being disjoint, still separates them.

And `direct` is unchanged: median AUROC 0.387 / 0.364, 51% of its mass on the border ring.

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
python fig1_panel.py --json outputs/fig1-multistep/all.json \
    --sample sample_007_row000007 --chain 2,3,4,5 --out outputs/fig1-multistep/panel-clevr
```

`--chain` is the N-object mode and `--rank`/`--sample` the two-region one; the second
needs exactly two regions because the crossover claim is about two regions swapping.

`test_fig1_multistep_cpu.py` gates the parts that can be wrong silently: which boxes
become the tight referent, the sign of the margin, and -- the one that actually bit -- how
an answer is extracted and graded. Base signs off with *"This is my answer."* on its own
line after the conclusion, and answers multiple choice with a paragraph followed by a bare
letter; the first version of the grader scored a correct baseline wrong on both.

[trained-heads-lean-on-the-border]: sink-location-by-image-type.md
