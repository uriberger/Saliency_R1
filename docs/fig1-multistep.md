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

It is the best *picture* of the behaviour by a distance. It is not a win over base: base's
chain is 12 steps long and, allowed to pick its own best step per object, matches it
(2.79 / 6.71 / 3.77 / 4.39). Use it to show what the model does, not to show that the
baseline cannot.

**`panel-dogbed/`** -- `val_natural` row 234, *"Where is on the dog bed?"*, gold `cat`.
The widest margin gap on an unambiguous question (+1.24 against base's **-0.64**), but
both models answer correctly and region B is scene furniture rather than something the
question needs.

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
