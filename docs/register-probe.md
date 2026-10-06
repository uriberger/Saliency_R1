# H1 — does the top-left visual token carry the whole picture? No.

**Answered on Qwen3-VL-8B, 2026-10-05, twice: by a linear probe on the encoder's output
and by a causal intervention through the whole model. They agree.** `register_probe.py`, data under
`outputs/register_probe/pope/`.

## The question, and why it needed asking

This project has been calling cell (0,0) a **register**. The evidence for that was a
signature, not a definition:

- the attention peak is cell (0,0) on **88.0%** of 1,800 pictures;
- its row's norm is **2.3×** a random row's;
- replacing its vector costs **1.56×** what replacing a random row costs (norm-matched);
- destroying its **pixels** costs **0.87×** — *less* than a random cell's.

So it is load-bearing and its content did not come from underneath it. Darcet's
definition of a register is a further claim: the token holds information about the
**whole image** rather than about its own patch. Nobody here had tested that.

## The test

Take **one square's vector** — 4,096 numbers, exactly as the language model receives them
from the vision tower — and train the simplest possible classifier to answer a question
about the entire picture.

**Label**: "Is there a *<object>* in the image?", ground truth from COCO's human
annotations by way of **POPE**. Unioning POPE's three splits recovers **5,127
(picture, object) labels over 500 pictures with zero disagreements between splits**, which
is the check that the parse is right. 16 object classes clear 60 pictures with 20 on the
minority side. (POPE misspells "image" as "imange" on 57 of its 9,000 rows, all on objects
taking "an"; parsing only the correct spelling silently drops real labels from exactly
those six classes.)

This is the right shape of question because the object is almost never in the corner: a
square at (0,0) has no local evidence for "is there a dog *somewhere* in this picture".

**Arms**, all one square, all scaled to unit length first because the top-left row's norm
is 2.3× a normal row's and a classifier can read scale:

| | |
|---|---|
| `tl` | cell (0,0) — the suspect |
| `mid` | a fixed middle cell — **the control that matters**, because it is *also* always in the same place |
| `rand` | a random cell per picture |
| `br` | cell (gh−1, gw−1), the last token |
| `mean` | the average of all cells — a reference, not a ceiling |

Scores are **balanced accuracy** on held-out pictures, so always answering the commoner
way scores 0.50. Five folds, the same folds for every arm; the regularisation strength is
chosen by an inner cross-validation inside each training fold, separately per arm.

**And the floor is measured, not assumed.** A synthetic check at n=240 put an
uninformative arm at 0.589, so the same pipeline is run on shuffled labels. On the real
corpus that floor is **0.498** (spread 0.062).

## The result

| object | with / without | tl | mid | rand | mean | br | tl, shuffled |
|---|---|---|---|---|---|---|---|
| person | 345 / 152 | **0.819** | 0.648 | 0.665 | 0.880 | 0.767 | 0.475 |
| car | 63 / 410 | 0.657 | 0.610 | 0.619 | 0.858 | 0.642 | 0.480 |
| dining table | 61 / 370 | 0.560 | 0.592 | 0.634 | 0.836 | 0.589 | 0.552 |
| chair | 44 / 367 | 0.585 | 0.583 | 0.577 | 0.765 | 0.582 | 0.509 |
| truck | 46 / 85 | 0.524 | 0.527 | 0.608 | 0.621 | 0.427 | 0.478 |
| bottle | 43 / 87 | 0.608 | 0.509 | 0.554 | 0.651 | 0.542 | 0.489 |
| backpack | 48 / 75 | 0.606 | 0.548 | 0.570 | 0.671 | 0.531 | 0.492 |
| handbag | 32 / 70 | 0.529 | 0.505 | 0.479 | 0.631 | 0.562 | 0.454 |
| bowl | 47 / 48 | 0.511 | 0.473 | 0.521 | 0.673 | 0.558 | 0.527 |
| tv | 32 / 58 | 0.609 | 0.474 | 0.493 | 0.808 | 0.492 | 0.435 |
| couch | 32 / 51 | 0.554 | 0.474 | 0.466 | 0.834 | 0.613 | 0.555 |
| traffic light | 22 / 56 | 0.635 | 0.690 | 0.479 | 0.712 | 0.570 | 0.541 |
| dog | 22 / 49 | 0.634 | 0.692 | 0.794 | 0.722 | 0.523 | 0.469 |
| spoon | 29 / 39 | 0.615 | 0.589 | 0.691 | 0.766 | 0.624 | 0.524 |
| bicycle | 20 / 46 | 0.538 | 0.672 | 0.437 | 0.662 | 0.494 | 0.482 |
| sports ball | 38 / 25 | 0.780 | 0.725 | 0.605 | 0.881 | 0.787 | 0.499 |
| **MEAN** | | **0.610** | **0.582** | **0.575** | **0.748** | **0.581** | **0.498** |

Two findings, and they point in different directions.

### 1. A single visual token knows a surprising amount about the whole picture

Every arm sits far above the 0.498 floor. **One 32×32 square's vector predicts whether a
person, a car or a dog is *anywhere* in the picture at 0.57–0.61**, and that is true of a
random square as much as of the corner. The vision tower does not keep a patch's vector
about that patch; it spreads picture-level information into all of them.

The `mean` arm at **0.748** says the information is there in quantity and is **distributed**
— averaging all the squares beats any one of them by 14 points.

### 2. The top-left token is NOT special, which is the answer to H1

| | tl − mid |
|---|---|
| unweighted over 16 classes | **+0.028** |
| weighted by pictures | +0.044 |
| 95% interval, clustered on class | **[−0.041, +0.098]** |
| the same difference with labels shuffled | +0.000 (spread 0.084) |
| top-left wins in | 11 / 16 classes, 95% interval [0.44, 0.86] |

**The gap does not clear its own null.** Prior work on this comparison reports registers
beating ordinary tokens by 20–30 points; we measure 2.8, with an interval that contains
zero. The bottom-right corner (0.581) is indistinguishable from the middle (0.582) too, so
it is not a corner effect in either direction.

`person` is the one class with a large gap (+0.171) and it is also the class with by far
the most pictures (497 against 60–131 for most). The six classes with ≥130 pictures
average +0.048 and the ten smaller ones +0.017, which is consistent with a small real
effect *and* with the big classes simply being measured more precisely.

## What this changes

**The word "register" has to come off cell (0,0)**, in the sense the literature defines it.
What survives is narrower and still interesting: that token is attended 13× its share, is
a 2.3× norm outlier, matters 1.56× a random token when replaced, and does **not** take its
content from its own pixels — but it holds no more about the picture than its neighbours
do. That combination is the "pure norm sink" that the attention-sink literature describes
(high norm, low probe accuracy) rather than Darcet's information-carrying register.

It also means the project's Q2 reasoning does not stand up: the argument was that the
first visual token carries the summary, so the rest might be redundant. The rest are not
redundant — they carry the same kind of information, and the average of them beats the
corner by 14 points.

## The objection, and the causal test that answers it

A probe only sees what a *linear* classifier can read off a vector **in isolation**. The
encoder and the language model were trained together, so the model might extract
whole-picture information from that cell by a route no probe would see. That objection is
right, and it needs an intervention rather than a classifier.

**The test.** Move ONE cell a fixed distance toward a donor picture's cell at the same
index — in the rows the language model consumes **and** in all three DeepStack injections,
since Qwen3-VL feeds those into the decoder's early layers under the same indexing, and
moving the pooled row alone would leave three quarters of the token's content in place.
The distance is common to all three cells and is the **smallest** of that picture's three
literal-swap distances (mean 19.16), so no arm is hit harder than another. Then ask the
model POPE's own question in words and read the first answer token.

**5,096 questions over 497 pictures, every arm.** The mass on {yes, no} is 1.0000, so the
model answers the question asked and nothing is being renormalised away.

| arm | accuracy | P(correct) | Δ accuracy vs clean | Δ P(correct) vs clean |
|---|---|---|---|---|
| clean | 0.9129 | 0.9092 | — | — |
| **tl** | 0.9150 | 0.9104 | **+0.0022** [−0.0001, +0.0044] | **+0.0012** [+0.0003, +0.0021] |
| mid | 0.9121 | 0.9083 | −0.0008 [−0.0022, +0.0007] | −0.0009 [−0.0017, −0.0001] |
| rand | 0.9111 | 0.9079 | −0.0018 [−0.0034, −0.0001] | −0.0013 [−0.0023, −0.0003] |
| tl_raw (literal swap) | 0.9141 | 0.9098 | +0.0012 [−0.0016, +0.0039] | +0.0007 [−0.0005, +0.0019] |

Intervals are clustered on picture, because the ~10 questions about one picture share its
cell.

**Corrupting the top-left cell costs nothing.** If anything it helps by a hair:
tl − mid on P(correct) is **+0.0021** [+0.0010, +0.0033], tl − rand **+0.0025**
[+0.0013, +0.0038]. Both in the model's favour.

**And the instrument is not blind** — this is what makes the null worth something.
Corrupting a *random* cell measurably hurts (−0.0018 accuracy, −0.0013 P(correct), both
intervals excluding zero). A random cell sometimes holds the object; the corner never
holds anything the answer needs. So the probe and the intervention agree, by two methods
that could easily have disagreed.

### The cells themselves

| cell | mean row norm | × the mean cell |
|---|---|---|
| top-left | 36.04 | **1.71** |
| middle | 24.72 | 1.18 |
| random | 20.46 | 0.97 |

The top-left cell is a norm outlier at 1.71× here (this project has 2.3× on record from a
different corpus), **and it is never the largest**: the largest-norm cell of each picture
is 3.19× the mean, sits at cell (0,0) on **0.0%** of pictures, and moves around with the
content — 174 distinct cells take it, 89.6% of them in the interior. Those are two
different phenomena and only one of them is in a fixed place. The second has not been
looked at here at all.

### One tension to state rather than bury

`token_mediation_probe` found that swapping this same cell costs **1.56×** what swapping a
random cell costs, measured as log-KL along a teacher-forced chain. Here it costs *less*
than a random cell. Both are right and they measure different things: log-KL over a long
free-form chain moves on any shift in the output distribution, including generic ones,
while a yes/no answer about object presence is a task-specific readout. The corner shifts
what the model says without carrying what the answer needs — which is the norm-sink
picture, stated twice.

## What would overturn it

**Sample size, and only sample size.** With 500 pictures the interval on tl − mid is about
±0.07. That rules out the 20–30 point effect prior work reports; it does not rule out a
5-point one. The fix is more labelled pictures: COCO's instance annotations cover ~40,000
val2014 images against POPE's 500, the login node reaches the Hub, and nothing else about
the pipeline changes. If the register effect is real but small, that is what would show it.

## Running it

```fish
python register_probe.py labels  --out-dir outputs/register_probe/pope    # ~1 min, CPU
bash launch_register_probe_job.sh --out-dir outputs/register_probe/pope --stage extract
bash launch_register_probe_job.sh --out-dir outputs/register_probe/pope --stage probe \
    --duration 2 -- --jobs 16
python register_probe.py report  --out-dir outputs/register_probe/pope
```

`extract` is one prefill per picture: **3 minutes on one GPU**, 763 MB of rows. It keeps
*all* rows rather than the five arms currently asked for, so a new arm needs no GPU.
`probe` is ~40 minutes on a `cpu_short` node. Neither belongs on the login node.

## Caveats

- **One model, one checkpoint.** Qwen3-VL-8B-Instruct.
- **500 pictures**, 60–500 per class. Sized for a large effect.
- **POPE gives presence, not boxes**, so pictures where the object really does sit in the
  top-left corner cannot be filtered out. The pixel arm argues that matters little —
  destroying that square's own pixels changed the model *less* than destroying a random
  square's — but it is not nothing.
- **The rows are the vision tower's output.** Nothing here is about what the language
  model does with them afterwards, which is a separate question and the same extraction
  would answer it with one more hook.
- **A linear probe is a lower bound on what is present.** Information the classifier
  cannot read linearly is still information; this measures what is linearly available,
  which is what the comparison between arms needs and not the same as what is there.
  The intervention is what closes that gap, and it agrees.
- **The intervention's task is easy** — object presence, answered at 91%. A harder or
  finer-grained question might be more sensitive to the same perturbation. What the
  random-cell arm establishes is that the instrument can see *something* at this
  difficulty, not that it could see everything.
