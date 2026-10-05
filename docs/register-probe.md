# H1 — does the top-left visual token carry the whole picture? No.

**Answered on Qwen3-VL-8B, 2026-10-05.** `register_probe.py`, data under
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
