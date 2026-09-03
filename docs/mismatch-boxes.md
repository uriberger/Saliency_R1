# Mismatched boxes: does it matter that DINO read *that sentence*?

The attention-overlap reward runs Grounding-DINO on each observe step's own text and
rewards the policy for attending inside the boxes that come back. Every variant in this
repo — `mean_in`, `mean_in_v2`, `auroc`, the roll-null, the gradient and GLIMPSE maps — is
downstream of one assumption: that grounding **that sentence** is what makes the boxes
mean anything.

Two offline results say it may not be.

- `docs/step-box-similarity.md`: two steps of one chain get union masks no more alike
  than two steps of two *different* chains about the same picture (closeness 0.72 vs
  0.70), and a step's map scores higher on its own mask than on another step's only
  52.6% of the time — a coin flip.
- `trl/rewards/maskfree_rewards.py`: the best single predictor of `mean_in` is
  `mean(m)/max(m)`, which never sees a box (r = +0.69 within group), while a genuinely
  relocated union predicts it at r = −0.015.

Neither can say what a policy *trained* against wrong boxes would do — both re-score maps
that a DINO-trained policy already produced. This is that run.

    bash launch_mismatch_bank.sh --out-dir outputs/mismatch_bank/8k          # once, ~1-2 h
    bash launch_grpo_qwen3_overlap_colocated_job.sh \
        --mismatch-bank outputs/mismatch_bank/8k/bank.json --lora-targets q_proj,v_proj

## What it is

Each observe step is scored against **real Grounding-DINO output on a real photograph for
a real cold-start sentence** — grounded offline, before training — that belongs to a
**different question about a different picture**. Only the pairing is wrong.

That places it between two controls that already exist:

| control | the union it scores against |
|---|---|
| the reference | the step's own text, this picture |
| `--placebo roll` | the step's own union, moved to a wrong place (same area, same shape) |
| **`--mismatch-bank`** | **another chain's real union, from another picture** |
| `--maskfree` | no union at all |

`roll` keeps the correct union's size and shape and breaks only its position, and it still
runs DINO on the step's text — so the boxes still respond to what the policy writes.
`--mismatch-bank` severs that entirely: **no Grounding-DINO is loaded during training**,
which is 16.6 s off a 40.5 s optimizer step.

## The measurements the design rests on

All from re-scoring maps already on disk — the `base_coldstart` entries of
`outputs/overlap_probe/20260805-030416-crossrun-val_natural`, `grad_spread` and
`mean_in_v2_spread`. No GPU, no new generation.

### 1. Observe-step counts, and why the length tail is guaranteed to miss

880 cold-start chains:

| n observe steps | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9–14 |
|---|---|---|---|---|---|---|---|---|---|---|
| share of chains | 3.1% | 10.8% | 21.7% | 23.5% | 17.4% | 10.5% | 6.3% | 3.1% | 1.5% | 2.3% |

Median 3, 96.3% at 7 or fewer, **maximum 14**. The *trained* checkpoints in the same
probes reach **70, 83, 85, 87**. So no bank built from the cold-start model can hold a
chain as long as the policy will eventually write, at any bank size, and "no chain of this
length" is guaranteed to happen and to get more common as a run drifts. The design has to
answer it as a normal case, not an error.

### 2. Donor length barely affects the score it hands out

Mean `mean_in` a foreign map gets on a donor's union, by that donor's chain length
(250 recipient maps × 538 donor chains, different image and different question,
124,385 pairs):

| donor length | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| mean score | .0603 | .0603 | .0614 | .0610 | .0613 | .0609 | .0596 | .0621 |

Flat to ±0.002. Relaxing the length match is cheap.

### 3. Donor *identity* is worth as much as the entire reward

The completion-level score, spread **across the 8 rollouts of one prompt** — the only
quantity GRPO sees, since the group mean is subtracted before anything else (53 groups,
24 donor chains each):

| | within-group sd |
|---|---|
| real reward — each completion on its own boxes | **0.0115** |
| mismatched — all 8 rollouts sharing **one** donor chain | **0.0094** |
| donor lottery — one completion across 24 donor chains | **0.0117** |

Two things follow.

**The control keeps 82% of the reference's tie-breaking strength** after losing the
pairing entirely. That is the offline headline, and the reason the run is worth its GPUs.

**A donor drawn per completion would drown it.** Independent donors put the third row
inside the group; roughly 60% of the reward's within-group variance becomes which donor a
rollout happened to draw. That run is `--placebo random` with a box-shaped distribution,
and it already exists. So the donor is fixed **per prompt row** and shared by every
rollout of it.

### 4. Where the donor variance lives, and what that buys the length ladder

Spread of the mean score a donor hands out, decomposed:

    across all donor chains                          0.0051
      between donor images                           0.0036
      between chains of ONE donor image              0.0024

Consistent with `step_box_similarity`: the union is mostly a property of the picture. It
also prices the two ways of serving a step count the bank cannot match exactly:

| | cost | × the real reward's 0.0115 |
|---|---|---|
| a wrong-length chain from the **same** donor row | 0.0024 | 0.21× |
| hopping to another donor row for an **exact** length | 0.0117 | 1.02× |

Chasing the exact length across donors costs five times what accepting the length mismatch
costs. **So the donor is picked first and the length second**, and a thin or empty
length pool is never consulted — which is also why "only one chain of this length exists"
never arises as a case. Had it, the cost would have been the worst of both: every
completion of that length in the entire run carrying one fixed union's ~0.6 σ offset as a
pure length signal, inside groups that mix lengths.

The third option — leave the completion unscored when no chain of length *n* exists — is
the one that must not be taken. The policy chooses its own observe-step count, so an
unservable count is a free exit from the reward, and it would be found.

## The rule, in full

For a completion with *n* observe steps, on training row `(dataset, question_id)`:

1. **Donor row.** `blake2b(seed, "mismatch-donor", row_key) % D` picks a starting index
   into the bank's donor list; scan forward to the first donor whose question *and*
   picture both differ from this row's. Deterministic — the same donor in every epoch, on
   every rank, after any restart — and identical for all rollouts of the row.
2. **Chain.** The donor row's chain of length exactly *n* when it has one; otherwise the
   nearest length it does have, ties going to the longer chain (which needs no wrap).
   Which of that length's stored variants is a second hash of the row key.
3. **Step.** Recipient step *i* takes donor step `i % L` — wrapped when the donor chain is
   shorter, cut short when longer.
4. **Score.** `overlap_rewards._union_mask` then `overlap_rewards._step_score`, i.e. the
   run's own `--overlap-metric`, mass floor, `--max-box-area` and `--max-union-area`, then
   the mean over the completion's scored steps and the multiplicative format gate.

Assignment is positional, not by a hash of the step text. By text hash a repeated sentence
would get a repeated box, which is the property that makes the duplicate-step hack work on
the real reward — and re-attaching the text to the boxes is the dependence this control
exists to sever.

### Identity

`(dataset, question_id)`, which the trainer forwards to every reward as a dataset column.
Unique across all 8080 rows of `saliency-r1-8k`; `problem` is not (6844 unique texts, one
question repeated 41 times).

"Different picture" is enforced, not assumed: 793 of the corpus's 6714 images carry more
than one question, up to 10, so excluding the row itself would still let a row be scored
against its own picture through a sibling. The bank ships an index of every row key to a
hash of its encoded image bytes, and a row the index does not cover **raises** rather than
falling back to a question-only exclusion.

## What is held equal to the reference, and the one thing that is not

Held equal, by reading `overlap_rewards._CFG` rather than duplicating it: the metric, the
mass floor, the box-area and union-area caps, `--overlap_natural_only`, and the three ways
a completion can go unscored (an ungroundable step is skipped, not scored 0; a completion
with no usable step returns `None` and is neutral in the advantage). `configure()` refuses
a bank whose `--box_threshold` differs from the run's, because that filter was applied by
DINO when the bank was written and cannot be re-applied to stored boxes.

**Not** held equal: *which* completions the skips land on. Here they follow the donor;
there they follow the completion's own text. `placebo_rewards` closes exactly this gap by
running the real pipeline as a gate — this control cannot, because not running DINO is the
point. The size of the gap, measured on the cold start: DINO leaves **3.4% of steps**
ungrounded and **0.5%** of completions that have at least one observe step with nothing
scored at all. The bank stores an ungroundable donor step as an empty box list, so the
*rate* is inherited even though the incidence is not.

## Reading the result

Against `overlap mean_in w0.4` on the same corpus, at `w 0.49` (= 0.4 × 0.0115/0.0094, so
both runs apply the same tie-breaking pressure), and against the accuracy-only control
(`--w-overlap 0`). Not against `baseline/grpo-no-saliency`, which starts from vanilla
Qwen3-VL-8B rather than the cold-start merge.

- **Control ≈ reference** → the pairing never mattered. Together with `--maskfree`, that
  puts the reward's whole effect in "point attention at *a* plausible region" — and under
  `mean_in`, most likely in flattening.
- **Reference > control** → read it carefully. This control **cannot be hacked the way its
  reference can**: repeating a trivially-groundable sentence and describing the background
  both stop working once the boxes no longer respond to the text. So a gap is either "the
  sentence mattered" or "the reference was hacking". The duplicate-sentence fraction and
  the union area, both already logged, separate the two.

One more caveat that is not optional: **a seed is a random pairing, and one pairing is not
a result.** Run two seeds (`--mismatch-seed 0`, `--mismatch-seed 1`) before believing
either direction.

An offline hint about which way it will go, and a warning about it. Paired over 996
cold-start steps, a step's map scores **0.0508** on its own DINO union and **0.0572** on a
same-length foreign chain's — the foreign union scores *higher*, and the own union wins on
only 31.9% of steps. Treat that as an unpinned observation rather than a prediction: it
mixes grid shapes across two probe corpora, and the union sizes were not matched.

## Diagnostics

Logged as `mismatch/*`:

| key | what it says |
|---|---|
| `exact_len_frac` | share of scored completions whose step count the donor row matched exactly. **The one to watch** — it falls as the policy's chains outgrow the cold start's, which is a description of the drift, not a failure. |
| `len_delta` | mean (donor length − completion length) |
| `wrap_frac` | share where the donor chain was shorter and wrapped |
| `union_frac` | mean grid coverage of the donor union — directly comparable to the reference's |
| `step_skip_frac` | share of steps whose donor boxes gave no usable union |

## Building the bank

`build_mismatch_bank.py`, in four phases (`launch_mismatch_bank.sh` runs all of them):

    --plan     CPU, ~2 min. Hashes every image in --index-dataset, picks --n-donors rows
               from distinct pictures.
    --shard    GPU, one per card. Generates --n-generations chains per donor row with the
               cold-start model at the training sampling settings, segments observe steps
               with the same FLAN-T5 classifier the trainer uses, grounds every step once.
    --merge    CPU. Shards + plan -> bank.json.
    --verify   CPU. Re-runs the real donor resolution over 2000 sampled rows and asserts
               every one lands on a different question and a different picture; asserts
               step counts 1..19, 40, 85 and 200 all resolve; prints the length coverage.

Defaults are 256 donor rows × 64 chains = 16,384 chains, ~1–2 hours on 8 cards.

**256 donor rows** is enough because a donor row is reused by ~31 training rows and the
corpus is still spread over 256 distinct wrong regions. Raising it buys diversity, not
correctness.

**64 chains per donor row** is where the exact-length service rate stops improving. Chains
per donor row against the share of completions whose step count its own donor row can
match exactly, at the cold-start length mix (analytic from the histogram above; the
builder's `--verify` prints the same figure measured from the bank it actually built, and
the two agreed to 0.8 points on a 40-donor test):

| chains per donor | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|
| exact-length service | 49% | 71% | 87% | 94% | **97%** | 98% |

At 64, lengths 1–6 are covered by essentially every donor row and n = 7 by 87%; the
misses are the tail the cold start itself barely produces. 128 buys 1.3 points for double
the GPU time. Everything past the bank's longest chain is unservable at any size — see the
length ladder above, which is what makes that acceptable rather than fatal.

Everything that decides what a chain *is* comes from `overlap_probe.py` rather than being
reimplemented — the same `SYSTEM_PROMPT`, the same 512px image cap, the same sampling
parameters, the same format regex, the same segmentation. A bank built off-distribution
would make the control differ from its reference in a second way.

## Files

    trl/rewards/mismatch_rewards.py   the reward
    build_mismatch_bank.py            the offline bank builder
    launch_mismatch_bank.sh           plan + shard + merge + verify on one node
    test_mismatch_reward_cpu.py       31 CPU checks; T4 pins the shared donor, T2 pins
                                      that no step count can go unscored
