# Arm 0 — does accuracy depend on where the answer sits on the patch grid?

Every geometric result in this project is about **attention**: the border draws 2.6x its
share, the single corner patch 13x, the mark is written by the vision encoder and the
language model attends to it. None of it says the bias **costs** anything, and a reviewer
will ask. This is the first of four arms built to answer that, and the cheapest by a long
way — the boxed corpus already carries a human answer box per picture, four models have
already answered all 1,800, and their completions are on disk as token ids. Nothing is
regenerated and no GPU is used.

Instrument: `box_position_vs_correct.py`, on `llm_judge.py` and `analysis_stats.py`.

```fish
python box_position_vs_correct.py build  --out-dir outputs/box_position/arm0
set -x NVIDIA_API_KEY ...
python box_position_vs_correct.py judge  --out-dir outputs/box_position/arm0 \
    --seed-from 'outputs/human_box_correct/*.judge_cache.json'
python box_position_vs_correct.py report --out-dir outputs/box_position/arm0
```

`build` is ~30 s on a login node. `judge` is 6,151 gpt-4o-mini calls (well under a dollar)
and is cached on content, so a rerun is free. `report` is instant and can be regenerated
from the cache with no key in the environment.

## Where this arm sits

| arm | stimulus | what it can reach |
|---|---|---|
| **0, this one** | the existing boxed corpus, no new images | coarse centre-vs-periphery; prices the effect and sizes Arm 2 |
| 1 | SALBench P3 pop-out, already cached | target position varies uniformly, background constant by construction |
| 2 | the slide ladder — fixed canvas, content slid in whole cells | whole-scene position, paired within picture |
| 3 | targeted paste (rendered word / object cutout) | the only arm that can put the answer in cell (0,0) |

## The ceiling, measured before anything else

The corpus was built to measure attention, not to vary position, and Visual-CoT's answer
regions are centrally placed — `sink_three_legs` already scores them at 0.71–0.76 against
a translation null, i.e. genuinely central. On Qwen3-VL's own grids:

| box centroid | n = 1800 |
|---|---|
| in cell (0,0) | **0** (0.00%) |
| in any corner cell | **1** (0.06%) |
| anywhere on the one-patch ring | 84 (4.7%) |
| coarse 3×3 centre bin | 660 (37%) |

**So this arm cannot test the corner/register claim.** There is no mass to regress on and
no amount of statistics recovers it; only Arms 1 and 3 can reach cell (0,0). What Arm 0
can test is the coarse centre-versus-periphery contrast — the **ring** claim — where the
four corner bins hold 65–93 pictures each.

It also constrains Arm 2's design: because human boxes sit centrally *within their own
picture*, sliding the whole scene moves the answer region but never to the corner. Arm 2
measures scene position, not answer position, and the write-up has to say so.

## What had to be got right

**The gold.** The manifest stores the question and the box but not the answer. It is
recovered by joining `(set, dataset, question_id)` back into `cold_data/grpo_sets` —
**exact on 1800/1800 rows, with zero question mismatches and zero bbox mismatches**, which
`build` verifies and refuses to continue without. A silently wrong join would attach the
right geometry to the wrong gold and every number below would be noise with a plausible
shape.

**The answer, which is a different shape in every family.** `answer_grading.extract_answer`
is Qwen-shaped and is right for two of the four:

| model | format | what the generic rule did |
|---|---|---|
| Qwen3-VL-8B | bare prose, no think block | fine |
| InternVL3.5-8B | bare prose | fine |
| GLM-4.1V-9B | `<think>…</think><answer>… <\|begin_of_box\|>X<\|end_of_box\|>.</answer>` | returned the whole `<answer>` paragraph with its opening tag attached |
| Nemotron-Omni-30B | prompt opens `<think>`; completion is the chain then `</think>` ANSWER | where the cap cut the chain, returned the last line of the *reasoning* — e.g. `'*   Let'` graded as an answer |

The decode therefore keeps special tokens (GLM's `<|begin_of_box|>` **is** one), and the
extractor is family-aware. Strict accuracy on GLM and the Nemotron moves from near-zero to
0.267 / 0.226 on that change alone.

**Unfinished chains.** The scans ran at `--max-new-tokens 1024` and the models spend it
very differently:

| model | hit the cap | produced no answer at all |
|---|---|---|
| Qwen3-VL-8B | 14 | 0 |
| InternVL3.5-8B | 0 | 0 |
| GLM-4.1V-9B | 82 | 82 |
| Nemotron-Omni-30B | 456 | **453 (25%)** |

Those rows are counted **wrong** — a model that produced nothing did not answer — but for
a reason that has nothing to do with the box, so the report repeats the contrast with them
removed, and checks whether running out of budget depends on position at all
(ρ = −0.046 / −0.047, p ≈ 0.05 — a weak tendency for peripheral boxes to run long).

**The label is the judge, and that is not cosmetic.** These scans ran with
`--system-prompt none`, so Qwen3-VL and InternVL answer in prose: exact match scores
Qwen3-VL at **0.006**. A word-boundary substring rule mis-scores in both directions
(gold `car` against "the undercarriage of a vehicle" reads wrong; gold `man` fires on a
word elsewhere in the span). And 327 of the 1,800 rows are flickr30k, whose gold is a full
sentence — they are real VQA with real questions, the judge grades them fine, and a string
rule scores them at **0.003** and would have forced dropping 18% of the corpus.

## The result — 2026-10-05

Judge run: 7,152 of 7,200 rows scored, 48 gateway failures excluded rather than counted
wrong. `outputs/box_position/arm0/{table.jsonl,judge_cache.json,report.txt}`.

**The judge was worth buying.** It agrees with the soft string grade on 76.2% of rows, and
the disagreement is one-sided: 18.4% of rows are judge-right/soft-wrong against 5.4% the
other way. Accuracy by model, judge vs soft vs strict:

| model | judge | soft | strict | hit the 1024 cap | produced no answer |
|---|---|---|---|---|---|
| GLM-4.1V-9B | **0.585** | 0.387 | 0.267 | 82 | 82 |
| InternVL3.5-8B | **0.554** | 0.429 | 0.048 | 0 | 0 |
| Nemotron-Omni-30B | **0.493** | 0.331 | 0.226 | 456 | 453 |
| Qwen3-VL-8B | **0.461** | 0.422 | 0.006 | 14 | 0 |

A string rule would have under-read every model by 6 to 20 points, and would have reversed
the ordering of Qwen3-VL and the Nemotron.

### Centre vs periphery

Centre bin against the eight other 3×3 bins, same 1,800 pictures. MH holds picture
difficulty (how many of the *other* three models answered that picture correctly, 0–3)
crossed with the box's area quartile.

| model | centre | periphery | diff | Fisher p | MH OR | MH p |
|---|---|---|---|---|---|---|
| InternVL3.5-8B | 0.597 | 0.530 | +0.067 | 0.007 | 1.14 | 0.33 |
| GLM-4.1V-9B | 0.623 | 0.563 | +0.060 | 0.014 | 1.17 | 0.26 |
| Qwen3-VL-8B | 0.498 | 0.440 | +0.059 | 0.018 | 1.07 | 0.62 |
| Nemotron-Omni-30B | 0.511 | 0.482 | +0.028 | 0.260 | 0.84 | 0.16 |

The raw gap is +2.8 to +6.7 points and significant in three of four. Every stratified odds
ratio lands between 0.84 and 1.28 with p ≥ 0.16 — and the Nemotron's points the other way.
Dropping its 453 unfinished chains does not rescue it (+0.021, p = 0.48).

### Threshold-free, and then the number that settles it

Unlike the soft label, the continuous version is significant everywhere: Spearman ρ between
the box centroid's radial depth (0 = on the border, 1 = dead centre) and the judge's own
0–1 score is positive in all four models, p ≤ 0.036.

But radial depth and box **area** are correlated at **+0.375** on this corpus — central
boxes are bigger boxes — and area predicts correctness on its own at ρ = +0.069 to +0.138,
*larger* than the depth correlation in three of the four. So the question reduces to a
partial correlation, on the whole rank ordering rather than in four bins:

| model | ρ(depth, score) | p | **ρ(depth, score \| area)** | p | survives |
|---|---|---|---|---|---|
| Qwen3-VL-8B | 0.1013 | <0.0001 | **0.0431** | 0.077 | 43% |
| InternVL3.5-8B | 0.0647 | 0.008 | **0.0083** | 0.734 | 13% |
| GLM-4.1V-9B | 0.0624 | 0.011 | **0.0446** | 0.069 | 71% |
| Nemotron-Omni-30B | 0.0510 | 0.036 | **0.0139** | 0.567 | 27% |

**Hold box size fixed and the position effect does not reach significance in any of the
four models**, and how much of it survives — 13% to 71% — is not consistent across them.
Accuracy by area quartile is monotone or near-monotone in all four (Qwen3-VL runs
0.364 → 0.434 → 0.527 → 0.520).

### What Arm 0 concludes

**On this corpus, at this granularity, the centre advantage is the size of the answer
region and not its position.** The upper bound on a position effect is about ρ = 0.045,
which is an AUC of roughly 0.52 — nothing you would design a system around.

Three things that does and does not license:

1. It does **not** say the geometric bias is harmless. The corpus cannot put an answer in
   cell (0,0), where the register sits and where the effect would be largest if it exists,
   and a 3×3 bin is a very blunt instrument for a one-patch phenomenon.
2. It **does** say the confound is real and dominant, and that any arm which varies
   position must hold the target's size fixed. Arm 2's slide ladder does this by
   construction — the content is shrunk once and only the offset varies — which is now a
   requirement rather than a preference.
3. It **does** say the effect, if present, is small. Arms 1–3 should be sized for a few
   points, not for a dramatic one.

## Power — what Arm 2 has to be sized for

Taking each model's own raw centre-vs-periphery gap as the effect to detect, two-sided,
α = 0.05, power 0.80, two independent groups:

| model | centre | periphery | gap | n per group |
|---|---|---|---|---|
| InternVL3.5-8B | 0.597 | 0.530 | +0.067 | 868 |
| GLM-4.1V-9B | 0.623 | 0.563 | +0.060 | 1,042 |
| Qwen3-VL-8B | 0.498 | 0.440 | +0.059 | 1,129 |
| Nemotron-Omni-30B | 0.511 | 0.482 | +0.028 | 4,929 |

Arm 2 is **paired within picture**, so its real requirement is lower than this by roughly
the within-picture correlation — but the order of magnitude is the point: a few hundred
pictures will not settle a 4-point effect, and the pilot should be sized against this
table rather than against intuition.

## Caveats

- **Observational.** Box position is not randomised and is confounded with what the
  question is about. Difficulty and box size are two nuisance routes closed explicitly;
  neither closes the confound itself.
- **MH over-controls.** Difficulty is defined from the other models' correctness, so if
  position hurts every model at once the stratification removes part of the signal. The
  raw and stratified rows are both printed and neither is the answer on its own.
- **The four models answered the same pictures**, so the four rows are one test read four
  ways, not four independent tests.
- **The judge does not see the image.** It compares the extracted answer with the
  Visual-CoT gold string — fair across formats, blind to a right answer phrased about a
  different object.
