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

## The result — pending the judge

`judge` has not been run: it needs `NVIDIA_API_KEY`, which is supplied at run time and is
not in this environment. Everything else is built and verified, and the report below is
the **soft string grade**, which is the noisy label — it is here to show the shape of the
answer and the size of the confound, and every number in it should be re-read off the
judge before it is quoted.

Centre vs the eight other 3×3 bins, same 1,800 pictures, four models:

| model | centre | periphery | diff | Fisher p | MH OR | MH p |
|---|---|---|---|---|---|---|
| Qwen3-VL-8B | 0.448 | 0.407 | +0.041 | 0.092 | 0.96 | 0.82 |
| InternVL3.5-8B | 0.468 | 0.406 | +0.062 | 0.012 | 1.14 | 0.39 |
| GLM-4.1V-9B | 0.421 | 0.368 | +0.054 | 0.027 | 1.11 | 0.54 |
| Nemotron-Omni-30B | 0.355 | 0.318 | +0.037 | 0.119 | 0.99 | 0.99 |

MH holds picture difficulty (how many of the *other* three models answered that picture
correctly, 0–3) crossed with the box's area quartile.

Read the last two columns against the first two. **The raw centre advantage is +3.7 to
+6.2 points in all four models and it dissolves once difficulty and box size are held
fixed** — every common odds ratio lands between 0.96 and 1.14 with p ≥ 0.39. The
threshold-free version agrees: Spearman ρ between the centroid's radial depth and
correctness is 0.012–0.056, and the AUC separating right from wrong answers by depth is
0.507–0.532.

And the confound is visible and large: **ρ(radial depth, box area) = +0.375**. Central
boxes are *bigger* boxes, bigger boxes are answered better (Q4 accuracy is the highest
quartile in all four models), and that alone reproduces the raw gap.

So the provisional reading — to be confirmed on the judge label — is that **the raw centre
advantage in this corpus is mostly the size of the region, not its position.** That is a
useful negative: it is exactly the confound the interventional arms are built to remove,
and Arm 2's slide ladder removes it by construction, since the content is shrunk once and
only the offset varies.

## Power — what Arm 2 has to be sized for

Taking each model's own raw centre-vs-periphery gap as the effect to detect, two-sided,
α = 0.05, power 0.80, two independent groups:

| model | centre | periphery | gap | n per group |
|---|---|---|---|---|
| InternVL3.5-8B | 0.468 | 0.406 | +0.062 | 1,003 |
| GLM-4.1V-9B | 0.421 | 0.368 | +0.054 | 1,301 |
| Qwen3-VL-8B | 0.448 | 0.407 | +0.041 | 2,234 |
| Nemotron-Omni-30B | 0.355 | 0.318 | +0.037 | 2,557 |

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
