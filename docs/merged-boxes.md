# Every step against the whole chain's union — 2026-09-21

`--overlap_merge_boxes` (`--merge-boxes` on the launcher), and the offline measurement
that says what it will do before a GPU is spent on it.

Tool: `mask_variance_probe.py`, scheme `chain_union`. CPU only, seconds, no GPU and no
Grounding-DINO — it reads the maps `overlap_probe.py --store-maps` already wrote and
builds every mask by importing `trl/rewards/overlap_rewards.py`, so a number here is a
number about the code a run will execute.

```fish
set PY /home/uberger/scratch/miniconda3/envs/saliency_r1_qwen3_vllm/bin/python
$PY mask_variance_probe.py \
    outputs/overlap_probe/20260809-021810-crossrun-val_natural-plus/probe_merged.json \
    --question-boxes outputs/question_boxes/val_natural_bt0.10.json
```

## The arm

Ground every observe step exactly as the incumbent does — same calls, same sentences,
same count — then **merge** the completion's box lists and score every one of its steps
against that single union. "Did this step look where its own sentence points" becomes
"did this step look anywhere the chain ever mentions".

It is **not** a rung of the [per-completion-masks.md](per-completion-masks.md) ladder.
Those four (per step, per completion, per row, no detector) all trade detector calls for
a coarser mask. This one buys nothing on cost and changes only the target, which is what
makes it the arm that separates granularity from budget:

| | detector calls | mask varies within a prompt's 8 rollouts? |
|---|---|---|
| per step (incumbent) | one per step | yes, per step |
| `--overlap_chain_boxes last` | one per **completion** | yes, per completion |
| `--overlap_question_boxes` | none (precomputed per row) | **no** |
| `--overlap_rect_frac`, centred | none | **no** |
| **`--overlap_merge_boxes`** | **one per step** (unchanged) | **yes, per completion** |

Two further behavioural differences from the incumbent, both consequences of one mask per
completion rather than one per step:

- a step that grounds nothing is **no longer skipped** — its neighbours' boxes give it a
  mask. Only a completion where *nothing* grounded is lost, so the scored set is larger.
- `--max_union_area` applies **per completion**, because the merged list is the same for
  every step and so is the cap's verdict.

## What it does to the mask — 11 checkpoints, val_natural

Identical generations, only the mask varied. Coverage is of the patch grid, over **every**
completion including the ones the reward then refuses; `sat%` is the share whose merged
union covers the grid outright, which `_union_mask` refuses and which costs the whole
completion. Ordered by chain length, and that ordering is the finding:

| checkpoint | steps/comp | merged cover | p90 | sat% | sd vs per-step | w to match | r(flat) |
|---|---|---|---|---|---|---|---|
| `auroc_set_a_2500` | 1.1 | 0.519 | 0.73 | **0.0%** | 0.95 | 0.42 | 0.762 |
| `auroc_set_a_2000` | 1.5 | 0.532 | 0.79 | 0.8% | 0.95 | 0.42 | 0.541 |
| `mean_in_saliency_r1_8k` | 2.1 | 0.720 | 0.95 | 2.7% | 1.00 | 0.40 | 0.714 |
| `auroc_set_a_1000` | 2.4 | 0.672 | 0.93 | 3.0% | 1.00 | 0.40 | 0.644 |
| `mean_in_set_a_1000` | 3.4 | 0.753 | 1.00 | 10.5% | 1.02 | 0.39 | 0.715 |
| **`base_coldstart`** | **3.7** | **0.754** | **0.97** | **6.5%** | **1.09** | **0.37** | **0.611** |
| `mean_in_v2_set_a_1000` | 3.7 | 0.807 | 1.00 | 16.6% | 1.12 | 0.36 | 0.687 |
| `mean_in_v2_beta_004` | 4.4 | 0.844 | 1.00 | 22.7% | 1.21 | 0.33 | 0.567 |
| `mean_in_v2_set_a_1500` | 5.6 | 0.924 | 1.00 | 50.4% | 1.02 | 0.39 | 0.643 |
| `mean_in_set_a_2000` | 13.2 | 0.945 | 1.00 | **64.9%** | 0.74 | 0.54 | 0.778 |
| `mean_in_v2_set_a_1700` | 14.1 | 0.946 | 1.00 | **67.9%** | 1.18 | 0.34 | 0.730 |
| median | | **0.754** | | 10.5% | 1.02 | 0.39 | 0.687 |
| *per-step union, for reference* | | *0.568* | | *—* | *1.00* | *0.40* | *0.723* |

### 1. Saturation is a deterministic function of chain length

Spearman between observe steps per completion and the saturation rate: **+0.991** over
the 11 checkpoints (and +0.991 with coverage). That is not a correlation to be argued
about; merging only ever grows the mask, and the per-step unions being merged are the
*least* alike masks in the corpus — two steps of one chain sit at closeness 0.614 against
0.842 for two chains' first steps ([step-box-similarity.md](step-box-similarity.md)) — so
every extra step adds genuinely new area.

This is the risk that decides the arm, and it is a moving one. The overlap reward is known
to lengthen chains: the wov0.4 / set_a run went from 163 to 356 mean completion tokens and
0.00 to 0.19 duplicate-sentence fraction over steps 1000–2000. On this arm that drift does
not merely dilute the reward, it **switches it off**, completion by completion, exactly
for the completions that ramble. The reward is masked (neutral), not zeroed, so nothing in
the loss says so — `mask/merged_unscored_frac` is the only thing that will.

The cliff has a ramp in front of it, and the ramp points the wrong way. Under `mean_in` a
growing union *raises* the score (r +0.17 with the area fraction), so adding observe steps
pays right up until the union hits the grid and the completion drops out.

### 2. `--max_union_area` cannot be sized to fix it

Share of completions a cap would drop (on top of the saturation already lost):

| checkpoint | median cover | cap 0.7 | cap 0.8 | cap 0.9 | already ==1.0 |
|---|---|---|---|---|---|
| `base_coldstart` | 0.773 | 68.0% | 45.9% | 26.4% | 6.5% |
| `mean_in_saliency_r1_8k` | 0.750 | 54.3% | 37.4% | 19.2% | 2.7% |
| `mean_in_v2_set_a_1500` | 1.000 | 92.9% | 84.5% | 73.1% | 50.4% |
| `auroc_set_a_2500` | 0.500 | 12.9% | 3.4% | 0.0% | 0.0% |

The same objection the per-step path already has, one size up. A cold-start run capped at
0.9 starts with a quarter of its completions unscored; at 0.8, nearly half. The cap
*bounds* the damage, it does not remove it, and every value of it changes which
completions are scored — so it is a second experimental variable, not a safety net.

### 3. What the arm keeps, and it is more than the other arms keep

- **`w_overlap` transfers.** Median within-group sd ratio 1.02, matched weight 0.39 —
  the incumbent's 0.4, unchanged. This is the only mask source measured for which that is
  true; every other arm needs its own weight (`chain_last` 0.32, `qbox` 0.55,
  `rect_centre` 0.60). Read it as ±25% like all the others.
- **It is not a flatness run in disguise.** r with the box-blind `flatness` statistic
  0.687, *below* the per-step union's own 0.723, and below its own per-step reference in
  8 of the 11 checkpoints. Compare the fixed-mask arms at 0.89–0.93.
- **It reproduces the incumbent's ranking more closely than any other arm.** Correlation
  with the per-step reward, group-centred: 0.827, against `chain_last` 0.705, `qbox`
  0.727, `rect_centre` 0.710. GRPO only ever sees that ranking, so this cuts both ways —
  it is the least confounded comparison available *and* the least likely to land
  anywhere different.

### 4. The one thing that gets worse

Ring coverage — the share of the grid's one-patch border the mask takes — rises from the
per-step union's median **0.403** to **0.569**, and the mask share on the ring from 0.189
to 0.216. The border is 30% of a 10x16 grid, holds 48–52% of the attention mass and 76–85%
of map peaks, and `mean_in` divides by that peak. A mask that reaches further into the
sink scores more of the sink against itself. This is the dimension the interior rectangle
arms were built to hold at zero, and this arm moves it in the wrong direction.

## Running it

```fish
bash launch_grpo_qwen3_overlap_colocated_job.sh \
    --saliency-method attention --overlap-metric mean_in \
    --merge-boxes --w-overlap 0.4 \
    --num-gpus 8 --lora-targets q_proj,v_proj \
    --dataset_name peterant330/saliency-r1-8k
```

`--w-overlap 0.4` is the measured match, not the launcher's 0.2 default, so it has to be
passed. The detector is still needed, so the GPU layout is a normal DINO run's. The run
name gets `_mergebox`, so it can never share a checkpoint directory or a wandb run with a
per-step one.

Consider `--max-union-area 0.9` with eyes open about §2: it converts a silent late
collapse into a known up-front 26% masking rate, which is the more readable failure but a
different experiment from the uncapped arm.

**What to watch, in this order.** `mask/merged_cover` and `mask/merged_unscored_frac`
first, then the reward. If `merged_unscored_frac` climbs past ~0.3 the arm has stopped
being the experiment it was launched as — most of the batch is neutral on the overlap
dimension, and the runs it is compared against are scoring completions it is not.
`mask/union_frac` and `mask/ring_frac` are over the completions that survived, so they
read low exactly when things are going wrong; that is why `merged_cover` is recorded
before the refusal and is the one to read.

## Pre-registered

- **`merge_boxes` ≈ the per-step reference on both benchmark suites** ⇒ the per-step
  *target* was not buying anything either, and with `chain_last` (the per-step *call*) that
  closes both halves of the granularity question at once.
- **`merge_boxes` > the reference on natural** ⇒ the chain-wide union is the better target
  and the per-step mask was too tight — the first positive result any mask-source arm
  would have produced.
- **`merge_boxes` < the reference** ⇒ read `merged_unscored_frac` before concluding
  anything about targets. Below ~0.1 it is a result about the mask; above ~0.3 it is a
  result about how much of the batch the arm stopped scoring, and the two are not
  distinguishable after the fact.

Traps carried over from [next-reward-experiments.md](next-reward-experiments.md): the
benchmark cannot resolve differences below ~0.013 (the measured seed variance between two
runs of one identical config), and the comparator is Uri's `--w-overlap 0` control, not
`baseline/grpo-no-saliency`, which starts from a different model.

## Caveats

- **One corpus.** Every number is `val_natural`, 30 images x 8 generations per checkpoint.
  The 11 checkpoints are one cold start plus ten of its descendants, so "8 of 11" is a
  consistency statement across a training trajectory, not eleven independent replications.
- **Steps DINO could not ground are absent from the probe**, so the coverage and
  saturation numbers are over completions with at least one grounded step. That is the
  right denominator for what this arm loses *extra*, but it means the total unscored rate
  in a run will be a little higher than `sat%` here.
- **The maps are quantised** to 1/255 of their own peak; the probe reports
  `|recomputed - stored|`, which is ≤0.0011 against values of ~0.04, i.e. pure rounding.
- **Nothing here is a benchmark result.** It is all about the reward's shape.
