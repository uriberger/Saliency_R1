# A mask per completion — 2026-09-04

Two new arms, `--overlap_chain_boxes` and `--overlap_rect_placement`, and the measurement
that says why the two arms already running (`--overlap_question_boxes`, `--overlap_rect_frac`)
cannot answer the question they were launched for on their own.

Tool: `mask_variance_probe.py`. CPU only, seconds, no GPU and no Grounding-DINO — it reads
the maps `overlap_probe.py --store-maps` already wrote and builds every mask by importing
`trl/rewards/overlap_rewards.py`, so a number here is a number about the code a run will
execute. Report: `outputs/mask_variance/per_completion/val_natural.txt`.

```fish
set PY /home/uberger/scratch/miniconda3/envs/saliency_r1_qwen3_vllm/bin/python
$PY mask_variance_probe.py \
    outputs/overlap_probe/20260809-021810-crossrun-val_natural-plus/probe_merged.json \
    --question-boxes outputs/question_boxes/val_natural_bt0.10.json
```

## The question the running arms cannot answer

GRPO subtracts the group mean before anything else. The only thing a reward contributes is
how it varies **between the 8 rollouts of one prompt**. So a mask that is the same for all
8 cannot contribute *through the mask* at all — it cancels, and what survives is a property
of the map.

`--overlap_question_boxes` (one union per row) and `--overlap_rect_frac` (a fixed
rectangle) both have that property. 11 checkpoints, identical generations, only the mask
varied; group-centred Pearson against `flatness = mean(m)/max(m)`, the statistic
`--maskfree flatness` rewards, and the pooled within-group sd the trainer logs as
`rewards/<f>/within_group_std`:

| mask | r with flatness | sd_within vs per-step DINO | median w to match at w_ref 0.4 |
|---|---|---|---|
| per-step DINO union (incumbent) | 0.723 | 1.00 | 0.40 |
| `--overlap_question_boxes` | 0.900 | 0.73 | 0.55 |
| `--overlap_rect_frac`, centred | 0.932 | 0.67 | 0.60 |
| **`--overlap_chain_boxes last`** | **0.498** | **1.45** | 0.28 |
| `--overlap_chain_boxes first` | 0.562 | 1.14 | 0.35 |
| **`--overlap_rect_placement interior_hash`** | **0.709** | 0.85 | 0.47 |
| `--overlap_rect_placement interior_centre` | 0.891 | 0.69 | 0.58 |
| whole grid (`flatness`) | 1.000 | 0.75 | 0.53 |

Medians over the 11 checkpoints. Every fixed-mask arm sits at 0.89–0.93 with the box-blind
statistic; the per-completion arms pull back to the per-step union's own 0.72 or below.
`chain_boxes` is the only source that *raises* the spread rather than lowering it (11 of 11
checkpoints above 1.0 for `last`, 10 of 11 for `first`).

**This does not say the running arms are worthless.** It says a null from either of them is
not evidence that the detector is unnecessary — it is also consistent with "the contrast
GRPO saw was mostly box-blind". The two new arms separate those.

## Why `last`, not `first`

The obvious way to ground once per completion is on its first observe step. It is the worst
single choice available. Mask closeness between chains of the same image
(`step_box_similarity.py`'s measure, so mask size is already controlled for), median over
the 11 checkpoints:

| pairing | closeness |
|---|---|
| two steps of ONE chain | 0.614 |
| **first step vs first step, different chains** | **0.842** |
| random step vs random step | 0.754 |
| last step vs last step | 0.699 |

Two chains' opening sentences produce masks that are more alike than two steps of one chain
do. Pairwise, first beats last in 9 of the 11 checkpoints. Grounding on the first step hands
a prompt's 8 rollouts the most nearly identical masks available, which is the opposite of
what a per-completion mask is for. `first` is kept as a flag value only because it is the
naive choice and running it costs nothing extra.

## Why the interior, and not just any offset

This is the part that reversed a design. The detector-free way to give each completion its
own mask is to move the rectangle, and the obvious move is a random in-frame offset. That is
wrong, and the reason is the edge sink.

The border of the patch grid is one patch wide — 48 of 160 patches on a 10x16 grid, **30%** —
and it carries **48–52% of the attention mass**, 2.6–3.1x the interior's per-patch density,
with **76–85% of map peaks** on it. `mean_in` divides by that peak. So where a mask sits
relative to the border is not a detail of the design; it is most of what the reward is
measuring.

Geometry first, exact and needing no sampling. A rectangle covering 0.565 of a 10x16 grid
rounds to 8x12 and has **15** in-frame placements:

| | border patches covered | interior patches covered |
|---|---|---|
| centred (the incumbent) | **0.000** | 0.857 |
| averaged over all 15 in-frame placements | 0.228 | 0.760 |
| whole grid (`flatness`) | 1.000 | 1.000 |

So an in-frame offset **does** reach the border — only 21% of draws miss it entirely — and
averaging over placements converges not to `flatness` but to a centre-weighted mask with the
border discounted about fourfold.

And it costs the mechanism. Group-centred correlation of the reward with a completion's own
border mass (more negative = the reward rewards getting mass off the border), median over
the 11 checkpoints, with the pairwise counts because the ranges overlap:

| mask | border covered | r(reward, border mass) |
|---|---|---|
| `interior_centre` | 0.000 | −0.714 |
| centred rectangle | 0.000 | −0.684 |
| **`interior_hash`** | **0.000** | **−0.605** |
| `--overlap_question_boxes` | 0.394 | −0.555 |
| whole grid (`flatness`) | 1.000 | −0.443 |
| per-step DINO union | 0.26–0.53 | −0.426 |
| **any in-frame offset (rejected)** | **0.21** | **−0.408** |

- an in-frame offset tracks border mass **less** strongly than `flatness` in **10 of 11**
  checkpoints, and less than the DINO union in 8 of 11. It dilutes the mechanism it was
  meant to preserve.
- the interior draw tracks it **more** strongly than `flatness` in **8 of 11** and more than
  the DINO union in **11 of 11**.

Restricting the draw to placements that touch no border patch holds coverage at exactly 0
for every completion, so ring coverage stops being a variable and only placement moves.

What that buys and costs, all pairwise against its own centred control `interior_centre`:

- within-group sd rises in **11 of 11** (median ratio 0.85 vs 0.69)
- correlation with `flatness` falls in **11 of 11** (0.709 vs 0.891) — down to the per-step
  DINO union's own 0.723
- the border term weakens in **11 of 11** (−0.605 vs −0.714). That is the price of the
  position varying, and it is the number to watch if the arm underperforms.

## The two flags, and only two runs

```fish
# one DINO call per completion, on its LAST observe step
bash launch_grpo_qwen3_overlap_colocated_job.sh \
    --saliency-method attention --overlap-metric mean_in \
    --chain-boxes last --w-overlap 0.32 \
    --num-gpus 8 --lora-targets q_proj,v_proj \
    --dataset_name peterant330/saliency-r1-8k

# a detector-free rectangle that moves per completion, never onto the border
bash launch_grpo_qwen3_overlap_colocated_job.sh \
    --saliency-method attention --overlap-metric mean_in \
    --overlap-rect-frac 0.565 --rect-placement interior_hash --w-overlap 0.32 \
    --num-gpus 7 --lora-targets q_proj,v_proj \
    --dataset_name peterant330/saliency-r1-8k
```

`--num-gpus` differs because only the first needs the detector; both give 6 training procs
and `gen_batch` 48. `--lora-targets q_proj,v_proj` matters — the launcher's default is
`q_proj,k_proj,v_proj`, a different experiment that adds `_loraqkv` to the name.
`--w-overlap` must be passed: the launcher's default is 0.2 and `mean_in` does not override
it, which is why the reference runs pass 0.4 by hand.

### `interior_centre` is NOT the control to run — 2026-09-04

The first draft of this page called for a third run, `--rect-placement interior_centre`, as
the matched control for `interior_hash`: same area, same zero border coverage, placement
fixed instead of per completion. **Do not run it.** The already-running `--overlap-rect-frac
0.565` arm does that job, and the residual confound is measured and small.

`interior_hash` differs from that running arm in two things — mask AREA (0.412 of the grid
against 0.600) and PLACEMENT. Separated pairwise over the 11 checkpoints:

| quantity | AREA only (0.600 → 0.412) | PLACEMENT only | ratio |
|---|---|---|---|
| within-group sd | +0.010 [−0.040, +0.050] | **+0.180** [+0.080, +0.240] | 18x |
| r with `flatness` | −0.041 [−0.056, −0.017] | **−0.178** [−0.296, −0.129] | 4.3x |
| r with border mass | −0.021 [−0.050, +0.053] | **+0.136** [+0.055, +0.207] | 6.5x |
| border covered | 0.000 → 0.000 | 0.000 → 0.000 | — |

The placement effect moves the same way in 11 of 11 checkpoints on all three rows. The area
effect is 4–18x smaller and is not even sign-consistent on the sd (7 of 11). The last row is
the one that matters most for this page's argument: the CENTRED rectangle already has zero
border coverage, so the edge-sink dimension is matched between the two arms for free.

It could not have been designed away in any case. On a 10x16 grid the interior is 8x14 =
112 patches and the running rectangle is 8x12 = 96. It fits — with 1 x 3 = 3 placements and
no vertical freedom at all. Same area *and* room to move does not exist on this grid, which
is why the fraction is read against the interior.

Recompute the table from the report with `rect_centre` (the running arm) and `rect_inctr`
(area changed, placement not) as the two rungs; both rows are already in it.

### Weights

`chain last` is calibrated the family's usual way — the cold-start
`0.4 x sd_within(per-step DINO) / sd_within(scheme)`, so it is pressure-matched to
`mean_in w0.4` like every other arm. 0.32, bracket [0.19, 0.34] over the 11 checkpoints.

`interior_hash` is **not**, deliberately: its comparator is the running rectangle arm, not
`mean_in`, so it is matched to that instead —
`0.4 x sd_within(rect_centre) / sd_within(interior_hash)`. That also happens to be 0.32, and
it is a much steadier number than the `mean_in` match because it is the same statistic on
two similar masks:

| calibrated against | cold start | range over 11 checkpoints |
|---|---|---|
| the running `rect_frac` arm (**use this**) | 0.32 | 0.28 – 0.36 |
| `mean_in` w0.4, the family convention | 0.37 | 0.37 – 0.58 |

Read either as ±25%, as everywhere else. If a third run ever happens and the comparison is
against `mean_in` rather than against the rectangle, use 0.37 and say so in the run name.

Details that will otherwise be discovered the hard way:

- **`--chain-boxes` still needs the detector.** It is the only one of the four mask sources
  that does. The sidecar stays up and the GPU layout is a normal DINO run's; what falls is
  the number of calls, from the completion's step count (2.1 on the trained 8k policy, 3.7 at
  the cold start) to one.
- **A completion whose chosen step grounds nothing is unscored as a whole**, masked not
  zeroed, with no fallback — a fallback would cost a second call and make the cost claim
  untrue. `mask/chain_ungrounded_frac` logs the rate; read it before reading the reward.
  `--max_union_area` becomes per completion for the same reason.
- **The rectangle's fraction is read against the INTERIOR** under both interior modes, so at
  `--overlap-rect-frac 0.565` the mask is 6x11 = **0.412** of a 10x16 grid, not the centred
  8x12 = 0.600. That is what leaves it room to move.
- **12 placements on the modal grid.** The per-completion mask takes about a dozen distinct
  values, which bounds how much variation this can restore. `mask/n_placements` logs it.
- `--rect-placement centre` is byte-identical to the incumbent, and no existing command line
  or checkpoint directory moves.

## What to log, and what would falsify each arm

New `mask/*` metrics, live: `ring_frac` (share of the scored mask on the border — the
contract is 0.000 for both interior modes, and it drifting off zero means the arm is no
longer the control it is named for), `union_frac`, `n_placements`, `chain_ungrounded_frac`.

Pre-registered:

- `chain last` ≈ the per-step reference on both benchmark suites ⇒ the per-step call is
  decorative but the per-**completion** one is not, and DINO calls can drop 2–4x for free.
- `chain last` ≈ `question_boxes` (i.e. both below the reference) ⇒ granularity is not what
  matters and the ladder ends; look elsewhere.
- `interior_hash` > the running `rect_frac` arm on natural ⇒ per-completion mask variation
  is worth something even with no detector at all, and every fixed-mask arm was handicapped.
- `interior_hash` ≈ `rect_frac` ⇒ variation is not the missing ingredient, and `rect_frac`
  can be read at face value after all.

The second reading has one escape hatch, and it is the only thing `interior_centre` would
be for: a difference SMALLER than the ~0.013 seed floor cannot be told apart from the
uncontrolled area change (0.600 → 0.412 of the grid). The table above bounds that channel at
4–18x below the placement channel, which is why it is a bound rather than a control — so if
the two arms land within noise of each other, run `interior_centre` at w 0.45 before
concluding that placement does nothing. If they land far apart, do not bother: the area
channel is too small to have carried it.

Traps carried over from [next-reward-experiments.md](next-reward-experiments.md): the
benchmark cannot resolve differences below ~0.013 (the measured seed variance between two
runs of one identical config), so do not report a ranking without a stated uncertainty; and
compare against Uri's `--w-overlap 0` control, not `baseline/grpo-no-saliency`, which starts
from a different model.

## Caveats

- **One corpus.** Every number above is `val_natural`, 30 images x 8 generations per
  checkpoint. The 11 checkpoints are not independent — they are one cold start plus ten of
  its descendants — so "11 of 11" is a consistency statement across a training trajectory,
  not eleven independent replications.
- **Ranges overlap; the pairwise counts are the claim.** The three collapsed set_a
  checkpoints sit at one end of nearly every column, which is why medians and per-checkpoint
  pairings are reported rather than min–max alone.
- **The maps are quantised** to 1/255 of their own peak. Every scheme's `true` value was
  recomputed from the stored bytes and compared with what the probe wrote: largest
  disagreement 0.00093 against a value of ~0.04, i.e. pure rounding.
- **Steps DINO could not ground are absent** from the probe, so `chain_ungrounded_frac`
  cannot be predicted from it — it has to be read off the run.
- **Nothing here is a benchmark result.** It is all about the reward's shape. Whether any of
  it moves the model is what the runs are for.
