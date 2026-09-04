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

## The two flags

```fish
# one DINO call per completion, on its LAST observe step
bash launch_grpo_qwen3_overlap_colocated_job.sh --chain-boxes last --w-overlap 0.32

# a detector-free rectangle that moves per completion, never onto the border
bash launch_grpo_qwen3_overlap_colocated_job.sh \
    --overlap-rect-frac 0.565 --rect-placement interior_hash --w-overlap 0.37
# ... and its matched control, identical in every way but the placement
bash launch_grpo_qwen3_overlap_colocated_job.sh \
    --overlap-rect-frac 0.565 --rect-placement interior_centre --w-overlap 0.45
```

Weights are the cold-start match (`0.4 x sd_within(per-step) / sd_within(scheme)`), which is
the convention every other arm's weight was set by; brackets over the 11 checkpoints are
[0.19, 0.34] for `chain last`, [0.37, 0.58] for `interior_hash`, [0.42, 0.73] for
`interior_centre`. Read them as ±25%, as everywhere else.

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
- `interior_hash` > `interior_centre` on natural ⇒ per-completion mask variation is worth
  something even with no detector at all, and the fixed-mask arms were handicapped.
- `interior_hash` ≈ `interior_centre` ⇒ variation is not the missing ingredient, and the
  running `rect_frac` arm can be read at face value after all.

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
