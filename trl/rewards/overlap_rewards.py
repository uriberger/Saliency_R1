# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Attention-overlap reward (reward_variant="ours"), the flag-selectable alternative
to Saliency-R1's think_saliency_reward.

The trainer (grpo_trainer_qwen3.py, reward_variant="ours" branch) does the attention
surgery and hands this reward, per completion, a list of per-observe-step saliency
maps + the step text:

    saliency_map[c] = [{"map": np.ndarray (grid_h, grid_w) float32, "text": str}, ...]

Each map is raw observe-token -> image-patch attention at LAYER 22, mean of the
configured heads (default (22,28)+(22,31)), ReLU, token-reduced over the step's tokens.
This reward grounds each step's text with Grounding-DINO (per step, in the loop),
builds the union mask of boxes >= box_threshold with area <= max_box_area (and, with
--max_union_area, drops steps whose union covers too much of the image -- see the
coverage section below), and scores
each step with one of three metrics (--overlap_metric), then averages over the
completion's grounded observe steps (steps DINO can't ground are SKIPPED, not scored
0). The result is gated by format validity (multiplicative, like their valid_list).
Zero grounded observe steps -> None (masked, neutral in the GRPO advantage — NaN is
nan-summed out).

  mean_in  (DEFAULT, the incumbent)
      mean over the box of the MAX-normalized (/max -> [0,1]) map. Mirrors the offline
      reference in the vlm_reasoning repo (analysis/aggregation_correlation.py:
      _score_saliency_flat) so the online reward matches the fit it was validated on.

      KNOWN WEAKNESS: it divides by the map's own PEAK, not by its surroundings, so a
      map that merely FLATTENS scores higher inside the box without attending it any
      better. Measured offline, mean_in moves 32x further under pure flattening than
      under a genuine 5% transfer of attention mass into the box, and the wov0.2 /
      wov0.4 runs did exactly that: their MMStar gain disappears once chance-corrected.
      See wiki/lmms-eval-overlap-comparison.md in the vlm_reasoning repo.

      CONFIRMED ONLINE: the wov0.4 / trmean / 50k set_a run took this route at ~step
      1200. It doubled the overlap reward (0.044 -> 0.085) by repeating observe steps,
      and its cp_2000 scores BELOW the cold-start parent on the benchmarks (p3 60.7 ->
      34.3, i.e. chance; mme 715 -> 642) while the training accuracy reward barely moved
      -- the hack costs 0.006 accuracy on set_a and lives inside <think>, which
      accuracy_reward never parses. See wiki/overlap-reward-hack-set-a.md (same repo)
      for the full mechanism, and note the enabling condition is not the metric alone:
      with scale_rewards=True, groups whose accuracy has saturated hand the overlap term
      the entire, std-renormalized advantage (75% of groups by step 1500).

  mean_in_v2  (--overlap_metric mean_in_v2)
      mean over the box divided by the mean over the WHOLE map, i.e. the same numerator
      as mean_in but normalized by the map's average instead of its peak. Chance = 1.0
      (a map with no preference for the box scores 1.0); the ceiling is
      n_patches / n_in_box, so the scale depends on how large the box union is.

      It closes mean_in's flattening hole from the other side than auroc does: both the
      numerator and the denominator are means, so any rescale m -> c*m cancels exactly,
      and a uniform flattening moves the score TOWARD 1.0 rather than up. Unlike auroc
      it still sees magnitudes, so a map that concentrates more mass in the box (not
      just ranks it higher) is rewarded for it.

      MEASURED (overlap_metric_spread.py over 1074 grounded steps of the cold-start
      policy on set_a, 40 samples x 8 generations):

        range      p10 0.41 / median 0.74 / p99 1.36 / max 2.33. Unbounded in principle,
                   but the median box union covers 56% of the image, which puts the
                   ceiling n/k at ~1.8 -- no clamp is needed in practice.
        spread     per-sample sd 0.105, i.e. 12x mean_in's 0.0086, so w_overlap 0.033
                   reproduces mean_in's wov0.4 pressure (the launchers apply this).
                   The same script re-derives auroc's documented 0.11 as 0.089, so
                   treat 0.033 as +-25%.
        box size   r +0.38 with the box area fraction, vs +0.17 for mean_in and -0.11
                   for auroc. Growing the box union does raise the score while the map
                   is BELOW chance, but that pull dies at 1.0 instead of diverging: full
                   coverage gives exactly 1.0. (Dividing by the outside SUM rather than
                   the overall mean removes that limit -- it equals
                   (mean_in/mean_out)/(n-k), which diverges as the union grows and has
                   no fixed chance level. That is why the denominator is the whole map.)

      Not covered by the offline attack/utility screen that produced the mean_in and
      auroc correlation numbers below.

  auroc  (--overlap_metric auroc)
      P(a random in-box patch outranks a random out-box patch), average ranks for ties.
      Chance = 0.5. Depends only on the ORDER of the patches, so it is exactly
      invariant to m -> m**gamma: the flattening route is closed by construction, not
      by tuning. Scored 0.00 on every reshaping attack in the offline simulation, and
      predicts correctness more stably than mean_in (mean |r| 0.238 vs 0.181 over four
      powered datasets, sd 0.028 vs 0.089, and mean_in flips sign on Visual-CoT/DINO).

      NOT immune to the union-growth hack, despite all of the above. The offline screen
      covers RESHAPING the map for a FIXED box; it says nothing about the policy changing
      the TEXT so that DINO returns a bigger box. The wov0.11 / auroc / 50k set_a run did
      exactly that: the overlap reward jumped around step 2200 and every observe step had
      become a description of the BACKGROUND, which grounds to huge boxes. Rank-invariance
      does not help when the ranking is taken over a different, much larger in-box set.
      So do not read auroc's r -0.11 with the box-area fraction (cited below) as evidence
      that it needs no union cap -- that number is from static offline collections and it
      did not predict this. See --max_union_area.

Optional mass floor (--mass_floor_tau, applies to any metric; off by default):

      score *= min(1, image_mass / tau)      image_mass = step_map.sum()

  Because attention rows are a softmax over ALL keys and only the image columns are
  kept, image_mass is the fraction of the row spent on the image. AUROC is rank-based
  and therefore blind to a model that withdraws attention from the image toward text
  tokens while keeping a good ranking; mean_in_v2 is a ratio of two means and is blind
  to it for the same reason (any rescale cancels). This floor closes that for both. It
  is not a pure guard:
  image_mass is itself predictive of correctness (r +0.22..+0.29), so the floor also
  RAISES the correlation (0.227 -> 0.238) rather than costing anything. Recommended
  tau = 0.0022, the 10th percentile of the reference model's image_mass (stable at
  0.0018-0.0029 across all seven offline collections). Keep tau near p10: much above
  p25 it stops being a floor and "raise image attention uniformly" becomes its own
  exploitable direction.

Box-coverage caps (two independent filters, both in _union_mask):

  --box_threshold (default 0.10)
      DINO confidence floor. Applied server-side too when --dino_api_base is set.

  --max_box_area (default 0.5, set to 0 to DISABLE)
      Drops any INDIVIDUAL box whose area exceeds this fraction of the image, before
      rasterisation. Note this is a per-box filter and says nothing about the union:
      ten disjoint boxes at 0.1 each pass it and cover the whole image between them.

  --max_union_area (default None = OFF)
      Skips the whole step when the RASTERISED union covers more than this fraction
      of the patch grid. Returns None from _union_mask, so the step takes the same
      path as an ungroundable one: SKIPPED, not scored 0, exactly like the existing
      degenerate-union guard (which only fires at 100% coverage).

      Why it is worth having: the per-box cap leaves the union unbounded, and the
      measured median union already covers 56% of the image (overlap_metric_spread.py,
      1074 grounded steps of the cold-start policy on set_a). A near-full union makes
      the score meaningless -- everything is "inside the box".

      CONFIRMED ONLINE, under AUROC. The wov0.11 / auroc / 50k set_a run jumped in
      overlap reward around step 2200, and its observe steps had all turned into
      descriptions of the BACKGROUND -- background phrases ground to huge boxes, and a
      huge scored region is easier to rank well against. This is the metric-INDEPENDENT
      hole: the offline area-fraction correlations (mean_in +0.17, mean_in_v2 +0.38,
      auroc -0.11) are about reshaping the map for a fixed box and did not predict it.
      auroc's negative number in particular is not protection.

      Why it is still OFF by default (2026-08-04): the right value is not yet known.
      The cold-start policy's median union is ALREADY 0.562, so any cap tight enough to
      look principled masks a large share of completions before training does anything
      -- and a step the cap drops leaves the mean entirely, so an aggressive cap turns
      the reward off for whole completions rather than merely trimming it. Size it from
      the measured distribution first: union_size_report.py over an overlap_probe run
      (which must itself be run with --max_union_area 0, or it cannot see the tail it
      is being used to measure).

      Turning it on also changes WHICH steps are scored, so it shifts the reward's
      scale -- re-check w_overlap against a probe run rather than assuming the
      incumbent weight transfers, and note the launchers add _mu<x> to the run name so
      a capped run never shares a checkpoint dir or wandb name with an uncapped one.

A FIXED RECTANGLE INSTEAD OF THE BOXES (--overlap_rect_frac, default None = OFF):

      Replaces the Grounding-DINO union with an axis-aligned rectangle in the middle
      of the patch grid, covering `frac` of it at the grid's own aspect:

          rows = round(grid_h * sqrt(frac)),  cols = round(grid_w * sqrt(frac))

      centred, clamped to at least one patch. Nothing else changes: the same
      --overlap_metric scores it, the same format gate multiplies it, the same
      --overlap_natural_only masks it, and it occupies the same reward_funcs slot, so
      --reward_weights lines up with a DINO reference run unchanged. _dino_boxes is
      never called and the detector is never constructed, so the run needs no DINO
      GPU and no DINO server.

      WHY. centre_box_probe.py (outputs/centre_box_probe/report.txt) measured what
      the boxes are worth to this reward, on the val_natural probe. Two findings,
      pointing opposite ways:

        - The rectangle is NOT DINO's mask. Given the step's own union AREA, so that
          only placement is left to differ, its closeness to that union is 0.230 --
          against a different-image floor of 0.235. It is no closer to what DINO drew
          than boxes drawn on an unrelated picture are.
        - But the reward cannot tell. The per-completion reward built on the
          rectangle reproduces the real per-step-DINO reward's ranking of a group's
          8 rollouts at rho 0.651, against 0.621 for running DINO once per chain --
          and 0.638 for NO MASK AT ALL. GRPO only ever sees that ranking.

      This flag is the training arm of that measurement: if a rectangle trains as
      well as the boxes, the detector was not buying the gradient. 0.565 is the
      fraction the probe used -- the mean union coverage DINO produces on these runs
      -- so the rectangle gives away nothing on mask SIZE and differs from its
      reference in mask PLACEMENT alone.

      Two behavioural differences to expect, both because the mask no longer depends
      on the step's sentence:

        - Nothing is ever ungroundable, so EVERY observe step is scored and no
          completion is masked for having none. The scored set is therefore larger
          than its DINO reference's -- the point, not a bug, but it does mean the two
          runs' logged reward means are over different populations of steps. (The one
          exception is a grid so coarse that the rectangle rounds up to all of it: at
          0.565 that is a 2x2 grid and nothing else, and such a step takes the same
          skipped-not-zeroed path as a degenerate union. Real patch grids are ~10x16.)
        - --box_threshold, --max_box_area and --dino_api_base are dead knobs here.
          --max_union_area is REFUSED when it would drop the rectangle: the mask is
          identical on every step, so that cap is all-or-nothing and would silently
          leave every completion unscored.

      w_overlap does not transfer from a DINO run unread -- the value distribution
      moves with the mask. Re-measure the within-group sd (overlap_metric_spread.py)
      rather than assuming the incumbent weight.

WHERE THE RECTANGLE SITS (--overlap_rect_placement, default 'centre'):

      The centred rectangle is the same for all 8 rollouts of a prompt, and GRPO
      subtracts the group mean, so a constant mask cancels out of the advantage except
      through the map. Measured on the val_natural cross-run probe (11 checkpoints,
      identical generations, only the mask varied -- mask_variance_probe.py), the
      surviving contrast is 0.90-0.97 correlated with `flatness` = mean(m)/max(m), the
      mask-free statistic --maskfree rewards, against 0.57-0.78 for the per-step DINO
      union. This flag restores a per-COMPLETION mask without a detector.

        centre           the incumbent: rows = round(grid_h * sqrt(frac)), centred on the
                         grid. Byte-identical to what --overlap_rect_frac did before this
                         flag existed, so an existing run's semantics do not move.
        interior_centre  the same construction on the INTERIOR of the grid (everything but
                         the one-patch border), sized to `frac` of the interior's area and
                         centred in it. The matched control for the mode below: same area,
                         same zero border coverage, no per-completion variation.
        interior_hash    those dims, placed at one of the strictly-interior positions,
                         chosen by blake2b(seed | completion text). Deterministic per
                         completion and stable across restarts and ranks.

      WHY THE INTERIOR, and not just any offset. The border of the patch grid is 30% of a
      10x16 grid and carries 48-52% of the attention mass -- 2.6-3.1x the interior's
      per-patch density -- and 76-85% of map peaks sit on it. `mean_in` divides by that
      peak, so where a mask sits relative to the border is not a detail. Border coverage
      (share of border patches the mask takes) and the group-centred correlation of the
      resulting reward with a completion's own border mass, over the same 11 checkpoints:

          mask                                border covered     r(reward, border mass)
          centred rectangle f=0.565               0.000            -0.32 .. -0.78
          box among INTERIOR placements           0.000            -0.21 .. -0.70
          DINO union                              0.26 .. 0.53     -0.08 .. -0.56
          rectangle at any IN-FRAME offset        0.21             -0.02 .. -0.46
          whole grid (`flatness`)                 1.000            -0.08 .. -0.57

      Ranges overlap because the collapsed set_a checkpoints sit at one end of all of
      them, so the claim is made pairwise on the same generations instead. A rectangle
      merely displaced inside the frame tracks border mass LESS strongly than `flatness`
      does in 10 of 11 checkpoints and less strongly than the DINO union in 8 of 11: it
      dilutes the mechanism it was meant to preserve, because it lets a different slice of
      the border back in on every draw. The interior draw goes the other way -- stronger
      than `flatness` in 8 of 11 and stronger than the DINO union in 11 of 11.

      What the per-completion draw costs, also pairwise:

        * against its own centred control it is weaker on the border term in 11 of 11
          (median r -0.605 vs -0.714). That is the price of the position varying.
        * it buys back spread: within-group sd rises against interior_centre in 11 of 11
          (median ratio to the per-step reward 0.85 vs 0.69), and it pulls the
          flatness correlation down in 11 of 11, from 0.891 to 0.709 -- which is the
          per-step DINO union's own 0.723.

      Two consequences worth knowing:

        * `frac` is read as a fraction of the INTERIOR under both interior_* modes, so
          the mask is smaller than the same `frac` gives under 'centre' -- 6x11 = 0.412
          of a 10x16 grid at frac 0.565, against the centred 8x12 = 0.600. That is what
          makes it fit with room to move. Re-measure w_overlap, against whichever run is
          the comparator: matched to a CENTRED --overlap_rect_frac arm the cold-start
          weights are 0.32 for interior_hash and 0.45 for interior_centre;
          matched to mean_in w0.4 they are 0.37 and 0.45. The first is the steadier
          number -- 0.28-0.36 over the 11 checkpoints against 0.37-0.58 -- because it is
          the same statistic on two similar masks. docs/per-completion-masks.md says why
          the centred arm, not interior_centre, is the control worth spending a GPU on.
        * a coarse grid leaves few positions -- 12 on the modal 10x16. The per-completion
          mask takes about a dozen distinct values, which is a real limit on how much
          variation this can restore, and `mask/n_placements` logs it.

      On a grid with no interior at all (grid_h < 3 or grid_w < 3) both interior modes
      return None, and the step takes the same skipped-not-zeroed path as a degenerate
      union.

There is deliberately NO step-count term. The observe-step count carries essentially
no correctness signal (r -0.004..-0.022), so an anti-brevity multiplier costs 24% of
the reward's predictive value and a hard gate costs 50-70%, to close a step-dropping
hole that is already ~5x smaller under auroc (0.20) than under mean_in (1.07). Monitor
the observe-step count as a training diagnostic instead.

The hole that actually opened online was the opposite one: DUPLICATED steps. The score
is a mean over grounded steps, so re-quoting one trivially-groundable generic sentence
pulls it up and dilutes the genuine, hard perception steps -- which is what the wov0.4 /
set_a run learned (duplicate-sentence fraction 0.00 -> 0.19 over steps 1000-2000, mean
completion length 163 -> 356 tokens). Log the duplicate fraction alongside the step
count, and consider deduping identical steps before the mean.
See wiki/overlap-reward-hack-set-a.md.

Natural-images-only gating (--overlap_natural_only, OFF by default):

      With a mixed corpus (cold_data/grpo_sets/set_b = 80% natural + 20% charts /
      documents / diagrams), Grounding-DINO is being asked to localise phrases on
      imagery it was never trained for, so on the non-natural rows the box union --
      and therefore the whole overlap score -- is noise. Turning this on returns
      None for every row whose `natural` column is False: those rows keep exactly
      the other three rewards (format, accuracy, judge) and contribute nothing to
      the overlap term. It is a masking, not a zeroing: a zero would be identical
      for the advantage (a per-group constant cancels in reward - group_mean) but
      would drag the logged rewards/think_overlap_reward/mean down with rows the
      reward was never evaluated on.

One grounding per COMPLETION instead of one per step (--overlap_chain_boxes, OFF by
default):

      The middle rung of the ladder that --overlap_question_boxes ends: ground once per
      step (incumbent), once per COMPLETION (here), once per ROW (question boxes), or
      not at all (--overlap_rect_frac). Grounding-DINO is called on ONE observe step's
      sentence per completion and the union it returns is scored against every step of
      that completion. Detector calls fall from the completion's step count -- 2.1 on
      the trained 8k policy, 3.7 at the cold start -- to exactly one.

      WHICH STEP, and why not the first. Mask closeness between chains of the same image
      (step_box_similarity.py's measure, so mask size is already controlled for), median
      over the 11 checkpoints of the val_natural probe:

          two steps of ONE chain                          0.614
          first step vs first step, different chains      0.842   <- the most alike
          random step vs random step                      0.754
          last step vs last step                          0.699

      The opening sentence of a chain is the most stereotyped thing in it -- two chains'
      first steps are more alike than two steps of one chain -- so grounding on it gives
      the 8 rollouts of a prompt the most nearly identical masks available, which is the
      opposite of what this flag is for. It holds pairwise: first beats last in 9 of the
      11 checkpoints. 'last' is the default for that reason. 'first' is kept because it is
      the naive choice and the comparison is cheap.

      Consequences:

        * WHICH completions are scored changes, at completion granularity. If the chosen
          step grounds nothing, the whole completion is unscored (masked, not zeroed) --
          where per-step grounding would have skipped that one step and kept the rest.
          There is deliberately NO fallback to another step: a fallback costs a second
          detector call and would make "one call per completion" untrue. The rate is
          logged as `mask/chain_ungrounded_frac`; read it before reading the reward.
        * --max_union_area now applies per completion, not per step, for the same reason.
        * The step-duplication hack loses its lever on the mask -- repeating a
          trivially-groundable sentence no longer changes which boxes are used -- but a
          duplicated step still enters the per-completion mean, exactly as under
          --overlap_question_boxes.
        * Unlike the two flags around it this one still needs Grounding-DINO, so the
          sidecar stays up. It is cheaper, not detector-free.

      This is the only mask source measured that RAISES the reward's within-group spread
      rather than lowering it: median ratio to the per-step reward 1.45 for 'last' (11 of
      11 checkpoints above 1) and 1.14 for 'first' (10 of 11), where every fixed-mask arm
      sits at 0.67-0.75. Correlation with `flatness` falls to 0.498 / 0.562 against the
      per-step union's own 0.723 -- the fixed-mask arms are at 0.89-0.93. w_overlap needs
      re-measuring: 0.4 x sd(per-step)/sd(scheme) puts 'last' at 0.32 on the cold start,
      bracketed by [0.19, 0.34] over the 11 checkpoints.

One grounding per QUESTION instead of one per step (--overlap_question_boxes, OFF by
default):

      Normally this reward calls Grounding-DINO once per observe step, on that step's
      own sentence. dino_text_sensitivity.py measured what that buys on the cold-start
      policy (30 images, 859 steps): re-grounding a step with the sample's QUESTION
      instead of its own sentence recovers the step's real mask at IoU 0.649 /
      closeness 0.785, against 0.635 / 0.721 for a DIFFERENT REAL STEP OF THE SAME
      CHAIN -- at the same mask size (median union 0.568 vs the real 0.578). Grounding
      on the single word "object" scores 0.636 / 0.740, also at the per-step level. The
      per-step call is therefore not buying a per-step mask.

      With this flag the reward reads one box list per dataset ROW from a file built
      ahead of the run by precompute_question_boxes.py, and scores every step of that
      row's completions against that one union. Consequences worth knowing:

        * Grounding-DINO is never loaded. No detector on the training device, no
          per-batch grounding call.
        * WHICH steps are scored changes. Per-step grounding skips (not zeroes) a step
          whose own sentence grounds nothing; here the question either grounds for the
          whole row or for none of it, so a row is scored on all of its steps or masked
          entirely. --max_union_area likewise now applies per row, not per step.
        * The step-duplication hack loses its lever on the mask, though not on the mean:
          repeating a trivially-groundable sentence no longer changes which boxes are
          used, but a duplicated step still enters the per-completion mean.
        * The cache stores RAW boxes, before --max_box_area, so the two area caps stay
          run-time knobs. --box_threshold is baked in by DINO and cannot be, so the
          loader refuses a cache built at a different threshold.

w_overlap is applied by the trainer via --reward_weights, not here.
"""

from __future__ import annotations

import contextlib
import hashlib
import math
import os

import numpy as np

from . import roll_null as _RN

GROUNDING_DINO_HF_ID = "IDEA-Research/grounding-dino-base"


@contextlib.contextmanager
def _no_deepspeed_zero3_init():
    """Temporarily hide HF's global ZeRO-3 config from ``from_pretrained``.

    The trainer runs under DeepSpeed ZeRO-3 (accelerate ``zero3_init_flag: true``),
    which registers a process-global HfDeepSpeedConfig. Every subsequent
    ``from_pretrained`` — including this auxiliary Grounding-DINO model — would then
    be wrapped in ``deepspeed.zero.Init`` and have its parameters partitioned into
    1-D shards, so a sharded weight is no longer 2-D and the forward pass raises
    ``RuntimeError: 'weight' must be 2-D``. DINO is a small, frozen, single-device
    model that must be fully materialised, so we null the weakref for the duration
    of the load and restore it (no-op if transformers lacks deepspeed integration).
    Mirrors overlap_steps._no_deepspeed_zero3_init (kept local to avoid importing
    the trainer package from the rewards package).
    """
    try:
        import transformers.integrations.deepspeed as _ds
    except Exception:
        yield
        return
    saved = getattr(_ds, "_hf_deepspeed_config_weak_ref", None)
    _ds._hf_deepspeed_config_weak_ref = None
    try:
        yield
    finally:
        _ds._hf_deepspeed_config_weak_ref = saved

# Config, set by grpo_vlm_qwen3.py via configure() from the CLI flags. box_threshold /
# max_box_area default to the flagship offline filter (honest |r|~0.22 combo).
_CFG = {
    "box_threshold": 0.10,
    "max_box_area": 0.5,     # per-box area cap; None or <= 0 disables it
    "max_union_area": None,  # per-step union coverage cap; None or <= 0 disables it
    # "mean_in" (incumbent, default) | "mean_in_v2" (/mean instead of /max) | "auroc"
    # | "logratio" (the roll-null, chance 0 -- see roll_null.py and the knobs below)
    "metric": "mean_in",
    # Roll-null knobs. Read ONLY when metric == "logratio"; the gradient reward keeps its
    # own copies under --grad_* because there the roll-null is the reward, not a choice.
    "null_offsets": 16,      # K translates of the union forming the null
    "logratio_clip": 1.0,    # +-c on log(N(U)/N_0); 1.0 == a ratio of e ~ 2.7
    "inframe_rolls": True,   # keep the translate inside the grid (no border wrap)
    "min_inframe": 4,        # below this many in-frame offsets, fall back to toroidal
    "roll_seed": 0,
    "mass_floor_tau": None,  # None/0 disables the image-mass floor; recommended 0.0022
    "dino_api_base": None,   # if set, hit a served batched DINO endpoint; else local
    "dino_device": None,     # local device override; default cuda if available
    "dino_batch_size": 32,
    "natural_only": False,   # True -> mask (None) the reward on rows with natural=False
    # None or <= 0 -> score the DINO union (the incumbent). A fraction in (0, 1] ->
    # score a centred rectangle of that area instead, and never call DINO at all.
    "rect_frac": None,
    # Where that rectangle sits: "centre" (the incumbent, unchanged), "interior_centre"
    # (same construction on the grid's interior, so it covers no ring patch) or
    # "interior_hash" (those dims at a per-COMPLETION interior position, drawn from a
    # stable hash of the completion's text). Read only when rect_frac is active.
    "rect_placement": "centre",
    "rect_seed": 0,          # mixed into the interior_hash draw; a second run, a second lottery
    # Path to a precompute_question_boxes.py file. Set -> one grounding per dataset ROW,
    # read from disk, and no DINO at all at training time. See the module docstring.
    # Mutually exclusive with rect_frac: both replace the per-step union, in the same
    # place, and _validate_rect refuses the pair.
    "question_boxes": None,
    # None (incumbent, one DINO call per observe step) | "first" | "last" -> one call per
    # COMPLETION, on that step's sentence, reused for every step of the completion. Still
    # needs the detector; see the module docstring for why "last" is the default.
    "chain_boxes": None,
}

# Lazily-loaded local Grounding-DINO singleton (one per training process).
_DINO = {"proc": None, "model": None, "device": None}


# Roll-null by-products, logged only when metric == "logratio". FIXED key set and always
# all of it, for the same NCCL reason grad_rewards.DIAG_KEYS documents: the trainer
# gathers these across ranks, and a key set that depended on what a rank happened to see
# would mean a rank-dependent number of collectives, which hangs rather than fails.
#
# `toroidal_frac` is the one to watch. It says the in-frame control pool was too small --
# a near-full-frame union -- so the null wrapped across the image border and stopped being
# the same question. It rising means the scores are no longer comparable to earlier ones.
ROLL_DIAG_KEYS = ("logratio_raw", "clip_frac", "toroidal_frac", "n_offsets",
                  "union_frac", "ecc", "n_image")
_DIAG: dict[str, list[float]] = {}
_ROLL_RNG = np.random.default_rng(0)


def _diag(key: str, value: float):
    _DIAG.setdefault(key, []).append(float(value))


# Where the MASK came from, for --overlap_rect_placement and --overlap_chain_boxes. A
# second FIXED key set, drained through its own gated block in the trainer, for the same
# NCCL reason as above.
#
# `ring_frac` is the one to watch. It is the share of the scored mask's patches that lie
# on the grid's one-patch border -- the edge sink that holds ~half the attention mass and
# 76-85% of map peaks. 0.000 is the contract for both interior placements; anything above
# it means the mask reached the sink `mean_in` divides by, and the arm has stopped being
# the control it was named for. `chain_ungrounded_frac` is the other one: it says how
# often --overlap_chain_boxes lost a whole completion because its chosen step grounded
# nothing, which the per-step path would have survived.
MASK_DIAG_KEYS = ("union_frac", "ring_frac", "n_placements", "chain_ungrounded_frac")
_MASK_DIAG: dict[str, list[float]] = {}


def _mask_diag(key: str, value: float):
    _MASK_DIAG.setdefault(key, []).append(float(value))


def mask_diag_active() -> bool:
    """True when a mask-source flag is installed, so the trainer may branch its logging
    collectives on it. Rank-uniform: it is a CLI decision made on every process."""
    return rect_active() or bool(_CFG.get("chain_boxes"))


def pop_mask_diagnostics() -> dict[str, float]:
    """Mean of each mask-source diagnostic since the last call, then clear.

    Always all of MASK_DIAG_KEYS; NaN for a key nothing was recorded under.
    """
    out = {k: (float(np.mean(_MASK_DIAG[k])) if _MASK_DIAG.get(k) else float("nan"))
           for k in MASK_DIAG_KEYS}
    _MASK_DIAG.clear()
    return out


def pop_diagnostics() -> dict[str, float]:
    """Mean of each roll-null diagnostic since the last call, then clear.

    Always all of ROLL_DIAG_KEYS; NaN for a key nothing was recorded under, including
    every key when the configured metric is not "logratio".
    """
    out = {k: (float(np.mean(_DIAG[k])) if _DIAG.get(k) else float("nan"))
           for k in ROLL_DIAG_KEYS}
    _DIAG.clear()
    return out


def configure(**kwargs):
    """Set reward config from the CLI flags. None values are ignored (keep defaults)."""
    global _ROLL_RNG
    for k, v in kwargs.items():
        if v is not None:
            _CFG[k] = v
    _ROLL_RNG = np.random.default_rng(int(_CFG["roll_seed"]))
    _validate_rect()
    _validate_chain_boxes()


def rect_active() -> bool:
    """True when the reward scores a fixed rectangle instead of Grounding-DINO boxes."""
    f = _CFG.get("rect_frac")
    return f is not None and float(f) > 0


RECT_PLACEMENTS = ("centre", "interior_centre", "interior_hash")
CHAIN_SELECTORS = ("first", "last")


def chain_boxes_active() -> bool:
    """True when Grounding-DINO is called once per COMPLETION instead of once per step."""
    return bool(_CFG.get("chain_boxes"))


def _validate_chain_boxes():
    """--overlap_chain_boxes conflicts with every other source of the mask."""
    sel = _CFG.get("chain_boxes")
    if not sel:
        return
    if sel not in CHAIN_SELECTORS:
        raise ValueError(
            f"--overlap_chain_boxes must be one of {'|'.join(CHAIN_SELECTORS)}, got {sel!r}.")
    other = ("--overlap_rect_frac" if rect_active()
             else "--overlap_question_boxes" if _CFG.get("question_boxes") else None)
    if other:
        raise ValueError(
            f"--overlap_chain_boxes {sel} and {other} both decide where the mask comes "
            "from, and only one of them can win. They are consecutive rungs of one "
            "experiment -- per step, per completion, per row, no detector -- so run them "
            "as separate runs, not as one configuration.")


def _validate_rect():
    """Refuse the --overlap_rect_frac settings that fail silently rather than loudly."""
    placement = _CFG.get("rect_placement") or "centre"
    if placement not in RECT_PLACEMENTS:
        raise ValueError(
            f"--overlap_rect_placement must be one of {'|'.join(RECT_PLACEMENTS)}, "
            f"got {placement!r}.")
    if not rect_active():
        if placement != "centre":
            raise ValueError(
                f"--overlap_rect_placement {placement} needs --overlap_rect_frac: it says "
                "where the rectangle sits, and without a fraction there is no rectangle. "
                "The flag would be silently dead.")
        return
    if _CFG.get("question_boxes"):
        raise ValueError(
            "--overlap_rect_frac and --overlap_question_boxes both REPLACE the per-step "
            "Grounding-DINO union, in the same place. Whichever won, the other arm's name "
            "would be on a run it did not describe. Pick one.")
    f = float(_CFG["rect_frac"])
    if f > 1.0:
        raise ValueError(
            f"--overlap_rect_frac must be in (0, 1], got {f}: it is the fraction of the "
            "patch grid the rectangle covers.")
    cap = _CFG.get("max_union_area")
    if cap is not None and float(cap) > 0 and f > float(cap):
        raise ValueError(
            f"--overlap_rect_frac {f} exceeds --max_union_area {cap}. The rectangle is the "
            "SAME on every step, so unlike a DINO union that cap is all-or-nothing: it "
            "would drop every step of every completion and the reward would be None "
            "everywhere, which reads in the logs as a run with no overlap signal rather "
            "than as a misconfiguration. Lower the fraction or drop the cap. (Under the "
            "interior placements `f` is a fraction of the INTERIOR, so the realised "
            "coverage is smaller and this check is conservative -- it refuses a little "
            "early rather than letting an all-or-nothing cap through.)")


# ---------------------------------------------------------------------------
# Grounding-DINO (batched)
# ---------------------------------------------------------------------------

def _load_dino_local():
    if _DINO["model"] is None:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        device = _CFG.get("dino_device") or ("cuda" if torch.cuda.is_available() else "cpu")
        proc = AutoProcessor.from_pretrained(GROUNDING_DINO_HF_ID)
        # Load fully materialised: never let DeepSpeed ZeRO-3 partition this auxiliary
        # detector (would 1-D-shard its weights -> "'weight' must be 2-D" at forward).
        with _no_deepspeed_zero3_init():
            model = AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_DINO_HF_ID).to(device).eval()
        _DINO.update(proc=proc, model=model, device=device)
    return _DINO["proc"], _DINO["model"], _DINO["device"]


def _dino_boxes_local(images, texts):
    """Batched local Grounding-DINO. Returns list (per item) of [x1,y1,x2,y2] rel boxes.

    box_threshold is applied here; area filtering is applied by the caller.
    """
    import torch

    proc, model, device = _load_dino_local()
    prompts = [(t.strip() + (".") if not t.strip().endswith(".") else t.strip()) for t in texts]
    out_boxes = [None] * len(images)

    def _run_chunk(start, n):
        imgs = images[start:start + n]
        txts = prompts[start:start + n]
        inputs = proc(
            images=imgs, text=txts, return_tensors="pt",
            padding=True, truncation=True, max_length=256,
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        target_sizes = [(im.size[1], im.size[0]) for im in imgs]  # (h, w)
        results = proc.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            threshold=float(_CFG["box_threshold"]),
            text_threshold=float(_CFG["box_threshold"]),
            target_sizes=target_sizes,
        )
        for j, res in enumerate(results):
            w, h = imgs[j].size
            boxes = []
            for box in res["boxes"].tolist():
                x1, y1, x2, y2 = box
                boxes.append([x1 / w, y1 / h, x2 / w, y2 / h])
            out_boxes[start + j] = boxes

    # Deformable attention materialises one contiguous
    # (batch, queries, heads, levels, points) tensor, so peak memory scales with the
    # batch AND with the images' native resolution: a batch that fits on one caller
    # can OOM on the next. Callers that co-reside with a big model on the same GPU
    # (the probe: 8B VLM + attention re-forward) cannot pick a batch size that is
    # both safe and fast, so halve on OOM instead of dying. A single item that still
    # OOMs is a real failure and is re-raised.
    bs = int(_CFG["dino_batch_size"])
    start = 0
    while start < len(images):
        n = min(bs, len(images) - start)
        while True:
            try:
                _run_chunk(start, n)
                break
            except torch.cuda.OutOfMemoryError:
                if n == 1:
                    raise
                torch.cuda.empty_cache()
                n = max(1, n // 2)
                print(f"[dino] CUDA OOM; retrying batch at offset {start} with size {n}",
                      flush=True)
        start += n
    return out_boxes


def _dino_boxes_served(images, texts):
    """Batched served Grounding-DINO endpoint (preferred layout: DINO on a GPU outside
    the training allocation; see grpo-reward-port-plan memory). Posts base64 images +
    texts + thresholds, expects per-item relative-coord box lists back.

    Kept minimal on purpose; the local path is the tested one. Enable by setting
    --dino_api_base (OVERLAP_DINO_API_BASE). Falls back to local on any error so
    training never dies on a reward-server hiccup.
    """
    import base64
    import io

    import requests

    payload_images = []
    for im in images:
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        payload_images.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    resp = requests.post(
        _CFG["dino_api_base"].rstrip("/") + "/ground",
        json={
            "images": payload_images,
            "texts": list(texts),
            "box_threshold": float(_CFG["box_threshold"]),
            "text_threshold": float(_CFG["box_threshold"]),
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["boxes"]


def _dino_boxes(images, texts):
    if not images:
        return []
    if _CFG.get("dino_api_base"):
        try:
            return _dino_boxes_served(images, texts)
        except Exception as e:  # noqa: BLE001
            print(f"[overlap_reward] served DINO failed ({e}); falling back to local")
    return _dino_boxes_local(images, texts)


# ---------------------------------------------------------------------------
# Precomputed per-question boxes (--overlap_question_boxes)
# ---------------------------------------------------------------------------
# One box list per dataset ROW, grounded once on the row's question by
# precompute_question_boxes.py before the run, and reused for every observe step of
# every completion of that row. See the module docstring for why the per-step call is
# not buying a per-step mask.

QBOX_VERSION = 1

# The row identity. Every corpus this trainer accepts carries all three -- the
# saliency-r1-8k default and every cold_data/grpo_sets/* built by build_grpo_sets.py --
# and the triple is unique in each of them (the builder checks, and so does the loader).
# `problem` is deliberately NOT part of it: questions repeat across images (35343 distinct
# strings over set_a's 50000 rows), so keying on the text would collapse different
# pictures onto one box list.
QBOX_KEY_COLUMNS = ("dataset", "split", "question_id")

# Loaded once per process, on first use.
_QBOX: dict = {"path": None, "boxes": None, "meta": None}


def qbox_key(dataset, split, question_id) -> str:
    """The cache key for one dataset row.

    Joined with '|' rather than JSON-encoded so the file stays readable; no corpus here
    has a separator in any of the three fields, and the builder refuses one that does
    instead of silently producing a key that two rows could share.
    """
    parts = [str(dataset), str(split), str(question_id)]
    for name, part in zip(QBOX_KEY_COLUMNS, parts):
        if "|" in part:
            raise ValueError(
                f"question-box key column `{name}` contains the '|' separator: {part!r}. "
                "Two rows could then collide onto one key."
            )
    return "|".join(parts)


def load_question_boxes(path: str, box_threshold=None, max_image_side=None) -> dict:
    """Read (once per process) and validate a precomputed question-box file.

    Both checks are hard failures, because both fail SILENTLY otherwise -- the run would
    train happily on boxes that are not the ones its own configuration describes:

      box_threshold   applied inside DINO, so it cannot be re-applied here. A cache built
                      at 0.10 cannot serve a run asking for 0.25.
      max_image_side  the detector sees a different picture at a different resolution.
                      The trainer passes its own constant so the two cannot drift.
    """
    if _QBOX["path"] == path and _QBOX["boxes"] is not None:
        return _QBOX["boxes"]

    import json

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"--overlap_question_boxes {path} does not exist. Build it first with "
            f"precompute_question_boxes.py (it needs a GPU, and it is a separate job)."
        )
    with open(path) as f:
        d = json.load(f)

    got_v = d.get("version")
    if got_v != QBOX_VERSION:
        raise ValueError(f"{path}: question-box file version {got_v}, expected {QBOX_VERSION}")

    meta = d.get("config") or {}
    if box_threshold is not None:
        want, have = float(box_threshold), meta.get("box_threshold")
        if have is None or abs(float(have) - want) > 1e-9:
            raise ValueError(
                f"{path} was built with box_threshold={have}, but this run asks for "
                f"{want}. The threshold is applied inside Grounding-DINO, so the cached "
                "boxes cannot be re-filtered to it -- rebuild the cache, or match the flag."
            )
    if max_image_side is not None:
        want, have = int(max_image_side), meta.get("max_image_side")
        if have is None or int(have) != want:
            raise ValueError(
                f"{path} was built with max_image_side={have}, but the trainer resizes to "
                f"{want}. Grounding-DINO sees a different picture at a different "
                "resolution -- rebuild the cache against this trainer."
            )

    boxes = d.get("boxes")
    if not isinstance(boxes, dict) or not boxes:
        raise ValueError(f"{path}: no `boxes` mapping")

    _QBOX.update(path=path, boxes=boxes, meta=meta)
    n_empty = sum(1 for v in boxes.values() if not v)
    print(f"[overlap_reward] question boxes: {len(boxes)} rows from {path} "
          f"(box_threshold={meta.get('box_threshold')}, "
          f"max_image_side={meta.get('max_image_side')}, "
          f"{n_empty} rows grounded nothing). Grounding-DINO will not be loaded.",
          flush=True)
    return boxes


def _question_boxes_per_row(kwargs, wanted):
    """-> {completion index: raw box list} for the completions in `wanted`.

    Only the completions that actually have steps to score are looked up, for the same
    reason masked rows never reach DINO on the per-step path: a row this call is not
    scoring must not be able to fail it.

    A key the cache does not hold IS a hard failure. It means the cache was built for a
    different corpus, and masking those rows instead would show up only as a quietly
    smaller reward on part of the batch.
    """
    if not wanted:
        return {}
    absent = [c for c in QBOX_KEY_COLUMNS if kwargs.get(c) is None]
    if absent:
        raise KeyError(
            f"--overlap_question_boxes needs the {', '.join(QBOX_KEY_COLUMNS)} columns to "
            f"identify a row, but {', '.join(absent)} did not reach the reward function. "
            "Use a corpus built by build_grpo_sets.py (cold_data/grpo_sets/*) or the "
            "saliency-r1-8k default."
        )
    boxes = load_question_boxes(_CFG["question_boxes"])
    cols = [kwargs[c] for c in QBOX_KEY_COLUMNS]
    out = {}
    for c in sorted(wanted):
        key = qbox_key(*(col[c] for col in cols))
        if key not in boxes:
            raise KeyError(
                f"row {key!r} is not in {_CFG['question_boxes']}. The cache does not cover "
                "the dataset being trained on -- rebuild it for this corpus."
            )
        out[c] = boxes[key]
    return out


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------

def _box_area(b):
    x1, y1, x2, y2 = b
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _union_mask(boxes, grid_h, grid_w, apply_union_cap=True):
    """Boolean (grid_h, grid_w) union of area-filtered boxes; None if degenerate.

    Rasterisation matches analysis/aggregation_correlation.py exactly.

    Two independent filters, both disabled by None or a non-positive value:

      max_box_area    per-box, applied to the raw relative-coordinate box BEFORE
                      rasterisation. Drops individual boxes; the surviving ones
                      still form a union.
      max_union_area  per-step, applied to the RASTERISED union. Rejects the whole
                      step (-> None -> skipped, not scored 0) when the union covers
                      more than this fraction of the patch grid. The per-box cap
                      does not bound the union: N disjoint boxes each under the cap
                      can cover the image between them (measured median union
                      coverage on set_a is 56%), and under `mean_in` a growing union
                      raises the score (r +0.17 with the area fraction). This is the
                      only filter that closes that.

    The union is measured on the patch grid, not on the box geometry, because the
    grid mask is what the metric actually scores -- and rasterisation inflates it:
    every surviving box claims at least one patch row and column, so many small
    boxes cover more grid than their summed area suggests.

    apply_union_cap=False skips only the max_union_area check, so a caller can see the
    mask the cap rejected (overlap_probe uses this to distinguish "the cap dropped this
    step" from "DINO grounded nothing"). The reward path always leaves it True.
    """
    max_area = _CFG.get("max_box_area")
    if max_area is not None and float(max_area) > 0:
        boxes = [b for b in boxes if _box_area(b) <= max_area]
    if not boxes:
        return None
    mask = np.zeros((grid_h, grid_w), dtype=bool)
    for x1, y1, x2, y2 in boxes:
        r0 = max(0, int(y1 * grid_h))
        r1 = min(grid_h, max(r0 + 1, round(y2 * grid_h)))
        c0 = max(0, int(x1 * grid_w))
        c1 = min(grid_w, max(c0 + 1, round(x2 * grid_w)))
        mask[r0:r1, c0:c1] = True
    n_in = int(mask.sum())
    if n_in == 0 or n_in == grid_h * grid_w:
        return None
    max_union = _CFG.get("max_union_area")
    if apply_union_cap and max_union is not None and float(max_union) > 0:
        if n_in > float(max_union) * grid_h * grid_w:
            return None
    return mask


# One rectangle per (grid, fraction); the grids repeat all run long and the mask is a
# pure function of them.
_RECT_CACHE: dict = {}


def _centre_rect_mask(grid_h, grid_w, frac):
    """Centred axis-aligned rectangle covering ~frac of the patch grid; None if degenerate.

    The area is split equally between the two axes (sqrt(frac) on each), so the
    rectangle keeps the frame's aspect and is not secretly a wide or tall band: the
    only thing it differs from a DINO union in is WHERE it sits, which is what the
    --overlap_rect_frac arm is testing.

    Rasterised on the patch grid, and rounded to whole patches like _union_mask, for
    the same reason -- the grid mask is what the metric scores. On a grid this coarse
    (10x16 is typical) the rounding moves the realised area a few points off `frac`;
    that is deliberate and shared with centre_box_probe.py, which this reproduces
    exactly so the training arm scores the same rectangle the probe measured.
    """
    key = (int(grid_h), int(grid_w), round(float(frac), 6))
    hit = _RECT_CACHE.get(key)
    if hit is not None:
        return hit
    s = math.sqrt(min(1.0, max(0.0, float(frac))))
    rows = min(grid_h, max(1, int(round(grid_h * s))))
    cols = min(grid_w, max(1, int(round(grid_w * s))))
    mask = np.zeros((grid_h, grid_w), dtype=bool)
    r0, c0 = (grid_h - rows) // 2, (grid_w - cols) // 2
    mask[r0:r0 + rows, c0:c0 + cols] = True
    n_in = int(mask.sum())
    if n_in == 0 or n_in == grid_h * grid_w:
        # Same refusal as a degenerate union: "inside vs outside" is not a question on
        # a mask that covers everything or nothing. Not cached -- it is a config error,
        # and _validate_rect catches the frac that causes it before training starts.
        return None
    _RECT_CACHE[key] = mask
    return mask


def _rect_dims(extent_h, extent_w, frac):
    """Rows and columns covering ~frac of an extent_h x extent_w area, at its aspect."""
    s = math.sqrt(min(1.0, max(0.0, float(frac))))
    return (min(extent_h, max(1, int(round(extent_h * s)))),
            min(extent_w, max(1, int(round(extent_w * s)))))


def interior_placements(grid_h, grid_w, frac):
    """(rows, cols, n_rows_of_positions, n_cols_of_positions) for the interior rectangle.

    The interior is the grid minus its one-patch border -- the edge sink that carries
    ~half the attention mass and most of the peaks (see the module docstring). `frac` is
    read as a fraction of that interior, not of the whole grid, which is what leaves the
    rectangle room to move without ever touching the ring.

    None when the grid has no interior at all (either side below 3 patches).
    """
    ih, iw = int(grid_h) - 2, int(grid_w) - 2
    if ih < 1 or iw < 1:
        return None
    rows, cols = _rect_dims(ih, iw, frac)
    return rows, cols, ih - rows + 1, iw - cols + 1


def _placed_rect_mask(grid_h, grid_w, rows, cols, r0, c0):
    """Boolean grid with a rows x cols block at (r0, c0); None if degenerate."""
    key = (int(grid_h), int(grid_w), int(rows), int(cols), int(r0), int(c0))
    hit = _RECT_CACHE.get(key)
    if hit is not None:
        return hit
    mask = np.zeros((int(grid_h), int(grid_w)), dtype=bool)
    mask[r0:r0 + rows, c0:c0 + cols] = True
    n_in = int(mask.sum())
    if n_in == 0 or n_in == int(grid_h) * int(grid_w):
        return None
    _RECT_CACHE[key] = mask
    return mask


def _interior_rect_mask(grid_h, grid_w, frac, index=None):
    """The interior rectangle, centred (index=None) or at placement `index`.

    Placements are numbered row-major over the strictly-interior positions, so every
    index gives a mask with ring coverage exactly 0 -- the property that separates this
    from a rectangle merely displaced inside the frame, which lets a different slice of
    the ring back in on every draw and dilutes the reward's ring signal below the
    box-blind statistic's. See the module docstring for the measured table.
    """
    got = interior_placements(grid_h, grid_w, frac)
    if got is None:
        return None
    rows, cols, n_r, n_c = got
    if index is None:
        r_i, c_i = (n_r - 1) // 2, (n_c - 1) // 2
    else:
        i = int(index) % (n_r * n_c)
        r_i, c_i = divmod(i, n_c)
    return _placed_rect_mask(grid_h, grid_w, rows, cols, 1 + r_i, 1 + c_i)


def _completion_text(completion) -> str:
    """The assistant text of one completion, in either shape the trainer may hand over.

    A local copy of placebo_rewards._completion_text: that module imports this one, so
    importing it back would be a cycle.
    """
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        return completion[0].get("content", "") or ""
    return completion if isinstance(completion, str) else ""


def _blake_u64(*parts: str) -> int:
    """Stable 64-bit digest. Stable across processes, ranks and restarts, unlike Python's
    own hash(), which is salted per interpreter -- so the same completion draws the same
    rectangle on every rank and after every resume."""
    return int.from_bytes(
        hashlib.blake2b("\x1f".join(parts).encode("utf-8"), digest_size=8).digest(), "big")


def _ring_frac(mask) -> float:
    """Share of a mask's patches that lie on the grid's one-patch border."""
    n_in = int(mask.sum())
    if n_in == 0:
        return float("nan")
    ring = np.zeros_like(mask)
    ring[0, :] = ring[-1, :] = True
    ring[:, 0] = ring[:, -1] = True
    return float(np.logical_and(mask, ring).sum()) / n_in


def _rect_mask_for(grid_h, grid_w, text):
    """The configured rectangle for one step, under --overlap_rect_placement."""
    frac = _CFG["rect_frac"]
    placement = _CFG.get("rect_placement") or "centre"
    if placement == "centre":
        return _centre_rect_mask(grid_h, grid_w, frac)
    if placement == "interior_centre":
        return _interior_rect_mask(grid_h, grid_w, frac)
    # interior_hash: the position is a pure function of the completion's own text, so it
    # is fixed for a completion and varies between the 8 rollouts of a prompt -- which is
    # the whole point, since a mask constant inside a group cancels out of the advantage.
    idx = _blake_u64("overlap-rect", str(_CFG.get("rect_seed", 0)), text)
    return _interior_rect_mask(grid_h, grid_w, frac, index=idx)


def _mean_in(step_map, mask):
    """mean of MAX-normalized (/max -> [0,1]) saliency inside the mask."""
    vmax = float(step_map.max())
    m = step_map / vmax if vmax > 0 else step_map
    inside = m[mask]
    return float(inside.mean()) if inside.size > 0 else None


def _mean_in_v2(step_map, mask):
    """mean of the saliency inside the mask, divided by its mean over the whole map.

    Chance = 1.0; unbounded above (ceiling n_patches / n_in). Both terms are means of
    the SAME map, so the normalisation constant cancels -- no /max, no separate peak
    to inflate, and the value is invariant to m -> c*m (see the module docstring).
    """
    v = np.asarray(step_map, dtype=np.float64)
    inside = v[np.asarray(mask, dtype=bool)]
    if inside.size == 0:
        return None
    denom = float(v.mean())
    if denom <= 0:
        return None  # all-zero map: the ratio is undefined -> skip this step
    return float(inside.mean()) / denom


def _auroc(step_map, mask):
    """P(random in-box patch outranks a random out-box patch); 0.5 == chance.

    Average ranks for ties -- attention maps have many near-identical near-zero
    patches and argsort would break those ties arbitrarily, biasing the estimate.
    Pure numpy (no scipy) to avoid adding a dependency to the training env; this is
    the same computation as the offline screen, so the offline attack simulation and
    utility screen predict this reward exactly.
    """
    v = np.asarray(step_map, dtype=np.float64).ravel()
    m = np.asarray(mask, dtype=bool).ravel()
    n_in = int(m.sum())
    n_out = v.size - n_in
    if n_in == 0 or n_out == 0:
        return None
    order = np.argsort(v, kind="stable")
    ranks = np.empty(v.size, dtype=np.float64)
    ranks[order] = np.arange(1, v.size + 1, dtype=np.float64)
    _uniq, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
    sums = np.zeros(cnt.size, dtype=np.float64)
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    u = ranks[m].sum() - n_in * (n_in + 1) / 2.0
    return float(u / (n_in * n_out))


def _mass_gate(step_map):
    """min(1, image_mass / tau); 1.0 when the floor is disabled.

    image_mass = the map's total, i.e. the fraction of the softmax row spent on image
    tokens (the row is a softmax over all keys; only image columns were kept).
    """
    tau = _CFG.get("mass_floor_tau")
    if not tau or float(tau) <= 0:
        return 1.0
    return min(1.0, float(np.asarray(step_map).sum()) / float(tau))


def _roll_logratio(step_map, mask):
    """The roll-null score with THIS module's knobs, logging THIS module's diagnostics.

    `roll_null.py` holds the definition and the reasoning; it is shared with the gradient
    reward, where the same score is the only scoring mode rather than one metric of four.
    Chance is exactly 0, so unlike the other three this metric is already centred.

    Random by construction: it draws control placements. A caller that needs the value
    twice must keep it, not call twice, or the step gets two different scores.
    """
    r, info = _RN.logratio(step_map, mask, _ROLL_RNG,
                           n_offsets=int(_CFG["null_offsets"]),
                           clip=float(_CFG["logratio_clip"]),
                           inframe=bool(_CFG["inframe_rolls"]),
                           min_inframe=int(_CFG["min_inframe"]))
    if r is None:
        return None
    for key in ROLL_DIAG_KEYS:
        _diag(key, info[key])
    return r


def _step_score(step_map, mask, metric=None):
    """Per-step reward: the configured metric, times the optional mass floor.

    `metric` overrides `_CFG["metric"]` for this call. The gradient reward passes its own,
    because its historical default is the roll-null while this module's is `mean_in`: with
    one flag now serving three maps, the alternative is two copies of the setting that
    have to agree, and they eventually would not.
    """
    metric = metric or _CFG.get("metric")
    if metric == "auroc":
        v = _auroc(step_map, mask)
    elif metric == "mean_in_v2":
        v = _mean_in_v2(step_map, mask)
    elif metric == "logratio":
        v = _roll_logratio(step_map, mask)
    else:
        v = _mean_in(step_map, mask)
    if v is None:
        return None
    return v * _mass_gate(step_map)


def think_overlap_reward(
    completions=None, saliency_map=None, valid_list=None, image=None, natural=None, **kwargs
):
    """Per-completion overlap reward. See module docstring.

    Returns a list (len == n completions) of floats, or None where there is no grounded
    observe step, or where --overlap_natural_only masks a non-natural row (masked ->
    neutral in GRPO). w_overlap is applied by --reward_weights.

    Under --overlap_rect_frac the mask is a rectangle rather than the DINO union, and
    Grounding-DINO is not called at all. Every step then has a mask, so the only
    remaining route to None is a completion with no observe steps, or the
    --overlap_natural_only mask -- unless --overlap_rect_placement is an interior mode on
    a grid too small to have an interior, which skips the step like a degenerate union.

    Under --overlap_chain_boxes the detector IS called, once per completion instead of
    once per observe step, and the union it returns is scored against every step of that
    completion. A completion whose chosen step grounds nothing is unscored as a whole.
    """
    n = len(saliency_map)
    if valid_list is None:
        valid_list = [True] * n

    # --overlap_natural_only: score only the photographic rows. `natural` arrives as a
    # per-row dataset column (the trainer forwards every column as a reward kwarg).
    if _CFG.get("natural_only"):
        if natural is None:
            raise KeyError(
                "--overlap_natural_only requires a boolean 'natural' column in the dataset, "
                "but none reached the reward function. Use a corpus built by "
                "build_grpo_sets.py (cold_data/grpo_sets/*), or drop the flag."
            )
        scored = [bool(x) for x in natural]
    else:
        scored = [True] * n

    rect = rect_active()

    qbox = bool(_CFG.get("question_boxes"))

    if rect or qbox:
        # Neither arm reads the step's TEXT, so neither builds the image/text lists and
        # neither calls _dino_boxes. That is what makes both runs detector-free --
        # _load_dino_local is lazy, so not calling it is the whole mechanism, and the
        # launcher's WANT_DINO=false path can then give the GPU DINO would have held to
        # training. Masked rows are excluded here exactly as they are on the DINO path.
        flat_owner = [(c, si) for c, steps in enumerate(saliency_map)
                      if steps and scored[c] for si in range(len(steps))]
        if rect:
            # --overlap_rect_frac: the mask is a function of the GRID alone, and is built
            # in the scoring loop below rather than from any boxes.
            boxes_per_item = [None] * len(flat_owner)
        else:
            # --overlap_question_boxes: one grounding per ROW, done before the run. Every
            # step of a completion gets the same box list, so the loop below is unchanged:
            # it rasterises that one list onto each step's grid (identical for all steps of
            # a completion -- same image, same patch grid) and scores each step's own map
            # against it. Deliberately NOT hoisted out of the loop: a duplicated
            # `_union_mask` on a 16x10 grid costs nothing, and keeping one scoring path
            # means the cached and per-step runs differ in where the boxes came from and in
            # nothing else.
            per_row = _question_boxes_per_row(kwargs, {c for c, _si in flat_owner})
            boxes_per_item = [per_row[c] for c, _si in flat_owner]
    elif chain_boxes_active():
        # --overlap_chain_boxes: ONE grounding call per completion, on the chosen step's
        # own sentence, then every step of that completion is scored against it. The
        # scoring loop below is untouched -- the per-completion box list is simply
        # repeated across the completion's steps -- so this arm differs from the per-step
        # reference in which sentence was grounded and in nothing else.
        sel = _CFG["chain_boxes"]
        sel_c, sel_images, sel_texts = [], [], []
        for c, steps in enumerate(saliency_map):
            if not steps or not scored[c]:
                continue
            si = 0 if sel == "first" else len(steps) - 1
            sel_c.append(c)
            sel_images.append(image[c])
            sel_texts.append(steps[si]["text"])
        got = _dino_boxes(sel_images, sel_texts) if sel_images else []
        per_comp = dict(zip(sel_c, got))
        # Ungrounded here costs the WHOLE completion, not one step: there is no fallback
        # to a second sentence, because a fallback would cost a second detector call and
        # make "one call per completion" untrue. Logged rather than papered over.
        _mask_diag("chain_ungrounded_frac",
                   float(np.mean([not per_comp[c] for c in sel_c])) if sel_c else float("nan"))
        flat_owner = [(c, si) for c in sel_c for si in range(len(saliency_map[c]))]
        boxes_per_item = [per_comp[c] for c, _si in flat_owner]
    else:
        # Flatten every (completion, observe-step) into one batched DINO call. Masked rows
        # never reach DINO -- the trainer normally hands them no maps anyway, but a row
        # masked here must not cost a grounding call even if it does.
        flat_images, flat_texts, flat_owner = [], [], []
        for c, steps in enumerate(saliency_map):
            if not steps or not scored[c]:
                continue
            img = image[c]
            for si, st in enumerate(steps):
                flat_images.append(img)
                flat_texts.append(st["text"])
                flat_owner.append((c, si))

        boxes_per_item = _dino_boxes(flat_images, flat_texts) if flat_images else []

    # --overlap_rect_placement interior_hash draws the rectangle's position from the
    # completion's own text, so the texts have to be here. Resolved once, up front, so a
    # missing `completions` fails before any scoring rather than on one unlucky row.
    diag_on = mask_diag_active()
    hashed_rect = rect and (_CFG.get("rect_placement") == "interior_hash")
    if hashed_rect:
        if completions is None:
            raise KeyError(
                "--overlap_rect_placement interior_hash needs the completions to place the "
                "rectangle, but none reached the reward function. The position IS the "
                "completion's identity here; without it every rollout of a prompt would "
                "get the same mask, which is the arm this one exists to differ from.")
        texts = [_completion_text(x) for x in completions]
    else:
        texts = [""] * n

    # Gather grounded mean_in per completion.
    per_completion = [[] for _ in range(n)]
    for (c, si), boxes in zip(flat_owner, boxes_per_item):
        step_map = saliency_map[c][si]["map"]
        gh, gw = step_map.shape
        mask = (_rect_mask_for(gh, gw, texts[c]) if rect
                else _union_mask(boxes, gh, gw))
        if mask is None:
            continue  # DINO couldn't ground this step -> skip (do NOT score 0)
        if diag_on:
            _mask_diag("union_frac", float(mask.sum()) / mask.size)
            _mask_diag("ring_frac", _ring_frac(mask))
            if hashed_rect:
                # How many distinct masks the draw could have produced on THIS grid. A
                # coarse grid leaves about a dozen, which bounds how much per-completion
                # variation the arm can restore -- read it before reading the spread.
                got = interior_placements(gh, gw, _CFG["rect_frac"])
                _mask_diag("n_placements", float(got[2] * got[3]) if got else 1.0)
            elif rect:
                _mask_diag("n_placements", 1.0)
        s = _step_score(step_map, mask)
        if s is not None:
            per_completion[c].append(s)

    rewards = []
    for c in range(n):
        if not scored[c]:
            rewards.append(None)  # non-natural under --overlap_natural_only -> mask
            continue
        vals = per_completion[c]
        if not vals:
            rewards.append(None)  # zero grounded observe steps -> mask (neutral)
            continue
        overlap = float(np.mean(vals))
        rewards.append(overlap * (1.0 if valid_list[c] else 0.0))  # format gate (multiplicative)
    return rewards
