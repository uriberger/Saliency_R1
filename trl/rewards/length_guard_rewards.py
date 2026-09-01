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

"""A length REGULATOR (--length-guard): zero for normal completions, negative for runaways.

WHAT IT IS. An extra reward term, ADDED alongside whatever auxiliary reward a run is using
rather than replacing it. It is exactly 0.0 while a completion's token count stays within a
wide multiplicative band around what the BASE policy -- the cold-start model, before GRPO
-- produced on this corpus, and goes negative outside it. So it does nothing during normal
training and only pushes back once lengths run away.

"Does nothing normally" is the design, and it is what separates this from --placebo length
(placebo_rewards.py), which scored EVERY completion on a straight line in length. That
reward drove completions 173 -> 13.5 tokens, entropy 0.654 -> 0.116, and was the worst arm
on both benchmark suites. Brevity is a real direction and it points the wrong way. This is
not a brevity reward; it is a two-sided leash whose middle is empty.

THE NOTATION, one symbol at a time:

    n          the completion's token count -- the same count the trainer logs as
               train/completions/mean_length.
    l_ref      REFERENCE LENGTH, in tokens: the base policy's MEAN completion length on
               the corpus being trained on. One constant for the whole run. Read it off
               completions/mean_length at step 0 of a matched run, or from
               overlap_metric_spread.py, which prints it. There is no safe default.
    band_lo    lower edge of the free window, as a MULTIPLE of l_ref (0.30 = "a third of
               the base length is still fine").
    band_hi    upper edge, likewise (3.0 = "three times the base length is still fine").
    d          log length ratio, d = ln(n / l_ref). Zero at the reference; +0.69 is twice
               as long; -0.69 is half as long. LOG, not percent -- see WHY LOG below.
    e          EXCESS: how far outside the band, in the same log units. 0 inside.
    knee       excess at which the penalty stops growing quadratically and grows linearly.

    d = ln(n / l_ref)
    e = max( 0,  d - ln(band_hi),  ln(band_lo) - d )     # at most one term is positive

    reward = -( e^2                            if e <= knee
                knee^2 + 2*knee*(e - knee)     if e >  knee )

The reward is never positive: the best a completion can do is 0, by staying in the band.
This function returns the UNWEIGHTED shape; the strength k is the term's entry in
--reward_weights, which is how every other reward here is calibrated and which keeps
train/rewards/length_guard_reward/within_group_std meaning the shape's own spread.

WHY LOG, AND WHY THE BAND IS SO WIDE. Both were forced by measurement, and the first draft
of this reward (a +-20% / -50% band on a plain n/l_ref - 1) was wrong on both counts. The
per-completion length distribution is heavily RIGHT-SKEWED, so a band that looks generous
against the mean is not generous against the population. On the cold-start policy, 240
completions, val_natural (outputs/overlap_probe/20260805-030416-crossrun-val_natural):

    median 195    mean 221    p95 403 (1.8x)    p99 771 (3.5x)    max 1024 (4.6x)
    p5      117 (0.53x)       p1  103 (0.47x)

A +-20%/-50% band would have penalised 22% of HEALTHY completions and applied 0.078 of
effective pressure at the cold start -- 28x the mean_in reward's own 0.0028, and 37x the
0.0021 at which --placebo length destroyed a run. It would have been the thing it was
built to prevent. A log ratio also makes the two sides symmetric in the only sense that
matters here: "half as long" and "twice as long" are the same distance from the reference,
which a percentage deviation cannot express (it is bounded at -100% on one side and
unbounded on the other, so a plain-ratio guard saturates on the short side exactly where
every real collapse happened).

WHAT IT ACTUALLY CATCHES -- the honest asymmetry. Measured on the same probe, one row per
model, l_ref fixed at the COLD START's 221 tokens for every row (which is what a run does:
the reference does not move as the policy drifts). `pressure` is k x sd_within at k=0.20,
against mean_in w0.4's 0.0028:

    model                      mean len   frac_penalised   sd_within   pressure
    base_coldstart                  221        0.013         0.0178     0.0036
    mean_in saliency_r1_8k (good)   153        0.004         0.0043     0.0009
    mean_in set_a cp1000 (pre-hack) 177        0.033         0.0060     0.0012
    mean_in_v2 set_a cp1700 (BAD)   318        0.050         0.0247     0.0049
    mean_in set_a cp2000 (BAD)      342        0.071         0.0323     0.0065
    auroc set_a cp2000 (BAD)         56        0.787         0.0584     0.0117
    auroc set_a cp2500 (BAD)         49        0.963         0.0316     0.0063

  * On LENGTH COLLAPSE it is decisive: the auroc runs have 79-96% of their completions
    outside the band, against 1.3% at the cold start, and 3-7x the within-group spread.
  * On LENGTH INFLATION it is weak: the two padding runs sit at 1.4-1.8x the cold start.

That asymmetry is structural, not a tuning failure, and it is the right shape. Inflation
ALREADY has a hard guard: max_completion_length truncates, a truncated completion loses
`</think>`, and it therefore scores 0 on BOTH accuracy and format -- about 2.0 reward units.
With l_ref = 221 there is less than half an octave between the healthy p99 (3.5x) and that
cap (4.6x), so all this term can add on the long side is a soft ramp into an existing
cliff. Collapse has NO existing guard at all: a 13-token completion can be perfectly
formatted and scored correct. And that is where every recorded degeneration the accuracy
reward could not see actually went -- auroc to 49, --maskfree mass to 31, --placebo length
to 13.5. The guard is strong exactly where nothing else is.

WHAT IT DOES NOT CATCH, stated so it is not discovered later. set_c -- the 2x corpus whose
run collapsed from a bench mean of 0.6508 at step 1500 to 0.5600 by 5100 -- moved its mean
length only 178 -> 254 (1.15x l_ref) and its long tail to ~400 (1.81x). BOTH are inside
this band, and widening the band to reach them is exactly what the measurements above rule
out. This term would not have saved that run. The regulator for a set_c-style entropy
collapse (0.697 -> 0.123) is --beta, and the two are meant to be run together for that
reason. `lenguard/frac_penalized` is logged so this is measured on the run rather than
assumed: 0.00 means the guard is inert and the run's length behaviour is entirely the other
rewards' doing.

WHY QUADRATIC. --placebo length self-extinguished: being linear in length, once a group
converged on one length there was no spread left between its 8 completions, the advantage
went to zero, and the term stopped working (frac_reward_zero_std 0.000 -> 0.64, worse than
accuracy-only's 0.60; pressure 0.0021 -> 0.0001). A quadratic's SLOPE grows with the
excess, so the same within-group scatter produces a larger spread in penalty the further
out the group is. The guard strengthens as the problem worsens instead of switching itself
off.

WHY THE KNEE. scale_rewards=True divides every advantage by the group's standard deviation
-- the amplifier overlap-reward-hack-set-a.md blames for turning a 0.012-reward-unit
overlap spread into a +-2.9 advantage. An unbounded penalty on one completion would inflate
that std and shrink EVERY other completion's advantage in the group. Going linear past
`knee` bounds that. The two pieces agree in value AND slope at e = knee, so the penalty is
C1 and there is no kink for the optimiser to sit in.

SCORED SET: every completion, always, and never None. A deliberate difference from
placebo_rewards, which had to return unscored on exactly the completions the overlap reward
leaves unscored because it was a CONTROL on that reward. This is a constraint, not a
control, and masking it to the DINO-scored set would make "produce no groundable observe
step" an escape hatch from the leash. Returning a float for every row also means the
pre-8489767 `nansum`-vs-imputing-fold question never arises for this term.

COST: none. It reads `completion_ids` and nothing else -- no attention map, no
Grounding-DINO, no dataset column, no GPU.

CALIBRATING k. Re-measure on the corpus you train on with `overlap_metric_spread.py`,
which prints l_ref and two lenguard rows by importing this module. The default k = 0.20
was set so that:

  1. at the cold start the pressure is 0.0036 -- the same order as mean_in w0.4's 0.0028
     and above --placebo length's destructive 0.0021, but concentrated on the 1.3% of
     completions outside the band instead of spread over all of them as a brevity
     gradient. On the healthy 8k run it is 0.0009.
  2. a collapse is expensive: 0.018 reward units at 49 tokens, 0.12 at 31, 0.45 at 13.

If those two conflict on your corpus, WIDEN the band rather than lowering k -- the fix for
"too noisy at baseline" is a bigger free window, not a weaker penalty on real runaways.

WHY l_ref IS ONE CONSTANT AND NOT PER-PROMPT. Some questions naturally want short answers
and others long ones, so a single l_ref looks like it would permanently penalise the
naturally-long prompts. It mostly cancels, for a reason specific to GRPO: all 8 completions
in a group answer the SAME prompt, so a naturally-long prompt penalises all 8 roughly
equally and the advantage subtracts the group mean before anything else. What survives is
second-order -- the penalty's slope is steeper for naturally-long prompts. Measure it
before fixing it: if `lenguard/frac_penalized` is high and concentrated in one dataset
source, a per-prompt reference table (one offline cold-start generation pass) is the answer.
"""

from __future__ import annotations

import math

import numpy as np

_CFG = {
    # None = disabled; length_guard_reward is then never installed by grpo_vlm_qwen3.py.
    # In TOKENS: the base policy's mean completion length on the corpus being trained on.
    # No default is possible -- 221 is the cold-start model's on val_natural and would be
    # wrong on a corpus whose chains are half as long -- so the launcher requires it.
    "l_ref": None,
    # The free window, as MULTIPLES of l_ref. Both measured, not guessed: see WHY LOG.
    # 0.30 clears the healthy 8k run's own short tail (its p1 is 0.37x) while still sitting
    # above every collapse on record (auroc 0.22x, maskfree mass 0.14x, placebo length
    # 0.06x). 3.0 sits between the cold start's p99 (3.5x) and its bulk, and below the
    # max_completion_length cap at 4.6x.
    "band_lo": 0.30,
    "band_hi": 3.0,
    # Quadratic below this excess (in LOG units), linear above. See WHY THE KNEE.
    "knee": 1.0,
}

# FIXED key set, always all of it, for the reason grad_rewards.DIAG_KEYS and
# maskfree_rewards.DIAG_KEYS document: the trainer gathers these across ranks, so a key set
# that depended on what a rank happened to see would mean a rank-dependent number of
# collectives, which hangs rather than fails.
#
# `frac_penalized` is the one to read. It says what share of completions the guard is
# actually touching -- the open question about this term, since it is silent on set_c-style
# inflation by construction. `frac_long` vs `frac_short` says which failure mode is
# developing, and they are separate because the two have opposite fixes.
DIAG_KEYS = ("frac_penalized", "frac_long", "frac_short", "mean_penalty", "mean_logratio",
             "mean_len", "p95_len", "p05_len")
_DIAG: dict[str, list[float]] = {}


def _diag(key: str, value: float):
    _DIAG.setdefault(key, []).append(float(value))


def pop_diagnostics() -> dict[str, float]:
    """Mean of each diagnostic since the last call, then clear.

    Always all of DIAG_KEYS; NaN for a key nothing was recorded under. Values are per-CALL
    aggregates (one generation batch on this rank), so the mean here is over calls and the
    trainer's gather makes it a mean over ranks.
    """
    out = {k: (float(np.mean(_DIAG[k])) if _DIAG.get(k) else float("nan")) for k in DIAG_KEYS}
    _DIAG.clear()
    return out


def is_active() -> bool:
    """True when the length guard is installed. Rank-uniform (it is set from the CLI on
    every process), so the trainer may branch logging collectives on it."""
    return _CFG["l_ref"] is not None


def configure(**kwargs):
    """Set the length-guard config from the CLI flags. None values are ignored."""
    for k, v in kwargs.items():
        if v is not None:
            _CFG[k] = v
    if _CFG["l_ref"] is None:
        return
    l_ref = float(_CFG["l_ref"])
    if not np.isfinite(l_ref) or l_ref <= 0:
        raise ValueError(f"--length-guard must be a positive token count (got {l_ref!r})")
    lo, hi = float(_CFG["band_lo"]), float(_CFG["band_hi"])
    for name, v in (("band-lo", lo), ("band-hi", hi)):
        if not np.isfinite(v) or v <= 0:
            raise ValueError(f"--length-guard-{name} is a MULTIPLE of the reference length "
                             f"and must be > 0 (got {v!r})")
    # lo >= hi is an empty band: every completion is outside it, on one side or the other,
    # and the guard becomes a pure brevity/verbosity reward -- the --placebo length failure.
    # lo > 1 or hi < 1 puts the reference length itself outside its own band.
    if lo >= hi:
        raise ValueError(f"--length-guard-band-lo ({lo}) must be < --length-guard-band-hi "
                         f"({hi}); an empty band penalises every completion, which is the "
                         "brevity reward this term exists to not be.")
    if not (lo <= 1.0 <= hi):
        raise ValueError(f"the free band [{lo}, {hi}] x l_ref does not contain the reference "
                         "length itself, so a completion of exactly l_ref tokens would be "
                         "penalised. Band edges are MULTIPLES of l_ref, not token counts.")
    knee = float(_CFG["knee"])
    if not np.isfinite(knee) or knee <= 0:
        raise ValueError(f"--length-guard-knee must be > 0 (got {knee!r})")


def penalty(n_tokens: int,
            l_ref: float | None = None,
            band_lo: float | None = None,
            band_hi: float | None = None,
            knee: float | None = None) -> float:
    """The unweighted shape, for ONE completion of `n_tokens` tokens. Always <= 0.0.

    Split out from the reward so tests and `overlap_metric_spread.py` can score a length
    without installing a config -- the same reason `maskfree_rewards.flatness` is its own
    function. Unset arguments fall back to `_CFG`.
    """
    l_ref = float(_CFG["l_ref"] if l_ref is None else l_ref)
    band_lo = float(_CFG["band_lo"] if band_lo is None else band_lo)
    band_hi = float(_CFG["band_hi"] if band_hi is None else band_hi)
    knee = float(_CFG["knee"] if knee is None else knee)

    # A zero-token completion is scored, not skipped, and log(0) is not a number: clamp to
    # one token, which is already ~5 knee-widths outside any sane band. Not rounded to an
    # int -- the trainer only ever passes integer counts, but rounding would move a
    # completion sitting exactly on a band edge to the wrong side of it.
    d = math.log(max(float(n_tokens), 1.0) / l_ref)
    e = max(0.0, d - math.log(band_hi), math.log(band_lo) - d)
    if e <= 0.0:
        return 0.0
    # Quadratic up to the knee, then linear at the quadratic's slope there. Value and first
    # derivative both agree at e == knee, so there is no kink.
    return -(e * e if e <= knee else knee * knee + 2.0 * knee * (e - knee))


def length_guard_reward(completions=None, completion_ids=None, **kwargs):
    """Per-completion length penalty. See the module docstring.

    Reads `completion_ids` and nothing else: no attention map, no Grounding-DINO, no dataset
    column. Returns a float for EVERY completion and never None -- see SCORED SET.
    """
    if _CFG["l_ref"] is None:
        raise ValueError(
            "length_guard_reward was called with no reference length. It is installed by "
            "grpo_vlm_qwen3.py only when --length-guard is passed, and that path calls "
            "configure(l_ref=...) first."
        )
    if completion_ids is None:
        raise KeyError(
            "--length-guard needs the per-completion token ids (the trainer passes them as "
            "completion_ids), but none reached the reward function."
        )

    lengths = [len(ids) for ids in completion_ids]
    rewards = [penalty(n) for n in lengths]

    # Per-call aggregates. Recorded once per generation batch rather than once per
    # completion so that `frac_*` means a fraction and not the mean of a 0/1 column gathered
    # at a different rate from the others.
    if lengths:
        l_ref = float(_CFG["l_ref"])
        lo_edge, hi_edge = l_ref * float(_CFG["band_lo"]), l_ref * float(_CFG["band_hi"])
        _diag("frac_penalized", float(np.mean([r < 0.0 for r in rewards])))
        _diag("frac_long", float(np.mean([n > hi_edge for n in lengths])))
        _diag("frac_short", float(np.mean([n < lo_edge for n in lengths])))
        _diag("mean_penalty", float(np.mean(rewards)))
        _diag("mean_logratio", float(np.mean([math.log(max(n, 1) / l_ref) for n in lengths])))
        _diag("mean_len", float(np.mean(lengths)))
        _diag("p95_len", float(np.percentile(lengths, 95)))
        _diag("p05_len", float(np.percentile(lengths, 5)))

    return rewards
