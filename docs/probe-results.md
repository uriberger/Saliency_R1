# Probe results: the intervention null, the all-head scan, and the indirect-flow maps

Results log for the experiments planned in
[attention-intervention-plan.md](attention-intervention-plan.md). Three runs so far,
all on the cold-start model, `cold_data/grpo_sets/set_a`, per-step Grounding-DINO
boxes, the same 1,157 prepared cases in `outputs/intervene_probe/coldstart_setA_v2`.

Read 1 and 2 together. The first is a *layer-level* null; the second shows why that
null does not license the conclusion the plan drew from it. 3 replaces the direct
attention map with three indirect ones and is the first place a signal survives
correction. 6 scores a fourth, GLIMPSE, and is the first map here that is grounded at
all — which turns out not to be the good news it sounds like. 7 is the only one that is
not a screen: it sets the `w_overlap` a GLIMPSE run needs, on its own draw of completions
rather than the shared 1,157 cases.

---

## 1. Layer-level intervention (2026-08-05) — a tight null

`intervene_probe.py`, 1,157 cases x 36 layers x (box, roll, perm + alpha=0 baseline)
= **166,608 forwards**, plus an alpha sweep at L22 (11 conditions x 4 strengths).

Forcing an observe step's attention onto the objects that step names does **exactly as
much to the answer as forcing it onto an equal-area region elsewhere**:

```
pooled box - roll over all 35 non-trivial layers:  -0.00006 nats  [-0.00076, +0.00064]   n=40,495
per layer:  mean -0.00006,  sd 0.0023,  max |0.0067| at L9
95% CI excludes 0 at 2/36 layers (chance ~1.8);  Bonferroni survivors: 0
```

The alpha sweep at L22 (all 32 heads) is flat at every strength, and two further
controls are also zero:

| | |
|---|---|
| box - roll, alpha 0.25 / 0.5 / 0.75 / 1.0 | -0.0002 / +0.0024 / -0.0001 / +0.0010, every CI spanning 0 |
| shape - box (alpha=1) | -0.0018 [-0.0060, +0.0024] — within-box structure buys nothing |
| **box - image (alpha=1)** | **-0.0004 [-0.0043, +0.0036]** — all the mass in the named object's box is indistinguishable from spreading it over the whole image |

Behaviourally negligible: 0.25-0.31% of answers flip, net movement toward gold +14
(box) / +11 (roll) / +3 (perm) out of 41,652 comparisons.

**Harness validation.** L35 gives exactly 0.0000 under every condition, which theory
demands (modifying the last layer's output at chain positions cannot reach the answer
position). |d logp| decays monotonically with depth, 0.051 at L0 -> 0.013 at L34.
`--stage selftest` gates the run: the rebuilt attention rows match the module's own
eager output to **one bf16 ULP (2^-8)**, and an alpha=0 repeat is bit-identical, so the
shape-dependent matmul rounding is a fixed per-case offset that cancels in every
paired delta.

### What this does NOT show

An earlier version of this page claimed `perm` proved the layer was causally live.
The alpha sweep withdraws that: mean |d logp| is 0.0378 at alpha=0.25 and 0.0402 at
alpha=1.0, so a **4x stronger perturbation produces 6% more disturbance**. A real
causal response scales with intervention strength; an alpha-independent magnitude is
what a numerical noise floor looks like. So the location claim is solid -- it is a
*difference* between conditions with identical noise characteristics -- but "the
answer depends on this attention at all" is not established. Closing that needs a
positive control the current parameterisation cannot express (it preserves image_mass
by construction): blinding, i.e. scaling the image block toward zero and renormalising
over the text keys.

---

## 2. All-head correlation scan (2026-08-06)

`head_correlation_probe.py`, all 36x32 heads over 1,157 completions / 3,471 observe
steps. Correctness is the trainer's `accuracy_reward` on the model's own greedy
answer (accuracy 0.461), not a first-token match -- capitalisation biases that to 0.38
against a true 0.55. Output in `outputs/head_corr/coldstart_setA`.

### The rewarded heads rank near the bottom

| | L22H28 | L22H31 |
|---|---|---|
| mean_in_v2, step | +0.029 — rank 592/1152 | +0.017 — rank 839 |
| mean_in_v2, completion | +0.046 — rank 321 | +0.020 — rank 821 |
| **auroc, step** | -0.002 — rank **1109** | -0.001 — rank **1132** |
| **auroc, completion** | +0.004 — rank **1076** | +0.028 — rank **1100** |

Under `auroc` -- the metric the `wov0.11` run trained on -- the two rewarded heads are
in the **bottom 4% of all 1152 heads**, with r indistinguishable from zero, in exactly
the configuration the reward used.

### Step-level results are inside the chance envelope

Steps of one completion share a label, so the effective sample size is 1,157
completions, not 3,471 steps:

```
sd(r | H0) = 1/sqrt(1157-3) = 0.0294
Bonferroni threshold over 1152 heads:  |r| >= 0.120

mean_in_v2  step        max |r| 0.0995    heads over threshold:   0 / 1152
auroc       step        max |r| 0.1066    heads over threshold:   5 / 1152
mean_in_v2  completion  max |r| 0.1462    heads over threshold:  11 / 1152
auroc       completion  max |r| 0.1553    heads over threshold:  32 / 1152
```

**Do not read the step-level tables as findings.** The naive threshold using n=3,471
(0.069) would have passed dozens of heads spuriously. Completion level clears the
correct threshold, most convincingly under `auroc`.

### Heads that survive the held-out split

Ranked on odd `row_index`, re-scored on even. Most of the top 20 collapse -- mean
held-out/selection ratio **0.39-0.54**, which is what selecting from 1152 candidates
does to you. These held:

| head | metric / setup | select | **held out** | all |
|---|---|---|---|---|
| L0H19 | auroc, completion | -0.200 | **-0.113** | -0.155 |
| **L1H4** | mean_in_v2, completion | +0.173 | **+0.119** | +0.146 |
| L18H0 | mean_in_v2, completion | -0.158 | **-0.127** | -0.144 |
| L0H9 | mean_in_v2, completion | -0.148 | **-0.124** | -0.135 |
| L0H17 | auroc, completion | -0.138 | **-0.121** | -0.128 |
| L0H10 | auroc, completion | -0.121 | **-0.106** | -0.114 |
| L23H9 | auroc, step | -0.107 | **-0.104** | -0.106 |

**Most survivors are NEGATIVE** -- more overlap with the named objects predicts being
*wrong*. Rewarding those would push the model backwards. `L1H4` is the one strong
positive.

Layers ranking highest at completion level under both metrics: **0, 1, 18, 19, 20, 21,
23, 24**. **L22 is not in the top 8 under either metric.**

### Caveats

- Best |r| is ~0.15, materially weaker than the **0.19-0.28** the offline screen
  reported for `auroc` at (22,28) on saliency_r1_8k / visual_cot. Different corpus,
  different box source, and correctness measured on this model's own greedy chains
  rather than a static collection. Reconcile before treating 0.15 as a ceiling.
- **Layer 0 is suspect.** It is the strongest `auroc` layer and every survivor there is
  negative, but at layer 0 the residual stream is close to raw embeddings, so "attention
  to the object's patches" may be measuring image statistics rather than grounding.

---

## 3. Indirect-flow maps (2026-08-06) — the first surviving signal, and it is small

`flow_correlation_probe.py` over the same 1,157 completions / 3,471 observe steps and
the same DINO unions, so every number is directly comparable with result 2. Three
replacement maps, each scored by `mean_in_v2` and `auroc`, at step and completion
level: `rollout_mean` (layer-wise rollout, heads merged by the mean), `rollout_wnorm`
(the same, merged by `‖Σ_h A^h W_O^h v^h‖`) and `grad` (`‖∂ log P(step)/∂e_j‖`).
Output in `outputs/flow_corr/coldstart_setA`. ~45 s per shard — the whole thing is
under 10 minutes on 8 GPUs, unlike every other probe here.

### The premise fails first, before any correlation

The rollout maps put **less** weight on the objects a step names than on the rest of
the image, at every layer, and it worsens monotonically with depth:

| auroc level (0.5 = the union ranks no higher than the rest of the image) | L0 | L8 | L16 | L22 | L28 | L35 |
|---|---|---|---|---|---|---|
| `rollout_mean` | 0.487 | 0.470 | 0.458 | 0.455 | 0.453 | 0.452 |
| `rollout_wnorm` | 0.490 | 0.458 | 0.439 | **0.434** | 0.432 | **0.430** |

Every cell is below 0.5 by many standard errors (2 SE ≈ 0.004–0.007). The direct map
agrees and is worse at the rewarded heads: **L22H28 sits at 0.410, L22H31 at 0.392**,
against 0.518 averaged over all 1152 heads. So the reward was pointing at two of the
*least* object-aligned heads in the model.

**The gradient map is the exception, and the only map above chance**: `gnorm` 0.545,
`gnorm_ds` 0.553, `gxi` 0.556, `gxi_ds` **0.574**. Differentiating the step's own
log-probability does find it more sensitive to the named object's patches than to the
background. That is the first positive grounding measurement in this line of work, and
it is the one method with no rollout approximation, no α and no head-merge convention.

### Exactly one column survives multiplicity correction

308 tests across the three variants (maps × metrics × step/completion × columns).
Bonferroni at α=0.05 needs |r| ≥ 0.105 at the effective n of 1,157 completions — steps
of one completion share a label, so completions are the unit in both set-ups.

| variant | metric | setup | column | r(all) | p | r(held out) | r(partial) |
|---|---|---|---|---|---|---|---|
| `rollout_wnorm` | auroc | completion | **`inc`** | **+0.117** | 6.3e-05 | **+0.143** | +0.119 |
| `grad` | auroc | completion | `gxi` | −0.098 | 8.9e-04 | −0.104 | −0.124 |
| `grad` | auroc | step | `gxi` | −0.087 | 3.1e-03 | −0.078 | −0.100 |
| `rollout_wnorm` | auroc | step | `inc` | +0.077 | 9.1e-03 | +0.086 | +0.080 |

Only the first clears the threshold. Nothing in `rollout_mean` clears it at all, so the
value-norm head merge is doing the work — the plain mean over heads sees +0.052 for the
same column.

`inc` is the **increment**: the step's own mean flow map minus the map at the token
immediately before it, i.e. what the step's span newly pulled in from the image rather
than what it inherited. It holds up under everything asked of it:

```
permutation, 20k shuffles of the completion labels          p = 0.00005
Spearman instead of Pearson                                 +0.122
1st-99th percentile trimmed (n=1133)                        +0.112
200 random half-splits    median +0.118, range [+0.033, +0.200], both halves same sign 100%
partial: union area / step count / union+steps              +0.117 / +0.119 / +0.119
partial: + base L35 auroc, + base L0 auroc                  +0.102 / +0.105
by completion length   1 step +0.119 (n=161) | 2-3 +0.115 (n=639) | 4+ +0.129 (n=357)
```

**The increment is the carrier, not the cumulative map.** Holding `inc` fixed collapses
the base map's correlation from +0.065 to **+0.025**; holding the base map fixed leaves
`inc` at +0.102. The last layer's weak association is inherited from the increment, not
the other way round.

### What the effect actually is

`inc` averages **0.461** — below chance. Correct completions sit at 0.479, wrong ones
at 0.445; the gap is 0.034, or **0.235 sd**. So the honest statement is *less
anti-grounding goes with being right*, inside a regime that is anti-grounded
throughout. It is not "grounding predicts correctness". The report now prints `level`
beside every r so this cannot be read off wrong again.

Splitting the increment by position: step 0, which differences against a *prompt*
token, sits at chance (0.504); steps ≥1, which difference against a previous step, sit
at 0.439. Both correlate with correctness at about the same strength (+0.073 / +0.064),
so the effect is not an artefact of step-to-step differencing.

### The two survivors disagree in sign, and nobody has explained that

`inc` says more union weight → more likely right. `gxi` — the assumption-free
measure, and the only one above chance in level — says more union weight → more likely
**wrong**, and gets *stronger* when union area and step count are held fixed (−0.124).
This is the same sign puzzle as result 2's held-out survivors, which were mostly
negative, and it is unresolved. Candidate readings, none tested: the gradient is large
where the model is uncertain and leaning on the image, so it tracks difficulty that
union area does not capture; or the two measure genuinely different things and the
rollout's below-chance regime makes its sign hard to interpret at all.

### Comparison with the direct map

Result 2's best single head is |r| = 0.155, but selected from 1152 candidates. The
median head is 0.039 and the 95th percentile 0.094, so the flow increment's 0.117 lands
around the 99th percentile of the direct-head distribution while needing a 308-fold
correction rather than a 1152-fold one. The indirect map does beat the direct map — by
a margin far too small to carry the project's premise.

### The re-scan: `inc` is a ramp, not a spike, and mass does not explain it

`outputs/flow_corr/coldstart_setA_v2`, same 1,157 completions / 3,471 steps / 0 dropped,
now with a per-layer `incL` and the `mass` covariate. Both open caveats close.

**The increment has a clean monotonic layer curve**, so L35 was the top of a ramp rather
than a lone spike:

| `incL`, auroc/completion | 0 | 5 | 10 | 15 | 20 | 25 | 30 | 33 | **34** | 35 |
|---|---|---|---|---|---|---|---|---|---|---|
| r(all) | −0.012 | +0.038 | +0.054 | +0.080 | +0.089 | +0.082 | +0.093 | +0.107 | **+0.124** | +0.117 |
| r(held out) | −0.044 | +0.017 | +0.055 | +0.055 | +0.067 | +0.062 | +0.073 | +0.103 | **+0.138** | +0.143 |
| level | 0.485 | 0.499 | 0.494 | 0.479 | 0.455 | 0.436 | 0.430 | 0.443 | 0.453 | 0.461 |

Against the higher threshold the extra columns demand (216 tests, |r| ≥ 0.108), **two**
now clear it: `inc34` (+0.124, held out +0.138) and `inc35` (+0.117, held out +0.143).
They are the same signal — the two columns correlate at **+0.968** — and `inc33` sits
just under. All three behave identically under stress: permutation p ≤ 3e-4 over 20k
shuffles, 200 random half-splits agreeing in sign 100% of the time, and 0.21–0.25 sd of
separation between correct and wrong.

**Holding the column's own image mass fixed does not remove it**: `inc34` goes +0.124 →
**+0.117** partialled, `inc35` +0.117 → +0.108. That was the last uncontrolled candidate
confound, and it is not the explanation.

Two secondary observations. The increment's *level* curve is non-monotonic where its
*correlation* curve is not — it passes through chance around L5–L10 (`inc7` = 0.512, the
only rollout column anywhere above chance by 2 SE, which at 1 of 36 columns is exactly
the false-positive rate to expect) before falling away with depth. And the two curves run
opposite: the deeper the layer, the more anti-grounded the increment *and* the more its
variation predicts correctness. Nothing here explains that.

`grad` reproduces exactly (`gxi` −0.098, held out −0.105, partial −0.123) and
`rollout_mean` still clears nothing, so the value-norm head merge remains load-bearing.

### The union was never capped, and the level statement depends on it (2026-08-07)

**No probe here applied a maximum union size.** `intervene_probe.py --stage prepare`
configures the per-**box** cap (`max_box_area=0.5`) and never `max_union_area`, which
defaults to `None`; the only steps dropped are unions that are literally empty or
literally the whole grid. All four methods read that one prepared case set, so all four
inherit it. The per-box cap does not bound the union — N boxes each under it can cover
the image between them. (The *training* runs are not uniform on this: `auroc_maxun_03`
and `auroc_maxun_05` exist.)

The unions are large: percentiles 0.19 / **0.538** / 0.688 / 0.819 at p10 / p50 / p75 /
p90. Every map falls monotonically as the union grows — `auroc` by union decile, same
3,471 steps:

| union bin | n | `wnorm` L22 | `wnorm` L35 | `inc35` | `gxi_ds` | L22H28 | L22H31 | mean of 1152 heads |
|---|---|---|---|---|---|---|---|---|
| 0.02–0.19 | 331 | **0.541** | **0.525** | 0.485 | **0.689** | 0.392 | 0.361 | **0.601** |
| 0.32–0.42 | 339 | 0.473 | 0.466 | 0.461 | 0.593 | 0.421 | 0.402 | 0.539 |
| 0.48–0.54 | 339 | 0.441 | 0.436 | 0.463 | 0.577 | 0.427 | 0.410 | 0.524 |
| 0.66–0.73 | 343 | 0.381 | 0.379 | 0.447 | 0.538 | 0.391 | 0.379 | 0.487 |
| 0.82–0.99 | 352 | 0.364 | 0.370 | 0.451 | 0.492 | 0.379 | 0.368 | 0.452 |
| **ALL** | 3471 | 0.434 | 0.430 | 0.461 | 0.575 | 0.408 | 0.391 | 0.518 |

r(union, auroc) over steps: −0.55 all-head mean, −0.50 `gxi_ds`, −0.28 `wnorm` L22,
**−0.039 `inc35`**.

This is not an arithmetic artefact. Midrank AUROC has chance exactly 0.5 for a mask of
any size at a random location: averaged over toroidal shifts every pair contributes
symmetrically, because the mask's autocorrelation is symmetric. The curve is real
map/mask structure. What it does mean is that the two ends answer different questions —
above ~0.5 coverage the union has stopped localising anything (those steps read *"The
image shows a group of people outdoors, possibly at a wedding"*, against *"There's also
a small piece of white onion"* below 0.2).

**So the section heading above is over-general.** "The rollout puts less weight on the
objects a step names, at every layer" holds at the population's median union of 0.54;
it does not hold on localised steps, where `rollout_wnorm` is at 0.537 ± 0.020 (2 SE
clustered by completion) at L22 and the average head is at 0.598. The depth ordering
survives the cap; the below-chance **sign** does not. Note also that the average of all
1152 direct heads was already above chance uncapped (0.518) — what is anti-grounded is
the rollout and the two rewarded heads, not the model's attention as such.

What does not move:

- **The rewarded heads are anti-aligned in every union bin** (0.407/0.378 at u < 0.19).
  Against the all-head mean this *widens* under a cap — 0.60 vs 0.39 rather than 0.52
  vs 0.40.
- **`inc34`/`inc35`**: +0.124/+0.117 uncapped → +0.136/+0.150 at `--max-union 0.5`,
  still clearing the threshold that rises to 0.130 as the sample falls to 807
  completions. `inc35` is the one column essentially flat in union size, which is a new
  argument that it is the right column rather than an artefact.
- **`gxi`'s negative sign**, which sharpens: −0.098 → −0.115 (cap 0.5) → −0.155 (0.25).
- **The causal null** (result 4): re-aggregated at case-level union caps off/0.6/0.5/
  0.35 it gives 14/15/16/16 nominal hits against ~14 expected and 0 over Bonferroni.
  `box − roll` is a same-size same-shape contrast, so union size cancels by
  construction.

`--max-union` now exists on the `report` stage of `flow_correlation_probe.py` and
`head_correlation_probe.py` (default 0 = off, what every number above used), and both
reports open with the decile table on the uncapped data. **Fix a threshold before
looking at `val_natural`** — chosen afterwards it is a researcher degree of freedom, and
the confirmation draw is single use.

### Caveats

- The DEEPSTACK caveat in the probe's docstring applies to both rollout variants and
  not to `grad`, which is another reason the two can disagree.
- `r ≈ 0.12` at ~0.25 sd of separation is not a basis for a training signal.
- Everything above is still **the same 1,157 cases the columns were selected on**.
  `inc34` in particular is a fresh selection from 36 increment columns. The odd/even
  split is internal to that set; a real confirmation needs the image-disjoint
  `val_natural` draw.
- A union cap is also a **selection on step semantics**, not just on mask size: small
  unions are single localised objects, large ones are scene-level statements. "Grounded
  on localised steps" is a claim about a different step population, not a corrected
  measurement of the same one.

---

## 4. Per-head intervention (2026-08-07) — the null holds at head granularity

The direct test of result 1's invalid gate. `intervene_probe.py`, 9 layers x 32 heads
x (box, roll) on 1,157 cases = **842,296 records**, 0 errors, 0 duplicates. Layers
0, 1, 18, 19, 20, 21, 22, 23, 24: those result 2 ranked highest, plus L22 as the
incumbent control.

```
cells (layer, head)                              288      n = 1,157 each
per-case sd 0.0731   SE 0.00215   Bonferroni |box-roll| threshold = 0.00806

cells over Bonferroni                            0 / 288
cells at nominal p<0.05 (|z|>1.96)              15         chance predicts ~14
pooled over all 288 cells                       +0.000022
strongest cell                                  L18H12, -0.00654, z -2.88
```

Fifteen nominal hits against fourteen expected is the textbook signature of no effect.
Per-layer means are all within +-0.0005 and per-layer mean |box-roll| is uniform at
0.0014-0.0021 — no layer separates.

**L0 and L1 come back empty.** They carried result 2's largest correlations, and the
concern was that at near-embedding depth "attention to the object's patches" measures
image statistics rather than grounding. Causally they are the *weakest* layers scanned
(mean |box-roll| 0.0019, max 0.0044, no cell even nominal), which supports the artefact
reading — something result 2 alone could not settle.

**L22 shows the cancellation, without a surviving effect.** Its two rewarded heads rank
4th and 7th of 288 and pull in opposite directions:

| | box - roll | z |
|---|---|---|
| L22H28 | **+0.00495** | +2.38 |
| L22H31 | **-0.00549** | -2.79 |

Neither clears correction, and two of 288 draws landing in the top ten is unremarkable.
But this is exactly the head cancellation that made result 1's gate invalid: at layer
level L22 gave +0.0010 because its heads oppose. The mechanism is real and visible; it
just does not sum to an effect in either head.

### The same 288 cells, ranked by ACCURACY instead of log P

`box - roll` in case-insensitive first-token accuracy (exact-token match is biased low
by capitalisation -- the model writes " Bus" where gold is "bus"):

| rank | layer | head | d accuracy | z | top-1 changed | d log P |
|---|---|---|---|---|---|---|
| 1 | 18 | 9 | **+0.17%** | +1.42 | 4 / 1157 | -0.0011 |
| 2 | 18 | 31 | **+0.17%** | +1.42 | 5 / 1157 | -0.0007 |
| 3 | 19 | 8 | **+0.17%** | +1.42 | 4 / 1157 | +0.0009 |
| 4 | 0 | 2 | +0.09% | +1.00 | 4 / 1157 | -0.0002 |
| 5 | 0 | 16 | +0.09% | +1.00 | 3 / 1157 | +0.0001 |
| 6 | 0 | 26 | +0.09% | +1.00 | 3 / 1157 | +0.0001 |
| 7 | 1 | 3 | +0.09% | +1.00 | 4 / 1157 | +0.0033 |
| 8 | 1 | 7 | +0.09% | +1.00 | 3 / 1157 | -0.0039 |
| 9 | 1 | 24 | +0.09% | +1.00 | 3 / 1157 | -0.0012 |
| 10 | 1 | 26 | +0.09% | +1.00 | 3 / 1157 | -0.0001 |

**+0.17% is two cases of 1,157**, and the whole table spans +-2 flipped answers. Read
it as noise:

```
cells at |z| > 1.96           0 / 288    (chance ~14 -- accuracy is too quantised to have power)
best z                        +1.42      (Bonferroni needs 3.46)
largest LOSS                  -0.17%     same magnitude as the largest gain, 3 cells
pooled over 288 cells         -0.00005
top-1 changed at all          866 / 333,216 pairs = 0.26%
```

Two independent reasons not to trust the ranking. The losses mirror the gains, which is
a symmetric noise distribution. And **the accuracy ranking disagrees with the log P
ranking** -- L18H9 tops this table with d log P *negative*, while L18H12, the strongest
log P cell at z -2.88, does not appear at all. Two orderings of the same 288 cells that
disagree is what you get when both are noise.

The 0.26% tie-rate is structural: the model sits at P ~ 0.999 when right and about -8
nats when wrong, so accuracy is a thresholded readout with very little power here. That
is why log P is the primary measure -- and log P returns 0/288 as well.

### What is now closed, and what is not

Three independent lines agree that overlap at these heads has no causal path to the
answer: the layer-level null with its alpha sweep (result 1), the correlation ranking
(result 2), and 0/288 here. With the benchmark collapses (-11.5% and -8.5%), the reward
has been optimising a quantity that cannot help accuracy.

Still open: **the positive control**. |d logp| does not scale with alpha, so a single
head's attention may simply be too small a perturbation for this readout to register.
Blinding -- scale the image block toward zero and renormalise over the text keys --
remains the test, and needs the ~20-line extension the mass-preserving parameterisation
cannot express.

Note this is an **activation-level** intervention: attention values are overwritten
mid-forward, weights are frozen, no gradient is taken. No experiment run so far changes
weights. That is plan Stage 4 (the supervised attention loss), still unbuilt.

## 5. Indirect-path intervention (2026-08-07) — the positive control finally works, and the answer still does not care

`flow_intervene_probe.py` on the same 1,157 cases. At every layer up to a cutoff, each
head's mass over image-*carrying* keys (image tokens **and** earlier text positions that
absorbed image content) is held fixed and re-allocated toward the keys holding the most
boxed-object content. 32,396 forwards, 0 dropped, ~40 min on 8 GPUs. Output in
`outputs/flow_intervene/coldstart_setA`.

### This is the positive control results 1 and 4 lacked

The edit moves the answer, hard: mean **|Δ log P(gold)| = 0.726 nats**, 8/8 selftest
cases, up to 2.03. Against the direct intervention's 0.0378 at α=0.25 versus 0.0402 at
α=1.0 — a 4× stronger perturbation buying 6% more, which was read as a noise floor.

So the readout is not blind, and the earlier nulls were not "we never moved anything."
That question has been open since 2026-08-05 and is now closed.

The manipulation also lands the right way round. Union-traceable mass at the step's own
positions rises **13–71%**, and it rises *faster than* total image-traceable mass in
every cell, so `ushare` climbs because the numerator grew — the model reading more from
the boxes — not because the denominator shrank.

### Pointing it at the boxes rather than anywhere else buys almost nothing

`box − roll` on log P(gold), paired per case, n=1,157. 12 cells, so Bonferroni needs
|t| ≥ 2.87:

| cut | α=0.25 | α=0.50 | α=1.00 |
|---|---|---|---|
| 8 | +0.0033 (t 1.07) | +0.0041 (t 1.24) | +0.0158 (t 2.83) |
| 16 | +0.0086 (t 2.65) | +0.0093 (t 2.59) | **−0.0488 (t −3.78)** |
| 24 | +0.0068 (t 2.04) | **+0.0126 (t 3.01)** | −0.0031 (t −0.25) |
| 35 | +0.0046 (t 1.37) | +0.0050 (t 1.20) | −0.0005 (t −0.05) |

One cell clears correction with the intended sign, at **+0.0126 nats** — under 2% of the
0.726 the edit moves overall. Normalised by manipulation strength the estimate is flat at
**~+0.2 millinats per 1% of union mass** across every cutoff and both usable α.

**Behaviourally it is nothing.** Over 13,884 box/roll comparisons, **98.96% produce an
identical top-1 token** and correctness differs on **0.14%**. No accuracy cell clears
correction.

### α=1.0 is off-manifold and should not be read

Against the α=0 baseline, at α=1.0:

| cut | box − base | roll − base |
|---|---|---|
| 24 | **+0.671** | **+0.674** |
| 35 | **+0.939** | **+0.939** |

Concentrating *all* eligible mass at every layer improves log P(gold) by ~0.9 nats — and
box and roll agree to within 0.0003, so it has nothing to do with where the mass points.
That is an activation the model never sees, exactly the failure the direct probe's
docstring predicted for α=1. Read α=0.25 and 0.5 only; the −0.0488 outlier at cut 16 sits
in the same unusable regime.

### The depth prediction fails

Result 3's pre-registered test: the correlation ramps with depth (`inc0` −0.012 →
`inc34` +0.124), so if it is causal, editing deeper should buy more.

```
alpha 0.25   cut8 +0.0033   cut16 +0.0086   cut24 +0.0068   cut35 +0.0046
alpha 0.5    cut8 +0.0041   cut16 +0.0093   cut24 +0.0126   cut35 +0.0050
```

A hump at 16–24 and no ramp; cut 35, which edits every layer, is near the bottom. **The
causal effect does not track the correlation curve**, which is what the plan said would
mark the correlation as epiphenomenal.

### The one caveat, and it cuts toward the null

The control is a *weaker* manipulation than the treatment: `d ushare(box)` runs
+0.013…+0.040 while `d rshare(roll)` runs only +0.004…+0.010, roughly 3–4×. Both
redistribute identical attention mass, so this says content from the DINO-boxed objects
propagates into text positions more readily than content from equal-area background —
interesting in itself, and something no correlational measure here would have shown. But
it biases `box − roll` **in box's favour**, and box − roll is still ~+0.01 nats with no
behavioural effect. The bias works against the null and the null holds anyway.

---

## 6. GLIMPSE (2026-08-12) — the map is finally grounded, and it predicts being wrong

`flow_correlation_probe.py --map glimpse` over the same 1,157 completions / 3,471
observe steps and the same DINO unions, so every number below is directly comparable
with results 2 and 3. Output in `outputs/flow_corr/glimpse_screen/glimpse`. 0 cases
dropped, no OOM.

Run as a **screen for a reward that does not exist yet**: the GLIMPSE grounding reward
brief proposes `mean_in_v2`, and `auroc` was added as a second variant. This scan
already computes both, at step and completion level, so one run ranks the two variants
against each other and against chance before either is trained.

Cost, since it is the one map here that is not cheap: 10.1 s per case, 0.198 s per
target token, 62,159 target tokens — **~27 min on 8 GPUs**, against under 10 minutes
for all three of result 3's variants combined. GLIMPSE pays one backward plus a
per-layer eager replay per *target token*, where `grad` amortises one backward over a
whole step.

### The level: the strongest grounding ever measured here

| map | `auroc` level |
|---|---|
| direct, the rewarded heads L22H28 / L22H31 | 0.410 / 0.392 |
| `rollout_wnorm`, L0 → L35 | 0.490 → 0.430 |
| `grad`, best column (`gxi_ds`) | 0.574 |
| **`glimpse`** | **0.567** |
| **`glimpse`, union ≤ 0.5** | **0.626** |

Every result-3 map put *less* weight on the objects a step names than on the rest of the
image. GLIMPSE does not. It is the second map to clear chance and the first to clear it
by a margin, and on the half of steps whose union still localises anything it reaches
0.626. The union curve is monotone over all ten deciles:

| mean union | 0.11 | 0.26 | 0.37 | 0.45 | 0.51 | 0.56 | 0.62 | 0.69 | 0.77 | 0.89 |
|---|---|---|---|---|---|---|---|---|---|---|
| `glimpse` | 0.712 | 0.647 | 0.591 | 0.579 | 0.574 | 0.558 | 0.526 | 0.516 | 0.505 | 0.471 |

`r(union) = −0.487` over steps, close to `gxi_ds`'s −0.50 and unlike `rollout_wnorm`'s
−0.28 — GLIMPSE inherits the gradient map's union dependence, which is what a
gradient-weighted map should do.

### The correlation: negative, and it survives correction

Union ≤ 0.5 (1,534 steps, 807 completions; Bonferroni |r| ≥ 0.0881):

| metric | setup | r(all) | r(held out) | r(partial) | level |
|---|---|---|---|---|---|
| `auroc` | step | **−0.1120** | −0.1257 | −0.1067 | 0.6259 |
| `auroc` | completion | **−0.1003** | −0.1209 | −0.1011 | 0.6232 |
| `mean_in_v2` | step | −0.0532 | −0.0490 | −0.0550 | 1.5574 |
| `mean_in_v2` | completion | −0.0365 | −0.0118 | −0.0532 | 1.5551 |

Both `auroc` rows clear the threshold and survive the held-out half and partialling out
union area, step count and the map's own mass. Uncapped, nothing clears (best is `auroc`
at completion level, −0.0644 against a 0.0735 threshold), so the cap is doing work — but
it is result 3's threshold, argued from unions above ~0.5 having stopped localising the
thing the step names, not a value fitted here.

**So the better GLIMPSE says the model looked at the object it just named, the less
likely the answer is right.** The sign replicates `grad`'s `gxi` (−0.098 raw, −0.124
partialled) and now matches its magnitude — and unlike `gxi`, it clears correction.
Two independent gradient-based map families, same sign, same size.

### What this settles about the two reward variants

**`auroc` is the variant with a real result, and the result is negative.** It is the
only column in this scan with a multiplicity-corrected association with correctness, and
that association runs the wrong way. Training to maximise it is training toward what
goes with being wrong. This is a *stronger* negative than the gradient reward ever
produced, whose −0.098 never cleared result 3's threshold.

**`mean_in_v2` is a null** — |r| between 0.01 and 0.06, clearing nothing at either cap.
Its level reads high (1.275 uncapped, 1.557 capped) but that *rise* on capping is the
mechanical ceiling: `mean_in_v2` is bounded by `n_patches / n_in`, so restricting to
small unions inflates it. Its level is not evidence of grounding the way `auroc`'s is,
and the offline `r +0.38` with box-area fraction says the same thing from the other side.

Two limits on how hard to read this. **r = −0.11 is a reliable sign, not a large
effect** — about 1% of variance. And this is correlational on the frozen cold-start
policy: it says nothing about what training on it does, which is a separate question
that results 1, 4 and 5 answer in the negative for the direct path. The map's own bf16
rounding noise (0.063–0.089, docs/glimpse-handoff.md) attenuates rather than inflates,
so the true |r| is if anything slightly larger.

**What GLIMPSE is good for is the level, not the reward.** 0.626 is the first real
grounding measurement in this line of work; it makes GLIMPSE the right map for
diagnostic use — per-step LOC, `ov_share` — even though it is the wrong thing to hand a
policy to maximise.

---

## 7. GLIMPSE reward weights (2026-08-24) — all four spreads, and the union cap that moves them

`overlap_probe.py --map glimpse --trained-adapter none` on set_a, base cold-start, 40
samples × 8 generations → **1,099 grounded steps / 307 completions**, at the reward's
default knobs (`layer_frac 1.0`, no token cap) and with no union cap. SLURM job 32822918,
8 GPUs, **28 min**. Output in `outputs/overlap_probe/glimpse_spread`, maps stored.

Run to fill one hole: `mean_in` on the GLIMPSE map had never been weighted, so
`--saliency-method glimpse --overlap-metric mean_in` could not be launched at all. Because
`score_steps` records all four `*_raw` metrics on every grounded step whatever the map, the
same run re-derives the other three on identical maps, masks and completions.

### The weights

`w = 0.4 × 0.0086 / sd_per_sample`, anchored on the ATTENTION map's `mean_in` spread — the
one comparison that cannot be paired, so every number here inherits ±25%.

| metric | level | sd/sample | sd_within | w | w with `--max-union-area 0.5` |
|---|---|---|---|---|---|
| `mean_in` | 0.1426 | 0.0268 | 0.0297 | **0.13** | **0.088** |
| `mean_in_v2` | 1.1910 | 0.1407 | 0.1698 | 0.024 | 0.013 |
| `auroc` | 0.5523 | 0.0485 | 0.0512 | 0.071 | 0.060 |
| `logratio` | 0.0998 | 0.1071 | 0.1230 | 0.032 | 0.020 |

Both anchors agree on the new cell — 0.128 from `sd/sample` against 0.0086, 0.132 from
`sd_within` against 0.0098 — the same convergence the `grad`/`auroc` calibration showed.

**The two cells already on record reproduce.** This draw levels `mean_in_v2` at 1.1910
against 2026-08-12's 1.2122 and `auroc` at 0.5523 against 0.5596, on 1,099 steps against
1,077. Their re-derived weights come out at 0.024 and 0.071 against the 0.020 and 0.063 the
runs were launched with — 13–20% high, inside the ±25%, so nothing in flight needs
changing. That agreement is the reason to trust the `mean_in` row beside them.

### The union cap is the fork

`--max-union-area 0.5` is not a refinement of those numbers, it is a different column.
Median union area is 0.569, so the cap skips **61% of the grounded steps** and 32% of the
completions, leaving 428 steps / 210 completions — and the spread of what survives moves
every weight by up to 1.5×. Every glimpse run on record passes the cap while every weight
on record was measured without it. For `auroc` the mismatch happens to land well (0.063
launched against 0.060 measured under the cap); nothing guarantees that for the next
metric, and for `mean_in` it is the difference between 0.13 and 0.088.

`overlap_metric_spread.py` has no `--max-union-area`, so its table is always the uncapped
one. `box_area_frac` on each step record in `probe_merged.json` is exactly the union
fraction the cap gates on, which is what the capped column above was recomputed from.

### `mean_in` is weighted now, and still not recommended

It divides by the map's own peak, so a policy that merely flattens the map scores higher —
the mechanism behind the `wov0.4` hack, and the launcher warns on the combination. What
this run adds is that on this map it is at least not perverse: `mean_in` levels at 0.1409
per completion against the roll placebo's 0.1232, and `logratio` clears its own roll-null
(0.0998 against a chance of 0, 72.3% of steps above it). The GLIMPSE map does put more mass
in the union a step names than in a rolled equal-area control. But `mean_in`'s within-group
spread is within 5% of that placebo's (0.0297 against 0.0283), so the variation GRPO's
advantage actually sees is the size of a direction control's — which is the argument
against training it, independently of the weight now existing.

---

## Why result 1 does not close the question

The plan's Stage 0 -> Stage 1 gate said: *if forcing every head at layer L does nothing,
no single head there will.* **That is invalid.** Forcing all 32 heads at once is a
different manipulation from forcing one: heads can carry opposing contributions that
cancel, and `o_proj` mixes them, so a zero net effect at the layer is fully compatible
with real per-head effects. The gate treated a sum as an upper bound on its parts.

Result 1 therefore stands as a **layer-level** result and nothing more. The per-head
intervention was therefore run anyway — that is result 4, which reaches the same
conclusion at head granularity, and which found the predicted head cancellation at L22
along the way.

## Next

**A. ~~Re-scan with per-layer increments~~ — done 2026-08-07**, written up above.

**B. Confirm `inc34`/`inc35` on held-out images.** Everything so far is selection and
re-test on one set of 1,157 cases. `val_natural` (256 rows, image-disjoint from set_a by
content hash) is the cheap strong claim; the flow probe reads an `intervene_probe`
`prepare` out-dir, so it needs one built for those rows first. Pre-register the two
columns and the direction before running it — the whole point is that nothing gets
selected twice.

**C. ~~Per-head intervention~~ — done 2026-08-07**, and it is result 4 above: 0 of 288
cells survive.

**D. ~~Intervene on the INDIRECT path~~ — done 2026-08-07**, and it is result 5 above:
the edit is causally powerful, `box − roll` is ~+0.01 nats, and 98.96% of comparisons
give an identical answer token. What remains on this thread is **necessity, not
sufficiency**: blinding — scale the image block toward zero and renormalise over the text
keys — which no probe here can express yet. Result 5 tested whether pointing the flow at
the boxes *helps*; blinding tests whether the image is being used at all.

Original plan for D, kept for the command: Every causal result so far (1 and 4) edits the
direct path — the step's own tokens attending to image tokens — which is precisely the
path result 3 says carries little of the traffic. Both nulls are therefore consistent
with never having moved the thing that matters. `flow_intervene_probe.py` edits the
indirect path instead: at every layer up to a cutoff it holds fixed how much each head
reads from image-*carrying* keys (image tokens **and** earlier text positions that
absorbed image content) and re-allocates that mass toward the keys holding the most
boxed-object content. `roll` is the matched-area, wrong-place control, as before.

```bash
bash launch_flow_intervene.sh --stage selftest --gpus 1 \
    --out-dir outputs/flow_intervene/coldstart_setA \
    --cases-dir outputs/intervene_probe/coldstart_setA_v2
bash launch_flow_intervene.sh --stage run --gpus 8 \
    --out-dir outputs/flow_intervene/coldstart_setA \
    --cases-dir outputs/intervene_probe/coldstart_setA_v2
```

1,157 cases x 28 units (4 cutoffs x (1 baseline + 2 conditions x 3 alphas)) = **32.4k**
forwards, ~25–35 min on 8 GPUs. The cutoff sweep turns result 3's layer curve into a
prediction: if the causal effect tracks the correlation ramp, that corroborates it; if
it is flat while the correlation ramps, the correlation is epiphenomenal.

**Read the manipulation columns before the logp column.** Unlike the direct probe, the
actuator (attention) and the measurement (traceable mass) are different objects here, so
a cell whose `d ushare(box)` is ~0 is a failed intervention, not a null about grounding.
The selftest gates on exactly that and must pass before the run.

This still does not answer the positive control that results 1 and 4 leave open, and it
is still activation-level: weights are frozen and no gradient is taken. Blinding, and
plan Stage 4, remain unbuilt.
