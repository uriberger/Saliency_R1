# Probe results: the intervention null, the all-head scan, and the indirect-flow maps

Results log for the experiments planned in
[attention-intervention-plan.md](attention-intervention-plan.md). Three runs so far,
all on the cold-start model, `cold_data/grpo_sets/set_a`, per-step Grounding-DINO
boxes, the same 1,157 prepared cases in `outputs/intervene_probe/coldstart_setA_v2`.

Read 1 and 2 together. The first is a *layer-level* null; the second shows why that
null does not license the conclusion the plan drew from it. 3 replaces the direct
attention map with three indirect ones and is the first place a signal survives
correction.

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

### Caveats

- The DEEPSTACK caveat in the probe's docstring applies to both rollout variants and
  not to `grad`, which is another reason the two can disagree.
- `r ≈ 0.12` at ~0.25 sd of separation is not a basis for a training signal.
- Everything above is still **the same 1,157 cases the columns were selected on**.
  `inc34` in particular is a fresh selection from 36 increment columns. The odd/even
  split is internal to that set; a real confirmation needs the image-disjoint
  `val_natural` draw.

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

**C. Per-head intervention** on **L0, L1 and L18-L24** -- the layers result 2 flags, plus L22
as the incumbent control. L0/L1 are included despite the near-embedding concern above
precisely because they carry the largest correlations: if that signal is an image-
statistics artefact rather than grounding, the intervention is what shows it.

```bash
bash launch_intervene_probe.sh --stage run --gpus 8 \
    --out-dir outputs/intervene_probe/coldstart_setA_v2 \
    --layers 0,1,18,19,20,21,22,23,24 --head-mode each \
    --conditions box,roll --alphas 1.0
```

1,157 x 9 x 64 new forwards (the alpha=0 baselines for these layers already exist from
result 1) ~= **666k**, about 5h on 8 GPUs at the measured 37.2 it/s -- past the 4-hour
interactive limit, so either split it over two nodes (`--num-nodes 2 --node-index 0|1`,
~2h30m) or expect one resume.

**Before reading its output, `--stage report` needs the odd/even split-half.** 7 layers
x 32 heads = 288 tests is the same selection problem result 2 just demonstrated, and
the report does not yet do it.
