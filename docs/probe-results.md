# Probe results: the layer-level intervention null, and the all-head correlation scan

Results log for the experiments planned in
[attention-intervention-plan.md](attention-intervention-plan.md). Two runs so far,
both on the cold-start model, `cold_data/grpo_sets/set_a`, per-step Grounding-DINO
boxes, 1,157 prepared cases in `outputs/intervene_probe/coldstart_setA_v2`.

Read these two together. The first is a *layer-level* null; the second shows why that
null does not license the conclusion the plan drew from it.

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

## Why result 1 does not close the question

The plan's Stage 0 -> Stage 1 gate said: *if forcing every head at layer L does nothing,
no single head there will.* **That is invalid.** Forcing all 32 heads at once is a
different manipulation from forcing one: heads can carry opposing contributions that
cancel, and `o_proj` mixes them, so a zero net effect at the layer is fully compatible
with real per-head effects. The gate treated a sum as an upper bound on its parts.

Result 1 therefore stands as a **layer-level** result and nothing more. The per-head
intervention is still required, and result 2 is what selects the layers for it.

## Next

Per-head intervention on **L18-L24** -- the layers result 2 flags, plus L22 as the
incumbent control, excluding L0/L1 for the near-embedding concern above:

```bash
bash launch_intervene_probe.sh --stage run --gpus 8 \
    --out-dir outputs/intervene_probe/coldstart_setA_v2 \
    --layers 18,19,20,21,22,23,24 --head-mode each \
    --conditions box,roll --alphas 1.0
```

1,157 x 7 x 64 new forwards (the alpha=0 baselines for these layers already exist from
result 1) ~= **518k**, about 3h50m on 8 GPUs at the measured 37.2 it/s.

**Before reading its output, `--stage report` needs the odd/even split-half.** 7 layers
x 32 heads = 224 tests is the same selection problem result 2 just demonstrated, and
the report does not yet do it.
