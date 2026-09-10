# LASER on Qwen3-VL-8B: does the pathology it corrects exist here? — 2026-09-10

LASER (*A Corrective Lens for LVLMs via Visual Attention Preservation and Sink
Suppression*, ECCV 2026, [arXiv:2607.01707](https://arxiv.org/abs/2607.01707), Apache 2.0,
`laser_repo/`) adds two attention-derived terms to a GRPO reward. No checkpoint was
released, so the deliverable is a training run — and a training run is the most expensive
thing this project buys.

`../vlm_reasoning/wiki/laser-implementation.md` is the port plan. Its validation ladder
opens with a go/no-go:

> **Does Qwen3-VL-8B actually show these pathologies?** Both rewards only have headroom if
> visual attention decays over the chain and sinks dominate. If Qwen3-VL is already flat
> and sink-free, skip the rest.

This document is that go/no-go. Sections 1–9 are the design, written before the run;
**section 10 is the result.**

Everything here is measured on **the reward's own construction**: eight sampled rollouts
per prompt at the trainer's settings, then one teacher-forced forward over prompt ++
rollout, layer-mean and head-mean over raw softmax probabilities. That is what
`compute_slice_for_sample` computes upstream, so these are LASER's numbers on our model,
not a proxy for them.

---

## 1. What the released code actually computes

The wiki page warns that the paper and the code disagree and says to follow the code. The
code disagrees with the wiki page too, in two places that change the experiment. Read from
`laser_repo/verl/workers/actor/dp_actor.py` and
`laser_repo/verl/utils/reward_score/openr1_verl.py`.

| | wiki page says | `laser_repo` does |
|---|---|---|
| window | 20, stride 10 | **10, stride 5.** The signature default is 20; both call sites set `window_size = 10` |
| sink identification | massive-activation dims `D*` in the **final layer's hidden states**, then a top-k percentile `η` | **no hidden states at all.** `a_bos[j]`, the layer- and head-mean attention from the **last prompt token** to visual token `j`, then `j ∈ S ⟺ a_bos[j] > mean(a_bos) + 2·sd(a_bos)` |
| `α'_t` | `Σ_{j∈V'}` — a sum | **a mean** over non-sink visual tokens |
| `R_supp` | `exp(−γ·max(0, ā_S/ā_V − 1))`, one scalar | per generated token: `exp(−β·max(0, r_t − τ))` with **τ = 0.9**, then **averaged over tokens** |
| `ω` | "folded into the per-reward `scale`" | separate: `ATTENTION_SCORE_WEIGHT = 0.05`, `SUPPRESSION_SCORE_WEIGHT = 0.1`, on top of the scales |

Written out, exactly as the code has it:

```
A[t, j]   = (1/L)·Σ_l (1/H)·Σ_h  softmax(q_t·k^T·s + mask)[j]      j ∈ V
a_bos[j]  = A at the single query position prompt_len − 1
S         = { j : a_bos[j] > mean(a_bos) + 2·sd(a_bos) },   V' = V \ S

alpha[t]  = mean_{j ∈ V'} A[t, j]                       t over response indices 1 .. n−2
w         = alpha.unfold(size=10, step=5).mean(-1)      # 0 if n_query <= 10
R_vis     = 0.1 · Σ_i  g_i · ( exp(−5·(1 − w_i / max(w).detach())) − 0.015 )
            g_i = 1, or the early-weighted exp(−1.5·i/(W−1)) renormalised to sum to W

r_t       = mean_{j∈S} A[t, j] / mean_{j∈V} A[t, j]
R_supp    = 1.0 · mean_t exp(−0.5 · max(0, r_t − 0.9))

total     = acc + 0.3·format + 0.05·R_vis·acc·format + 0.1·R_supp·acc·format + repetition
```

Three consequences worth stating before any number is collected.

- **`R_supp`'s ratio is an enrichment.** `r_t` is the sinks' mean per-patch attention over
  *all* visual tokens' mean per-patch attention — the same quantity
  [sink-location-by-image-type.md](sink-location-by-image-type.md) calls `E`, on a
  different patch set. Every enrichment in that document is directly comparable to `τ`.
- **`τ = 0.9` penalises a uniform map.** A picture whose sinks pull exactly their fair
  share has `r_t = 1.0` and scores `exp(−0.05) = 0.951`, not 1. The reward's ceiling is
  reachable only by pushing the selected tokens *below* their share.
- **`R_vis` cannot see magnitude.** `target_level` is `w.max().detach()`, so a rollout
  whose visual attention is uniformly low scores the same as one that is uniformly high.
  It rewards flatness. Whether that is a problem is an empirical question this probe can
  answer, because it can measure the two apart.

## 2. Why the answer is not obvious in either direction

[sink-location-by-image-type.md](sink-location-by-image-type.md) already measured, on
1,800 pictures across twelve image types, that within the picture attention concentrates
on the **border**: `E_ring` 2.1–3.6, twelve types of twelve, and 99.7% of the 1,152
layer×head cells above 1. That is `R_supp`'s premise looking healthy.

The same document also concluded, against a pre-registered threshold fixed in advance,
that **it is not a sink**: across-query CV 1.2–2.6 where a sink needs ≤ 0.5. `R_supp`
suppresses tokens on the assumption that they absorb attention *regardless of the
question*. If the high-attention tokens here move with the query, suppressing them removes
signal, not noise — and the correctness gate would not catch it, because a reward that
quietly degrades grounding still gets paid on the rollouts that were already right.

So the two halves of the ladder's question point opposite ways, and the discriminating
measurement is not "is there a peak" — that is settled — but "**is LASER's `S` the border,
and does `S` move with the query**".

Finding 1 has no prior here at all. Nothing in this project has plotted visual attention
against generation step.

## 3. The measurement

`n_prompts` prompts × `G = 8` rollouts, sampled at the trainer's own settings
(`temperature 1.0`, `max_completion_length 512`, `num_generations 8` from `run_grpo.sh`).
Generation runs through the fused kernel; one teacher-forced forward per rollout then
produces everything, which is the construction `grpo_trainer_qwen3.py` uses for its own
reward and the one [sink-location §17](sink-location-by-image-type.md) validated.

Per rollout, stored: `A[t, j]` (float16, `n_response × m`), `a_bos[j]`, the row totals per
step, the response text, accuracy, and format validity. Everything below is computed
offline from those, so a threshold can be swept without touching a GPU again.

### 3.1 Finding 1 — does visual attention decay?

- `alpha[t]` against `t`, per rollout normalised by its own maximum window, pooled.
- Per-rollout Spearman `ρ(alpha_t, t)`; the **distribution**, not just its mean. A model
  where half the rollouts rise and half fall has no decay to fix even if the pooled mean
  slopes down.
- Which window is the argmax — first, middle or last. Finding 1 is specifically that
  *early* decay is the damaging kind, and the early-weighted variant only makes sense if
  the peak is early.
- **The share of rollouts with `n_query <= 10`**, which score exactly zero.

### 3.2 The headroom — can GRPO move it?

Decay existing is not sufficient. GRPO learns from *within-group* spread, and this project
has already been bitten by the other side of that: [overlap-reward-hack-set-a.md](overlap-reward-hack-set-a.md)
showed that when accuracy is constant across a group, a reward with a 0.012 spread takes
100% of the advantage and gets amplified ~240× by `scale_rewards`. So:

- within-group sd of `R_vis` and of `R_supp`, over all groups and over **correct-only**
  groups (the gate is multiplicative on `acc`, so wrong rollouts contribute nothing);
- within-group sd of the accuracy term, for scale;
- **the share of groups in which accuracy and format are constant across all 8 rollouts**
  — in those, the attention terms are the entire gradient;
- `r(R_vis, response length)` and `r(R_vis, mean alpha)`. The first is the padding hack
  channel the `penalty` term exists to close; the second says whether `R_vis` is measuring
  magnitude or, as §1 predicts, only flatness.

### 3.3 Finding 2 — is there a sink, and is it the border?

- `|S|` and its share of the patches; `ā_S / ā_V` — the quantity `τ` is compared against.
- `R_supp`'s distribution, and how much of it is above vs below `τ`.
- **The border cross-check the ladder asks for.** `S` against
  `sink_location.ring_set`: the share of `S` on the border, its enrichment there, and the
  agreement between the two masks. If they are the same tokens, sink-location's CV result
  transfers directly and argues against suppressing them.
- **Query-dependence, measured on the same statistic sink-location used.** `S` is defined
  from one query (the last prompt token). Its across-query CV is computed over the
  generated tokens of the *same* rollout, and over the eight *different* rollouts of the
  same prompt. Pre-registered threshold 0.5, taken unchanged from
  [sink-location §16.4](sink-location-by-image-type.md).
- `S`'s stability across the eight rollouts of one prompt: `a_bos` does not depend on the
  rollout at all (it is a prompt-level statistic), so this is a *control* — it must come
  back as exact agreement, and if it does not, the collector is wrong.

## 4. Pre-registered thresholds

Fixed here, before the run, so that the verdict is a lookup and not a judgement call.

| # | claim | passes if |
|---|---|---|
| **T1** | visual attention decays over the chain | median per-rollout `ρ(alpha_t, t) ≤ −0.20` and the bootstrap CI excludes 0 |
| **T2** | the decay is early | the argmax window is in the first third for ≥ 50% of rollouts |
| **T3** | `R_vis` is learnable | within-group sd among correct rollouts ≥ 0.010 reward units after the 0.05 weight |
| **T4** | `R_vis` is not just a length reward | \|`r(R_vis, response length)`\| ≤ 0.3 |
| **T5** | sinks pull more than their share | `ā_S / ā_V ≥ 1.5` |
| **T6** | `R_supp` is learnable | within-group sd among correct rollouts ≥ 0.010 reward units after the 0.1 weight |
| **T7** | the sinks are **sinks**, not peaks | across-query CV of the `S` columns ≤ 0.50 |
| **T8** | `S` is not simply the border | Jaccard(`S`, ring) ≤ 0.5 |
| **C1** | control: `S` is identical across a prompt's 8 rollouts | exact agreement |
| **C2** | control: the collector reproduces stock SDPA | greedy tokens identical, logits no further from the fused kernel than stock eager is |

**T7 and T8 are the ones that matter, and they are written to fail.** If `S` is the border
ring and its CV is 1.2–2.6, then `R_supp` on Qwen3-VL is a reward for looking away from
the tokens the model's queries are aimed at, and it should not be run. Recording that in
advance is the point of writing them down before the numbers exist.

## 5. What would falsify what

- **T1 fails (attention is flat or rises).** `R_vis` has nothing to preserve. It becomes a
  flatness reward on an already-flat quantity, its within-group spread is noise, and by
  [overlap-reward-hack-set-a.md](overlap-reward-hack-set-a.md)'s mechanism that noise
  still captures the whole advantage in saturated groups. Do not run it.
- **T1 passes, T3 fails.** The pathology is real and GRPO cannot see it. Raise `ω`, or
  drop the arm — but a raised `ω` on a low-variance term is exactly the configuration that
  produced the set_a hack, so this is a stop, not a knob.
- **T5 passes and T7 fails.** The expected outcome given §2. `R_supp` is aimed at a
  query-dependent peak. The port should either drop `R_supp` or replace its sink
  identification with one that has query-invariance in it — which is a different method,
  and should be described as one.
- **T7 passes.** Sink-location's conclusion does not survive contact with LASER's own
  criterion, which selects on a single query rather than averaging over the question's
  tokens. That is a real finding about both, and it licenses `R_supp`.
- **T8 fails and T7 fails together.** `S` is the border ring, and the border ring is a
  vision-encoder positional stamp ([sink-location §16.8](sink-location-by-image-type.md)).
  Suppressing it is suppressing the encoder, not the model's attention habits.
- **C1 fails.** Stop; the collector's prompt/response boundary is wrong and no number in
  the report means anything.

## 6. Two things this probe deliberately does not do

- **It does not run LASER.** It measures the inputs its rewards read and computes the
  rewards those inputs imply, on rollouts from an untrained checkpoint. A reward with no
  spread today can acquire spread once training moves the policy. That is the standing
  limitation of every go/no-go of this shape, and the thresholds are set where they are
  because the *cost* on the other side is a multi-day training run.
- **It does not use LASER's data.** LASER trains on 45K filtered from MMR1-RL and
  ReVisual-R1. This runs on `val_natural` / `val_nonnatural`, which are what we would
  actually train against and are image-disjoint from `set_a`/`set_b`, so a GRPO checkpoint
  can be measured with the same command. If the two corpora differ in chain length they
  will differ in `R_vis`, and §3.1's short-rollout share is the number that shows it.

## 7. Cost

One prompt is 8 sampled rollouts plus 8 teacher-forced forwards at batch 1. Budget ~6 s
per prompt on an H100, so 256 prompts is ~25 min on one GPU and ~4 min on eight. No
Grounding-DINO, no judge, no training.

## 8. Files

| file | what |
|---|---|
| `laser.py` | LASER's two rewards, transcribed from `laser_repo` and pure-numpy; the collector that produces `A[t, j]` and `a_bos[j]` from one teacher-forced forward |
| `laser_probe.py` | `--stage selftest / collect / report / monitor` |
| `test_laser_cpu.py` | the rewards against hand-computed values, the window arithmetic, the sink rule, the geometry cross-check against `sink_location` |
| `launch_laser.sh` | shards `collect` over a node's GPUs; selftest gates the run |
| `laser_repo/` | the upstream checkout, read-only reference (Apache 2.0) |

## 9. Caveats stated in advance

- **One checkpoint.** The cold start. LASER's ablation reports gains from both an instruct
  model and a cold-start checkpoint, so the base model is worth a second pass if the first
  is ambiguous.
- **`max_completion_length 512`**, matching our trainer, against LASER's 2048. Windows are
  10 tokens wide, so 512 gives ~100 windows and the difference should not bind — but the
  short-rollout share is reported precisely because that assumption is checkable.
- **Temperature 1.0**, ours, against LASER's 0.7. Higher temperature widens within-group
  spread, so T3 and T6 are measured under conditions that *favour* passing.
- **`S` is defined from one query position.** That is upstream's choice, kept deliberately.
  Averaging over the question's tokens would be a better estimator and a different method;
  §3.3 reports both so the difference is visible rather than assumed away.
- **The correctness gate here is `accuracy_reward` alone.** Our trainer's `reward_funcs`
  are `[think_format_reward, think_overlap_reward, accuracy_reward, openai_reward]`, and
  the last is an LLM judge that credits an answer which is right but does not parse.
  Running it needs a key and a per-rollout API call, which is out of proportion to a
  go/no-go, so it is omitted. The effect is one-directional: judge-only-correct rollouts
  are counted as ungated, which shrinks the pool T3 and T6 are estimated on and cannot
  make either threshold look better than it is.

---

# 10. The result — 2026-09-10

Cold start (`coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged`), 256 prompts — 128
`val_natural`, 128 `val_nonnatural` — × 8 rollouts at the trainer's own settings, 2,048
scored rollouts. One GPU, 35 minutes. Selftest passed, including the check that `A[t, j]`
reproduces transformers' eager attention and that the sink set drawn from it is the same
set. `outputs/laser/coldstart/report.txt`.

**Finding 1 replicates on Qwen3-VL-8B and is the strongest number here. Finding 2's
premise fails exactly where §2 predicted. And the reward built on Finding 1 is, on this
model, mostly a length reward.**

| | | |
|---|---|---|
| **T1** decay | **PASS** | median per-rollout `ρ(alpha_t, t)` = **−0.537** [−0.547, −0.525] |
| **T2** early | **PASS** | peak window in the first third for **86.4%** of rollouts |
| **T3** `R_vis` learnable | **FAIL** | within-group sd × ω = **0.0053**, against 0.010 |
| **T4** not a length reward | **FAIL** | `r(R_vis, length)` = **+0.534**, against 0.30 |
| **T5** sinks pull more | **PASS** | `ā_S/ā_V` = **6.00** [5.89, 6.11], against 1.5 |
| **T6** `R_supp` learnable | **FAIL** | within-group sd × ω = **0.0043**, against 0.010 |
| **T7** sinks are sinks | **FAIL** | across-query CV **1.222** [1.211, 1.233], against 0.50 |
| **T8** `S` is not the border | pass, **on the wrong statistic** | Jaccard 0.033 — but **99.3%** of `S` is on the ring |

## 10.1 Visual attention decays, hard, and early

`alpha_t` falls with the step index in **99.7%** of rollouts. It is not a tendency: `ρ ≤
−0.2` in 95.4%, `ρ ≥ +0.2` in **0.0%**, and the decile spread is −0.70 to −0.30. The peak
window sits in the first third of the chain 86.4% of the time and in the last third 2.1%.

So the pathology LASER describes for Qwen2.5-VL-7B is present in Qwen3-VL-8B, at the
reward's own readout, on the tokens the model actually wrote. Nothing in this project had
measured it before, and it would be worth knowing even if every other line below were a
no-go. The picture takes **0.0297** of a generated token's attention row (against
[sink-location §16.1](sink-location-by-image-type.md)'s 0.087 from the question's tokens),
and `alpha`, a per-patch mean, is 1.87e−4.

## 10.2 `R_vis` measures length, and it measures flatness, and it does not measure looking

Two of the three correlations in §3.2 came back at almost exactly the values §1 predicted,
and the third is worse than expected.

- **`r(R_vis, mean alpha) = −0.045`.** Dead on. `target_level` is the rollout's own peak,
  so `R_vis` is blind to whether the model looked at the picture a lot or barely at all.
  A model that halved its visual attention everywhere would score identically. The paper's
  framing — *visual attention preservation* — is not what the released reward computes.
- **`r(R_vis, response length) = +0.534.`** T4's threshold was 0.30 and this is not close.
  The mechanism is arithmetic, not statistics: a window pays
  `exp(−5·err) − 0.015`, which stays **positive until a window falls below 16% of the
  peak** (`err > ln(1/0.015)/5 = 0.84`). Almost every window clears that, so `R_vis` is
  very nearly `0.1 × n_windows × (a discount)`, and `n_windows` is the response length over
  5. The `penalty` term is the one part of the design meant to close the padding channel,
  and at 0.015 it is two orders of magnitude too small to make an extra window cost
  anything. Median 42 windows here, and R_vis averages 0.50.
- `r(early-weighted, flat) = +0.931`, so `APPLY_EARLY_WEIGHTED_STABILITY` is close to a
  no-op on this data — the choice the wiki page flagged as the one matching Finding 1
  barely moves the number.

## 10.3 It is too weak to teach and strong enough to hack

These are the same fact read twice, and both readings matter.

- **Too weak.** Within-group sd among gated rollouts is 0.106, which at ω = 0.05 is
  **0.0053 reward units** — half the pre-registered floor. And the gate bites harder than
  the sd suggests: only **27.0%** of groups contain two or more gated rollouts, and a
  group with one correct rollout gives an attention term nothing to discriminate between.
- **Strong enough to hack.** In **50.4%** of groups, accuracy *and* format are identical
  across all eight rollouts. In those groups the attention terms are the **entire**
  advantage, and `R_vis`'s within-group sd over all rollouts (0.135) is nearly **twice**
  the accuracy term's (0.077). This is precisely the mechanism
  [overlap-reward-hack-set-a.md](overlap-reward-hack-set-a.md) measured — a small reward
  spread captured 100% of the advantage in saturated groups and was amplified ~240× by
  `scale_rewards` — and here the reward doing the capturing is one that correlates +0.53
  with response length. A length-correlated term taking the whole advantage in half the
  groups is the set_a failure with the sign written on the tin.

## 10.4 `S` is the border, it is tiny, and it moves with the query

`|S|` is **1.06%** of the patches — a median of **2 patches of 160** — and it pulls
**6.0×** the average visual token's attention. T5 passes by a wide margin, so there is
something concentrated there.

It is the border. **99.3%** [99.0, 99.5] of `S` sits on the one-patch ring, at **3.26×**
the ring's area share. T8's Jaccard of 0.033 says otherwise only because Jaccard charges
for the size mismatch between two patches and a third of the grid; **the threshold was
mis-specified, and it is left as recorded rather than rewritten after the fact.**
Containment is the statistic that answers the question T8 names.

And the across-query CV is **1.222** [1.211, 1.233], against the 0.50 that
[sink-location §16.4](sink-location-by-image-type.md) pre-registered and that this document
took unchanged. That is the same 1.2–2.6 band sink-location found for the border, on a
different corpus, with a different selection rule, at the reward's own readout. `S` is a
**peak**, not a sink.

Put together with sink-location's causal arms — the border mark survives `rot180`, follows
the patch embedding under permutation, and arrives from the vision tower with a *smaller*
norm — `R_supp` on Qwen3-VL-8B is a reward for looking away from a **vision-encoder
positional stamp** that the model's queries are deliberately aimed at, on two patches, in a
query-dependent way. The correctness gate does not protect against that: a rollout that was
already right still gets paid for looking away.

## 10.5 What this changes

- **`R_supp`: do not run it.** T5 passes and T7 fails, which §5 named in advance as the
  outcome that says drop the term or replace its sink identification with one that has
  query-invariance in it. The second is a different method and should be described as one.
- **`R_vis`: the finding is worth having, the reward is not.** Finding 1 is real and large
  on our model. But the released reward is blind to magnitude (§10.2), dominated by length
  (T4), invisible to GRPO where it would help (T3), and load-bearing where it would hurt
  (§10.3). Porting it as released would buy the set_a hack again.
- **What is worth building instead**, if the decay is to be acted on: a term with an
  **absolute floor** rather than a self-normalised target — the wiki page's own "if we want
  magnitude, add an absolute floor term" — and a length control that actually costs
  something, since `penalty = 0.015` demonstrably does not. That is no longer LASER, and
  the honest way to run it is as our own reward with LASER cited for the finding.
- **The wiki page needs the §1 corrections**, independently of any of this: the window is
  10 and not 20, and the sink set is an attention-column rule and not massive activations.
  A port written from the page as it stands would implement neither the reward LASER
  released nor the one it describes.

## 10.6 Caveats on the result

- **T3 and T6 are estimated on 69 groups**, because only 22.4% of rollouts clear the gate,
  and the gate here is `accuracy_reward` without the trainer's judge (§9). Both numbers are
  lower bounds on the spread a full reward stack would see. They are not lower bounds by a
  factor of two: the sd among gated rollouts (0.106) is not far off the sd over all
  rollouts (0.135), so widening the gate mostly adds groups rather than spread.
- **One checkpoint, one temperature, one corpus.** The cold start at T = 1.0 on our
  validation sets, not LASER's 45K of MMR1-RL and ReVisual-R1 at T = 0.7. Chain length
  drives `R_vis` directly (§10.2), and a corpus with longer chains would raise it and its
  length correlation together.
- **A no-go on an untrained policy is not a no-go forever** (§6). `ATTENTION_START_STEP`
  exists because LASER expects 20 steps of plain GRPO first, and the spread these terms
  carry at step 20 is not measured here. What is measured is that at step 0 the term is
  half the floor while sitting on half the advantage, and that is the configuration this
  project has already paid for once.
- **`R_supp`'s no-go is the robust one.** It rests on a CV of 1.222 against a threshold of
  0.50 — a factor of 2.4, replicating an independent prior measurement — rather than on a
  spread estimate that more data could move.
