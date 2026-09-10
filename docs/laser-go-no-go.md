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
