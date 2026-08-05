# Does making the model look at what it mentions make it more accurate?

The plan of record for `intervene_probe.py`, written 2026-08-05. Stages P0–P5, with
the measurement that motivated each and the gate that decides whether to run the next.

Companion pages in the `vlm_reasoning` repo:
[hack-resistant-overlap-reward-plan](../../vlm_reasoning/wiki/hack-resistant-overlap-reward-plan.md)
(the offline metric screen), [overlap-reward-hack-set-a](../../vlm_reasoning/wiki/overlap-reward-hack-set-a.md)
(the wov0.4 collapse), [grpo-training](../../vlm_reasoning/wiki/grpo-training.md).

## The question

The project's premise is that in reasoning steps the model does not really look at the
objects it names, and that making it look should make it more accurate. Three GRPO runs
on `50k set_a` — `mean_in` (wov0.4), `auroc` (wov0.11), `mean_in_v2` (wov0.033) — raised
their overlap reward 25–60% and **did not** answer the question, for a structural reason:

> The GRPO gradient is `∇ log π(y|x) · A`. The reward enters only as a scalar, so the
> optimiser can do exactly one thing — raise the log-probability of the token sequences
> that scored well. **A policy-gradient reward can only change the text.** Any attention
> change is a second-order side effect of changing which tokens are produced.

That is what the runs did. The text moved enormously and the attention moved a little,
and the accuracy effect — the thing we wanted to measure — was swamped by the damage
the text change did.

### What the three runs actually did (2026-08-05 measurements)

Behaviour on `val_natural`, 30 samples x 8 generations
(`outputs/overlap_probe/20260805-030416-crossrun-val_natural`):

| | steps/completion | background-phrase share | dup-step frac | mean union | train length |
|---|---|---|---|---|---|
| cold start | 3.6 | 13.5% | 0.00 | 0.565 | 237 |
| mean_in cp2000 | **13.7** | **75.6%** | 0.19 | 0.573 | 342 |
| mean_in_v2 cp1700 | **14.9** | **63.4%** | 0.07 | 0.539 | 341 |
| auroc cp2500 | **1.1** | **97.6%** | 0.00 | **0.465** | **49** |

The auroc jump at ~step 2200 is **not** union growth — the union shrank. It is step
pruning plus a switch to an always-groundable sentence frame ("The background has ___.").
Under `mean_in`/`mean_in_v2` the same pressure produces padding instead of pruning,
because a broad generic sentence scores ~21% above the completion mean under those two
and *below* it under auroc.

Costs, all unambiguous:

- `mean_in_v2` greedy val_natural accuracy **0.547 (step 1500) -> 0.340 -> 0.172 (1700)**,
  with `ungraded` flat at 0.000. Sampled T=1 accuracy at the same checkpoint is 0.571 —
  **never judge these runs by sampled accuracy**.
- lmms-eval vs cold start: mean_in cp2000 **-11.5%** mean over 13 tasks (p3 60.7 -> 34.3,
  i.e. chance), auroc cp2500 **-8.5%** (p3 -> 29.2, below chance). The 8k runs: +3.6% / -1.1%.

### Why set_a degenerated and saliency_r1_8k did not

Not group saturation — 8k is *more* saturated (uniform-accuracy groups 0.78 -> 0.98 vs
set_a 0.67 -> 0.81). The discriminator is the overlap term's share of the within-group
reward spread, `ov_share = w_overlap * sd(overlap) / sd(total)`, with masked overlap
counted as 0 (which is what `nansum` at `grpo_trainer_qwen3.py:2355` does):

| steps | mean_in setA | auroc setA | mean_in_v2 setA | mean_in 8k |
|---|---|---|---|---|
| 0–499 | 0.38 | 0.52 | 0.49 | 0.22 |
| 1000–1499 | **0.65** | 0.59 | **0.63** | 0.50 |
| 1500–1999 | **0.72** | **0.71** | **0.67** | 0.53 |
| 2500–2999 | — | **0.84** | — | 0.64 |

It crosses ~0.65 at exactly the step each set_a run turns. Underneath: set_a is 7.6-word
questions with 1-word answers that the model answers at 0.6 with no chain, so `<think>`
is decorative and free to degenerate.

### Three things found in passing, all still true of the current code

1. **`beta = 0` in every run**, never set in any launcher, and at `beta == 0` the trainer
   sets `self.ref_model = None` (`grpo_trainer_qwen3.py:778`) — the reference model is
   never loaded. There is no KL anchor, nothing computed, nothing logged.
2. **PPO clipping is a provable no-op** here: `num_iterations=1` means the sampling policy
   is the current policy, the ratio is identically 1, and `train/clip_ratio/region_mean`
   is 0.000 in every run. The only implicit constraint is LoRA r=16 on `q_proj, v_proj`
   — note `k_proj` is **not** adapted, so K only moves indirectly.
3. **A masked completion is scored 0, not neutral.** `None -> NaN -> nansum`. With
   auroc's +0.5 offset that costs `0.11 x 0.5 = 0.055` against an informative spread of
   `0.11 x 0.02 = 0.002`. Rewarding `metric - chance` fixes it in one line.
   `--max_union_area` makes it acute: the caps mask **65%** (0.5) and **68%** (0.3) of
   completions on set_a, dropping `ov_share` to 0.37 / 0.29 and pushing
   `frac_reward_zero_std` from 0.000 to 0.12–0.20.

### The measurement the plan is built on

Per-step faithfulness, on the step's own DINO union, per-completion aggregated:

| | ID accuracy | DISC |
|---|---|---|
| cold start | **0.534** | +0.0087 [+0.0065, +0.0110] |
| shuffled-map null | 0.494 | -0.0023 |
| **oracle** (map replaced by its own blurred DINO mask) | **0.953** | **+0.225** |
| mean_in cp2000 | 0.512 | +0.0075 |
| mean_in_v2 cp1700 | 0.529 | +0.0100 |

`ID accuracy` = for each step, the normalised rank of `auroc(map_i, M_i)` among
`{auroc(map_i, M_j)}` over the chain's other steps. It answers: *given the attention
while the model was writing this step, can you tell which step's objects it was writing
about?* The cold-start model sits at **4% of the achievable value**, and no run improved
it. That is the premise, quantified — and the reason the intervention below is worth
running before anything is trained.

Caveat carried forward: heads (22,28)/(22,31) were selected because their `mean_in`
**correlates** with correctness. A correlate need not be a cause; a head can be a
read-only indicator of a computation happening elsewhere, in which case shaping it moves
the metric and nothing else — which is exactly the observed pattern. Nothing so far tests
that. Stages 0–2 do.

---

## The intervention

`intervene_probe.py`. For each sample: one greedy chain, observe steps segmented exactly
as the reward does, each grounded with DINO, then the forward re-run with the layer-L
attention of that step's query rows replaced, and the effect on the answer read off.

A mixture over the **image keys only**:

```
A'[k] = (1-a)*A[k] + a*m*T[k]     k in image keys      (T sums to 1 over them)
A'[k] = A[k]                      k in text/sink keys  (untouched)
m     = sum of A over the image keys
```

The row still sums to 1, `image_mass` is preserved exactly at every alpha, and no text or
sink column can be touched by construction. `a=1` is "all the image mass, inside the box,
spread equally" — the supervision target. `0<a<1` keeps every weight strictly positive,
i.e. on the manifold softmax can produce, which matters because a null at `a=1` alone
could just mean layers L+1.. were handed an activation they have never seen.

| target `T` | what it is | what it is for |
|---|---|---|
| `box` | uniform over the step's DINO union | the supervision target |
| `roll` | that union rolled to a random offset, same area | **the key control**: matched area, wrong place |
| `shape` | the step's own in-box weights, renormalised | all mass in box *without* forcing flatness inside it |
| `image` | uniform over the whole image | controls for "any redistribution moves logits" |
| `perm` | the row's image weights randomly permuted | does this head affect the output **at all** |

**Readout** is `log P(gold answer tokens | prompt, chain)`, paired against an `a=0`
baseline run through the identical rebuild at the same layer, so the eager-vs-sdpa
numerical difference cancels. `first_correct` (argmax at the first answer position) is
recorded as the accuracy proxy.

**Scope note.** Teacher-forcing holds the text fixed, so the counterfactual is "if the
model had looked in the right place *while writing this chain*, would it answer better".
That is the right isolation for the hypothesis, but it is not "would a model trained this
way be better" — in reality different attention would also have produced different text.
Stage 2's effect size is the ceiling on the mechanism, not a prediction of the training
outcome.

---

## P0 — prepare (once)

```bash
DIR=outputs/intervene_probe/coldstart_setA
bash launch_intervene_probe.sh --stage prepare --n-samples 1200 --out-dir $DIR
```

Everything downstream reads `cases/shard*.json`, so the chain and boxes are identical
across every stage, layer and condition. 1200 rather than 1000 because ~10–15% drop out
(bad format, no observe steps, nothing groundable); the drop histogram is printed.

## P1 — selftest (gate, do not skip)

```bash
bash launch_intervene_probe.sh --stage selftest --out-dir $DIR
```

`a=0` must reproduce the un-hooked forward (|delta log P| under tolerance, identical
top-1) and `a=1` whole-layer must move it. Rebuilding a module's output from its
attention weights requires `v_proj`, the GQA expansion and `o_proj` to be exactly right;
a mistake there corrupts every number the probe produces and raises nothing.

## Stage 0 — which layers matter at all

All 32 heads of a layer at once, every layer. This bounds the entire search: if forcing
every head at layer L onto the named objects does not move the answer, no single head
there will.

```bash
bash launch_intervene_probe.sh --stage run --out-dir $DIR \
     --layers 0-35 --head-mode layer --conditions box,roll,perm --alphas 1.0
```

~1000 cases x 36 layers x (3 variants + 1 baseline) ~= **144k forwards**.

`perm` is here to make a null informative: it destroys the spatial arrangement while
preserving mass and the multiset of values. If `perm` does not change the answer either,
the model does not care about this layer's visual attention at all and a null on `box`
means nothing.

**Gates**
- `box - roll` > 0, CI excluding zero, at some layers -> Stage 1 on the top 4–6.
- `perm` moves but `box - roll` does not -> the model is sensitive to attention
  perturbation but not to whether attention sits on the *named* objects. Premise
  falsified at the mechanism level. **Stop.**
- Nothing moves anywhere, `perm` included -> distrust the harness before the model.

## Stage 1 — which heads

```bash
bash launch_intervene_probe.sh --stage run --out-dir $DIR \
     --layers <Stage-0 layers> --head-mode each --conditions box,roll --alphas 1.0
```

~1000 x 4 layers x (32 heads x 2 + 1 baseline) ~= **260k forwards**.

The deliverable is a per-head causal map. This is a head-selection criterion never used
before: not box alignment (the 2026-07-27 scan's axis), not correlation with correctness
(the axis that picked 22/28 and 22/31, i.e. the procedure under suspicion). Where the
incumbent heads land in the ranking is its own answer.

**Add before analysing.** 32 heads x 4–6 layers is 128–192 tests, so selection optimism
is real. Split cases by `row_index` parity — select on the odd half, report effect sizes
on the even half. `--stage report` does not do this yet; it is a small addition and
should land before Stage 1 finishes.

**Gate.** At least one head-slot with a confirmed positive `box - roll` on the held-out
half.

## Stage 2 — effect size and the ceiling

```bash
bash launch_intervene_probe.sh --stage run --out-dir $DIR \
     --layers 22 --head-mode 28,31 \
     --conditions box,roll,shape,image,perm --alphas 0.25,0.5,0.75,1.0
```

One invocation per surviving head-set; ~**115k forwards** total.

- the **alpha sweep** turns a point into a curve; monotone means a causal channel rather
  than "any big perturbation shifts the logits";
- **`shape` vs `box`** decides the Stage-4 target — all mass in the box preserving the
  within-box shape, vs forcing uniformity;
- **`image`** and **`perm`** are the nulls.

**The number this produces** — `delta log P(gold)` for `box - roll` at `a=1` — is the
**ceiling on the mechanism**. No training method, supervised or RL, can beat perfectly
forcing the attention. If it is ~0, stop here and every stage below is saved.

## Stage 3 — validate the selected heads on held-out data

Selection happened on set_a. "Our method" is three separate claims, and each needs data
that did not pick the heads.

**(a) Causal replication.**

```bash
DIR3=outputs/intervene_probe/heldout_val_natural
bash launch_intervene_probe.sh --stage prepare --out-dir $DIR3 \
     --dataset <abs>/cold_data/grpo_sets/val_natural --split all --n-samples 256
bash launch_intervene_probe.sh --stage run --out-dir $DIR3 \
     --dataset <abs>/cold_data/grpo_sets/val_natural --split all --n-samples 256 \
     --layers <L> --head-mode <h1,h2> \
     --conditions box,roll,shape,image,perm --alphas 0.25,0.5,0.75,1.0
```

`val_natural` is image-disjoint from set_a by content hash, so it is the clean
disjointness check — but only 256 rows. If Stage 2's effect is marginal, run the powered
replication on `set_b` (80% natural + 20% charts/documents, which also tests whether the
effect is specific to photographic imagery) and keep val_natural for disjointness.
`set_a --split holdout` is the same-distribution control.

*Gate:* same sign, CI excluding zero, on image-disjoint data. If it holds only on set_a
it is a set_a artifact.

**(b) Does the metric at these heads still predict correctness?** The gate that killed
head re-selection in July, re-asked for causally-selected heads. Run
`overlap_probe.py --overlap-layer L --overlap-heads h1,h2` on val_natural and correlate
the per-completion overlap score with `accuracy_reward`, paired against the incumbent
heads on the same completions.

*Flagged in advance:* R3 in
[hack-resistant-overlap-reward-plan](../../vlm_reasoning/wiki/hack-resistant-overlap-reward-plan.md)
reports alignment and predictiveness as near-orthogonal (rho -0.25 to -0.31 on DINO
boxes), and causal influence may track alignment. So there is a real chance the
causally-selected heads correlate *worse* with correctness. That is not automatically
disqualifying — the correlational criterion is what produced the current mess — but it
is a fork to decide deliberately, not to discover in Stage 5.

**(c) Step-discriminability at the new heads.** ID accuracy, as defined above: 0.534 at
the incumbent heads against a 0.953 oracle. If the new heads are also at 0.53 the
supervised loss has just as little to amplify and lambda will have to carry everything;
at 0.7 it is the strongest green light available.

**Tooling gap.** (b) and (c) were run ad hoc during the 2026-08-05 session and are not in
the repo. They need one small script reading an `overlap_probe` merged JSON. Write it
before Stage 3 runs.

**Output:** the frozen configuration for Stage 4 — layer, head set, target form, metric.

## Stage 4 — supervised attention training

Only if Stage 2's ceiling is non-trivial and Stage 3 replicates.

The attention term must be a **loss, not a reward**: as a reward it can only reach the
model through the text (see the top of this page); as a differentiable loss it acts
directly on the attention computation. Softmax is differentiable everywhere, so this is
ordinary; the non-differentiable parts of the current pipeline are the `no_grad` /
`.numpy()` boundary (removable) and AUROC itself, which is a rank statistic and needs a
smooth surrogate. DINO is non-differentiable but only produces the target.

Teacher-force the cold-start model's own chains, with DINO boxes precomputed:

```
L = KL(pi_ref || pi)  +  lambda * L_attn
```

- **KL to a frozen reference**, not CE on self-generated chains — CE on your own samples
  is self-distillation, it sharpens and reduces entropy, the same direction the auroc run
  went. KL is zero-gradient at initialisation and penalises movement either way.
- **`L_attn` should be contrastive across the chain's own steps** — InfoNCE over
  (step, box) pairs, `s_ij = log(mass of step i's attention on box j)`. A per-step
  *absolute* target is satisfiable by a text-independent saliency detector, which would
  score perfectly while leaving ID accuracy at 0.50. The contrastive form closes that by
  construction.
- Supervise the **renormalised image distribution**, not raw logits: softmax is
  shift-invariant per row, so an absolute target on logits is ill-posed. Decompose into
  *where* (`p = A_image / sum A_image`) and *how much* (`image_mass`, itself predictive of
  correctness at r +0.22–0.29).
- **Add `k_proj` to `--lora_target_modules`.** It is currently absent, so K only moves
  indirectly through the residual stream.
- lambda needs a sweep with the benchmarks as readout: too large and it breaks whatever
  the supervised heads were doing, costing accuracy for reasons unrelated to the
  hypothesis.

Because nothing is sampled during training, **the text distribution cannot drift** — which
is the property no RL variant can offer. Not written yet.

## Stage 5 — the accuracy test

Cold-start vs. attention-supervised on the benchmark suite, with the sentence statistics
(observe-step count, step length, background-phrase share, duplicate fraction) verified
unchanged. This is the clean version of what the three GRPO runs were trying to be:

- ID accuracy rises toward the oracle **and** accuracy improves -> hypothesis confirmed,
  and RL becomes an optional fine-tune on top rather than the mechanism;
- localisation improves and accuracy does not -> hypothesis falsified cleanly, with the
  text confound removed.

Either verdict is worth having, and the three GRPO runs could produce neither.

---

## Analysis rules, all stages

- **Primary readout is `box - roll`, paired by row.** `box` alone also moves under any
  large perturbation; only the gap is location-specific.
- **`delta log P(gold)`, not accuracy.** Continuous and paired; accuracy at n=1000 with a
  ~0.55 base rate has SE ~= 0.016 and most answers will not flip. Accuracy flips are the
  secondary readout, with McNemar on the discordant pairs.
- **Stratify by union size.** At `a=1` you relocate roughly `(1-union)*m` of the image
  mass, and the cold-start median union is 0.562 — on broad boxes the intervention is
  barely one. `--stage report` prints a `tight` column (mean union < `--tight-union`,
  default 0.35). Pre-register that split rather than slicing after seeing the pooled
  number.

## Cost

Stages 0–2 total ~520k forwards; Stage 3 adds a fraction. The per-forward estimate is
0.15–0.4 s (seq ~= 700, 8B, one eager layer), so 1.5–4 hours on 16 GPUs for Stages 0–2.
Treat that as a guess: the probe prints a live rate and ETA after the first ~25 units, so
the real number is known a minute after Stage 0 starts and Stages 1–3 can be resized
from it.

## Operational notes

- **Sharding.** One shard per GPU, no IPC. Two nodes: same command on each with
  `--num-nodes 2 --node-index 0|1` and a shared `--out-dir`; shards are numbered globally
  so results and heartbeats interleave without collision.
- **Resume.** Re-run the identical command. `run` skips every
  `(case, layer, head, variant)` already in the append-only `results/shard*.jsonl`, and a
  torn final line from a killed process is discarded rather than crashing the loader.
  `prepare` skips a shard whose case file exists (`--overwrite` to redo).
- **Progress.** Each shard writes a heartbeat; the launcher runs `--stage monitor` in the
  foreground and prints one aggregated line — completed/total, it/s, shards alive, ETA
  (the max over shards, since they run in parallel, not the sum). It exits on completion
  or when every heartbeat goes stale, so it cannot wedge an interactive shell.
- **CPU tests.** `test_intervene_probe_cpu.py` covers the intervention algebra (every
  target is a distribution; image mass preserved at every alpha; strictly positive for
  alpha<1; alpha=0 bit-identical; the scatter touches only the selected heads x rows x
  image columns), the grid construction, the heartbeat/ETA round trip, torn-JSONL
  recovery, and the report recovering a planted effect. It does not cover anything
  needing a GPU — that is what P1 is for.
