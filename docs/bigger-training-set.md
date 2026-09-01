# A training set larger than saliency-r1-8k that does not invite the reward hack

Written 2026-09-01, from the five GRPO runs that differ only in their training corpus
(all `mean_in`, `w_overlap 0.4`, `token_reduction mean`, L22 heads 28/31, `beta 0`,
`scale_rewards True`, `lr 1e-5` **linear to zero**, `warmup 0`, `gen_batch 48`).

Companion pages: [overlap-reward-hack-set-a](../../vlm_reasoning/wiki/overlap-reward-hack-set-a.md)
(the mechanism), [attention-intervention-plan](attention-intervention-plan.md) §"Why
set_a degenerated and saliency_r1_8k did not" (the `ov_share` claim this page revises).

Everything below is re-measured from wandb: the scalar histories for all five runs, and
1,100–1,300 logged completion-table groups each for the 8k, set_c and set_d (a group =
one prompt's 8 completions = exactly the contrast GRPO acts on). Scripts in
`outputs/hackset_analysis/`.

---

## 1. The five runs, and which ones ran away

| run | rows | `num_train_epochs` | linear-decay length | ran to | `mean_length` min → later | max dup-sentence frac | hacked |
|---|---|---|---|---|---|---|---|
| `saliency_r1_8k` | 8,080 | 3 | **3,990** | 3,990 (3.00 ep) | 146 → 147 | 0.002 | no |
| `set_d` | 8,080 | 3 (`--max_steps 3990`) | **3,990** | 3,255 (2.45 ep) | 164 → 176 (+7%) | 0.003 | no |
| `set_c_ms3900` | 16,160 | 3 (`--max_steps 3990`) | **3,990** | 1,710 (0.63 ep) | still falling | 0.003 | no |
| `set_c` | 16,160 | 3 | **8,080** | 5,280 (1.97 ep) | 174 → 261 (+50%) | **0.093** | **yes, from ~2,400** |
| `50k set_a` | 50,000 | 1 | **8,333** | 2,010 (0.24 ep) | 169 → 321 (+90%) | **0.239** | **yes, from ~1,500** |

`set_c` is the run nobody wrote up. It is the configuration this session was about to
repeat — **twice the 8k's rows, three epochs** — and it degenerated the same way set_a
did, more slowly: overlap reward 0.051 → 0.119 (+131%) with accuracy flat at 0.26,
entropy 0.700 → 0.122, `think_format_reward` 0.995 → 0.974, `clipped_ratio` 0.005 →
0.021, and the duplicated-sentence fraction climbing 0.000 → 0.093.

## 2. The variable that separates them is the length of the LR schedule

`set_c` and `set_c_ms3900` are **the same 16,160 rows in the same order with the same
seed**. The only difference is `--max_steps 3990`, i.e. how fast `lr = 1e-5·(1 − s/N)`
decays. They are indistinguishable at step 500 and already apart at step 1,500:

| step | | `set_c_ms` (N=3990) | `set_c` (N=8080) | 8k (N=3990) |
|---|---|---|---|---|
| 500 | lr / entropy / len | 8.4e-6 / 0.653 / 198 | 9.2e-6 / 0.649 / 199 | 8.4e-6 / 0.645 / 192 |
| 1000 | | 7.2e-6 / 0.600 / 184 | 8.6e-6 / **0.537** / 185 | 7.2e-6 / 0.604 / 178 |
| 1500 | | 6.0e-6 / 0.523 / 166 | 8.0e-6 / **0.429** / 174 | 5.9e-6 / 0.553 / 155 |

Entropy collapse rate tracks the learning rate, and the length attractor follows the
entropy collapse.

**Both run-aways turn between step 1,250 and 1,750.** So do the clean runs pass through
that window — the difference is the learning rate they still have when they get there:

| | step 1,250 | 1,500 | 1,750 | turned? |
|---|---|---|---|---|
| 8k / set_d / set_c_ms (N≈3990) | 6.87e-6 | 6.24e-6 | 5.61e-6 | no |
| set_c (N=8080) | 8.45e-6 | 8.14e-6 | 7.83e-6 | **yes** |
| set_a (N=8333) | 8.50e-6 | 8.20e-6 | 7.90e-6 | **yes** |

Four runs cannot locate a threshold better than *somewhere between 6.9e-6 and 7.8e-6*.
The operational form:

> **Keep the total optimizer-step count near 4,000 with the standard 1e-5 linear-to-zero
> schedule.** That is the only schedule with three clean runs behind it, and it is the
> thing a bigger corpus silently breaks.

`steps = rows × epochs ÷ prompts_per_step`, and `prompts_per_step = gen_batch ÷ 8 =
per_device × train_procs × grad_accum ÷ 8` — which is 6 in every run above.
16,160 rows × 3 epochs ÷ 6 = 8,080 steps, i.e. set_c exactly.

## 3. Two things that do *not* separate them

**Not epochs, and not data repetition.** The 8k completed 3.0 epochs cleanly. Both
run-aways turned at a *lower* epoch count than the 8k ever reached — set_a at 0.18,
set_c at 0.89.

**Not group saturation, and not `ov_share`.** `ov_share = w·sd_within(overlap) ÷
sd_within(total reward)` per group (unscored overlap folded as 0, which is what the
trainer's `nansum` does) — with `scale_rewards=True` every group is renormalised to unit
advantage, so the mean over groups is the fraction of the gradient that is pure
overlap-maximisation. Re-measured on ~1,200 groups per run:

| step bin | 8k | set_d | set_c_ms | set_c | set_a |
|---|---|---|---|---|---|
| 0–499 | 0.228 | 0.221 | 0.201 | 0.236 | **0.462** |
| 500–999 | 0.311 | 0.316 | 0.349 | 0.328 | 0.478 |
| 1000–1499 | 0.245 | 0.317 | 0.560 | 0.425 | 0.575 |
| 1500–1999 | 0.478 | 0.694 | 0.566 | 0.618 | 0.541 |
| 2500–2999 | **0.736** | 0.717 | — | 0.611 | — |
| 3000–3499 | **0.729** | 0.597 | — | 0.676 | — |

(The first two bins carry 590–605 groups per run; from step 1,000 on, the tables were
sampled and each bin is 12–42 groups, so read the later rows as ±0.1.)

The 8k reaches **0.74** — higher than set_c anywhere — and never hacks. The fraction of
groups with uniform accuracy is 0.62–1.00 in *every* run including the clean ones. So the
attention-intervention page's reading (`ov_share` crosses ~0.65 exactly when a run turns;
the 8k stays ≤0.53) does not survive re-measurement at this fold: it describes set_a,
whose cold start really is an outlier at 0.46, but it does not separate set_c from the 8k,
and set_c hacked.

`ov_share` is still the right description of the *direction* the gradient points. What
the runs say is that the direction is present everywhere and the **learning rate decides
whether the policy can travel far enough along it to reach the attractor**.

## 4. Which rows are the fuel

The hack channel is: pad `<think>` with broad, generic, trivially-groundable sentences;
DINO grounds them widely; `mean_in` averages over observe steps and the mean goes up.
Per-source, measured near the cold start (steps ≤ 1200) on two independent corpus draws:

| source | 8k share | acc | accSD | uniAcc | jdgSD | overlap level | **overlap sd_within** | **ov_share** |
|---|---|---|---|---|---|---|---|---|
| flickr30k | 33.6% | 0.006 | 0.014 | 0.96 | 0.175 | 0.042 | 0.0094 | 0.174 / 0.184 |
| gqa | 21.8% | 0.288 | **0.180** | 0.55 | 0.193 | 0.042 | 0.0098 | 0.229 / 0.229 |
| openimages | 10.6% | 0.107 | 0.072 | 0.82 | 0.178 | 0.038 | 0.0087 | 0.255 / 0.175 |
| **docvqa** | 8.3% | 0.300 | 0.124 | 0.69 | 0.136 | **0.085** | **0.0306** | **0.410 / 0.511** |
| **textcap** | 7.9% | 0.374 | 0.175 | 0.58 | 0.132 | **0.063** | **0.0211** | **0.366 / 0.384** |
| v7w | 7.5% | 0.017 | 0.016 | 0.96 | 0.174 | 0.048 | 0.0109 | 0.247 / 0.338 |
| **textvqa** | 4.6% | 0.542 | 0.083 | 0.80 | 0.099 | **0.059** | **0.0210** | **0.636 / 0.616** |
| **infographicsvqa** | 3.7% | 0.319 | 0.056 | 0.85 | 0.120 | **0.064** | **0.0191** | **0.519 / 0.338** |
| cub | 1.0% | 0.642 | 0.196 | 0.53 | 0.196 | 0.044 | 0.0097 | 0.535 / 0.550 |
| vsr | 0.9% | 0.250 | 0.258 | 0.38 | 0.223 | 0.036 | 0.0095 | 0.017 / 0.423 |
| **corpus** | | | | | | | | **0.272 / 0.285** |

(`ov_share` column: the 8k draw / the set_c draw. `cub` and `vsr` have 8–15 groups each
and their numbers are noise; every other row has 40+ and the two draws agree.)

Two readings, both robust across the draws:

1. **The four OCR / document sources carry 2.5–4× the within-group overlap spread** of
   every photographic source (0.019–0.031 against 0.0087–0.0109), and the highest
   `ov_share` (0.34–0.64 against 0.17–0.34). They are 24.5% of the corpus. On scanned
   pages and infographics Grounding-DINO is answering a question it was not trained for,
   and on textvqa/textcap the step text is about *written words*, which grounds just as
   loosely. That variance is not signal; it is the hack's raw material.
2. **flickr30k is a third of the corpus and contributes no accuracy signal at all.** Its
   answers are full sentences (12.5 words, 0% of them ≤3 words), and `accuracy_reward`
   grades by `math_verify` or exact string match — so it scores 0.006 with sd 0.014, and
   96% of its groups have zero accuracy variance. It trains the judge and the overlap
   term and nothing else. Its `ov_share` is nevertheless moderate (0.18) because the judge
   does vary there.

The 24.5% in group 1 is already reachable without a rebuild: the `natural` column plus
`--natural-only` masks the overlap reward to `None` on those rows, which under either
fold gives them zero within-group overlap spread — `ov_share` exactly 0. The set on disk
marks only docvqa and infographicsvqa as non-natural; textvqa and textcap are marked
natural. Projecting the table above with all four masked:

    corpus ov_share  0.272  ->  0.162   (-41%)

Reweighting *among* the photographic sources buys almost nothing by comparison — they all
sit at 0.17–0.29 — so source reweighting is worth doing for the **accuracy** signal
(gqa, vsr and cub are the only sources with live within-group accuracy variance), not for
hack resistance.

---

## 5. The proposal

Three options, cheapest first. All three hold the step count at ~4,000 and all three take
the `natural`-column change; they differ in how the rows are obtained.

### Option A — no build: `set_c ∪ set_d`, one epoch

`set_c` (16,160) and `set_d` (8,080) are already on disk, share the identical feature
schema, and were built image-disjoint from each other and from the 8k. Concatenated:
**24,240 rows, 3× the 8k**, at ~1.7 questions per image.

Verified here rather than taken from the builder: SHA-256 over the stored image bytes of
all 24,240 rows gives 7,160 distinct pictures in set_c and 6,946 in set_d, sharing
**0**. (set_d's 6,946 against its recipe's 6,953 is seven pictures stored twice under
different names — the same thing `build_set_d.py --index` reports for the 8k.)

```
24,240 rows x 1 epoch / 6 prompts-per-step = 4,040 steps
```

Same schedule, same geometry, same wall clock as the 8k reference, **3× the distinct
prompts**, zero build cost. The only new code is a ~20-line concat that also flips
`natural` to `False` on textcap and textvqa. One epoch instead of three is not a real
loss: the 8k's accuracy plateaus by step ~2,500 (epoch 1.9) and its last epoch buys
nothing measurable.

**This is the recommendation** if the goal is "bigger, safely, this week".

### Option B — what was asked for: 2× rows, 3 epochs, doubled generation batch

If each row really should be seen three times, keep the step count at 4,000 by taking
twice as many prompts per step:

```bash
bash launch_grpo_qwen3_overlap_colocated_job.sh \
    --dataset_name .../set_e --grad-accum 16 --natural-only \
    --lora-targets q_proj,v_proj --num-gpus 8
```

`gen_batch = per_device x train_procs x grad_accum = 1 x 6 x 16 = 96` → 12 prompts per
step → `16,160 x 3 / 12 = 4,040 steps`, and the LR-versus-step curve is the 8k's. Each
optimizer step averages over 12 groups instead of 6, which is strictly less gradient
noise. Per-step wall clock roughly doubles and the step count halves, so the run costs
what it costs today. **Smoke-test first**: the banner must print `gen_batch=96`, and vLLM
now gets 96 sequences per request against the 48 the training path is known to sustain
(256 has wedged it before).

Second-choice variant if the batch change is unwelcome: keep `gen_batch 48`, run the full
8,080 steps, and **halve the peak learning rate to `--learning-rate 5e-6`**. Total
learning budget `∫lr ds` = 0.0202 against the 8k's 0.0200 — the same amount of learning
spread over twice as many steps, with the LR never entering the 7–8e-6 band that both
run-aways turned in. Nothing has been run at 5e-6, so this trades a proven schedule for
an argued one.

### Option C — build `set_e`: a fresh 16,160-row corpus, recomposed

Worth the build only if the composition change matters on its own. `build_set_d.py`
already has everything: the archive SHA-256 index, the two-mechanism disjointness proof
against the 8k, and the per-source pools. Extend its exclusion list with set_d's own
images and change two constants.

Free images after excluding the 8k, set_c and the validation draws
(`python build_set_d.py --report`): gqa 49,712 · flickr30k 22,694 · textcap 14,844 ·
textvqa 12,735 · v7w 11,229 · openimages 8,403 · docvqa 8,042 · cub 4,853 ·
infographicsvqa 3,043 · vsr 1,569 — 137k images against the ~14k the recipe needs.

| source | rows | share | vs 8k | `natural` | why |
|---|---|---|---|---|---|
| gqa | 5,000 | 31.0% | +9.2 | true | the only large source with live accuracy variance (accSD 0.180); 49.7k free images |
| flickr30k | 3,000 | 18.6% | −15.0 | true | keep for caption-style diversity; halve it because it grades 0.006 |
| openimages | 2,000 | 12.4% | +1.8 | true | lowest overlap sd_within (0.0087) |
| v7w | 1,500 | 9.3% | +1.8 | true | question-type diversity; 11.2k free images |
| docvqa | 1,340 | 8.3% | 0.0 | **false** | keep for non-natural benchmark coverage, unscored for overlap |
| textcap | 1,280 | 7.9% | 0.0 | **false** | same |
| textvqa | 740 | 4.6% | 0.0 | **false** | same |
| infographicsvqa | 600 | 3.7% | 0.0 | **false** | same |
| vsr | 400 | 2.5% | +1.6 | true | accSD 0.258, the highest measured |
| cub | 300 | 1.9% | +0.9 | true | accSD 0.196 |
| **total** | **16,160** | | | | 24.5% overlap-masked, the same OCR/document share the 8k has |

The non-natural *share* is deliberately unchanged, so nothing is removed from what the
model sees — only from what the overlap reward is allowed to score. Target ~1.15
questions per image, as the 8k has; set_c's 2.26 was suspected once and cleared by
`set_c_ms3900`, but there is no reason to spend the packing.

**Optional pre-build screen, if a GPU hour is available.** The mechanism says the fuel is
*images where a generic sentence grounds broadly*. That is directly measurable with
Grounding-DINO alone (1.6 GB, no LLM, no attention): score every candidate image with one
fixed generic caption — `"the background. a person. a building. trees. a street."` — and
drop the top quartile by union area. Nothing in this repo has measured per-source DINO
union area on the Visual-CoT sources, so this would also be the first direct check of the
claim in §4 rather than the inference from within-group spread.

### Not recommended

**Do not filter to "hard" prompts.** The intuition that set_a hacked because it was easy
is half right about set_a and wrong as a design rule: the 8k is *more* group-saturated
than set_a (uniform-accuracy groups reach 0.94–1.00 against set_a's 0.50–0.80) and it did
not hack. Selecting for difficulty would shrink the pool and buy nothing that §2 does not
buy for free.

---

## 6. What to watch, and when to stop

Length turns 400–900 steps before accuracy shows anything, in both run-aways. Live alarms,
in order of how early they fire:

| signal | clean (8k / set_d) | hacked (set_c / set_a) | abort at |
|---|---|---|---|
| `train/completions/mean_length` vs its running minimum | +0% / +7% | **+50% / +90%** | **+20%** |
| duplicate-sentence fraction in the `completions` table | ≤0.003 | 0.093 / 0.239 | **0.005** |
| `train/entropy` | 0.44 / 0.45 at the end | 0.12 at step 5,280 | below 0.30 |
| `train/rewards/think_overlap_reward/mean` vs step 0 | +23% / +31% | **+131% / +83%** | +60% |
| `train/completions/clipped_ratio` | 0.002 | 0.021 / 0.029 | 0.015 |

Duplicate fraction is not logged today; it is four lines over the `completions` table the
trainer already writes (split on `[.\n]`, keep segments >25 chars, `1 − len(set)/len(all)`),
and it is the only one of these that is unambiguous rather than suggestive.

## 7. What this does not establish

- **Four runs locate the LR threshold only to within 6.9–7.8e-6.** The rule "≈4,000 steps
  at 1e-5 linear" is the configuration with evidence, not a calibrated boundary.
- `set_c_ms3900` **stopped at step 1,710** (0.63 epoch). It proves the schedule diverges
  the two runs on identical data; it does not prove it would have stayed clean to 3,990.
  If Option B is chosen, that run is worth resuming to 3,990 first — it is the cheapest
  possible confirmation and it already exists.
- The §4 per-source reading is correlational, on the model's own chains, and infers "broad
  DINO unions" from within-group overlap spread rather than measuring union area. The
  screen at the end of Option C is what would test it.
- `--natural-only` masks 24.5% of rows, so in groups where accuracy, format and judge all
  tie the advantage becomes exactly zero and the group is skipped. `frac_reward_zero_std`
  will rise off 0.000 — expect roughly `0.245 × tied`, i.e. ~5% early and ~17% late.
  Whether losing those groups costs anything depends on whether the overlap term was
  doing anything in them, which [sharpness-results.md](sharpness-results.md) and
  [next-reward-experiments.md](next-reward-experiments.md) both argue it was not.
- Every run compared here is `--overlap-metric mean_in` on the attention map. The
  schedule result should carry to the glimpse/grad/auroc runs — it is a property of the
  optimiser, not the metric — but no long-schedule run exists for those.
