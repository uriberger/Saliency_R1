# Handoff — overlap-reward investigation, state as of 2026-08-06

Read this first, then the two pages it points at. Written to hand the thread to a
fresh session without re-deriving anything.

## The question

Does making the model look at the objects its reasoning steps *name* make it more
accurate? The project rewards attention-DINO overlap at **L22 heads 28/31** during
GRPO. Three reward variants were trained on `cold_data/grpo_sets/set_a` (50k):
`mean_in` (wov0.4), `auroc` (wov0.11), `mean_in_v2` (wov0.033).

## What is now established

**1. All three GRPO runs degraded the model, by three different textual routes.**

| | steps/completion | background-phrase share | dup-step frac | train length | lmms-eval vs cold start |
|---|---|---|---|---|---|
| cold start | 3.6 | 13.5% | 0.00 | 237 | — |
| mean_in cp2000 | 13.7 | 75.6% | 0.19 | 342 | **−11.5%** (p3 60.7→34.3 = chance) |
| mean_in_v2 cp1700 | 14.9 | 63.4% | 0.07 | 341 | greedy val 0.547→0.340→**0.172** |
| auroc cp2500 | **1.1** | **97.6%** | 0.00 | **49** | **−8.5%** (p3 → 29.2, below chance) |
| mean_in 8k | 2.0 | 13.2% | 0.00 | 153 | **+3.6%** |

The auroc jump at ~step 2200 is **not** union growth — the union *shrank* (0.565 →
0.465). It is step pruning plus a switch to an always-groundable sentence frame
("The background has ___."). Discriminator between set_a and 8k is `ov_share` =
`w_overlap · sd(overlap)/sd(total reward)` per group: it crosses ~0.65 exactly when
each set_a run turns; the 8k run stays ≤0.53. Details in the memory files.

**2. The structural reason RL could only ever move the text.** The GRPO gradient is
`∇ log π(y|x) · A`; the reward enters as a scalar, so the optimiser can only raise the
log-probability of sampled token sequences. It has no channel to shape attention
directly. Any attention change is a second-order side effect. Hence: an attention term
should be a **differentiable loss, not a reward**.

**3. Per-step faithfulness is nearly absent at the rewarded heads.** ID accuracy —
"given the attention while writing step *i*, can you tell which step's objects it was
about?" — is **0.534** against a **0.953** oracle and a 0.494 shuffled null. No run
improved it.

**4. Causal intervention returns a tight null at BOTH granularities** — layer-level
(36 layers, alpha sweep) and per-head (288 cells, 0 surviving). Note both are
**activation-level**: attention values are overwritten mid-forward with weights frozen
and no gradient taken. **No experiment so far changes weights** — that is plan Stage 4.
Results 1 and 4 in [probe-results.md](probe-results.md) with the statistics and the
caveats. **Do not read the layer-level null as bounding per-head effects** — that
inference was made and retracted; forcing all 32 heads at once is a different
manipulation from forcing one.

**5. The rewarded heads rank ~1100/1152** on correlation with correctness under `auroc`
— the bottom 4% of every head in the model, in exactly the configuration the reward
used. Result 2 in [probe-results.md](probe-results.md).

**6. The indirect-flow maps do not rescue the premise.** Attention rollout — which
follows image content through intermediate text positions, the path a direct map at L22
cannot see — puts *less* weight on the objects a step names than on the rest of the
image, at every layer, worsening with depth (wnorm auroc 0.490 at L0 → 0.430 at L35;
the rewarded heads sit at 0.410/0.392). The gradient map is the only one above chance
(gxi_ds 0.574) and it correlates **negatively** with correctness. Exactly one column
out of 308 clears a Bonferroni threshold: the rollout-wnorm *increment*, r = +0.117,
held out +0.143, and it too sits below chance in level — so the finding is "less
anti-grounding goes with being right", at ~0.25 sd of separation. Result 3 in
[probe-results.md](probe-results.md). The 2026-08-07 re-scan (`coldstart_setA_v2`)
closed both of that result's open caveats and neither rescued it: the increment has a
clean **monotonic layer curve** peaking at L34/L35 rather than a lone spike, and it
**survives holding its own image mass fixed** (+0.124 → +0.117). Two columns clear the
threshold, `inc34` and `inc35`, correlated at +0.968 — one signal, not two. Real, and
small.

**7. No probe capped the DINO union, and the "anti-grounded" level depends on that.**
`prepare` applies the per-*box* cap only; the median step's union covers **54%** of the
patch grid and the top decile 89%. Every map reads lower the larger it gets
(r(union, auroc) = −0.55 for the mean over all 1152 heads). Below 0.19 coverage
`rollout_wnorm` sits at **0.537**, not 0.434, and the average head at 0.598 — so result
3's level claim holds at the median union and not on localised steps. The depth ordering
survives; the sign does not. What survives unchanged: the rewarded heads are below
chance in every union bin, `inc34`/`inc35` hold (+0.136/+0.150 at a 0.5 cap), `gxi`'s
negative sign sharpens, and the causal null is untouched (`box − roll` cancels union
size). `--max-union` is now a `report`-stage flag on both correlation probes, off by
default, and both reports open with the union-decile table. **Pick a threshold before
`val_natural` runs.**

**8. The indirect path is causally live, and the answer still does not care where it
points.** `flow_intervene_probe.py` re-allocates each head's mass over image-*carrying*
keys toward boxed-object content, at every layer up to a cutoff. It moves log P(gold) by
**0.726 nats** on average — **the positive control this project has lacked since
2026-08-05**, and 18× the direct intervention's α=1 response. Union-traceable mass at the
step rises 13–71%, faster than total image mass, so the manipulation lands the right way
round. And `box − roll` is **+0.0126 nats** in the one cell of 12 that clears Bonferroni
with the right sign, with **98.96% of 13,884 comparisons giving an identical top-1
token**. The depth prediction from result 6 fails: the causal effect does not track the
correlation ramp. Stratifying by union area does not rescue it either — below 0.5
coverage **0 of 3,960 comparisons change the answer**, and the largest `t` values sit in
the *worst*-contrast stratum. Result 5 in [probe-results.md](probe-results.md). **Read
α=0.25/0.5 only** — at α=1.0 box and roll both gain ~0.9 nats and agree to 0.0003, which
is off-manifold, not grounding.

**9. Half of result 3 is a fact about the TARGETS, not about the attention.** Two steps of
one chain get DINO unions that sit **72%** of the way from "unrelated given their sizes"
to "as identical as their sizes allow" — and two steps of two *different* chains about the
same image sit at **70%**. The gap is nothing, in all 14 model-by-run cells measured, and
negative in 10 of them. Same on the raw box lists (best-match IoU 0.480 within vs 0.485
across chains) so it is not the patch grid blurring them together. Different image drops to
0.235, so the measure is not saturated. Consequences: a step scores higher on its own mask
than on a sibling's only **52.6%** of the time (`mean_in`; **53.2%** under `auroc`, which
reproduces result 3's 0.534 independently), and running DINO **once per chain** instead of
once per step keeps the within-group reward ranking at rho **0.62**. The per-step structure
of the reward is close to decorative. The 0.953 oracle never contradicted this — it scores
each mask against itself, which stays maximal however alike the masks are. Full numbers and
caveats in [step-box-similarity.md](step-box-similarity.md); it also puts a condition on
open question 4, since a contrastive `L_attn` across a chain's own steps needs those steps'
targets to differ.

Definitions of every map named above — the rollout, the two head merges, the increment,
the gradient map and the intervention edit — are in
[saliency-maps.md](saliency-maps.md), with the notation stated once.

## What is running / pending

- **DONE (2026-08-07)**: per-head intervention over 9 layers x 32 heads, 842,296
  records. **0 of 288 cells survive Bonferroni**; 15 nominal hits against ~14 expected
  by chance. L0/L1 — result 2's strongest correlational layers — are the weakest here.
  L22's two rewarded heads rank 4th and 7th of 288 in *opposite* directions, which is
  the head cancellation that made the layer-level gate invalid, visible but not
  amounting to an effect. Ranked by ACCURACY instead of log P the best cell gains
  **+0.17% = two cases of 1,157**, 0/288 reach even nominal significance, the losses
  mirror the gains, and the two rankings disagree with each other. Top-1 changed in
  0.26% of all pairs. Result 4 in [probe-results.md](probe-results.md).
- **DONE 2026-08-07**: the flow re-scan with per-layer increments and the mass
  covariate, in `outputs/flow_corr/coldstart_setA_v2`. Read it with `--stage report
  --all-columns`; the launcher's own `report.txt` thins 72 columns down to ~18.
- **NEXT**: confirmation runs. `val_natural` (256 rows, image-disjoint from set_a by
  content hash — the strong claim, ~1h) and a fresh disjoint 2,000-row set_a draw
  (`--exclude-cases-dir`, the powered claim, ~8.4h). Then
  `--stage report --out-dir <selection> --confirm-dir <confirmation>`. The
  confirmation set is **single use**.
- **KNOWN BUG, cosmetic**: `Progress(already_done=len(done))` counts every line in a
  shard's results file rather than the units in the current grid, so any run sharing
  an out-dir with a previous grid shows a nonsense percentage and ETA (`1266.7%`
  once). It has misled twice; the underlying work was correct both times.
- **STORAGE**: the `nvr_israel_rlop` project quota is at **79.96T / 80T**. Free space
  oscillates 7–90 GB while other users' jobs cycle. A full filesystem silently
  truncated a file to 0 bytes here (`ast.parse("")` succeeds — verify writes landed
  non-empty).

  A full `du` walk (2026-08-06, 1h40m) says this is **not fixable from this tree**:

  ```
  59T   users/jonathanp/     <- 74% of the whole allocation
  16T   sweep_checkpoints/   <- 20%
   2.3T datasets/
   2.0T users/uberger/       <- 2.5%
  800G  experiments/   272G code/   270G users/gdalal/   218G dockers/
  145G  users/esharony/   80G users/igreenberg/
  ```

  Those top two are **94%** of 80 TB; everything else including all five other users
  is under 5 TB. Deleting every `_merged` model in this repo (289 GB, 16 of 17
  reconstructible via `merge_lora_grpo_qwen3.sh`) recovers 0.36%. The ask belongs with
  `jonathanp` and whoever owns `sweep_checkpoints/` — likely unpruned intermediate
  checkpoints, since pruning 7 runs here removed 120 of them. Caveats for that
  conversation: `du` reports apparent size and Lustre striping can shift it ~10%, and
  nothing here says that 59 TB is abandoned rather than active.

## Environment facts that cost time to rediscover

- **Claude's Bash tool runs on a login node with no GPU.** Hand GPU commands to the
  user; never launch them, and never report a job running without evidence from its
  own log.
- `beta = 0` in every GRPO run — at `beta == 0` the trainer sets `ref_model = None`,
  so there is no KL anchor at all. PPO clipping is also inactive (`num_iterations=1`
  ⇒ ratio ≡ 1, `clip_ratio` 0.000 in every run). LoRA r=16 on `q_proj,v_proj` (+
  `k_proj` since 437899e) is the only implicit constraint.
- A masked overlap reward is **not** neutral: `None → NaN → nansum` scores it 0, and
  with auroc's +0.5 offset that is ~25× the informative within-group spread.
- `train/rewards/think_overlap_reward/mean` under-reports the real per-completion
  reward ~4×; use the logged completion tables.

## Tooling

| file | what |
|---|---|
| `intervene_probe.py` | causal intervention; `prepare` / `selftest` / `run` / `report` / `monitor`. `--stage selftest` gates every run and must pass. |
| `head_correlation_probe.py` | all-1152-head correlation scan of the DIRECT map; `scan` / `report` |
| `flow_correlation_probe.py` | the same correlation test on INDIRECT maps — layer-wise attention rollout (heads merged by mean, or by `‖Σ_h A^h W_O^h v^h‖`) and the input-gradient map. Readout is the last layer, so no layer is selected. `scan` / `report` |
| `test_flow_correlation_cpu.py` | CPU checks for the rollout algebra, the wnorm Gram expansion, the increment and the report |
| `flow_intervene_probe.py` | causal test of the INDIRECT path: re-allocates each head's mass over image-*carrying* keys toward union-carriers, at every layer up to a cutoff. `selftest` / `run` / `report` / `monitor`. Reports `ushare`/`rshare` on every forward — a null with a flat `ushare` is a failed intervention, not a result |
| `test_flow_intervene_cpu.py` | CPU checks for the edit algebra against naive references, the carried scalars, the deepstack re-seed and the report pairing |
| `overlap_probe.py` | the original offline reward probe (generation + per-step breakdown) |
| `step_box_similarity.py` | do a chain's steps get DIFFERENT boxes? CPU-only, reads `probe_merged.json` written with `--store-maps`. Within-chain vs same-image vs different-image mask overlap, plus the mask-swap rescoring. `--verify` checks every metric it recomputes against the value the probe stored |
| `dino_text_sensitivity.py` | the GPU follow-up: re-grounds the same images with wrong / scrambled / empty text and measures how far the boxes move. DINO only — no VLM, no generation. `real` must reproduce the stored boxes or the run is misconfigured |
| `mask_variance_probe.py` | what each MASK SOURCE leaves in the reward: within-group sd, correlation with the box-blind `flatness` statistic, and border coverage, for all four granularities at once. CPU-only, reads `probe_merged.json`, and builds every mask by importing `trl/rewards/overlap_rewards.py` so the probe and the training arm cannot drift. `docs/per-completion-masks.md` is the write-up |
| `--overlap_chain_boxes last` | one DINO call per COMPLETION instead of one per step; the rung between the incumbent and `--overlap_question_boxes`. `last`, not `first`: two chains' opening steps are the most alike masks in the corpus |
| `--overlap_rect_placement interior_hash` | the detector-free arm with a per-completion mask: the rectangle drawn among placements that touch no border patch, from a hash of the completion's text. `interior_centre` is its matched control |
| `test_chain_boxes_cpu.py` | CPU checks for the per-completion grounding: one call per completion at the detector boundary, an identical score to the per-step path given the same boxes, and the whole-completion masking when the chosen step grounds nothing |
| `precompute_question_boxes.py` | acts on that result: grounds each dataset row ONCE, on its question, and writes the boxes for training to read. `--shard`/`--num-shards` + `--merge`; `launch_precompute_question_boxes_job.sh` fans it over a node's GPUs. Stores RAW boxes so the area caps stay run-time knobs; `--box-threshold` and the 512px cap are baked in and the trainer refuses a mismatch |
| `--overlap_question_boxes <file>` | the training side (launchers: `--question-boxes`). Every observe step of a row is scored against that row's one question union, no Grounding-DINO is loaded, and the colocated launcher gives DINO's GPU to training. Changes WHICH steps are scored — a row grounds for all of its steps or none — so `_qbox` is in the run name |
| `test_question_boxes_cpu.py` | CPU checks for both sides: keying, sharding, merge, the loader's threshold/resolution refusals, and that a cached run and a per-step run given the SAME boxes score identically |
| `build_mismatch_bank.py` | the TRAINING-time version of the same question: grounds cold-start chains offline so a run can be trained against boxes from another question about another picture, with no DINO loaded. `--plan` / `--shard` / `--merge` / `--verify` |
| `trl/rewards/mismatch_rewards.py` | that control as a reward (`--mismatch-bank`). One donor row per PROMPT, shared by all 8 rollouts — a donor per rollout adds 0.0117 of draw noise to a 0.0115 reward, i.e. re-runs `--placebo random`. Step count matched inside that donor, nearest length wrapped when it cannot be |
| `docs/mismatch-boxes.md` | the design, every number behind it and how to read the run |
| `docs/per-completion-masks.md` | why a mask that is constant across a generation group cancels out of the advantage, what that costs the two arms now running, and the two flags that fix it |
| `test_intervene_probe_cpu.py` | CPU checks for the intervention algebra, resume, report |
| `docs/saliency-maps.md` | **every map defined in one place** — direct, rollout-mean, rollout-wnorm, increment, gradient, and the intervention edit — with the notation, the scoring, the fixed-vs-swept parameters and the deepstack facts stated once. Start here before reading any probe |
| `docs/attention-intervention-plan.md` | the P0–P5 plan, with its Stage 0→1 gate marked invalid |
| `docs/probe-results.md` | results 4 and 5 in full |

Companion analysis lives in the sibling `vlm_reasoning` repo under `wiki/` —
`overlap-reward-hack-set-a.md`, `hack-resistant-overlap-reward-plan.md`,
`lmms-eval-overlap-comparison.md`.

## Open questions worth new ideas

1. ~~**Is the attention causally live at all?**~~ **CLOSED by result 7.** The direct α
   sweep looked like a noise floor (0.0378 at α=0.25, 0.0402 at α=1.0). The indirect-path
   edit moves log P(gold) by **0.726 nats**, so the readout registers a real causal
   response and the earlier nulls are not blindness. Blinding is no longer needed as the
   positive control, though it remains the cleanest test of *necessity* (result 7 tests
   sufficiency: pointing the flow somewhere specific).
2. **Most heads that survive the held-out split correlate NEGATIVELY** — more overlap
   predicts being *wrong*. Nobody has explained that. Result 6 sharpens it rather than
   settling it: the gradient map, the one measure with no rollout approximation and the
   only one above chance in level, is also negative (−0.098, −0.124 partialled), while
   the rollout increment is positive. The two survivors disagree in sign.
3. **Layer 0 is the strongest `auroc` layer**, but sits near raw embeddings, so it may
   be measuring image statistics rather than grounding.
4. **The supervised route** (plan Stage 4) is unbuilt: teacher-forced chains,
   `KL(π_ref‖π) + λ·L_attn`, with `L_attn` contrastive across the chain's own steps
   (InfoNCE over (step, box) pairs) so a text-independent saliency detector cannot
   satisfy it. This is the only design that can hold the sentence distribution fixed
   while changing attention.
5. **Best |r| is ~0.15**, materially below the 0.19–0.28 the offline screen reported
   for `auroc` at (22,28). Unreconciled.
