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
Also, and **5. the rewarded
heads rank ~1100/1152** on correlation with correctness under `auroc`. Both in
[probe-results.md](probe-results.md) with the statistics and the caveats. **Do not
read result 4 as bounding per-head effects** — that inference was made and retracted;
forcing all 32 heads at once is a different manipulation from forcing one.

**6. The indirect-flow maps do not rescue the premise.** Attention rollout — which
follows image content through intermediate text positions, the path a direct map at L22
cannot see — puts *less* weight on the objects a step names than on the rest of the
image, at every layer, worsening with depth (wnorm auroc 0.490 at L0 → 0.430 at L35;
the rewarded heads sit at 0.410/0.392). The gradient map is the only one above chance
(gxi_ds 0.574) and it correlates **negatively** with correctness. Exactly one column
out of 308 clears a Bonferroni threshold: the rollout-wnorm *increment*, r = +0.117,
held out +0.143, and it too sits below chance in level — so the finding is "less
anti-grounding goes with being right", at ~0.25 sd of separation. Result 3 in
[probe-results.md](probe-results.md).

The 2026-08-07 re-scan (`coldstart_setA_v2`) closed both of that result's open caveats
and neither rescued it. The increment has a clean **monotonic layer curve** peaking at
L34/L35, not a lone spike, and it **survives holding its own image mass fixed**
(+0.124 → +0.117). Two columns now clear the threshold, `inc34` and `inc35`, correlated
at +0.968 — one signal, not two. It is real and it is small.

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
| `overlap_probe.py` | the original offline reward probe (generation + per-step breakdown) |
| `test_intervene_probe_cpu.py` | CPU checks for the intervention algebra, resume, report |
| `docs/attention-intervention-plan.md` | the P0–P5 plan, with its Stage 0→1 gate marked invalid |
| `docs/probe-results.md` | results 4 and 5 in full |

Companion analysis lives in the sibling `vlm_reasoning` repo under `wiki/` —
`overlap-reward-hack-set-a.md`, `hack-resistant-overlap-reward-plan.md`,
`lmms-eval-overlap-comparison.md`.

## Open questions worth new ideas

1. **Is the attention causally live at all?** The α sweep showed |Δ logp| is
   0.0378 at α=0.25 and 0.0402 at α=1.0 — a 4× stronger perturbation buying 6% more
   disturbance, which is a noise floor, not a response. The design lacks a positive
   control; the natural one is **blinding** (scale the image block toward zero and
   renormalise over text keys), which the current mass-preserving parameterisation
   cannot express.
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
