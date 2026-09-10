# Replicating EASE from our cold-start checkpoint

> **Status 2026-09-10: in progress on branch `feat/ease-replication`.**
> Source corpora are downloaded (47 GB). The `ease` conda env is building. The
> annotation pipeline, the parquet converter and the training launcher are not
> written yet. **Blocked on a `GOOGLE_API_KEY`** for annotation Step 3.

The plan of record for running [EASE](https://arxiv.org/abs/2605.30912) in *their*
framework on *their* data, changed in exactly one place: it starts from our SFT
cold-start checkpoint instead of stock Qwen3-VL.

## What EASE is

*Attend to Evidence: Evidence-Anchored Spatial Attention Supervision for Multimodal
RLVR* — arXiv:2605.30912, v1 2026-05-29, v2 2026-09-03. Hu, Wang, Wei, Bai, Yu,
Huang, Wang, Wang (Harbin Institute of Technology / Zhongguancun Academy et al.).
EASE = **E**vidence-**A**nchored **S**patial Att**E**ntion.

It adds a **reward-gated auxiliary attention loss** to a DAPO-style multimodal RL
objective. Annotated evidence boxes are converted into a smoothed Gaussian-mixture
target over visual tokens; on trajectories that clear the verifier reward gate, a
KL term pulls response-to-vision attention at one decoder layer toward that target.
The boxes are **privileged training-only metadata** — they are never rendered into
the image and never enter the prompt, so inference is unchanged.

This is the closest published cousin of our overlap-reward work: same evidence-box
signal on the same backbone, but applied to *internal attention as an auxiliary
loss* rather than *to the reward*. That contrast is the reason to run it.

Reported result: **+2.5 to +3.1 average points over DAPO** across Qwen2.5-VL-7B,
Qwen3-VL-4B and Qwen3-VL-8B.

## What they released, and what they didn't

| | |
|---|---|
| Code | ✅ [github.com/Nrich-sunny/Attend-to-Evidence](https://github.com/Nrich-sunny/Attend-to-Evidence), Apache-2.0 |
| Trained checkpoint | ❌ none, anywhere — no HF repo, no GitHub release, not indexed on hf.co/papers |
| Evidence-box annotations | ❌ never released |
| Annotation pipeline code | ❌ not in the repo — described in prose only (Appendix B) |
| Source corpus **names** and counts | ✅ Appendix B.4, Table 5 |

The repo is a fork of [EasyR1](https://github.com/hiyouga/EasyR1) (verl). The
EASE-specific delta is small and self-contained:

| File | |
|---|---|
| `verl/workers/actor/evidence_mask.py` | new, 352 lines — box → Gaussian-mixture target |
| `verl/workers/actor/trainable_attention.py` | new, 220 lines — forward hook capturing pre-RoPE Q/K on the autograd graph |
| `verl/workers/actor/dp_actor.py` | ≤451 changed lines, ~95 added lines evidence/attention-specific |
| `actor/config.py`, `utils/dataset.py` | +58, +49 |

Everything else differs from upstream EasyR1 by ≤17 lines.

## Decisions taken

| Question | Decision |
|---|---|
| Framework | **Theirs** (EasyR1 fork), in a separate conda env. Not ported to our TRL stack. |
| Base model | **Ours**: `checkpoint/coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged` |
| Training data | **Theirs** — rebuild the pools from the five named source corpora |
| Step 3 validator | **Gemini 2.5 Flash-Lite**, as in the paper |

The framework choice was deliberate. Porting the ~570 lines into our GRPOTrainer was
the cheaper option, but running their code unmodified is what makes the result
attributable to EASE rather than to our reimplementation of it.

## The data

Appendix B.4, Table 5 — the annotated **pool**, before training sampling:

| Source corpus | Multi | Single | Total | Public source | License |
|---|---|---|---|---|---|
| ZwZ74K | 0 | 74,000 | 74,000 | `inclusionAI/ZwZ-RL-VQA` | Apache-2.0 |
| ViRL39K | 9,978 | 28,886 | 38,864 | `TIGER-Lab/ViRL39K` | MIT |
| CLEVR | 4,528 | 1,531 | 6,059 | standard release | CC BY |
| SuperCLEVR | 3,992 | 926 | 4,918 | standard release | MIT |
| SpaCE-10 | 1,759 | 2,371 | 4,130 | `Cusyoung/SpaCE-10` | MIT |
| **Total** | **20,257** | **107,714** | **127,971** | | |

**ZwZ74K** is "Zooming without Zooming" ([arXiv:2602.11858](https://arxiv.org/abs/2602.11858),
Wei et al.) — the same group, since Lai Wei and Weiran Huang are EASE co-authors.

### 127,971 is the pool, NOT the training set

The paper never states how many examples it trained on. It says the training
distribution is sampled from the two pools **in a balanced 1:1 ratio**, and the
multi-evidence pool holds only 20,257. So the training set is **at most ~40,514**,
and how far below that they landed is unpublished. No step count is reported either.

At their rollout batch size of 512 over 2 epochs, that ceiling is **≤~158 steps**.

This is the single biggest reproducibility gap. We cannot match their training-set
size; we can only state ours.

### Annotation workload is much smaller than 128k

About 85k of the pool arrives with gold boxes and needs only Step 3 validation:

- **ZwZ-RL-VQA ships a `bbox` column** — no GPT, no Grounding DINO for its 74,000.
- **CLEVR and SuperCLEVR ship gold scene graphs** — boxes are derived, not predicted.
  Per the paper, "existing evidence boxes are normalized and passed through the same
  validation stage."

The full Steps 1–3 pipeline runs on **ViRL39K (38,864) + SpaCE-10 (4,130) ≈ 43k only**.

### ZwZ: take `original_images/`, never `images/`

ZwZ-RL-VQA publishes two image trees. `images/` (178 GB) has the evidence boxes
**burned into the pixels** — that is how ZwZ itself trains. `original_images/`
(48 GB) is unmarked.

EASE requires the unmarked one: *"evidence boxes are never rendered into the image"*,
and `scripts/prepare_ease_dataset.py:54` rejects paths matching
`with_boxes,marked,annotated,visualized` unless `--allow_marked_images` is passed.
So we take `original_images/` plus the parquet `bbox` column as metadata. That is
both the correct input and a 4.7× smaller download.

### Two provenance notes

- **ZwZ-RL-VQA has 110,988 rows; EASE used 74,000.** The subset selection rule is
  undocumented. We must pick our own and record it.
- **SpaCE-10 publishes only a `test` split of 4,132 rows, and EASE reports 4,130
  training examples.** They trained on essentially the whole benchmark. We reproduce
  it, but it is their methodological choice, not a defensible one.

## The annotation pipeline (Appendix B)

Not released as code. Reconstructed from prose:

1. **Evidence phrase extraction** — GPT-4.1-mini enumerates the minimal set of
   visible entities needed to verify the answer. These are temporary localization
   queries, not labels. Drop phrases naming the whole scene, background, the image
   itself, or non-localizable abstractions; merge near-duplicates by string overlap
   and head-noun match.
2. **Box localization** — Grounding DINO is primary, top prediction kept when
   confidence > **δ_det = 0.35**. A locally deployed Qwen3.5-27B gives a complementary
   box from the same phrase. Merged per query: **IoU ≥ 0.7 → union** (conservative);
   disagreement → DINO wins; neither returns a valid box → drop the example.
3. **Quality validation** — Gemini 2.5 Flash-Lite receives image, question, reference
   answer, phrase and proposed box, and judges whether the region holds answer-relevant
   content. Rejected boxes removed. A different model family on purpose, so one model's
   bias does not determine both the query and the box.

Pools split by validated box count: exactly 1 → single-evidence, ≥2 → multi-evidence.

**Their cost: ~$48 total** for 127,971 examples (~$18 GPT-4.1-mini, ~$30 Gemini,
Qwen3.5-27B local and free). Human audit: three master's students over 1,000
instances / 1,586 boxes.

We already run Grounding DINO (`serve_grounding_dino.py`), which is Step 2's primary
model.

## Hyperparameters

From Appendix C.4 and `examples/config.yaml` / `examples/train_ease_dapo_qwen3vl.sh`:

| | |
|---|---|
| Objective | DAPO-style, `clip_ratio_low` 0.2 / `clip_ratio_high` 0.3, KL disabled |
| Learning rate | 1e-6 |
| Epochs | 2 |
| Rollout batch size | 512 |
| Rollout n | 5 |
| Max response length | 1,024 |
| λ_attn | 0.001 |
| Background smoothing α | 0.1 |
| Attention layer | ⌊2L/3⌋ (`layer_index: -1` resolves to this) |
| Attention target | `bbox_weight_mode: gaussian`, `gaussian_sigma_scale: 0.25` |
| Loss | `attn_loss_mode: vision_kl`, `kl_direction: model_to_target` |
| Reward gate τ | 0.5, applied only to reward-positive trajectories |
| Response tokens sampled for the aux loss | ≤64 |
| Eval decoding | greedy, temperature 0.0 |
| `padding_free` | **false** — the attention loss needs genuinely padded batches |

Their reward-gate coverage, for comparison when ours runs: **47.4%** of sampled
rollouts are reward-positive and **92.3%** of groups contain at least one
gate-passing rollout.

## Environment

A **new** conda env, `ease`. It does not touch `saliency_r1_qwen3` or
`saliency_r1_qwen3_vllm`, which are shared by every session and running job.

Separation is mandatory, not tidiness: EASE pins `transformers>=4.54.0,<=4.57.0`,
and Qwen3-VL exists in transformers **v4.57.0 but not v4.56.0** (verified against the
upstream tags). That range therefore collapses to **exactly 4.57.0**. Our other envs
run `transformers 5.13.0.dev0`. They cannot coexist.

| | |
|---|---|
| Python | 3.12 (matches their NGC base image) |
| torch | 2.8.0 + cu128 |
| vllm | 0.11.0 |
| transformers | **exactly 4.57.0** |
| tokenizers | pinned `>=0.22.0,<=0.23.0` |
| flash-attn | 2.8.3, release wheel, not compiled |

Built by `setup_ease_env.sh`. We deliberately do **not** use their Dockerfile as-is:
it points apt and pip at Tsinghua mirrors.

## Layout

| Path | |
|---|---|
| `ease_repo/` | the EasyR1 fork that actually executes; shared + gitignored like `trl_repo` |
| `cold_data/ease/raw/` | downloaded corpora as published |
| `cold_data/ease/images/` | extracted image trees |
| `download_ease_sources.sh` | sbatch, `cpu_datamover` — pulls and extracts the three HF corpora |
| `setup_ease_env.sh` | sbatch, `cpu` — builds the `ease` conda env |

Both `ease_repo` and `cold_data/ease` are symlinked from the central tree via
`.worktree-links`, so deleting the worktree never takes 47 GB of downloads with it.

## Gotchas already paid for

- **`hf download --include` must be repeated per pattern.** Space-separated patterns
  after one `--include` are parsed as positional *filenames*; the CLI warns
  "Ignoring `--include` since filenames have been explicitly set" and silently
  downloads only the first. This would have skipped all 48 GB of ZwZ images.
- **sbatch scripts run from the Slurm spool dir**, so `BASH_SOURCE` does not locate
  the repo. Anchor on `SLURM_SUBMIT_DIR`.
- **vllm 0.11.0 resolves `tokenizers` to 0.23.2**, which transformers 4.57.0 rejects
  at import time. Installing transformers with `--no-deps` does not correct it; pin
  tokenizers explicitly afterwards.
- **`transformers==4.57.0` is not a range, it is a point.** Anything that upgrades
  transformers in this env breaks Qwen3-VL support in one direction and EASE's pin
  in the other.

## Open questions before results mean anything

1. **The in-stack DAPO baseline is not optional.** Their EASE is DAPO-in-EasyR1; our
   runs are GRPO-in-TRL from an SFT cold-start checkpoint. Comparing EASE directly
   against our overlap runs confounds method with framework. Run
   `examples/train_dapo_baseline_qwen3vl.sh` on the same data from the same base, and
   compare **(EASE − DAPO) inside EasyR1** against **(overlap − placebo) inside ours**.
   We already have placebo-length and placebo-random arms, so the design is symmetric.
2. **Benchmark overlap with our harness is POPE and nothing else.** `eval_mini/benchmarks.py`
   registers mme, mmerealworld, pope, realworldqa, mmstar, algopuzzlevqa, chartqa,
   illusionvqa_soft_localization, mathvision_testmini, mmmu_pro_standard, p3,
   scienceqa_img, visulogic. Theirs: HR-Bench, V*, CV-Bench, POPE, HallusionBench-Image,
   MathVerse-V, MathVista, WeMath, MMK12, LogicVista, Geo3K. So score the EASE
   checkpoint with **our** harness and compare to **our** runs; their published table
   is not a reference point.
3. **The 1:1 mixture caps the training set at ~40.5k** and their actual size is
   unpublished. Decide and record ours.
4. **The ZwZ 74k subset rule is unknown.** Ours must be stated explicitly.
5. **Porting caution, if we ever move this into TRL:** `trainable_attention.py`
   captures Q/K *pre-RoPE* off `q_norm`/`k_norm` and reads `position_embeddings` from
   self_attn kwargs with an `args[7]` positional fallback. That signature was written
   against transformers 4.5x and would need checking against our patched 5.x file.

## Next steps

1. Annotation pipeline, Steps 1–3 — **needs `GOOGLE_API_KEY` for Step 3**.
2. Source CLEVR and SuperCLEVR and derive boxes from their scene graphs.
3. Parquet converter into the schema `scripts/prepare_ease_dataset.py` expects
   (question, answer, image path, `evidence_bboxes` in pixel coords).
4. Slurm launcher for `verl.trainer.main` on 8 GPUs, `MODEL_PATH` pointed at our
   cold-start merged checkpoint.
5. The DAPO baseline arm, per open question 1.
6. Score with `run_bench_eval.sh` and compare against our overlap runs.
