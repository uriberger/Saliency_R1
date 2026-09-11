# Replicating EASE from our cold-start checkpoint

> **Status 2026-09-10: runnable, on branch `feat/ease-replication`.**
> The route taken is **their framework on *our* corpus** — saliency-r1-8k already
> ships evidence boxes, so EASE's unreleased annotation pipeline is off the critical
> path entirely and the `GOOGLE_API_KEY` block is gone. Data is built (7,984 train /
> 95 val), the `ease` env works, the cold-start checkpoint is staged for
> transformers 4.57, and both training arms have launchers. **Nothing has been
> trained yet.** See [Running EASE on saliency-r1-8k](#running-ease-on-saliency-r1-8k).
>
> The original plan — rebuilding their five source corpora and re-running their
> three-step annotation pipeline — is still documented below and still valid, but it
> is now the *slower alternative*, not the next step. The 47 GB of corpora are
> downloaded and keep.

The plan of record for running [EASE](https://arxiv.org/abs/2605.30912) in *their*
framework, changed in exactly two places: it starts from our SFT cold-start
checkpoint instead of stock Qwen3-VL, and it trains on saliency-r1-8k instead of
their evidence pools.

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
| Training data | **Ours**: `peterant330/saliency-r1-8k`, which already carries boxes |
| Reward | Their rule matcher with **our gpt-4o-mini judge** behind it |
| Step 3 validator | **Gemini 2.5 Flash-Lite**, if the five-corpus rebuild is ever run |

The framework choice was deliberate. Porting the ~570 lines into our GRPOTrainer was
the cheaper option, but running their code unmodified is what makes the result
attributable to EASE rather than to our reimplementation of it.

The data choice was revised on 2026-09-10. Training on *our* corpus rather than a
reconstruction of theirs makes EASE and our overlap runs differ in exactly one
thing — where the box signal enters, an auxiliary attention loss versus the reward
— on the same 8k rows from the same cold start. That is a sharper contrast than
matching their data could have given, and it deletes the annotation pipeline, the
Gemini validation, the ZwZ subset rule and the `GOOGLE_API_KEY` block along with it.

## Running EASE on saliency-r1-8k

### Why this works at all

`peterant330/saliency-r1-8k` ships a `bbox` column. Every one of its 8,080 rows has
a box — measured, not assumed: 0 empty, 0 unparseable, 0 degenerate, across all ten
source corpora. The corpus *is* an evidence-box corpus, so EASE's Steps 1–3 have
nothing to do.

The box is a JSON string of four floats normalized to `[0, 1]`, and it is a **union**
of whatever boxes the source corpus carried (`union_bbox` in `build_grpo_sets.py`).
So every row is single-evidence, K=1. The 1:1 single/multi mixture the paper samples
has no counterpart here, and the per-entity Gaussian mixture EASE builds for its
multi-evidence pool degenerates to one component. That is the largest fidelity cost
of this route and it should be stated in any writeup.

### The four commands

```fish
bash patch_ease_repo.sh                                   # once, additive
bash stage_ease_checkpoint.sh                             # once, ~seconds
sbatch --cpus-per-task=32 --time=03:00:00 prepare_ease_saliency_data.sh
bash launch_ease_train.sh --arm ease --exp preflight --preflight   # CPU, ~2 min
env NVIDIA_API_KEY=$NVIDIA_API_KEY bash launch_ease_train_job.sh --arm ease --exp ease_8k
env NVIDIA_API_KEY=$NVIDIA_API_KEY bash launch_ease_train_job.sh --arm dapo --exp dapo_8k
```

Both arms, always. `(EASE − DAPO)` inside EasyR1 is the only quantity comparable to
`(overlap − placebo)` inside ours; a lone EASE number confounds method with framework.

### What the data build produced

| | |
|---|---|
| rows exported | 8,079 of 8,080 |
| train / val | **7,984 / 95** |
| distinct images | 6,713 (1.20 questions/image) |
| boxes per row | 1, for every row |
| boxes clamped into [0,1] | 31 |
| rows dropped | 1 |
| prompt_length | median ~343, max ~360, against `max_prompt_length` 2048 |

The 31 clamps and the 1 drop are the same phenomenon at two magnitudes. Upstream
`round(x, 3)` at the image edge pushes a coordinate slightly past 1.0 (max 1.002–1.005),
and their `box_to_pixels` decides normalized-vs-pixel by `max(abs(coords)) <= 1.0` —
so an unclamped 1.002 would be read as *pixels* and collapse into a sub-pixel box in
the top-left corner, silently, because the result is still a valid non-degenerate box.
The single drop is `gqa` question 249365, whose box is `[1.064, 0.48, 1.334, 0.749]`:
genuinely off the right edge of its image, not a rounding artifact, and zero-width
after clamping. Their converter would have kept it as a 0.27-pixel box.

Prompt lengths land at ~343 tokens against a 2048 cap, so `filter_overlong_prompts`
never fires and `truncation: error` cannot trigger. Worth knowing, because
`data.min_pixels` is 262144 and saliency-r1-8k ships images at a long side of ≤512 —
their loader **upscales** nearly every picture. Both arms see that, but our overlap
runs did not.

### The reward, and why it is not theirs

`ease/reward_function/judged_perception.py` runs their `perception.py` matcher first
(imported from `ease_repo/`, not copied) and falls through to our gpt-4o-mini judge
only when the rule scores 0. `accuracy = max(rule, judge)`, so it is ≥ their reward
on every sample and never below it.

The reason is flickr30k: 2,715 of the 8,080 rows, and **exactly one** of those 2,715
has a gold answer of ≤3 words. Under a rule-only reward a third of the corpus scores 0
on essentially every rollout, which costs twice — the GRPO group has no advantage
spread, *and* the sample never clears EASE's τ=0.5 gate, so it contributes no
attention supervision either. A third of the corpus would burn rollout compute to
teach nothing.

Per source, share of rows whose gold answer is ≤3 words:

| source | rows | ≤3 words |
|---|---|---|
| flickr30k | 2,715 | 0% |
| gqa | 1,765 | 100% |
| openimages | 860 | 100% |
| docvqa | 670 | 84% |
| textcap | 640 | 96% |
| v7w | 610 | 86% |
| textvqa | 370 | 92% |
| infographicsvqa | 300 | 94% |
| cub | 80 | 100% |
| vsr | 70 | 100% |
| **total** | **8,080** | **63%** |

`--no-judge` gives their reward byte-for-byte. Pass it to **both** arms or neither.

### The two changes to their code

`patch_ease_repo.sh` makes both, and neither is in the EASE method.
**`verl/workers/actor/` is untouched** — `evidence_mask.py`,
`trainable_attention.py` and `dp_actor.py` are the method itself.

1. **`verl/workers/reward/function.py`** — add `question` and `data_source` to the
   dicts `AutoRewardManager` hands the reward function. Their interface passes only
   `{response, response_length, ground_truth}`, which is all a rule matcher needs and
   not enough for a judge. Additive, so their own `perception.py` is unaffected and
   the DAPO arm runs on unmodified behaviour.

2. **`verl/utils/vllm_utils.py`** — `from vllm.lora.lora_model import LoRAModel` does
   not resolve under **vllm 0.11.0, the version their own Dockerfile pins**: 0.11.0
   keeps `LoRAModel` in `vllm.lora.models`, and only a later refactor split it out.
   The import is unconditional and sits under `verl/workers/rollout/__init__.py`, so
   the trainer dies on `import verl` — before any config is read, and whether or not
   LoRA is used (it is not; `lora.rank` is 0). Replaced with a `try`/`except` that
   accepts either path. **Their repo as published cannot start under its own pin.**

### Preflight

`launch_ease_train.sh --preflight` runs `verify_ease_setup.py` against the exact
override list the real command would use, on CPU, in about two minutes: verl
imports, the config parses through `deep_post_init()` (OmegaConf rejects unknown
keys, so a mistyped override is caught here rather than 20 minutes into an
allocation), the `RLHFDataset` builds against the staged checkpoint, and EASE's own
`get_attention_target_distribution` runs on real rows.

That last check is the one worth having. If `bbox`, `image_height` or `image_width`
failed to survive the parquet → dataset → actor path, the target silently falls back
to uniform over vision tokens and the aux loss trains toward nothing. Measured on
our val rows: 252 vision tokens per image, all mass inside the vision span, and the
peak token carries 28–149× uniform for boxes covering 0.2–4.9% of the image (5.8× for
a box covering 40.7%). The target concentrates where the box is.

It exercises neither FSDP, vLLM, nor the judge.

### The checkpoint had to be restaged

Our merged checkpoints are written by transformers 5.13.0.dev0; the `ease` env is
pinned to exactly 4.57.0. Three metadata files changed schema between them and the
model will not load without fixing all three:

| file | 5.x | 4.57 |
|---|---|---|
| `config.json` | `text_config.rope_parameters{rope_theta,…}` | `text_config.rope_scaling` + `rope_theta` |
| `config.json` | `vision_config.model_type: qwen3_vl_vision` | `qwen3_vl` |
| `tokenizer_config.json` | `extra_special_tokens` is a list | must be a dict — 4.57 calls `.keys()` |
| `processor_config.json` | nests image/video processors | separate `preprocessor_config.json`, `video_preprocessor_config.json` |

Without the first, 4.57 dies in `Qwen3VLTextRotaryEmbedding` with
`'NoneType' object has no attribute 'get'`.

All three are metadata; the weights are unaffected. `stage_ease_checkpoint.sh` takes
the stock `Qwen/Qwen3-VL-8B-Instruct` 4.x metadata, symlinks our 17 GB of weights
beside it, and refuses to stage unless the stock chat template is byte-identical to
ours, the tokenizer vocab/merges/added-tokens match, and all **750** tensors line up
by name and shape. All four checks pass for the cold start.

One more 4.57 quirk, worked around inside that script: `AutoModelForImageTextToText`
cannot resolve `Qwen3VLForConditionalGeneration` out of the lazy module until
`transformers.models.qwen3_vl.modeling_qwen3_vl` has actually been imported. verl hits
the same path at `fsdp_workers.py:205`, so if a run dies there, that is why.

### Smoke run, 2026-09-10 (3 steps, 8 GPUs, rollout batch 16)

Clean. The whole path runs: FSDP + vLLM load the staged checkpoint, the aux loss
fires, the judge answers.

**The judge decision is confirmed by the gate, not just by the reward.**

| step | rule alone | with judge | gate passed | `attn_mask_loss` |
|---|---|---|---|---|
| 1 | 0.125 | **0.447** | 41/80 (51%) | 3.422 |
| 2 | 0.275 | **0.572** | 52/80 (65%) | 3.385 |
| 3 | 0.100 | **0.469** | 40/80 (50%) | 2.937 |

`judge_failed: 0.0` throughout; `judge_called` 0.73–0.90. The paper reports **47.4%**
of rollouts reward-positive, so the judged reward lands in their regime while
rule-only would have gated on ~a quarter of rollouts and starved the attention loss
of exactly the trajectories it exists to shape. `attn_mask_loss` falls across the
three steps, so the KL to the evidence target moves.

Resource facts: **26–29 GB allocated of 80** per GPU (large headroom at 8 GPUs /
TP 4), 301 GB host RAM of 2 TB, steps 39.0 / 27.9 / 27.3 s at rollout batch 16.
`format: 0.95` — the cold start adapts to their `<answer>` template. 3.8% of
responses hit the 1024 cap.

Scaling 16 → 128 puts the real run near **3 min/step, ~6 h per arm**, so 2–3
resubmits each against 4 h allocations. Four model-only checkpoints per arm, ~68 GB.

**`JUDGE_MAX_WORKERS` defaults to 64 because of this run.** At rollout batch 128 a
step judges ~480 completions; 32 workers is ~15 sequential rounds. Note that
`launch_ease_train_job.sh` must write every judge setting into the runner explicitly
— submit_job does not carry the submitting shell's environment into the allocation,
and the failure mode is quiet.

### The first pair of runs reward-hacked, 2026-09-10

`ease_8k` and `dapo_8k` both ran to 124/124 and **both are unusable**. Kept for the
record; not benchmarked.

| | step ~1 | step ~124 |
|---|---|---|
| `format` (emits `<answer>…</answer>`) | 0.97 | **0.16** EASE / **0.05** DAPO |
| `response_length` mean | 190 | **~1000** of 1024 |
| fraction truncated at the cap | 0.01 | **0.87 / 0.92** |
| `rule_accuracy` | 0.22 | **0.000** |
| judged `accuracy` | 0.57 | **0.88 / 0.82** |

The shape, from `checkpoints/generations.log`, scored **1.0**:

```
<answer> Down <answer> The direction mentioned in one of the book titles is "Down."
<answer> Down <answer> The direction mentioned ... [to the 1024-token cap]
```

Note what is missing: a **closing** `</answer>`. Their `perception.py` is immune to
this by construction — its regex needs the closing tag, so an unclosed one falls
through to `answer_text = response.strip()` and is then killed by
`len(answer_text) < 300`. Rambling earns nothing there, so rambling never pays.

The first version of `judged_perception.py` kept that whole-response fallback and
dropped the length guard, so the judge was handed a 1000-token ramble and asked
whether it matched the gold answer. It does — the answer is somewhere inside — so
gpt-4o-mini returned 5/5. With `format_weight` at 0.0 (their default, safe only
alongside their guarded matcher) nothing pushed back. **This was a bug in our
reward, not in EASE.**

Their unmodified reward is not the alternative: `rule_accuracy` reaches 0.000 by
mid-run for everything, so GRPO would have seen all-zero groups, zero advantage and
no gradient. Neither reward works on this corpus as-is.

**The fix is a span gate.** The judge now sees a *closed* `<answer>…</answer>`, or
text after `</think>` (our cold start's own format), and nothing else — no
whole-response fallback — capped at their 300 characters, which still admits
flickr30k's 100–150-character sentence answers. Everything else scores 0 with no API
call. The rule half still runs their code verbatim, so the DAPO arm's reward is
unchanged. A new `no_answer_span` metric is the canary.

`ease/test_judged_perception.py` replays the real `generations.log` from both runs:
all 12 hacked completions that scored 1.0 now yield no span, and 5/5 well-formed
completions from the pre-degeneration smoke still pass.

**What three steps of smoke could not catch.** At step 3 `format` was still 0.95;
the collapse begins around step 30–40. Any future reward change here needs ~40 steps
of canary, watching `format`, `response_length` and `no_answer_span` — not three.

**Still open:** a closed `<answer>` under 300 characters listing several candidate
answers ("Down, Up, North, …") could still shotgun the judge. The judge prompt is
deliberately byte-identical to `trl/rewards/openai_rewards.py` so the two stacks'
accuracy means the same thing, so this is watched rather than prompted away.

### Deviations from their recipe, in full

| | |
|---|---|
| data | saliency-r1-8k, all single-evidence (K=1) |
| base model | our SFT cold start, staged for 4.57 |
| reward | their matcher + our judge (`--no-judge` for theirs) |
| rollout batch | **128**, not 512 |
| val ratio | 0.0124 (~95 rows), not 0.1 (808), to match our TRL runs' held-out 100 |

Everything else is `examples/config.yaml` and `train_ease_dapo_qwen3vl.sh`: lr 1e-6,
2 epochs, n=5, clip 0.2/0.3, KL off, λ_attn 0.001, background α 0.1, σ scale 0.25,
layer ⌊2L/3⌋, τ 0.5, ≤64 response tokens for the aux loss, `padding_free: false`.

**The rollout-batch change is not an optimizer change.** At 512, our 8,080 rows give
31 steps over two epochs against the ≤~158 their own run had. But EasyR1 multiplies
`worker.actor.global_batch_size` by `rollout.n` internally
(`fsdp_workers.py:143`), so with `global_batch_size=64` the update sees 64 prompts ×
5 rollouts either way and the run takes **252 optimizer steps** at 512 or at 128.
What 128 buys is reporting granularity: 126 steps to checkpoint and log against
instead of 31. Gradient noise per update is unchanged.

## Their data (the slower alternative)

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
| `setup_ease_env.sh` | sbatch, `cpu` — builds the `ease` conda env |
| `patch_ease_repo.sh` | the one additive edit to their reward interface; idempotent |
| `stage_ease_checkpoint.sh` | 5.x checkpoint → 4.57-loadable, weights symlinked, four checks |
| **the saliency-r1-8k route** | |
| `export_saliency_r1_8k_for_ease.py` | corpus → raw parquet + content-hashed images |
| `add_ease_prompt_length.py` | their `prompt_length`, in a process pool |
| `prepare_ease_saliency_data.sh` | sbatch, `cpu` — the three data steps end to end |
| `ease/reward_function/judged_perception.py` | their matcher + our judge |
| `launch_ease_train.sh` | one arm, inside an allocation |
| `launch_ease_train_job.sh` | asks SLURM for the allocation |
| `cold_data/ease/saliency_r1_8k/` | `raw/`, `images/`, `parquet/{train,val}.parquet` |
| **the five-corpus route (not on the critical path)** | |
| `download_ease_sources.sh` | sbatch, `cpu_datamover` — pulls and extracts the three HF corpora |
| `cold_data/ease/raw/`, `cold_data/ease/images/` | downloaded corpora as published |

Both `ease_repo` and `cold_data/ease` are symlinked from the central tree via
`.worktree-links`, so deleting the worktree never takes 47 GB of downloads with it.
`stage_ease_checkpoint.sh` resolves with `pwd -P` for the same reason: a logical path
would point the staged weight symlinks through `.worktrees/<branch>/`, and
`worktree.sh done` would break a checkpoint that outlives the branch.

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
3. ~~**The 1:1 mixture caps the training set at ~40.5k**~~ — moot on the
   saliency-r1-8k route. Ours is **7,984 train / 95 val**, all single-evidence.
4. ~~**The ZwZ 74k subset rule is unknown.**~~ — moot; we do not use ZwZ.
5. **Every row is K=1**, so EASE's multi-evidence Gaussian mixture never mixes. The
   paper's own ablations treat single-vs-multi as a real axis, and this route can only
   ever exercise one side of it. If the mixture turns out to be where the method's
   gain lives, that has to come from the five-corpus route.
6. **Porting caution, if we ever move this into TRL:** `trainable_attention.py`
   captures Q/K *pre-RoPE* off `q_norm`/`k_norm` and reads `position_embeddings` from
   self_attn kwargs with an `args[7]` positional fallback. That signature was written
   against transformers 4.5x and would need checking against our patched 5.x file.

## Next steps

1. **A short smoke run first.** Nothing has touched a GPU yet. Two things are
   untested end to end and both fail late: whether verl's FSDP + vLLM path loads the
   staged checkpoint, and whether the judge keeps up inside a Ray reward actor at
   640 completions per step. Run a handful of steps before committing 8 GPUs for
   hours — `--rollout-batch 16 --epochs 1 -- trainer.max_steps=3`.
2. **Both arms**, `--arm ease` and `--arm dapo`, same data, same staged checkpoint,
   same reward. Resubmit the same `--exp` to resume; do not restart.
3. **Score with our harness**, `run_bench_eval.sh`, and report
   (EASE − DAPO) against (overlap − placebo). Their published table is not a
   reference point — benchmark overlap with `eval_mini/benchmarks.py` is POPE alone.
4. **Sanity-check the gate.** They report 47.4% of rollouts reward-positive and 92.3%
   of groups with at least one gate-passer. Our judged reward is graded rather than
   binary, so the τ=0.5 gate admits a 3-of-5 judge score; the logged
   `judge_called` / `rule_accuracy` metrics are there to tell how the two halves of
   the reward split.
5. The five-corpus route stays available if the K=1 limitation turns out to matter
   (open question 5). It needs `GOOGLE_API_KEY`, CLEVR/SuperCLEVR scene graphs, and
   a ZwZ subset rule.
