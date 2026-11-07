# Training the overlap reward on GLM-4.1V and InternVL3.5

Written 2026-11-06, branch `feat/second-backbones`. The harness is in place end to end
for both; **nothing has been trained yet**, and §5 is the step that must happen before a
run means anything.

`vlm_family.py` is the seam all of this goes through — the trainer stopped reading
`image_grid_thw` during the Omni port (`docs/omni-training-harness.md`) and asks the
family instead. That is why most of this document is *facts that were checked* rather
than code that was written.

## 1. The two backbones, and what each one is for

| | Qwen3-VL-8B (incumbent) | GLM-4.1V-9B-Thinking | InternVL3.5-8B |
|---|---|---|---|
| checkpoint | `Qwen/Qwen3-VL-8B-Instruct` | `zai-org/GLM-4.1V-9B-Thinking` | `OpenGVLab/InternVL3_5-8B-HF` |
| decoder | Qwen3, 36 x 32 | GLM-4, **40 x 32** | Qwen3, 36 x 32 |
| vision tower | Qwen3-VL ViT + DeepStack | GLM ViT, native-res | InternViT-300M, 448px tiles |
| grid | `image_grid_thw`, 2x2 merge | **same** | **fixed 16x16** |
| image tokens / picture | ~1,280 at 512px | ~250-280 | **256, always** |
| `vlm_family` | `Qwen3VL` | `Glm4V` (subclasses it) | `InternVL` |

The two are deliberately opposite choices. **GLM changes the language model** — GLM-4 is
neither Qwen nor Llama — while keeping Qwen3-VL's exact geometry, so anything that
survives it is not a property of the Qwen decoder. **InternVL changes the eyes** and
keeps a Qwen3 text tower of the same shape as the incumbent's, so anything that survives
it is not a property of the vision encoder. Together they bracket the claim in a way a
single second model cannot.

Use `-HF` for InternVL and not `OpenGVLab/InternVL3_5-8B`. The `-HF` repackaging has
`model_type: internvl`, which is the key `vlm_family.InternVL`, LLaMA-Factory's
`intern_vl` plugin and vLLM's `InternVLForConditionalGeneration` all dispatch on; the
original is `InternVLChatModel` behind `trust_remote_code` and shares none of it.

## 2. What was already there, and what this branch adds

Already there, and the reason the port is small: the trainer's geometry seam; the generic
SDPA identity patch (`patch_transformers_qwen3.sh`), which makes `output_attentions=True`
work on any family routed through `ALL_ATTENTION_FUNCTIONS`; `merge_lora.py`, which is
`AutoModelForImageTextToText` and model-agnostic; and `head_correlation_probe.py`, which
was made family-driven for the Omni.

This branch adds:

* `vlm_family.InternVL` — the trainer-side surface it did not have (one row per picture,
  the fixed grid, the LoRA scope, the tiling guard);
* `vlm_family.Glm4V` — two explicit statements rather than inherited accidents, plus the
  measured evidence that the rest of the inheritance is right;
* `trl/grpo_trainer_qwen3.py` — `family.proc_defaults` at the processor call, and three
  `== 151655` literals that now ask the family;
* `train/cold_start/{glm4v_9b_thinking_sft,internvl3_5_8b_sft}/train.yaml`;
* `coldstart_gate.py` + `test_coldstart_gate_cpu.py` — the acceptance gate for a
  shortened cold start, and 22 CPU checks on its judgement;
* `test_vlm_geometry_cpu.py` — 25 more checks, no GPU and no weights.

## 3. Cold start

```fish
bash launch_coldstart_job.sh --config train/cold_start/glm4v_9b_thinking_sft/train.yaml
bash launch_coldstart_job.sh --config train/cold_start/internvl3_5_8b_sft/train.yaml
```

Every hyperparameter matches `qwen3_vl_8b_instruct_sft/train.yaml`; only the backbone and
its chat template differ. The job name now follows the config's directory, so the two do
not append to the Qwen3-VL run's log file (it resolves byte-identically for the incumbent
config).

Four things about the templates were checked rather than assumed, because every one of
them fails *silently* — the model simply never learns the tags, and the first sign is a
flat `format` reward in GRPO days later.

**`glm4v` is a `ReasoningTemplate`, and that is correct here.** LF warns to add a
`_nothink` suffix when the base has a separate Instruct variant; GLM-4.1V does not, so the
reasoning template is the right one. What it does to targets that already carry literal
`<think>` tags: `enable_thinking` defaults `True`, so the branch that *strips* the thought
out of the target does not fire; `preserve_thinking` defaults `False`, which strips
history turns — `range(1, len(messages) - 2, 2)`, empty on this single-turn corpus; and
the "add an empty cot" branch is skipped because it tests for exactly the tags these
targets have. Measured on the first row of `saliency_r1_llava_cot_full`: prompt 457
tokens, response 170, **one `<think>` and one `</think>`**. `intern_vl` is a plain
`Template` and the question does not arise.

**GLM's `resize_vocab` warning is spurious.** LF logs "Add `<|user|>,<|observation|>,
</answer>` to stop words" then "New tokens have been added, make sure `resize_vocab` is
True". On this tokenizer nothing was added: `<think>`, `</think>`, `<answer>` and
`</answer>` are already single tokens (151345–151348), `len(tokenizer)` is unchanged at
151352, and the embedding has 151552 rows. InternVL's tokenizer behaves exactly like
Qwen3-VL-8B-Instruct's (`<think>`/`</think>` single, `<answer>` three pieces), which is
the incumbent's situation and known to work.

**The cold start does not LoRA either vision tower.** `lora_target: all` returns bare
names, which peft would suffix-match anywhere — but LF resolves them to full module paths
through `patch_target_modules` first. Counted on all three: 252 / 240 / 252 resolved
targets, **0 in the vision tower**.

**InternVL's cold start sees one tile, like its GRPO.** LF's `InternVLPlugin` calls the
*image* processor directly with `getattr(processor, "crop_to_patches", False)`, and the
image processor honours its own `crop_to_patches: False`. Measured: `pixel_values`
`(1, 3, 448, 448)`, 256 image tokens. The trap is one level up — see §4.

### The budget is 2,000 steps, not 2 epochs

Both configs carry `max_steps: 2000`. **This is the one place the two new arms
deliberately depart from the Qwen3-VL recipe**, and the reason is that the Qwen3-VL arm
exists to be compared against Saliency-R1's published numbers while these two do not —
they only have to learn the format and a reasoning style the overlap reward can score.

What the full recipe bought, off the Qwen3-VL run's own `trainer_log.jsonl` (share of the
total loss drop, 1.7276 → 0.4728):

| step | epoch | share | | step | epoch | share |
|---|---|---|---|---|---|---|
| 100 | 0.02 | **79.9%** | | 3,000 | 0.70 | 96.7% |
| 300 | 0.07 | 89.7% | | 4,258 | 1.00 | 97.5% |
| 1,000 | 0.23 | 92.5% | | 8,516 | 2.00 | 100% |
| 2,000 | 0.47 | 94.3% | | | | |

The entire second epoch is 3.8 hours for the last 2.5%. At ~1,124 steps/h, 2,000 steps is
**~1.8 h — one 4-hour allocation, so the autoresume cycle never starts**, which on a
contended cluster is worth more than the 4× compute saving. `save_steps` drops to 250 to
match.

`max_steps` rather than a fraction of an epoch, and the difference matters: the cosine
schedule and warmup are computed against it, so the LR anneals to zero over exactly these
2,000 steps. Stopping an 8,516-step run at 2,000 would leave the adapter mid-schedule at a
high LR. `num_train_epochs: 2` stays in the yaml and is ignored, so deleting one line
restores the full recipe.

The cut is a bet, and §3.5 is how it gets settled.

Then merge, exactly as for Qwen3-VL (the script reads the base out of
`adapter_config.json`, so it is already model-agnostic):

```fish
bash merge_lora_grpo_qwen3.sh \
    $REPO/checkpoint/coldstart_glm4v_9b_thinking_sft_2ksteps_lr5e5 \
    $REPO/checkpoint/coldstart_glm4v_9b_thinking_sft_2ksteps_lr5e5_merged
```

## 3.5. The gate — settle the 2,000-step bet before spending a GPU-day on it

```fish
python coldstart_gate.py --model <merged> \
    --reference checkpoint/coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged
```

~15 minutes on one GPU, 100 held-out prompts from `cold_data/grpo_sets/val_natural`,
**sampled at temperature 1.0** because that is what GRPO generates at and greedy decoding
overstates format compliance. Exits non-zero on a fail. `--reference` scores the Qwen3-VL
cold start alongside, which is the only one of the three with a known-good GRPO run behind
it, so its numbers are what "good enough" means in practice rather than in principle.

Three criteria, each a mechanism the reward depends on, and each one silent when it fails:

| criterion | threshold | what it being wrong looks like later |
|---|---|---|
| `format_ok` | ≥ 95% | every reward masked; `format` flat near 0 |
| median observe steps | ≥ 2 | `think_overlap_reward = nan`, nothing in the log says why |
| `<answer>` tags in the answer | ≤ 2% | `accuracy_reward` exactly 0.0 forever, no error |

**That third row is the one worth knowing about.** The trainer's format regex is
`^<think>…</think>…$`, with no `<answer>` in it — but it does not *reject* those tags
either, it swallows them into the answer capture group. `accuracy_reward` then takes
everything after `</think>` and, for the non-LaTeX gold that is most of this corpus,
string-compares it against the solution. `"<answer> flower </answer>"` is never
`"flower"`. And `<think>…</think><answer>…</answer>` is **GLM-4.1V-Thinking's native
format** — it was trained upstream to write it — so a *shortened* cold start is precisely
the condition under which it creeps back. The gate therefore counts the tags itself rather
than trusting the format regex to catch them.

`test_coldstart_gate_cpu.py` pins all of this with 22 checks and no GPU, including that
the tagged completion really does pass the format regex, and a drift guard that makes the
gate refuse to run if the trainer's pattern or system prompt ever moves away from the one
it scores.

On a fail: re-run the cold start at `max_steps=4000` (+1.8 h) and gate again. Do not go on
to §5 — a weak cold start does not announce itself in GRPO, it looks like a reward that
does not work.

## 4. The two traps, and where each is now stopped

**InternVL's processor tiles unless told not to, and the attribute is a lie.**
`image_processor.crop_to_patches` is `False` on this checkpoint, and `InternVLProcessor.
__call__` carries its own default and overrides it. Measured on one 512x384 picture:

```
bare call                       13 tiles   3,328 image tokens
crop_to_patches=False            1 tile      256 image tokens
```

Nothing downstream would say so: `mm_slice` would hand the forward one tile of some other
sample's picture, `token_grid` would keep answering 16x16, and every per-step map would
come out 3,328 long against a 256-cell grid, be dropped by the length check, and surface
as `think_overlap_reward = nan`. Two things now stop it — the trainer applies
`family.proc_defaults` at the processor call (a no-op for every other family), and
`InternVL.after_processor` *refuses* a batch whose image rows outnumber its prompts.

**InternVL's vision tower uses bare `q_proj`/`k_proj`/`v_proj`.** The same collision that
put 144 of 180 Omni LoRA tensors on an audio tower, on a tower that actually runs. Counted
on the checkpoints: of the modules ending in `q_proj`/`v_proj`, Qwen3-VL has 72 and
**all 72** are in the language model, GLM has 80 and **all 80** are (its
`Glm4vVisionAttention` uses a fused `qkv`), and InternVL has 120 of which **48 are in the
InternViT**. Left bare, 40% of the adapter trains the vision encoder — and because this
tower runs on every batch it would *train*, not sit at zero gradient, quietly making the
arm a different experiment from the one it is meant to compare against.
`InternVL.lora_target_modules` scopes it to `.*language_model\..*\.(q_proj|v_proj)`. The
leading `.*` is load-bearing: peft `fullmatch`es the whole path and the path here is
`model.language_model....`, so the Omni's own anchor-less pattern would match *nothing*.

What is **not** a trap, having been checked: vLLM serves InternVL through its `interns1`
path, which reads `image_processor.crop_to_patches` and `min_patches`/`max_patches` off
the config rather than through the full processor. `get_number_of_image_patches` returns
**1** for every picture tried, so the colocated launcher's generation server agrees with
training without any `mm_processor_kwargs` plumbing.

## 5. Head and layer selection — do this before any run is a baseline

**`--overlap-layer 22 --overlap-heads 28,31` are Qwen3-VL-8B numbers.** They were selected
on that model by `head_correlation_probe.py`; on GLM-4.1V (40 x 32) and InternVL3.5
(36 x 32) the same three indices name arbitrary heads, and the layer and the head pair
**are the reward's definition**. `docs/omni-head-selection.md` is the precedent, including
the sentence worth repeating: a run launched on the incumbent's numbers is a harness smoke
test and not a baseline to continue from.

The probe is already family-driven and takes the model and the environment as arguments:

```fish
bash launch_head_correlation.sh --gpus 8 --out-dir DIR --cases-dir PROBE_DIR \
    --base-model $REPO/checkpoint/coldstart_glm4v_9b_thinking_sft_2ksteps_lr5e5_merged
python head_correlation_probe.py --stage report --out-dir DIR --max-union 0.5
```

Both backbones are dense, so unlike the Omni's 192 cells the search is the full
1,280 (GLM) / 1,152 (InternVL). Read the parity split, not the ranking.

## 6. GRPO

Nothing in the launcher is Qwen3-VL-specific once the layer and heads are known — the
LoRA scope, the processor keywords and the grid all come from `vlm_family` inside the
trainer:

```fish
bash launch_grpo_qwen3_overlap_job.sh \
    --model $REPO/checkpoint/coldstart_glm4v_9b_thinking_sft_2ksteps_lr5e5_merged \
    --overlap-layer <L> --overlap-heads <a,b> --overlap-metric mean_in_v2
```

**One thing to check before the first launch, and it is not specific to these backbones.**
The shared `trl_repo/` currently has **no `trl/trainer/vlm_family.py` at all**, and its
`grpo_trainer_qwen3.py` is the pre-seam 3,427-line version. The launcher copies the
seamed trainer in at launch time, and that file imports `.vlm_family` at module scope — so
without the module beside it the run dies on import, on Qwen3-VL as much as on anything
new. This branch adds `vlm_family.py` and `nemotron_loader.py` to the launcher's own copy
list, which closes it for the non-colocated launcher. The colocated one syncs nothing and
still relies on `patch_trl_qwen3.sh` having been run.

`patch_trl_qwen3.sh` re-patches the clone **every session and every running job** shares,
so run it deliberately rather than as part of a launch, and not while a job is mid-step.

The first scored step prints

```
[overlap] first scored case: 8/8 completions format-valid, think tokens A-B,
          N observe steps, map P patches against grid GxH
```

On a new backbone that line is the one to read. `P != G*H` is flagged in the line itself
and means every map is being dropped. For InternVL `P` must be 256 and the grid 16x16;
for GLM it varies per picture, like Qwen3-VL's.

`--reward_variant ours` is the only variant defined on InternVL. `grad`, `glimpse` and the
original `saliency_r1` readout all index `image_grid_thw` and fold pixel gradients back
with Qwen3-VL's patch packing, so `InternVL.supports_saliency_r1()` is `False` and they
refuse with a one-line message instead of failing thirty frames down. On GLM all of them
are *defined* — dense decoder, and it emits `image_grid_thw` — but none has ever been run,
so treat a number off them as unvalidated.

## 7. Benchmarks

`run_bench_eval.sh` infers the lmms-eval wrapper from the checkpoint path and that
inference only knows Qwen; `BENCH_MODEL_TYPE` overrides it.

```fish
BENCH_MODEL_TYPE=glm4v            bash run_bench_eval.sh ...
BENCH_MODEL_TYPE=internvl_hf_chat bash run_bench_eval.sh ...
```

`internvl_hf_chat` is the wrapper that loads `InternVLForConditionalGeneration`;
`internvl3_5` is for the `trust_remote_code` repo and is the wrong one here.

**And it must be told not to tile.** `internvl_hf_chat` calls the full processor and
defaults `max_patches=12`, which gives 13 tiles and 3,328 image tokens — 13x what the
model was cold-started and GRPO'd on. Pass `max_patches=1,min_patches=1` in
`--model_args` (measured: that gives the 1 tile / 256 tokens training used). Without it
the InternVL arm is evaluated off-distribution and will read as a bad backbone rather
than a bad harness.

## 8. What has been verified, and what has not

Verified, on the real checkpoints: both load and their configs resolve; both processors'
batch shapes and grids (and that GLM's packed `pixel_values` rows are exactly its grid
products); the LoRA suffix counts; the LF template encodings end to end on a real
cold-start row; the vocab arithmetic; LF's resolved cold-start LoRA targets; vLLM's
registry entries and InternVL's patch count; and 63 CPU checks in
`test_vlm_geometry_cpu.py`.

**Not verified, because it needs GPUs:** a cold-start step, a GRPO step, the
`[overlap]` shape line on either backbone, and anything about reward quality. The next
action is §3 on both, then §5, then §6 — in that order, because §5 needs the merged
cold-start checkpoint as its base model.
