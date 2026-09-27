# What stands between here and a real GRPO step on the Omni

Written 2026-09-27, branch `probe/omni-quant`. Two blockers, both concrete, neither one a
day's work.

## 1. The installed vLLM cannot serve the Omni

`omni_train_step_bench.py` measures the training side only, and this is why. The
generation server in the `nemotron` environment is vLLM 0.11.0, and its registry has:

```
NemotronForCausalLM
NemotronHForCausalLM
NemotronH_Nano_VL_V2          <- the 12B VL
Llama_Nemotron_Nano_VL        <- the 8B VL
```

`NemotronH_Nano_Omni_Reasoning_V3` is **not** there. vLLM's current documentation does
list it, with LoRA support, so this is a version gap rather than a missing feature:
upgrading vLLM inside the (isolated) `nemotron` environment should close it. That is the
fix to try, and it is safe to try there in a way it would not have been in
`saliency_r1_qwen3_vllm`.

## 2. The trainer has Qwen3-VL's geometry threaded through it

`trl_repo/trl/trainer/grpo_trainer_qwen3.py` is 3,427 lines and mentions
`image_grid_thw` or `spatial_merge_size` 35 times — not in one place, but through the
batching, the micro-batching and the saliency read:

```
 349:    lengths = batch["image_grid_thw"].prod(dim=1).tolist()
1160:        if image_grid_thw is not None and pixel_values is not None:
1251:            start_pixel_idx = image_grid_thw[:start].prod(-1).sum().item()
1590:        thw = prompt_inputs.get("image_grid_thw")
```

`image_grid_thw` is Qwen3-VL's way of saying how a picture was cut up. **The Omni has no
such field.** Its processor emits `pixel_values`, `num_patches`, `num_tokens` and
`imgs_sizes`, and its grid is native-resolution — a different size per picture. So every
place that slices a batch's pixels by grid, and the place that maps attention columns
back onto patches, needs the Omni's geometry instead.

`vlm_family.py` already holds exactly that knowledge for the measuring side
(`NemotronVL.grids_for` reads `imgs_sizes`). The port is to make the trainer ask the
family rather than read `image_grid_thw` — the same seam, applied one level up. Doing it
that way is also what would let a third model in later without a third copy of the file.

## What this does NOT block

Everything on the training side: the forward, the backward through 23 Mamba layers, the
LoRA, the optimizer step, and the memory footprint. Those are what the quantization
decision turns on, and `omni_train_step_bench.py` measures all of them with the real
settings. See `docs/omni-quantization.md` for the answer it gave.

## One trap found on the way, worth keeping

The launcher's LoRA targets are the bare names `q_proj,k_proj,v_proj`, which peft matches
by suffix **anywhere in the model**. That is safe on Qwen3-VL, whose vision tower uses a
fused `qkv`. It is not safe on the Omni, which carries an **audio tower** whose 24 layers
use exactly those three names. The first run put 144 of its 180 LoRA tensors on that
tower, where an image-only batch never runs, and they came back with no gradient at all —
a configuration that trains almost nothing while looking correct in every log line.

Scope the targets to the decoder (`language_model\..*\.(q_proj|k_proj|v_proj)`) and
assert they land on the six attention layers `[5, 12, 19, 26, 33, 42]`. The benchmark
does both and refuses to run otherwise.
