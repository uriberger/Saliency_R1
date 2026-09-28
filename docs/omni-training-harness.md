# The harness a Nemotron-Omni GRPO run needs, and why it is not `trl_repo`

Written 2026-09-28, branch `feat/omni-grpo-plan-a`. `docs/omni-gpu-layout.md` says how the
eight cards divide, `docs/omni-quantization.md` says not to quantize, and
`docs/omni-training-blockers.md` named the two things standing in the way. This is what
closing them actually took.

The run is `launch_grpo_omni_overlap_job.sh`, and its header is the operating manual. This
file is the record of the decisions behind it.

## The shape of it

```
GPU 0      Grounding-DINO             nemotron        (before it starts: the preflight)
GPU 1      vLLM generation server     nemotron_vllm
GPU 2-7    6 training processes       nemotron        multi_gpu.yaml, one whole copy each
```

Every training hyper-parameter is the Qwen3-VL runs': LoRA r=16 alpha=32 on q/k/v_proj,
lr 1e-5, per_device 1, grad_accum 8, num_generations 8, max_completion_length 1024,
beta 0, 48 rollouts per update. Five things differ and each is forced.

## 1. Two environments, because vLLM and the trainer cannot share one

vLLM 0.11 — what `nemotron` had — has `NemotronHForCausalLM`, `NemotronH_Nano_VL_V2` and
`Llama_Nemotron_Nano_VL` but not `NemotronH_Nano_Omni_Reasoning_V3`. The registry entry is
not the problem: **0.11's `nemotron_h` has no MoE layer type at all**, and the Omni's
`hybrid_override_pattern` is `MEMEM*EMEM...` — half of it is `E`. Backporting would mean
implementing the mixture of experts, not adding a line.

vLLM 0.20.0 is the earliest release that registers the arch (it maps it onto
`nano_nemotron_vl.NemotronH_Nano_VL_V2`, i.e. the image+text path; the audio tower is not
served, which is exactly what this run needs). It pins **torch 2.11**, a CUDA 13 build.
That runs here — the nodes are on driver 580.105.08, which reports CUDA 13.0 — but it also
drags transformers from 5.13.0.dev0 down to 4.57.6.

The training side is the part that is measured (34.0 s a step), shimmed against
transformers 5.13 in seven places, and working. So it does not move:

| env | torch | transformers | vLLM | role |
|---|---|---|---|---|
| `nemotron` | 2.8.0+cu128 | 5.13.0.dev0 | 0.11.0 | the trainer, DINO, the preflight |
| `nemotron_vllm` | 2.11.0+cu130 | 4.57.6 | 0.20.2 | the generation server |

`nemotron_vllm` is a clone of `nemotron` with vLLM upgraded into it, so nothing about
`nemotron` changed except which TRL it imports (below).

**The one coupling with no precedent** is the weight sync: TRL's client opens a NCCL
communicator from the trainer to the server, which is NCCL 2.27 (torch 2.8, cu12) talking
to NCCL 2.28 (torch 2.11, cu13). Both ends of the HTTP protocol are TRL code from the same
clone, so the API cannot drift; the collective is the part to suspect if a step ever hangs
in `init_communicator`.

## 2. `trl_repo_nemotron`, so nothing is rewritten under a running job

`trl_repo/` is shared by every session and every running job, and `patch_trl_qwen3.sh`
rewrites it in place. Trying an Omni change there would hand it to six in-flight Qwen3-VL
ranks mid-step.

`trl_repo_nemotron/` is the same fork at the same branch, patched from the same tracked
sources by `patch_trl_nemotron.sh`, and nothing but the Omni launcher points at it. It is
in `.worktree-links`, so it is shared between worktrees the same way — and carries the
same caution.

`import trl` is resolved by a PEP 660 **meta-path finder**, not a `sys.path` entry, so
`PYTHONPATH` cannot redirect it. Both environments have `pip install -e trl_repo_nemotron`
instead. Nothing else in either environment imports `trl` as a package (the probes load
`trl/overlap_steps.py` by path).

Two files land in `trl_repo_nemotron` that the Qwen3-VL patcher does not install:

* `trl/scripts/vllm_serve.py` — a local copy that runs under **both** 0.11 and 0.20. The
  three differences are spellings, not behaviour: `GuidedDecodingParams` →
  `StructuredOutputsParams` (and the `SamplingParams` kwarg renamed with it),
  `vllm.utils.get_open_port` → `vllm.utils.network_utils.get_open_port`, and `top_k = -1`
  ("off") → `top_k = 0`, which the newer release rejects. It also grows two arguments
  upstream's does not expose, `--max_num_seqs` and `--kernel_config`, both of which the
  Omni needs (§5).
* `vlm_family.py` and `nemotron_loader.py` — the seam, §3 and §4.

## 3. The trainer asks the family for the geometry

`trl/grpo_trainer_qwen3.py` read `image_grid_thw` in thirty places, across the batching,
the micro-batching and the saliency read. The Omni has no such field: its processor emits
`pixel_values`, `num_patches`, `num_tokens` and `imgs_sizes`, and its grid is
native-resolution — a different size per picture.

The port is the seam `vlm_family.py` already was for the measuring side, applied one level
up. `Family` grew the trainer's half of it:

```
mm_inputs            which processor outputs are carried into every forward
packed_inputs        which of those are NOT row-aligned with the batch
geometry_inputs      which are carried for geometry and must never reach forward()
mm_lengths/mm_slice  how a micro-batch is cut out of them
token_grid           a sample's (gh, gw)
batch_image_arg      nested per sample, or flat
collapse_image_run   what a decoded prompt looks like on the wire (§6)
forward_defaults     arguments this family always wants (the Omni: use_cache=False)
decoder              where the layer stack is
lora_target_modules  §4
supports_saliency_r1 whether the dense-decoder readouts are even defined here
```

The difference between the two families is smaller than it looks and runs the other way
from expectation: **Qwen3-VL is the complicated one.** It stacks a whole batch's patches
into one flat `pixel_values`, so a micro-batch's slice of it is only findable through the
grid; the Omni emits one row per picture in every field, so a micro-batch is an ordinary
slice and nothing has to be cut.

`test_vlm_geometry_cpu.py` is what holds the Qwen3-VL side still. Each replaced expression
is restated there and checked against the family's answer, because `patch_trl_qwen3.sh`
now installs `vlm_family.py` into the shared clone too, and a seam that quietly changed a
micro-batch's pixel slice would move a live run's gradient without moving anything
visible.

**Three readouts are refused rather than ported.** `grad` folds a pixel gradient back onto
patches with Qwen3-VL's `patch_size` × `temporal_patch_size` packing; `glimpse` propagates
gradient-weighted attention across every layer; the original Saliency-R1 readout
multiplies each layer's attention by that layer's value states and pushes it through
`o_proj`. On a hybrid the last two have no object at 46 of 52 layers. Each would produce a
number rather than an error, which is the failure worth refusing by name.

## 4. The loader, in one place, because the trainer builds the model too

The trainer resolves its architecture as `getattr(transformers, config.architectures[0])`,
which raises `AttributeError` on a `trust_remote_code` checkpoint. So it needs the same
repairs `sink_location_probe.load_model` already had, and a second copy of them would be a
second thing to fix the next time transformers drifts. They moved verbatim into
`nemotron_loader.py`; `sink_location_probe` re-exports it and behaves identically.

The seven, in the order a load meets them: the vendored `mamba_ssm` (the decoder raises at
*import* without `rmsnorm_fn`), the missing `all_tied_weights_keys`, RADIO's
`summary_idxs` buffer, `create_causal_mask`'s renamed and dropped arguments, a one-rank
process group, `past_key_values` as an alias of `cache_params`, and the `cache_position`
5.13 stopped passing.

### The LoRA target scope, which is the trap that cost a run

peft matches bare target names by **suffix, anywhere in the model**. `q_proj,k_proj,v_proj`
is safe on Qwen3-VL (fused `qkv` in the vision tower) and safe on RADIO. The Omni carries a
**24-layer audio encoder** using exactly those three names, which an image-only batch never
runs: 144 of 180 adapter tensors landed there and came back with no gradient at all, while
every log line looked right.

`NemotronVL.lora_target_modules` returns `language_model\..*\.(q_proj|k_proj|v_proj)` — a
single **string**, which peft reads as a regex over the whole path rather than a list of
suffixes — and the trainer then asserts the adapters landed on the six attention layers
`[5, 12, 19, 26, 33, 42]` and refuses to start otherwise.

### Gradient checkpointing, and the order it has to happen in

`NemotronHPreTrainedModel` never sets `supports_gradient_checkpointing`, a plain class flag
defaulting to False, so transformers' own switch refuses on a model whose blocks **are**
`GradientCheckpointingLayer`s. The flag is flipped after the blocks are confirmed to exist,
on the language model rather than the wrapper, **before** the peft wrap.

Then `get_peft_model` sees checkpointing already on and re-installs the
`enable_input_require_grads` hook — so stripping it before wrapping accomplishes nothing.
The hook forces the embedding output to require grad, and this wrapper scatters the picture
into those embeddings in place, which raises. `after_peft_wrap` strips every one, reading
each module's own `__dict__` because peft delegates attribute lookup and `del` would
otherwise raise on a name the wrapper never owned.

## 5. What the generation server needs on this cluster

Three failures, none of them in any documentation, each found by `omni_vllm_probe.py`:

* **`VLLM_ENABLE_V1_MULTIPROCESSING=0`.** The EngineCore *child* hangs: it reaches the
  worker's memory snapshot and then sits in a futex with 43 sleeping threads while the
  parent prints `Waiting for 1 local core engine proc(s) to start` forever. In-process it
  loads normally. ptrace is off on these nodes, so py-spy and gdb cannot see into that
  child at all — which is why the probe grew `--watchdog`, a `faulthandler` timer that
  dumps every thread's stack from *inside* the process.
* **the triton MoE backend.** `moe_backend: auto` picks FlashInfer's CUTLASS path, which
  JIT-compiles on first use and dies in `get_cuda_path()`: there is no `/usr/local/cuda` on
  these nodes and the only system toolkit is CUDA 12.4 against a torch built on 13. Triton
  ships its own compiler. It is the portable MoE backend, not the fastest one; installing a
  CUDA 13 `nvcc` into `nemotron_vllm` is the lever if generation ever needs to be faster.
* **`max_num_seqs` ≤ the Mamba state-block count.** A hybrid needs one state block per
  concurrently decoding sequence, and this card has 914. vLLM's default is 1024, so CUDA
  graph capture refuses before anything runs. A GRPO step asks for 48; the launcher uses 64.

Plus `VLLM_USE_DEEP_GEMM=0`: kernel warmup calls DeepGEMM's FP8 path on a bfloat16 model
and raises "DeepGEMM backend is not available or outdated".

### And the number Plan A turned on

Measured on one H100, `gpu_memory_utilization 0.90`, `max_model_len 4096`, eager:

```
engine up                 87 s
GPU KV cache          1,188,848 tokens
8 rollouts of one real prompt   7.0-9.8 s
```

A step needs about 48 × 1,360 ≈ 65,000 tokens of cache. There is **18x** the room. The open
question in `docs/omni-gpu-layout.md` — "it may be enough at max_model_len 4096 with 8
rollouts; it has not been measured" — is closed, and **Plan A' is not needed**: the
generation server takes one card and the trainer keeps six.

## 6. The prompt that crosses to the server

The trainer decodes `prompt_ids` back to text to send to the generation server, and by then
the HF processor has expanded the template's single `<image>` into `<img>` + 270 image
tokens + `</img>`. vLLM's own processor substitutes **its** replacement onto the target
`<image>` — so a wrapper left in the text comes back out as `<img><img>...</img></img>`:
two indicator tokens the training forward never sees, on every prompt of every step.

`collapse_image_run` folds the whole run, delimiters included, back to the chat template's
own spelling, which is what both sides expand from. Qwen3-VL's override is the `re.sub` it
replaces, verbatim.

## 7. Weight sync: 18 tensors, not 7,349

`_move_model_to_vllm` pushes every named parameter to the server, one HTTP round trip and
one NCCL broadcast each. On Qwen3-VL-8B that is ~700 tensors. **The Omni has 7,349**, of
which 5,934 are expert weights — at a few milliseconds apiece the sync alone would be tens
of seconds on a step whose whole budget is ~47 s.

Under LoRA the base is frozen and the server loaded that same base from the same checkpoint
at startup, so after `merge_adapter()` the only weights that differ are the ones an adapter
sits on: 18 tensors here. `SR1_VLLM_SYNC_LORA_ONLY=1` is that filter. It refuses itself —
falls back to pushing everything, with a warning — if any parameter outside an adapter
still requires grad, because then the base is not frozen and the server's copy would go
stale. Off unless a launcher asks, so existing runs are unchanged.

## 8. The preflight

The riskiest unknown was never the step time: the learning signal has to travel back
through 23 Mamba layers on a torch fallback to reach the 6 attention layers the LoRA sits
on. `omni_train_step_bench.py --grads-only` is that check — load, scoped LoRA, recompute on,
one optimizer step, one backward, and a refusal if any adapter tensor comes back missing,
non-finite or exactly zero. The launcher runs it on GPU 0 before any sidecar takes a card,
and will not start the job without it. ~4 minutes.

**Known and unresolved:** the Omni does not reproduce its own gradients run to run, at
bfloat16 as well as at 8 bits, so it is not a quantization artefact. It is measured as a
worst-per-tensor relative difference over 36 tensors, which whichever tensor has the
smallest norm dominates — the minimum norms are ~2e-04 against a median of ~3e-02 — and a
global metric over the concatenated gradient was never computed and would very likely be
far smaller. Probably benign for GRPO. Do not chase it unless something else points there.

## The one number that could not be carried over

The Qwen3-VL runs read layer 22 of 36. **The Omni has an attention matrix at only 6 of its
52 layers**, at `[5, 12, 19, 26, 33, 42]`. 22 is a Mamba layer here; pointing the reward at
it attaches the capture hook to nothing, so the trainer refuses rather than training on a
reward that is silently zero everywhere.

The default is 33, because it is the nearest attention layer to the same relative depth
(63% against Qwen3-VL's 61%), and that is the whole of the argument for it — there is no
head-selection probe behind it. The head pair carries over even less: 28 and 31 were chosen
on Qwen3-VL-8B by a probe, and on any other model the same two indices name two arbitrary
heads. They are kept so the command line differs in as little as possible. **Any attention
number read off this run has to say so.**
