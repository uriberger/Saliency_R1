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

## The one number that could not be carried over — now selected

The Qwen3-VL runs read layer 22 of 36. **The Omni has an attention matrix at only 6 of its
52 layers**, at `[5, 12, 19, 26, 33, 42]`. 22 is a Mamba layer here; pointing the reward at
it attaches the capture hook to nothing, so the trainer refuses rather than training on a
reward that is silently zero everywhere.

Through 2026-09-28 the defaults were **33** — the nearest attention layer to the same
relative depth, 63% against Qwen3-VL's 61%, and that was the whole of the argument — and
heads **28,31**, which were chosen on Qwen3-VL-8B and on any other model name two arbitrary
heads. The 30-step run in §11 trained those, so **any attention number read off it has to
say so**.

**They were replaced on 2026-09-29 by `--overlap-layer 19 --overlap-heads 4,9`**, selected
on this model by `head_correlation_probe` over all 192 cells — 6 attention layers × 32
heads, so the layer is selected too rather than argued from depth.
`docs/omni-head-selection.md` is the scan. In short: L19 h4 and h9 are the only two cells
positive on all four halves of the parity split under both correctness labels, their
held-out r exceeds their select-half r, and the old pair ranks 148–176 of 192.

Two things that doc says and this one must not lose:

* **Layers 26 and 42 are poison.** All 32 heads of 26 and most of 42 correlate *negatively*
  with correctness under both labels, and it is neither the union-size confound nor image
  mass.
* **The selection corpus has a 52%-disputed correctness label**, because the Omni is a base
  checkpoint whose prose answers `accuracy_reward`'s exact-string fallback cannot read.
  Cold-starting the Omni is the real fix; `19 / 4,9` is a defensible improvement on an
  inherited pair, not a settled one.

## 9. What actually stopped it, in the order it stopped

Every one of these was a hard failure on a run that looked correct up to it. They are
listed because each is invisible from the code and expensive from the logs.

| where | what | fix |
|---|---|---|
| engine start | vLLM's EngineCore child hangs in a futex, 43 sleeping threads, forever | `VLLM_ENABLE_V1_MULTIPROCESSING=0` |
| engine start | MoE backend `auto` JIT-compiles FlashInfer CUTLASS, no `nvcc` on the node | `--kernel_config` triton |
| engine start | `max_num_seqs` 1024 exceeds the 914 Mamba state blocks the card has | `--max_num_seqs 64` |
| engine start | kernel warmup calls DeepGEMM's FP8 path on a bf16 model | `VLLM_USE_DEEP_GEMM=0` |
| trainer init | `accelerate.is_peft_model` imports deepspeed, which needs a toolkit | `setup_cuda_home.sh` |
| trainer init | `Trainer.train()` re-enables checkpointing on a wrapper that refuses it | tell it not to |
| weight sync | NCCL 2.27.3 (torch 2.8) cannot pair with 2.28.9 (torch 2.11) | `VLLM_NCCL_SO_PATH` |
| step 1 | **format 0.000 on every rollout** -- the prompt already opens `<think>` | §6 below |
| every step | **each rank built its own Grounding-DINO on its own full card** | `--dino_api_base` |

The last two are the ones worth remembering, because both produce a run that looks healthy.

**The format one.** The Omni's chat template ends the generation prompt at
`<|im_start|>assistant\n<think>\n`. The assistant turn starts INSIDE the reasoning block,
so the completion carries only `</think>`, and `judge_format` -- which wants exactly one of
each tag -- scored every rollout of a perfectly well-behaved model as malformed. The
overlap reward then came back NaN on all of them, because the per-step maps are only built
where the format is valid. Nothing in the logs says "template": it says `format 0.000`.
The trainer now reads the fact off the actual prompt (`<think>\s*$`, anchored so the system
prompt's own tags stay out of it) rather than declaring it per family.

**The detector one.** The launcher started the Grounding-DINO server on GPU 0 and never
passed `--dino_api_base`, so every training rank built its own detector on the card already
holding 62 GB of Omni. It cost **~8 GB per rank** -- peak 74.1 GB against 66.4 with the flag
-- and the visible symptom was 123 `[dino] CUDA OOM; retrying batch at ... size 7` lines in
thirty minutes, one step completed, while the detector's own card sat idle at 1.4 GB with
nothing but a health check in its log.

## 10. The memory, measured

Per training rank, from `SR1_MEM_REPORT`, before a micro-step's forward:

```
                              allocated   reserved   peak      of 79.2 GB
with a per-rank DINO             62.4       62.5     74.1
through the DINO server          61.6       61.6     66.4
```

The forward itself adds **0.3 GB** of saved activations -- all 52 blocks recompute, so
gradient checkpointing is doing exactly what `docs/omni-quantization.md` measured. What
fills the card is the WORKING SET: a 128-expert mixture materialises ~11 GB of
intermediates for a ~1,400-position sequence and the backward asks for one further ~5.5 GB
block, and both are linear in tokens.

Six other levers were applied and each is still in place because each helped, in rough
order of size: no autocast (the model is already bf16, and autocast promotes a log-softmax
over 131,072 classes to fp32), the allocator's cache released before each micro-step (9.4 GB
was held and owned by nothing), `logits_to_keep` forwarded so the lm_head stops running over
the prompt, the completion padding the loss already masks no longer forwarded, the T5
classifier moved to CPU, and NCCL bound to each rank's own device (five 520 MB contexts
belonging to ranks 1-5 were sitting on rank 0's card).

Two were tried and REMOVED because they made it fail earlier:
`garbage_collection_threshold` (does not compose with expandable segments) and capped NCCL
channels.

**`--max-completion-length` is the one training hyper-parameter that did not carry over**
— and until 2026-09-30 the launcher dropped it on the way to the node, so the paragraph
below describes an intention rather than any run that happened. See §12.1.
At 1024 the run reaches step 12-15 and then dies on whichever micro-step is carrying a
completion that reached the cap -- and it gets there later, not never, because this reward
lengthens chains, so the peak grows with them. 768 is the value the runs use; the policy's
longest TERMINATED completion measured 594 tokens, so nothing that finishes is truncated,
and what changes is where a runaway chain is cut. It is in the run name (`_c768`) so a run
that differs in it can never be mistaken for one that does not, and any comparison against a
Qwen3-VL run has to state it.

## 11. Where the run got to

> **Read §12 before using any number in this section.** The run below trained at 1024, not
> at the 768 its directory name claims; its resumes died in NCCL and not in the allocator;
> and the "97-124 s on a bare `srun`" is a run with the overlap reward switched off.


50 steps were asked for; **30 were done**, `checkpoint-30` is on disk, and every logged step
carried a live reward:

```
                    first 5      last 5      (30 logged steps, overlap never NaN)
overlap reward       0.0820      0.0770
within-group sd      0.0298      0.0270
total reward          1.058       1.180
format reward         0.646       0.679
mean length            375         340
```

Step time **215.7 s** on the submitted (container) path and **97-124 s** on a bare `srun`,
against the 34.0 s training-side benchmark. The difference is the reward, not the model: the
saliency capture is eight full teacher-forced forwards of a 33B model per generation batch,
which is nothing like the ~13 s of generation and reward a Qwen3-VL step pays. Per-card
memory 65-78 GB training, 74.2 GB generation, 1.8 GB detector.

The 2-hour wall arrived at step 30 and the auto-resume worked -- a second node picked the
checkpoint up and reached a backward -- but **every resume past step 30 OOMs on its first
backward**, at 768 as at 1024. That is the §10 dynamic and not a new fault: the reward
lengthens chains, the working set is linear in tokens, so a fixed cap is reached eventually
rather than never. Three ways forward, none free:

* `--length-guard`, which is this repo's own answer to "this reward lengthens chains" and
  which the Omni launcher does not wire in. It is a reward term, so it is a second
  hyper-parameter change -- but it is the one that makes the cap hold instead of drifting.
* a lower cap (512), which truncates completions that finish.
* a card bigger than 80 GB, which changes nothing else.

## 12. Three things §10 and §11 got wrong, and how each was found

Everything below came out of evidence that already existed on 2026-09-28 — the job logs in
`outputs/omni_grpo_plan_a/job_logs/` and the `.wandb` transaction logs under
`trl_repo_nemotron/wandb/`. No GPU was used to establish any of it. `omni_step_budget.py`
is the reader.

### 12.1 No submitted run has ever trained at 768

`launch_grpo_omni_overlap_job.sh` submits a job whose command re-executes the same script
with `--direct`, rebuilding the argument list from an `ARGS` array. **That array did not
include `--max-completion-length`.** A flag passed on the submit line and not repeated in
`ARGS` is silently replaced by the default at the top of the file, and nothing warns: the
banner prints the default, the run trains, and the only record of what happened is the
wandb config.

The 30-step run in §11 was launched with `--max-completion-length 768`. It trained at
**1024**:

| evidence | value |
|---|---|
| `wandb-metadata.json` argv, `--max_completion_length` | `1024` |
| `completions/max_length`, over its 30 logged steps | `1024` on 29 of 30 |
| wandb run id | `..._mean_in` — the `_c768` suffix is computed on the node, and was lost |

The output *directory* says `_c768` because the parent computed it before submitting and
passed it explicitly. So the one hyper-parameter §10 calls "the deviation, and the ONLY
one" had never reached a node, and **"768 OOMs on resume as well as 1024" is a statement
about 1024 both times.**

No submitted run has ever used 768. The only run on record that did (`run-20260928_055534`)
was a `--direct` one, started before `--dino_api_base` existed, and it reached step 7.

Six other parsed flags were missing the same way — `--num-generations`,
`--per-device-batch`, `--learning-rate`, `--token-reduction`, `--box-threshold`,
`--max-box-area` and the `--` passthrough. All of them happened to be sitting at their
defaults, so only the completion cap bit. All seven are forwarded now, the default is 768,
and the cap is on the node's own banner so a job log answers the question by itself.

### 12.2 The resumes died in NCCL, not in the allocator

§11 reads "every resume past step 30 OOMs on its first backward" and attributes it to the
§10 dynamic — a working set linear in tokens reaching a fixed cap. The traceback says
otherwise, identically in all three resume logs:

```
torch.distributed.DistBackendError: NCCL error in: .../ProcessGroupNCCL.cpp:3699,
unhandled cuda error, NCCL version 2.27.3
ncclUnhandledCudaError: Call to CUDA function failed.
Last error:
Cuda failure 2 'out of memory'
```

raised from DDP's gradient allreduce inside `loss.backward()`. That is **not**
`torch.OutOfMemoryError`, which is what a working set that does not fit produces and which
names the size it wanted. And the last memory line before it reads `peak 68.0 of 79.2`.

NCCL allocates its communicator and channel buffers with its own `cudaMalloc`, outside the
caching allocator. The three columns `_mem_report` printed — allocated, reserved, peak —
are all torch's view of torch's own pool, and none of them can see that memory or the
margin it needs. A report saying "68.0 of 79.2" was describing a card that had nothing
left to give NCCL.

`_mem_report` now prints `cuda.mem_get_info()`'s **free** as a fourth column, on every
rank rather than the main process, for every micro-step rather than the first six, with
the micro-step's own forwarded token count. `omni_mem_series.py` plots it.

What this does *not* yet establish is the mechanism — whether reserved growth crowds NCCL
out, whether it is specific to the resume path, or whether a fresh start is equally
exposed and got lucky. The instrumentation is there to answer it on the next failure
instead of after it.

### 12.3 There is no container-vs-`srun` gap

§11's "215.7 s on the submitted (container) path and 97-124 s on a bare `srun`" compares
two runs that were not doing the same work. The fast number comes from
`run-20260928_024250`, whose profiling reads:

```
_compute_overlap_step_maps    0.0 s
think_overlap_reward          0.0 s
```

and whose logged rewards are `think_format_reward/mean: 0` and
`think_overlap_reward/mean: nan` on every step. It is the pre-fix run from §9's table —
the one where the chat template already opens `<think>`, so every rollout scored as
malformed and **no saliency map was built at all**. The saliency capture is the largest
single item in the step, and that run skipped it.

Median step time from the wandb timestamps, same code path, same cluster:

| run | cap | overlap reward | step |
|---|---|---|---|
| `024250` | 1024 | **off** (format 0, overlap NaN) | **84.1 s** |
| `055534`, direct | 768 | live | **184.2 s** |
| `070121`, submitted | 1024 | live | **208.5 s** |

The 184 → 208 difference is the completion cap, not the container: at 768
`vLLM.generate` drops 37.4 → 27.9 s and the saliency capture 57.1 → 25.6 s. So the
container costs nothing measurable, and the honest reading of "97-124 s" is *what a step
costs with the reward under test turned off*.

### 12.4 The step budget, as far as rank 0 can see it

From `run-20260928_070121` (cap 1024, 29 profiled steps, medians; `_prepare_inputs` and
`compute_loss` are summed over their 8 calls per optimizer step):

| | s | |
|---|---|---|
| `_prepare_inputs` | **168.6** | envelope |
| ├ `_compute_overlap_step_maps` | 57.1 | 8 teacher-forced forwards + T5 segmentation |
| ├ `vLLM.generate` | 37.4 | |
| ├ `_calculate_rewards` | 6.9 | of which judge 4.6, DINO-backed overlap 2.0 |
| ├ `_move_model_to_vllm` | 0.2 | the 18-tensor sync is doing its job |
| └ **unattributed** | **67.0** | inline code with no method to wrap |
| `compute_loss` | 15.0 | forward only; `_get_per_token_logps_and_entropies` 9.2 of it |
| **outside both** | **~24.9** | backward, optimizer, DDP allreduce, dataloader, logging |
| observed step | **208.5** | |

Two corrections to the handoff's estimates, both large and both in the same direction:
generation is **37 s and not 7-9.8 s**, and the saliency capture is **57 s and not ~5 s**.
The ~5 s was extrapolated from a head-selection scan at 0.65 s per case; the trainer's
capture is an order of magnitude more per case, and the scan is not a measurement of it.

The 67 s is the target of `SR1_LAP` (see the block above `_mem_report` in
`grpo_trainer_qwen3.py`). It is inline code in the middle of
`_generate_and_score_completions` — the processor on eight native-resolution pictures,
`gather_object` on the images, the decode-and-pad, the logging gathers — none of it a
method, so `profiling_context` never saw it. The leading hypothesis is **straggler time**:
every number here is rank 0's, `_compute_overlap_step_maps` ranges 11-270 s across runs,
and a step costs the slowest rank. `_lap_barrier` is placed immediately after the capture
precisely to split "this rank's work" from "waiting for the worst one".

### 12.5 What this changes about the three levers

§11 lists `--length-guard` first. It bounds the **mean**, and the allocation that decides
whether a micro-step fits is set by that micro-step's **own** completion — `per_device` is
1 and `SR1_TRIM_COMPLETION_PADDING` trims to the sequence actually forwarded. A guard makes
cap-length micro-steps rarer; it cannot make the worst one smaller. Only the cap can.

And §11's evidence for drift does not survive its own table: mean length went **375 → 340**
over the 30 steps. "This reward lengthens chains" is not what that run shows.

## 13. The 50-step run: job 7103770, and what it cost

`grpo-omni30b-overlap__wov0.2_L19_h4-9_mean_in_c768`, the selected heads, **a real 768**,
one 4 h allocation on pool1-00186. **50/50 steps, no OOM, no requeue**, 3 h 05 m wall at
**222.5 s a step**. `checkpoint-50` is on disk and `completions/max_length` is 768 on every
step, so the cap was binding rather than decorative.

```
                    first 5    last 5      (50 logged steps, overlap never NaN)
overlap reward       0.0478    0.0535
within-group sd      0.0233    0.0259
total reward         0.880     0.947
format reward        0.542     0.583
judge reward         0.317     0.337
mean length            349       335
clipped ratio        0.188     0.154
```

**Read no attention conclusion off this.** §9 of `docs/omni-head-selection.md` applies in
full: `19 / 4,9` was selected on a corpus whose correctness label is 52% disputed, because
the Omni is a base checkpoint. This run establishes that the arm RUNS, at a stated cap,
with a live reward. It does not establish that the heads are the right ones.

### 13.1 The memory question, answered: it is a knife edge, not a drift

`outputs/omni_grpo_plan_a/.../mem_series.{csv,png}`, 7,200 reports, 6 ranks, 50 steps.
Worst rank per step:

| | step 0 | median over 50 | worst over 50 |
|---|---|---|---|
| allocated, entering compute_loss | 61.6 | 61.6 | 61.6 |
| **reserved**, entering compute_loss | 77.4 | 75.9 | **77.4** |
| **free (driver)** | 0.1 | 0.8 | **0.0** |

**Free reaches 0.0 GB on 9 of the 50 steps, and never exceeds 6 GB on any of them.** The
card is completely full at the driver level for part of every step, while torch's
*allocated* sits at 61.6 GB — the weights and nothing else. The gap is the caching
allocator: 15.8 GB reserved and owned by nothing, left behind by the previous micro-step's
backward. `SR1_EMPTY_CACHE_PER_MICROSTEP` does return it (free goes back to 15.1 at
"before the forward"), but it runs at the *start of the next micro-step* — after the
allreduce that needed the room.

So §10's model is wrong in the way that matters. The drift test, over the 50 steps:

| | r |
|---|---|
| reserved (worst rank) vs **step number** | **−0.073** |
| free (worst rank) vs **step number** | **+0.123** |
| `completions/mean_length` vs step number | +0.079 |
| reserved vs `completions/mean_length` | +0.375 |
| free vs `completions/mean_length` | −0.429 |

**Nothing climbs.** Memory does track length across steps — the bottom two rows — but
length itself is flat, and the relationship is loose because the batch MEAN is not the
variable that fills a card: the longest rollout on the fullest rank is. (That is why the
plot's middle panel measures the same physics against the micro-step's own token count,
where it is a straight line, and the right-hand panel is the looser view the question was
originally asked in.)

Mean completion length FALLS over this run (349 → 335) as it did over the previous run's
30 (375 → 340). The failure mode is not "a cap reached eventually", it is **a margin of
zero that is crossed when something asks for memory at the wrong instant** — which is
exactly what `ncclUnhandledCudaError` from DDP's allreduce is (§12.2). The honest answer to
"was it about to fail?" is: it was exactly as close to failing at step 49 as at step 0, and
that distance was zero on nine of the fifty.

The working set is linear in the sequence, cleanly, and now on the axis that decides it:

```
reserved after the forward  =  +7.90 GB per 1,000 tokens forwarded
```

which is what makes the cap the only lever that moves the peak. At 768 the longest
micro-step is ~1,190 positions and reserved after the forward reaches 72.2; the backward
adds ~5.2 more, to 77.4 of 79.2. At 1024 the same arithmetic gives ~1,446 positions,
~74.2 after the forward and **~79.4 after the backward, against 79.18 available.** That is
the 2026-09-28 failure, to within the rounding of these numbers.

**Caveat on one column.** `peak` is `torch.cuda.max_memory_allocated()`, a high-water mark
since the process started, so the flat 72.3 across all 50 steps means only "no micro-step
ever exceeded what step 0 already reached". It is not a per-step series and is not plotted
as one. A true per-step peak needs `reset_peak_memory_stats()` once per step; that is not
wired, and `reserved` and `free` — both instantaneous — carry the argument without it.

### 13.2 The step budget, closed

The `SR1_LAP` spans partition `_generate_and_score_completions` end to end, and on the mean
they sum to the envelope exactly:

| span | mean s | of the step |
|---|---|---|
| `prep_prompts` | 0.1 | |
| `generate_block` (incl. `vLLM.generate` 28.4) | 28.7 | 13% |
| `post_generate` | 0.1 | |
| `saliency_block` — this rank's 8 teacher-forced forwards | 48.8 | 22% |
| **`wait_ranks_after_saliency`** — waiting for the slowest rank | **48.3** | **22%** |
| `rewards_block` (judge 13.9, DINO-backed overlap 4.9) | 18.9 | 9% |
| `epilogue` (advantages, ~15 scalar gathers, 3 `gather_object`) | 38.8 | 17% |
| **`_prepare_inputs`** | **183.7** | |
| `compute_loss`, 8 micro-steps, forward only | 15.7 | 7% |
| outside both: backward, optimizer, DDP allreduce, dataloader | 23.1 | 10% |
| **step** | **222.5** | |

0.1 + 28.7 + 0.1 + 48.8 + 48.3 + 18.9 + 38.8 = 183.7, against `_prepare_inputs` at 183.7.
Nothing is left over.

**The single largest item is not compute.** The saliency capture costs this rank 48.8 s and
then this rank waits 48.3 s for a slower one — so the slowest rank spends about 97 s where
the main process spends 49, and the step pays the maximum. Every profiled number before
this was the main process's own share, which is why the capture looked like half of what it
costs. **Load imbalance is 22% of the step**, and it is imbalance rather than work: the six
ranks hold eight completions each whose lengths differ by a factor of ten, and the capture
is linear in tokens.

### 13.3 Two named fixes, with the measurement behind each

1. **Balance the saliency capture by token count, not by row count.** Measured cost of the
   present imbalance: **48.3 s a step, 22%.** The rollouts are handed out without regard to
   length and the capture is linear in length, so the upper bound on the recovery is most
   of that 48.3 s — a ~174 s step. Not yet implemented; it touches how the generation batch
   is split across ranks, which is a correctness-sensitive place.
2. **The 38.8 s epilogue — measured, but NOT yet attributed, and the first guess at it was
   wrong.** The span is real: 38.8 s, 17% of the step. The guess was that
   `self._logs["image"].extend(gather_object(images))` dominated it. Two things say
   otherwise, both checked after the fact:

   * **The payload is small.** set_a's images are capped at 512 px on the long side and
     pickle to a median of **0.50 MB**, so the whole step gathers **24 MB** across six
     ranks. Over NVLink that is milliseconds, and pickling 48 PIL images is a memcpy.
     Nothing there costs 38.8 s.
   * **The span holds 24 `.item()` calls and 12 `gather()` calls**, against 3
     `gather_object()`. Every `.item()` is a host-device synchronisation, and the first one
     after a stretch of asynchronous GPU work waits for all of it to drain. That is the
     better candidate, and it means the time may not belong to the epilogue at all — it may
     be the saliency capture's queued work, finally being waited on.

   So: no fix is named here yet. The span is now split three ways (`epilogue_metrics` /
   `epilogue_log_text` / `epilogue_log_images`) and the next run says which it is, at no
   cost. One small thing IS worth changing regardless: `if has_images:` should also test
   `self.log_completions`, because `self._logs` is a `deque(maxlen=48)` that is read only
   under that flag — a run without it does the gather for nothing. That is correctness, not
   a speed-up; **this run had `--log_completions` on, so its images were used.**

**`--length-guard` is still not wired, and neither run gives a reason to wire it.** §11
reaches for it against "this reward lengthens chains"; mean length fell in both runs (375 →
340 over 30 steps, 349 → 335 over 50) and `clipped_ratio` fell too (0.188 → 0.154). It
would also not help the memory: `per_device` is 1 and the padding is trimmed, so the
allocation that decides a micro-step is set by that micro-step's OWN completion. A guard
makes cap-length micro-steps rarer; only the cap makes the worst one smaller.

### 13.4 Is 222.5 s a step fast enough?

For this task, yes: it put 50 steps on the board inside one allocation with the margin
measured. For anything longer it is not — a 4,000-step run at this rate is ten days.

Of the 222.5 s, **one item has a named fix with a number behind it**: the 48.3 s of
straggler wait, 22%. The 38.8 s epilogue is 17% more that is measured but not yet
attributed, so it is a lead and not a fix. The model-compute floor (`compute_loss` + the
backward + the capture's own work) is around 90 s, so even taking both there is no version
of this that runs at Qwen3-VL's 22.5 s a step.

## 14. The fused Mamba kernels, and what they moved the bottleneck to

`grpo-omni30b-overlap__wov0.2_L19_h4-9_mean_in_c768_fused`, job 7185146: the same reward,
the same cap, the same heads as §13, with `causal_conv1d` and `mamba_ssm` installed and the
`vendor/mamba_ssm_min` stub no longer prepended to `PYTHONPATH`. **50/50 steps, no OOM, one
allocation, 2 h 13 m.**

### 14.1 What it bought

| | §13, naive | §14, fused | |
|---|---|---|---|
| step time | 222.5 s | **160.3 s** | **−28%** |
| wall for 50 steps | 3 h 05 m | **2 h 13 m** | |
| reserved, worst rank | 71.5–77.4 GB | **63.3–63.8 GB** | a 0.4 GB band |
| **free, worst rank** | **0.0–5.9 GB** | **13.1–14.0 GB** | |
| steps with ~0 free | **9 of 50** | **0 of 50** | |
| **memory per 1,000 tokens** | **+7.90 GB** | **+2.46 GB** | **3.2× flatter** |

The last row is the one with the longest reach. The naive path's working set grew more than
three times faster with sequence length, so the cap was not just low, it was low *because*
of the kernels. At +2.46 GB/1,000 and 13.1 GB spare there is room for roughly **5,300 more
tokens** — 1024 is comfortable, and the §10 memory argument that forced 768 no longer binds.

The one-card A/B that justified the switch (`bench_mamba_kernels_ab.sh`, job 7184255) said
31.21 → 19.39 s and 71.2 → 63.0 GB peak. The full run beat the memory prediction (13.1 GB
free against 8.2 GB predicted) and under-delivered on speed (−28% against −1.61×), and
§14.2 is why.

Rewards stayed healthy: total 0.9245 → 1.0246, format 0.5625 → 0.6208, judge 0.3428 →
0.3732, mean length 328 → 303 (falling again, for the third run running), entropy flat.
Do not compare the overlap reward against §13 in detail — different kernels give different
numbers, and 50 steps is not a trend.

### 14.2 The capture was never mostly model compute

`OVERLAP_PROFILE=1` has been in `grpo_trainer_qwen3.py` since it was written and had never
been switched on. Over 260 scored cases in 47 profiled steps:

| | total | per case | share |
|---|---|---|---|
| model forwards (one layer, no_grad) | 135.5 s | **0.52 s** | 8% |
| FLAN-T5 observe-step segmentation, on CPU | **1,611.6 s** | **6.20 s** | **92%** |

A 110M-parameter encoder on CPU costs twelve times a 30B model's forward pass. That is why
the fused kernels moved the step 28% and not 61%: they made the 8% cheaper.

And it is not steady work, it is contention. Per-case T5 ranges **0.53 s to 45.36 s**, a
85× spread with a median of 3.94 — six ranks on one node, each torch process sizing its
thread pool for all 96 cores. `OMP_NUM_THREADS` is not set anywhere in the launcher.

### 14.3 The budget, closed again

Lap means, which sum to `_prepare_inputs` exactly (54.2 + 28.5 + 35.7 + 8.5 + 4.0 + 0.1 +
0.1 = 131.1 against 131.1):

| span | naive | fused | |
|---|---|---|---|
| `wait_ranks_after_saliency` | 48.3 | **54.2** | **up** — see below |
| `saliency_block` | 48.8 | 35.7 | 92% of it is CPU T5 |
| `generate_block` | 28.7 | 28.5 | unchanged; vLLM has its own kernels |
| `compute_loss` | 15.7 | 11.0 | |
| `rewards_block` | 18.9 | 8.5 | |
| `epilogue*` | 38.8 | **4.0** | |
| outside both | 23.1 | ~18 | backward, optimizer, dataloader |
| **step** | **222.5** | **160.3** | |

Two rows deserve comment.

**The epilogue fell 38.8 → 4.0 s, and nothing was done to it.** §12.5 guessed the image
all-gather dominated it; that was withdrawn when the payload turned out to be 24 MB, and
the replacement hypothesis was that the span's 24 `.item()` calls were host-device syncs
draining GPU work queued earlier. This run settles it: `epilogue_log_images` and
`epilogue_log_text` both measure **0.0 s**, and the whole span collapsed once the GPU work
in front of it got faster. It was never the pictures.

**The straggler wait went UP, 48.3 → 54.2 s, and is now the largest single item at 34% of
the step.** That follows: the capture's GPU half got 1.6× faster while its CPU half did
not, so the imbalance is a larger share of a smaller step. The `[lap]` lines show how
extreme it is — rank 0 scored **zero** cases at step 1 and still waited 49.2 s — because
only format-valid completions are scored, so a rank's workload swings from 0 to 8 cases.

### 14.4 Where the time is now

T5 and the wait it causes are **54.2 + 0.92 × 35.7 ≈ 87 s of a 160 s step**. The two
changes that address it are both cheap and neither touches the objective:

1. **`OVERLAP_STEPS_DEVICE=cuda`.** The launcher pins the classifier to CPU and says why:
   "the training card is at 65-73 GB of 79 ... a 110M-parameter encoder is not worth one of
   the ~6 GB that are left". There are now **13 GB** left, and the premise is gone.
2. **`OMP_NUM_THREADS`.** Six ranks each sizing a thread pool for 96 cores is the obvious
   reading of an 85× per-case spread, and capping it costs nothing.

If T5 stops being the capture, the capture becomes 0.52 s/case — about 3 s a step — and
most of the 54.2 s wait goes with it, because the imbalance being waited on is T5's.
