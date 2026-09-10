# Running LASER's own trainer on Qwen3-VL-8B — 2026-09-10

[laser-go-no-go.md](laser-go-no-go.md) measured LASER's two rewards on our model and
returned NO-GO on both. This document is the other thing: **how to train their method
anyway**, in their code, on Qwen3-VL-8B — because "we ran it and it did X" is worth more
as a baseline than "we predicted it would not work", and because the go/no-go is a
statement about an untrained policy.

**The port is one file.** Everything else in their fork already speaks Qwen3-VL.

## 1. What was already there, and what the wiki page got wrong about it

`../vlm_reasoning/wiki/laser-implementation.md` called Path B blocked by
`verl/workers/actor/attention_capture.py` and its deliberate `transformers==4.57.6` pin.
Both halves of that are wrong:

- **4.57.6 ships Qwen3-VL.** Their own `verl/models/transformers/qwen3_vl.py` does
  `from transformers.models.qwen3_vl.modeling_qwen3_vl import ...`. The pin is not a
  Qwen2.5-VL ceiling.
- **Their verl is already Qwen3-VL-aware end to end.** `qwen3_vl.py` provides
  `get_rope_index` (interleaved MRoPE), `_get_input_embeds`, `Qwen3VLCausalLMOutputForPPO`
  and three backend forwards; `models/mcore/registry.py`, `models/transformers/monkey_patch.py`,
  `utils/flops_counter.py`, `utils/vllm/patch.py` and `utils/dataset/rl_dataset.py` all
  reference it. verl 0.7.0.dev, vllm 0.11.0, torch 2.8.0.
- **Nothing else is Qwen2.5-shaped.** `dp_actor.py`, `fsdp_workers.py` and
  `reward_score/openr1_verl.py` contain zero `Qwen2_5` references. The image token is
  `<|image_pad|>` in both families.

So the only Qwen2.5-VL-specific code in the repo is the attention capture, and `train.sh`
defaults `APPLY_HOOK_ATTENTION=True`, so that is the path that runs.

## 2. The one change

`laser/attention_capture.py` in this repo, installed by `patch_laser_qwen3.sh`.

Upstream mirrors `Qwen2_5_VLAttention.forward` byte-for-byte — 266 lines, and the reason
for the pin — purely to get post-RoPE `query_states` and `key_states` on their way past.
**That mirror is unnecessary.** An attention interface function already receives them,
post-RoPE and (on Qwen3-VL) post-QK-norm:

```python
query_states = self.q_norm(self.q_proj(h).view(shape)).transpose(1, 2)
key_states   = self.k_norm(self.k_proj(h).view(shape)).transpose(1, 2)
query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
...
attention_interface(self, query_states, key_states, value_states, attention_mask, ...)
```

Intercepting `attention_interface` therefore gets exactly what the mirror produced, with
`q_norm`/`k_norm` applied — the easy thing to drop when re-deriving by hand, and worth
0.5–2.2 in max absolute deviation on a random head, i.e. not a rounding difference — and
with interleaved MRoPE already baked into `cos`/`sin`. None of the wiki page's ten-point
delta (`mrope_section`, the cache API, `get_interface`, the output reshape) has to be
restated, which is also why this file should not need touching the next time Qwen3-VL's
forward changes.

**It wraps the function behind the existing `_attn_implementation` entry rather than
registering a new implementation.** That is the part to not "simplify" later: verl
branches on that string — padding-free packing, Ulysses SP monkey patches, which kwargs
reach FlashAttention — so repointing the config would change the forward's semantics while
looking like a no-op. `ALL_ATTENTION_FUNCTIONS.__setitem__` writes an instance-local
override that `__getitem__` consults first; 5.x's `get_interface` ends in `super().get()`,
so the same swap covers both dispatch paths. 4.57.6's *eager* path never consults the
registry at all, so the module-level `eager_attention_forward` symbol is wrapped instead.

Text attention only, filtered by **module identity**: Qwen3-VL's vision tower dispatches
through the same registry entry, and upstream got that exclusion for free from a
class-name mismatch it no longer has.

## 3. Setting it up

```fish
# the fork (already done; origin is upstream, work is on branch qwen3-vl)
git clone https://github.com/KeViNYuAn0314/LASER laser_fork
cd laser_fork; and git checkout -b qwen3-vl; and cd ..

bash patch_laser_qwen3.sh                    # installs the one file, checks the API
python test_laser_capture_cpu.py             # CPU, no GPU, no checkpoint
```

The env is **separate on purpose** — `requirements_laser.txt` pins torch 2.8.0,
transformers 4.57.6 and vllm 0.11.0, none of which match `saliency_r1_qwen3_vllm`.
Build it as its own conda env; do not `pip install` into a shared one
([CLAUDE.md](../CLAUDE.md): the envs are global, not worktree-local).

Then in `train.sh`: `MODEL_PATH` to
`checkpoint/coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged`, `EXP_NAME` to
something Qwen3-shaped, and the data to their 45K filtered from MMR1-RL and ReVisual-R1
(both public; the composition table is in their supplement).

## 4. What the CPU test already establishes

`test_laser_capture_cpu.py`, on a 3-layer Qwen3-VL text stack with 4 query heads and 2 KV
heads so grouped-query attention is genuinely exercised:

- **`compute_slice_for_sample` reproduces transformers' own `output_attentions=True`
  attention to `max |delta| = 0.00e+00`.** This is the whole risk of the swap. If the
  interface had handed back pre-RoPE or pre-QK-norm tensors, the reward would have been
  computed from plausible-looking wrong numbers and no training curve would have shown it.
- The forward stays **bit-identical** while capturing.
- `_attn_implementation` is never renamed.
- The override is removed on exit, including when the body raises, and a *prior* local
  override is restored rather than deleted.
- Padding-masked keys receive exactly zero, which is what verl's left-padded prompts need.
- The eager path captures too.

## 5. What is NOT established, in the order it will bite

1. **The capture under FSDP.** `find_text_attention_modules` walks `root.modules()` and
   FSDP wraps modules without renaming their classes, so it should hold — but it is
   untested against a real `FSDPModule`, and it is the first thing a smoke run exercises.
   Upstream's version walked the same tree, so this is inherited risk, not new risk.
2. **vLLM 0.11.0 rolling out Qwen3-VL inside their verl.** Their `utils/vllm/patch.py`
   references qwen3_vl, which is a good sign and not a guarantee.
3. **The env building at all** — torch 2.8.0 / vllm 0.11.0 / transformers 4.57.6 against
   this cluster's CUDA. Their README says the wheel is compiled in a container.
4. **The data.** Not downloaded. 45K from MMR1-RL + ReVisual-R1.
5. **`_generate_attentions_experimental_impl`** — the NON-hooked path, which
   `APPLY_HOOK_ATTENTION=False` selects. It stacks `output.attentions` and then indexes
   `layer_avg_attentions[:, i, bos_pos, visual_token_indices]`, whose comment claims the
   stack is `(num_layers, batch, seq, seq)` when HF returns `(B, H, T, T)` per layer. If
   that is right, the head dimension is being indexed with `bos_pos`. **Leave
   `APPLY_HOOK_ATTENTION=True`** and do not use that path without reading it first.

## 6. The smoke run

**It needs GPUs** — FSDP plus a vLLM rollout of an 8B VLM; there is no CPU version of
this rung. What CPU buys you is §4, which is the part most likely to be silently wrong.

Smallest thing that proves the port:

```fish
# 1 node, 2 GPUs, a handful of steps, tiny batches
env ATTENTION_START_STEP=1 APPLY_HOOK_ATTENTION=True \
    APPLY_EARLY_WEIGHTED_STABILITY=True \
    MODEL_PATH=... TRAIN_FILES=... bash train.sh
```

`ATTENTION_START_STEP=1` matters: at its default of 20 the first twenty steps are plain
GRPO, so a broken capture would look like a working run for twenty steps and then fail.

Watch, in this order: the job reaches step 1 without an `AttentionSliceCapturer` import or
discovery error; `attention_score` and `suppression_attention_score` appear in the reward
extra-info and are **not identically zero** (zero is what a silently empty capture looks
like); and `attention_score` lands near the 0.4–0.5 our own measurement saw, with
`suppression_attention_score` near 0.36. Those two numbers are the cheapest available
cross-check that the port computes what `laser.py` computed independently, on a different
code path, from the same model.

## 7. Cross-check available for free

`laser.py` + `laser_probe.py` in this repo compute the same two rewards from a completely
separate implementation (registered attention impl, our own trainer's teacher-forced
construction, pure-numpy reward math checked against a torch rewrite of upstream's lines
to 4e-15). If the fork's `attention_score` disagrees materially with
`outputs/laser/coldstart/report.txt`'s `R_vis` distribution on the same checkpoint, one of
the two is wrong and the disagreement is worth more than either number alone.
