# GRPO overlap-reward training — performance notes

Diagnosed 2026-07-22. This page tracks why the `reward_variant=ours` (attention-overlap)
GRPO run is slow and the optimizations to fix it, ranked by payoff.

## Setup (so we don't re-litigate it)

- vLLM runs in **server mode** (`--vllm_mode server`), on its own GPU. GPU layout for
  the 8-GPU launcher (`launch_grpo_qwen3_overlap_colocated_job.sh`): `DINO=gpu0,
  vLLM=gpu1, train=gpu2-7` (6 training procs). "Colocated" in the launcher name only
  means the DINO + vLLM **sidecars share the node** — it is *not* TRL `vllm_mode=colocate`.
  **Generation is not the bottleneck.**
- Batch: `per_device=1 × num_generations=8 × grad_accum=8` ⇒ **48 completions scored per
  optimizer step per device**. `max_prompt_length=2048`, `max_completion_length=1024`.
- Reward stack (weights `1.0 0.2 1.0 1.0` = format, overlap, accuracy, judge):
  - **overlap** — VLM self-attention at **layer 22, heads [28,31]** scored against
    Grounding-DINO boxes. Attention comes from a re-forward of the policy (see below).
  - **judge** — LLM-as-judge via the NVIDIA/OpenAI gateway (`openai_rewards.py`).
  - DINO grounding runs on its own GPU over HTTP (batched, off the training GPUs).
  - FLAN-T5 observe-step classifier runs on **CPU** (`OVERLAP_STEPS_DEVICE=cpu`).

## Where the wall-clock goes

Per optimizer step, per training GPU, two things run **48× serially**:

1. **Layer-22 attention re-forward** — `_compute_overlap_step_maps`
   (`trl/grpo_trainer_qwen3.py`). One separate full 8B forward *per completion* to get
   attention. `output_attentions=True` forces **eager attention across all 36 layers**
   and materializes every layer's `[heads, seq, seq]` map — but only layer 22 is used.
   ≈ `160h / 3990 ≈ 2.4 min/step`, which matches ~48 eager 8B forwards.
2. **Judge reward** — `openai_rewards.py:openai_reward` is a **serial list-comp**, one
   synchronous `chat.completions.create` per completion (`max_tokens=512`, `timeout=120`,
   retry/backoff). 48 network round-trips on the critical path.

Smaller, also on the path: FLAN-T5 classifier on CPU per completion; DINO HTTP.

## Optimizations (ranked)

| # | Change | Status |
|---|--------|--------|
| 1 | **Layer-22 only**: recompute attention for just layer 22, keep the fast attn impl for the other 35 layers | ✅ done (this branch) |
| 2 | **Batch the re-forward**: the per-completion `for` loop does 48 separate forwards on equal-length padded seqs — batch them | pending |
| 3 | **Fuse into an existing forward**: attach the layer-22 capture to the logprob forward GRPO already runs and drop `reforward_saliency` (check the `--reforward_saliency False` path first) | pending |
| 4 | **Judge**: run the 48 API calls concurrently (thread pool / async) and drop `max_tokens` to ~8 (only needs `Score: <1-5>`) | pending |
| 5 | **T5 classifier off CPU / batched**: move to a GPU or batch all sentences in one forward | pending |
| 6 | **Cap image resolution** (`max_pixels`): fewer image patches speeds generation, the training forward, *and* the attention re-forward's key dim | pending |

### Attribution kill-switches
- `DISABLE_OVERLAP_FORWARD=1` — skips the whole re-forward (overlap reward → 0); isolates cost #1.
- Zero the judge reward weight — isolates cost #2.

## #1 — implemented

`_compute_overlap_step_maps` no longer calls `output_attentions=True`. Instead it:
- runs the per-completion forward on the model's normal attn impl (flash/sdpa — fast, no
  weight materialization), and
- registers a forward hook on the **layer-22** `Qwen3VLTextAttention` module that re-runs
  *only that one module* in eager mode to recover its softmax weights.

The re-run reuses the module's own `q/k/v` projections, `q_norm`/`k_norm` and rotary, so
the weights are numerically identical to the all-layer path (verified on a tiny CPU model:
max abs diff ≈ 1.8e-7, including left-padding). The one thing supplied explicitly is the
additive **causal + padding mask** (`0` where attended, `finfo.min` where masked), because
the fast attention paths may hand the layer a `None` mask. KV cache is bypassed on the
re-run (`past_key_values=None`, `use_cache=False`) to avoid double-updating it.

Net cost per completion: **~1 fast forward + 1 layer of eager attention** (was: 1 all-eager
36-layer forward with 36× weight materialization). Requires all-full-attention layers
(Qwen3-VL-8B has no sliding-window layers — verified). If the layer-22 module isn't found
(unexpected model layout), it falls back to the old `output_attentions=True` path.
