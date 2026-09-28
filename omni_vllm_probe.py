#!/usr/bin/env python
"""Can one 80 GB card serve the Omni for generation, and how fast?

This is the question `docs/omni-gpu-layout.md` left open and `docs/omni-training-blockers.md`
calls the first of two blockers. Plan A gives the generation server ONE card, and the Omni
is 62 GB of weights at full size, which leaves under 18 GB for the key-value scratch space
generation needs. If that is not enough the trainer loses a card: Plan A', five training
processes at grad_accum 10, ~42 s a step instead of 34.

    python omni_vllm_probe.py --stage engine --gpu-mem 0.90
    python omni_vllm_probe.py --stage engine --gpu-mem 0.95 --max-model-len 4096

WHAT IT MEASURES, in the order the answer depends on them:

    the arch resolves          vLLM 0.11 has no NemotronH_Nano_Omni_Reasoning_V3 and its
                               nemotron_h has no MoE layer type at all, so this is a real
                               check and not a formality
    the KV cache fits          printed in tokens, against the longest real sequence
    it generates               8 rollouts of one real prompt with one real picture, which
                               is exactly the shape a GRPO step asks for
    how long that takes        against the ~13 s of generation + reward the Qwen3-VL run
                               pays per step

Run it in the `nemotron_vllm` environment, not `nemotron`: the Omni needs vLLM >= 0.20,
which pins torch 2.11, and the trainer stays on the 2.8 the 34.0 s figure was measured on.
"""
import argparse
import json
import os
import time
from pathlib import Path

MODEL = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"
CORPUS = Path("outputs/sink_location/xmodel/boxed/corpus")

SYSTEM_PROMPT = (
    "A conversation between user and assistant. The user asks a question, and the "
    "assistant solves it. The assistant first thinks about the reasoning process in the "
    "mind and then provides the user with the answer. The reasoning process and answer "
    "are enclosed within <think></think> tags, "
    "i.e., <think>\nThis is my reasoning.\n</think>\nThis is my answer."
)


def one_real_prompt():
    """A real picture and a real question, templated the way the trainer templates them.

    Not a synthetic string: the prompt length is set by the picture, the Omni is
    native-resolution, and the whole memory question turns on how many tokens a picture
    becomes.
    """
    from PIL import Image
    from transformers import AutoProcessor

    row = json.loads(open(CORPUS / "manifest.jsonl").readline())
    im = Image.open(CORPUS / row["image"]).convert("RGB")
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "image"},
                                         {"type": "text", "text": row["question"]}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    # How many tokens the PICTURE becomes, straight off the processor -- the number the
    # headroom argument in docs/omni-quantization.md rests on.
    got = proc(text=[text], images=[im], return_tensors="pt")
    n_img = int((got["input_ids"] == proc.tokenizer.convert_tokens_to_ids("<image>")).sum())
    return text, im, int(got["input_ids"].shape[1]), n_img, row["question"]


def kernel_config():
    """Keep vLLM off the kernels that want a CUDA toolkit this cluster does not have.

    The Omni is a mixture of experts, and vLLM's MoE backend `auto` picks FlashInfer's
    CUTLASS path, which JIT-COMPILES it on first use:

        flashinfer/jit/cpp_ext.py  get_cuda_path()
        RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda'
                      doesn't exist

    The nodes have no `/usr/local/cuda`; the only system toolkit is CUDA 12.4, against a
    torch built on 13. Triton's fused-MoE kernels need no toolkit at all -- triton ships
    its own compiler -- so that is what this asks for. Autotuning is off for the same
    reason: it is FlashInfer's.

    Worth knowing rather than silently avoiding: `triton` is the portable MoE backend, not
    the fastest one. If generation ever needs to be faster than it is, installing a CUDA 13
    `nvcc` into `nemotron_vllm` and dropping this is the lever.
    """
    return {"moe_backend": "triton", "enable_flashinfer_autotune": False}


def arm_watchdog(seconds):
    """Dump EVERY thread's Python stack after `seconds`, then exit.

    Engine startup on this model went quiet for 15 minutes with the process asleep in a
    futex, and none of the usual ways in were available: py-spy and gdb both need ptrace,
    which is off here. `faulthandler` is the one that needs no permission at all -- it is
    inside the process already -- and `dump_traceback_later` is a timer on it. With
    VLLM_ENABLE_V1_MULTIPROCESSING=0 the engine runs in THIS process, so the dump covers
    the thread that is actually stuck instead of a parent waiting on a pipe.
    """
    import faulthandler

    faulthandler.enable()
    faulthandler.dump_traceback_later(seconds, exit=True)
    print(f"[watchdog] armed: every thread's stack will be dumped after {seconds}s",
          flush=True)


def stage_engine(args):
    import torch
    from vllm import LLM, SamplingParams
    from vllm.model_executor.models.registry import ModelRegistry

    archs = ModelRegistry.get_supported_archs()
    print(f"vLLM registry: NemotronH_Nano_Omni_Reasoning_V3 present = "
          f"{'NemotronH_Nano_Omni_Reasoning_V3' in archs}")
    if "NemotronH_Nano_Omni_Reasoning_V3" not in archs:
        raise SystemExit("this vLLM cannot serve the Omni; upgrade it (>= 0.20) first")

    text, image, prompt_len, n_img, question = one_real_prompt()
    print(f"prompt: {prompt_len} tokens, of which {n_img} are the picture")
    print(f"question: {question[:90]}")

    t0 = time.time()
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        enforce_eager=bool(args.enforce_eager),
        dtype="bfloat16",
        trust_remote_code=True,
        enable_prefix_caching=True,
        limit_mm_per_prompt={"image": 1},
        # A Mamba hybrid needs ONE state block per concurrently decoding sequence, and on
        # this card there are 914 of them. vLLM's default `max_num_seqs` is 1024, so CUDA
        # graph capture refuses before anything runs. A GRPO step asks for 48 at most (6
        # prompts x 8 rollouts); 64 leaves margin and is nowhere near the cap.
        max_num_seqs=args.max_num_seqs,
        kernel_config=kernel_config(),
    )
    load_s = time.time() - t0
    print(f"engine up in {load_s:.0f}s")

    # The KV cache vLLM actually got, which is the whole question. `num_gpu_blocks` x
    # `block_size` is how many TOKENS it can hold at once across all sequences.
    try:
        cache = llm.llm_engine.vllm_config.cache_config
        blocks = getattr(cache, "num_gpu_blocks", None) or getattr(
            cache, "num_gpu_blocks_override", None)
        print(f"KV cache: {blocks} blocks x {cache.block_size} = "
              f"{(blocks or 0) * cache.block_size} tokens")
    except Exception as exc:                       # a private path; never fatal
        print(f"(could not read the cache config: {type(exc).__name__}: {exc})")

    sp = SamplingParams(n=args.n, temperature=1.0, top_p=1.0, top_k=0,
                        max_tokens=args.max_tokens)
    row = {"prompt": text, "multi_modal_data": {"image": image}}

    for trial in range(args.trials + 1):
        torch.cuda.synchronize()
        t0 = time.time()
        out = llm.generate([row], sp)
        torch.cuda.synchronize()
        dt = time.time() - t0
        lens = [len(c.token_ids) for c in out[0].outputs]
        label = "warmup" if trial == 0 else f"trial {trial}"
        print(f"    {label}: {dt:.1f}s for {len(lens)} rollouts, "
              f"lengths {min(lens)}-{max(lens)} (mean {sum(lens)/len(lens):.0f})",
              flush=True)

    print("\nfirst rollout, first 400 chars:")
    print(out[0].outputs[0].text[:400])

    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"\npeak torch-allocated on this process: {peak:.1f} GB "
          "(vLLM pre-reserves its pool, so this is a floor, not the card's total)")
    print(f"\nVERDICT: one card at gpu_memory_utilization={args.gpu_mem} "
          f"max_model_len={args.max_model_len} SERVED the Omni. Plan A holds.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="engine", choices=["engine"])
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--enforce-eager", type=int, default=0)
    ap.add_argument("--n", type=int, default=8, help="rollouts per prompt, as GRPO does")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--max-num-seqs", type=int, default=64,
                    help="concurrent sequences; one Mamba state block each")
    ap.add_argument("--watchdog", type=int, default=0,
                    help="seconds before dumping every thread's stack and exiting; 0 off")
    args = ap.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if args.watchdog:
        arm_watchdog(args.watchdog)
    stage_engine(args)


if __name__ == "__main__":
    main()
