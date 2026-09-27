#!/usr/bin/env python
"""How long is a training step on an 8-bit Omni that fits on one card?

WHAT THIS IS, AND WHAT IT IS NOT. The question is whether shrinking the frozen base to 8
bits -- 34.8 GB, so a whole copy fits on one 80 GB card -- removes the cost that makes a
Nemotron step slow. That cost is the trainer cutting 66 GB of weights across cards and
re-collecting them 16 times a step. It lives entirely on the TRAINING side.

So this measures the training side, faithfully, and nothing else:

    a saliency re-forward with no gradient   (the run does one; --reforward_saliency True)
    a forward with gradient
    a backward
    x8 micro-steps                           (per_device_train_batch_size 1, grad_accum 8)
    an optimizer step
    x20 steps

with the real settings off `launch_grpo_qwen3_overlap_colocated_job.sh`: LoRA r=16
alpha=32 on q_proj/k_proj/v_proj, lr 1e-5, completions of 1024 tokens, no gradient
checkpointing, beta 0 so there is no reference-model forward.

IT DOES NOT generate the rollouts or score them. Those are a fixed cost that the
quantization decision does not touch -- on the Qwen3-VL run they are about 13 s of a
22.5 s step -- and standing them up for the Omni is blocked on two separate things, both
recorded in docs/omni-training-blockers.md. Adding this number to that fixed cost is the
honest estimate of a whole step; calling this number "a GRPO step" would not be.

IT ALSO ANSWERS THE RISKIEST UNKNOWN, which is not speed. The learning signal has to
travel backwards through 23 Mamba layers to reach the 6 attention layers the LoRA sits
on, and those layers are running a torch fallback rather than the fused kernels. If that
produces a crash, a non-finite gradient, or a zero gradient, the plan is dead whatever
the step time is. `--check-grads` is that test and it runs by default.

    python omni_train_step_bench.py --steps 20
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import sink_location_probe as SLP

MODEL = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"
CORPUS = Path("outputs/sink_location/xmodel/boxed/corpus")

# Straight off the launcher -- but SCOPED TO THE DECODER, which the launcher does not
# have to be. `q_proj,k_proj,v_proj` as bare names matches by suffix anywhere in the
# model, and this is an OMNI: it carries an audio tower whose 24 layers use exactly those
# names. On an image-only batch that tower never runs, so 144 of 180 LoRA tensors came
# back with no gradient at all -- not a broken backward pass, just a LoRA bolted onto
# code that was never executed. Qwen3-VL never showed this because its vision tower uses
# a fused `qkv`, and RADIO uses one too; the audio tower is what is new here.
LORA_TARGETS = r"language_model\..*\.(q_proj|k_proj|v_proj)"
LORA_R, LORA_ALPHA = 16, 32
LR = 1e-5
GRAD_ACCUM = 8
COMPLETION_LEN = 1024


def enable_grad_ckpt(model):
    """Recompute the decoder's intermediates instead of storing them.

    Worth doing because the measured footprint says the weights are not the problem: at 8
    bits the model is 34.8 GB and a training step peaks at 63.8, so ~30 GB is
    intermediate results held for the backward pass. That 30 GB is what puts a 16-bit run
    over an 80 GB card, not the 62 GB of weights.

    `NemotronHBlock` inherits `GradientCheckpointingLayer`, so transformers' own switch
    reaches it -- a plain grep for "gradient_checkpointing" in that file finds nothing and
    says otherwise, which is why this is asserted rather than assumed.

    Enabled on the LANGUAGE MODEL, not the wrapper: the vision tower runs under
    `no_grad` and has nothing to recompute, and narrowing it keeps the switch away from
    code whose behaviour under recomputation nobody here has checked.
    """
    lm = getattr(model, "language_model", None)
    if lm is None:
        raise SystemExit("no .language_model to enable gradient checkpointing on")
    lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    on = sum(1 for m in lm.modules() if getattr(m, "gradient_checkpointing", False))
    print(f"gradient checkpointing ON for {on} decoder blocks")
    if on == 0:
        raise SystemExit("gradient_checkpointing_enable() left every block untouched -- "
                         "it silently did nothing, and the memory would be unchanged")
    return on


def build(device, bits, grad_ckpt=False):
    from peft import LoraConfig, get_peft_model
    from transformers import BitsAndBytesConfig

    quant = None
    if bits == 8:
        quant = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_skip_modules=["vision_model", "mlp1", "lm_head"])
    elif bits != 16:
        raise SystemExit(f"--bits {bits}: expected 8 or 16")

    t0 = time.time()
    proc, model = SLP.load_model(MODEL, None, device, "sdpa", quant=quant)
    load_s = time.time() - t0
    if grad_ckpt:
        enable_grad_ckpt(model)

    # Only the LoRA trains. Everything else is frozen, which is what licenses shrinking
    # the base at all -- nothing is ever written back into it.
    cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=LORA_TARGETS,
                     lora_dropout=0.0, bias="none")
    model = get_peft_model(model, cfg)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    trained = [n for n, p in model.named_parameters() if p.requires_grad]
    hit = sorted({n.split(".lora_")[0].split(".")[-1] for n in trained})
    layers = sorted({int(n.split("layers.")[1].split(".")[0])
                     for n in trained if "layers." in n})
    print(f"loaded in {load_s:.0f}s | trainable {n_train/1e6:.2f}M of {n_total/1e9:.1f}B "
          f"({n_train/n_total*100:.4f}%)")
    print(f"LoRA landed on {hit} in decoder layers {layers}")

    # The audio tower is the trap this guards. Anything outside `language_model` is code
    # an image-only batch never runs, so it would train nothing while looking like it was
    # configured to.
    stray = sorted({n.split(".lora_")[0] for n in trained if "language_model." not in n})
    if stray:
        raise SystemExit(f"LoRA landed outside the decoder on {len(stray)} modules, "
                         f"e.g. {stray[:3]} -- those never run on an image-only batch")
    want = [5, 12, 19, 26, 33, 42]
    if layers != want:
        raise SystemExit(f"LoRA is on decoder layers {layers}, not the attention layers "
                         f"{want}: q/k/v_proj exist only inside attention, so anything "
                         "else means the match went somewhere unintended")
    model.train()
    return proc, model, load_s


def one_batch(proc, model, device):
    """One real picture, one real question, and a completion of the real length.

    The completion ids are sampled rather than generated: this is a timing and gradient
    test, and what the model would have written changes neither the shapes nor the graph.
    The PROMPT is real, because its length is set by the picture and the Omni's grid is
    native-resolution.
    """
    import vlm_family as VF

    row = json.loads(open(CORPUS / "manifest.jsonl").readline())
    im = Image.open(CORPUS / row["image"]).convert("RGB")
    fam = VF.family_for(model.base_model.model, proc)
    fam.bind(model=model.base_model.model, processor=proc,
             config=model.base_model.model.config)
    inputs = fam.build_inputs(proc, [im], row["question"], device)
    prompt_len = int(inputs["input_ids"].shape[1])

    g = torch.Generator(device="cpu").manual_seed(0)
    vocab = int(model.base_model.model.config.llm_config.vocab_size)
    comp = torch.randint(0, vocab, (1, COMPLETION_LEN), generator=g).to(device)
    ids = torch.cat([inputs["input_ids"], comp], dim=1)

    case = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    for k in ("pixel_values", "image_flags"):
        if inputs.get(k) is not None:
            case[k] = inputs[k]
    n_img = int((inputs["input_ids"] == fam.image_token_id).sum())
    print(f"batch: prompt {prompt_len} ({n_img} picture tokens) + completion "
          f"{COMPLETION_LEN} = {ids.shape[1]} positions")
    return case, prompt_len


def loss_of(model, case, prompt_len):
    """GRPO's shape: per-token log-probabilities of the completion, times an advantage.

    Not a cross-entropy against the batch's own ids, because that is a different graph
    from the one the real objective builds. The advantage is a constant here -- its VALUE
    is what training would learn from and has nothing to do with what a step costs.
    """
    out = model(**case, use_cache=False)
    logits = out.logits[:, prompt_len - 1:-1, :]
    tgt = case["input_ids"][:, prompt_len:]
    logp = torch.log_softmax(logits.float(), dim=-1)
    tok = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return -(tok * 1.0).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--grad-ckpt", type=int, default=0,
                    help="recompute the decoder's intermediates instead of storing them")
    ap.add_argument("--reforward", type=int, default=1,
                    help="the no-grad saliency forward the run does each micro-step")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    device = "cuda:0"

    # `reset_peak_memory_stats` on a device that has never been touched raises "Invalid
    # device argument", so the context has to exist first. Reset BEFORE the load, because
    # loading is where the base weights land and their footprint is half the question.
    torch.cuda.init()
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    proc, model, load_s = build(device, args.bits, bool(args.grad_ckpt))
    case, prompt_len = one_batch(proc, model, device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)

    # --- the gradient test, before any timing -------------------------------
    # Riskiest unknown first: does the signal survive the trip back through the Mamba
    # layers at all? A step time is meaningless if the answer is no.
    #
    # ONE OPTIMIZER STEP FIRST, and it is not a warm-up. LoRA initialises `lora_B` to
    # zero, so at step 0 the branch contributes nothing and the gradient w.r.t. `lora_A`
    # is exactly zero BY CONSTRUCTION -- half the tensors would read as dead in a
    # perfectly healthy model. After one step `lora_B` is non-zero and every tensor
    # should carry signal, which is the condition worth asserting.
    (loss_of(model, case, prompt_len)).backward()
    opt.step()
    opt.zero_grad(set_to_none=True)

    loss = loss_of(model, case, prompt_len)
    loss.backward()
    grads = [(n, p.grad) for n, p in model.named_parameters() if p.requires_grad]
    missing = [n for n, g in grads if g is None]
    nonfinite = [n for n, g in grads if g is not None and not torch.isfinite(g).all()]
    norms = np.array([float(g.norm()) for _n, g in grads if g is not None])
    n_zero = int((norms == 0).sum())
    print(f"\ngradient check on {len(grads)} LoRA tensors, after one optimizer step")
    print(f"    loss {float(loss.detach()):.4f}")
    print(f"    no gradient at all : {len(missing)}")
    print(f"    non-finite         : {len(nonfinite)}")
    print(f"    exactly zero       : {n_zero}")
    if norms.size:
        print(f"    norm  median {np.median(norms):.3e}  min {norms.min():.3e}  "
              f"max {norms.max():.3e}")
    if missing or nonfinite or n_zero:
        raise SystemExit(
            "the learning signal does not reach the LoRA intact -- stop here, the step "
            "time does not matter. Check the backward path through the Mamba layers "
            "before reading anything else in this file.")
    print("    PASS: the signal reaches every LoRA tensor, finite and non-zero,")
    print("          which means the backward pass through the 23 Mamba layers works")
    opt.zero_grad(set_to_none=True)

    # --- the timing ---------------------------------------------------------
    print(f"\n{args.steps} steps of {GRAD_ACCUM} micro-steps"
          f"{' (+1 no-grad saliency forward each)' if args.reforward else ''}")
    times = []
    for step in range(args.steps + 1):          # step 0 is warmup, discarded
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(GRAD_ACCUM):
            if args.reforward:
                with torch.no_grad():
                    model(**case, use_cache=False)
            (loss_of(model, case, prompt_len) / GRAD_ACCUM).backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        dt = time.time() - t0
        if step == 0:
            print(f"    warmup {dt:.1f}s (discarded)", flush=True)
            continue
        times.append(dt)
        if step % 5 == 0 or step == 1:
            print(f"    step {step:>3}  {dt:.1f}s", flush=True)

    t = np.array(times)
    peak = torch.cuda.max_memory_allocated(device) / 2**30
    ck = "recompute ON" if args.grad_ckpt else "recompute OFF"
    print(f"\n{'='*70}\nTRAINING SIDE OF ONE STEP, {args.bits}-bit base, one card, "
          f"no weight-splitting, {ck}\n{'='*70}")
    print(f"    median   {np.median(t):.1f}s")
    print(f"    mean     {t.mean():.1f}s   sd {t.std():.1f}s")
    print(f"    min/max  {t.min():.1f}s / {t.max():.1f}s")
    print(f"    peak GPU {peak:.1f} GB of 79.2 -- headroom {79.2-peak:.1f} GB")
    print(f"\n    3,990 steps at the median = {np.median(t)*3990/3600:.0f} h, "
          f"training side only")
    print("    generation and reward are extra and unchanged by this decision "
          "(~13s/step on the Qwen3-VL run)")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"bits": args.bits, "grad_ckpt": bool(args.grad_ckpt), "steps": args.steps, "reforward": bool(args.reforward),
             "median_s": float(np.median(t)), "mean_s": float(t.mean()),
             "sd_s": float(t.std()), "peak_gb": float(peak), "load_s": load_s,
             "prompt_len": prompt_len, "completion_len": COMPLETION_LEN,
             "grad_accum": GRAD_ACCUM, "times": [float(x) for x in t]}, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
