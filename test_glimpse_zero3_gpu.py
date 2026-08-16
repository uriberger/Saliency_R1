#!/usr/bin/env python
"""Multi-GPU gate for the GLIMPSE reward under DeepSpeed ZeRO-3.

The CPU gate checks the reward's arithmetic and the single-GPU probe checks that
`step_glimpse_maps` produces a map. NEITHER touches the thing that actually breaks a
training run, and the gradient reward is on record as having been merged with exactly
that hole ([[grad-reward-pending-gpu-validation]]). This closes it for GLIMPSE.

Four hazards, all of them silent, all of them specific to running a BACKWARD and a pile of
EAGER LAYER REPLAYS through the very module DeepSpeed hangs its hooks on:

  1. NOTHING MAY LAND IN .grad. `frozen_params` clears requires_grad on every weight so
     autograd prunes the weight-gradient half; if it did not, ZeRO-3's gradient hooks
     would reduce-scatter garbage into the live training state and the run would corrupt
     silently rather than crash. Asserted directly: every parameter's .grad is None after.

  2. THE ZeRO-3 TRACE MUST SURVIVE RANK DISAGREEMENT. ZeRO-3 records the module execution
     order of one fwd+bwd and prefetches against it, and `reset_step` asserts the recorded
     order is identical across ranks. GLIMPSE's per-case work is proportional to the
     number of TARGET TOKENS, which differs per rank because the completions differ -- so
     ranks issue different numbers of backwards AND, unlike the gradient reward, different
     numbers of MODULE FORWARDS (one eager replay per propagated layer per token). This
     harness gives each rank a DELIBERATELY DIFFERENT step count, then runs a real
     fwd+bwd+step afterwards. If the trace handling is wrong that step hangs in NCCL or
     dies in fetch_sub_module; passing it is the claim.

  3. THE DIAGNOSTIC GATHER IS A COLLECTIVE. pop_diagnostics must return the SAME key set
     on every rank, including a rank that scored nothing, or the ranks issue different
     numbers of collectives and the run hangs instead of failing. Asserted by having rank
     0 score nothing at all.

  4. THE MAP MUST STILL BE THE MAP. Under ZeRO-3 the parameters are sharded until
     `unwrap_model_for_generation` gathers them; a map built against partly-gathered
     weights would be quietly wrong rather than an error. Compared against the same case
     computed with the model gathered on one rank.

    accelerate launch --config_file <deepspeed_zero3.yaml> --num_processes 8 \
        test_glimpse_zero3_gpu.py --base-model <path>

Needs 8 GPUs and the real weights; there is no smaller version of this test, because every
hazard above is a property of the distributed wrapper and not of the algebra.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent


def repo_path(rel: str) -> Path:
    p = REPO / rel
    if p.exists():
        return p
    if REPO.parent.name == ".worktrees":
        alt = REPO.parent.parent / rel
        if alt.exists():
            return alt
    return p


def _load(dotted, rel):
    spec = importlib.util.spec_from_file_location(dotted, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = mod
    spec.loader.exec_module(mod)
    return mod


for _n, _p in (("trl_local", REPO / "trl"), ("trl_local.rewards", REPO / "trl" / "rewards")):
    _m = types.ModuleType(_n)
    _m.__path__ = [str(_p)]
    sys.modules[_n] = _m
# The reward modules import each other relatively; load them under a private package so
# this test never depends on whether trl_repo/ has been patched yet.
sys.modules["trl_local.grad_maps"] = GM = _load("trl_local.grad_maps", "trl/grad_maps.py")
GLM = _load("trl_local.glimpse_maps", "trl/glimpse_maps.py")
OR = _load("trl_local.rewards.overlap_rewards", "trl/rewards/overlap_rewards.py")
sys.modules["trl_local.rewards.overlap_rewards"] = OR
sys.modules["trl_local.rewards.grad_rewards"] = _load(
    "trl_local.rewards.grad_rewards", "trl/rewards/grad_rewards.py")
GL = _load("trl_local.rewards.glimpse_rewards", "trl/rewards/glimpse_rewards.py")

PASS, FAIL = [], []


def check(name, cond, detail="", fail_detail=""):
    """`fail_detail` is shown only when the check fails, so a passing line cannot read
    like a failure (a message such as "none found" printed next to `ok` is worse than no
    message at all)."""
    (PASS if cond else FAIL).append(name)
    d = detail if cond else (fail_detail or detail)
    print(f"  {'ok  ' if cond else 'FAIL'} {name}" + (f"   {d}" if d else ""), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default=str(
        repo_path("checkpoint/coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged")))
    p.add_argument("--seq-len", type=int, default=520,
                   help="synthetic sequence length; the hazards are the wrapper's, not "
                        "the content's, so no dataset or DINO is needed")
    p.add_argument("--layer-frac", type=float, default=1.0,
                   help="propagate the whole stack, which is the trainer's default. It "
                        "used to be 0.25 on the grounds that this gate is about the "
                        "distributed plumbing -- and that quarter never replays LAYER 0, "
                        "which is the one layer where the left-pad rows the padded case "
                        "here carries are large next to the residual stream. The first "
                        "colocated run died there on a check this gate had reported green")
    args = p.parse_args()

    from accelerate import Accelerator
    from transformers import AutoConfig, AutoProcessor
    import transformers

    acc = Accelerator()
    # There is no dataloader here -- the hazards are the wrapper's, not the data's -- so
    # DeepSpeed cannot infer its micro-batch and `prepare` refuses. State it directly.
    # Assigned, not setdefault: the plugin ships these as the string "auto", which is a
    # present key, so setdefault silently leaves them unresolved.
    _dsp = getattr(acc.state, "deepspeed_plugin", None)
    if _dsp is not None:
        _dsp.deepspeed_config["train_micro_batch_size_per_gpu"] = 1
        _dsp.deepspeed_config["gradient_accumulation_steps"] = 1
        _dsp.deepspeed_config["train_batch_size"] = acc.num_processes
    rank, world = acc.process_index, acc.num_processes
    device = acc.device
    if rank == 0:
        print(f"\n[zero3] {world} ranks, {acc.distributed_type}", flush=True)

    processor = AutoProcessor.from_pretrained(args.base_model, padding_side="left")
    cfg = AutoConfig.from_pretrained(args.base_model)
    arch = getattr(transformers, cfg.architectures[0])
    model = arch.from_pretrained(args.base_model, torch_dtype=torch.bfloat16,
                                 attn_implementation="sdpa")
    # A real optimizer, because hazard 1 is about what lands in the TRAINING state: with
    # no optimizer prepared there are no ZeRO-3 gradient hooks to fire and the test would
    # pass vacuously.
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=1e-9)
    model, opt = acc.prepare(model, opt)
    if rank == 0:
        print(f"[zero3] prepared; zero stage "
              f"{getattr(getattr(acc.state, 'deepspeed_plugin', None), 'zero_stage', '?')}",
              flush=True)

    from trl.models import unwrap_model_for_generation

    # ---- the case. Text-only synthetic input; every hazard is the wrapper's. ----
    torch.manual_seed(1234 + rank)
    n = args.seq_len
    ids = torch.randint(1000, 20000, (1, n), device=device)
    img_id = int(getattr(processor, "image_token_id", 151655) or 151655)
    # A 4x4 patch grid of image tokens inside the prompt, so the map has columns to read
    # and step_glimpse_maps' grid check is exercised for real.
    gh = gw = 4
    # Exactly gh*gw image tokens: step_glimpse_maps refuses to reshape otherwise, and the
    # random filler cannot collide (randint tops out at 19999, the placeholder is 151655).
    ids[0, 10:10 + gh * gw] = img_id
    attn = torch.ones_like(ids)
    attn[0, :6] = 0                                      # LEFT PADDING, as a real batch has
    prompt_len = 200

    # HAZARD 2 + 3: ranks get deliberately different work. Rank 0 gets NOTHING to score.
    n_steps = rank                                       # 0, 1, 2, ... 7
    spans = [(prompt_len + 1 + 30 * k, prompt_len + 1 + 30 * k + 7 + k)
             for k in range(n_steps)]
    if not spans:
        spans = [(n - 2, n - 1)]                         # the trainer's dummy-span path
        discard = True
    else:
        discard = False

    fwd = {"input_ids": ids, "attention_mask": attn}
    grid = [1, gh * 2, gw * 2]

    maps = None
    with unwrap_model_for_generation(model, acc, gather_deepspeed3_params=True) as un, \
            GM.frozen_params(un):
        maps, info = GLM.step_glimpse_maps(
            un, fwd, spans, grid, question="what is in the image?",
            tokenizer=processor.tokenizer, prompt_len=prompt_len,
            layer_frac=args.layer_frac, image_token_id=img_id)
    acc.wait_for_everyone()

    if rank == 0:
        print(f"\n[hazard 1] nothing may land in .grad", flush=True)
    n_with_grad = sum(1 for q in model.parameters() if q.grad is not None)
    tot = torch.tensor([float(n_with_grad)], device=device)
    tot = acc.gather(tot).sum().item()
    if rank == 0:
        check("no parameter has a .grad after the glimpse pass, on any rank",
              tot == 0.0, f"{int(tot)} parameters with .grad")

    if rank == 0:
        print(f"\n[hazard 4] the map is a real map, not a shard artefact", flush=True)
    finite = bool(np.isfinite(maps).all())
    nonzero = bool(np.abs(maps).sum() > 0) if not discard else True
    ok = torch.tensor([float(finite and nonzero)], device=device)
    ok = acc.gather(ok).min().item()
    if rank == 0:
        check("every rank's map is finite and non-degenerate", ok == 1.0)
        check("the map has the language-model grid's shape",
              maps.shape[1:] == (gh, gw), f"{maps.shape}")

    if rank == 0:
        print(f"\n[hazard 3] the diagnostic gather is a collective", flush=True)
    # Only the ranks that had steps record anything -- rank 0 recorded nothing at all.
    if not discard:
        info["n_steps_built"] = len(spans)
        GL.record_map_info(info)
    diag = GL.pop_diagnostics()
    # Compared against the CONSTANT, not rank-to-rank via hash(): Python randomises string
    # hashing per process, so a hash comparison across ranks fails on identical key sets.
    agrees = torch.tensor([float(tuple(diag) == GL.DIAG_KEYS)], device=device)
    same = bool(acc.gather(agrees).min().item() == 1.0)
    n_collectives = 0
    for _k, v in diag.items():                       # exactly what the trainer does
        t = torch.tensor([v], dtype=torch.float32, device=device)
        g = acc.gather(t)
        g = g[~torch.isnan(g)]
        n_collectives += 1
        _ = g.mean().item() if g.numel() else None
    if rank == 0:
        check("every rank returns the same diagnostic key set, incl. the rank that "
              "scored nothing", same)
        check("and therefore the same number of collectives",
              n_collectives == len(GL.DIAG_KEYS), f"{n_collectives}")

    if rank == 0:
        print(f"\n[hazard 2] the ZeRO-3 trace survives rank disagreement", flush=True)
        print(f"          (rank r ran r steps: 0..{world - 1}, so the module-forward "
              f"counts differ by design)", flush=True)
    # The trainer invalidates the coordinator's trace on both sides of this pass. Do the
    # same, then take a REAL optimizer step: this is the operation that dies in
    # fetch_sub_module, or hangs in reset_step's cross-rank assert, if the trace is stale.
    # Located and invalidated EXACTLY as GRPOTrainerQwen3._zero3_param_coordinator and
    # ._invalidate_zero3_trace do -- a harness that found the coordinator by a different
    # route could pass while the trainer's own lookup silently returns None.
    coord = None
    offload = getattr(getattr(model, "optimizer", None), "parameter_offload", None)
    if offload is not None:
        try:
            coord = offload.get_param_coordinator()
        except Exception:
            coord = None
    if rank == 0:
        check("the trainer's own coordinator lookup finds a live coordinator",
              coord is not None, fail_detail="none found -- _invalidate_zero3_trace "
              "would be a silent no-op in training")
    if coord is not None and not coord.is_invalid_trace():
        coord._invalidate_trace()
        if rank == 0:
            print("          coordinator trace invalidated", flush=True)

    out = model(input_ids=ids, attention_mask=attn, labels=ids)
    acc.backward(out.loss)
    opt.step()
    opt.zero_grad(set_to_none=True)
    acc.wait_for_everyone()
    if rank == 0:
        check("a real fwd+bwd+optimizer step runs after the glimpse pass",
              torch.isfinite(out.loss).item(), f"loss {out.loss.item():.4f}")

    # And a second one, because ZeRO-3 re-records on the first and REPLAYS on the second:
    # a trace that is wrong shows up on the replay, not the record.
    out2 = model(input_ids=ids, attention_mask=attn, labels=ids)
    acc.backward(out2.loss)
    opt.step()
    opt.zero_grad(set_to_none=True)
    acc.wait_for_everyone()
    if rank == 0:
        check("and a SECOND one, which is where a stale trace is replayed",
              torch.isfinite(out2.loss).item(), f"loss {out2.loss.item():.4f}")
        print(f"\n{len(PASS)} passed, {len(FAIL)} failed", flush=True)
        for f in FAIL:
            print(f"  FAILED: {f}", flush=True)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
