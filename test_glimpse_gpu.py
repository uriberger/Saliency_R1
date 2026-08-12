#!/usr/bin/env python
"""Gate the GLIMPSE memory rework on a GPU: same map, far smaller peak.

    srun --jobid=$JOBID --overlap -n1 bash -lc '... python test_glimpse_gpu.py'

Two parts, each of which can fail the run:

  equivalence  the grad-cache map against the all-eager baseline commit, on real samples.
               The two are the same quantity by the chain rule but not the same floating
               point path -- the baseline runs the whole stack in eager, this runs it in
               sdpa and replays one layer at a time -- so the gate is correlation, not
               equality. Anything below --min-corr is a real disagreement, not noise.

  scaling      peak allocated as a function of sequence length, against the baseline's
               measured curve. The point of the rework is that this stops growing with
               the layer count, so the baseline's OOM at N=3600 has to become a fit.

The baseline is materialised from git rather than kept as a second copy in the tree, so
it cannot drift: `git show <commit>:saliency_viz.py`. It is written next to this file
because saliency_viz.py resolves its sibling modules relative to its own __file__.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parent
BASELINE_NAME = "_glimpse_baseline.py"

sys.path.insert(0, str(REPO))
import saliency_viz as SV                                            # noqa: E402

PROBE, IV, GM = SV.PROBE, SV.IV, SV.GM

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


def load_baseline(commit: str):
    """-> the pre-rework module, or None if this commit has no glimpse_map."""
    path = REPO / BASELINE_NAME
    src = subprocess.run(["git", "-C", str(REPO), "show", f"{commit}:saliency_viz.py"],
                         capture_output=True, text=True)
    if src.returncode != 0:
        raise SystemExit(f"cannot read saliency_viz.py at {commit}: {src.stderr.strip()}")
    path.write_text(src.stdout)
    import importlib.util
    spec = importlib.util.spec_from_file_location("_glimpse_baseline", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_glimpse_baseline"] = mod
    spec.loader.exec_module(mod)
    return mod


def glimpse_args(a):
    return SimpleNamespace(glimpse_temp=a.glimpse_temp,
                           glimpse_depth_temp=a.glimpse_depth_temp,
                           glimpse_layer_frac=a.glimpse_layer_frac,
                           glimpse_target=a.glimpse_target,
                           glimpse_token_weight=a.glimpse_token_weight)


def synthetic_steps(n_chain: int, n_steps: int, per_step: int):
    """Spans over the chain, in the (text, a, b) shape glimpse_map expects.

    The step TEXT is never read by glimpse_map -- only the spans are -- so this test does
    not have to load the FLAN-T5 classifier to compare two implementations on identical
    input. What it does have to do is stay inside the chain.
    """
    steps, a = [], 1
    for i in range(n_steps):
        b = min(a + per_step, n_chain)
        if b <= a:
            break
        steps.append((f"step {i}", a, b))
        a = b
    return steps


def run_one(mod, model, processor, inputs, ids, prompt_len, steps, gh, gw, question,
            args, device):
    """-> (map, peak GiB) or (None, peak GiB) if it ran out of memory."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        out, _info = mod.glimpse_map(model, processor, inputs, ids, prompt_len, steps,
                                     gh, gw, question, glimpse_args(args), device)
    except torch.cuda.OutOfMemoryError:
        out = None
    peak = torch.cuda.max_memory_allocated(device) / 2 ** 30
    torch.cuda.empty_cache()
    return out, peak


def test_equivalence(args, device):
    print(f"\n[equivalence] grad cache vs the all-eager baseline at {args.baseline}")
    base = load_baseline(args.baseline)
    if not hasattr(base, "glimpse_map"):
        raise SystemExit(f"{args.baseline} has no glimpse_map to compare against")

    processor, model = PROBE.load_model(args.base_model, None, device, "sdpa")
    model.requires_grad_(False)
    rows = PROBE.load_samples(args.dataset, args.n_samples, args.seed,
                              cache_tag="_glimpsegpu", split="all")

    worst_corr, worst_dev = 1.0, 0.0
    for i, row in enumerate(rows[: args.n_samples]):
        inputs, prompt_len, comp_ids = IV.greedy_chain(
            processor, model, row["image"], row["question"], args.max_new_tokens, device)
        ids = torch.tensor([inputs["input_ids"][0].tolist() + list(comp_ids)],
                           device=device)
        gh = int(inputs["image_grid_thw"][0, 1].item()) // 2
        gw = int(inputs["image_grid_thw"][0, 2].item()) // 2
        steps = synthetic_steps(len(comp_ids), args.n_steps, args.step_tokens)
        if not steps:
            print(f"  sample {i}: chain too short ({len(comp_ids)} tokens), skipped")
            continue

        n = ids.shape[1]
        new, new_peak = run_one(SV, model, processor, inputs, ids, prompt_len, steps,
                                gh, gw, row["question"], args, device)
        old, old_peak = run_one(base, model, processor, inputs, ids, prompt_len, steps,
                                gh, gw, row["question"], args, device)
        if new is None:
            check(f"sample {i}: the grad cache fits", False, f"OOM at N={n}")
            continue
        if old is None:
            print(f"  sample {i}: N={n}, baseline OOM ({old_peak:.1f} GiB), "
                  f"grad cache {new_peak:.1f} GiB -- nothing to compare")
            continue

        corr_i, dev_i = 1.0, 0.0
        for si in range(len(steps)):
            a, b = new[si].ravel(), old[si].ravel()
            corr = float(np.corrcoef(a, b)[0, 1]) if a.std() and b.std() else 1.0
            dev = float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-12))
            corr_i, dev_i = min(corr_i, corr), max(dev_i, dev)
        worst_corr, worst_dev = min(worst_corr, corr_i), max(worst_dev, dev_i)
        print(f"  sample {i}: N={n}, {len(steps)} step(s), grid {gh}x{gw} | "
              f"peak {new_peak:.1f} vs {old_peak:.1f} GiB "
              f"({old_peak / max(new_peak, 1e-9):.1f}x) | corr {corr_i:.5f} "
              f"| max dev {dev_i:.3g}")
        check(f"sample {i}: the grad cache costs less than the all-eager graph",
              new_peak < old_peak, f"{new_peak:.1f} vs {old_peak:.1f} GiB")

    check("every step correlates with the baseline map",
          worst_corr >= args.min_corr, f"worst r = {worst_corr:.5f}")
    check("no step deviates beyond bf16 noise",
          worst_dev <= args.max_dev, f"worst relative deviation = {worst_dev:.3g}")
    return processor, model


def test_scaling(args, model, device):
    """Peak allocated vs sequence length, driving the cache directly on text-only input.

    The map needs an image; the memory does not care -- what scales is `L x H x N^2` in
    the graph, and this isolates it. The baseline column is the measured all-eager curve.
    """
    print("\n[scaling] peak allocated vs sequence length")
    baseline = {1200: 32.5, 1600: 42.3, 2000: 54.3, 2400: 68.6, 3600: float("inf")}
    n_layers = sum(1 for m in model.modules()
                   if type(m).__name__ == "Qwen3VLTextAttention")
    dtype = next(model.parameters()).dtype

    for n in args.scale:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        ids = torch.randint(1000, 20000, (1, n), device=device)
        cap = SV.GlimpseGradCache(model, 0, args.glimpse_temp)
        ok, peak = True, 0.0
        try:
            with SV.checkpointing_off(model), torch.enable_grad():
                res = model(input_ids=ids, use_cache=False,
                            logits_to_keep=torch.tensor([n - 2, n - 1], device=device))
                z = res.logits[0].float()
                del res
                cap.check()
                cap.mask = IV.causal_mask(n, dtype, device)
                g_out = cap.layer_grads(z[0, 100])
                for li in cap.layers:
                    e, _g1 = cap.edge(li, g_out.pop(li))
                    del e
                del g_out, z
            peak = torch.cuda.max_memory_allocated(device) / 2 ** 30
        except torch.cuda.OutOfMemoryError:
            ok = False
        finally:
            cap.close()
            torch.cuda.empty_cache()
        was = baseline.get(n)
        ref = "" if was is None else (" (baseline OOM'd here)" if was == float("inf")
                                      else f" vs {was:.1f} all-eager")
        if ok:
            print(f"  N={n:5d}  layers={n_layers}  peak {peak:6.2f} GiB{ref}")
        else:
            print(f"  N={n:5d}  OOM")
        check(f"N={n} fits", ok, "out of memory")
        if ok and was is not None and was != float("inf"):
            check(f"N={n} costs less than the all-eager graph", peak < was,
                  f"{peak:.1f} vs {was:.1f} GiB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default=str(SV.repo_path(
        "checkpoint/coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged")))
    p.add_argument("--dataset", default=str(SV.repo_path("cold_data/grpo_sets/val_natural")))
    p.add_argument("--baseline", default="feb1f4d",
                   help="the commit whose glimpse_map holds every layer in one graph")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n-samples", type=int, default=2)
    p.add_argument("--seed", type=int, default=1234)
    # Short on purpose: the baseline has to survive for there to be a comparison.
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--n-steps", type=int, default=2)
    p.add_argument("--step-tokens", type=int, default=4)
    p.add_argument("--scale", type=int, nargs="*", default=[1600, 2400, 3600, 4800])
    p.add_argument("--min-corr", type=float, default=0.99)
    p.add_argument("--max-dev", type=float, default=0.05)
    p.add_argument("--glimpse-temp", type=float, default=0.5)
    p.add_argument("--glimpse-depth-temp", type=float, default=0.2)
    p.add_argument("--glimpse-layer-frac", type=float, default=1.0)
    p.add_argument("--glimpse-target", default="logit")
    p.add_argument("--glimpse-token-weight", default="full")
    p.add_argument("--skip-scaling", action="store_true")
    args = p.parse_args()

    device = args.device
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        _processor, model = test_equivalence(args, device)
        if not args.skip_scaling:
            test_scaling(args, model, device)
    finally:
        (REPO / BASELINE_NAME).unlink(missing_ok=True)

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
        sys.exit(1)


if __name__ == "__main__":
    main()
