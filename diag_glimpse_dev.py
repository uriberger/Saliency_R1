#!/usr/bin/env python
"""Is a max-dev failure in the GLIMPSE equivalence gate precision, or a real bug?

This is the harness that calibrated `test_glimpse_gpu.py`'s thresholds. It is kept
because those thresholds are only defensible with a measurement behind them: the gate
originally demanded max dev <= 0.05 on a quantity whose own bf16 noise is 0.063-0.089,
which no correct implementation can pass. Rerun this before changing --min-corr,
--max-dev or --equiv-dtype, rather than tuning them to whatever a run happened to print.

    srun --jobid=$JOBID --overlap -n1 bash -lc '... python diag_glimpse_dev.py --fp32
        --dataset <...>/cold_data/grpo_sets/val_natural'

Three questions, in order of decisiveness:

  1. determinism   -- rerun each path twice. Establishes each one's own noise floor.
                      Bit-exact within a process; the bf16 gate reading still moved
                      0.0707 -> 0.02675 BETWEEN processes on one sample.
  2. distribution  -- is the deviation one outlier pixel, or the whole map? It is the
                      peak pixel, with 0 of 130-176 pixels over 0.05.
  3. fp32          -- run BOTH implementations at float32 on identical ids. If they
                      converge there, the bf16 gap is the floating-point path, and the
                      chain-rule algebra is identical. This is the one that decides it,
                      and it did: corr 1.000000, max dev 1.3e-06.

Measured on an H100-80GB at commit a8cd0e8. --fp32 needs ~37 GiB: it casts the 8B model
to float32, so run it on a card with room, and note that TF32 is left off on purpose --
it would put a ~bf16 mantissa back into every matmul and hide the very thing being
measured.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import test_glimpse_gpu as T                                          # noqa: E402
import saliency_viz as SV                                             # noqa: E402

PROBE, IV = SV.PROBE, SV.IV


def cmp(a_maps, b_maps, label):
    """corr / dev exactly as the gate computes them, per step."""
    worst_corr, worst_dev = 1.0, 0.0
    for a_m, b_m in zip(a_maps, b_maps):
        a, b = a_m.ravel(), b_m.ravel()
        corr = float(np.corrcoef(a, b)[0, 1]) if a.std() and b.std() else 1.0
        dev = float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-12))
        worst_corr, worst_dev = min(worst_corr, corr), max(worst_dev, dev)
    print(f"  {label:<34} corr {worst_corr:.6f}   max dev {worst_dev:.4g}")
    return worst_corr, worst_dev


def describe(a_maps, b_maps):
    """Where does the deviation live -- the peak, or the tail?"""
    for si, (a_m, b_m) in enumerate(zip(a_maps, b_maps)):
        a, b = a_m.ravel(), b_m.ravel()
        peak = max(np.abs(b).max(), 1e-12)
        d = np.abs(a - b) / peak
        j = int(d.argmax())
        over = int((d > 0.05).sum())
        print(f"    step {si}: {a.size} px | {over} px over 0.05 "
              f"({100.0 * over / a.size:.2f}%) | worst px {j}: "
              f"new {a[j]:+.5g} vs old {b[j]:+.5g}, |b|/peak = {abs(b[j]) / peak:.3f}")
        print(f"             deviation pctiles /peak: "
              f"p50 {np.percentile(d, 50):.2e}  p90 {np.percentile(d, 90):.2e}  "
              f"p99 {np.percentile(d, 99):.2e}  max {d.max():.2e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default=str(SV.repo_path(
        "checkpoint/coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged")))
    p.add_argument("--dataset", required=True)
    p.add_argument("--baseline", default="feb1f4d")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n-samples", type=int, default=2)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--n-steps", type=int, default=2)
    p.add_argument("--step-tokens", type=int, default=4)
    p.add_argument("--glimpse-temp", type=float, default=0.5)
    p.add_argument("--glimpse-depth-temp", type=float, default=0.2)
    p.add_argument("--glimpse-layer-frac", type=float, default=1.0)
    p.add_argument("--glimpse-target", default="logit")
    p.add_argument("--glimpse-token-weight", default="full")
    p.add_argument("--fp32", action="store_true", help="also run both paths at float32")
    a = p.parse_args()

    device = a.device
    # Explicit, not inherited from the torch default: TF32 carries a 10-bit mantissa,
    # about bf16, so leaving it on would make the fp32 arm measure the noise it exists
    # to rule out. test_glimpse_gpu.py sets it True for its scaling half.
    torch.backends.cuda.matmul.allow_tf32 = False

    base = T.load_baseline(a.baseline)
    processor, model = PROBE.load_model(a.base_model, None, device, "sdpa")
    model.requires_grad_(False)
    rows = PROBE.load_samples(a.dataset, a.n_samples, a.seed,
                              cache_tag="_glimpsegpu", split="all")

    for i, row in enumerate(rows[: a.n_samples]):
        inputs, prompt_len, comp_ids = IV.greedy_chain(
            processor, model, row["image"], row["question"], a.max_new_tokens, device)
        ids = torch.tensor([inputs["input_ids"][0].tolist() + list(comp_ids)],
                           device=device)
        gh = int(inputs["image_grid_thw"][0, 1].item()) // 2
        gw = int(inputs["image_grid_thw"][0, 2].item()) // 2
        steps = T.synthetic_steps(len(comp_ids), a.n_steps, a.step_tokens)
        if not steps:
            continue
        n = ids.shape[1]
        print(f"\n=== sample {i}: N={n}, grid {gh}x{gw}, {len(steps)} step(s) ===")

        def run(mod):
            out, _ = T.run_one(mod, model, processor, inputs, ids, prompt_len, steps,
                               gh, gw, row["question"], a, device)
            return out

        print("  [bf16]")
        new1, new2 = run(SV), run(SV)
        old1, old2 = run(base), run(base)
        cmp(new1, new2, "new vs new (determinism)")
        cmp(old1, old2, "old vs old (determinism)")
        cmp(new1, old1, "new vs old  <- the gate")
        describe(new1, old1)

        if a.fp32:
            print("  [fp32] casting the model -- both paths, identical ids")
            model.float()
            torch.cuda.empty_cache()
            new_f, old_f = run(SV), run(base)
            if new_f is None or old_f is None:
                print("    fp32 run OOM'd -- inconclusive")
            else:
                cmp(new_f, old_f, "new32 vs old32  <- DECIDES IT")
                describe(new_f, old_f)
                cmp(new1, new_f, "new16 vs new32 (new's own error)")
                cmp(old1, old_f, "old16 vs old32 (old's own error)")
            model.to(torch.bfloat16)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
