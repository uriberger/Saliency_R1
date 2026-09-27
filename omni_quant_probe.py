#!/usr/bin/env python
"""Does shrinking the Omni's frozen weights change where it looks?

THE ARGUMENT THIS IS FOR. Training the overlap reward on
`Nemotron-3-Nano-Omni-30B-A3B` is expensive for one reason: 66 GB of bfloat16 weights do
not fit beside a generation server on an 80 GB card, so the trainer has to cut them into
pieces across cards and re-collect them 16 times a step. Under LoRA the base is FROZEN --
nothing is ever written back into it -- so it can be stored in fewer bits. At 8 bits it
is 33 GB and at 4 bits 17 GB, and either fits whole on one card, which would remove the
cutting-up and the traffic with it.

That is only worth building if the shrunken model still LOOKS in the same places, because
where it looks is what the reward scores. So: the same pictures, the same prompts, the
same scan, at bfloat16 and at each shrunken setting, compared per (layer, head).

WHAT IS COMPARED, AND WHY EACH ONE.

    maps            the per-layer attention map over the patch grid. Correlation is the
                    summary, but a correlation of 0.99 can still move an argmax, so:
    peak            the single most-attended patch per (layer, head). The reward divides
                    by the maximum, so if the peak moves off the corner the reward's
                    whole lever moves. This is the strictest thing here and the one to
                    read first.
    corner/ring     the statistics the cross-model table quotes. Their ABSOLUTE values
                    are what a future table would carry, so a small correlation change
                    that shifts these systematically still matters.

The vision tower is deliberately NOT shrunk. It is ~600M parameters, it is where the
border stamp comes from, and the geometry every patch statistic is defined on is its
output. Shrinking it would confound the measurement with the thing being measured.

    python omni_quant_probe.py --n 12 --modes bf16,int8,nf4
"""
import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import sink_location as SL
import sink_location_probe as SLP

MODEL = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"
CORPUS = Path("outputs/sink_location/xmodel/boxed/corpus")

#: the statistics the cross-model tables quote, plus the two the reward leans on
WATCH = ("corner_share", "corner_tl_share", "ring_share", "peak_in_ring",
         "first_patch_share", "peak_share", "image_mass", "entropy_norm")


def quant_config(mode):
    """`mode` -> a bitsandbytes config, or None for the full-size reference.

    `llm_int8_skip_modules` names what stays full-size. `vision_model` is the RADIO
    tower and `mlp1` is the projector that turns its output into the rows the decoder
    reads; together they are the picture's whole path into the model, and they are small.
    `lm_head` is skipped by the usual convention -- it is the widest single matrix and
    quantising it costs accuracy for no memory worth having.
    """
    if mode == "bf16":
        return None
    from transformers import BitsAndBytesConfig

    skip = ["vision_model", "mlp1", "lm_head"]
    if mode == "int8":
        return BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=skip)
    if mode == "nf4":
        return BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=skip)
    raise SystemExit(f"unknown mode {mode!r}: expected bf16, int8 or nf4")


def rows(n):
    got = [json.loads(l) for l in open(CORPUS / "manifest.jsonl")]
    by = {}
    for r in got:
        by.setdefault(r["type"], []).append(r)
    out = []
    while len(out) < n and any(by.values()):
        for k in sorted(by):
            if by[k] and len(out) < n:
                out.append(by[k].pop(0))
    return out[:n]


def run_mode(mode, picked, device="cuda:0"):
    """Load at `mode`, measure every picture, hand back the arrays and the cost."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    proc, model = SLP.load_model(MODEL, None, device, "sdpa", quant=quant_config(mode))
    load_s = time.time() - t0
    fam = SLP.load_family(model, proc, "none")
    scan = SL.install(model, family=fam)
    layers = fam.attention_layers(model)

    out, t1 = [], time.time()
    for r in picked:
        im = Image.open(CORPUS / r["image"]).convert("RGB")
        got = SLP.measure(model, proc, [im], r["question"], device, scan,
                          want_hidden=False)
        if got is None:
            raise SystemExit(f"{mode}: measure() returned None on {r['key']} -- the scan "
                             "did not fire, so there is nothing to compare")
        out.append({"key": r["key"], "grid": tuple(got["grid"]),
                    "stats": np.asarray(got["stats"], dtype=np.float64),
                    "peak": np.asarray(got["peak"]),
                    "maps": np.asarray(got["maps"], dtype=np.float64)})
    measure_s = (time.time() - t1) / max(1, len(picked))
    peak_gb = torch.cuda.max_memory_allocated(device) / 2**30

    scan.uninstall()
    del model, proc, scan
    gc.collect()
    torch.cuda.empty_cache()
    return out, {"load_s": load_s, "measure_s": measure_s, "peak_gb": peak_gb,
                 "layers": layers}


def compare(ref, got, mode):
    """Print the three readings, most decisive last."""
    print(f"\n{'='*78}\n{mode}  vs  bf16\n{'='*78}")

    # 1. the maps, per layer
    cors = []
    for a, b in zip(ref, got):
        if a["grid"] != b["grid"]:
            raise SystemExit(f"{a['key']}: grid moved {a['grid']} -> {b['grid']}")
        for la, lb in zip(a["maps"], b["maps"]):
            if la.std() > 0 and lb.std() > 0:
                cors.append(float(np.corrcoef(la, lb)[0, 1]))
    cors = np.asarray(cors)
    print(f"attention map correlation, per picture x layer  (n={cors.size})")
    print(f"    mean {cors.mean():.4f}   min {cors.min():.4f}   "
          f"below 0.99: {(cors < 0.99).mean()*100:.1f}%")

    # 2. the peak patch -- the reward's own lever
    same = np.concatenate([(a["peak"] == b["peak"]).reshape(-1)
                           for a, b in zip(ref, got)])
    print(f"\nmost-attended patch identical, per picture x layer x head  (n={same.size})")
    print(f"    {same.mean()*100:.1f}% agree")

    # ... and, when it moves, does it still land on the corner?
    tl_a = np.concatenate([(a["peak"] == 0).reshape(-1) for a in ref])
    tl_b = np.concatenate([(b["peak"] == 0).reshape(-1) for b in got])
    print(f"    peak sits on the top-left patch:  bf16 {tl_a.mean()*100:.1f}%   "
          f"{mode} {tl_b.mean()*100:.1f}%")

    # 3. the quoted statistics, absolute
    print(f"\n{'statistic':<20} {'bf16':>10} {mode:>10} {'diff':>10} {'rel':>8}")
    print("-" * 62)
    for name in WATCH:
        i = SL.STAT_INDEX[name]
        va = np.mean([a["stats"][..., i] for a in ref])
        vb = np.mean([b["stats"][..., i] for b in got])
        rel = (vb - va) / va if abs(va) > 1e-12 else float("nan")
        print(f"{name:<20} {va:>10.4f} {vb:>10.4f} {vb-va:>+10.4f} {rel*100:>+7.1f}%")
    return {"map_corr_mean": float(cors.mean()), "map_corr_min": float(cors.min()),
            "peak_agree": float(same.mean()), "peak_tl_ref": float(tl_a.mean()),
            "peak_tl_got": float(tl_b.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--modes", default="bf16,int8,nf4")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    modes = args.modes.split(",")
    if modes[0] != "bf16":
        raise SystemExit("bf16 must come first: it is the reference everything else is "
                         "compared against")
    picked = rows(args.n)
    print(f"{len(picked)} pictures, types: "
          f"{sorted({r['type'] for r in picked})}\nmodel: {MODEL}\n")

    results, cost, summary = {}, {}, {}
    for m in modes:
        print(f"[{m}] loading...", flush=True)
        results[m], cost[m] = run_mode(m, picked)
        c = cost[m]
        print(f"[{m}] peak GPU {c['peak_gb']:.1f} GB | load {c['load_s']:.0f}s | "
              f"{c['measure_s']:.1f}s per picture | attention layers {c['layers']}",
              flush=True)

    for m in modes[1:]:
        summary[m] = compare(results["bf16"], results[m], m)

    print(f"\n{'='*78}\nCOST\n{'='*78}")
    print(f"{'mode':<8} {'peak GPU':>10} {'fits 80GB':>11} {'s/picture':>11} "
          f"{'vs bf16':>9}")
    base = cost["bf16"]["measure_s"]
    for m in modes:
        c = cost[m]
        print(f"{m:<8} {c['peak_gb']:>9.1f}G {'yes' if c['peak_gb'] < 78 else 'NO':>11} "
              f"{c['measure_s']:>11.1f} {c['measure_s']/base:>8.2f}x")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"cost": {k: {kk: vv for kk, vv in v.items()} for k, v in cost.items()},
             "summary": summary, "n": len(picked)}, indent=2, default=str))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
