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

ONE MODE PER PROCESS, deliberately. bf16 peaks at 66.8 GB of an 80 GB card, so the next
load has to start from a genuinely empty GPU -- and it does not, because
`SinkScan.install` registers a closure over itself in transformers'
`ALL_ATTENTION_FUNCTIONS`, which is module-global and outlives any `del model`. The first
attempt died exactly there: "this process has 65.21 GiB in use" while loading the second
model. Chasing that reference is the wrong fix when process exit is free and total.

    python omni_quant_probe.py --mode bf16 --n 8 --save out/bf16.npz
    python omni_quant_probe.py --mode int8 --n 8 --save out/int8.npz
    python omni_quant_probe.py --compare out/bf16.npz,out/int8.npz,out/nf4.npz
"""
import argparse

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
    return out, {"load_s": load_s, "measure_s": measure_s, "peak_gb": peak_gb,
                 "layers": layers}


def save(path, per_picture, cost, mode):
    """One npz, one entry PER PICTURE per field.

    Not stacked. The Omni is native-resolution -- it resizes each picture to its own
    aspect-matched size, so the grids differ (14x19, 13x21, 18x15 ...) and `maps` has a
    different width per picture. Stacking raised, which is the right failure; storing
    them separately is the fix.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    blob = {"mode": np.asarray([mode]),
            "keys": np.asarray([p["key"] for p in per_picture]),
            "grid": np.asarray([p["grid"] for p in per_picture]),
            "cost": np.asarray([json.dumps(cost, default=str)])}
    for i, p in enumerate(per_picture):
        for f in ("stats", "peak", "maps"):
            blob[f"{f}_{i}"] = p[f]
    np.savez_compressed(path, **blob)
    print(f"wrote {path}")


def load_saved(path):
    z = np.load(path, allow_pickle=False)
    cost = json.loads(str(z["cost"][0]))
    per = [{"key": str(k), "grid": tuple(g),
            "stats": z[f"stats_{i}"], "peak": z[f"peak_{i}"], "maps": z[f"maps_{i}"]}
           for i, (k, g) in enumerate(zip(z["keys"], z["grid"]))]
    return str(z["mode"][0]), per, cost


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
    ap.add_argument("--mode", default="", help="bf16 | int8 | nf4 -- measure and save")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--save", default="")
    ap.add_argument("--compare", default="",
                    help="comma-separated npz files; the FIRST is the reference")
    args = ap.parse_args()

    if args.mode:
        picked = rows(args.n)
        print(f"{len(picked)} pictures, types: {sorted({r['type'] for r in picked})}")
        print(f"model: {MODEL}\n[{args.mode}] loading...", flush=True)
        per, cost = run_mode(args.mode, picked)
        print(f"[{args.mode}] peak GPU {cost['peak_gb']:.1f} GB | "
              f"load {cost['load_s']:.0f}s | {cost['measure_s']:.1f}s per picture | "
              f"attention layers {cost['layers']}", flush=True)
        if args.save:
            save(args.save, per, cost, args.mode)
        return 0

    if not args.compare:
        raise SystemExit("pass --mode to measure, or --compare to read the saved runs")
    paths = args.compare.split(",")
    loaded = [load_saved(p) for p in paths]
    ref_mode, ref, ref_cost = loaded[0]
    if ref_mode != "bf16":
        raise SystemExit(f"the first file is {ref_mode!r}; bf16 is the reference "
                         "everything else is compared against")
    for mode, per, _c in loaded[1:]:
        if [p["key"] for p in per] != [p["key"] for p in ref]:
            raise SystemExit(f"{mode} measured different pictures from the reference")
        compare(ref, per, mode)

    print(f"\n{'='*78}\nCOST\n{'='*78}")
    print(f"{'mode':<8} {'peak GPU':>10} {'fits 80GB':>11} {'load s':>8} "
          f"{'s/picture':>11} {'vs bf16':>9}")
    base = ref_cost["measure_s"]
    for mode, _per, c in loaded:
        print(f"{mode:<8} {c['peak_gb']:>9.1f}G "
              f"{'yes' if c['peak_gb'] < 78 else 'NO':>11} {c['load_s']:>8.0f} "
              f"{c['measure_s']:>11.1f} {c['measure_s']/base:>8.2f}x")
    return 0


if __name__ == "__main__":
    main()
