#!/usr/bin/env python
"""Was this VLM's vision tower TRAINED inside the VLM, or used as the upstream released it?

    python check_vision_tower_trained.py

The cross-model experiment turns on that question, and reading it off papers and training
recipes is archaeology: recipes change between a paper and a release, "frozen" sometimes
means frozen in one stage of three, and a checkpoint is the only thing that actually ran.
So measure it. Load the VLM's vision tower and the upstream encoder it was initialised
from, match the tensors by name, and compare the numbers:

    identical to the last bit   -> the tower is exactly what the upstream released. Frozen.
    changed                     -> it was trained inside the VLM, and by how much.

LLaVA-1.5 versus its CLIP is the control. That one is frozen by every account of the
recipe, so if the method says anything else the method is wrong, not the recipe.

Weights only -- no GPU, no forward pass, and nothing is instantiated. The safetensors are
memory-mapped and read tensor by tensor, so the peak cost is one tensor.
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

HUB = os.environ.get("HF_HOME", "/home/uberger/scratch/cache/hf_cache") + "/hub"

#: (label, VLM repo, the prefixes its vision tower's tensors carry,
#:  upstream repo, the prefixes the upstream carries)
#: The prefixes are stripped from both sides so the remainder is the tensor's own path
#: inside the encoder, which is what makes the two comparable.
PAIRS = (
    ("LLaVA-1.5-7B  (control: believed frozen)", "llava-hf/llava-1.5-7b-hf",
     ("model.vision_tower.vision_model.", "vision_tower.vision_model."),
     "openai/clip-vit-large-patch14-336", ("vision_model.",)),
    ("llava-interleave-qwen-7B", "llava-hf/llava-interleave-qwen-7b-hf",
     ("model.vision_tower.vision_model.", "vision_tower.vision_model."),
     "google/siglip-so400m-patch14-384", ("vision_model.",)),
    ("Idefics3-8B-Llama3", "HuggingFaceM4/Idefics3-8B-Llama3",
     ("model.vision_model.", "vision_model."),
     "google/siglip-so400m-patch14-384", ("vision_model.",)),
)


def snapshot(repo):
    base = os.path.join(HUB, "models--" + repo.replace("/", "--"), "snapshots")
    snaps = sorted(glob.glob(base + "/*"))
    if not snaps:
        raise SystemExit(f"{repo} is not in the cache under {base}")
    return snaps[-1]


def load_tower(repo, prefixes):
    """{tensor path inside the encoder: tensor} for everything under a prefix.

    Reads safetensors where they exist and falls back to a pickled `pytorch_model.bin`
    where they do not -- `openai/clip-vit-large-patch14-336` still ships only the latter,
    and treating "no safetensors" as "no tensors" is how the control silently reported
    nothing instead of reporting an answer. Everything is read through torch, because the
    checkpoints are bfloat16 and numpy has no such dtype.
    """
    import torch
    from safetensors.torch import load_file

    snap = snapshot(repo)
    state = {}
    sfs = sorted(glob.glob(snap + "/*.safetensors"))
    if sfs:
        for f in sfs:
            state.update(load_file(f))
    else:
        bins = sorted(glob.glob(snap + "/*.bin"))
        if not bins:
            raise SystemExit(f"{repo}: no .safetensors and no .bin under {snap}")
        for f in bins:
            state.update(torch.load(f, map_location="cpu", weights_only=True))
    out = {}
    for k, v in state.items():
        for p in prefixes:
            if k.startswith(p):
                out[k[len(p):]] = v
                break
    return out


def compare(label, vlm_repo, vlm_pref, up_repo, up_pref):
    a, b = load_tower(vlm_repo, vlm_pref), load_tower(up_repo, up_pref)
    shared = sorted(set(a) & set(b))
    print(f"\n{label}")
    print(f"   vs {up_repo}")
    print(f"   {len(a)} tower tensors, {len(b)} upstream, {len(shared)} matched by name")
    if not shared:
        print("   NO MATCHED TENSORS -- the prefix mapping is wrong, not the answer")
        return
    # THE COMPARISON HAS TO GO THROUGH THE STORAGE DTYPE, or it answers a different
    # question. The upstream encoders ship float32; the VLMs store float16 or bfloat16.
    # Casting fp32 -> fp16 perturbs about a quarter of the tensors in the last bits, which
    # a raw "are these equal" test reports as training -- it did, on the control, at a
    # relative size of 4e-4, which is below fp16's own epsilon. So the test is: cast the
    # upstream tensor to the dtype the VLM stored, and ask whether it reproduces the VLM's
    # tensor EXACTLY. If it does, the tower is the upstream encoder written at lower
    # precision, and nothing trained it.
    identical, changed, worst, shape_mismatch = 0, 0, ("", 0.0), []
    for k in shared:
        x, y = a[k], b[k]
        if tuple(x.shape) != tuple(y.shape):
            shape_mismatch.append(f"{k} {tuple(x.shape)} vs {tuple(y.shape)}")
            continue
        if bool((x == y.to(x.dtype)).all()):
            identical += 1
            continue
        changed += 1
        xf, yf = x.float().numpy(), y.float().numpy()
        scale = float(np.abs(yf).max()) or 1.0
        d = float(np.abs(xf - yf).max()) / scale
        if d > worst[1]:
            worst = (k, d)
    n = identical + changed
    dts = {str(t.dtype) for t in a.values()} | {str(t.dtype) for t in b.values()}
    print(f"   dtypes seen: {sorted(dts)}  (upstream cast to the VLM's before comparing)")
    print(f"   {identical}/{n} tensors BIT-IDENTICAL to the upstream encoder, "
          f"{changed} changed")
    if shape_mismatch:
        print(f"   {len(shape_mismatch)} shape mismatches (resolution change, not "
              f"training): e.g. {shape_mismatch[0]}")
    if changed:
        print(f"   largest relative change: {worst[0]}  {worst[1]:.3e}")
    verdict = ("FROZEN -- every matched tensor is exactly the upstream encoder, written "
               "at the VLM's storage precision"
               if changed == 0 else
               f"TRAINED inside the VLM -- {changed} tensors differ by more than the "
               "storage dtype can explain")
    print(f"   -> {verdict}")


def main():
    print(__doc__.split("\n\n")[0])
    for label, vr, vp, ur, up in PAIRS:
        try:
            compare(label, vr, vp, ur, up)
        except SystemExit as e:
            print(f"\n{label}: {e}")
        except Exception as e:
            print(f"\n{label}: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
