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


def tensor_index(repo, prefixes):
    """{tensor path inside the encoder: (file, key)} for every tensor under a prefix."""
    snap = snapshot(repo)
    idx = os.path.join(snap, "model.safetensors.index.json")
    if os.path.exists(idx):
        weight_map = json.load(open(idx))["weight_map"]
    else:
        files = [os.path.basename(f) for f in glob.glob(snap + "/*.safetensors")]
        from safetensors import safe_open
        weight_map = {}
        for f in files:
            with safe_open(os.path.join(snap, f), framework="np") as h:
                for k in h.keys():
                    weight_map[k] = f
    out = {}
    for k, f in weight_map.items():
        for p in prefixes:
            if k.startswith(p):
                out[k[len(p):]] = (os.path.join(snap, f), k)
                break
    return out


def compare(label, vlm_repo, vlm_pref, up_repo, up_pref, sample=None):
    from safetensors import safe_open

    a, b = tensor_index(vlm_repo, vlm_pref), tensor_index(up_repo, up_pref)
    shared = sorted(set(a) & set(b))
    print(f"\n{label}")
    print(f"   vs {up_repo}")
    print(f"   {len(a)} tower tensors, {len(b)} upstream, {len(shared)} matched by name")
    if not shared:
        print("   NO MATCHED TENSORS -- the prefix mapping is wrong, not the answer")
        return
    keys = shared if sample is None else shared[:: max(1, len(shared) // sample)]
    identical, changed, worst, shape_mismatch = 0, 0, ("", 0.0), []
    handles = {}

    def get(path, key):
        if path not in handles:
            handles[path] = safe_open(path, framework="np")
        return handles[path].get_tensor(key)

    for k in keys:
        x, y = get(*a[k]), get(*b[k])
        if x.shape != y.shape:
            shape_mismatch.append(f"{k} {x.shape} vs {y.shape}")
            continue
        xf, yf = x.astype(np.float32), y.astype(np.float32)
        d = float(np.abs(xf - yf).max())
        scale = float(np.abs(yf).max()) or 1.0
        if d == 0.0:
            identical += 1
        else:
            changed += 1
            if d / scale > worst[1]:
                worst = (k, d / scale)
    n = identical + changed
    print(f"   {identical}/{n} tensors BIT-IDENTICAL to the upstream encoder, "
          f"{changed} changed")
    if shape_mismatch:
        print(f"   {len(shape_mismatch)} shape mismatches (resolution change, not "
              f"training): e.g. {shape_mismatch[0]}")
    if changed:
        print(f"   largest relative change: {worst[0]}  {worst[1]:.3e}")
    verdict = ("FROZEN -- every matched tensor is exactly what the upstream released"
               if changed == 0 else
               "TRAINED inside the VLM -- the weights moved")
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
