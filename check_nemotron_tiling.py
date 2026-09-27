#!/usr/bin/env python
"""Does NVIDIA-Nemotron-Nano-12B-v2-VL tile, and if so how does its token grid differ
from Nemotron-3-Nano-Omni's?

`vlm_family.NemotronVL` derives one (1, H//32, W//32) grid from the processor's
`imgs_sizes`. That is right for the Omni. This asks whether it is right for the 12B, by
running BOTH processors over the same pictures from the boxed corpus and printing, per
picture, what each one actually emits.

CPU only: the image processors do not need the weights.
"""
import json
import os
import sys

from PIL import Image
from transformers import AutoImageProcessor

CORPUS = ("/lustre/fs1/portfolios/nvr/projects/nvr_israel_rlop/users/uberger/research/"
          "saliency_r1/outputs/sink_location/xmodel/boxed/corpus")

TWELVE = "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"
OMNI = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"


def sample(n):
    """n pictures spread across the corpus's sources, so aspect ratios vary."""
    rows = [json.loads(l) for l in open(f"{CORPUS}/manifest.jsonl")]
    by_src = {}
    for r in rows:
        key = r.get("source") or r.get("image", "").split("-")[0]
        by_src.setdefault(key, []).append(r)
    out = []
    while len(out) < n and any(by_src.values()):
        for k in sorted(by_src):
            if by_src[k] and len(out) < n:
                out.append(by_src[k].pop(0))
    return out


def image_path(row):
    for key in ("image", "image_path", "path", "file"):
        v = row.get(key)
        if v:
            p = v if os.path.isabs(v) else f"{CORPUS}/images/{os.path.basename(v)}"
            if os.path.exists(p):
                return p
    raise KeyError(f"no image path in {list(row)}")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    rows = sample(n)

    proc12 = AutoImageProcessor.from_pretrained(TWELVE, trust_remote_code=True)
    procom = AutoImageProcessor.from_pretrained(OMNI, trust_remote_code=True)
    tok12 = proc12.num_image_token      # tokens contributed by ONE tile

    print(f"12B  image_size={proc12.image_size} max_num_tiles={proc12.max_num_tiles} "
          f"use_thumbnail={proc12.use_thumbnail} tokens/tile={tok12}")
    print()
    hdr = (f"{'picture':<22} {'orig WxH':>11} | {'12B tiles':>9} {'+thumb':>6} "
           f"{'12B tokens':>10} {'implied grid':>13} | {'Omni HxW':>11} "
           f"{'Omni grid':>10} {'Omni tokens':>11}")
    print(hdr)
    print("-" * len(hdr))

    tot12 = totom = 0
    for r in rows:
        p = image_path(r)
        im = Image.open(p).convert("RGB")
        w, h = im.size

        o12 = proc12(images=[im], return_tensors="pt")
        npatch = int(o12["num_patches"][0]) if "num_patches" in o12 else \
            int(o12["pixel_values"].shape[0])
        thumb = proc12.use_thumbnail and npatch != 1
        ntok12 = npatch * tok12

        # what the tiles cover, before the thumbnail: blocks laid out cols x rows
        real = npatch - (1 if thumb else 0) if thumb else npatch
        grid12 = f"{real}x{tok12}"

        oom = procom(images=[im], return_tensors="pt")
        sizes = oom.get("imgs_sizes")
        th, tw = (int(x) for x in sizes[0])
        gh, gw = th // 32, tw // 32
        ntokom = gh * gw

        tot12 += ntok12
        totom += ntokom
        print(f"{os.path.basename(p):<22} {f'{w}x{h}':>11} | {real:>9} "
              f"{'yes' if thumb else 'no':>6} {ntok12:>10} {grid12:>13} | "
              f"{f'{th}x{tw}':>11} {f'{gh}x{gw}':>10} {ntokom:>11}")

    print()
    print(f"mean visual tokens   12B {tot12/len(rows):8.0f}   Omni {totom/len(rows):8.0f}")
    print()
    print("Does the 12B emit `imgs_sizes` (what vlm_family.grids_for reads)? ",
          "imgs_sizes" in proc12(images=[Image.open(image_path(rows[0])).convert("RGB")],
                                 return_tensors="pt"))
    print("Keys the 12B processor emits: ",
          list(proc12(images=[Image.open(image_path(rows[0])).convert("RGB")],
                      return_tensors="pt").keys()))
    print("Keys the Omni processor emits:",
          list(procom(images=[Image.open(image_path(rows[0])).convert("RGB")],
                      return_tensors="pt").keys()))


if __name__ == "__main__":
    main()
