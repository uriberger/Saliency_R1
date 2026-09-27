#!/usr/bin/env python
"""Why does `measure()` come back None on the 12B? Print the scan's state, not a guess."""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sink_location as SL
import sink_location_probe as SLP
import vlm_family as VF

M = "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"
IMG = ("outputs/sink_location/xmodel/boxed/corpus/images/gqa-00001.png")

proc, model = SLP.load_model(M, None, "cuda", "sdpa")
fam = SLP.load_family(model, proc, "none")
scan = SL.install(model, family=fam)
print("family            :", fam.name, fam.fixed_grid, "img token", fam.image_token_id)
print("attn classes      :", fam.attn_classes)
print("attention_layers  :", fam.attention_layers(model))
print("q_sets            :", scan.q_sets)

im = Image.open(IMG).convert("RGB")
inputs = SLP.build_inputs(fam, proc, [im], "What is in this picture?", "cuda")
ids = inputs["input_ids"]
print("\ninput_ids         :", tuple(ids.shape),
      "image tokens:", int((ids == fam.image_token_id).sum()))
runs, grids = SL.locate_image_runs(ids, inputs, fam)
print("locate_image_runs : runs", [int(r.numel()) for r in runs], "grids", grids)

scan.reset()
with torch.no_grad():
    model(**inputs, output_hidden_states=False, use_cache=False)

print("\nAFTER THE FORWARD")
print("  scan.img_cols   :", None if scan.img_cols is None else int(scan.img_cols.numel()))
print("  scan.grids      :", getattr(scan, "grids", "UNSET"))
print("  scan.n_forwards :", scan.n_forwards)
for q in scan.q_sets:
    acc = scan._acc[q]
    print(f"  _acc[{q!r}] layers: {sorted(acc)}")
res = scan.result()
print("  result()        :", "None" if res is None else f"grids={res['grids']} "
      f"kv_len={res['kv_len']} n_image_tokens={res['n_image_tokens']}")

got = SLP.measure(model, proc, [im], "What is in this picture?", "cuda", scan,
                  want_hidden=False)
print("  measure()       :", "None" if got is None else f"grid={got['grid']}")
