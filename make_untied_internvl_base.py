#!/usr/bin/env python
"""Build an InternVL3.5 base dir whose config does NOT tie the output head.

    python make_untied_internvl_base.py            # build it, print what it did
    python make_untied_internvl_base.py --check     # verify only, exit non-zero if wrong

WHY THIS EXISTS, AND WHY IT IS NOT OPTIONAL

`OpenGVLab/InternVL3_5-8B-HF`'s `config.json` has **no `tie_word_embeddings` key**, so
transformers falls back to its default of `True` -- while the checkpoint ships BOTH
`language_model.lm_head.weight` and `language_model.model.embed_tokens.weight`, and they
differ (max|diff| = 0.9448). Plain `from_pretrained` notices the conflict and declines to
tie them, warning:

    The tied weights mapping and config for this model specifies to tie
    model.language_model.embed_tokens.weight to lm_head.weight, but both are present in
    the checkpoints with different values, so we will NOT tie them. You should update the
    config with `tie_word_embeddings=False` to silence this warning.

**DeepSpeed ZeRO-3 cannot make that comparison**, because its init never holds both full
tensors on one rank. It ties them, and the cold start then trains against a wrong output
head. Nothing errors. The symptoms, all observed on the 2026-11-07 run:

  * step-10 loss **13.898**, against 1.578 for GLM on the same corpus. `ln(151936)` is
    11.93, so the model began WORSE THAN UNIFORM -- that is the tell, and it is visible
    sixty seconds into the run;
  * loss nonetheless converges normally (0.873 by step 2,000) because the LoRA learns to
    compensate for the broken head, so a low FINAL loss proves nothing here;
  * at inference the head loads untied, the compensation is meaningless, and the merged
    model emits token salad. `coldstart_gate.py` scored format_ok 0.0%.

GLM-4.1V and Qwen3-VL are unaffected -- both set `tie_word_embeddings: false` explicitly.

The output is a directory of SYMLINKS to the HF cache plus one real, patched
`config.json`, so it costs no disk and never goes stale against the cached weights.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.realpath(__file__))
SRC_REPO = "OpenGVLab/InternVL3_5-8B-HF"
HUB_DIR = "models--OpenGVLab--InternVL3_5-8B-HF"
DEST = os.path.join(REPO, "checkpoint", "base_models", "InternVL3_5-8B-HF-untied")


def snapshot_dir() -> str:
    hf_home = os.environ.get("HF_HOME", "/home/uberger/scratch/cache/hf_cache")
    snaps = sorted(glob.glob(os.path.join(hf_home, "hub", HUB_DIR, "snapshots", "*")))
    if not snaps:
        sys.exit(
            f"no cached snapshot of {SRC_REPO} under {hf_home}/hub/{HUB_DIR}.\n"
            f"Stage it first, e.g.:\n"
            f"  HF_HUB_DISABLE_XET=1 python -c \""
            f"from huggingface_hub import snapshot_download; snapshot_download('{SRC_REPO}')\"")
    return snaps[-1]


def build(dest: str = DEST) -> str:
    snap = snapshot_dir()
    os.makedirs(dest, exist_ok=True)
    linked = 0
    for name in os.listdir(snap):
        dst = os.path.join(dest, name)
        if os.path.lexists(dst):
            os.remove(dst)
        if name == "config.json":
            continue  # the one file that is ours
        os.symlink(os.path.realpath(os.path.join(snap, name)), dst)
        linked += 1

    cfg = json.load(open(os.path.join(snap, "config.json")))
    # Both levels: the composite config is what `tie_weights()` consults, and the text
    # config is what the language model carries if it is rebuilt from its own sub-config.
    cfg["tie_word_embeddings"] = False
    cfg.setdefault("text_config", {})["tie_word_embeddings"] = False
    with open(os.path.join(dest, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)

    print(f"source snapshot : {snap}")
    print(f"built           : {dest}")
    print(f"  {linked} symlinks + 1 patched config.json")
    print(f"  tie_word_embeddings = False (top level and text_config)")
    return dest


def check(dest: str = DEST) -> int:
    if not os.path.isdir(dest):
        print(f"MISSING: {dest} -- run `python make_untied_internvl_base.py`")
        return 1
    cfg_path = os.path.join(dest, "config.json")
    if os.path.islink(cfg_path):
        print(f"BAD: {cfg_path} is a symlink; it must be the patched copy")
        return 1
    cfg = json.load(open(cfg_path))
    bad = []
    if cfg.get("tie_word_embeddings") is not False:
        bad.append(f"top-level tie_word_embeddings={cfg.get('tie_word_embeddings')!r}")
    if cfg.get("text_config", {}).get("tie_word_embeddings") is not False:
        bad.append("text_config.tie_word_embeddings is not False")
    weights = glob.glob(os.path.join(dest, "*.safetensors"))
    if not weights:
        bad.append("no *.safetensors present")
    if bad:
        print("FAIL: " + "; ".join(bad))
        return 1
    print(f"OK: {dest} -- tie_word_embeddings False, {len(weights)} weight file(s)")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify only; exit non-zero if wrong")
    ap.add_argument("--dest", default=DEST)
    args = ap.parse_args()
    sys.exit(check(args.dest) if args.check else (build(args.dest) and 0))
