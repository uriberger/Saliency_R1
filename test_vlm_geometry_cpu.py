#!/usr/bin/env python
"""The trainer's geometry seam: Qwen3-VL unchanged, and the Omni actually different.

`trl/grpo_trainer_qwen3.py` used to read `image_grid_thw` in thirty places. It now asks
`vlm_family` instead, and TWO claims hold the whole port up:

  1. **Qwen3-VL's answers are exactly what the file used to hardcode.** The trainer runs
     in flight on that model, `patch_trl_qwen3.sh` rewrites the shared clone under it, and
     a seam that quietly changed a micro-batch's pixel slice would move a live run's
     gradient without moving anything visible. So each of the replaced expressions is
     restated here and checked against the family's answer.

  2. **The Omni's are different in the specific way it needs.** Its processor emits one
     row per picture for everything, so a micro-batch is an ordinary slice; its grid comes
     from `imgs_sizes` over a 32px token, a different size per picture; and `image_flags`
     has to be synthesised because the forward requires a key the processor never emits.

Plus the trap that cost a run: the LoRA target scope. peft matches bare names by SUFFIX
anywhere in the model, and the Omni carries a 24-layer AUDIO tower using exactly
`q_proj`, `k_proj`, `v_proj`.

CPU only, no model, no weights: every one of these is a pure function of a batch dict.

    python test_vlm_geometry_cpu.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import vlm_family as VF  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{('   ' + detail) if detail else ''}")


def qwen_batch(grids, seq=40):
    """A Qwen3-VL prompt batch, shaped the way its processor shapes one.

    `pixel_values` is ONE flat tensor of every picture's patches stacked together, which
    is the reason `image_grid_thw` was load-bearing: a sample's slice of it is only
    findable through the grid.
    """
    thw = torch.tensor(grids, dtype=torch.long)
    total = int(thw.prod(dim=1).sum())
    return {
        "input_ids": torch.zeros(len(grids), seq, dtype=torch.long),
        "pixel_values": torch.arange(total * 3, dtype=torch.float32).reshape(total, 3),
        "image_grid_thw": thw,
        "mm_token_type_ids": torch.zeros(len(grids), seq, dtype=torch.long),
    }


def omni_batch(sizes, seq=40):
    """An Omni prompt batch: one ROW per picture in every multimodal field."""
    n = len(sizes)
    return {
        "input_ids": torch.zeros(n, seq, dtype=torch.long),
        "pixel_values": torch.arange(n * 3 * 4 * 4, dtype=torch.float32).reshape(n, 3, 4, 4),
        "imgs_sizes": torch.tensor(sizes, dtype=torch.long),
        "num_patches": torch.ones(n, dtype=torch.long),
        "num_tokens": torch.tensor([h // 32 * (w // 32) for h, w in sizes]),
    }


# ---------------------------------------------------------------------------
def test_qwen_is_unchanged():
    print("\nQwen3-VL: the family's answers ARE the expressions they replaced")
    fam = VF._REGISTRY["qwen3_vl"]()
    grids = [(1, 8, 12), (1, 4, 6), (1, 10, 10)]
    batch = qwen_batch(grids)
    thw = batch["image_grid_thw"]

    # was: lengths = batch["image_grid_thw"].prod(dim=1).tolist()   (split_pixel_values_by_grid)
    check("mm_lengths == image_grid_thw.prod(dim=1)",
          fam.mm_lengths(batch) == thw.prod(dim=1).tolist())

    # was, in _get_per_token_logps_and_entropies:
    #   start_pixel_idx = image_grid_thw[:start].prod(-1).sum()
    #   end_pixel_idx   = image_grid_thw[:start+bs].prod(-1).sum()
    for lo, hi in [(0, 1), (1, 3), (0, 3), (2, 3)]:
        want_a = int(thw[:lo].prod(-1).sum())
        want_b = int(thw[:hi].prod(-1).sum())
        mm = fam.mm_slice(batch, lo, hi)
        ok = (torch.equal(mm["pixel_values"], batch["pixel_values"][want_a:want_b])
              and torch.equal(mm["image_grid_thw"], thw[lo:hi])
              and torch.equal(mm["mm_token_type_ids"], batch["mm_token_type_ids"][lo:hi]))
        check(f"mm_slice({lo},{hi}) reproduces the old pixel slice", ok,
              f"rows {want_a}:{want_b}")

    # was: gh = thw[case_id, 1] // 2 ; gw = thw[case_id, 2] // 2   (2x2 spatial merge)
    ok = all(fam.token_grid(batch, i) == (g[1] // 2, g[2] // 2)
             for i, g in enumerate(grids))
    check("token_grid == (h//2, w//2)", ok)

    # was: kwargs = {"images": [[img] for img in images]}
    imgs = ["a", "b", "c"]
    check("batch_image_arg nests one list per sample",
          fam.batch_image_arg(imgs) == [["a"], ["b"], ["c"]])

    # was: the five explicit `if "<key>" in prompt_inputs` carries into the loss pass
    check("the carried keys are the same five",
          set(fam.mm_inputs) == {"pixel_values", "image_grid_thw", "pixel_attention_mask",
                                 "image_sizes", "mm_token_type_ids"}
          and fam.geometry_inputs == ())

    check("no extra forward kwargs are introduced", fam.forward_defaults == {})
    check("the LoRA targets are passed through untouched",
          fam.lora_target_modules(["q_proj", "k_proj", "v_proj"])
          == ["q_proj", "k_proj", "v_proj"])
    check("after_processor is a no-op", fam.after_processor(batch) is batch)
    check("the Saliency-R1 readout is still available", fam.supports_saliency_r1())


def test_omni_geometry():
    print("\nOmni: one row per picture, and the grid comes from imgs_sizes")
    fam = VF._REGISTRY["NemotronH_Nano_Omni_Reasoning_V3"]()
    # Two DIFFERENT native resolutions, which is the case a fixed grid gets wrong.
    sizes = [(416, 672), (352, 224), (512, 512)]
    batch = omni_batch(sizes)

    check("nothing is packed behind a grid", fam.packed_inputs == ())
    check("mm_lengths has nothing to say", fam.mm_lengths(batch) is None)

    for lo, hi in [(0, 1), (1, 3), (0, 3)]:
        mm = fam.mm_slice(batch, lo, hi)
        ok = torch.equal(mm["pixel_values"], batch["pixel_values"][lo:hi])
        check(f"mm_slice({lo},{hi}) is an ordinary row slice", ok)

    # The geometry key must be carried by the batch and NEVER reach forward(): the
    # wrapper's signature has no `imgs_sizes` and passing it raises TypeError.
    check("imgs_sizes is carried as geometry, not as a forward kwarg",
          "imgs_sizes" in fam.geometry_inputs and "imgs_sizes" not in fam.mm_inputs)
    check("no geometry key leaks into mm_slice",
          not (set(fam.mm_slice(batch, 0, 3)) & set(fam.geometry_inputs)))

    ok = all(fam.token_grid(batch, i) == (h // 32, w // 32)
             for i, (h, w) in enumerate(sizes))
    check("token_grid == (H//32, W//32) per picture", ok,
          str([fam.token_grid(batch, i) for i in range(3)]))
    check("the three pictures really do get three different grids",
          len({fam.token_grid(batch, i) for i in range(3)}) == 3)

    check("batch_image_arg is FLAT", fam.batch_image_arg(["a", "b"]) == ["a", "b"])
    check("use_cache is forced off on every forward",
          fam.forward_defaults.get("use_cache") is False)
    check("the Saliency-R1 readout is refused", not fam.supports_saliency_r1())

    # image_flags: the forward opens with `image_flags.squeeze(-1)`, so [N] would collapse
    # a single picture to a 0-d tensor. It has to be [N, 1].
    out = fam.after_processor(dict(batch))
    flags = out.get("image_flags")
    check("after_processor synthesises image_flags", flags is not None)
    check("image_flags is [N, 1], not [N]",
          flags is not None and tuple(flags.shape) == (3, 1), str(tuple(flags.shape)))
    one = fam.after_processor({"pixel_values": torch.zeros(1, 3, 4, 4)})["image_flags"]
    check("one picture still gives a 2-d flag", tuple(one.shape) == (1, 1))
    kept = fam.after_processor({"pixel_values": torch.zeros(2, 3, 4, 4),
                                "image_flags": torch.full((2, 1), 7)})
    check("an existing image_flags is left alone", int(kept["image_flags"][0, 0]) == 7)


def test_lora_scope_keeps_the_audio_tower_out():
    print("\nThe trap: peft matches bare names by suffix, anywhere in the model")
    import re

    fam = VF._REGISTRY["NemotronH_Nano_Omni_Reasoning_V3"]()
    pat = fam.lora_target_modules(["q_proj", "k_proj", "v_proj"])
    check("a single string, so peft reads it as a regex", isinstance(pat, str), pat)

    wanted = ["language_model.backbone.layers.5.mixer.q_proj",
              "language_model.backbone.layers.42.mixer.v_proj",
              "language_model.model.layers.19.self_attn.k_proj"]
    # The 24-layer audio encoder, which an image-only batch never runs, and the RADIO
    # tower, which uses a fused qkv but is worth pinning anyway.
    unwanted = ["sound_encoder.layers.3.q_proj",
                "sound_encoder.layers.23.self_attn.v_proj",
                "sound_projection.k_proj",
                "vision_model.radio_model.blocks.1.attn.qkv",
                "vision_model.radio_model.blocks.1.attn.q_proj"]
    check("every decoder projection matches",
          all(re.fullmatch(pat, n) for n in wanted))
    check("no audio or vision module matches",
          not any(re.fullmatch(pat, n) for n in unwanted),
          str([n for n in unwanted if re.fullmatch(pat, n)]))
    check("a narrower target list narrows the regex",
          re.fullmatch(fam.lora_target_modules(["k_proj"]),
                       "language_model.backbone.layers.5.mixer.k_proj") is not None
          and re.fullmatch(fam.lora_target_modules(["k_proj"]),
                           "language_model.backbone.layers.5.mixer.q_proj") is None)


def test_v2_does_not_inherit_the_omni_grid():
    print("\nThe 12B VL shares the decoder and NOT the geometry")
    fam = VF._REGISTRY["NemotronH_Nano_VL_V2"]()
    # It reports tile COUNTS; `imgs_sizes` is an Omni-only key, so inheriting the Omni's
    # token_grid would raise on the first picture.
    check("token_grid is the fixed grid", fam.token_grid({}, 0) == (16, 16))
    check("it carries no geometry key", fam.geometry_inputs == ())


def main():
    test_qwen_is_unchanged()
    test_omni_geometry()
    test_lora_scope_keeps_the_audio_tower_out()
    test_v2_does_not_inherit_the_omni_grid()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for n in FAIL:
            print(f"  FAILED: {n}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
