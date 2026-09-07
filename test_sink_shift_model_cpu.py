#!/usr/bin/env python
"""Integration checks for sink_shift against a REAL Qwen3-VL module tree, on CPU.

    python test_sink_shift_model_cpu.py

`test_sink_shift_cpu.py` exercises the algebra and the attention function directly. That
leaves the riskiest part untested: `install()` has to find the right attention modules,
put the implementation name where transformers will look for it, leave the vision tower
alone, get `input_ids` and `image_grid_thw` out of a forward it does not control, and
survive `generate`'s KV cache. None of that can be checked without a model.

So this builds a randomly-initialised Qwen3-VL with tiny dimensions -- 3 text layers, 4
heads, hidden size 64 -- and runs the real forward and the real `generate` through it. It
needs no GPU, no checkpoint and no download; the weights are noise, which is fine, because
every assertion here is about wiring and invariants rather than about outputs being good.

Takes a couple of minutes, most of it importing transformers and initialising the
151,936-row embedding, which cannot shrink: the image placeholder is token 151655.
"""

from __future__ import annotations

import sys
import time

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import sink_shift as SS  # noqa: E402

FAILURES = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""),
          flush=True)
    if not ok:
        FAILURES.append(name)


def tiny_model():
    from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration
    from transformers.models.qwen3_vl.configuration_qwen3_vl import (
        Qwen3VLConfig, Qwen3VLTextConfig, Qwen3VLVisionConfig)

    vis = Qwen3VLVisionConfig(hidden_size=64, intermediate_size=128, num_heads=4, depth=2,
                              out_hidden_size=64, patch_size=16, spatial_merge_size=2,
                              temporal_patch_size=2, deepstack_visual_indexes=[0, 1])
    txt = Qwen3VLTextConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                            max_position_embeddings=4096)
    torch.manual_seed(0)
    return Qwen3VLForConditionalGeneration(Qwen3VLConfig(vision_config=vis,
                                                         text_config=txt)).eval()


def tiny_inputs(gh=6, gw=8, prefix=5, suffix=6):
    """A prompt with one picture in the middle, shaped the way the processor shapes one."""
    n_img = gh * gw
    ids = torch.full((1, prefix + n_img + suffix), 100, dtype=torch.long)
    ids[0, prefix:prefix + n_img] = SS.IMAGE_TOKEN_ID
    return dict(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        # M-RoPE refuses to run without this, so it is part of the contract, not a detail.
        mm_token_type_ids=(ids == SS.IMAGE_TOKEN_ID).long(),
        pixel_values=torch.randn(n_img * 4, 3 * 2 * 16 * 16),
        image_grid_thw=torch.tensor([[1, gh * 2, gw * 2]]),
    )


def main():
    t0 = time.time()
    print("sink_shift integration checks (tiny random Qwen3-VL, CPU)")
    torch.set_num_threads(4)
    model = tiny_model()
    kw = tiny_inputs()
    print(f"  built in {time.time() - t0:.0f}s, "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params", flush=True)

    with torch.no_grad():
        ref = model(**kw).logits

    ss = SS.install(model, arm="centre", alpha=0.0, layers=[1], heads=[2, 3])
    text_impl = model.config.text_config._attn_implementation
    vis_impl = model.config.vision_config._attn_implementation
    check("the text decoder is switched over", text_impl == SS.IMPL_NAME, text_impl)
    check("the vision tower is left alone", vis_impl != SS.IMPL_NAME, vis_impl)
    with torch.no_grad():
        zero = model(**kw).logits
    check("alpha=0 reproduces the un-hooked forward",
          torch.allclose(ref, zero, atol=1e-4),
          f"max |diff| {float((ref - zero).abs().max()):.2e}")
    check("the picture is located from the forward's own inputs",
          ss.img_cols is not None and ss.img_cols.numel() == 48
          and ss.grids == [(1, 6, 8)] and ss.first_row == 53,
          f"cols={None if ss.img_cols is None else ss.img_cols.numel()} "
          f"grids={ss.grids} first_row={ss.first_row}")
    # 6x8 is deliberately a grid where the centred rectangle ROUNDS ONTO the border
    # (rows 0-4 of 6), which the modal 10x16 grid does not. The destination must have
    # those border patches removed, or `centre` hands the sink's mass back to the sink.
    frame68, rect68 = SS.frame_set(6, 8), SS.rect_set(6, 8)
    check("the rectangle really does touch the border on this grid",
          int((frame68 & rect68).sum()) > 0, f"{int((frame68 & rect68).sum())} shared")
    check("the destination excludes the source",
          int(ss.src_cols.sum()) == int(frame68.sum())
          and int(ss.dst_cols.sum()) == int((rect68 & ~frame68).sum()),
          f"src={int(ss.src_cols.sum())} dst={int(ss.dst_cols.sum())} "
          f"(rect alone would be {int(rect68.sum())})")
    ss.uninstall()

    ss = SS.install(model, arm="centre", alpha=1.0, layers=[1], heads=[2, 3])
    with torch.no_grad():
        edited = model(**kw).logits
    d = ss.diagnostics()
    check("alpha=1 changes the forward",
          not torch.allclose(ref, edited, atol=1e-4),
          f"max |diff| {float((ref - edited).abs().max()):.2e}")
    check("alpha=1 empties the border", d["frame_share_after"] < 1e-6,
          f"{d['frame_share_before']:.5f} -> {d['frame_share_after']:.5f}")
    check("only the requested layer was touched", d["layers_touched"] == [1],
          str(d["layers_touched"]))
    ss.uninstall()

    with torch.no_grad():
        back = model(**kw).logits
    check("uninstalling restores the forward exactly", torch.equal(ref, back),
          f"max |diff| {float((ref - back).abs().max()):.2e}")
    check("uninstalling restores the implementation name",
          model.config.text_config._attn_implementation != SS.IMPL_NAME,
          model.config.text_config._attn_implementation)

    # generate() is the case the module exists for: a prefill followed by decode steps
    # that read a KV cache. A forward hook that re-runs the module cannot do this.
    with torch.no_grad():
        plain = model.generate(**kw, do_sample=False, max_new_tokens=6)
    ss = SS.install(model, arm="centre", alpha=1.0, layers=[1],
                    heads=[2, 3]).collect(1, [2, 3])
    with torch.no_grad():
        moved = model.generate(**kw, do_sample=False, max_new_tokens=6)
    d2, smap, n_maps = ss.diagnostics(), SS.collected_map(ss), len(ss._maps)
    ss.uninstall()
    check("generate() reaches the decode steps", d2["rows_edited"] >= 6,
          f"{d2['rows_edited']} rows over {d2['forwards']} forwards")
    check("the border is emptied during generation", d2["frame_share_after"] < 1e-6,
          f"{d2['frame_share_before']:.5f} -> {d2['frame_share_after']:.5f}")
    check("the edit reaches the tokens that are written",
          plain.tolist() != moved.tolist(),
          "identical tokens would mean the decode path was never edited")
    # N generated tokens give N-1 maps: the last token is never a query position, because
    # nothing follows it. Expecting N here would be expecting causal generation to run one
    # forward it has no reason to run.
    check("a map is collected for every generated token but the last",
          n_maps == 5, f"{n_maps} maps for 6 tokens")
    check("the collected map is on the patch grid",
          smap is not None and smap.shape == (6, 8),
          str(None if smap is None else smap.shape))

    ss = SS.install(model, arm="centre", alpha=0.5, layers=None, heads=None)
    with torch.no_grad():
        model(**kw)
    d3 = ss.diagnostics()
    ss.uninstall()
    check("scope=all reaches every text layer", d3["layers_touched"] == [0, 1, 2],
          str(d3["layers_touched"]))
    check("scope=all halves the border", abs(d3["frame_share_after"]
                                             - 0.5 * d3["frame_share_before"]) < 1e-6,
          f"{d3['frame_share_before']:.5f} -> {d3['frame_share_after']:.5f}")

    print(f"\n{'ALL PASS' if not FAILURES else f'{len(FAILURES)} FAILED: {FAILURES}'} "
          f"({time.time() - t0:.0f}s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
