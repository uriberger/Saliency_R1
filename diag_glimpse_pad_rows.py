#!/usr/bin/env python
"""Where does the GLIMPSE replay disagree with the forward on a PADDED case -- and by how
much? The measurement behind `glimpse_maps.valid_row_mask`.

The first colocated GLIMPSE training run died on every rank in `_check_replay`, at
0.084-0.090 relative against a 0.05 tolerance, "the recorded kwargs or the causal mask are
wrong". They were not. The trainer's cases are left-padded, and a LEFT-PAD query row is
masked everywhere -- causality leaves it only pad columns and the padding takes those away:

  * the forward runs sdpa with the BOOL mask transformers builds, and for a fully-masked
    row torch >= 2.5 returns EXACTLY ZERO (pytorch#110213, fixed there; before it, NaN);
  * the replay runs eager with `causal_mask`, which restores the diagonal, so the row
    attends to itself and returns its own value vector.

Neither is "the layer's output", nothing downstream reads those rows (their dz/dh is zero),
and the check was comparing them. This harness measures the split -- real rows, pad rows,
both -- so the fix is a measurement rather than a raised threshold.

No GPU: it loads layer 0 alone out of the checkpoint (the embedding plus one decoder
layer, ~2 GB) and runs it twice on the same [1, N, d] input. ~2 minutes on a login node.
Layer 0 is the point: it is the layer `layer_frac=1.0` replays first, and its residual
stream is the smallest in the stack, so the same absolute junk is largest there in
relative terms. Deeper layers carry outliers that hide it -- which is exactly why the
ZeRO-3 gate, at `--layer-frac 0.25`, reported green on a padded case.

    python diag_glimpse_pad_rows.py

Measured at the fix's commit (N=1700, 120 pad rows, fp32):

    real rows only    4.063e-07      <- the kwargs and the mask are right
    pad rows only     0.3849
    ALL rows          0.1667         <- what the check saw, tolerance 0.05

    attention out on the pad rows: sdpa 0 exactly, replay 0.4481;
    on the real rows the two attention paths agree to 3e-07.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextDecoderLayer,
    Qwen3VLTextRotaryEmbedding,
)

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from trl.glimpse_maps import causal_mask, valid_row_mask                  # noqa: E402

PAD_ID = 151643
IMG_ID = 151655


def rel(got, ref, rows=None):
    """`_check_replay`'s own metric: max |diff| over max |ref|, on `rows`."""
    if rows is not None:
        got, ref = got[:, rows], ref[:, rows]
    return float((got - ref).abs().max()) / max(float(ref.abs().max()), 1e-6)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default=str(
        REPO / "checkpoint/coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged"))
    p.add_argument("--seq-len", type=int, default=1700)
    p.add_argument("--n-pad", type=int, default=120, help="left pad rows, as a real batch")
    p.add_argument("--n-image", type=int, default=400)
    p.add_argument("--layer", type=int, default=0)
    args = p.parse_args()

    torch.set_num_threads(8)
    n, dtype = args.seq_len, torch.float32
    cfg = AutoConfig.from_pretrained(args.base_model).text_config

    # The case: left-padded, with a block of image placeholders where a real prompt has one.
    torch.manual_seed(0)
    ids = torch.randint(1000, 100000, (1, n))
    ids[0, :args.n_pad] = PAD_ID
    ids[0, args.n_pad + 20:args.n_pad + 20 + args.n_image] = IMG_ID
    attn2d = torch.ones(1, n, dtype=torch.long)
    attn2d[0, :args.n_pad] = 0

    f = safe_open(f"{args.base_model}/model.safetensors", "pt")
    pre = f"model.language_model.layers.{args.layer}."
    cfg._attn_implementation = "sdpa"
    layer = Qwen3VLTextDecoderLayer(cfg, args.layer).to(dtype).eval()
    layer.load_state_dict({k: f.get_tensor(pre + k).to(dtype) for k in layer.state_dict()})
    h = f.get_tensor("model.language_model.embed_tokens.weight")[ids[0]].to(dtype)[None]
    pos = torch.arange(n)[None, None].expand(3, 1, n)
    pe = Qwen3VLTextRotaryEmbedding(cfg).to(dtype)(h, pos)

    # The mask the real forward builds: bool, causal AND padding-aware, so a left-pad row
    # is all-False -- and that is the row sdpa answers with zero.
    q, k = torch.arange(n)[:, None], torch.arange(n)[None, :]
    bool_mask = (k <= q) & attn2d[0].bool()[None, :]
    empty = int((~bool_mask.any(-1)).sum())
    print(f"layer {args.layer}, N={n}, {args.n_pad} pad rows -> {empty} fully-masked rows "
          f"in the forward's mask")

    with torch.no_grad():
        ref = layer(h, attention_mask=bool_mask[None, None], position_ids=pos[0],
                    position_embeddings=pe)
        ref = ref[0] if isinstance(ref, tuple) else ref
        cfg._attn_implementation = "eager"
        got = layer(h, attention_mask=causal_mask(n, dtype, h.device, attention_mask=attn2d),
                    position_ids=pos[0], position_embeddings=pe)
        got = got[0] if isinstance(got, tuple) else got

    valid = valid_row_mask(n, attn2d, h.device)
    print(f"  real rows only    {rel(got, ref, valid):.4g}     <- the kwargs and the mask")
    print(f"  pad rows only     {rel(got, ref, ~valid):.4g}")
    print(f"  ALL rows          {rel(got, ref):.4g}     <- what the check saw, tol 0.05")

    # And the mechanism itself, one level down: the ATTENTION module's own output on the
    # pad rows, which is what the two masks actually disagree about.
    with torch.no_grad():
        hn = layer.input_layernorm(h)
        cfg._attn_implementation = "sdpa"
        a_ref = layer.self_attn(hn, position_embeddings=pe,
                                attention_mask=bool_mask[None, None])[0]
        cfg._attn_implementation = "eager"
        a_got = layer.self_attn(hn, position_embeddings=pe,
                                attention_mask=causal_mask(n, dtype, h.device,
                                                           attention_mask=attn2d))[0]
    print(f"  attention out on the pad rows: sdpa max|.| "
          f"{float(a_ref[:, ~valid].abs().max()):.4g} (zero, by pytorch#110213), "
          f"replay {float(a_got[:, ~valid].abs().max()):.4g}; on the real rows they agree "
          f"to {float((a_got - a_ref)[:, valid].abs().max()):.3g}")


if __name__ == "__main__":
    main()
