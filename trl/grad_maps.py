# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Per-observe-step pixel-gradient maps -- the map the gradient reward scores.

One map per step, on the language model's image-token grid:

    F_S = mean_{n in S} [ z_{t_n, n} - (1/V) sum_v z_{v, n} ]      the centered logit
    G_j = || dF_S / d(pixels of image token j) ||_2

`S` is an observe step's token span, `t_n` the token the policy actually generated at
position `n`, `z_{v,n}` the raw logit of vocabulary item `v` there, and `V` the vocab
size. `trl/rewards/grad_rewards.py` turns `G` into a reward against a Grounding-DINO
box union; this module only produces `G`.

Three choices are baked in here, each for a reason that is not obvious:

**The centered logit, not the log-prob.** `d log P(t)/dz = onehot(t) - p`, which goes to
zero as the model becomes certain of `t` -- confidence lives in the *differences* between
logits, not their level, so a log-prob map shrinks everywhere for a step the model is sure
about and the reward ends up paying for uncertainty. The raw logit has no such factor but
carries a common-mode component (whatever the pixels do to the hidden state's overall
magnitude), which is shared by every item in the vocabulary and is therefore *the same map
for every step in the completion* -- shaping it once would lift every step's score at no
per-step cost. Subtracting the vocabulary mean removes that channel and keeps the
non-saturating property. `--grad_target logit|logprob` selects the other two for probes.

**Mean over the step's tokens, not sum.** The two differ by a factor `1/|S|`, which the
reward's roll-null ratio cancels exactly; the mean is used so that the *logged* raw norms
are comparable across steps of different lengths.

**Differentiate w.r.t. pixels, not image embeddings.** The vision tower stays inside the
graph, so Qwen3-VL's deepstack taps (vision layers 8/16/24 added into LM layers 0/1/2) are
counted automatically and there is no `_ds` variant to pick. It costs ~10%: the tower is
0.55 B params over ~1024 patches against 8 B over ~900 tokens.

Cost, per completion, at the 512px cap (256 image tokens, ~700-1000 sequence positions):
one forward plus one backward per chunk of `span_chunk` steps, via a vmapped VJP with a
one-hot cotangent per step (`is_grads_batched`). Batching every step into a single backward
is what the first GPU run tried, and it OOMed an 80 GB card: the vmap holds one set of
backward intermediates per cotangent at once, so peak memory grows with the completion's
step count while the forward graph below stays fixed. No parameter gradients are needed, so
`_frozen_params` clears `requires_grad` on every weight and autograd prunes every
weight-gradient matmul -- which both makes the backward cost about the same as a forward
rather than double it, and keeps DeepSpeed ZeRO-3's gradient hooks out of the pass
entirely (nothing accumulates into `.grad`). Measured budget in docs/: ~1.8 s per rank per
optimizer step against the 1.0 s attention capture it replaces.

The next lever, if this ever needs to be cheaper, is batching completions into one forward
(they are all generations of ONE prompt and share one image, so the tower could run once);
it is not done here because the retained activation graph is ~3.4 GB per sequence and the
trainer's loop is per-case for ZeRO-3 trace reasons.
"""

from __future__ import annotations

import contextlib

import numpy as np
import torch

GRAD_TARGETS = ("clogit", "logit", "logprob")

# How many steps share one vmapped backward. `is_grads_batched` runs one backward per
# cotangent *simultaneously*, so peak memory is the retained forward graph (~3.4 GB, fixed)
# plus n_spans x the backward's own intermediates -- the term that OOMed an 80 GB card at
# 76 GB on a completion with many observe steps, while shorter completions on the same card
# passed. Chunking bounds that term without changing any result: the steps' gradients are
# independent, so grouping them differently is exactly the same arithmetic. `None` or 0
# restores the all-at-once behaviour.
SPAN_CHUNK_DEFAULT = 4


def _at_least_fp32(t: torch.Tensor) -> torch.Tensor:
    """Promote bf16/fp16 to fp32 and leave fp32/fp64 alone.

    A plain `.float()` would silently DOWNCAST a float64 tensor, which is what the CPU
    test runs in -- and a test that has to loosen its tolerance to accommodate the code
    under test is not testing much.
    """
    return t if t.dtype in (torch.float32, torch.float64) else t.float()


def pixel_regroup(grad: torch.Tensor, grid_thw, ps: int, tps: int) -> torch.Tensor:
    """[n_patch, C*T*ps*ps] pixel gradient -> [gh, gw] per language-model token.

    The processor's flatten order is [gh, gw, merge, merge, C, T, ps, ps] (identically
    in the slow and fast Qwen2-VL image processors), so a language-model token owns
    `merge*merge` consecutive rows. The temporal axis holds `tps` copies of the SAME
    pixels, so the chain rule says the gradient w.r.t. a pixel is the SUM over that
    axis; summing before the norm rather than after is the difference between the
    gradient w.r.t. the image and the gradient w.r.t. the processor's buffer.

    `saliency_viz.py --stage selftest` gates this against the real processor with a
    synthetic image; it needs no GPU and no weights.
    """
    n, d = grad.shape
    gh2, gw2 = int(grid_thw[1]), int(grid_thw[2])          # pre-merge patch grid
    gh, gw = gh2 // 2, gw2 // 2
    c = d // (tps * ps * ps)
    if c * tps * ps * ps != d or gh * gw * 4 != n:
        raise RuntimeError(f"pixel_values {tuple(grad.shape)} does not match grid "
                           f"{gh2}x{gw2} with patch={ps} temporal={tps}")
    g = grad.reshape(n, c, tps, ps * ps).sum(dim=2)        # collapse the duplicate frames
    per_patch_sq = (_at_least_fp32(g) ** 2).sum(dim=(1, 2))   # [n_patch]
    # a token's pixels are the union of its 2x2 patches, so its gradient norm is the
    # norm of the concatenation: sqrt of the summed squares.
    return per_patch_sq.reshape(gh, gw, 4).sum(dim=2).clamp_min(0).sqrt()


@contextlib.contextmanager
def frozen_params(model):
    """Clear `requires_grad` on every parameter, then restore it exactly.

    Two things depend on this, not one:

      * autograd prunes the weight-gradient half of the backward, so the pass costs
        about one forward instead of two, and
      * no AccumulateGrad node exists for any weight, so DeepSpeed ZeRO-3's gradient
        hooks never fire and nothing lands in `.grad`. Running a backward through the
        wrapped policy without this would reduce-scatter gradients into the live
        training state.

    Restoring per parameter (rather than calling `requires_grad_(True)` on the model)
    matters under PEFT: only the adapter weights were trainable to begin with.
    """
    saved = [(p, p.requires_grad) for p in model.parameters()]
    try:
        for p, _ in saved:
            p.requires_grad_(False)
        yield
    finally:
        for p, flag in saved:
            p.requires_grad_(flag)


def _row_scalars(logits: torch.Tensor, target_ids: torch.Tensor, target: str) -> torch.Tensor:
    """[n_rows, V] logits + the token each row predicts -> [n_rows] per-token scalar."""
    z = _at_least_fp32(logits)
    picked = z.gather(-1, target_ids[:, None]).squeeze(-1)
    if target == "clogit":
        return picked - z.mean(dim=-1)
    if target == "logit":
        return picked
    if target == "logprob":
        return picked - z.logsumexp(dim=-1)
    raise ValueError(f"grad target {target!r} not in {GRAD_TARGETS}")


def step_grad_maps(
    model,
    forward_inputs: dict,
    spans: list[tuple[int, int]],
    grid_thw,
    ps: int,
    tps: int,
    *,
    target: str = "clogit",
    batched: bool = True,
    span_chunk: int | None = SPAN_CHUNK_DEFAULT,
) -> np.ndarray:
    """-> [n_steps, gh, gw] float32, the map `G` for each span.

    `forward_inputs` is one case (batch of 1): `input_ids` [1, L], `attention_mask`, and
    the image entries `pixel_values` / `image_grid_thw` (plus `mm_token_type_ids` when the
    processor emits it). `spans` are absolute `[a, b)` positions in that sequence: the
    step occupies positions `a .. b-1`, so the logit rows that *predict* those tokens are
    `a-1 .. b-2`.

    The caller is responsible for `frozen_params(model)` -- it wraps the whole per-case
    loop in the trainer, not one call.

    `span_chunk` caps how many steps share one vmapped backward (see SPAN_CHUNK_DEFAULT).
    It is a pure memory/speed dial: the result is identical for every value, because each
    step's gradient depends only on its own cotangent. It does NOT affect ZeRO-3's trace
    invariant -- the single forward per case is unchanged, and only the number of backward
    calls varies, which already varies across ranks with the per-case step count.
    """
    if not spans:
        return np.zeros((0, int(grid_thw[1]) // 2, int(grid_thw[2]) // 2), dtype=np.float32)
    if target not in GRAD_TARGETS:
        raise ValueError(f"grad target {target!r} not in {GRAD_TARGETS}")

    ids = forward_inputs["input_ids"]
    device = ids.device
    if any(a <= 0 or b <= a or b > ids.shape[1] for a, b in spans):
        raise ValueError(f"spans {spans} outside 1..{ids.shape[1]} or empty")

    pv = forward_inputs["pixel_values"]
    # fp32 leaf, cast into the model's dtype INSIDE the graph, so the gradient that
    # arrives back at the leaf is fp32 even though the tower runs in bf16.
    leaf = _at_least_fp32(pv.detach()).clone().requires_grad_(True)
    fwd = dict(forward_inputs)
    fwd["pixel_values"] = leaf.to(pv.dtype)

    # Only the rows the steps actually need reach `lm_head`. The full completion would be
    # ~700 rows x 151,936 vocab, and the fp32 upcast for the centering term would cost
    # ~0.4 GB for positions no step scores. `logits_to_keep` takes a tensor of absolute
    # positions (transformers >= 4.52); an int would keep a trailing slice instead.
    rows = torch.cat([torch.arange(a - 1, b - 1, device=device) for a, b in spans])
    targets = torch.cat([ids[0, a:b] for a, b in spans])

    with torch.enable_grad():
        out = model(**fwd, use_cache=False, logits_to_keep=rows)
        per_token = _row_scalars(out.logits[0], targets, target)
        del out

        # F_S = mean over the step's tokens. segment_sum via a [n_steps, n_rows] indicator
        # keeps this one graph node instead of a Python loop of slices.
        lens = torch.tensor([b - a for a, b in spans], device=device, dtype=torch.float32)
        owner = torch.repeat_interleave(
            torch.arange(len(spans), device=device), lens.long()
        )
        f = torch.zeros(len(spans), device=device, dtype=per_token.dtype)
        f = f.index_add(0, owner, per_token) / lens

        n = len(spans)
        chunk = n if not span_chunk else min(int(span_chunk), n)
        if batched and n > 1:
            grads = []                                       # [n_steps][n_patch, D]
            for i in range(0, n, chunk):
                j = min(i + chunk, n)
                # One-hot rows i..j-1 of the identity: this chunk's cotangents, still
                # indexing into the full f, so the graph is the one shared by every chunk.
                cot = torch.zeros(j - i, n, device=device, dtype=f.dtype)
                cot[torch.arange(j - i, device=device), torch.arange(i, j, device=device)] = 1
                (g,) = torch.autograd.grad(
                    f, leaf, grad_outputs=cot, is_grads_batched=True, retain_graph=(j < n)
                )
                grads.extend(list(g))
                del g, cot
        else:
            grads = []
            for k in range(n):
                (gk,) = torch.autograd.grad(f[k], leaf, retain_graph=(k < n - 1))
                grads.append(gk)

    maps = torch.stack([pixel_regroup(gk, grid_thw, ps, tps) for gk in grads])
    return maps.detach().float().cpu().numpy()
