"""Post-RoPE Q/K capture for **Qwen3-VL**, without mirroring the attention forward.

Drop-in replacement for LASER's `verl/workers/actor/attention_capture.py`, installed
into the fork by `patch_laser_qwen3.sh`. Same public API — `find_text_attention_modules`,
`AttentionSliceCapturer`, `compute_slice_for_sample` — so `dp_actor.py` needs no changes.

WHY THIS IS SHORTER THAN WHAT IT REPLACES

Upstream mirrors `Qwen2_5_VLAttention.forward` byte-for-byte (266 lines, and the reason
`transformers==4.57.6` is pinned) purely to get at post-RoPE `query_states` and
`key_states` on their way past. That mirror is unnecessary: **an attention interface
function is handed `query` and `key` already post-RoPE and, on Qwen3-VL, already
post-QK-norm.** Qwen3-VL's forward is

    query_states = self.q_norm(self.q_proj(h).view(shape)).transpose(1, 2)
    key_states   = self.k_norm(self.k_proj(h).view(shape)).transpose(1, 2)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    ...
    attention_interface(self, query_states, key_states, value_states, attention_mask, ...)

so intercepting `attention_interface` gets exactly the two tensors the mirror existed to
produce — with `q_norm`/`k_norm` applied (the easy thing to drop when re-deriving by hand)
and with interleaved MRoPE already baked into `cos`/`sin` upstream. Nothing about
`mrope_section`, the cache API or the output reshape has to be re-stated here, which is
also why this file does not have to move again the next time Qwen3-VL's forward does.

HOW IT INTERCEPTS, AND WHY NOT BY RENAMING THE IMPLEMENTATION

The obvious hook is to register a new attention implementation and point
`config._attn_implementation` at it. **Do not.** verl branches on that string in several
places — padding-free/varlen packing, the Ulysses sequence-parallel monkey patches, which
kwargs reach FlashAttention — so renaming it silently changes the forward's semantics
while looking like a no-op.

Instead this swaps the *function* the existing name maps to.
`ALL_ATTENTION_FUNCTIONS.__setitem__` writes to an instance-local override that
`__getitem__` consults before the global mapping, which is precisely the "local update of
the default functions without impacting other instances" the transformers docstring
describes. `config._attn_implementation` keeps whatever value it had — `flash_attention_2`
under verl — every downstream check sees the string it expects, and the real kernel still
runs. The capture is a side channel on the way past, and it is removed on `__exit__`.

The shim fires for **text** attention modules only, filtered by module identity: the
vision tower dispatches through the same registry entry and its attention is not what the
reward reads.
"""

from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

#: Qwen3-VL's text decoder attention, dense and MoE. The vision tower's
#: `Qwen3VLVisionAttention` is excluded by name AND by the `layer_idx` guard, exactly as
#: upstream excluded `Qwen2_5_VLVisionAttention`.
TEXT_ATTENTION_CLASSES = ("Qwen3VLTextAttention", "Qwen3VLMoeTextAttention")


def find_text_attention_modules(root: nn.Module) -> dict:
    """Return ``{layer_idx: attn_module}`` for every Qwen3-VL text attention module.

    Walks the tree the same way upstream did, so it keeps working under FSDP wrapping:
    FSDP wraps modules, it does not rename their classes.
    """
    found: dict[int, nn.Module] = {}
    for module in root.modules():
        if module.__class__.__name__ in TEXT_ATTENTION_CLASSES and hasattr(module, "layer_idx"):
            found[int(module.layer_idx)] = module
    return found


def _modeling_module(modules: dict):
    """The `modeling_qwen3_vl*` module object the attention classes were defined in.

    Taken from the class rather than imported by name, so the MoE variant resolves to its
    own module and a rename upstream surfaces as an AttributeError here instead of a
    silently un-hooked capture.
    """
    import sys

    any_module = next(iter(modules.values()))
    return sys.modules[type(any_module).__module__]


def _impl_name(modules: dict) -> str:
    """The attention implementation the text decoder is configured with.

    Read off the modules themselves rather than the top-level config, because that is what
    the forward reads and the two can differ — `sink_location.py` in the parent project
    exists partly because of a bug where they did.
    """
    names = {getattr(m.config, "_attn_implementation", None) for m in modules.values()}
    names.discard(None)
    if len(names) != 1:
        raise RuntimeError(
            f"text attention layers disagree about _attn_implementation ({sorted(names)}); "
            "refusing to guess which one the forward will dispatch to")
    return names.pop()


def _make_capturing_interface(inner, target_ids: set, captures: dict):
    """`inner`, plus a side channel that stashes post-RoPE Q/K for the target modules."""

    def laser_capture_attention_forward(module, query, key, value, attention_mask,
                                        *args, **kwargs):
        if id(module) in target_ids:
            # detach() so nothing retains the autograd graph. The calling context is
            # already no_grad; this is defensive, and it is what upstream did.
            captures[int(module.layer_idx)] = {
                "q": query.detach(),
                "k": key.detach(),
                "num_key_value_groups": int(getattr(module, "num_key_value_groups", 1)),
                # `scaling` arrives as a kwarg from Qwen3-VL's forward. Falling back to
                # the module attribute keeps this correct if a caller passes it
                # positionally, since the two are the same number.
                "scaling": float(kwargs.get("scaling") or getattr(module, "scaling", 1.0)),
            }
        return inner(module, query, key, value, attention_mask, *args, **kwargs)

    return laser_capture_attention_forward


class AttentionSliceCapturer:
    """Context manager that captures post-RoPE Q and K from the text decoder.

    Same contract as upstream's: enter, run one forward, call :meth:`get_captures`, exit.
    The attention kernel the model is configured with (FlashAttention / SDPA / eager) is
    unchanged — only the function object behind its registry entry is wrapped, and only
    for the duration of the ``with``.
    """

    def __init__(self, model: nn.Module, layer_indices: Optional[Iterable[int]] = None):
        self._all_modules = find_text_attention_modules(model)
        if not self._all_modules:
            raise RuntimeError(
                "No Qwen3-VL text attention modules found while installing "
                f"AttentionSliceCapturer (looked for {TEXT_ATTENTION_CLASSES}). If this is "
                "a Qwen2.5-VL model, use upstream's attention_capture.py instead.")
        if layer_indices is None:
            self._target_indices = sorted(self._all_modules.keys())
        else:
            self._target_indices = [int(i) for i in layer_indices if int(i) in self._all_modules]
        self._target_ids = {id(self._all_modules[i]) for i in self._target_indices}
        self._captures: dict[int, dict] = {}
        self._registry = None
        self._impl = None
        self._had_local = False
        self._prior = None
        self._eager_owner = None
        self._prior_eager = None

    @property
    def captured_layer_indices(self) -> list:
        return list(self._target_indices)

    def __enter__(self) -> "AttentionSliceCapturer":
        self._captures = {}
        mod = _modeling_module(self._all_modules)
        self._impl = _impl_name(self._all_modules)
        if self._impl == "eager":
            # The eager path never reaches the registry: the forward assigns
            # `attention_interface = eager_attention_forward` from the module globals and
            # only overwrites it when the impl is not "eager". So the symbol itself is
            # what has to be wrapped.
            self._eager_owner = mod
            self._prior_eager = mod.eager_attention_forward
            mod.eager_attention_forward = _make_capturing_interface(
                self._prior_eager, self._target_ids, self._captures)
            return self
        # Mutate the very instance the modeling module dispatches through, not whichever
        # one `transformers.modeling_utils` re-exports. They are normally the same object;
        # "normally" is not a property worth relying on for a side channel that fails
        # silently when it misses.
        self._registry = mod.ALL_ATTENTION_FUNCTIONS
        self._had_local = self._impl in getattr(self._registry, "_local_mapping", {})
        self._prior = self._registry[self._impl]
        self._registry[self._impl] = _make_capturing_interface(
            self._prior, self._target_ids, self._captures)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._eager_owner is not None:
            self._eager_owner.eager_attention_forward = self._prior_eager
            self._eager_owner = self._prior_eager = None
        elif self._registry is not None:
            if self._had_local:
                self._registry[self._impl] = self._prior      # restore the prior override
            else:
                del self._registry[self._impl]                # drop ours; global reappears
            self._registry = self._prior = None
        # The captures are dropped too: holding a forward's Q and K past the block that
        # asked for them is how a 36-layer capture turns into a memory leak.
        self._captures = {}

    def get_captures(self) -> dict:
        """``{layer_idx: {'q': (B, H_q, T, D), 'k': (B, H_kv, T, D), ...}}``.

        Only layers actually visited by the most recent forward appear, which is what
        `compute_slice_for_sample` iterates over.
        """
        return dict(self._captures)


# ---------------------------------------------------------------------------
# Everything below is upstream's, unchanged. `compute_slice_for_sample` works purely off
# `q`, `k`, `num_key_value_groups` and `scaling`, all of which Qwen3-VL exposes under the
# same names -- the payoff for how they factored it.
# ---------------------------------------------------------------------------
def _repeat_kv(k: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Equivalent to transformers' `repeat_kv`."""
    if n_rep == 1:
        return k
    bsz, h_kv, slen, d = k.shape
    return k[:, :, None, :, :].expand(bsz, h_kv, n_rep, slen, d).reshape(bsz, h_kv * n_rep, slen, d)


def _build_additive_mask(
    query_positions: torch.Tensor,  # (T_q,) long, abs positions in [0, T_k)
    key_valid_mask: torch.Tensor,   # (T_k,) bool, True where the key is real (not padding)
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Additive attention bias of shape ``(T_q, T_k)``: 0 where the key may be
    attended (causal AND not padding), ``finfo(dtype).min`` otherwise.

    This handles both left- and right-padding because the validity comes from
    the explicit ``key_valid_mask`` rather than a single right-edge cutoff.
    """
    T_k = int(key_valid_mask.shape[0])
    device = key_valid_mask.device
    key_positions = torch.arange(T_k, device=device)
    causal_ok = query_positions[:, None] >= key_positions[None, :]
    allowed = causal_ok & key_valid_mask[None, :]
    neg_inf = torch.finfo(dtype).min
    return torch.where(
        allowed,
        torch.zeros((), dtype=dtype, device=device),
        torch.full((), neg_inf, dtype=dtype, device=device),
    )


def compute_slice_for_sample(
    captures: dict,
    sample_idx: int,
    query_indices: torch.Tensor,    # (T_q,) long, abs positions
    visual_indices: torch.Tensor,   # (V,) long, abs positions
    key_valid_mask: torch.Tensor,   # (T_capture,) bool, True where the key token is real
    layer_indices: Optional[Iterable[int]] = None,
) -> torch.Tensor:
    """Compute layer- and head-averaged attention from ``query_indices`` to
    ``visual_indices`` for one sample, using captured post-RoPE Q/K.

    The math mirrors :func:`eager_attention_forward` from transformers:
    ``softmax((Q @ Kᵀ) * scaling + mask)`` in fp32, then slice and head-mean.

    Returns a ``(T_q, V)`` fp32 tensor. Peak memory is one layer's logits in
    fp32 — ``(1, H_q, T_q, T_capture)`` — plus the running ``(T_q, V)``
    accumulator.
    """
    assert query_indices.dim() == 1 and visual_indices.dim() == 1
    if layer_indices is None:
        layer_indices = sorted(captures.keys())
    else:
        layer_indices = [i for i in layer_indices if i in captures]
    T_q = int(query_indices.numel())
    V = int(visual_indices.numel())
    if not layer_indices or T_q == 0 or V == 0:
        device = next(iter(captures.values()))["q"].device if captures else key_valid_mask.device
        return torch.zeros((T_q, V), dtype=torch.float32, device=device)

    device = captures[layer_indices[0]]["q"].device
    T_capture = int(captures[layer_indices[0]]["q"].shape[-2])
    assert int(key_valid_mask.shape[0]) == T_capture, (
        f"key_valid_mask length {key_valid_mask.shape[0]} != capture length {T_capture}"
    )

    q_idx = query_indices.to(device=device, dtype=torch.long)
    v_idx = visual_indices.to(device=device, dtype=torch.long).clamp_(max=T_capture - 1)
    mask = key_valid_mask.to(device=device, dtype=torch.bool)

    # Built once for this sample; broadcasts over batch and head dims.
    bias = _build_additive_mask(q_idx, mask, dtype=torch.float32)  # (T_q, T_capture)

    accum: Optional[torch.Tensor] = None
    n_layers = 0
    for idx in layer_indices:
        qk = captures[idx]
        q = qk["q"][sample_idx : sample_idx + 1]  # (1, H_q, T_capture, D)
        k = qk["k"][sample_idx : sample_idx + 1]  # (1, H_kv, T_capture, D)
        n_rep = int(qk["num_key_value_groups"])
        scaling = float(qk["scaling"])

        q_sel = q.index_select(2, q_idx)  # (1, H_q, T_q, D)
        k_full = _repeat_kv(k, n_rep)  # (1, H_q, T_capture, D)

        # Eager-attention math in fp32 for softmax numerical stability.
        logits = torch.matmul(q_sel.float(), k_full.float().transpose(-1, -2)) * scaling
        logits = logits + bias  # broadcasts (T_q, T_capture) -> (1, H_q, T_q, T_capture)
        probs = F.softmax(logits, dim=-1)  # (1, H_q, T_q, T_capture)

        visual_probs = probs.index_select(-1, v_idx)  # (1, H_q, T_q, V)
        head_avg = visual_probs.mean(dim=1).squeeze(0)  # (T_q, V)

        accum = head_avg if accum is None else accum + head_avg
        n_layers += 1

        del logits, probs, visual_probs, q_sel, k_full

    return accum / max(n_layers, 1)
