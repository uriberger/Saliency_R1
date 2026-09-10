#!/usr/bin/env python
"""CPU checks for the Qwen3-VL attention capture in LASER's fork. No GPU, no checkpoint.

    python test_laser_capture_cpu.py

Builds a TINY randomly-initialised Qwen3-VL text stack (3 layers, 4 heads, 2 KV heads --
so grouped-query attention is genuinely exercised) and asks the only question that
matters: does `compute_slice_for_sample`, fed by the captured post-RoPE Q/K, reproduce
the attention transformers itself reports under `output_attentions=True`?

That is the whole risk in this swap. Upstream got post-RoPE Q and K by mirroring
`Qwen2_5_VLAttention.forward` byte-for-byte; the replacement gets them by wrapping the
attention interface, on the theory that the interface already receives them post-RoPE and
post-QK-norm. If that theory is wrong -- if, say, `q_norm`/`k_norm` were applied after
the interface call, or MRoPE were not yet baked into cos/sin -- the reward would be
computed from plausible-looking wrong numbers and no training curve would show it.

  api         `dp_actor.py` imports exactly three names; they exist with the right shapes.
  discovery   text attention modules are found by layer index, and the VISION tower is
              not. Upstream relied on a class-name mismatch for that; Qwen3-VL's vision
              attention dispatches through the SAME registry entry, so the replacement
              filters by module identity instead and this is where that is checked.
  capture     Q and K come back post-RoPE and post-QK-norm, at the right shapes, for
              every layer, once per forward.
  slice       THE CHECK. `compute_slice_for_sample` against `output_attentions=True`.
  restore     the registry override is removed on exit, including when the body raises,
              and a prior local override is put back rather than deleted.
  semantics   `config._attn_implementation` is UNCHANGED by the capture -- the property
              verl's padding-free and Ulysses-SP branches depend on.

The test runs on whatever transformers is installed. The fork targets 4.57.6, where the
forward dispatches with `ALL_ATTENTION_FUNCTIONS[impl]`; 5.x uses
`ALL_ATTENTION_FUNCTIONS.get_interface(impl, eager_attention_forward)`, which ends in
`super().get(...)` and therefore consults the same instance-local override. Both paths
are covered by the same code, which is why this test is meaningful on either.
"""

from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "laser"))

import attention_capture as AC  # noqa: E402

FAILURES = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def tiny_text_model(impl="sdpa"):
    """A 3-layer Qwen3-VL text stack on CPU. Random weights: only the algebra is on test."""
    from transformers import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

    cfg = Qwen3VLTextConfig(
        hidden_size=64, num_hidden_layers=3, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, intermediate_size=128,
        vocab_size=256, max_position_embeddings=256,
    )
    cfg._attn_implementation = impl
    torch.manual_seed(0)
    model = Qwen3VLTextModel(cfg).eval()
    for m in model.modules():
        m.config = getattr(m, "config", cfg)
    return model, cfg


def a_forward(model, ids, **kw):
    with torch.no_grad():
        return model(input_ids=ids, attention_mask=torch.ones_like(ids), **kw)


def test_api():
    print("\napi -- the three names dp_actor.py imports")
    for n in ("find_text_attention_modules", "AttentionSliceCapturer",
              "compute_slice_for_sample"):
        check(f"{n} is exported", hasattr(AC, n))
    # Behavioural, not textual: the file NAMES Qwen2_5_VLAttention in its docstring,
    # because saying what it replaces is the point of the docstring. What must be true is
    # that nothing dispatches on it.
    check("discovery targets Qwen3-VL classes only",
          all(n.startswith("Qwen3VL") for n in AC.TEXT_ATTENTION_CLASSES)
          and "Qwen3VLTextAttention" in AC.TEXT_ATTENTION_CLASSES,
          f"{AC.TEXT_ATTENTION_CLASSES}")

    class Qwen2_5_VLAttention(torch.nn.Module):      # noqa: N801 -- the name IS the test
        pass

    stale = Qwen2_5_VLAttention()
    stale.layer_idx = 0
    holder = torch.nn.Module()
    holder.add_module("attn", stale)
    check("a Qwen2.5 attention module is no longer discovered",
          AC.find_text_attention_modules(holder) == {})


def test_discovery():
    print("\ndiscovery -- text attention only, keyed by layer index")
    model, cfg = tiny_text_model()
    found = AC.find_text_attention_modules(model)
    check("every text layer is found, keyed by layer_idx",
          sorted(found) == list(range(cfg.num_hidden_layers)), f"{sorted(found)}")
    check("the modules really are the text attention class",
          all(type(m).__name__ == "Qwen3VLTextAttention" for m in found.values()))

    # The vision tower dispatches through the same registry entry, so filtering by
    # registry alone would capture it too and `layer_idx` collisions would overwrite real
    # text layers. Upstream got this for free from a class-name mismatch; here it is the
    # id() filter's job, so check the tower is excluded from discovery in the first place.
    class FakeVisionAttention(torch.nn.Module):
        pass

    v = FakeVisionAttention()
    v.layer_idx = 0
    model.add_module("visual_attn", v)
    check("a non-text attention module is not discovered",
          sorted(AC.find_text_attention_modules(model)) == list(range(cfg.num_hidden_layers)))


def test_capture():
    print("\ncapture -- post-RoPE Q/K, right shapes, once per forward")
    model, cfg = tiny_text_model()
    ids = torch.randint(0, cfg.vocab_size, (2, 17))
    with AC.AttentionSliceCapturer(model) as cap:
        a_forward(model, ids)
        got = cap.get_captures()
    check("every layer captured", sorted(got) == list(range(cfg.num_hidden_layers)),
          f"{sorted(got)}")
    q, k = got[0]["q"], got[0]["k"]
    check("q is (B, H_q, T, D)", tuple(q.shape) == (2, 4, 17, 16), f"{tuple(q.shape)}")
    check("k is (B, H_kv, T, D) -- GQA is real here",
          tuple(k.shape) == (2, 2, 17, 16), f"{tuple(k.shape)}")
    check("the GQA group size and scaling came along",
          got[0]["num_key_value_groups"] == 2
          and abs(got[0]["scaling"] - cfg.head_dim ** -0.5) < 1e-9,
          f"n_rep={got[0]['num_key_value_groups']} scaling={got[0]['scaling']:.4f}")
    check("captures do not survive the context manager",
          cap.get_captures() == {})

    # Post-QK-norm is the easy thing to lose when re-deriving by hand, and a wrong answer
    # here looks entirely plausible. RMSNorm over the head dim pins ||q|| per head to
    # sqrt(D) times the learned gain, so an unnormed capture is detectable without
    # recomputing the forward.
    m0 = AC.find_text_attention_modules(model)[0]
    with torch.no_grad():
        h = torch.randn(1, 5, cfg.hidden_size)
        want = m0.q_norm(m0.q_proj(h).view(1, 5, -1, cfg.head_dim)).transpose(1, 2)
        raw = m0.q_proj(h).view(1, 5, -1, cfg.head_dim).transpose(1, 2)
    check("q_norm materially changes q, so capturing it matters",
          float((want - raw).abs().max()) > 1e-3,
          f"max|q_norm(q) - q| = {float((want - raw).abs().max()):.3f}")


def test_slice():
    """THE CHECK: the captured Q/K reproduce transformers' own attention."""
    print("\nslice -- compute_slice_for_sample vs output_attentions=True")
    model, cfg = tiny_text_model(impl="sdpa")
    B, T = 2, 19
    ids = torch.randint(0, cfg.vocab_size, (B, T))

    with AC.AttentionSliceCapturer(model) as cap:
        a_forward(model, ids)
        captures = cap.get_captures()

    # The reference runs eager so transformers will hand back its own softmax weights.
    ref_model, _ = tiny_text_model(impl="eager")
    ref_model.load_state_dict(model.state_dict())
    out = a_forward(ref_model, ids, output_attentions=True)
    att = torch.stack(out.attentions, dim=0)              # [L, B, H, T, T]

    q_idx = torch.arange(4, T)
    v_idx = torch.tensor([1, 3, 5, 8])                    # stands in for the visual span
    valid = torch.ones(T, dtype=torch.bool)
    worst = 0.0
    for s in range(B):
        got = AC.compute_slice_for_sample(captures, s, q_idx, v_idx, valid)
        want = att[:, s][:, :, q_idx][..., v_idx].mean(dim=1).mean(dim=0)
        worst = max(worst, float((got - want).abs().max()))
    check("the captured slice reproduces transformers' attention", worst < 1e-5,
          f"max |delta| {worst:.2e} over {B} samples x {len(q_idx)} queries")

    # Padding is the other thing the slice has to get right, and verl left-pads prompts.
    valid_lp = valid.clone()
    valid_lp[:3] = False
    got = AC.compute_slice_for_sample(captures, 0, q_idx, v_idx, valid_lp)
    check("masked-out keys are excluded, not merely down-weighted",
          torch.isfinite(got).all() and float(got[:, 0].abs().max()) == 0.0,
          "a key marked invalid receives exactly zero")
    check("a layer subset is honoured",
          AC.compute_slice_for_sample(captures, 0, q_idx, v_idx, valid,
                                      layer_indices=[0]).shape == (len(q_idx), len(v_idx)))
    check("an empty query or visual set returns zeros rather than raising",
          AC.compute_slice_for_sample(captures, 0, torch.empty(0, dtype=torch.long),
                                      v_idx, valid).shape == (0, len(v_idx)))


def test_restore():
    print("\nrestore -- the override is local, reversible, and exception-safe")
    from transformers.models.qwen3_vl import modeling_qwen3_vl as mod

    model, cfg = tiny_text_model(impl="sdpa")
    ids = torch.randint(0, cfg.vocab_size, (1, 9))
    reg = mod.ALL_ATTENTION_FUNCTIONS
    before = reg["sdpa"]
    with AC.AttentionSliceCapturer(model):
        during = reg["sdpa"]
        impl_during = AC.find_text_attention_modules(model)[0].config._attn_implementation
    check("the registry entry is swapped inside the block", during is not before)
    check("and restored on exit", reg["sdpa"] is before)
    check("_attn_implementation is NEVER renamed", impl_during == "sdpa",
          "verl branches on this string; renaming it would change the forward")

    try:
        with AC.AttentionSliceCapturer(model):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    check("restored even when the body raises", reg["sdpa"] is before)

    sentinel = lambda *a, **k: None                       # noqa: E731
    reg["sdpa"] = sentinel
    try:
        with AC.AttentionSliceCapturer(model):
            pass
        check("a PRIOR local override is put back, not deleted", reg["sdpa"] is sentinel)
    finally:
        del reg["sdpa"]
    check("and the global entry reappears once ours is dropped", reg["sdpa"] is before)

    # Numerical no-op: the model's own output must be untouched by the capture.
    base = a_forward(model, ids).last_hidden_state
    with AC.AttentionSliceCapturer(model):
        during_out = a_forward(model, ids).last_hidden_state
    after = a_forward(model, ids).last_hidden_state
    check("the capture leaves the forward bit-identical",
          torch.equal(base, during_out) and torch.equal(base, after),
          f"max|delta| {float((base - during_out).abs().max()):.1e}")


def test_eager_path():
    print("\neager -- 4.57.6 bypasses the registry when impl == 'eager'")
    model, cfg = tiny_text_model(impl="eager")
    ids = torch.randint(0, cfg.vocab_size, (1, 11))
    with AC.AttentionSliceCapturer(model) as cap:
        a_forward(model, ids)
        got = cap.get_captures()
    check("capture works under eager too", sorted(got) == list(range(cfg.num_hidden_layers)),
          "the module-level eager_attention_forward symbol is wrapped, since 4.57.6's "
          "forward never consults the registry for it")
    from transformers.models.qwen3_vl import modeling_qwen3_vl as mod
    check("the eager symbol is restored",
          mod.eager_attention_forward.__name__ != "laser_capture_attention_forward",
          mod.eager_attention_forward.__name__)


def main():
    print("laser fork: Qwen3-VL attention capture, CPU checks")
    import transformers
    print(f"  transformers {transformers.__version__}  "
          f"(the fork pins 4.57.6; both dispatch paths are covered)")
    test_api()
    test_discovery()
    test_capture()
    test_slice()
    test_restore()
    test_eager_path()
    print(f"\n{'ALL PASS' if not FAILURES else f'{len(FAILURES)} FAILED: {FAILURES}'}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
