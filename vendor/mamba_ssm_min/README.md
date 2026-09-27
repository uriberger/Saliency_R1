# `mamba_ssm_min` — one upstream file, so the 12B Nemotron VL can be imported

`NVIDIA-Nemotron-Nano-12B-v2-VL`'s `modeling_nemotron_h.py` ends its import block with

```python
try:
    from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn
except ImportError:
    raise ImportError("mamba-ssm is required by the Mamba model but cannot be imported")
```

and `MambaRMSNormGated.forward` — every Mamba layer, on every forward — is nothing but a
call to `rmsnorm_fn`. So this is not an optional fast path that degrades: without the
function the checkpoint cannot be loaded at all, and a hand-written replacement would be
changing the model's arithmetic in the one place we are trying to measure.

Its Omni sibling does not have this problem. `Nemotron-3-Nano-Omni-30B-A3B`'s copy of the
same file lazy-loads the kernels and falls back to a torch-native gated RMSNorm, which is
how the Omni row of `docs/sink-location-cross-model.md` was measured — `mamba_ssm`,
`causal_conv1d` and `kernels` are all absent from every env on this machine.

## What is here, and what it deliberately is not

`mamba_ssm/ops/triton/layernorm_gated.py`, verbatim from
[state-spaces/mamba](https://github.com/state-spaces/mamba) at tag **v2.2.5**, Apache-2.0,
© 2024 Tri Dao. Unmodified: `sink_selftest_mamba_rmsnorm.py` checks it byte-for-byte
against the upstream URL, and checks the Triton kernel against `rms_norm_ref` — the pure
torch reference that ships in the same file — so a silent numerical drift fails loudly.

It needs only `torch`, `triton` and `einops`, all already in `saliency_r1_qwen3_vllm`.

**There is no dist-info, and that is the point.** `transformers.utils.is_mamba_2_ssm_available()`
resolves through `importlib.metadata`, so a path-vendored package reads as *absent*:
`selective_state_update` and `mamba_chunk_scan_combined` stay `None`,
`is_fast_path_available` stays False, and the Mamba mixers run the same torch-native
`torch_forward` the Omni ran. That keeps the two Nemotron rows measured on one code path
rather than making the 12B the only model in the panel on fused SSM kernels.

## Why vendored rather than installed

`pip install mamba-ssm` compiles CUDA extensions and the conda envs here are **shared** —
per `CLAUDE.md` an install is global, not worktree-local, and it would change the code
path of any Nemotron job another session is running. A directory on `sys.path` is visible
only to the process that adds it.

If a real Nemotron *training* run is ever set up, install the package properly and delete
this: the fused path is worth a large factor at training scale, and the torch fallback
here is a measurement convenience, not a recommendation.
