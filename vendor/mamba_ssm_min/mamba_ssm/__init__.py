"""NOT mamba-ssm. One upstream Triton file; see ../README.md.

`__version__` is what actually makes `transformers` read this as "present but far too
old", and it has to live HERE rather than only in the sibling dist-info.
`_is_package_available` resolves a distribution name through
`PACKAGE_DISTRIBUTION_MAPPING`, which is built once when transformers is imported --
before this directory is appended to `sys.path` -- so the lookup KeyErrors and the
fallback branch reads `__version__` off the imported module instead. Without it that
branch yields the string "N/A", which `packaging.version.parse` refuses, and the 12B
cannot load on any GPU.

Being below 2.0.4 is the point: `is_mamba_2_ssm_available()` then returns False, the
fused SSM kernels stay unimported, and the Mamba mixers run the torch-native path that
the Omni row of docs/sink-location-cross-model.md was measured on.
"""

__version__ = "0.0.0+layernorm.only"
