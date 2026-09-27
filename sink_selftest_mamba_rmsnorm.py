#!/usr/bin/env python
"""Is `vendor/mamba_ssm_min` still the upstream kernel, and does it still compute it?

`vendor/mamba_ssm_min/README.md` claims two things that nothing else would catch if they
stopped being true:

  1. the vendored file is state-spaces/mamba v2.2.5's `layernorm_gated.py`, UNMODIFIED;
  2. path-vendoring it leaves `is_mamba_2_ssm_available()` False, so the 12B runs the
     same torch-native Mamba path the Omni row of the cross-model table was measured on.

A quiet edit to (1) or a transformers change to (2) would not raise anywhere -- the model
would load and produce numbers, on a different code path from the model it is being
compared against. Hence a selftest.

    python sink_selftest_mamba_rmsnorm.py          # all of it, needs a GPU for Triton
    python sink_selftest_mamba_rmsnorm.py --no-net # skip the upstream byte comparison
"""
import argparse
import hashlib
import sys
from pathlib import Path

VENDOR = Path(__file__).resolve().parent / "vendor" / "mamba_ssm_min"
REL = "mamba_ssm/ops/triton/layernorm_gated.py"
UPSTREAM = ("https://raw.githubusercontent.com/state-spaces/mamba/v2.2.5/"
            "mamba_ssm/ops/triton/layernorm_gated.py")


def check_verbatim():
    """Byte-for-byte against the tag the README names."""
    import urllib.request

    local = (VENDOR / REL).read_bytes()
    with urllib.request.urlopen(UPSTREAM, timeout=30) as r:
        remote = r.read()
    lh, rh = hashlib.sha256(local).hexdigest(), hashlib.sha256(remote).hexdigest()
    print(f"  local  sha256 {lh}")
    print(f"  v2.2.5 sha256 {rh}")
    if lh != rh:
        raise SystemExit(
            "vendored layernorm_gated.py is NOT upstream v2.2.5. Either it was edited -- "
            "in which case the 12B's Mamba arithmetic is ours, not NVIDIA's -- or the tag "
            "moved. Re-vendor or update the README; do not paper over it.")
    print("  VERBATIM: yes")


def check_absent_to_transformers():
    """The whole reason it is a path and not an install -- tested on the GPU branch.

    `is_mamba_2_ssm_available()` is

        is_torch_cuda_available() and is_available and parse(version) >= parse("2.0.4")

    and `and` short-circuits, so on a CPU-only box it returns False at the FIRST operand
    and never touches the rest. That is not the question: the model always runs on a GPU,
    where the third operand is evaluated. Asking this on the login node as-is passes for
    a reason that has nothing to do with the vendored package -- which is exactly how the
    first attempt shipped a version string of 'N/A' straight into `packaging.parse` and
    died on the node, having 'passed' here. So CUDA is forced true and the cache cleared.
    """
    import importlib.util

    # ORDER MATTERS, and it is the job's order: transformers is imported (and builds its
    # `PACKAGE_DISTRIBUTION_MAPPING` once) BEFORE the vendored directory joins sys.path.
    # Appending first would test a mapping that already knows about the package, which is
    # a situation no real run is ever in.
    from transformers.utils import import_utils as iu

    iu._is_package_available("mamba_ssm", return_version=True)
    sys.path.append(str(VENDOR))

    assert importlib.util.find_spec("mamba_ssm") is not None, \
        "vendored mamba_ssm is not importable -- the path append did not take"
    found, ver = iu._is_package_available("mamba_ssm", return_version=True)
    print(f"  importable                    : {found}")
    print(f"  version metadata              : {ver!r}")
    if ver == "N/A":
        raise SystemExit(
            "the vendored package has no dist-info, so transformers reports its version "
            "as 'N/A' and `packaging.version.parse` raises InvalidVersion on any GPU -- "
            "the model cannot load at all. Restore "
            "vendor/mamba_ssm_min/mamba_ssm-*.dist-info/METADATA.")

    iu.is_torch_cuda_available = lambda: True          # the branch a GPU node takes
    iu.is_mamba_2_ssm_available.cache_clear()
    try:
        available = iu.is_mamba_2_ssm_available()
    except Exception as e:
        raise SystemExit(f"is_mamba_2_ssm_available() raises on a GPU: {e!r}")
    print(f"  is_mamba_2_ssm_available()    : {available}  (CUDA forced true)")
    if available:
        raise SystemExit(
            "transformers now reports mamba_ssm as available from a path-vendored "
            "package. The 12B would take the FUSED SSM path while the Omni row it is "
            "compared against ran torch-native. Install mamba-ssm properly for both, or "
            "force the slow path -- but do not leave the two rows on different kernels.")


def check_numerics():
    """The Triton kernel against `rms_norm_ref`, the torch reference in the same file."""
    import torch

    sys.path.append(str(VENDOR))
    from mamba_ssm.ops.triton.layernorm_gated import rms_norm_ref, rmsnorm_fn

    if not torch.cuda.is_available():
        print("  SKIPPED (no GPU; Triton needs one)")
        return

    torch.manual_seed(0)
    # the 12B's own shape: hidden 5120, n_groups 8 -> group_size 640
    n, d, g = 4, 5120, 640
    for dtype in (torch.float32, torch.bfloat16):
        x = torch.randn(n, d, device="cuda", dtype=dtype)
        z = torch.randn(n, d, device="cuda", dtype=dtype)
        w = torch.randn(d, device="cuda", dtype=dtype)
        got = rmsnorm_fn(x=x, weight=w, bias=None, z=z, eps=1e-5,
                         group_size=g, norm_before_gate=False)
        ref = rms_norm_ref(x, w, None, z=z, eps=1e-5, group_size=g,
                           norm_before_gate=False)
        err = (got.float() - ref.float()).abs().max().item()
        tol = 2e-5 if dtype is torch.float32 else 5e-2
        print(f"  {str(dtype):>16}  max|triton-ref| = {err:.3e}  (tol {tol:g})")
        if not (err < tol):
            raise SystemExit(f"vendored rmsnorm_fn disagrees with its own reference "
                             f"at {dtype}: {err:.3e}")
    print("  NUMERICS: ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-net", action="store_true",
                    help="skip the upstream byte comparison (offline nodes)")
    args = ap.parse_args()

    print("[1/3] vendored file is upstream v2.2.5")
    if args.no_net:
        print("  SKIPPED (--no-net)")
    else:
        check_verbatim()
    print("[2/3] transformers still sees it as absent (slow path, as the Omni ran)")
    check_absent_to_transformers()
    print("[3/3] Triton kernel against the file's own torch reference")
    check_numerics()
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
