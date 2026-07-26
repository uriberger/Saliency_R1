#!/bin/bash
# Derive CUDA_HOME and fix PATH so the CUDA toolkit and the active conda env's
# interpreter coexist without shadowing each other.
#
# SOURCE this (do not execute it) AFTER `conda activate <env>` -- it relies on
# CONDA_PREFIX and its exports must land in the caller's shell:
#     conda activate "$CONDA_ENV"
#     source "$REPO/setup_cuda_home.sh"
#
# Preference order for CUDA_HOME:
#   1. explicit CUDA_HOME from the environment (manual override still wins)
#   2. the active conda env, if it ships nvcc -- the normal, self-consistent
#      case: install nvcc into the training env so toolkit and runtime are ONE
#      env and no cross-env CUDA_HOME juggling is needed, e.g.
#          mamba install -n <env> -c nvidia -c conda-forge \
#              cuda-nvcc=<ver> cuda-cudart-dev=<ver> cuda-cccl=<ver>
#   3. a system toolkit on the node, if one exists
#   4. legacy hardcoded fallback (check_cuda_home.sh fails fast if it's absent)
if [ -z "${CUDA_HOME:-}" ]; then
    if [ -x "${CONDA_PREFIX:-}/bin/nvcc" ]; then
        export CUDA_HOME="$CONDA_PREFIX"
    elif [ -x /usr/local/cuda/bin/nvcc ]; then
        export CUDA_HOME=/usr/local/cuda
    fi
fi
export CUDA_HOME="${CUDA_HOME:-/cm/shared/apps/cuda12.4/toolkit/12.4.1}"
echo "Using CUDA_HOME=$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# CUDA_HOME may be a *conda env* (the only place with nvcc on offline nodes),
# whose bin/ also holds python/pip/accelerate and would shadow the active env's
# interpreter via the prepend above. Re-assert the active env's bin at the FRONT
# so python/torchrun/etc always resolve to the active env, regardless of which
# nvcc-bearing env CUDA_HOME points at.
if [ -n "${CONDA_PREFIX:-}" ]; then
    export PATH="$CONDA_PREFIX/bin:$PATH"
    hash -r 2>/dev/null || true
fi
