#!/bin/bash
# Does ONE 80 GB card serve the Omni for generation? -- the open question in
# docs/omni-gpu-layout.md, answered on a real node.
#
#   srun ... bash run_omni_vllm_probe.sh [--gpu-mem 0.90] [--max-model-len 4096]
#
# Runs in `nemotron_vllm`, NOT `nemotron`: the Omni needs vLLM >= 0.20, which pins torch
# 2.11, and the trainer stays on the 2.8 that the 34.0 s step was measured on. Two
# environments is the price of not re-validating the training side; see
# docs/omni-training-harness.md.
set -euo pipefail

REPO="$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")" && pwd)"
CONDA_SH=/home/uberger/scratch/miniconda3/etc/profile.d/conda.sh
export HF_HOME=${HF_HOME:-/home/uberger/scratch/cache/hf_cache}
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
# The remote code imports a decoder that raises without `rmsnorm_fn`; the vendored
# layernorm-only mamba_ssm is what supplies it, and having no dist-info is deliberate
# (`is_mamba_2_ssm_available()` keeps reading False, so the fused kernels stay off).
export PYTHONPATH="$REPO/vendor/mamba_ssm_min:${PYTHONPATH:-}"

# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate nemotron_vllm

# The environment the Omni's generation server needs on THIS cluster. Every one of these
# is a hard failure without it, and omni_vllm_probe.py is what found each:
#
#   VLLM_ENABLE_V1_MULTIPROCESSING=0   the EngineCore CHILD hangs -- it reaches the
#                                      worker's memory snapshot and then sits in a futex
#                                      with 43 sleeping threads while the parent prints
#                                      "Waiting for 1 local core engine proc" forever.
#                                      In-process it loads normally.
#   VLLM_USE_DEEP_GEMM=0               kernel warmup calls DeepGEMM's FP8 path on a
#                                      bfloat16 model and raises "DeepGEMM backend is not
#                                      available or outdated".
#
# The other two are arguments rather than variables: `--kernel_config` picks the triton
# MoE backend (the default `auto` JIT-compiles FlashInfer CUTLASS and needs an nvcc these
# nodes do not have), and `--max_num_seqs` stays under the Mamba state-block count.
export VLLM_ENABLE_V1_MULTIPROCESSING=${VLLM_ENABLE_V1_MULTIPROCESSING:-0}
export VLLM_USE_DEEP_GEMM=${VLLM_USE_DEEP_GEMM:-0}

# Startup diagnostics are opt-in: SR1_VLLM_DEBUG=1 turns on vLLM's own DEBUG log and
# NCCL's, which is what tells a slow 62 GB load apart from a hang in the collective setup.
if [ "${SR1_VLLM_DEBUG:-0}" = "1" ]; then
    export VLLM_LOGGING_LEVEL=DEBUG
    export NCCL_DEBUG=INFO
    export NCCL_DEBUG_SUBSYS=INIT,ENV
    # Run the engine IN THIS PROCESS. The engine normally lives in a child, and a child
    # that goes quiet leaves the parent printing "Waiting for 1 local core engine proc"
    # forever with no way in -- ptrace is off here, so py-spy and gdb both refuse. In-
    # process, `--watchdog` can dump the stack of the thread that is actually stuck.
    export VLLM_ENABLE_V1_MULTIPROCESSING=0
    export PYTHONFAULTHANDLER=1
fi

cd "$REPO"
echo "=== node $(hostname) ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
python -c "import torch, vllm; print('torch', torch.__version__, '| vllm', vllm.__version__, '| cuda', torch.version.cuda)"

exec python omni_vllm_probe.py --stage engine "$@"
