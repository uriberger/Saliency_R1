#!/usr/bin/env bash
# Patch vLLM 0.11.0 for transformers 5.x compatibility.
#
# vLLM's get_cached_tokenizer() reads `tokenizer.all_special_tokens_extended`, which
# transformers 5.x REMOVED -> at server startup:
#     AttributeError: Qwen2Tokenizer has no attribute all_special_tokens_extended
# vLLM's own interface types this field as list[str], so we fall back to
# `all_special_tokens` (plain strings) when the extended attribute is absent.
#
# Idempotent: backs up tokenizer.py.orig on first run; safe to re-run.
# Usage:  bash patch_vllm_qwen3.sh [env_name]     # default env: saliency_r1_qwen3_vllm
set -euo pipefail

ENV=${1:-saliency_r1_qwen3_vllm}
source /home/uberger/scratch/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV"

TP=$(python -c "import vllm.transformers_utils.tokenizer as m; print(m.__file__)")
echo "=== patch_vllm_qwen3.sh: env=$ENV ==="
echo "  vllm tokenizer.py: $TP"

if grep -q 'all_special_tokens_extended", tokenizer.all_special_tokens' "$TP"; then
    echo "  (already patched - skipping)"
    exit 0
fi

[ -f "$TP.orig" ] || cp "$TP" "$TP.orig"

python - "$TP" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
needle = "tokenizer.all_special_tokens_extended)"
repl = ('getattr(tokenizer, "all_special_tokens_extended", '
        'tokenizer.all_special_tokens))  # transformers 5.x compat')
n = s.count(needle)
if n != 1:
    sys.exit(f"  EXPECTED exactly 1 occurrence of pattern, found {n} - vLLM version "
             f"differs; not patching (inspect {p} manually).")
open(p, "w").write(s.replace(needle, repl, 1))
print("  patched get_cached_tokenizer -> all_special_tokens fallback")
PY

echo "Done. Verify: python -c 'from vllm.transformers_utils.tokenizer import get_cached_tokenizer'"
