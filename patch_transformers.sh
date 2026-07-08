#!/usr/bin/env bash
# Patch the installed transformers with Saliency-R1's attention-output versions.
# Must run AFTER `pip install -r requirements_clean.txt` (transformers must exist).
# Idempotent: backs up originals to *.orig on first run.
set -euo pipefail

REPO=/home/uberger/scratch/research/saliency_r1
source /home/uberger/scratch/miniconda3/etc/profile.d/conda.sh
conda activate saliency_r1

SP=$(python -c "import transformers, os; print(os.path.dirname(transformers.__file__))")
echo "transformers at: $SP"

SDPA="$SP/integrations/sdpa_attention.py"
QWEN="$SP/models/qwen2_5_vl/modeling_qwen2_5_vl.py"

for f in "$SDPA" "$QWEN"; do
    [ -f "$f" ] || { echo "MISSING target: $f"; exit 1; }
    [ -f "$f.orig" ] || cp "$f" "$f.orig"
done

cp "$REPO/transformer/sdpa_attention.py" "$SDPA"
echo "  ✓ patched $SDPA"
cp "$REPO/transformer/modeling_qwen2_5_vl.py" "$QWEN"
echo "  ✓ patched $QWEN"
echo "Done. Restore with: mv <file>.orig <file>"
