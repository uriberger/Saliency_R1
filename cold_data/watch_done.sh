#!/bin/bash
DIR=~/scratch/research/saliency_r1/cold_data/LLaVA-CoT-100k
DLLOG=~/scratch/research/llavacot_dl.log
for i in $(seq 1 40); do
  if grep -q "^DONE" "$DLLOG" 2>/dev/null; then echo "=== DOWNLOAD DONE ==="; grep "^DONE" "$DLLOG"; break; fi
  sleep 60
done
echo "--- final listing ---"
ls -la "$DIR"/*.part-* 2>/dev/null | wc -l
ls "$DIR"/ 2>/dev/null
du -sh "$DIR" 2>/dev/null
# any leftover incomplete?
find "$DIR"/.cache -name "*.incomplete" 2>/dev/null | wc -l
