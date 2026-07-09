#!/bin/bash
DST=~/scratch/research/saliency_r1/cold_data/Saliency-R1-cold/llava_cot_images
LOG=~/scratch/research/llavacot_unzip.log
TARGET=1391473
for i in $(seq 1 6); do
  if grep -q "UNZIP DONE" "$LOG" 2>/dev/null; then echo "EXTRACTION COMPLETE"; break; fi
  sleep 300
  n=$(find "$DST" -type f 2>/dev/null | wc -l)
  pct=$(( n * 100 / TARGET ))
  echo "[$(date +%H:%M)] ${pct}%  ${n} / ${TARGET} files"
done
echo "=== monitor window end ==="
grep -q "UNZIP DONE" "$LOG" 2>/dev/null && tail -3 "$LOG"
