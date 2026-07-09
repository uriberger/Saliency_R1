#!/bin/bash
DIR=~/scratch/research/saliency_r1/cold_data/LLaVA-CoT-100k
DLLOG=~/scratch/research/llavacot_dl.log
TARGET=170000000000   # ~170 GB
prev=$(du -sb "$DIR" 2>/dev/null | cut -f1); t_prev=$(date +%s)
for i in $(seq 1 5); do
  sleep 300
  if grep -q "^DONE" "$DLLOG" 2>/dev/null; then echo "DOWNLOAD COMPLETE"; break; fi
  now=$(du -sb "$DIR" 2>/dev/null | cut -f1); t_now=$(date +%s)
  dt=$((t_now - t_prev)); db=$((now - prev))
  rate=$(( db / (dt>0?dt:1) ))
  pct=$(( now * 100 / TARGET ))
  rem=$(( (TARGET - now) / (rate>0?rate:1) ))
  echo "[$(date +%H:%M)] ${pct}%  $(( now/1000000000 ))GB / ~170GB  rate=$(( rate/1000000 ))MB/s  ETA=$(( rem/60 ))min"
  prev=$now; t_prev=$t_now
done
echo "=== monitor window end ==="
du -sh "$DIR" 2>/dev/null
