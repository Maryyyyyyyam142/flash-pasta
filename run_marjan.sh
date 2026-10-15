#!/usr/bin/env bash
# Training-throughput sweep on the Marjan Red Sea coral dataset (640x640 images), single GPU.
set -u
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
GPU=${GPU:-2}
mkdir -p results/train_marjan logs

for w in 4 8 16; do
  for m in none gpu cpu_ours cpu_official; do
    out=results/train_marjan/${m}_w${w}_640.json
    [ -f "$out" ] && { echo "skip $out"; continue; }
    echo "== marjan $m w$w $(date +%T)"
    $PY bench/train_yolo.py --data data/marjan.yaml --mode "$m" --workers "$w" --imgsz 640 --batch 16 \
      --fraction 0.25 --epochs 2 --device "$GPU" --out "$out" > "logs/marjan_${m}_w${w}.log" 2>&1 \
      || echo "FAILED $out (see logs/marjan_${m}_w${w}.log)"
    grep PASTA_TIMING "logs/marjan_${m}_w${w}.log" | tail -1 | cut -c1-200
  done
done
echo "== marjan done $(date)"
