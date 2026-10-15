#!/usr/bin/env bash
# Full benchmark on real Sim10k, single GPU (physical id 2). Sequential so runs don't share CPUs.
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
GPU=${GPU:-2}
mkdir -p results/train logs

echo "== loader sweep $(date)"
CUDA_VISIBLE_DEVICES=$GPU $PY bench/bench_loader.py --images data/sim10k/images/train --device cuda:0 \
  --out results/loader.json > logs/loader.log 2>&1 || echo "loader sweep FAILED"

run() {  # mode workers imgsz batch fraction
  out=results/train/${1}_w${2}_${3}.json
  [ -f "$out" ] && { echo "skip $out"; return; }
  echo "== train $1 w$2 $3px $(date +%T)"
  $PY bench/train_yolo.py --mode "$1" --workers "$2" --imgsz "$3" --batch "$4" --fraction "$5" \
    --epochs 2 --device "$GPU" --out "$out" > "logs/train_${1}_w${2}_${3}.log" 2>&1 \
    || echo "FAILED $out (see logs/train_${1}_w${2}_${3}.log)"
  grep PASTA_TIMING "logs/train_${1}_w${2}_${3}.log" | tail -1 | cut -c1-200
}

for w in 2 4 8 16 32; do
  for m in none gpu cpu_ours cpu_official; do run $m $w 640 16 0.2; done
done
for w in 8 16; do
  for m in none gpu cpu_ours cpu_official; do run $m $w 1280 8 0.1; done
done
echo "== done $(date)"
