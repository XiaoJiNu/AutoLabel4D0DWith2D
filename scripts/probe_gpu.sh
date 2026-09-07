#!/usr/bin/env bash
set -euo pipefail
LOG_DIR="${1:-/data/data/automomous/autolabel4d/logs}"
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
OUT="$LOG_DIR/probe_gpu_${STAMP}.txt"
{
  echo "=== hostname ==="
  hostname
  echo "=== date ==="
  date
  echo "=== nvidia-smi ==="
  nvidia-smi
  echo "=== query ==="
  nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv
  echo "=== disk /data ==="
  df -h /data
} | tee "$OUT"
echo "wrote $OUT"
