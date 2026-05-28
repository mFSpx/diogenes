#!/usr/bin/env bash
set -euo pipefail

PIPE_DIR="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/lucidota-mps}"
export CUDA_MPS_PIPE_DIRECTORY="$PIPE_DIR"

if ! pgrep -u "$USER" -f 'nvidia-cuda-mps-control' >/dev/null 2>&1; then
  echo "LUCIDOTA MPS is not running"
  exit 0
fi

echo quit | nvidia-cuda-mps-control
echo "LUCIDOTA MPS stopped"
