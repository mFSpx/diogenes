#!/usr/bin/env bash
set -euo pipefail

PIPE_DIR="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/lucidota-mps}"
LOG_DIR="${CUDA_MPS_LOG_DIRECTORY:-$HOME/.local/state/lucidota/mps}"

mkdir -p "$PIPE_DIR" "$LOG_DIR"

export CUDA_MPS_PIPE_DIRECTORY="$PIPE_DIR"
export CUDA_MPS_LOG_DIRECTORY="$LOG_DIR"

if pgrep -u "$USER" -f 'nvidia-cuda-mps-control' >/dev/null 2>&1; then
  echo "LUCIDOTA MPS already running"
  echo "CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY"
  echo "CUDA_MPS_LOG_DIRECTORY=$CUDA_MPS_LOG_DIRECTORY"
  exit 0
fi

nvidia-cuda-mps-control -d

echo "LUCIDOTA MPS started"
echo "CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY"
echo "CUDA_MPS_LOG_DIRECTORY=$CUDA_MPS_LOG_DIRECTORY"
