#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA="$ROOT/01_REPOS/llama.cpp"

cmake -S "$LLAMA" -B "$LLAMA/build-cuda" \
  -DGGML_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build "$LLAMA/build-cuda" --target llama-cli llama-server -j"$(nproc)"

"$LLAMA/build-cuda/bin/llama-cli" --version
"$LLAMA/build-cuda/bin/llama-server" --version
