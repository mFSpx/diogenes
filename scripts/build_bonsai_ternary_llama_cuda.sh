#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA="$ROOT/01_REPOS/prismml_llama.cpp"
BUILD="$LLAMA/build-cuda"
export CCACHE_DIR="${CCACHE_DIR:-$ROOT/04_RUNTIME/ccache}"
mkdir -p "$CCACHE_DIR"
cmake -S "$LLAMA" -B "$BUILD" -G Ninja \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_FORCE_MMQ=ON \
  -DCMAKE_CUDA_ARCHITECTURES="${LUCIDOTA_CUDA_ARCH:-75}" \
  -DGGML_CCACHE=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD" --target llama-server llama-cli -j"${LUCIDOTA_BUILD_JOBS:-$(nproc)}"
"$BUILD/bin/llama-server" --version
