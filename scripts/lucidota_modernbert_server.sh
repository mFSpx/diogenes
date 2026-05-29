#!/usr/bin/env bash
# ModernBERT encoder service via ONNX runtime on :8084
# GGUF conversion not possible (decoder.bias tensor mapping fails in current llama.cpp).
# Uses ONNX int8 model directly via Python http server.
set -euo pipefail
PORT="${1:-8084}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PY="${ROOT}/.venv/bin/python"

exec "${VENV_PY}" "${ROOT}/scripts/lucidota_modernbert_onnx_server.py" --port "${PORT}"
