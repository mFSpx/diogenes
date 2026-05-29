#!/usr/bin/env bash
# LUCIDOTA laptop safe-ops defaults.
# Source this before Claw, model servers, scrapers, or watchers.
# Goal: keep the machine responsive; stream/chunk work; avoid swap storms.

# Shell-safe if sourced multiple times.
export LUCIDOTA_SAFE_OPS_PROFILE="${LUCIDOTA_SAFE_OPS_PROFILE:-shitty_laptop_palantir}"
export LUCIDOTA_DB_IS_OS="${LUCIDOTA_DB_IS_OS:-1}"
export LUCIDOTA_RUST_CORE_TARGET="${LUCIDOTA_RUST_CORE_TARGET:-1}"
export LUCIDOTA_PYTHON_ROLE="${LUCIDOTA_PYTHON_ROLE:-wet_clay_operator_adapters_only}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export LLAMA_OFFLINE="${LLAMA_OFFLINE:-1}"

# CPU/RAM anti-thrash: deterministic, low concurrency by default.
export LUCIDOTA_MAX_WORKERS="${LUCIDOTA_MAX_WORKERS:-1}"
export LUCIDOTA_MAX_BATCH="${LUCIDOTA_MAX_BATCH:-16}"
export LUCIDOTA_STREAM_CHUNK_BYTES="${LUCIDOTA_STREAM_CHUNK_BYTES:-65536}"
export LUCIDOTA_MAX_FILE_BYTES="${LUCIDOTA_MAX_FILE_BYTES:-1500000}"
export LUCIDOTA_HYPERTIMELINE_BATCH_SIZE="${LUCIDOTA_HYPERTIMELINE_BATCH_SIZE:-512}"
export LUCIDOTA_JSON_STREAM_CHARS="${LUCIDOTA_JSON_STREAM_CHARS:-65536}"
export LUCIDOTA_JSON_STREAM_THRESHOLD_MB="${LUCIDOTA_JSON_STREAM_THRESHOLD_MB:-64}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export MALLOC_TRIM_THRESHOLD_="${MALLOC_TRIM_THRESHOLD_:-131072}"

# 4GB GTX 1650 operating envelope. Keep reserve real; evict transient lanes.
export LUCIDOTA_VRAM_BUDGET_MB="${LUCIDOTA_VRAM_BUDGET_MB:-4096}"
export LUCIDOTA_VRAM_RESERVE_MB="${LUCIDOTA_VRAM_RESERVE_MB:-768}"
export LUCIDOTA_LLAMA_CTX="${LUCIDOTA_LLAMA_CTX:-2048}"
export LUCIDOTA_MAMBA_CTX="${LUCIDOTA_MAMBA_CTX:-2048}"
export LUCIDOTA_MAMBA_GPU_NGL="${LUCIDOTA_MAMBA_GPU_NGL:-24}"
export LUCIDOTA_BONSAI_NGL="${LUCIDOTA_BONSAI_NGL:-0}"
export LUCIDOTA_BONSAI_CTX="${LUCIDOTA_BONSAI_CTX:-1024}"
export LUCIDOTA_LLAMA_PARALLEL="${LUCIDOTA_LLAMA_PARALLEL:-1}"
export LUCIDOTA_MAMBA_PARALLEL="${LUCIDOTA_MAMBA_PARALLEL:-1}"
export LUCIDOTA_LLAMA_BATCH="${LUCIDOTA_LLAMA_BATCH:-256}"
export LUCIDOTA_MAMBA_BATCH="${LUCIDOTA_MAMBA_BATCH:-128}"
export LUCIDOTA_LLAMA_UBATCH="${LUCIDOTA_LLAMA_UBATCH:-64}"
export LUCIDOTA_MAMBA_UBATCH="${LUCIDOTA_MAMBA_UBATCH:-32}"
export LUCIDOTA_BONSAI_BATCH="${LUCIDOTA_BONSAI_BATCH:-64}"
export LUCIDOTA_BONSAI_UBATCH="${LUCIDOTA_BONSAI_UBATCH:-16}"

# Hybrid graphics policy: keep display/video clients on the onboard path.
# NVIDIA VRAM is for model services only unless the operator deliberately opts out.
if [[ "${LUCIDOTA_ALLOW_DGPU_GRAPHICS:-0}" != "1" ]]; then
  export DRI_PRIME="0"
  export __NV_PRIME_RENDER_OFFLOAD="0"
else
  export DRI_PRIME="${DRI_PRIME:-0}"
  export __NV_PRIME_RENDER_OFFLOAD="${__NV_PRIME_RENDER_OFFLOAD:-0}"
fi
if command -v lspci >/dev/null 2>&1 && lspci | grep -qiE 'VGA compatible controller: Intel|3D controller: Intel|Display controller: Intel'; then
  export LIBVA_DRIVER_NAME="${LIBVA_DRIVER_NAME:-iHD}"
fi

# Scraper ladder: metadata/API/static adapters first; browser last.
export LUCIDOTA_SCRAPER_IDEAL="${LUCIDOTA_SCRAPER_IDEAL:-adapter_buddha_form}"
export LUCIDOTA_BROWSER_FALLBACK_TIER="${LUCIDOTA_BROWSER_FALLBACK_TIER:-playwright_desperation_only}"
export LUCIDOTA_ALLOW_AMBIENT_BROWSER="${LUCIDOTA_ALLOW_AMBIENT_BROWSER:-0}"
export LUCIDOTA_AHOY_PAUSED="${LUCIDOTA_AHOY_PAUSED:-1}"

# Ontology and truth boundary.
export LUCIDOTA_ACTIVE_ONTOLOGY="${LUCIDOTA_ACTIVE_ONTOLOGY:-GO-25}"
export LUCIDOTA_CANONICAL_GRAPH_DIRECT_WRITES="${LUCIDOTA_CANONICAL_GRAPH_DIRECT_WRITES:-blocked}"
export LUCIDOTA_DEFAULT_CASE_KEY="${LUCIDOTA_DEFAULT_CASE_KEY:-KE26-00001}"

# CUDA runtime path — required for llama-server (compiled against CUDA 12).
# libcudart.so.12 ships with Ollama at /usr/local/lib/ollama/cuda_v12/.
# Without this, ALL model server lanes fail to start with "cannot open shared object file".
export LD_LIBRARY_PATH="/usr/local/lib/ollama/cuda_v12${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# LUCIDOTA venv: activate so all scripts get psycopg, river, bytewax, etc.
# Safe to source multiple times; skip if already active.
_LUCIDOTA_VENV="/home/mfspx/LUCIDOTA/.venv"
if [[ "${VIRTUAL_ENV:-}" != "${_LUCIDOTA_VENV}" ]] && [[ -f "${_LUCIDOTA_VENV}/bin/activate" ]]; then
  source "${_LUCIDOTA_VENV}/bin/activate"
fi
unset _LUCIDOTA_VENV

# Bitloops daemon health gate.
# Starts daemon if not running; exports BITLOOPS_DAEMON_OK=1 on success.
_BITLOOPS_BIN="${BITLOOPS_BIN:-/home/mfspx/.local/bin/bitloops}"
if command -v "${_BITLOOPS_BIN}" >/dev/null 2>&1; then
  _BL_STATUS="$("${_BITLOOPS_BIN}" status 2>/dev/null | grep -c 'running' || true)"
  if [[ "${_BL_STATUS}" -eq 0 ]]; then
    ACCESSIBLE=1 "${_BITLOOPS_BIN}" start --no-telemetry >/dev/null 2>&1 &
    sleep 1
    _BL_STATUS="$("${_BITLOOPS_BIN}" status 2>/dev/null | grep -c 'running' || true)"
  fi
  export BITLOOPS_DAEMON_OK=$([ "${_BL_STATUS}" -gt 0 ] && echo "1" || echo "0")
else
  export BITLOOPS_DAEMON_OK="0"
fi
unset _BITLOOPS_BIN _BL_STATUS

# Provider lane URLs: local model servers and embedding services.
export LUCIDOTA_LLAMA_URL="${LUCIDOTA_LLAMA_URL:-http://127.0.0.1:8080/v1}"
export LUCIDOTA_MAMBA_RAM_URL="${LUCIDOTA_MAMBA_RAM_URL:-http://127.0.0.1:8081/v1}"
export LUCIDOTA_MAMBA_GPU_URL="${LUCIDOTA_MAMBA_GPU_URL:-http://127.0.0.1:8083/v1}"
export LUCIDOTA_BGE_EMBED_URL="${LUCIDOTA_BGE_EMBED_URL:-http://127.0.0.1:8082/v1}"
export LUCIDOTA_MODERNBERT_EMBED_URL="${LUCIDOTA_MODERNBERT_EMBED_URL:-http://127.0.0.1:8084/v1}"
export LUCIDOTA_BGE_MODEL_PATH="${LUCIDOTA_BGE_MODEL_PATH:-/home/mfspx/LUCIDOTA/04_RUNTIME/models/bge-m3}"
export LUCIDOTA_MODERNBERT_MODEL_PATH="${LUCIDOTA_MODERNBERT_MODEL_PATH:-/home/mfspx/LUCIDOTA/04_RUNTIME/models/modernbert-base}"
