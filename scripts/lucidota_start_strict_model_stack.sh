#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/lucidota_safe_ops_env.sh"
mkdir -p "$ROOT/04_RUNTIME/inference_os" "$ROOT/04_RUNTIME/needle_swarm" "$ROOT/05_OUTPUTS/model_runtime"
PY="$ROOT/.venv/bin/python"
[[ -x "$PY" ]] || PY="python3"

"$PY" "$ROOT/scripts/lucidota_strict_model_stack_admission.py" --run-diogenes-gate
source "$ROOT/04_RUNTIME/inference_os/strict_model_stack_admission.env"

start_server() {
  local name="$1" port="$2" pidfile="$3" log="$4"; shift 4
  if curl -fsS --max-time 1 "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then echo "$name online :$port"; return 0; fi
  if [[ -s "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then echo "$name pending pid $(cat "$pidfile")"; return 0; fi
  setsid "$@" >"$log" 2>&1 < /dev/null & echo $! > "$pidfile"
  echo "$name started pid $(cat "$pidfile") :$port"
}

LLAMA="$ROOT/01_REPOS/llama.cpp/build-cuda/bin/llama-server"
# build-cuda llama-server dynamically links libcudart.so.12 even for CPU (-ngl 0) lanes;
# without this the RAM/CPU lanes (mamba_ram, deepseek) die on load. Root-cause fix.
export LD_LIBRARY_PATH="$ROOT/01_REPOS/llama.cpp/build-cuda/bin:/usr/local/lib/ollama/cuda_v12:${LD_LIBRARY_PATH:-}"
# DeepSeek R1 1.5B: ON DISK PASSIVE — not loaded at launch (operator 2026-06-06)
# Falcon3-Mamba-7B (RAM + GPU): PURGED from disk and all active code 2026-06-06
start_server bonsai8b_1bit 8082 "$ROOT/04_RUNTIME/inference_os/bonsai8b_1bit.pid" "$ROOT/04_RUNTIME/inference_os/bonsai8b_1bit_llama_server.log" \
  "$ROOT/scripts/lucidota_start_bonsai_ternary_llama.sh"
LUCIDOTA_NEEDLE_COUNT=6 "$ROOT/scripts/lucidota_start_needle_swarm.sh"
"$ROOT/scripts/lucidota_start_indy_reads_watcher.sh"
python3 "$ROOT/scripts/lucidota_model_turbine_overseer.py"
