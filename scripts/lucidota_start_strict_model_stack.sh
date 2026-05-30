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
start_server deepseek 8080 "$ROOT/04_RUNTIME/inference_os/deepseek_q4.pid" "$ROOT/04_RUNTIME/inference_os/deepseek_q4_llama_server.log" \
  "$ROOT/scripts/lucidota_start_deepseek_llama.sh"
start_server mamba7b_ram 8081 "$ROOT/04_RUNTIME/inference_os/mamba7b_ternary.pid" "$ROOT/04_RUNTIME/inference_os/mamba7b_ternary_cpu_llama_server.log" \
  env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "$LLAMA" \
  -m "$ROOT/03_VAULT/models/tensorblock/Falcon3-Mamba-7B-Instruct-GGUF/Falcon3-Mamba-7B-Instruct-Q2_K.gguf" --host 127.0.0.1 --port 8081 -ngl 0 -c 256 --parallel 1 --batch-size 32 --ubatch-size 8 --cache-ram 0 --no-warmup
start_server bonsai4b_ram 8082 "$ROOT/04_RUNTIME/inference_os/bonsai4b_ternary.pid" "$ROOT/04_RUNTIME/inference_os/bonsai4b_ternary_cpu_llama_server.log" \
  "$ROOT/scripts/lucidota_start_bonsai_ternary_llama.sh"
if [[ "${LUCIDOTA_ENABLE_MAMBA_GPU_PARTIAL:-0}" == "1" ]]; then
  start_server mamba7b_gpu_partial 8083 "$ROOT/04_RUNTIME/inference_os/mamba7b_gpu.pid" "$ROOT/04_RUNTIME/inference_os/mamba7b_gpu_llama_server.log" \
    "$ROOT/scripts/lucidota_start_mamba_gpu_partial.sh"
else
  echo "mamba7b_gpu_partial skipped by strict admission"
fi
LUCIDOTA_NEEDLE_COUNT=6 "$ROOT/scripts/lucidota_start_needle_swarm.sh"
"$ROOT/scripts/lucidota_start_indy_reads_watcher.sh"
python3 "$ROOT/scripts/lucidota_model_turbine_overseer.py"
