#!/usr/bin/env bash
set -euo pipefail

echo "LUCIDOTA CUDA Inventory"
echo "======================="

if command -v nvidia-smi >/dev/null 2>&1; then
  echo
  echo "GPU"
  nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,driver_version,compute_cap,pstate,temperature.gpu,power.draw --format=csv,noheader,nounits || true
  echo
  echo "Processes"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits || true
else
  echo
  echo "GPU"
  echo "nvidia-smi not found"
fi

echo
echo "CUDA Tools"
for bin in nvidia-cuda-mps-control nvidia-cuda-mps-server nvcc python3; do
  if command -v "$bin" >/dev/null 2>&1; then
    printf "%-28s %s\n" "$bin" "$(command -v "$bin")"
  else
    printf "%-28s missing\n" "$bin"
  fi
done

echo
echo "Display Devices"
lspci | rg -i 'vga|3d|display|nvidia|intel|amd' || true

echo
echo "Python ML Imports"
python3 - <<'PY'
mods = ["jax", "torch", "transformers", "peft", "river", "treelite"]
for name in mods:
    try:
        mod = __import__(name)
        version = getattr(mod, "__version__", "unknown")
        print(f"{name:<14} {version}")
    except Exception as exc:
        print(f"{name:<14} missing ({exc.__class__.__name__})")
PY
