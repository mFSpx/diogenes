#!/bin/bash
cd "$(dirname "$(dirname "$(readlink -f "$0")")")"
source scripts/lucidota_safe_ops_env.sh 2>/dev/null || true

# 1. BGE embed fleet (6 wide)
if ! pgrep -f "bge_fleet" > /dev/null 2>&1; then
  bash scripts/lucidota_bge_fleet.sh 6 >> 04_RUNTIME/bge_fleet.log 2>&1 || true
  echo "[startup] BGE fleet x6 started"
else echo "[startup] BGE fleet already running"; fi

# 2. Governor daemon
if ! pgrep -f "lucidota_guvna.py" > /dev/null 2>&1; then
  nohup .venv/bin/python3 scripts/lucidota_guvna.py --interval 30 >> 04_RUNTIME/guvna_daemon.log 2>&1 &
  echo "[startup] governor pid=$!"
else echo "[startup] governor already running"; fi

# 3. RAM watchdog
if ! pgrep -f "lucidota_ram_watchdog" > /dev/null 2>&1; then
  WATCHDOG_FLOOR_MB=750 WATCHDOG_RESUME_MB=1300 WATCHDOG_TICKS=200 \
    nohup bash scripts/lucidota_ram_watchdog.sh >> 04_RUNTIME/ram_watchdog.log 2>&1 &
  echo "[startup] watchdog pid=$!"
else echo "[startup] watchdog already running"; fi

# 4. Mamba GPU (:8083)
if ! curl -sf http://127.0.0.1:8083/health > /dev/null 2>&1; then
  LUCIDOTA_MAMBA_GPU_NGL=99 \
    nohup bash scripts/lucidota_start_mamba_gpu_partial.sh \
    >> 04_RUNTIME/inference_os/mamba7b_gpu_llama_server.log 2>&1 &
  echo "[startup] mamba-gpu pid=$!"
else echo "[startup] mamba-gpu already live"; fi

# 5. 6x Needle workers (:8090-8095)
for i in 0 1 2 3 4 5; do
  port=$((8090 + i))
  if ! curl -sf http://127.0.0.1:$port/ > /dev/null 2>&1; then
    JAX_PLATFORMS=cpu PYTHONPATH=01_REPOS/needle \
      nohup .venv/bin/python3 scripts/lucidota_needle_worker.py \
      --port $port --instance needle-$i \
      >> 04_RUNTIME/needle_swarm/needle-$i.log 2>&1 &
    echo "[startup] needle-$i pid=$! :$port"
  fi
done

# 6. corpus→graph loop
if ! pgrep -f "corpus_to_graph_loop" > /dev/null 2>&1; then
  nohup bash scripts/corpus_to_graph_loop.sh >> 04_RUNTIME/corpus_to_graph_loop.log 2>&1 &
  echo "[startup] corpus_to_graph pid=$!"
else echo "[startup] corpus_to_graph already running"; fi

# 7. Corpus embed fill (batch BGE)
if ! pgrep -f "corpus_embed_fill" > /dev/null 2>&1; then
  LUCIDOTA_BGE_FLEET=$(cat 04_RUNTIME/inference_os/bge_fleet.endpoints 2>/dev/null || echo "http://127.0.0.1:8101/v1/embeddings") \
    nohup .venv/bin/python3 scripts/corpus_embed_fill.py --batch 32 --workers 12 \
    >> 04_RUNTIME/embed_fill.log 2>&1 &
  echo "[startup] embed_fill pid=$!"
else echo "[startup] embed_fill already running"; fi

# 8. Forge proxy (:9000 → :8083)
if ! curl -sf http://127.0.0.1:9000/health > /dev/null 2>&1; then
  nohup .venv/bin/python3 scripts/forge_proxy_start.py >> 04_RUNTIME/forge_proxy.log 2>&1 &
  echo "[startup] forge_proxy pid=$!"
else echo "[startup] forge_proxy already live"; fi

echo "[startup] done — $(date -u +%Y%m%dT%H%M%SZ)"
