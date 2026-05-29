#!/usr/bin/env bash
set -euo pipefail
cd /home/mfspx/LUCIDOTA
exec /home/mfspx/LUCIDOTA/.venv/bin/python /home/mfspx/LUCIDOTA/scripts/lucidota_ingest_watchdog.py --self-disable >> /home/mfspx/LUCIDOTA/04_RUNTIME/ingestion_watchdog.log 2>&1
