#!/usr/bin/env bash
# lucidota_capped_run.sh — run ANY heavy job under a HARD memory ceiling so it
# can never freeze the whole box again.
#
# Born from the 2026-05-29 OOM freeze: an unbounded ingest python (~10GB virtual)
# triggered a global OOM on this 7.6GB box; the COSMIC desktop thrashed to a
# frozen halt before the kernel killed anything -> hard reboot.
#
# This wrapper runs the job in a transient systemd --user scope with MemoryMax.
# If the job exceeds the cap, ONLY its own cgroup is OOM-killed; the desktop,
# Postgres, and model servers survive. The memory controller is delegated to
# user.slice on this box, so NO sudo is required.
#
# Usage:
#   scripts/lucidota_capped_run.sh python3 scripts/<ingest_script>.py [args...]
# Tunables (env):
#   LUCIDOTA_CAP_MEM   hard RSS kill ceiling   (default 4G)
#   LUCIDOTA_CAP_HIGH  soft throttle threshold (default 3500M)
#   LUCIDOTA_CAP_SWAP  swap ceiling (die, don't thrash zram) (default 512M)
#   LUCIDOTA_CAP_CPU   CPUQuota, leave cores for desktop (default 300%)
set -euo pipefail

ROOT="${LUCIDOTA_HOME:-/home/mfspx/LUCIDOTA}"
# Defense in depth: advisory safe-ops env caps (batch/thread limits) on top of the hard cap.
[[ -f "${ROOT}/scripts/lucidota_safe_ops_env.sh" ]] && source "${ROOT}/scripts/lucidota_safe_ops_env.sh"

CAP_MEM="${LUCIDOTA_CAP_MEM:-4.5G}"
CAP_HIGH="${LUCIDOTA_CAP_HIGH:-3500M}"
CAP_SWAP="${LUCIDOTA_CAP_SWAP:-512M}"
CAP_CPU="${LUCIDOTA_CAP_CPU:-300%}"

if [[ $# -eq 0 ]]; then
  echo "usage: $0 <command> [args...]   (runs under MemoryMax=${CAP_MEM} MemorySwapMax=${CAP_SWAP})" >&2
  exit 2
fi

# Probe: does systemd-run --user accept a MemoryMax scope on this box?
if systemd-run --user --scope --quiet -p MemoryMax=2G -- true >/dev/null 2>&1; then
  echo "[capped-run] HARD CAP active: MemoryMax=${CAP_MEM} MemoryHigh=${CAP_HIGH} MemorySwapMax=${CAP_SWAP} CPUQuota=${CAP_CPU}" >&2
  echo "[capped-run] -> $*" >&2
  exec systemd-run --user --scope --quiet \
    -p MemoryMax="${CAP_MEM}" \
    -p MemoryHigh="${CAP_HIGH}" \
    -p MemorySwapMax="${CAP_SWAP}" \
    -p CPUQuota="${CAP_CPU}" \
    -- "$@"
else
  # Fallback if user-scope memory control is unavailable: ulimit virtual-memory cap.
  VKB="${LUCIDOTA_CAP_VKB:-4194304}"  # 4 GiB in KB
  echo "[capped-run] systemd --user cap unavailable; ulimit -v ${VKB}KB fallback" >&2
  echo "[capped-run] -> $*" >&2
  ( ulimit -v "${VKB}"; exec "$@" )
fi
