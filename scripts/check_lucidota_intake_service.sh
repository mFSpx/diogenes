#!/usr/bin/env bash
set -euo pipefail
SERVICE="lucidota-intake.service"
echo "--- user-systemd service ---"
if [[ ! -f "$HOME/.config/systemd/user/$SERVICE" ]]; then
  echo "service_unit_missing=$HOME/.config/systemd/user/$SERVICE"
else
  systemctl --user --no-pager status "$SERVICE" || true
fi
echo "--- recent service logs ---"
journalctl --user -u "$SERVICE" --no-pager -n "${LOG_LINES:-40}" || true
echo "--- binary ---"
BIN="/home/mfspx/LUCIDOTA/01_REPOS/lucidota_etl/target/release/lucidota-intake"
if [[ -x "$BIN" ]]; then
  "$BIN" --help | head -20
else
  echo "binary_missing=$BIN"
fi
