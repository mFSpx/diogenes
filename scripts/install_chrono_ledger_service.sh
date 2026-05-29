#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$ROOT_DIR/01_REPOS/lucidota_etl"
UNIT_SRC="$WORKSPACE/deploy/systemd/user/lucidota-chrono-ledger.service"
UNIT_DST="$HOME/.config/systemd/user/lucidota-chrono-ledger.service"
ENV_DIR="$HOME/.config/lucidota"
ENV_FILE="$ENV_DIR/chrono-ledger.env"
ENABLE=0
START=0
BUILD=0

for arg in "$@"; do
  case "$arg" in
    --enable) ENABLE=1 ;;
    --start) START=1 ;;
    --build) BUILD=1 ;;
    --help|-h)
      cat <<USAGE
Usage: $0 [--build] [--enable] [--start]
Installs the LUCIDOTA Chrono-Ledger daemon as a user-systemd service.
USAGE
      exit 0
      ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

if [[ "$BUILD" == "1" ]]; then
  cargo build --release -p lucidota-chrono-ledger --manifest-path "$WORKSPACE/Cargo.toml"
fi

if [[ ! -x "$WORKSPACE/target/release/lucidota-chrono-ledger" ]]; then
  echo "missing release binary: $WORKSPACE/target/release/lucidota-chrono-ledger" >&2
  echo "run: $0 --build --enable --start" >&2
  exit 1
fi

mkdir -p "$(dirname "$UNIT_DST")" "$ENV_DIR"
cp "$UNIT_SRC" "$UNIT_DST"

if [[ ! -f "$ENV_FILE" ]]; then
  cat > "$ENV_FILE" <<ENV
DATABASE_URL=${DATABASE_URL:-postgresql:///lucidota_storage}
LUCIDOTA_CHRONO_REPO_ROOT=$ROOT_DIR
LUCIDOTA_CHRONO_CURSOR_NAME=chrono-ledger-daemon
LUCIDOTA_CHRONO_REPLAY_LIMIT=100000
ENV
fi

systemctl --user daemon-reload
if [[ "$ENABLE" == "1" ]]; then
  systemctl --user enable lucidota-chrono-ledger.service
fi
if [[ "$START" == "1" ]]; then
  systemctl --user restart lucidota-chrono-ledger.service
fi

systemctl --user --no-pager status lucidota-chrono-ledger.service || true
printf 'installed_unit=%s\nenv_file=%s\n' "$UNIT_DST" "$ENV_FILE"
