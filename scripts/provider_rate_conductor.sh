#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$ROOT_DIR/01_REPOS/lucidota_etl"
BIN="${LUCIDOTA_PROVIDER_RATE_CONDUCTOR_BIN:-$WORKSPACE/target/release/provider-rate-conductor}"

if [[ -x "$BIN" ]]; then
  exec "$BIN" "$@"
fi

BIN_DEBUG="$WORKSPACE/target/debug/provider-rate-conductor"
if [[ -x "$BIN_DEBUG" ]]; then
  exec "$BIN_DEBUG" "$@"
fi

exec cargo run --quiet --manifest-path "$WORKSPACE/Cargo.toml" -p lucidota-workers --bin provider-rate-conductor -- "$@"
