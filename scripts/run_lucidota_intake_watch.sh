#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="${LUCIDOTA_ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BIN="${LUCIDOTA_INTAKE_BIN:-$ROOT_DIR/01_REPOS/lucidota_etl/target/release/lucidota-intake}"
DROP_DIR="${LUCIDOTA_INTAKE_DROP_DIR:-$ROOT_DIR/03_VAULT/korpus_krampii/DROP}"
DIGESTED_DIR="${LUCIDOTA_INTAKE_DIGESTED_DIR:-$ROOT_DIR/03_VAULT/korpus_krampii/DIGESTED}"
QUARANTINE_DIR="${LUCIDOTA_INTAKE_QUARANTINE_DIR:-$ROOT_DIR/03_VAULT/korpus_krampii/QUARANTINE}"
QUIET_MS="${LUCIDOTA_INTAKE_QUIET_MS:-1500}"
DRY_RUN="${LUCIDOTA_INTAKE_DRY_RUN:-1}"
if [[ ! -x "$BIN" ]]; then
  echo "missing executable: $BIN" >&2
  exit 1
fi
args=(watch "$DROP_DIR" --quiet-ms "$QUIET_MS" --digested-dir "$DIGESTED_DIR" --quarantine-dir "$QUARANTINE_DIR")
if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" || "$DRY_RUN" == "yes" ]]; then
  args+=(--dry-run)
fi
exec "$BIN" "${args[@]}"
