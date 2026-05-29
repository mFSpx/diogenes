#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$ROOT_DIR/01_REPOS/lucidota_etl"
UNIT_SRC="$WORKSPACE/deploy/systemd/user/lucidota-intake.service"
UNIT_DST="$HOME/.config/systemd/user/lucidota-intake.service"
ENV_DIR="$HOME/.config/lucidota"
ENV_FILE="$ENV_DIR/intake.env"
EXECUTE=0
ENABLE=0
START=0
BUILD=0
SERVICE_DRY_RUN=1
for arg in "$@"; do
  case "$arg" in
    --dry-run) EXECUTE=0 ;;
    --execute) EXECUTE=1 ;;
    --enable) ENABLE=1 ;;
    --start) START=1 ;;
    --build) BUILD=1 ;;
    --live) SERVICE_DRY_RUN=0 ;;
    --dry-run-service) SERVICE_DRY_RUN=1 ;;
    --help|-h)
      cat <<USAGE
Usage: $0 [--dry-run|--execute] [--build] [--enable] [--start] [--live|--dry-run-service]
Default is command dry-run and service dry-run. --live sets LUCIDOTA_INTAKE_DRY_RUN=0 in the environment file.
USAGE
      exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done
if [[ "$BUILD" == "1" ]]; then
  cargo build --release -p lucidota-intake --manifest-path "$WORKSPACE/Cargo.toml"
fi
cat <<INFO
mode=$([[ "$EXECUTE" == "1" ]] && echo execute || echo dry_run)
unit_src=$UNIT_SRC
unit_dst=$UNIT_DST
env_file=$ENV_FILE
binary=$WORKSPACE/target/release/lucidota-intake
enable=$ENABLE
start=$START
service_dry_run=$SERVICE_DRY_RUN
INFO
if [[ "$EXECUTE" != "1" ]]; then
  echo "dry_run_no_systemd_mutation=1"
  exit 0
fi
if [[ ! -x "$WORKSPACE/target/release/lucidota-intake" ]]; then
  echo "missing release binary: $WORKSPACE/target/release/lucidota-intake" >&2
  echo "run: $0 --execute --build --enable" >&2
  exit 1
fi
mkdir -p "$(dirname "$UNIT_DST")" "$ENV_DIR"
cp "$UNIT_SRC" "$UNIT_DST"
cat > "$ENV_FILE" <<ENV
LUCIDOTA_ROOT_DIR=$ROOT_DIR
LUCIDOTA_INTAKE_DROP_DIR=$ROOT_DIR/03_VAULT/korpus_krampii/DROP
LUCIDOTA_INTAKE_DIGESTED_DIR=$ROOT_DIR/03_VAULT/korpus_krampii/DIGESTED
LUCIDOTA_INTAKE_QUARANTINE_DIR=$ROOT_DIR/03_VAULT/korpus_krampii/QUARANTINE
LUCIDOTA_INTAKE_QUIET_MS=1500
LUCIDOTA_INTAKE_DRY_RUN=$SERVICE_DRY_RUN
ENV
systemctl --user daemon-reload
if [[ "$ENABLE" == "1" ]]; then systemctl --user enable lucidota-intake.service; fi
if [[ "$START" == "1" ]]; then systemctl --user restart lucidota-intake.service; fi
systemctl --user --no-pager status lucidota-intake.service || true
printf 'installed_unit=%s\nenv_file=%s\n' "$UNIT_DST" "$ENV_FILE"
