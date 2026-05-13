#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL="$ROOT/01_REPOS/doggystyle"
CLAW="$ROOT/01_REPOS/claudecode/rust"
KERNEL_SMOKE_HOME="$(mktemp -d /tmp/lucidota-kernel-smoke.XXXXXX)"
CLAW_SMOKE_HOME="$(mktemp -d /tmp/lucidota-clawd-smoke.XXXXXX)"
trap 'rm -rf "$KERNEL_SMOKE_HOME" "$CLAW_SMOKE_HOME"' EXIT

cd "$KERNEL"
if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv
fi
. .venv/bin/activate
python -m pip install -e . pytest >/dev/null
python -m pytest -q
python scripts/diogenes_grpc_smoke.py --home "$KERNEL_SMOKE_HOME"

cd "$CLAW"
cargo test --workspace
cargo build --release -p claw-cli
./target/release/claw diogenes-smoke --home "$CLAW_SMOKE_HOME"
