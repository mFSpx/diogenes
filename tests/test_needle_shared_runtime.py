#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_needle_worker_exposes_shared_six_lane_batch_contract() -> None:
    worker = (ROOT / "scripts" / "lucidota_needle_worker.py").read_text(encoding="utf-8")
    assert "generate_batch" in worker
    assert "--slots" in worker
    assert '"/generate_batch"' in worker
    assert '"shared_server"' in worker
    assert '"slots"' in worker
    assert "rolling_window_500_token_chunks" in worker


def test_needle_start_script_defaults_to_one_shared_server() -> None:
    script = (ROOT / "scripts" / "lucidota_start_needle_swarm.sh").read_text(encoding="utf-8")
    assert 'SHARED_SERVER="${LUCIDOTA_NEEDLE_SHARED_SERVER:-1}"' in script
    assert 'SLOTS="${LUCIDOTA_NEEDLE_SLOTS:-6}"' in script
    assert '--slots "$SLOTS"' in script
    assert '--instance "needle-shared-${SLOTS}"' in script
    assert 'lucidota_needle_worker.py' in script
