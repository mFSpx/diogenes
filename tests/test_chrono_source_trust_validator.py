#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from chrono_source_trust_validator import validate_rows


def test_legacy_dbos_queue_event_bridge_is_accepted_at_same_weight_as_absurd():
    rows, blockers = validate_rows([
        {"evidence_source": "dbos_queue_event_bridge", "min_weight": 0.55, "max_weight": 0.55, "n": 3}
    ])
    assert blockers == []
    assert rows[0]["expected_weight"] == 0.55


def test_unknown_source_still_blocks():
    _, blockers = validate_rows([
        {"evidence_source": "mystery", "min_weight": 0.55, "max_weight": 0.55, "n": 1}
    ])
    assert blockers == ["unknown_evidence_source:mystery"]
