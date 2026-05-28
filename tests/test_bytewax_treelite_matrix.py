#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_bytewax_hint_contains_26_treelite_router_scores():
    import lucidota_bytewax_mini as bw

    event = {
        "source": "operator",
        "phase": "investigation",
        "status": "succeeded",
        "workflow_id": "wf",
        "detail": {"decision": "wire treelite matrix", "risk": "fallback fraud"},
    }
    hint = bw.hint(event)

    assert hint["detail"]["treelite_router_count"] == 26
    assert len(hint["detail"]["treelite_router_scores"]) == 26
    assert "treelite_matrix_score" in hint["detail"]
