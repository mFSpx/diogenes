#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_rete_bandit_cli_routes_inline_packet_without_graph_writes() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/rete_bandit_gate_cli.py",
            "--text",
            "CLAIM EVIDENCE EVENT TIME 2026-05-20 mtime_snapshot_v1",
            "--ontology-term",
            "CLAIM",
            "--ontology-term",
            "EVIDENCE",
            "--ontology-term",
            "TIME",
            "--source-ref",
            "pytest-inline",
            "--no-receipt",
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=5,
    )
    assert proc.returncode == 0, proc.stderr
    payload = next(json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{"))
    assert payload["schema"] == "lucidota.rete_bandit_gate_cli.receipt.v1"
    assert payload["canonical_graph_writes_performed"] is False
    assert payload["decision"]["execution_status"] == "succeeded"
    assert payload["decision"]["parallel_engine_plan"]["outbound_state"] == "draft_only"
    assert "treelite_date_router" in payload["decision"]["algorithm_pool"]
