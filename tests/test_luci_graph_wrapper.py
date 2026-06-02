from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_luci_shell_wrapper_exposes_graph_promote_class():
    proc = subprocess.run(
        [
            str(ROOT / "luci"),
            "graph",
            "promote",
            "--execute",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "GRAPH_PROMOTION_FULL_E2E=PASS" in proc.stdout
    assert "REPORT_PATH=" in proc.stdout
    assert proc.stderr == "" or "WARNING:" in proc.stderr


def test_luci_shell_wrapper_exposes_graph_promote_json_only():
    proc = subprocess.run(
        [
            str(ROOT / "luci"),
            "graph",
            "promote",
            "--execute",
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(proc.stdout)
    assert payload["status"] == "PASS"
    assert proc.stdout.lstrip().startswith("{")
    assert proc.stdout.rstrip().endswith("}")
    assert "GRAPH_PROMOTION_FULL_E2E=" not in proc.stdout
    assert proc.stderr == "" or "WARNING:" in proc.stderr


def test_luci_shell_wrapper_exposes_graph_edge_json_only():
    proc = subprocess.run(
        [
            str(ROOT / "luci"),
            "graph",
            "edge",
            "--execute",
            "--json",
            "--command-envelope-uuid",
            "a38fd6b2-9418-4742-b2f0-a0fc70b5b4eb",
            "--evidence-ref",
            "05_OUTPUTS/graph/graph_promotion_full_e2e_20260601T175707672290Z.json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(proc.stdout)
    assert payload["status"] == "PASS"
    assert payload["graph_writes_performed"] is True
    assert proc.stdout.lstrip().startswith("{")
    assert proc.stdout.rstrip().endswith("}")
    assert "HELPER_RECEIPT_UUID=" not in proc.stdout
    assert proc.stderr == "" or "WARNING:" in proc.stderr


def test_luci_shell_wrapper_exposes_graph_materialize_json_only():
    proc = subprocess.run(
        [
            str(ROOT / "luci"),
            "graph",
            "materialize",
            "--execute",
            "--json",
            "--command-envelope-uuid",
            "a38fd6b2-9418-4742-b2f0-a0fc70b5b4eb",
            "--candidate-payload-json",
            '{"term":"ENTITY","label":"front door materialize smoke","status":"staged"}',
            "--evidence-ref",
            "05_OUTPUTS/graph/graph_promotion_full_e2e_20260601T175707672290Z.json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(proc.stdout)
    assert payload["status"] == "PASS"
    assert payload["canonical_graph_writes_performed"] is True
    assert proc.stdout.lstrip().startswith("{")
    assert proc.stdout.rstrip().endswith("}")
    assert "PACKET_UUID=" not in proc.stdout
    assert proc.stderr == "" or "WARNING:" in proc.stderr
