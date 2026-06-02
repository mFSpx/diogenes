from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_graph_promotion_materialize_json_stdout_is_pure():
    proc = subprocess.run(
        [
            str(ROOT / ".venv" / "bin" / "python"),
            str(ROOT / "scripts" / "graph_promotion_materialize.py"),
            "--json",
            "materialize",
            "--command-envelope-uuid",
            "a38fd6b2-9418-4742-b2f0-a0fc70b5b4eb",
            "--candidate-payload-json",
            '{"term":"ENTITY","label":"json contract","status":"staged"}',
            "--evidence-ref",
            "05_OUTPUTS/graph/graph_promotion_full_e2e_20260601T175205695814Z.json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(proc.stdout)
    assert payload["action"] == "materialize"
    assert proc.stdout.lstrip().startswith("{")
    assert proc.stdout.rstrip().endswith("}")
    assert "REPORT_PATH=" not in proc.stdout
    assert proc.stderr == "" or "WARNING:" in proc.stderr


def test_graph_edge_materialize_json_stdout_is_pure():
    proc = subprocess.run(
        [
            str(ROOT / ".venv" / "bin" / "python"),
            str(ROOT / "scripts" / "graph_edge_materialize.py"),
            "--json",
            "--command-envelope-uuid",
            "a38fd6b2-9418-4742-b2f0-a0fc70b5b4eb",
            "--evidence-ref",
            "05_OUTPUTS/graph/graph_promotion_full_e2e_20260601T175205695814Z.json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(proc.stdout)
    assert payload["action"] == "edge_materialize"
    assert proc.stdout.lstrip().startswith("{")
    assert proc.stdout.rstrip().endswith("}")
    assert "REPORT_PATH=" not in proc.stdout
    assert proc.stderr == "" or "WARNING:" in proc.stderr
