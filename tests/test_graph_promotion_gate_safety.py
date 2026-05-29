from __future__ import annotations
import json, subprocess, sys, uuid
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def run(cmd, expect=0):
    p = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    assert p.returncode == expect, (cmd, p.returncode, p.stdout, p.stderr)
    return p

def report(stdout: str) -> dict:
    for line in stdout.splitlines():
        if line.startswith("REPORT_PATH="):
            return json.loads((ROOT / line.split("=",1)[1]).read_text())
    raise AssertionError(stdout)

def test_graph_gate_valid_dry_run_no_db_writes():
    p = run([sys.executable, "scripts/graph_promotion_gate.py", "gate", "--dry-run", "--candidate-payload-json", '{"label":"dry"}', "--evidence-ref", "README.md", "--authority-class", "operator_authored_assertion"])
    r = report(p.stdout)
    assert r["dry_run"] is True
    assert r["db_writes_performed"] is False
    assert r["canonical_graph_writes_performed"] is False

def test_graph_gate_dry_run_materialize_refuses():
    p = run([sys.executable, "scripts/graph_promotion_gate.py", "gate", "--dry-run", "--materialize", "--candidate-payload-json", '{"label":"bad"}', "--evidence-ref", "README.md", "--authority-class", "operator_authored_assertion"], expect=2)
    assert "dry_run_materialize_refused" in json.dumps(report(p.stdout))

def test_graph_gate_execute_no_materialize_no_canonical_write():
    p = run([sys.executable, "scripts/graph_promotion_gate.py", "gate", "--execute", "--candidate-payload-json", '{"label":"execute_no_materialize_' + str(uuid.uuid4()) + '"}', "--evidence-ref", "README.md", "--authority-class", "operator_authored_assertion", "--decision", "defer"])
    r = report(p.stdout)
    assert r["db_writes_performed"] is True
    assert r["canonical_graph_writes_performed"] is False
    assert r["graph_writes_performed"] is False

def test_graph_gate_bogus_authority_fails():
    p = run([sys.executable, "scripts/graph_promotion_gate.py", "gate", "--dry-run", "--candidate-payload-json", '{"label":"bad"}', "--evidence-ref", "README.md", "--authority-class", "bogus"], expect=2)
    assert "invalid_authority_class" in json.dumps(report(p.stdout))

def test_boring_beast_direct_graph_materialize_default_refuses():
    p = run([sys.executable, "scripts/boring_beast.py", "graph-promote", "--execute", "--materialize", "--candidate-payload-json", '{"term":"CLAIM","label":"blocked"}', "--evidence-ref", "README.md", "--operator-confirmed", "--command-envelope-uuid", str(uuid.uuid4())], expect=2)
    assert "legacy_direct_canonical_graph_materialization_disabled" in json.dumps(report(p.stdout))
