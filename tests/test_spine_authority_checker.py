from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
REG = ROOT / "00_PROJECT_BRAIN/spine_authority_registry.json"

def test_spine_authority_checker_passes_default_registry():
    p = subprocess.run([sys.executable, "scripts/spine_authority_checker.py"], cwd=ROOT, text=True, capture_output=True)
    assert p.returncode == 0, p.stdout + p.stderr

def test_duplicate_canonical_acceptor_fails(tmp_path):
    data = json.loads(REG.read_text())
    data["roles"].append({"path":"scripts/cep_full_e2e.py", "roles":["canonical_acceptor"], "canonical_acceptor": True, "canonical_graph_gate": False})
    f = tmp_path / "bad_registry.json"; f.write_text(json.dumps(data))
    p = subprocess.run([sys.executable, "scripts/spine_authority_checker.py", "--registry", str(f)], cwd=ROOT, text=True, capture_output=True)
    assert p.returncode == 4
    assert "SPINE_AUTHORITY_CHECK=FAIL" in p.stdout

def run_decision(*args, expect=0):
    p = subprocess.run([sys.executable, "scripts/spine_authority_checker.py", "--check-decision", *args], cwd=ROOT, text=True, capture_output=True)
    assert p.returncode == expect, p.stdout + p.stderr
    return p

def out_report(stdout: str) -> dict:
    for line in stdout.splitlines():
        if line.startswith("REPORT_PATH="):
            return json.loads((ROOT / line.split("=",1)[1]).read_text())
    raise AssertionError(stdout)

def test_valid_authority_decision_passes():
    r = out_report(run_decision("--authority-class", "operator_authored_assertion", "--effect", "queue_absurd_work_order", "--lane", "conversation_command_work_order", "--evidence-ref", "README.md").stdout)
    assert r["verdict"] == "PASS"
    assert r["authority_decision"]["registry_hash"]

def test_missing_authority_fails():
    r = out_report(run_decision("--effect", "queue_absurd_work_order", "--lane", "conversation_command_work_order", "--evidence-ref", "README.md", expect=4).stdout)
    assert "missing_authority" in r["blockers"]

def test_stale_registry_hash_fails():
    r = out_report(run_decision("--authority-class", "operator_authored_assertion", "--effect", "queue_absurd_work_order", "--lane", "conversation_command_work_order", "--evidence-ref", "README.md", "--expected-registry-hash", "deadbeef", expect=4).stdout)
    assert "stale_registry_hash" in r["blockers"]

def test_wrong_lane_fails():
    r = out_report(run_decision("--authority-class", "deterministic_metric", "--effect", "stage_graph_packet", "--lane", "conversation_command_work_order", "--evidence-ref", "metric.json", expect=4).stdout)
    assert "lane_not_permitted_for_authority" in r["blockers"]

def test_broader_effect_fails():
    r = out_report(run_decision("--authority-class", "operator_authored_assertion", "--effect", "materialize_canonical_graph", "--lane", "graph_promotion_execute", "--evidence-ref", "README.md", expect=4).stdout)
    assert "effect_broader_than_registry_permits" in r["blockers"]

def test_explicit_operator_override_passes_without_expansion():
    r = out_report(run_decision("--authority-class", "operator_authored_assertion", "--effect", "operator_override", "--lane", "operator_override", "--evidence-ref", "operator_note", "--operator-override").stdout)
    assert r["verdict"] == "PASS"
