from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "tests/fixtures/golden_path/valid_receipt_bundle.json"

def run(cmd, expect=0):
    p = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    assert p.returncode == expect, (cmd, p.returncode, p.stdout, p.stderr)
    return p

def report(stdout: str) -> dict:
    for line in stdout.splitlines():
        if line.startswith("REPORT_PATH="):
            return json.loads((ROOT / line.split("=",1)[1]).read_text())
    raise AssertionError(stdout)

def write_bundle(tmp_path: Path, mutator) -> Path:
    data = json.loads(BUNDLE.read_text())
    mutator(data)
    p = tmp_path / "bundle.json"
    p.write_text(json.dumps(data))
    return p

def test_valid_hash_linked_bundle_passes():
    p = run([sys.executable, "scripts/same_lineage_validator.py", "--receipt", str(BUNDLE)])
    r = report(p.stdout)
    assert r["verdict"] == "PASS"
    assert r["same_lineage"] is True
    assert len(r["receipt_hashes"]) == 7

def test_missing_lineage_fails(tmp_path):
    f = write_bundle(tmp_path, lambda d: d.pop("lineage_id", None))
    p = run([sys.executable, "scripts/same_lineage_validator.py", "--receipt", str(f)], expect=4)
    assert "missing_lineage_id" in report(p.stdout)["blockers"]

def test_missing_parent_fails(tmp_path):
    def m(d): d["receipts"][2]["parent_receipt_hash"] = "bad"
    f = write_bundle(tmp_path, m)
    r = report(run([sys.executable, "scripts/same_lineage_validator.py", "--receipt", str(f)], expect=4).stdout)
    assert any("parent_receipt_hash_mismatch" in b for b in r["blockers"])

def test_swapped_receipt_fails(tmp_path):
    def m(d): d["receipts"][3], d["receipts"][4] = d["receipts"][4], d["receipts"][3]
    f = write_bundle(tmp_path, m)
    r = report(run([sys.executable, "scripts/same_lineage_validator.py", "--receipt", str(f)], expect=4).stdout)
    assert "receipt_type_order_mismatch" in r["blockers"] or any("parent_receipt_hash_mismatch" in b for b in r["blockers"])

def test_future_evidence_fails(tmp_path):
    def m(d): d["receipts"][-1]["evidence_refs"][0]["created_at"] = "2999-01-01T00:00:00Z"
    f = write_bundle(tmp_path, m)
    r = report(run([sys.executable, "scripts/same_lineage_validator.py", "--receipt", str(f)], expect=4).stdout)
    assert any("future_evidence" in b for b in r["blockers"])

def test_duplicate_id_fails(tmp_path):
    def m(d): d["receipts"][1]["receipt_uuid"] = d["receipts"][0]["receipt_uuid"]
    f = write_bundle(tmp_path, m)
    r = report(run([sys.executable, "scripts/same_lineage_validator.py", "--receipt", str(f)], expect=4).stdout)
    assert any("duplicate_receipt_uuid" in b for b in r["blockers"])

def test_unrelated_child_report_collage_fails(tmp_path):
    collage = {"schema":"lucidota.golden_path.single_instruction.v1", "verdict":"REAL", "receipts":["a.json", "b.json"]}
    f = tmp_path / "collage.json"; f.write_text(json.dumps(collage))
    r = report(run([sys.executable, "scripts/same_lineage_validator.py", "--receipt", str(f)], expect=4).stdout)
    assert "missing_receipt_bundle_hash_chain" in r["blockers"]
