from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "tests/fixtures/golden_path/valid_receipt_bundle.json"

def report_from(stdout: str) -> dict:
    for line in stdout.splitlines():
        if line.startswith("REPORT_PATH="):
            p = Path(line.split("=",1)[1])
            return json.loads((ROOT / p if not p.is_absolute() else p).read_text())
    raise AssertionError(stdout)

def run_gate(receipt: Path | None = None):
    cmd = [sys.executable, "scripts/golden_path_regression_gate.py", "--dry-run"]
    if receipt: cmd += ["--receipt", str(receipt)]
    return subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)

def test_golden_path_regression_gate_dry_run_passes():
    p = run_gate()
    assert p.returncode == 0, p.stdout + p.stderr
    r = report_from(p.stdout)
    assert r["verdict"] == "PASS"
    assert r["same_lineage"] is True
    assert r["canonical_graph_writes_performed"] is False
    assert r["receipt_contract"] == "lucidota.lineage.receipt_bundle.v1"

def test_golden_path_regression_gate_rejects_missing_id(tmp_path):
    data = json.loads(BUNDLE.read_text())
    data.pop("lineage_id", None)
    f = tmp_path / "bad.json"; f.write_text(json.dumps(data))
    p = run_gate(f)
    assert p.returncode != 0
    r = report_from(p.stdout)
    assert "missing_lineage_id" in r["blockers"] or "same_lineage_validator_failed" in r["blockers"]

def test_golden_path_regression_gate_rejects_receipt_collage(tmp_path):
    f = tmp_path / "collage.json"
    f.write_text(json.dumps({"schema":"lucidota.golden_path.single_instruction.v1", "receipts":["loose.json"]}))
    p = run_gate(f)
    assert p.returncode != 0
    assert "unsupported_receipt_contract_schema" in report_from(p.stdout)["blockers"]

def test_golden_path_regression_gate_rejects_graph_mutation(tmp_path):
    data = json.loads(BUNDLE.read_text())
    data["canonical_graph_counts"]["after"]["graph_item"] = 1
    f = tmp_path / "mutated.json"; f.write_text(json.dumps(data))
    p = run_gate(f)
    assert p.returncode != 0
    r = report_from(p.stdout)
    assert "canonical_graph_counts_changed_fixture" in r["blockers"] or "same_lineage_validator_failed" in r["blockers"]

def test_repeat_run_lineage_shape_is_deterministic():
    a = report_from(subprocess.run([sys.executable, "scripts/same_lineage_validator.py", "--receipt", str(BUNDLE)], cwd=ROOT, text=True, capture_output=True, check=True).stdout)
    b = report_from(subprocess.run([sys.executable, "scripts/same_lineage_validator.py", "--receipt", str(BUNDLE)], cwd=ROOT, text=True, capture_output=True, check=True).stdout)
    assert a["receipt_hashes"] == b["receipt_hashes"]
    assert a["bundle_hash"] == b["bundle_hash"]
    assert a["command_uuid"] == b["command_uuid"]
