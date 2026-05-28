from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def report(stdout: str, root: Path = ROOT) -> dict:
    for line in stdout.splitlines():
        if line.startswith("REPORT_PATH="):
            p = Path(line.split("=",1)[1])
            return json.loads((root / p if not p.is_absolute() else p).read_text())
    raise AssertionError(stdout)

def test_default_canonical_graph_write_scanner_passes():
    p = subprocess.run([sys.executable, "scripts/canonical_graph_write_scanner.py"], cwd=ROOT, text=True, capture_output=True)
    assert p.returncode == 0, p.stdout + p.stderr
    r = report(p.stdout)
    assert r["verdict"] == "PASS"
    assert not r["blockers"]

def test_direct_graph_write_outside_allowlist_fails(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "bad.py").write_text("cur.execute('INSERT INTO lucidota_go.graph_item(term) VALUES (\\'x\\')')\n")
    allow = tmp_path / "allow.json"
    allow.write_text(json.dumps({"allowed_materialization_helpers": [], "staging_only_modules": [], "test_fixture_modules": [], "legacy_blocked_modules": []}))
    out = tmp_path / "scan.json"
    p = subprocess.run([sys.executable, str(ROOT/"scripts/canonical_graph_write_scanner.py"), "--scan-root", str(tmp_path), "--allowlist", str(allow), "--scan-dir", "scripts", "--output", str(out)], cwd=ROOT, text=True, capture_output=True)
    assert p.returncode == 4
    r = json.loads(out.read_text())
    assert r["blockers"][0]["classification"] == "unknown_danger"
