from pathlib import Path
import json
import subprocess
import sys


def test_mirror_classifies_and_excludes():
    from scripts.diogenes_mirror import classify, is_excluded

    assert classify(Path("scripts/foo.py"))[0] == "experimental"
    assert classify(Path("06_SCHEMA/001_lucidota_control.sql"))[0] == "db-native"
    assert classify(Path("scripts/legacy/dead.py"))[0] == "quarantine"
    assert is_excluded(Path("05_OUTPUTS/big.bin")).startswith("dir:") or is_excluded(Path("05_OUTPUTS/big.bin")).startswith("suffix:")


def test_mirror_dry_run_writes_receipt(tmp_path):
    root = tmp_path / "root"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts" / "example.py").write_text('"""Example script."""\n', encoding="utf-8")
    (root / "06_SCHEMA").mkdir()
    (root / "06_SCHEMA" / "001_example.sql").write_text("-- PURPOSE: example schema\n", encoding="utf-8")
    (root / "logs").mkdir()
    (root / "logs" / "exhaust.log").write_text("skip me\n", encoding="utf-8")
    out = tmp_path / "mirror.json"
    cmd = [sys.executable, "scripts/diogenes_mirror.py", "--dry-run", "--json-out", str(out), "--root", str(root)]
    res = subprocess.run(cmd, cwd=Path("."), capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    data = json.loads(out.read_text())
    assert data["schema"] == "lucidota.diogenes_mirror.receipt.v1"
    assert data["included_count"] >= 2
    assert data["excluded_count"] >= 1
