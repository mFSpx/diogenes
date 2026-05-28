import subprocess
import sys


def test_dev_library_scan_query_wraps_legacy_manifest_without_old_name_in_help():
    result = subprocess.run(
        [sys.executable, "scripts/dev_library_scan.py", "--query", "slop"],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "scripts/slop_audit_law.py" in result.stdout
