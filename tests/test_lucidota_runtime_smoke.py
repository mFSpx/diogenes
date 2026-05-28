from scripts import lucidota_runtime_smoke as smoke
from pathlib import Path


def test_runtime_smoke_checks_repo_absurd_spine_not_missing_package():
    assert "absurd" not in smoke.IMPORTS
    assert "dbos" in smoke.IMPORTS
    assert "absurd_queue_spine" in smoke.IMPORTS


def test_recovery_matrix_includes_runtime_smoke():
    assert "lucidota_runtime_smoke.py" in Path("scripts/recovery_matrix.py").read_text()
