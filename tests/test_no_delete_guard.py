from scripts import no_delete_guard


def test_no_delete_guard_detects_new_deleted_paths():
    baseline = {"old.md"}
    current = {"old.md", "new.py"}
    assert no_delete_guard.new_deletes(baseline, current) == ["new.py"]


def test_no_delete_guard_parses_git_porcelain_deletes():
    lines = [" D old.md", "D  staged.md", " M keep.py", "?? new.txt"]
    assert no_delete_guard.parse_deleted(lines) == {"old.md", "staged.md"}


def test_no_delete_guard_is_in_recovery_matrix():
    from pathlib import Path
    assert "scripts/no_delete_guard.py check" in Path("scripts/recovery_matrix.py").read_text()
