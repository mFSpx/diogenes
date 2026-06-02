from __future__ import annotations

import json
from pathlib import Path


def test_root_rotor_audit_dump_excludes_bulk_and_truncates_active_sources(tmp_path: Path) -> None:
    import scripts.root_rotor_audit_dump as dump

    (tmp_path / "scripts").mkdir()
    active = tmp_path / "scripts" / "worker.py"
    active.write_text("a" * 120, encoding="utf-8")
    small = tmp_path / "AGENTS.md"
    small.write_text("agent law", encoding="utf-8")
    for excluded_dir in ["KRAMPUSCHEWING", "05_OUTPUTS", ".venv", ".git", "__pycache__", "03_VAULT"]:
        d = tmp_path / excluded_dir
        d.mkdir()
        (d / "skip.py").write_text("skip", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"not source")
    out = tmp_path / "GOALS" / "audit.txt"

    result = dump.write_audit_dump(tmp_path, out, max_bytes=100)

    assert result["schema"] == "lucidota.root_rotor.audit_dump.v1"
    assert result["files_written"] == 2
    assert result["excluded_dirs"] == dump.DEFAULT_EXCLUDED_DIRS
    assert result["max_bytes_per_file"] == 100
    text = out.read_text(encoding="utf-8")
    assert "=== FILE: scripts/worker.py ===" in text
    assert "=== FILE: AGENTS.md ===" in text
    assert "KRAMPUSCHEWING" not in text
    worker_entry = next(item for item in result["files"] if item["path"] == "scripts/worker.py")
    assert worker_entry["truncated"] is True
    assert worker_entry["bytes_read"] == 100
    assert len(worker_entry["sha256"]) == 64
    manifest_path = Path(result["manifest_path"])
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["files_written"] == 2


def test_root_rotor_audit_dump_excludes_other_repos_but_keeps_claw_luci(tmp_path: Path) -> None:
    import scripts.root_rotor_audit_dump as dump

    active_script = tmp_path / "scripts" / "active.py"
    active_script.parent.mkdir()
    active_script.write_text("print('active')", encoding="utf-8")
    other_repo = tmp_path / "01_REPOS" / "PocketFlow" / "pocketflow" / "__init__.py"
    other_repo.parent.mkdir(parents=True)
    other_repo.write_text("# external repo", encoding="utf-8")
    claw_luci = tmp_path / "01_REPOS" / "claudecode" / "rust" / "crates" / "claw-cli" / "src" / "main.rs"
    claw_luci.parent.mkdir(parents=True)
    claw_luci.write_text("fn main() {}", encoding="utf-8")

    out = tmp_path / "GOALS" / "audit.txt"
    result = dump.write_audit_dump(tmp_path, out, max_bytes=100)

    paths = {item["path"] for item in result["files"]}
    assert "scripts/active.py" in paths
    assert "01_REPOS/claudecode/rust/crates/claw-cli/src/main.rs" in paths
    assert "01_REPOS/PocketFlow/pocketflow/__init__.py" not in paths
    text = out.read_text(encoding="utf-8")
    assert "01_REPOS/claudecode/rust/crates/claw-cli/src/main.rs" in text
    assert "PocketFlow" not in text


def test_root_rotor_audit_dump_skips_symlinks_that_resolve_outside_repo(tmp_path: Path) -> None:
    import scripts.root_rotor_audit_dump as dump

    outside = tmp_path.parent / "outside_root_rotor_target.py"
    outside.write_text("print('outside')", encoding="utf-8")
    link = tmp_path / "outside_link.py"
    link.symlink_to(outside)
    active = tmp_path / "active.py"
    active.write_text("print('active')", encoding="utf-8")

    result = dump.write_audit_dump(tmp_path, tmp_path / "GOALS" / "audit.txt", max_bytes=100)

    paths = {item["path"] for item in result["files"]}
    assert "active.py" in paths
    assert "outside_link.py" not in paths


def test_root_rotor_audit_dump_uses_current_roots_not_lab_or_session_bulk(tmp_path: Path) -> None:
    import scripts.root_rotor_audit_dump as dump

    keep = tmp_path / "00_PROJECT_BRAIN" / "KANT69.md"
    keep.parent.mkdir()
    keep.write_text("canon", encoding="utf-8")
    keep_script = tmp_path / "scripts" / "worker.py"
    keep_script.parent.mkdir()
    keep_script.write_text("print('worker')", encoding="utf-8")
    keep_algo = tmp_path / "ALGOS" / "bandit_router.py"
    keep_algo.parent.mkdir()
    keep_algo.write_text("# active reusable algo", encoding="utf-8")
    evolved = tmp_path / "ALGOS" / "evolved" / "generated.py"
    evolved.parent.mkdir(parents=True)
    evolved.write_text("# generated lab", encoding="utf-8")
    session = tmp_path / ".claw" / "sessions" / "session.json"
    session.parent.mkdir(parents=True)
    session.write_text("{}", encoding="utf-8")
    runtime_repo = tmp_path / "04_RUNTIME" / "models" / "repo" / "README.md"
    runtime_repo.parent.mkdir(parents=True)
    runtime_repo.write_text("external model repo", encoding="utf-8")

    result = dump.write_audit_dump(tmp_path, tmp_path / "GOALS" / "audit.txt", max_bytes=100)
    paths = {item["path"] for item in result["files"]}

    assert "00_PROJECT_BRAIN/KANT69.md" in paths
    assert "scripts/worker.py" in paths
    assert "ALGOS/bandit_router.py" in paths
    assert "ALGOS/evolved/generated.py" not in paths
    assert ".claw/sessions/session.json" not in paths
    assert "04_RUNTIME/models/repo/README.md" not in paths


def test_root_rotor_audit_dump_excludes_clean_nested_repos(tmp_path: Path) -> None:
    import scripts.root_rotor_audit_dump as dump

    phantom_file = tmp_path / "phantom" / "src" / "persona.py"
    phantom_file.parent.mkdir(parents=True)
    (tmp_path / "phantom" / ".git").mkdir()
    phantom_file.write_text("persona = {}", encoding="utf-8")
    active = tmp_path / "scripts" / "active.py"
    active.parent.mkdir()
    active.write_text("print('active')", encoding="utf-8")

    result = dump.write_audit_dump(tmp_path, tmp_path / "GOALS" / "audit.txt", max_bytes=100)
    paths = {item["path"] for item in result["files"]}

    assert "scripts/active.py" in paths
    assert "phantom/src/persona.py" not in paths


def test_root_rotor_audit_dump_includes_dirty_nested_repos(tmp_path: Path) -> None:
    import subprocess
    import scripts.root_rotor_audit_dump as dump

    dirty_repo = tmp_path / "phantom"
    dirty_file = dirty_repo / "src" / "persona.py"
    dirty_file.parent.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=dirty_repo, check=True, capture_output=True, text=True)
    dirty_file.write_text("persona = {}", encoding="utf-8")

    result = dump.write_audit_dump(tmp_path, tmp_path / "GOALS" / "audit.txt", max_bytes=100)
    paths = {item["path"] for item in result["files"]}

    assert "phantom/src/persona.py" in paths


def test_root_rotor_audit_dump_excludes_dirty_nested_repos_in_excluded_dirs(tmp_path: Path) -> None:
    import subprocess
    import scripts.root_rotor_audit_dump as dump

    dirty_repo = tmp_path / "03_VAULT" / "bad_actor"
    dirty_file = dirty_repo / "notes.py"
    dirty_file.parent.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=dirty_repo, check=True, capture_output=True, text=True)
    dirty_file.write_text("persona = {}", encoding="utf-8")

    result = dump.write_audit_dump(tmp_path, tmp_path / "GOALS" / "audit.txt", max_bytes=100)
    paths = {item["path"] for item in result["files"]}

    assert "03_VAULT/bad_actor/notes.py" not in paths
    assert not any(prefix.startswith("03_VAULT/") for prefix in result["included_prefixes"])
    assert not any(prefix.startswith("03_VAULT/") for prefix in result["dirty_nested_repo_prefixes"])


def test_root_rotor_audit_dump_keeps_explicitly_allowed_dirty_nested_repo(tmp_path: Path) -> None:
    import subprocess
    import scripts.root_rotor_audit_dump as dump

    dirty_repo = tmp_path / "01_REPOS" / "forge"
    dirty_file = dirty_repo / "src" / "guardian.py"
    dirty_file.parent.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=dirty_repo, check=True, capture_output=True, text=True)
    dirty_file.write_text("pass", encoding="utf-8")

    result = dump.write_audit_dump(
        tmp_path,
        tmp_path / "GOALS" / "audit.txt",
        max_bytes=100,
        included_prefixes=["01_REPOS/forge/"],
    )
    paths = {item["path"] for item in result["files"]}

    assert "01_REPOS/forge/src/guardian.py" in paths
    assert "01_REPOS/forge/" in result["dirty_nested_repo_prefixes"]
    assert "01_REPOS/forge/" in result["included_prefixes"]


def test_root_rotor_audit_dump_skips_paths_that_disappear_between_scan_and_read(monkeypatch, tmp_path: Path) -> None:
    import scripts.root_rotor_audit_dump as dump

    missing = tmp_path / "scripts" / "missing.py"
    missing.parent.mkdir()
    missing.write_text("gone", encoding="utf-8")
    active = tmp_path / "scripts" / "active.py"
    active.write_text("print('active')", encoding="utf-8")

    monkeypatch.setattr(dump, "iter_active_sources", lambda *args, **kwargs: [missing, active])
    missing.unlink()

    result = dump.write_audit_dump(tmp_path, tmp_path / "GOALS" / "audit.txt", max_bytes=100)
    paths = {item["path"] for item in result["files"]}

    assert paths == {"scripts/active.py"}


def test_root_rotor_audit_dump_includes_common_active_code_and_config_extensions(tmp_path: Path) -> None:
    import scripts.root_rotor_audit_dump as dump

    files = [
        tmp_path / "config" / "service.yaml",
        tmp_path / "config" / "service.yml",
        tmp_path / "config" / "service.ini",
        tmp_path / "phantom" / "src" / "runtime.ts",
        tmp_path / "phantom" / "src" / "view.tsx",
        tmp_path / "phantom" / "scripts" / "worker.js",
        tmp_path / "docs" / "notes.txt",
    ]
    (tmp_path / "phantom" / ".git").mkdir(parents=True)
    for path in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("active", encoding="utf-8")

    result = dump.write_audit_dump(tmp_path, tmp_path / "GOALS" / "audit.txt", max_bytes=100)
    paths = {item["path"] for item in result["files"]}

    assert "config/service.yaml" in paths
    assert "config/service.yml" in paths
    assert "config/service.ini" in paths
    assert "docs/notes.txt" in paths
    # Phantom is a clean nested repo in this fixture, so its TS/JS files stay excluded.
    assert "phantom/src/runtime.ts" not in paths
    assert "phantom/scripts/worker.js" not in paths
