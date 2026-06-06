from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SQL_ROOT = ROOT / "06_SCHEMA"


FORBIDDEN_PATTERNS = [
    "./luci",
    ".venv/bin/python scripts/",
    "curl the live route",
    "luci doctor --json",
    "luci status --json",
    "luci capability list --json",
]


def test_sql_surfaces_do_not_emit_hardcoded_shell_command_rails() -> None:
    offenders: list[str] = []
    for path in sorted(SQL_ROOT.glob("*.sql")):
        text = path.read_text(encoding="utf-8")
        for pattern in FORBIDDEN_PATTERNS:
            if pattern in text:
                offenders.append(f"{path}:{pattern}")

    assert not offenders, f"forbidden SQL command rails found: {offenders}"
