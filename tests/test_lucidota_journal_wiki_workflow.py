from __future__ import annotations

import json
import subprocess
import urllib.request
from pathlib import Path

from scripts import indy_reads


LIVE_BASE_URL = "http://127.0.0.1:3000"


def test_indy_journal_and_wiki_writers_create_files(tmp_path: Path) -> None:
    journal = indy_reads.write_journal_entry(
        title="Indy thought log",
        body="write this now",
        kind="note",
        journal_dir=tmp_path / "private_journal",
    )
    wiki = indy_reads.write_wiki_page(
        title="Indy thought log",
        body="wiki this now",
        wiki_dir=tmp_path / "wiki" / "pages",
    )

    assert journal["schema"] == "lucidota.indy_reads.journal_entry.v1"
    assert wiki["schema"] == "lucidota.indy_reads.wiki_page.v1"
    assert Path(journal["abs_path"]).exists()
    assert Path(wiki["abs_path"]).exists()


def test_indy_journal_wiki_and_backup_workflows_are_registered() -> None:
    with urllib.request.urlopen(f"{LIVE_BASE_URL}/workflow_registry?workflow_name=eq.indy-journal-wiki", timeout=5) as resp:
        journal_rows = json.loads(resp.read().decode("utf-8"))
    with urllib.request.urlopen(f"{LIVE_BASE_URL}/workflow_registry?workflow_name=eq.indy-daily-backup", timeout=5) as resp:
        backup_rows = json.loads(resp.read().decode("utf-8"))

    assert journal_rows and journal_rows[0]["status"] == "active"
    assert "journal" in journal_rows[0]["notes"].lower()
    assert backup_rows and backup_rows[0]["status"] == "active"
    assert backup_rows[0]["command"] == "scripts/lucidota_daily_backup.sh"


def test_wiki_query_covers_indy_journal_paths() -> None:
    proc = subprocess.run(
        [".venv/bin/python", "scripts/lucidota_wiki_query.py", "Indy journal wiki", "--json", "--limit", "2"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    searched = set(payload["searched_dirs"])
    assert "BOOKS/.indy_reads/wiki" in searched
    assert "BOOKS/.indy_reads/private_journal" in searched
