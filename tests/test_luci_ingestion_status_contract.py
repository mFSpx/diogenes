#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def test_luci_ingest_status_reports_repo_wide_completion_contract_json_only():
    proc = subprocess.run(
        [str(ROOT / "luci"), "ingest", "status", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert proc.stdout.lstrip().startswith("{")
    assert proc.stdout.rstrip().endswith("}")
    assert proc.stderr == "" or "WARNING:" in proc.stderr

    payload = json.loads(proc.stdout)
    assert payload["schema"] == "lucidota.ingestion_completion_contract.v1"
    assert payload["status"] in {"DONE", "IN_PROGRESS", "BLOCKED"}
    assert payload["done"] is (payload["status"] == "DONE")
    assert payload["false_victory_guard"] == "per-lane evidence only; no full-ingestion claim unless all required lanes are done"
    assert payload["receipt_path"].startswith("05_OUTPUTS/ingestion_status/")
    assert payload["db_counts"]["corpus_chunk"] >= 0
    assert payload["db_counts"]["markdown_artifact"] >= 0
    assert "markdown_archive" in payload["contracts"]
    assert payload["contracts"]["markdown_archive"]["active_candidates"] >= 0
    assert "krampus_top_level_documents" in payload["contracts"]
    assert "krampus_archive_members" in payload["contracts"]
    assert "embedding_backlog" in payload["contracts"]
    assert payload["contracts"]["krampus_top_level_documents"]["required_evidence"]
    assert payload["contracts"]["krampus_archive_members"]["required_evidence"]
    assert "archives are separate lanes" not in json.dumps(payload)
    assert payload["contracts"]["embedding_backlog"]["required_evidence"]
    assert "enqueue_receipt_count" in payload["contracts"]["embedding_backlog"]
    assert "latest_enqueue_receipt" in payload["contracts"]["embedding_backlog"]
    assert payload["contracts"]["embedding_backlog"]["quality_gate_error"] == ""
    assert "embedding_quality_status" in payload["contracts"]["embedding_backlog"]["quality_gate_sql"]
    assert isinstance(payload["next_actions"], list)
    assert payload["visible_response"]["work_order_id"]
    assert payload["visible_response"]["work_receipt_id"]


def test_krampus_archive_status_excludes_c_archive_email_lane(monkeypatch, tmp_path):
    import scripts.luci_ingestion_status as status

    krampus = tmp_path / "KRAMPUSCHEWING"
    krampus.mkdir()
    (krampus / "C_ARCHIVE.zip").write_bytes(b"email archive bytes")
    (krampus / "active.zip").write_bytes(b"active archive bytes")
    (krampus / "pending.zip").write_bytes(b"pending archive bytes")
    receipts = tmp_path / "05_OUTPUTS" / "receipts"
    receipts.mkdir(parents=True)
    (receipts / "krampus_archive_active_1.json").write_text(
        json.dumps({"status": "PASS", "dry_run": True, "members_seen": 4, "nested_archives_opened": 0})
    )

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def execute(self, *_args, **_kwargs):
            return None

        def fetchall(self):
            return [
                ("C_ARCHIVE.zip", 132769),
                ("KRAMPUSCHEWING/active.zip", 3),
            ]

    monkeypatch.setattr(status, "ROOT", tmp_path)
    payload = status.krampus_archive_contract(SimpleNamespace(cursor=lambda: FakeCursor()))

    assert payload["archives_present"] == 2
    assert payload["excluded_special_archive_count"] == 1
    assert payload["excluded_special_archives"] == ["KRAMPUSCHEWING/C_ARCHIVE.zip"]
    assert payload["archive_member_chunks"] == 3
    assert payload["archives_opened_with_chunks"] == 1
    assert payload["pending_unopened_archives"] == 1
    assert payload["latest_receipt_dry_run"] is True
