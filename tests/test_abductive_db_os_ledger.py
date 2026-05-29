from __future__ import annotations

import json
from pathlib import Path


def test_ledger_normalizes_model_audit_and_hunch_receipts(tmp_path: Path) -> None:
    from scripts.abductive_db_os_ledger import normalize_receipt, summarize_rows

    audit = tmp_path / "model_invocation_audit.json"
    audit.write_text(
        json.dumps(
            {
                "schema": "lucidota.model_invocation_audit.v1",
                "verdict": "FAIL",
                "status": "FAIL",
                "five_task_audit_blocks": [{"block_id": "task_block_0001", "audit_status": "MISSING_VALID_AUDIT_OUTPUT"}],
                "missing_dedicated_model_audit_blocks": 1,
            }
        ),
        encoding="utf-8",
    )
    hunch = tmp_path / "hunch_postgres_ingest.json"
    hunch.write_text(
        json.dumps(
            {
                "schema": "lucidota.hunch_postgres_ingest.receipt.v1",
                "records_upserted": 93,
                "graph_candidates_written": 93,
                "canonical_graph_writes_performed": False,
            }
        ),
        encoding="utf-8",
    )

    rows = [normalize_receipt(audit), normalize_receipt(hunch)]
    board = summarize_rows(rows)
    assert board["object_counts"]["Receipt"] == 2
    assert board["object_counts"]["ModelAuditBlock"] == 1
    assert board["object_counts"]["GraphCandidate"] == 93
    assert any("model audit" in b["summary"].lower() for b in board["open_blockers"])

