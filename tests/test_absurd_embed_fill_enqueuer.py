from __future__ import annotations

from pathlib import Path
import json
import sys


def test_embed_fill_enqueuer_build_job_rows_uses_canonical_payload_json():
    from scripts import absurd_embed_fill_enqueuer as enq

    rows = enq.build_job_rows(total_null=501, batch_size=500)

    assert rows == [
        (
            "embed_fill_batch:next_null:v3_quality:500:0",
            '{"batch_index":0,"job_kind":"embed_fill_batch","limit":500,"quality_gate":"readable_text_only","selection":"next_null"}',
        ),
        (
            "embed_fill_batch:next_null:v3_quality:500:1",
            '{"batch_index":1,"job_kind":"embed_fill_batch","limit":500,"quality_gate":"readable_text_only","selection":"next_null"}',
        ),
    ]


def test_embed_fill_enqueuer_can_cap_planned_jobs():
    from scripts import absurd_embed_fill_enqueuer as enq

    rows = enq.build_job_rows(total_null=2500, batch_size=500, max_jobs=2)

    assert len(rows) == 2
    assert rows[0][0] == "embed_fill_batch:next_null:v3_quality:500:0"
    assert rows[1][0] == "embed_fill_batch:next_null:v3_quality:500:1"


def test_embed_fill_enqueuer_json_dry_run_writes_receipt_without_queue_writes(monkeypatch, tmp_path, capsys):
    from scripts import absurd_embed_fill_enqueuer as enq

    monkeypatch.setattr(enq, "count_null_embeddings", lambda: 1200)
    monkeypatch.setattr(enq.psycopg2, "connect", lambda *a, **k: (_ for _ in ()).throw(AssertionError("dry-run must not connect state DB")))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "absurd_embed_fill_enqueuer.py",
            "--batch-size",
            "500",
            "--max-jobs",
            "2",
            "--dry-run",
            "--receipt-dir",
            str(tmp_path),
            "--json",
        ],
    )

    assert enq.main() == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["schema"] == "lucidota.embed_fill_enqueuer.v1"
    assert payload["dry_run"] is True
    assert payload["total_null"] == 1200
    assert payload["jobs_planned"] == 2
    assert payload["jobs_enqueued"] == 0
    assert payload["receipt_path"].endswith(".json")
    assert Path(payload["receipt_path"]).exists()


def test_embed_fill_enqueuer_uses_small_next_null_jobs_not_offsets():
    source = Path("scripts/absurd_embed_fill_enqueuer.py").read_text(encoding="utf-8")

    assert 'LUCIDOTA_EMBED_FILL_BATCH_SIZE' in source
    assert '"selection": "next_null"' in source
    assert '"required_fields":["limit"]' in source
    assert "embed_fill_batch:next_null:v3_quality" in source
    assert "embedding_quality_sql_where" in source
    assert "offset:" not in source
    assert "retire_legacy_offset_jobs" in source
    assert "status='cancelled'" in source
    assert "payload ? 'offset'" in source
