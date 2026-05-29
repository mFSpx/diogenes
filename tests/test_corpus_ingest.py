from __future__ import annotations

import json
from pathlib import Path


def test_build_selection_skips_huge_and_active_db_risk():
    import scripts.corpus_ingest as ingest

    corpus_map = {"recommendations": {"bypass_or_stage": ["skip.me"]}}
    rows = [
        {"path": "skip.me", "size_bytes": 1, "sha256": "a", "hash_status": "HASHED"},
        {"path": "huge.bin", "size_bytes": ingest.HUGE_FILE_THRESHOLD + 1, "sha256": None, "hash_status": "HASH_SKIPPED_SIZE"},
        {"path": "active.sqlite", "size_bytes": 10, "sha256": None, "hash_status": "HASHED"},
        {"path": "ok.md", "size_bytes": 10, "sha256": "b", "hash_status": "HASHED"},
    ]

    selected, skipped, meta = ingest.build_selection(rows, start_after="", batch_size=5, corpus_map=corpus_map)

    assert [r["path"] for r in selected] == ["ok.md"]
    assert meta["cursor_after"] == "ok.md"
    assert any(item["path"] == "skip.me" for item in skipped)
    assert any(item["path"] == "huge.bin" for item in skipped)


def test_guess_workflows_and_authority_map():
    import scripts.corpus_ingest as ingest

    normalized = {
        "labels": ["lane:GRAPH_STAGING", "kind:markdown"],
        "kind_guess": "markdown",
        "text_status": "PARSED_TEXT",
        "ocr_status": "OCR_SKIPPED",
        "contains_graph_terms": True,
        "normalized_case_guess": "NORTHERN_STRIKE",
        "normalized_dev_project_guess": "LUCIDOTA",
        "contains_instruction_law": True,
    }
    assert ingest.guess_workflows(normalized) == [
        "authority_index_flow",
        "case_attachment_flow",
        "graph_staging_flow",
        "parsed_text_ingest_flow",
        "project_workflow_flow",
    ]

    authority = ingest.build_current_authority_map(
        [
            {"path": "GOALS/CURRENT_HANDOFF.md", "authority_status": "active", "summary": "active law"},
            {"path": "GOALS/old.md", "authority_status": "superseded", "summary": "old law"},
        ],
        {"receipt_path": "05_OUTPUTS/goals/x.json"},
    )
    assert authority["groq_thinker_lane"]["configured"] is True
    assert authority["groq_thinker_lane"]["invoked"] is True
    assert authority["active_laws"][0]["path"] == "GOALS/CURRENT_HANDOFF.md"


def test_local_enforcer_receipt_flags_pass_and_recommendation_state():
    import scripts.corpus_ingest as ingest

    receipt = ingest.make_local_enforcer_receipt(
        run_receipt={
            "verdict": "PASS",
            "canonical_graph_writes": False,
            "db_writes_performed": False,
            "processed": 3,
            "paths": {"cursor": "04_RUNTIME/corpus_ingest/cursor.json"},
            "cursor_after": "foo",
        },
        authority_map={"active_laws": [{"path": "x"}], "historical_artifacts": []},
        groq_receipt={"receipt_path": "05_OUTPUTS/goals/groq.json"},
    )
    assert receipt["verdict"] == "PASS"
    assert receipt["checks"]["no_canonical_graph_writes"] is True
    assert receipt["groq_thinker_runnable"] is True


def test_tool_gap_report_mentions_missing_local_transcription_tools():
    import scripts.corpus_ingest as ingest

    gap = ingest.make_tool_gap_report()
    assert gap["groq_configured"] is True
    assert "ffmpeg" in gap["tool_checks"]
    assert isinstance(gap["missing_local_tools"], list)


def test_parse_and_label_populates_custody_id_for_staging(monkeypatch, tmp_path):
    import scripts.corpus_ingest as ingest

    root = tmp_path
    sample = root / "sample.txt"
    sample.write_text("Alpha one. Beta two.", encoding="utf-8")

    monkeypatch.setattr(ingest, "ROOT", root)
    monkeypatch.setattr(ingest, "inventory_path", lambda row: sample)
    monkeypatch.setattr(ingest, "guess_case", lambda text: None)
    monkeypatch.setattr(ingest, "guess_dev_project", lambda path: None)
    monkeypatch.setattr(ingest, "kind_guess", lambda path, text: "note")
    monkeypatch.setattr(ingest, "lane_guess", lambda *args, **kwargs: "FILE_ORGANIZATION")

    normalized, chunks, staging_rows, ocr_jobs = ingest.parse_and_label(
        {"path": "sample.txt", "inventory_key": "custody-123", "size_bytes": sample.stat().st_size},
        max_chunk_chars=500,
    )

    assert normalized["source_path"].endswith("sample.txt")
    assert chunks, "expected chunking for text file"
    assert staging_rows, "expected staging rows from chunked text"
    assert staging_rows[0]["source_ref"]["custody_id"] == "custody-123"
    assert ocr_jobs == []
