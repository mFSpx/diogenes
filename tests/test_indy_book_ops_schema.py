from __future__ import annotations

from pathlib import Path


def test_indy_book_ops_schema_defines_db_visible_book_work_tables_and_manual_surface() -> None:
    schema = Path("06_SCHEMA/152_indy_book_ops_and_operator_manual.sql").read_text(encoding="utf-8")
    for needle in [
        "CREATE TABLE IF NOT EXISTS lucidota_indy.book_scan",
        "CREATE TABLE IF NOT EXISTS lucidota_indy.book_read_queue",
        "CREATE TABLE IF NOT EXISTS lucidota_indy.book_note",
        "CREATE TABLE IF NOT EXISTS lucidota_indy.lora_candidate",
        "CREATE TABLE IF NOT EXISTS lucidota_indy.lora_adapter",
        "CREATE TABLE IF NOT EXISTS lucidota_indy.training_job",
        "CREATE TABLE IF NOT EXISTS lucidota_indy.book_receipt",
        "CREATE OR REPLACE VIEW lucidota_canon.manual_current",
        "LUCIDOTA Operator Manual",
        "BOOKS folder watcher authority",
        "/rpc/cloud_packet",
    ]:
        assert needle in schema


def test_indy_reads_book_watch_is_deprecated_and_service_no_longer_exposes_books_path() -> None:
    workflow_registry = Path("06_SCHEMA/006_workflow_registry.sql").read_text(encoding="utf-8")
    assert "indy-reads-book-watch" in workflow_registry
    assert "'deprecated'" in workflow_registry
    assert "DB-visible book_source/book_scan/book_read_queue/book_note/lora_candidate/lora_adapter/training_job/book_receipt rows are authoritative" in workflow_registry

    service = Path("services/ironclaw-indy-reads.service").read_text(encoding="utf-8")
    assert "ReadWritePaths=%h/LUCIDOTA/BOOKS" not in service
    assert "ReadWritePaths=%h/LUCIDOTA/04_RUNTIME %h/LUCIDOTA/05_OUTPUTS" in service
