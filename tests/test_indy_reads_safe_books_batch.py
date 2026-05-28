from __future__ import annotations

import json
from pathlib import Path


def test_safe_books_batch_uses_existing_extractor_and_pipeline_without_graph_writes(tmp_path: Path) -> None:
    from scripts.indy_reads_safe_books_batch import run_batch

    books = tmp_path / "BOOKS"
    books.mkdir()
    (books / "Abduction Field Notes -- Nancy Drew.txt").write_text(
        " ".join(["ENTITY CLAIM EVIDENCE PATTERN ACTION ROUTE"] * 240),
        encoding="utf-8",
    )

    receipt = run_batch(
        books_root=books,
        limit=1,
        max_tokens=500,
        overlap_tokens=25,
        out_dir=tmp_path / "out",
        extracted_root=tmp_path / "extracted",
    )

    assert receipt["status"] == "PASS"
    assert receipt["books_considered"] == 1
    assert receipt["books_processed"] == 1
    assert receipt["canonical_graph_writes_performed"] is False
    assert receipt["model_calls_performed"] is False
    assert receipt["entries"][0]["extract_method"] == "plain_text"
    assert receipt["entries"][0]["child_status"] == "PASS"
    assert receipt["entries"][0]["max_observed_chunk_tokens"] <= 500

    child = json.loads((Path(receipt["entries"][0]["child_receipt_path"])).read_text(encoding="utf-8"))
    assert child["canonical_graph_writes_performed"] is False
    assert child["max_observed_chunk_tokens"] <= 500
    assert Path(receipt["entries"][0]["extracted_text_path"]).exists()
