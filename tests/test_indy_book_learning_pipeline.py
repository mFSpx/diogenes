from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_token_chunker_caps_chunks_at_500_tokens() -> None:
    from ALGOS.indy_learning_vector import chunk_text_tokens

    text = " ".join(f"word{i}" for i in range(1200))
    chunks = chunk_text_tokens(text, max_tokens=500, overlap_tokens=25, source_ref={"custody_id": "book:test"})
    assert len(chunks) == 3
    assert all(c["token_count"] <= 500 for c in chunks)
    assert chunks[0]["token_start"] == 0
    assert chunks[1]["token_start"] == 475
    assert chunks[0]["source_ref"]["custody_id"] == "book:test"


def test_learning_vector_emits_go25_object_event_edges_without_graph_write() -> None:
    from ALGOS.indy_learning_vector import build_learning_vector

    chunks = [
        {"chunk_id": "chunk:1", "text": "Evidence supports a CLAIM about an ENTITY pattern.", "token_count": 8, "chunk_index": 0, "source_ref": {"custody_id": "book:test"}},
    ]
    vector = build_learning_vector(chunks=chunks, source_ref={"custody_id": "book:test", "title": "Test"})
    kinds = [p["kind"] for p in vector["jzloads"]]
    assert "OBJECT" in kinds
    assert "EVENT" in kinds
    assert "EDGE" in kinds
    assert all(p["kind"] in {"OBJECT", "EVENT", "EDGE"} for p in vector["jzloads"])
    assert {v["term"] for v in vector["ontology_hits"]} >= {"EVIDENCE", "CLAIM", "ENTITY", "PATTERN"}
    assert vector["canonical_graph_writes_performed"] is False


def test_pipeline_has_no_rights_class_or_license_gate() -> None:
    script = (ROOT / "scripts" / "indy_book_learning_pipeline.py").read_text(encoding="utf-8")
    forbidden = ["rights_class", "rights-class", "license_gate", "rights_class_not_allowed", "public_domain_only"]
    assert not any(token in script for token in forbidden)


def test_pipeline_compiles_book_chunks_lora_manifest_and_receipt(tmp_path: Path) -> None:
    src = tmp_path / "book.txt"
    src.write_text(" ".join(["ENTITY CLAIM EVIDENCE PATTERN ACTION"] * 220), encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/indy_book_learning_pipeline.py",
            "--source-file",
            str(src),
            "--title",
            "Research Book",
            "--author",
            "A. Tester",
            "--max-tokens",
            "500",
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=5,
    )
    assert proc.returncode == 0, proc.stderr
    payload = next(json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{"))
    assert payload["status"] == "PASS"
    assert payload["purpose"] == "research_private_study_education"
    assert payload["max_tokens_per_chunk"] == 500
    assert payload["chunks_written"] >= 2
    assert payload["max_observed_chunk_tokens"] <= 500
    assert payload["canonical_graph_writes_performed"] is False
    assert payload["lora_manifest"]
    assert (ROOT / payload["chunks_jsonl"]).exists()
    assert (ROOT / payload["lora_manifest"]).exists()
    assert {x["kind"] for x in payload["jzloads"]} <= {"OBJECT", "EVENT", "EDGE"}


def test_annas_archive_query_packet_is_api_key_env_backed_and_no_download_by_default() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/indy_book_learning_pipeline.py", "--annas-query", "graph theory", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=5,
    )
    assert proc.returncode == 0
    payload = next(json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{"))
    assert payload["status"] == "PASS_QUERY_PACKET"
    assert payload["download_performed"] is False
    assert payload["api_key_env_names"] == ["ANNAS_ARCHIVE_API_KEY", "ANNA_ARCHIVE_API_KEY"]
    assert payload["jzloads"][0]["kind"] == "OBJECT"
