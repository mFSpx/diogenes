#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ALGOS"))
sys.path.insert(0, str(ROOT / "scripts"))

from gliner_zero_shot_extractor import (
    CODE_ENTITY_LABELS,
    code_entity_fallback,
    code_entity_extract,
)
from runtime_caps import MAX_LABELS, MAX_SPANS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_PYTHON_CODE = '''"""
Sample module with various code constructs for testing.
"""
import os
import json
from typing import Any, Protocol
from dataclasses import dataclass

import psycopg

API_BASE_URL = os.environ["API_BASE_URL"]
DB_DSN = os.getenv("LUCIDOTA_GO_STATE_DSN")


class EntityProtocol(Protocol):
    """Protocol for graph entities."""
    def get_uuid(self) -> str:
        ...


class GraphItem:
    """A graph item in the LUCIDOTA ontology."""
    def __init__(self, term: str):
        self.term = term

    def get_evidence(self) -> list[str]:
        return []


class ValidationError(Exception):
    """Custom error type."""
    pass


@app.get("/api/v1/entities")
def list_entities():
    return {"entities": []}


@app.post("/api/v1/graph")
def create_graph_item():
    pass


CREATE TABLE lucidota_go.graph_item (
    uuid uuid PRIMARY KEY,
    term text NOT NULL
);

CREATE SCHEMA IF NOT EXISTS lucidota_control;

QUEUE_NAME = "krampus_ingest"
WORKFLOW_NAME = "krampus-ingest-route-extract"
JOB_KIND = "route_extract"
'''

SAMPLE_CODE_CHUNK = {
    "source_path": "tests/fixtures/sample_module.py",
    "chunk_id": "chunk:test001",
    "chunk_index": 0,
    "text": SAMPLE_PYTHON_CODE,
}

SAMPLE_JSONL_ROWS = [
    SAMPLE_CODE_CHUNK,
    {
        "source_path": "tests/fixtures/another_module.py",
        "chunk_id": "chunk:test002",
        "chunk_index": 1,
        "text": "class SimpleClass:\\n    def simple_method(self):\\n        pass\\n\\nfrom typing import List\\n",
    },
]


# ---------------------------------------------------------------------------
# Tests for code_entity_fallback
# ---------------------------------------------------------------------------

def test_code_entity_fallback_detects_class():
    """code_entity_fallback should detect CLASS_DEFINITION spans."""
    text = "class GraphItem:\n    pass\nclass ValidationError(Exception):\n    pass\n"
    spans = code_entity_fallback(text)
    class_spans = [s for s in spans if s.label == "CLASS_DEFINITION"]
    # GraphItem and ValidationError should be detected
    assert len(class_spans) >= 2
    labels_found = [s.text for s in class_spans]
    assert "GraphItem" in labels_found
    assert "ValidationError" in labels_found


def test_code_entity_fallback_detects_function():
    """code_entity_fallback should detect FUNCTION_DEFINITION spans."""
    text = "def list_entities():\n    pass\nasync def process_batch():\n    pass\n"
    spans = code_entity_fallback(text)
    func_spans = [s for s in spans if s.label == "FUNCTION_DEFINITION"]
    assert len(func_spans) >= 2
    labels_found = [s.text for s in func_spans]
    assert "list_entities" in labels_found
    assert "process_batch" in labels_found


def test_code_entity_fallback_detects_imports():
    """code_entity_fallback should detect IMPORT_STATEMENT spans."""
    text = "import os\nimport json\nfrom typing import Any, Protocol\n"
    spans = code_entity_fallback(text)
    import_spans = [s for s in spans if s.label == "IMPORT_STATEMENT"]
    assert len(import_spans) >= 2


def test_code_entity_fallback_detects_ontology_terms():
    """code_entity_fallback should detect ONTOLOGY_TERM spans."""
    text = "ENTITY, ATTRIBUTE, and RELATIONSHIP are GO-25 terms."
    spans = code_entity_fallback(text)
    onto_spans = [s for s in spans if s.label == "ONTOLOGY_TERM"]
    labels_found = [s.text for s in onto_spans]
    assert "ENTITY" in labels_found
    assert "ATTRIBUTE" in labels_found
    assert "RELATIONSHIP" in labels_found


def test_code_entity_fallback_detects_db_tables():
    """code_entity_fallback should detect DB_TABLE_NAME spans."""
    text = "CREATE TABLE IF NOT EXISTS lucidota_go.graph_item (\\n    uuid uuid PRIMARY KEY\\n);\\n"
    spans = code_entity_fallback(text)
    table_spans = [s for s in spans if s.label == "DB_TABLE_NAME"]
    assert len(table_spans) >= 1
    assert any("graph_item" in s.text for s in table_spans)


def test_code_entity_fallback_detects_queue_names():
    """code_entity_fallback should detect QUEUE_OR_WORKFLOW_NAME spans."""
    text = 'QUEUE_NAME = "krampus_ingest"\\nWORKFLOW_NAME = "krampus-ingest"\\n'
    spans = code_entity_fallback(text)
    queue_spans = [s for s in spans if s.label == "QUEUE_OR_WORKFLOW_NAME"]
    assert len(queue_spans) >= 1
    assert any("krampus_ingest" in s.text for s in queue_spans)


def test_code_entity_fallback_detects_env_vars():
    """code_entity_fallback should detect ENVIRONMENT_VARIABLE spans."""
    text = 'api_key = os.environ["API_KEY"]\ndb_url = os.getenv("DATABASE_URL")\n'
    spans = code_entity_fallback(text)
    env_spans = [s for s in spans if s.label == "ENVIRONMENT_VARIABLE"]
    assert len(env_spans) >= 1
    labels_found = [s.text for s in env_spans]
    assert any("API_KEY" in s or "DATABASE_URL" in s for s in labels_found)


def test_code_entity_fallback_detects_error_types():
    """code_entity_fallback should detect ERROR_EXCEPTION_TYPE spans."""
    text = "class ValidationError(Exception): pass\\nclass NotFoundError(Exception): pass\\n"
    spans = code_entity_fallback(text)
    error_spans = [s for s in spans if s.label == "ERROR_EXCEPTION_TYPE"]
    assert len(error_spans) >= 2


def test_code_entity_fallback_detects_protocols():
    """code_entity_fallback should detect PROTOCOL_OR_INTERFACE spans."""
    text = "class EntityProtocol(Protocol):\\n    def get_uuid(self) -> str: ...\\n"
    spans = code_entity_fallback(text)
    proto_spans = [s for s in spans if s.label == "PROTOCOL_OR_INTERFACE"]
    assert len(proto_spans) >= 1
    assert "EntityProtocol" in [s.text for s in proto_spans]


def test_code_entity_fallback_detects_api_endpoints():
    """code_entity_fallback should detect API_ENDPOINT_DEFINITION spans."""
    text = '@app.get("/api/v1/entities")\\ndef list_entities():\\n    pass\\n'
    spans = code_entity_fallback(text)
    api_spans = [s for s in spans if s.label == "API_ENDPOINT_DEFINITION"]
    assert len(api_spans) >= 1
    assert any("/api/v1/entities" in s.text for s in api_spans)


def test_code_entity_fallback_detects_algorithm_names():
    """code_entity_fallback should detect ALGORITHM_NAME spans for known algorithms."""
    text = "class BanditRouter:\\n    pass\\nclass HoeffdingTree:\\n    pass\\n"
    spans = code_entity_fallback(text)
    algo_spans = [s for s in spans if s.label == "ALGORITHM_NAME"]
    assert len(algo_spans) >= 1


def test_code_entity_fallback_detects_system_components():
    """code_entity_fallback should detect SYSTEM_COMPONENT_NAME spans."""
    text = "KRAMPUSCHEWING processes files through KORPUS."
    spans = code_entity_fallback(text)
    sys_spans = [s for s in spans if s.label == "SYSTEM_COMPONENT_NAME"]
    assert len(sys_spans) >= 2
    labels_found = [s.text for s in sys_spans]
    assert "KRAMPUSCHEWING" in labels_found
    assert "KORPUS" in labels_found


def test_code_entity_fallback_detects_db_schemas():
    """code_entity_fallback should detect DB_SCHEMA_NAME spans."""
    text = "CREATE SCHEMA IF NOT EXISTS lucidota_control;\\n"
    spans = code_entity_fallback(text)
    schema_spans = [s for s in spans if s.label == "DB_SCHEMA_NAME"]
    assert len(schema_spans) >= 1
    assert any("lucidota_control" in s.text for s in schema_spans)


# ---------------------------------------------------------------------------
# Tests for code_entity_extract
# ---------------------------------------------------------------------------

def test_code_entity_extract_full_pipeline():
    """code_entity_extract should return the full extraction result dict."""
    result = code_entity_extract(SAMPLE_PYTHON_CODE, no_fallback=False)
    assert "schema" in result
    assert "backend" in result
    assert "spans" in result
    assert result["span_count"] == len(result["spans"])
    assert result["span_count"] > 0
    # Should have found various entity types
    labels_found = {s["label"] for s in result["spans"]}
    assert "CLASS_DEFINITION" in labels_found
    assert "FUNCTION_DEFINITION" in labels_found


def test_code_entity_extract_caps_text():
    """code_entity_extract should cap text at MAX_TEXT_CHARS."""
    long_text = "class Foo:\\n    pass\\n" * 5000
    result = code_entity_extract(long_text, no_fallback=False)
    assert result["text_length"] <= 20_000  # MAX_TEXT_CHARS
    assert result["span_count"] <= MAX_SPANS


def test_code_entity_extract_custom_labels():
    """code_entity_extract should accept custom label list."""
    labels = ["CLASS_DEFINITION", "FUNCTION_DEFINITION"]
    result = code_entity_extract(SAMPLE_PYTHON_CODE, labels=labels, no_fallback=False)
    assert len(result["labels"]) == 2
    assert result["labels"] == labels
    # Spans should only have these labels
    for span in result["spans"]:
        assert span["label"] in labels


def test_code_entity_extract_no_text():
    """code_entity_extract should handle empty text gracefully."""
    result = code_entity_extract("", no_fallback=False)
    assert result["span_count"] == 0


def test_code_entity_extract_with_no_fallback():
    """code_entity_extract with no_fallback=True should return 0 spans when no GLiNER."""
    result = code_entity_extract(SAMPLE_PYTHON_CODE, no_fallback=True)
    assert result["backend"] == "code_entity_gliner_missing_or_unspecified"
    assert result["span_count"] == 0


def test_code_entity_extract_no_fallback_caps():
    """code_entity_extract no_fallback should still cap labels."""
    many_labels = [f"LABEL_{i}" for i in range(MAX_LABELS + 20)]
    result = code_entity_extract("test", labels=many_labels, no_fallback=True)
    assert len(result["labels"]) <= MAX_LABELS


# ---------------------------------------------------------------------------
# Tests for CODE_ENTITY_LABELS completeness
# ---------------------------------------------------------------------------

def test_code_entity_labels_defined():
    """CODE_ENTITY_LABELS should have all expected entity types."""
    expected = {
        "CLASS_DEFINITION",
        "FUNCTION_DEFINITION",
        "IMPORT_STATEMENT",
        "DECORATOR",
        "DB_TABLE_NAME",
        "DB_SCHEMA_NAME",
        "API_ENDPOINT_DEFINITION",
        "SQL_QUERY",
        "CONFIG_KEY",
        "ENVIRONMENT_VARIABLE",
        "ONTOLOGY_TERM",
        "ERROR_EXCEPTION_TYPE",
        "PROTOCOL_OR_INTERFACE",
        "ALGORITHM_NAME",
        "SYSTEM_COMPONENT_NAME",
        "QUEUE_OR_WORKFLOW_NAME",
        "SCHEMA_OR_CONTRACT",
    }
    label_set = set(CODE_ENTITY_LABELS)
    missing = expected - label_set
    extra = label_set - expected
    assert not missing, f"Missing expected labels: {missing}"
    # Extra labels are acceptable (user may add more), so no assertion on extra


def test_code_entity_label_count_within_bounds():
    """CODE_ENTITY_LABELS count should be within MAX_LABELS."""
    assert len(CODE_ENTITY_LABELS) <= MAX_LABELS


# ---------------------------------------------------------------------------
# Tests for source path and chunk helpers (imported from script)
# ---------------------------------------------------------------------------

def test_extract_source_path():
    """extract_source_path should find the source path in a chunk dict."""
    from scripts.krampus_gliner_pipeline import extract_source_path
    assert extract_source_path({"source_path": "foo.py"}) == "foo.py"
    assert extract_source_path({"path": "bar.py"}) == "bar.py"
    assert extract_source_path({"file": "baz.py"}) == "baz.py"
    assert extract_source_path({}) == "unknown_source"


def test_extract_source_path_fallback():
    """extract_source_path should fall back to default."""
    from scripts.krampus_gliner_pipeline import extract_source_path
    assert extract_source_path({"irrelevant": True}) == "unknown_source"


def test_extract_chunk_id_from_explicit():
    """extract_chunk_id should use explicit chunk_id if present."""
    from scripts.krampus_gliner_pipeline import extract_chunk_id
    assert extract_chunk_id({"chunk_id": "explicit_id"}) == "explicit_id"


def test_extract_chunk_id_synthesized():
    """extract_chunk_id should synthesize an ID when none is present."""
    from scripts.krampus_gliner_pipeline import extract_chunk_id
    result = extract_chunk_id({"source_path": "test.py"})
    assert result.startswith("chunk:")


def test_extract_text_for_chunk_direct():
    """extract_text_for_chunk should return the 'text' key directly."""
    from scripts.krampus_gliner_pipeline import extract_text_for_chunk
    assert extract_text_for_chunk({"text": "hello world"}) == "hello world"


def test_extract_text_for_chunk_empty():
    """extract_text_for_chunk should return empty string for empty input."""
    from scripts.krampus_gliner_pipeline import extract_text_for_chunk
    assert extract_text_for_chunk({}) == ""


def test_extract_text_for_chunk_from_spans():
    """extract_text_for_chunk should concatenate span texts."""
    from scripts.krampus_gliner_pipeline import extract_text_for_chunk
    chunk = {"spans": [{"text": "hello"}, {"text": "world"}]}
    result = extract_text_for_chunk(chunk)
    assert "hello" in result
    assert "world" in result


# ---------------------------------------------------------------------------
# Integration test with sample JSONL
# ---------------------------------------------------------------------------

def test_run_pipeline_dry_run():
    """Test the pipeline in dry-run mode with sample data."""
    from scripts.krampus_gliner_pipeline import run_pipeline
    import tempfile
    # Write sample JSONL
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for row in SAMPLE_JSONL_ROWS:
            f.write(json.dumps(row) + "\n")
        tmp_path = Path(f.name)

    try:
        result = run_pipeline(
            input_paths=[tmp_path],
            dry_run=True,
            no_fallback=False,
        )
        assert result["total_chunks"] == 2
        assert result["total_spans"] > 0
        assert result["dry_run"] is True
        assert result["db_inserts_performed"] == 0
        assert len(result.get("span_label_summary", {})) > 0
    finally:
        tmp_path.unlink(missing_ok=True)


def test_run_pipeline_limits_chunks():
    """Test the pipeline with chunk_limit."""
    from scripts.krampus_gliner_pipeline import run_pipeline
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for row in SAMPLE_JSONL_ROWS:
            f.write(json.dumps(row) + "\n")
        tmp_path = Path(f.name)

    try:
        result = run_pipeline(
            input_paths=[tmp_path],
            dry_run=True,
            no_fallback=False,
            chunk_limit=1,
        )
        assert result["total_chunks"] == 1
        assert result["total_entity_rows"] == 1
    finally:
        tmp_path.unlink(missing_ok=True)
