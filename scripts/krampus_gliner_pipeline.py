#!/usr/bin/env python3
"""GLiNER-style named entity extraction pipeline for ALL extracted code.

Takes JSONL extraction output (code chunks with source_path + text), runs
entity extraction on each chunk using code-specific entity labels, and writes
results to both Postgres (lucidota_go.graph_promotion_evidence_resolution) and
a JSONL receipt file at 05_OUTPUTS/gliner/entities_<stamp>.jsonl.

Receipt scope: LOCAL_FILE_PRODUCT + ABSURD_POSTGRES_RUNTIME
Mutation class: receipt_only, candidate_writer
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ALGOS"))

from ALGOS.gliner_zero_shot_extractor import Span, code_entity_extract, code_entity_fallback, CODE_ENTITY_LABELS, extract as ontology_extract  # noqa: E402
from ALGOS.runtime_caps import MAX_LABELS, MAX_SPANS  # noqa: E402

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
GLINER_OUTPUT_DIR = ROOT / "05_OUTPUTS" / "gliner"
DEFAULT_INPUT = ROOT / "05_OUTPUTS" / "runtime" / "krampus_gliner_spans_*.jsonl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def confidence_bps(score: float) -> int:
    return max(0, min(10000, int(round(float(score) * 10000))))


def resolve_input_paths(pattern: str) -> list[Path]:
    """Resolve a glob pattern to a list of input JSONL files."""
    p = Path(pattern)
    if p.exists() and p.is_file():
        return [p]
    if "*" in pattern or "?" in pattern:
        paths = sorted(ROOT.glob(pattern)) if not pattern.startswith("/") else sorted(Path(".").glob(pattern))
        if not paths:
            # Try relative to ROOT
            paths = sorted(ROOT.glob(pattern))
        return paths
    return [p]


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    """Load all lines from a JSONL file."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_text_for_chunk(chunk: dict[str, Any]) -> str:
    """Extract the text content from a chunk row.

    Handles various chunk formats:
    - Chunks with 'text' key
    - Chunks with 'spans' that have text
    - Nested 'chunk' keys
    """
    if "text" in chunk and isinstance(chunk["text"], str) and chunk["text"].strip():
        return chunk["text"]
    # Some formats nest the text inside a 'chunk' dict
    if "chunk" in chunk and isinstance(chunk["chunk"], dict):
        inner = chunk["chunk"]
        for key in ("text", "content", "body"):
            if key in inner and isinstance(inner[key], str) and inner[key].strip():
                return inner[key]
    # Last resort: the entire line content might be a concatenation of spans
    spans = chunk.get("spans", [])
    if spans and isinstance(spans, list):
        parts = []
        for s in spans:
            if isinstance(s, dict) and "text" in s:
                parts.append(s["text"])
        if parts:
            return " ".join(parts)
    return ""


def extract_source_path(chunk: dict[str, Any]) -> str:
    """Extract source path from a chunk, using various key names."""
    for key in ("source_path", "path", "source", "file", "document_path"):
        if key in chunk and isinstance(chunk[key], str):
            return chunk[key]
    return "unknown_source"


def extract_chunk_id(chunk: dict[str, Any]) -> str:
    """Extract or synthesize a chunk ID."""
    for key in ("chunk_id", "id", "row_id", "uuid"):
        if key in chunk and isinstance(chunk[key], str):
            return chunk[key]
    # Synthesize one from source path + sha256
    src = extract_source_path(chunk)
    txt_hash = sha256_text(dumps(chunk))
    return f"chunk:{sha256_text(src)}:{txt_hash[:12]}"


# ---------------------------------------------------------------------------
# DB writer
# ---------------------------------------------------------------------------

def write_evidence_resolution(conn: Any, evidence_ref: str, ref_kind: str,
                              resolved: bool, resolver: str,
                              detail: dict[str, Any]) -> dict[str, Any]:
    """Insert a row into graph_promotion_evidence_resolution."""
    import uuid as _uuidlib
    resolution_uuid = str(_uuidlib.uuid4())
    detail_json = json.dumps(detail, default=str)
    conn.execute(
        """
        INSERT INTO lucidota_go.graph_promotion_evidence_resolution
            (resolution_uuid, evidence_ref, ref_kind, resolved, resolver, detail)
        VALUES (%s, %s, %s, %s, %s, %s::jsonb)
        """,
        (resolution_uuid, evidence_ref, ref_kind, resolved, resolver, detail_json),
    )
    conn.commit()
    return {
        "resolution_uuid": resolution_uuid,
        "evidence_ref": evidence_ref,
        "ref_kind": ref_kind,
        "resolved": resolved,
    }


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def run_pipeline(*, input_paths: list[Path], db_dsn: str | None = None,
                 model_path: str | None = None, threshold: float = 0.35,
                 allow_remote: bool = False, dry_run: bool = False,
                 no_fallback: bool = False, chunk_limit: int = 0,
                 labels: list[str] | None = None,
                 extract_mode: str = "code") -> dict[str, Any]:
    """Run the GLiNER entity extraction pipeline over all input chunks.

    Parameters
    ----------
    input_paths : list[Path]
        Paths to JSONL files containing extraction chunks.
    db_dsn : str, optional
        Postgres DSN for writing to graph_promotion_evidence_resolution.
    model_path : str, optional
        Path to local GLiNER model.
    threshold : float
        GLiNER prediction threshold.
    allow_remote : bool
        Allow remote GLiNER model download.
    dry_run : bool
        If True, skip DB writes and JSONL output.
    no_fallback : bool
        If True, skip regex fallback when GLiNER unavailable.
    chunk_limit : int
        Max chunks to process (0 = unlimited).
    labels : list[str], optional
        Custom entity labels. Defaults to CODE_ENTITY_LABELS.
    extract_mode : str
        'code' for code entity extraction, 'ontology' for ontology label extraction.

    Returns
    -------
    dict with pipeline receipt data.
    """
    start_time = now_iso()
    total_chunks = 0
    total_spans = 0
    total_errors = 0
    entity_rows: list[dict[str, Any]] = []
    all_spans: list[dict[str, Any]] = []
    processed_paths: set[str] = set()
    db_inserts: list[dict[str, Any]] = []

    effective_labels = labels if labels else list(CODE_ENTITY_LABELS)

    # Load all rows from all input files
    rows: list[dict[str, Any]] = []
    for inp in input_paths:
        if not inp.exists():
            continue
        file_rows = load_jsonl_rows(inp)
        rows.extend(file_rows)
        processed_paths.add(str(inp))

    if chunk_limit > 0:
        rows = rows[:chunk_limit]

    # Determine the extract function
    if extract_mode == "ontology":
        extract_fn = ontology_extract
        extract_kwargs = {"labels": effective_labels}
    else:
        extract_fn = code_entity_extract
        extract_kwargs = {"labels": effective_labels}

    for chunk_idx, chunk in enumerate(rows):
        text = extract_text_for_chunk(chunk)
        source_path = extract_source_path(chunk)
        chunk_id = extract_chunk_id(chunk)

        if not text.strip():
            continue

        total_chunks += 1
        try:
            result = extract_fn(
                text,
                **extract_kwargs,
                model=model_path,
                threshold=threshold,
                allow_remote_model=allow_remote,
                no_fallback=no_fallback,
            )
        except Exception as exc:
            total_errors += 1
            entity_rows.append({
                "chunk_id": chunk_id,
                "source_path": source_path,
                "chunk_index": chunk_idx,
                "error": f"{type(exc).__name__}: {exc}",
                "span_count": 0,
                "spans": [],
            })
            continue

        spans = result.get("spans", [])
        span_count = result.get("span_count", 0)
        total_spans += span_count

        # Build output row for this chunk
        output_row = {
            "backend": result.get("backend", "unknown"),
            "chunk_id": chunk_id,
            "chunk_index": chunk_idx,
            "source_path": source_path,
            "span_count": span_count,
            "spans": spans,
            "text_sha256": result.get("text_sha256", sha256_text(text)),
            "generated_at": result.get("generated_at", now_iso()),
        }
        entity_rows.append(output_row)

        # Collect all spans for JSONL output
        for sp_idx, span in enumerate(spans):
            span_id = f"{chunk_id}:{span.get('start', 0)}:{span.get('end', 0)}:{span.get('label', 'unknown')}"
            all_spans.append({
                "span_id": span_id,
                "chunk_id": chunk_id,
                "chunk_index": chunk_idx,
                "chunk_path": source_path,
                "label": span.get("label"),
                "text": span.get("text"),
                "start": span.get("start"),
                "end": span.get("end"),
                "score": span.get("score"),
                "backend": span.get("backend", result.get("backend")),
                "confidence_bps": confidence_bps(span.get("score", 0)),
                "evidence_ref": f"gliner:code:{sha256_text(source_path)}:{span.get('start', 0)}:{span.get('end', 0)}:{span.get('label', '?')}",
            })

    # Build DB evidence rows (only if not dry_run and db_dsn provided)
    if not dry_run and db_dsn:
        try:
            import psycopg
            conn = psycopg.connect(db_dsn)
            for sp in all_spans:
                detail = {
                    "chunk_id": sp["chunk_id"],
                    "source_path": sp["chunk_path"],
                    "label": sp["label"],
                    "matched_text": sp["text"],
                    "span_start": sp["start"],
                    "span_end": sp["end"],
                    "score": sp["score"],
                    "confidence_bps": sp["confidence_bps"],
                    "extractor": "gliner_code_entity_pipeline",
                    "pipeline_version": "v1",
                }
                db_result = write_evidence_resolution(
                    conn,
                    evidence_ref=sp["evidence_ref"],
                    ref_kind="gliner_code_entity",
                    resolved=True,
                    resolver="scripts/krampus_gliner_pipeline.py",
                    detail=detail,
                )
                db_inserts.append(db_result)
            conn.close()
        except Exception as exc:
            total_errors += 1
            print(f"DB_WRITE_ERROR={exc}", file=sys.stderr)

    # Write output JSONL
    output_data = {
        "schema": "lucidota.proof_hoard.gliner_pipeline_output.v1",
        "generated_at": now_iso(),
        "started_at": start_time,
        "pipeline": "krampus_gliner_pipeline",
        "pipeline_version": "v1",
        "extract_mode": extract_mode,
        "labels": effective_labels[:MAX_LABELS],
        "backend_summary": _summarize_backends(entity_rows),
        "input_files": [str(p) for p in processed_paths],
        "total_chunks": total_chunks,
        "total_spans": total_spans,
        "total_errors": total_errors,
        "total_entity_rows": len(entity_rows),
        "total_distinct_spans": len(all_spans),
        "db_inserts_performed": len(db_inserts),
        "db_inserts": db_inserts if len(db_inserts) <= 20 else (db_inserts[:20] + [f"... {len(db_inserts) - 20} more"]),
        "dry_run": dry_run,
        "span_label_summary": _summarize_labels(all_spans),
    }

    if not dry_run:
        out_dir = GLINER_OUTPUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        # Write full entities JSONL (one row per chunk)
        entities_out = out_dir / f"entities_{stamp()}.jsonl"
        with entities_out.open("w", encoding="utf-8") as fh:
            for row in entity_rows:
                fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        output_data["entities_jsonl"] = str(entities_out.relative_to(ROOT))

        # Write summary receipt
        receipt_out = out_dir / f"pipeline_receipt_{stamp()}.json"
        receipt_out.write_text(
            json.dumps(output_data, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        output_data["receipt_path"] = str(receipt_out.relative_to(ROOT))
        print(f"RECEIPT_PATH={output_data['receipt_path']}")

    return output_data


def _summarize_backends(entity_rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in entity_rows:
        b = row.get("backend", "unknown")
        counts[b] = counts.get(b, 0) + 1
    return counts


def _summarize_labels(all_spans: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sp in all_spans:
        label = sp.get("label", "unknown")
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="GLiNER-style named entity extraction pipeline for ALL extracted code. "
                    "Takes JSONL extraction output and runs code entity extraction on each chunk.",
    )
    ap.add_argument("--input", "-i", default=None,
                    help="Path or glob pattern for input JSONL extraction chunks. "
                         f"Default: resolves to {DEFAULT_INPUT.relative_to(ROOT) if DEFAULT_INPUT else 'krampus_gliner_spans'}.")
    ap.add_argument("--db-dsn", default=os.environ.get("LUCIDOTA_GO_STATE_DSN"),
                    help="Postgres DSN for graph_promotion_evidence_resolution writes. "
                         "Defaults to LUCIDOTA_GO_STATE_DSN env var.")
    ap.add_argument("--model", default=os.environ.get("GLINER_MODEL_PATH"),
                    help="Local GLiNER model path/name. Use GLINER_MODEL_PATH env var or pass directly.")
    ap.add_argument("--threshold", type=float, default=0.35,
                    help="GLiNER prediction threshold (default: 0.35).")
    ap.add_argument("--allow-remote", action="store_true",
                    help="Allow GLiNER.from_pretrained to resolve a non-local model name; may download.")
    ap.add_argument("--no-fallback", action="store_true",
                    help="Skip regex fallback when GLiNER/model is unavailable.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Skip DB writes and JSONL file output; only print summary.")
    ap.add_argument("--pretty", action="store_true",
                    help="Pretty-print the receipt JSON to stdout.")
    ap.add_argument("--chunk-limit", type=int, default=0,
                    help="Limit chunks processed (0 = unlimited).")
    ap.add_argument("--extract-mode", choices=["code", "ontology"], default="code",
                    help="'code' for code entity extraction (default), 'ontology' for ontology label extraction.")
    ap.add_argument("--labels", default=None,
                    help="Custom comma-separated entity labels. Overrides the default label set for the mode.")
    return ap


def main() -> int:
    ap = build_parser()
    args = ap.parse_args()

    # Resolve input paths
    if args.input:
        input_pattern = args.input
    else:
        # Use the most recent krampus_gliner_spans JSONL from runtime output
        candidates = sorted(ROOT.glob("05_OUTPUTS/runtime/krampus_gliner_spans_*.jsonl"))
        if candidates:
            input_pattern = str(candidates[-1])
        else:
            # Provide a sensible fallback fallback
            input_pattern = str(ROOT / "05_OUTPUTS" / "runtime" / "krampus_gliner_spans_*.jsonl")

    input_paths = resolve_input_paths(input_pattern)
    if not input_paths:
        print(f"No input files found matching: {input_pattern}", file=sys.stderr)
        return 1

    for p in input_paths:
        if not p.exists():
            print(f"Input file does not exist: {p}", file=sys.stderr)
            return 1

    # Parse custom labels if provided
    custom_labels = None
    if args.labels:
        custom_labels = [part.strip() for part in args.labels.split(",") if part.strip()]

    result = run_pipeline(
        input_paths=input_paths,
        db_dsn=args.db_dsn,
        model_path=args.model,
        threshold=args.threshold,
        allow_remote=args.allow_remote,
        dry_run=args.dry_run,
        no_fallback=args.no_fallback,
        chunk_limit=args.chunk_limit,
        labels=custom_labels,
        extract_mode=args.extract_mode,
    )

    if args.pretty or args.dry_run:
        print(json.dumps(result, indent=2, sort_keys=True, default=str))

    summary = (
        f"GLINER pipeline summary: {result['total_chunks']} chunks, "
        f"{result['total_spans']} entity spans, "
        f"{result['total_errors']} errors, "
        f"{result['total_distinct_spans']} distinct entities, "
        f"{result['db_inserts_performed']} DB inserts."
    )
    print(summary)
    return 0 if result["total_errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
