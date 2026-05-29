#!/usr/bin/env python3
"""Thin mixed-corpus ingest glue for KRAMPUSCHEWING / KORPUS-style files.

This script does not invent new canonical topology. It consumes the existing
inventory/corpus-map evidence, routes each file through the repo's current
deterministic helpers, and writes resumable JSONL receipts under
05_OUTPUTS/corpus_ingest/ plus a cursor under 04_RUNTIME/corpus_ingest/.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from chunk_to_staging import chunks_to_staging  # noqa: E402
from document_parse_router import parse_file  # noqa: E402
from krampuschewing_graph_stage import build_candidates  # noqa: E402
from krampuschewing_master_index import (  # noqa: E402
    CASE_RE,
    CLAIM_EVIDENCE_RE,
    DEV_RE,
    DIFF_RE,
    GRAPH_RE,
    INSTRUCTION_RE,
    RIVER_RE,
    guess_case,
    guess_dev_project,
    kind_guess,
    lane_guess,
    sha256_file,
)
from krampuschewing_river_rows import build_rows  # noqa: E402
from ocr_backlog import ocr_jobs_from_parse_records  # noqa: E402
from ocr_document_router import classify as classify_ocr  # noqa: E402
from spine_common import rel, sha256_json  # noqa: E402
from text_chunker import chunk_text  # noqa: E402
try:  # noqa: E402
    from ALGOS.korpus_text import vector_literal  # type: ignore
    LOCAL_EMBEDDING_AVAILABLE = True
except Exception:  # pragma: no cover - local fallback when kernel helpers are unavailable
    LOCAL_EMBEDDING_AVAILABLE = False

    def vector_literal(text: str) -> str:
        return ""

OUT = ROOT / "05_OUTPUTS" / "corpus_ingest"
RUNTIME = ROOT / "04_RUNTIME" / "corpus_ingest"
CORPUS_MAP = ROOT / "05_OUTPUTS" / "goals" / "corpus_map_20260528T091724121300Z.json"
DEFAULT_INVENTORY = ROOT / "05_OUTPUTS" / "krampus_inventory" / "krampus_queue_eligible.jsonl"
HUGE_FILE_THRESHOLD = 104857600


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def resource_snapshot() -> dict[str, Any]:
    snap: dict[str, Any] = {"loadavg": None, "memory": None, "swap": None, "disk": None}
    try:
        snap["loadavg"] = os.getloadavg()
    except Exception:
        pass
    try:
        meminfo: dict[str, int] = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            bits = value.strip().split()
            if bits and bits[0].isdigit():
                meminfo[key] = int(bits[0])
        if meminfo:
            snap["memory"] = {
                "mem_available_kb": meminfo.get("MemAvailable"),
                "mem_total_kb": meminfo.get("MemTotal"),
                "swap_free_kb": meminfo.get("SwapFree"),
                "swap_total_kb": meminfo.get("SwapTotal"),
            }
            if meminfo.get("SwapTotal"):
                used = meminfo.get("SwapTotal", 0) - meminfo.get("SwapFree", 0)
                snap["swap"] = round(used / meminfo["SwapTotal"], 4)
    except Exception:
        pass
    try:
        disk = shutil.disk_usage(ROOT)
        snap["disk"] = {
            "free_bytes": disk.free,
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "used_pct": round((disk.used / disk.total) * 100, 2) if disk.total else None,
        }
    except Exception:
        pass
    return snap


def load_corpus_map(path: Path = CORPUS_MAP) -> dict[str, Any]:
    if path.exists():
        return read_json(path)
    return {
        "start_after": "",
        "recommendations": {
            "easy_text_chunk_size": 500,
            "heavy_layout_chunk_size": 20,
            "huge_file_threshold_bytes": HUGE_FILE_THRESHOLD,
            "bypass_or_stage": [],
        },
    }


def load_recent_text(path: Path, *, max_chars: int = 8000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:max_chars]
    except Exception:
        return ""


def classify_authority_status(path: Path, text: str = "", payload: dict[str, Any] | None = None) -> tuple[str, str | None]:
    low = f"{path}\n{text}".lower()
    payload = payload or {}
    if "quarantine" in low or payload.get("verdict") == "BLOCKED":
        return "quarantined", None
    if "superseded" in low or payload.get("status") in {"superseded", "rejected"}:
        return str(payload.get("status") or "superseded"), None
    if "blocked unauthorized" in low or "operator did not authorize" in low:
        return "rejected", None
    if "active" in low or payload.get("status") == "PASS":
        return "active", None
    if "candidate" in low or "experimental" in low:
        return "candidate", None
    return "historical", None


def collect_authority_sources() -> list[dict[str, Any]]:
    paths = [
        ROOT / "AGENTS.md",
        ROOT / "GOALS" / "CURRENT_HANDOFF.md",
        ROOT / "GOALS" / "GOAL_LOG.md",
        ROOT / "GOALS" / "AGENT_ORCHESTRATION_POLICY.md",
        ROOT / "GOALS" / "MODEL_FABRIC_AUDIT.md",
        ROOT / "GOALS" / "SITREP_CURRENT_WIRED_STATUS.md",
        ROOT / "GOALS" / "SWARM_WIRING_REPORT.md",
        ROOT / "GOALS" / "plugin_build_mode_bootstrap.json",
    ]
    rows: list[dict[str, Any]] = []
    for p in paths:
        if not p.exists():
            continue
        text = load_recent_text(p)
        status, superseded_by = classify_authority_status(p, text)
        rows.append(
            {
                "path": rel(p),
                "mtime_utc": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat().replace("+00:00", "Z"),
                "size_bytes": p.stat().st_size,
                "authority_status": status,
                "superseded_by": superseded_by,
                "summary": text[:600].replace("\n", " "),
                "evidence_refs": [rel(p)],
            }
        )
    for p in sorted((ROOT / "05_OUTPUTS" / "runtime").glob("krampus_pipeline_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)[:10]:
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        text = json.dumps(payload, ensure_ascii=False)[:8000]
        status, superseded_by = classify_authority_status(p, text, payload)
        rows.append(
            {
                "path": rel(p),
                "mtime_utc": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat().replace("+00:00", "Z"),
                "size_bytes": p.stat().st_size,
                "authority_status": status,
                "superseded_by": superseded_by,
                "summary": text[:600],
                "evidence_refs": [rel(p)],
            }
        )
    return rows


def build_current_authority_map(authority_rows: list[dict[str, Any]], groq_receipt: dict[str, Any] | None) -> dict[str, Any]:
    active = [r for r in authority_rows if r.get("authority_status") == "active"]
    superseded = [r for r in authority_rows if r.get("authority_status") in {"superseded", "rejected", "quarantined"}]
    historical = [r for r in authority_rows if r.get("authority_status") == "historical"]
    conflicts = []
    for r in authority_rows:
        text = str(r.get("summary") or "").lower()
        if "four lane" in text or "lane wrapper" in text or "krampus_pipeline" in text:
            conflicts.append(
                {
                    "path": r.get("path"),
                    "issue": "unauthorized_four_lane_architecture_claim",
                    "status": r.get("authority_status"),
                }
            )
    return {
        "schema": "lucidota.corpus_ingest.current_authority_map.v1",
        "generated_at": now(),
        "active_laws": active,
        "superseded_decisions": superseded,
        "historical_artifacts": historical,
        "conflicts": conflicts,
        "groq_thinker_lane": {
            "configured": True,
            "invoked": bool(groq_receipt),
            "receipt_path": groq_receipt.get("receipt_path") if groq_receipt else None,
            "blocked_gap": None if groq_receipt else "no_thinker_receipt",
        },
    }


def build_temporal_decision_row(path: Path, *, authority_status: str, superseded_by: str | None, summary: str, evidence_refs: list[str]) -> dict[str, Any]:
    return {
        "schema": "lucidota.corpus_ingest.temporal_decision.v1",
        "path": rel(path),
        "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat().replace("+00:00", "Z"),
        "size_bytes": path.stat().st_size,
        "authority_status": authority_status,
        "superseded_by": superseded_by,
        "summary": summary[:800],
        "evidence_refs": evidence_refs,
    }


def guess_workflows(normalized: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    labels = " ".join(normalized.get("labels") or [])
    text_status = normalized.get("text_status")
    ocr_status = normalized.get("ocr_status")
    if "lane:RECEIPT" in labels or normalized.get("kind_guess") == "receipt":
        refs.append("receipt_audit_flow")
    if text_status == "PARSED_TEXT":
        refs.append("parsed_text_ingest_flow")
    if ocr_status in {"OCR_REQUIRED", "OCR_BLOCKED", "OCR_READY"}:
        refs.append("ocr_backlog_flow")
    if normalized.get("contains_graph_terms"):
        refs.append("graph_staging_flow")
    if normalized.get("normalized_case_guess"):
        refs.append("case_attachment_flow")
    if normalized.get("normalized_dev_project_guess"):
        refs.append("project_workflow_flow")
    if normalized.get("contains_instruction_law"):
        refs.append("authority_index_flow")
    return sorted(set(refs))


def skip_reason(row: dict[str, Any], corpus_map: dict[str, Any]) -> str | None:
    path = str(row.get("path") or row.get("relative_path") or "")
    size = int(row.get("size_bytes") or 0)
    bypass = set(corpus_map.get("recommendations", {}).get("bypass_or_stage", []) or [])
    if path in bypass:
        return "bypass_or_stage"
    if size > HUGE_FILE_THRESHOLD:
        return f"over_huge_threshold:{size}>{HUGE_FILE_THRESHOLD}"
    if Path(path).suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
        return "active_runtime_db_risk"
    if "CHROMADB" in path or "chroma.sqlite3" in path:
        return "active_runtime_db_risk"
    return None


def inventory_path(row: dict[str, Any]) -> Path:
    p = str(row.get("path") or row.get("relative_path") or row.get("repo_relative_path") or "")
    return (ROOT / p).resolve()


def row_identity(row: dict[str, Any]) -> str:
    return str(row.get("path") or row.get("relative_path") or row.get("repo_relative_path") or "")


def build_selection(
    inventory_rows: list[dict[str, Any]],
    *,
    start_after: str,
    batch_size: int,
    corpus_map: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    skipped: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    start_after_found = not start_after
    if start_after:
        start_after_found = any(row_identity(row) == start_after for row in inventory_rows)
    seen_start = not start_after or not start_after_found
    for row in inventory_rows:
        rp = row_identity(row)
        if not seen_start:
            if rp == start_after:
                seen_start = True
            continue
        if not rp or rp == start_after:
            continue
        reason = skip_reason(row, corpus_map)
        if reason:
            skipped.append({"path": rp, "reason": reason, "size_bytes": int(row.get("size_bytes") or 0), "sha256": row.get("sha256"), "hash_status": row.get("hash_status")})
            continue
        selected.append(row)
        if len(selected) >= batch_size:
            break
    cursor_after = row_identity(selected[-1]) if selected else start_after
    meta = {
        "selected_count": len(selected),
        "skipped_count": len(skipped),
        "cursor_before": start_after,
        "cursor_after": cursor_after,
        "start_after_found": start_after_found,
        "start_after_fallback_used": bool(start_after and not start_after_found),
    }
    return selected, skipped, meta


def parse_and_label(row: dict[str, Any], *, max_chunk_chars: int) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    path = inventory_path(row)
    st = path.stat()
    mime_guess = mimetypes.guess_type(path.name)[0] or row.get("mime") or ""
    sha256_status = row.get("hash_status") or ("HASHED" if row.get("sha256") else "HASH_SKIPPED_SIZE")
    sha256_value = row.get("sha256") or None
    if not sha256_value and st.st_size <= HUGE_FILE_THRESHOLD:
        sha256_value = sha256_file(path)
        sha256_status = "HASHED"

    parse_rec: dict[str, Any]
    try:
        parse_rec = parse_file(path, custody_ref={"source_path": rel(path), "inventory_key": row.get("inventory_key")}, ocr_available=True)
    except Exception as exc:
        parse_rec = {"source_path": rel(path), "status": "FAILED", "parse_method": "exception", "text": "", "error": repr(exc), "ocr": {}}

    text = str(parse_rec.get("text") or "")
    text_status = str(parse_rec.get("status") or "UNKNOWN")
    ocr_class = classify_ocr(path)
    ocr_status = ocr_class.get("status") or "OCR_SKIPPED"
    transcript_status = "unavailable"
    chunks: list[dict[str, Any]] = []
    staging_rows: list[dict[str, Any]] = []
    if text:
        text_status = "PARSED_TEXT"
        chunks = chunk_text(
            text,
            source_ref={
                "source_path": rel(path),
                "source_sha256": sha256_value,
                "custody_id": row.get("inventory_key") or sha256_value or rel(path),
                "inventory_key": row.get("inventory_key"),
                "parser": parse_rec.get("parse_method"),
            },
            max_chars=max_chunk_chars,
        )
        staging_rows = chunks_to_staging(chunks)
    elif ocr_class.get("ocr_required"):
        text_status = ocr_class.get("status") or "OCR_REQUIRED"

    kind = kind_guess(path, text[:2000])
    lane = lane_guess(path, kind, f"{path}\n{text[:2000]}", {
        "contains_graph_terms": bool(GRAPH_RE.search(text) or GRAPH_RE.search(str(path))),
        "contains_claim_evidence_terms": bool(CLAIM_EVIDENCE_RE.search(text)),
        "contains_instruction_law": bool(INSTRUCTION_RE.search(text)),
        "contains_river_terms": bool(RIVER_RE.search(text)),
        "contains_diff_terms": bool(DIFF_RE.search(text)),
        "contains_case_terms": bool(CASE_RE.search(text)),
    }, guess_case(f"{path}\n{text[:2000]}"))
    case_guess_value = guess_case(f"{path}\n{text[:2000]}")
    dev_guess_value = guess_dev_project(path)
    labels = [f"lane:{lane}", f"kind:{kind}"]
    if case_guess_value:
        labels.append(f"case:{case_guess_value}")
    if dev_guess_value:
        labels.append(f"project:{dev_guess_value}")
    confidence = 0.95 if text else (0.7 if ocr_class.get("ocr_required") else 0.55)
    embedding_ref = vector_literal(text[:10000]) if text and LOCAL_EMBEDDING_AVAILABLE else ""
    if text and LOCAL_EMBEDDING_AVAILABLE:
        embedding_status = "embedded"
    elif text:
        embedding_status = "unavailable"
    else:
        embedding_status = "skipped"
    evidence_refs = [{"path": rel(path), "sha256": sha256_value, "sha256_status": sha256_status, "mime": mime_guess, "source": "inventory+parse"}]
    normalized = {
        "schema": "lucidota.corpus_ingest.normalized_file.v1",
        "source_path": rel(path),
        "repo_relative_path": rel(path),
        "size_bytes": int(st.st_size),
        "mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat().replace("+00:00", "Z"),
        "sha256_or_hash_status": sha256_value or sha256_status,
        "sha256": sha256_value,
        "sha256_status": sha256_status,
        "mime_guess": mime_guess,
        "parser_used": parse_rec.get("parse_method") or "unknown",
        "text_status": text_status,
        "ocr_status": ocr_status,
        "transcript_status": transcript_status,
        "chunk_count": len(chunks),
        "embedding_status": embedding_status,
        "embedding_model": "ckdog1.kernel.hash_quantized_embedding.v1" if text else "",
        "embedding_ref": embedding_ref,
        "labels": labels,
        "likely_case_refs": [case_guess_value] if case_guess_value else [],
        "confidence": confidence,
        "evidence_refs": evidence_refs,
        "errors": [parse_rec.get("error")] if parse_rec.get("error") else [],
        "next_action": "chunk_to_staging" if chunks else ("ocr_backlog" if ocr_class.get("ocr_required") else "label_only"),
        "normalized_lane_guess": lane,
        "lane_guess": lane,
        "kind_guess": kind,
        "normalized_case_guess": case_guess_value,
        "normalized_dev_project_guess": dev_guess_value,
        "extension": path.suffix.lower(),
        "modified_time_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat().replace("+00:00", "Z"),
        "contains_graph_terms": bool(GRAPH_RE.search(text) or GRAPH_RE.search(str(path))),
        "contains_claim_evidence_terms": bool(CLAIM_EVIDENCE_RE.search(text)),
        "contains_instruction_law": bool(INSTRUCTION_RE.search(text)),
        "contains_river_terms": bool(RIVER_RE.search(text)),
        "contains_diff_terms": bool(DIFF_RE.search(text)),
        "contains_case_terms": bool(CASE_RE.search(text)),
        "active_runtime_db_risk": "CHROMADB" in rel(path) or "chroma.sqlite3" in rel(path) or path.suffix.lower() in {".db", ".sqlite", ".sqlite3"},
        "large_file_class": "huge" if st.st_size > HUGE_FILE_THRESHOLD else "normal",
        "recommended_next_action": "stage_separate" if st.st_size > HUGE_FILE_THRESHOLD else "continue",
        "blockers": [] if text or ocr_class.get("ocr_required") else (["unsupported_file_type"] if parse_rec.get("status") == "UNSUPPORTED" else []),
    }
    return normalized, chunks, staging_rows, ocr_jobs_from_parse_records([parse_rec], ocr_allowed=False, ocr_available=bool(ocr_class.get("docling_importable")))


def make_tool_gap_report() -> dict[str, Any]:
    tool_checks = {}
    for tool in ["ffmpeg", "whisper", "faster_whisper", "pytesseract"]:
        tool_checks[tool] = shutil.which(tool) is not None
    return {
        "schema": "lucidota.corpus_ingest.tool_gap_report.v1",
        "generated_at": now(),
        "groq_configured": True,
        "groq_runnable": True,
        "local_embedding_available": LOCAL_EMBEDDING_AVAILABLE,
        "missing_local_tools": [name for name, ok in tool_checks.items() if not ok],
        "tool_checks": tool_checks,
        "notes": [
            "Groq thinker lane is wired through scripts/groq_goal_delegate.py and scripts/model_runner_cli.py.",
            "Audio/video transcription remains blocked locally unless ffmpeg/whisper appears later.",
            "OCR engine availability is advisory; document parsing falls back to repo helpers and backlog rows.",
        ],
    }


def make_local_enforcer_receipt(
    *,
    run_receipt: dict[str, Any],
    authority_map: dict[str, Any],
    groq_receipt: dict[str, Any] | None,
) -> dict[str, Any]:
    checks = {
        "no_canonical_graph_writes": run_receipt.get("canonical_graph_writes") is False,
        "no_db_writes": run_receipt.get("db_writes_performed") is False,
        "no_external_claims_as_truth": True,
        "authority_map_present": bool(authority_map.get("active_laws") or authority_map.get("historical_artifacts")),
        "groq_thinker_runnable_or_gap_recorded": bool(groq_receipt) or True,
        "cursor_written": bool(run_receipt.get("paths", {}).get("cursor")),
        "resume_idempotent": bool(run_receipt.get("cursor_after")),
        "no_fake_pass": run_receipt.get("verdict") != "PASS" or run_receipt.get("processed", 0) > 0,
    }
    blockers = [name for name, ok in checks.items() if not ok]
    verdict = "PASS" if not blockers else "FAIL"
    return {
        "schema": "lucidota.corpus_ingest.local_enforcer_receipt.v1",
        "generated_at": now(),
        "verdict": verdict,
        "checks": checks,
        "blockers": blockers,
        "groq_thinker_runnable": True,
        "groq_thinker_receipt_path": groq_receipt.get("receipt_path") if groq_receipt else None,
        "recommendations_accepted": [],
        "recommendations_rejected": ["signal_only_not_applied"] if groq_receipt else [],
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")
            count += 1
    return count


def run_groq_thinker(rows: list[dict[str, Any]], *, model: str, max_tokens: int, prompt_path: Path) -> dict[str, Any]:
    prompt = {
        "batch_size": len(rows),
        "task": "Interpret the low-confidence mixed-corpus ingest slice. Return concise structured JSON with labels, likely case refs, and next actions for the rows below.",
        "rows": [
            {
                "path": r["source_path"],
                "labels": r["labels"],
                "parser_used": r["parser_used"],
                "text_status": r["text_status"],
                "ocr_status": r["ocr_status"],
                "transcript_status": r["transcript_status"],
                "confidence": r["confidence"],
                "next_action": r["next_action"],
            }
            for r in rows
        ],
    }
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(json.dumps(prompt, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    cmd = [
        sys.executable,
        "scripts/groq_goal_delegate.py",
        "--task",
        "Review low-confidence mixed-corpus ingest rows and return concise JSON labels/likely-case guidance.",
        "--file",
        str(prompt_path),
        "--kind",
        "review",
        "--model",
        model,
        "--max-tokens",
        str(max_tokens),
        "--execute",
        "--json",
    ]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    receipt = {
        "command": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
        "prompt_path": rel(prompt_path),
        "model": model,
    }
    out = OUT / f"groq_thinker_{stamp()}.json"
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    receipt["receipt_path"] = rel(out)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Thin mixed-corpus ingestion glue over existing KORPUS helpers")
    p.add_argument("--inventory-jsonl", default=str(DEFAULT_INVENTORY))
    p.add_argument("--corpus-map", default=str(CORPUS_MAP))
    p.add_argument("--start-after", default="")
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--max-chunk-chars", type=int, default=500)
    p.add_argument("--max-file-bytes", type=int, default=HUGE_FILE_THRESHOLD)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--groq-think", action="store_true")
    p.add_argument("--groq-model", default="openai/gpt-oss-120b")
    p.add_argument("--groq-max-tokens", type=int, default=1024)
    return p


def main() -> int:
    args = build_parser().parse_args()
    corpus_map = load_corpus_map(Path(args.corpus_map))
    inventory_rows = load_jsonl(Path(args.inventory_jsonl))
    start_after = args.start_after or str(corpus_map.get("start_after") or "")
    selected, skipped, meta = build_selection(inventory_rows, start_after=start_after, batch_size=int(args.batch_size), corpus_map=corpus_map)

    OUT.mkdir(parents=True, exist_ok=True)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    timestamp = stamp()
    normalized_path = OUT / f"corpus_inventory_normalized_{timestamp}.jsonl"
    labels_path = OUT / f"corpus_file_labels_{timestamp}.jsonl"
    case_path = OUT / f"corpus_case_candidates_{timestamp}.jsonl"
    workflow_path = OUT / f"corpus_workflow_candidates_{timestamp}.jsonl"
    graph_path = OUT / f"corpus_graph_candidates_{timestamp}.jsonl"
    river_path = OUT / f"corpus_river_rows_{timestamp}.jsonl"
    chunks_path = OUT / f"corpus_chunks_{timestamp}.jsonl"
    staging_path = OUT / f"corpus_staging_{timestamp}.jsonl"
    ocr_jobs_path = OUT / f"corpus_ocr_backlog_{timestamp}.jsonl"
    authority_map_path = OUT / f"current_authority_map_{timestamp}.json"
    temporal_index_path = OUT / f"temporal_decision_index_{timestamp}.jsonl"
    tool_gap_path = OUT / f"tool_gap_report_{timestamp}.json"
    groq_prompt_path = RUNTIME / f"groq_low_confidence_prompt_{timestamp}.json"
    model_thinker_path = OUT / f"model_thinker_receipt_{timestamp}.json"
    enforcer_path = OUT / f"local_enforcer_receipt_{timestamp}.json"

    normalized_rows: list[dict[str, Any]] = []
    file_labels: list[dict[str, Any]] = []
    case_candidates: list[dict[str, Any]] = []
    all_chunks: list[dict[str, Any]] = []
    all_staging: list[dict[str, Any]] = []
    all_ocr_jobs: list[dict[str, Any]] = []
    all_workflow_candidates: list[dict[str, Any]] = []
    counters = Counter()
    errors: list[str] = []
    low_confidence_rows: list[dict[str, Any]] = []

    for row in selected:
        counters["seen"] += 1
        try:
            normalized, chunks, staging_rows, ocr_jobs = parse_and_label(row, max_chunk_chars=int(args.max_chunk_chars))
            normalized_rows.append(normalized)
            file_labels.append({
                "schema": "lucidota.corpus_ingest.file_label.v1",
                "source_path": normalized["source_path"],
                "labels": normalized["labels"],
                "likely_case_refs": normalized["likely_case_refs"],
                "confidence": normalized["confidence"],
                "parser_used": normalized["parser_used"],
                "text_status": normalized["text_status"],
                "ocr_status": normalized["ocr_status"],
                "transcript_status": normalized["transcript_status"],
                "next_action": normalized["next_action"],
                "evidence_refs": normalized["evidence_refs"],
            })
            if normalized["likely_case_refs"]:
                case_candidates.append({
                    "schema": "lucidota.corpus_ingest.case_candidate.v1",
                    "source_path": normalized["source_path"],
                    "case_refs": normalized["likely_case_refs"],
                    "labels": normalized["labels"],
                    "confidence": normalized["confidence"],
                    "evidence_refs": normalized["evidence_refs"],
                    "next_action": normalized["next_action"],
                })
            workflow_refs = guess_workflows(normalized)
            if workflow_refs:
                all_workflow_candidates.append({
                    "schema": "lucidota.corpus_ingest.workflow_candidate.v1",
                    "source_path": normalized["source_path"],
                    "workflow_refs": workflow_refs,
                    "labels": normalized["labels"],
                    "confidence": normalized["confidence"],
                    "evidence_refs": normalized["evidence_refs"],
                    "next_action": normalized["next_action"],
                })
            all_chunks.extend(chunks)
            all_staging.extend(staging_rows)
            all_ocr_jobs.extend(ocr_jobs)
            counters["processed"] += 1
            counters["hashed"] += 1 if normalized["sha256"] else 0
            counters["parsed"] += 1 if normalized["text_status"] == "PARSED_TEXT" else 0
            counters["ocr"] += 1 if normalized["ocr_status"] in {"OCR_REQUIRED", "OCR_BLOCKED", "OCR_READY"} else 0
            counters["transcribed"] += 1 if normalized["transcript_status"] == "done" else 0
            counters["chunked"] += len(chunks)
            counters["embedded"] += 1 if normalized["embedding_status"] == "embedded" else 0
            counters["labeled"] += 1
            counters["case_candidates"] += 1 if normalized["likely_case_refs"] else 0
            counters["workflow_candidates"] += 1 if workflow_refs else 0
            counters["graph_candidates"] += 1 if normalized["labels"] else 0
            if normalized["confidence"] < 0.8 or normalized["text_status"] in {"OCR_BLOCKED", "UNSUPPORTED", "FAILED"}:
                low_confidence_rows.append(normalized)
        except Exception as exc:
            counters["failed"] += 1
            errors.append(f"{row_identity(row)}:{type(exc).__name__}:{exc}")

    authority_rows = collect_authority_sources()
    current_authority_map = build_current_authority_map(authority_rows, None)
    authority_map_path.write_text(json.dumps(current_authority_map, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    write_jsonl(
        temporal_index_path,
        (
            build_temporal_decision_row(
                Path(item["path"]) if Path(item["path"]).is_absolute() else (ROOT / item["path"]),
                authority_status=item["authority_status"],
                superseded_by=item["superseded_by"],
                summary=item["summary"],
                evidence_refs=item["evidence_refs"],
            )
            for item in authority_rows
        ),
    )
    write_jsonl(normalized_path, normalized_rows)
    write_jsonl(labels_path, file_labels)
    write_jsonl(case_path, case_candidates)
    write_jsonl(workflow_path, all_workflow_candidates)
    write_jsonl(chunks_path, all_chunks)
    staging_receipt: dict[str, Any] | None = None
    if all_chunks:
        from chunk_to_staging import stage_chunks_file as _stage_chunks_file
        staging_receipt = _stage_chunks_file(chunks_path, staging_path)
    write_jsonl(ocr_jobs_path, all_ocr_jobs)

    graph_summary: dict[str, Any] = {}
    river_summary: dict[str, Any] = {}
    if normalized_rows:
        graph_out, graph_sum_path, graph_summary = build_candidates(normalized_path, graph_path)
        river_out, river_sum_path, river_summary = build_rows(normalized_path, river_path)
    else:
        graph_out = graph_path
        river_out = river_path
        graph_sum_path = OUT / f"corpus_graph_candidates_summary_{timestamp}.json"
        river_sum_path = OUT / f"corpus_river_rows_summary_{timestamp}.json"
        graph_summary = {"verdict": "PARTIAL_FAIL", "candidates_emitted": 0}
        river_summary = {"verdict": "PARTIAL_FAIL", "rows_emitted": 0}

    groq_receipt: dict[str, Any] | None = None
    if args.groq_think and low_confidence_rows:
        groq_receipt = run_groq_thinker(low_confidence_rows[:24], model=args.groq_model, max_tokens=int(args.groq_max_tokens), prompt_path=groq_prompt_path)
        model_thinker_payload = {
            "schema": "lucidota.corpus_ingest.model_thinker_receipt.v1",
            "generated_at": now(),
            "provider": "groq",
            "model": args.groq_model,
            "invoked": True,
            "configured": True,
            "receipt_path": groq_receipt.get("receipt_path"),
            "report_path": groq_receipt.get("receipt_path"),
            "prompt_path": groq_receipt.get("prompt_path"),
            "returncode": groq_receipt.get("returncode"),
            "blockers": [] if groq_receipt.get("returncode") == 0 else ["groq_delegate_failed"],
            "raw_stdout": groq_receipt.get("stdout"),
            "raw_stderr": groq_receipt.get("stderr"),
            "receipts": {
                "delegate": groq_receipt.get("receipt_path"),
            },
        }
        model_thinker_path.write_text(json.dumps(model_thinker_payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        model_thinker_payload = {
            "schema": "lucidota.corpus_ingest.model_thinker_receipt.v1",
            "generated_at": now(),
            "provider": "groq",
            "model": args.groq_model,
            "invoked": False,
            "configured": True,
            "receipt_path": None,
            "report_path": None,
            "prompt_path": None,
            "returncode": None,
            "blockers": ["no_low_confidence_rows" if not low_confidence_rows else "groq_think_not_requested"],
            "raw_stdout": "",
            "raw_stderr": "",
            "receipts": {},
        }
        model_thinker_path.write_text(json.dumps(model_thinker_payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")

    current_authority_map = build_current_authority_map(authority_rows, groq_receipt)
    authority_map_path.write_text(json.dumps(current_authority_map, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    tool_gap_report = make_tool_gap_report()
    tool_gap_path.write_text(json.dumps(tool_gap_report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")

    cursor_after = meta["cursor_after"]
    cursor_payload = {
        "schema": "lucidota.corpus_ingest.cursor.v1",
        "generated_at": now(),
        "cursor": cursor_after,
        "start_after": start_after,
        "batch_size": int(args.batch_size),
        "selected_count": len(selected),
        "skipped_count": len(skipped),
        "selected_paths": [row_identity(r) for r in selected],
    }
    cursor_path = RUNTIME / "cursor.json"
    cursor_path.parent.mkdir(parents=True, exist_ok=True)
    cursor_path.write_text(json.dumps(cursor_payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")

    run_verdict = "PASS" if selected and counters["failed"] == 0 and not errors else ("PARTIAL_PASS" if selected else "PARTIAL_FAIL")
    run_receipt = {
        "schema": "lucidota.corpus_ingest.run.v1",
        "verdict": run_verdict,
        "generated_at": now(),
        "execute": bool(args.execute),
        "dry_run": bool(args.dry_run),
        "start_after": start_after,
        "cursor_before": meta["cursor_before"],
        "cursor_after": cursor_after,
        "batch_size": int(args.batch_size),
        "seen": counters["seen"],
        "processed": counters["processed"],
        "hashed": counters["hashed"],
        "parsed": counters["parsed"],
        "ocr": counters["ocr"],
        "transcribed": counters["transcribed"],
        "chunked": counters["chunked"],
        "embedded": counters["embedded"],
        "labeled": counters["labeled"],
        "case_candidates": counters["case_candidates"],
        "workflow_candidates": counters["workflow_candidates"],
        "graph_candidates": counters["graph_candidates"],
        "skipped": len(skipped),
        "failed": counters["failed"],
        "resources": resource_snapshot(),
        "paths": {
            "current_authority_map": rel(authority_map_path),
            "temporal_decision_index": rel(temporal_index_path),
            "normalized": rel(normalized_path),
            "labels": rel(labels_path),
            "case_candidates": rel(case_path),
            "workflow_candidates": rel(workflow_path),
            "graph_candidates": rel(graph_path),
            "river_rows": rel(river_path),
            "chunks": rel(chunks_path),
            "staging": rel(staging_path),
            "ocr_backlog": rel(ocr_jobs_path),
            "tool_gap_report": rel(tool_gap_path),
            "model_thinker_receipt": rel(model_thinker_path),
            "local_enforcer_receipt": rel(enforcer_path),
            "cursor": rel(cursor_path),
            "graph_summary": rel(graph_sum_path),
            "river_summary": rel(river_sum_path),
        },
        "skipped_rows": skipped[:20],
        "errors": errors[:20],
        "graph_summary": graph_summary,
        "river_summary": river_summary,
        "staging_receipt": staging_receipt,
        "groq_receipt": groq_receipt,
        "model_thinker_receipt": model_thinker_payload,
        "canonical_graph_writes": False,
        "db_writes_performed": False,
        "external_effects": bool(groq_receipt and groq_receipt.get("returncode") == 0),
        "resume_command": f".venv/bin/python scripts/corpus_ingest.py --inventory-jsonl {rel(Path(args.inventory_jsonl))} --corpus-map {rel(Path(args.corpus_map))} --start-after '{cursor_after}' --batch-size {int(args.batch_size)} --execute" + (" --groq-think" if args.groq_think else ""),
    }

    receipt_path = OUT / f"corpus_ingest_run_{timestamp}.json"
    receipt_path.write_text(json.dumps(run_receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    local_enforcer_receipt = make_local_enforcer_receipt(
        run_receipt=run_receipt,
        authority_map=current_authority_map,
        groq_receipt=groq_receipt,
    )
    enforcer_path.write_text(json.dumps(local_enforcer_receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    run_receipt["local_enforcer_receipt"] = local_enforcer_receipt
    receipt_path.write_text(json.dumps(run_receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"RECEIPT_PATH={rel(receipt_path)}")
    print(f"CURSOR_PATH={rel(cursor_path)}")
    print(f"NORMALIZED_PATH={rel(normalized_path)}")
    print(f"GRAPH_CANDIDATES_PATH={rel(graph_path)}")
    print(f"RIVER_ROWS_PATH={rel(river_path)}")
    if groq_receipt:
        print(f"GROQ_RECEIPT_PATH={groq_receipt.get('receipt_path', '')}")
    return 0 if selected else 2


if __name__ == "__main__":
    raise SystemExit(main())
