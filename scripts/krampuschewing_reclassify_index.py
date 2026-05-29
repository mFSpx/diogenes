#!/usr/bin/env python3
"""Normalize/reclassify an existing KRAMPUSCHEWING index without touching files."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "krampuschewing"
VERSION = "krampuschewing.reclassify.v1"

SOURCE_KINDS = {"python", "rust", "sql", "shell", "source"}
RUNTIME_EXTS = {".log"}
MODEL_EXTS = {".pkl", ".pickle", ".model", ".bin", ".index", ".faiss", ".parquet", ".sqlite", ".sqlite3", ".db"}
ARCHIVE_KINDS = {"archive"}
MEDIA_KINDS = {"image", "audio", "video", "pdf"}
SAVED_KINDS = {"markdown", "json", "jsonl", "obsidian", "text", "unknown"} | MEDIA_KINDS | ARCHIVE_KINDS

CASE_RE = re.compile(r"\b(case|counsel|RTB|BCFSA|complaint|witness|landlord|tenant|property|investigation|forensic|chronology|timeline|claim|affidavit|exhibit|lease|evidence)\b", re.I)
DEV_RE = re.compile(r"\b(repo|src|scripts|cargo|pytest|bug|build|work order|phase|implementation|diff|patch|TODO|dev|rust|python|compile|test)\b", re.I)
PROMPT_RE = re.compile(r"CURRENT MODE|HARD LAW|DO NOT|MUST|one[- ]shot|operator instruction|assistant|Claude|Codex|ChatGPT|prompt", re.I)
GRAPH_RE = re.compile(r"graph[_ -]?(candidate|promotion|staging|item|edge|journal)|lucidota_go\.|node\b|edge\b", re.I)
CLAIM_EVIDENCE_RE = re.compile(r"\b(claim|evidence|custody|proof|source artifact|screenshot|exhibit|affidavit|receipt)\b", re.I)


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def load_large_validation(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for rec in data.get("records", []):
        key = rec.get("relative_path")
        if key:
            out[str(key)] = rec
    return out


def load_rows(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                yield {"schema": "lucidota.krampuschewing.master_index.row.v1", "relative_path": f"[malformed:{line_no}]", "blockers": [f"malformed_jsonl:{line_no}:{exc}"]}
                continue
            if isinstance(row, dict):
                yield row


def row_rel(row: dict[str, Any]) -> str:
    return str(row.get("relative_path") or row.get("repo_relative_path") or row.get("path") or "")


def no_root_bias_text(row: dict[str, Any]) -> str:
    # Use root-relative path only. Do not include the absolute path or repo root name
    # because /KRAMPUSCHEWING/ itself is not a case signal.
    return row_rel(row)


def normalized_case_guess(text: str) -> str | None:
    checks = [
        ("KRAMPUS_HOLIDAY_POGROMS", r"KRAMPUS_HOLIDAY_POGROMS|holiday pogrom|5490|Ash|PropertySage|Rakhra|RTB|BCFSA"),
        ("RICKSHAW_ROBBERY", r"RICKSHAW|Rickshaw|RR-|RTB910237063"),
        ("NORTHERN_STRIKE", r"NorthernStrike|NORTHERN_STRIKE"),
        ("CCTC", r"CCTC|Cute City|tricycle"),
        ("META_SUPPORT", r"Meta|Facebook|Messenger"),
        ("COOP_DETAT", r"COOP_DETAT|Coop d'Etat|Very English Coop"),
    ]
    for name, pat in checks:
        if re.search(pat, text, re.I):
            return name
    return None


def normalized_dev_project(text: str) -> str | None:
    lower = text.lower()
    for name in ["claudecode", "lucidota", "luci", "krampus_express", "lucidota_etl", "xgb_pipeline", "river", "absurd"]:
        if name in lower:
            return name
    parts = Path(text).parts
    for marker in ["PROJECTS", "01_REPOS", "CORE", "scripts", "ALGOS"]:
        if marker in parts:
            idx = parts.index(marker)
            return parts[idx + 1] if idx + 1 < len(parts) else marker
    return None


def active_db_risk(row: dict[str, Any], large_rec: dict[str, Any] | None, text: str) -> bool:
    # Active runtime DB risk is narrower than "any database-like file". The
    # current validated blocker is the large-file validation record for the
    # live-ish ChromaDB store. Other legacy chroma/sqlite files are excluded
    # from derived graph/River outputs by path, but are not counted as the
    # single active runtime DB risk unless the validator identified them.
    return bool(large_rec and large_rec.get("large_class") == "ACTIVE_RUNTIME_DB_RISK")


def normalized_lane(row: dict[str, Any], text: str, case_guess: str | None, dev_project: str | None, large_rec: dict[str, Any] | None) -> tuple[str, list[str]]:
    reasons: list[str] = []
    kind = str(row.get("kind_guess") or "unknown")
    ext = str(row.get("extension") or "").lower()
    lower = text.lower()
    if active_db_risk(row, large_rec, text):
        reasons.append("active_runtime_db_risk")
        return "MODEL_ARTIFACT", reasons
    if kind == "receipt" or "receipt" in lower or "report" in lower or "05_outputs" in lower:
        reasons.append("receipt_or_report")
        return "RECEIPT", reasons
    if kind in SOURCE_KINDS:
        reasons.append("source_kind")
        return "SOURCE_CODE", reasons
    if ext in RUNTIME_EXTS or "runtime" in lower or "journal" in lower or "service" in lower:
        reasons.append("runtime_log_path_or_ext")
        return "RUNTIME_LOG", reasons
    if bool(row.get("contains_graph_terms")) or GRAPH_RE.search(text):
        reasons.append("graph_terms")
        return "GRAPH_STAGING", reasons
    if bool(row.get("contains_instruction_law")) or PROMPT_RE.search(text):
        reasons.append("instruction_or_prompt_terms")
        return "PROMPT_NOTE", reasons
    if case_guess or bool(row.get("contains_case_terms")) or CASE_RE.search(text):
        reasons.append("case_terms_or_case_guess")
        return "CASE_WORK", reasons
    if dev_project or bool(row.get("contains_diff_terms")) or DEV_RE.search(text):
        reasons.append("dev_terms_or_project_guess")
        return "DEV_WORK", reasons
    if bool(row.get("contains_claim_evidence_terms")) or CLAIM_EVIDENCE_RE.search(text):
        reasons.append("proof_or_evidence_terms")
        return "PROOF_CANDIDATE", reasons
    if ext in MODEL_EXTS or kind == "MODEL_ARTIFACT":
        reasons.append("model_or_index_ext")
        return "MODEL_ARTIFACT", reasons
    if kind in SAVED_KINDS:
        reasons.append(f"saved_kind:{kind}")
        return "SAVED_FILE", reasons
    reasons.append("no_strong_signal")
    return "UNKNOWN", reasons


def recommended_action(row: dict[str, Any], lane: str, large_rec: dict[str, Any] | None, active_db: bool) -> str:
    large_class = large_rec.get("large_class") if large_rec else None
    if active_db:
        return "do_not_touch_active_db"
    if large_class == "LARGE_ARCHIVE":
        return "archive_later_candidate"
    if large_class in {"LARGE_MEDIA", "LARGE_UNKNOWN"}:
        return "proof_kernel_later"
    if lane in {"CASE_WORK", "DEV_WORK", "PROMPT_NOTE", "RECEIPT", "SOURCE_CODE", "GRAPH_STAGING", "PROOF_CANDIDATE"}:
        return "graph_stage_candidate"
    if lane in {"SAVED_FILE", "MODEL_ARTIFACT"}:
        return "proof_kernel_later"
    return "manual_review"


def normalize_row(row: dict[str, Any], large_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    text = no_root_bias_text(row)
    large_rec = large_map.get(row_rel(row))
    case_guess = normalized_case_guess(text)
    dev_project = normalized_dev_project(text)
    lane, reasons = normalized_lane(row, text, case_guess, dev_project, large_rec)
    db_risk = active_db_risk(row, large_rec, text)
    row = dict(row)
    row["normalized_lane_guess"] = lane
    row["normalized_case_guess"] = case_guess
    row["normalized_dev_project_guess"] = dev_project
    row["classification_version"] = VERSION
    row["classification_reason_codes"] = reasons
    row["active_runtime_db_risk"] = db_risk
    row["large_file_class"] = large_rec.get("large_class") if large_rec else None
    row["recommended_next_action"] = recommended_action(row, lane, large_rec, db_risk)
    return row


def reclassify(index_path: Path, large_file_validation: Path, output: Path | None = None, summary_output: Path | None = None) -> tuple[Path, Path, dict[str, Any]]:
    OUT.mkdir(parents=True, exist_ok=True)
    out = output or OUT / f"krampuschewing_normalized_index_{stamp()}.jsonl"
    summary_out = summary_output or OUT / f"krampuschewing_normalized_summary_{stamp()}.json"
    if not out.is_absolute():
        out = ROOT / out
    if not summary_out.is_absolute():
        summary_out = ROOT / summary_out
    large_map = load_large_validation(large_file_validation)
    rows_read = 0
    rows_norm = 0
    blockers: list[str] = []
    by_lane: Counter[str] = Counter()
    by_large: Counter[str] = Counter()
    actions: Counter[str] = Counter()
    active_db_count = 0
    with out.open("w", encoding="utf-8") as fh:
        for row in load_rows(index_path):
            rows_read += 1
            try:
                norm = normalize_row(row, large_map)
            except Exception as exc:
                norm = dict(row)
                norm["classification_version"] = VERSION
                norm["classification_reason_codes"] = ["normalization_exception"]
                norm["normalization_blocker"] = str(exc)
                blockers.append(f"row_normalization_failed:{row_rel(row)}:{exc}")
            rows_norm += 1
            by_lane[str(norm.get("normalized_lane_guess") or "UNKNOWN")] += 1
            if norm.get("large_file_class"):
                by_large[str(norm["large_file_class"])] += 1
            actions[str(norm.get("recommended_next_action") or "missing")] += 1
            if norm.get("active_runtime_db_risk"):
                active_db_count += 1
            fh.write(json.dumps(norm, sort_keys=False, ensure_ascii=False) + "\n")
    summary = {
        "schema": "lucidota.krampuschewing.normalized_summary.v1",
        "generated_at_utc": now(),
        "source_index": rel(index_path),
        "source_large_file_validation": rel(large_file_validation),
        "normalized_index": rel(out),
        "normalized_summary": rel(summary_out),
        "rows_read": rows_read,
        "rows_normalized": rows_norm,
        "classification_version": VERSION,
        "by_normalized_lane_guess": dict(sorted(by_lane.items())),
        "by_large_file_class": dict(sorted(by_large.items())),
        "by_recommended_next_action": dict(sorted(actions.items())),
        "active_runtime_db_risk_count": active_db_count,
        "files_moved": [],
        "files_deleted": [],
        "full_hashing_performed": False,
        "archive_unpacking_performed": False,
        "canonical_graph_materialization": False,
        "canonical_graph_writes": False,
        "verdict": "PASS" if not blockers and rows_read == rows_norm else "PARTIAL_FAIL",
        "blockers": blockers,
    }
    summary_out.write_text(json.dumps(summary, indent=2, sort_keys=False, ensure_ascii=False), encoding="utf-8")
    return out, summary_out, summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Normalize existing KRAMPUSCHEWING index without filesystem re-chew")
    ap.add_argument("--index", required=True)
    ap.add_argument("--summary", help="Accepted for provenance/CLI compatibility; this script does not reread files.")
    ap.add_argument("--large-file-validation", required=True)
    ap.add_argument("--output")
    ap.add_argument("--summary-output")
    args = ap.parse_args()
    index = Path(args.index)
    if not index.is_absolute():
        index = ROOT / index
    large = Path(args.large_file_validation)
    if not large.is_absolute():
        large = ROOT / large
    out, summary_out, summary = reclassify(index, large, Path(args.output) if args.output else None, Path(args.summary_output) if args.summary_output else None)
    print("NORMALIZED_INDEX_PATH=" + rel(out))
    print("NORMALIZED_SUMMARY_PATH=" + rel(summary_out))
    print("KRAMPUSCHEWING_RECLASSIFY_INDEX=" + summary["verdict"])
    print("ROWS_READ=" + str(summary["rows_read"]))
    print("ROWS_NORMALIZED=" + str(summary["rows_normalized"]))
    print("ACTIVE_RUNTIME_DB_RISK_COUNT=" + str(summary["active_runtime_db_risk_count"]))
    return 0 if summary["verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
