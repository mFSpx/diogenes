#!/usr/bin/env python3
"""Emit KRAMPUSCHEWING graph-staging candidates only; no DB writes."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "krampuschewing"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def load_rows(index: Path) -> Iterable[dict[str, Any]]:
    with index.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                yield item


def row_path(row: dict[str, Any]) -> str:
    return str(row.get("repo_relative_path") or row.get("relative_path") or row.get("path") or "")


def candidate_id(row: dict[str, Any], kind: str) -> str:
    basis = f"{kind}\0{row_path(row)}\0{row.get('sha256') or row.get('sha256_status')}\0{row.get('normalized_lane_guess') or row.get('lane_guess')}"
    return "kgc_" + hashlib.sha256(basis.encode("utf-8", errors="replace")).hexdigest()[:24]


def lane(row: dict[str, Any]) -> str:
    return str(row.get("normalized_lane_guess") or row.get("lane_guess") or "UNKNOWN")


def case_guess(row: dict[str, Any]) -> Any:
    return row.get("normalized_case_guess") if "normalized_case_guess" in row else row.get("case_guess")


def dev_project(row: dict[str, Any]) -> Any:
    return row.get("normalized_dev_project_guess") if "normalized_dev_project_guess" in row else row.get("dev_project_guess")


def kind_for(row: dict[str, Any]) -> str:
    ln = lane(row)
    if ln == "RECEIPT":
        return "RECEIPT"
    if ln == "PROMPT_NOTE":
        return "PROMPT"
    if ln == "CASE_WORK":
        return "CASE" if case_guess(row) else "EVIDENCE"
    if ln == "DEV_WORK":
        return "DEV_WORK"
    if ln == "SOURCE_CODE":
        return "SOURCE_CODE"
    if ln == "RUNTIME_LOG":
        return "WORK_SESSION"
    if ln == "GRAPH_STAGING":
        return "GRAPH_STAGING"
    if ln == "PROOF_CANDIDATE":
        return "PROOF_CANDIDATE"
    return "EVIDENCE"


def confidence_for(row: dict[str, Any], kind: str) -> str:
    if row.get("sha256") and lane(row) in {"RECEIPT", "SOURCE_CODE", "CASE_WORK", "PROMPT_NOTE"}:
        return "high"
    if row.get("sha256") and lane(row) != "UNKNOWN":
        return "medium"
    return "low"


def label_for(row: dict[str, Any], kind: str) -> str:
    rp = row_path(row)
    if kind == "CASE":
        return f"Case work: {case_guess(row) or rp}"
    if kind == "DEV_WORK":
        return f"Dev work artifact: {dev_project(row) or rp}"
    if kind == "SOURCE_CODE":
        return f"Source code artifact: {dev_project(row) or rp}"
    if kind == "PROMPT":
        return "Prompt / instruction-law candidate"
    if kind == "RECEIPT":
        return "Receipt / report candidate"
    if kind == "WORK_SESSION":
        return "Runtime/work-session log candidate"
    if kind == "GRAPH_STAGING":
        return "Graph-staging candidate/source"
    if kind == "PROOF_CANDIDATE":
        return "Proof/custody candidate"
    return "Evidence/proof candidate"


def why_for(row: dict[str, Any], kind: str) -> str:
    bits = [f"normalized_lane={lane(row)}", f"kind={row.get('kind_guess')}"]
    for key in ["classification_reason_codes", "large_file_class", "recommended_next_action"]:
        if row.get(key):
            bits.append(f"{key}={row.get(key)}")
    for key in ["contains_graph_terms", "contains_claim_evidence_terms", "contains_instruction_law", "contains_river_terms", "contains_diff_terms", "contains_case_terms"]:
        if row.get(key):
            bits.append(key)
    return "; ".join(bits)


def duplicate_signature(row: dict[str, Any], kind: str) -> tuple[Any, ...] | None:
    digest = row.get("sha256")
    if not digest or not row.get("duplicate_of"):
        return None
    # Keep duplicates when path/context is likely meaningful. Collapse generic file/log dupes.
    ln = lane(row)
    if ln in {"CASE_WORK", "PROMPT_NOTE", "SOURCE_CODE", "DEV_WORK", "RECEIPT"}:
        return None
    return (digest, kind, ln, Path(row_path(row)).name)


def build_candidates(index: Path, output: Path | None, summary_output: Path | None = None) -> tuple[Path, Path, dict[str, Any]]:
    OUT.mkdir(parents=True, exist_ok=True)
    out = output or OUT / f"krampuschewing_graph_candidates_{stamp()}.jsonl"
    summary_path = summary_output or OUT / f"krampuschewing_graph_candidates_summary_{stamp()}.json"
    if not out.is_absolute():
        out = ROOT / out
    if not summary_path.is_absolute():
        summary_path = ROOT / summary_path
    out.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    emitted = 0
    rows_seen = 0
    skipped_active_db = 0
    skipped_duplicates = 0
    seen_dups: set[tuple[Any, ...]] = set()
    with out.open("w", encoding="utf-8") as fh:
        for row in load_rows(index):
            rows_seen += 1
            rp = row_path(row)
            if row.get("active_runtime_db_risk") or "CHROMADB" in rp or "chroma.sqlite3" in rp:
                skipped_active_db += 1
                continue
            kind = kind_for(row)
            sig = duplicate_signature(row, kind)
            if sig is not None:
                if sig in seen_dups:
                    skipped_duplicates += 1
                    continue
                seen_dups.add(sig)
            cand = {
                "schema": "lucidota.krampuschewing.graph_candidate.v1",
                "candidate_id": candidate_id(row, kind),
                "candidate_kind": kind,
                "source_path": rp,
                "source_sha256": row.get("sha256"),
                "source_sha256_status": row.get("sha256_status"),
                "proposed_term": case_guess(row) or dev_project(row) or Path(str(rp)).stem[:160],
                "proposed_label": label_for(row, kind),
                "proposed_payload": {
                    "normalized_lane_guess": lane(row),
                    "original_lane_guess": row.get("lane_guess"),
                    "kind_guess": row.get("kind_guess"),
                    "normalized_case_guess": case_guess(row),
                    "normalized_dev_project_guess": dev_project(row),
                    "extension": row.get("extension"),
                    "size_bytes": row.get("size_bytes"),
                    "modified_time_utc": row.get("modified_time_utc"),
                    "large_file_class": row.get("large_file_class"),
                    "recommended_next_action": row.get("recommended_next_action"),
                    "classification_version": row.get("classification_version"),
                    "flags": {k: bool(row.get(k)) for k in ["contains_graph_terms", "contains_claim_evidence_terms", "contains_instruction_law", "contains_river_terms", "contains_diff_terms", "contains_case_terms"]},
                },
                "evidence_refs": [{"path": rp, "sha256": row.get("sha256"), "sha256_status": row.get("sha256_status"), "ref_kind": "source_index_row"}],
                "confidence": confidence_for(row, kind),
                "why": why_for(row, kind),
                "materialization_allowed_now": False,
            }
            fh.write(json.dumps(cand, sort_keys=False, ensure_ascii=False) + "\n")
            emitted += 1
            counts[kind] += 1
    emitted_any = emitted > 0
    summary = {
        "schema": "lucidota.krampuschewing.graph_stage.summary.v1",
        "generated_at_utc": now(),
        "index": rel(index),
        "output_path": rel(out),
        "summary_path": rel(summary_path),
        "rows_seen": rows_seen,
        "candidates_emitted": emitted,
        "by_candidate_kind": dict(sorted(counts.items())),
        "active_runtime_db_excluded": skipped_active_db > 0,
        "active_runtime_db_skipped_count": skipped_active_db,
        "duplicates_skipped_count": skipped_duplicates,
        "db_writes_performed": False,
        "canonical_graph_materialization": False,
        "canonical_graph_writes": False,
        "verdict": "PASS" if emitted_any else "PARTIAL_FAIL",
        "blockers": [] if emitted_any else ["no_candidates_emitted"],
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=False, ensure_ascii=False), encoding="utf-8")
    return out, summary_path, summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Emit KRAMPUSCHEWING graph candidates without DB writes")
    ap.add_argument("--index", required=True)
    ap.add_argument("--output")
    ap.add_argument("--summary-output")
    args = ap.parse_args()
    index = Path(args.index)
    if not index.is_absolute():
        index = ROOT / index
    out, summary_path, summary = build_candidates(index, Path(args.output) if args.output else None, Path(args.summary_output) if args.summary_output else None)
    print("GRAPH_CANDIDATES_PATH=" + rel(out))
    print("GRAPH_CANDIDATES_SUMMARY_PATH=" + rel(summary_path))
    print("GRAPH_CANDIDATES=" + str(summary["candidates_emitted"]))
    print("ACTIVE_RUNTIME_DB_SKIPPED=" + str(summary["active_runtime_db_skipped_count"]))
    print("KRAMPUSCHEWING_GRAPH_STAGE=" + summary["verdict"])
    return 0 if summary["verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
