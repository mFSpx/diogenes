#!/usr/bin/env python3
"""Emit River-ready KRAMPUSCHEWING training-row candidates; does not train."""
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
OUT_RIVER = ROOT / "05_OUTPUTS" / "krampuschewing" / "river"


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


def row_id(row: dict[str, Any]) -> str:
    basis = f"{row_path(row)}\0{row.get('sha256') or row.get('sha256_status')}\0river\0{row.get('normalized_lane_guess') or row.get('lane_guess')}"
    return "krr_" + hashlib.sha256(basis.encode("utf-8", errors="replace")).hexdigest()[:24]


def size_bucket(size: Any) -> str:
    try:
        s = int(size)
    except Exception:
        return "unknown"
    if s < 10_000:
        return "tiny"
    if s < 250_000:
        return "small"
    if s < 5_000_000:
        return "medium"
    if s < 100_000_000:
        return "large"
    return "huge"


def normalized_lane(row: dict[str, Any]) -> str:
    return str(row.get("normalized_lane_guess") or row.get("lane_guess") or "UNKNOWN")


def training_lane(row: dict[str, Any]) -> str:
    ln = normalized_lane(row)
    if ln in {"SOURCE_CODE", "DEV_WORK"} or row.get("contains_diff_terms"):
        return "DEV_WORK"
    if ln == "CASE_WORK" or row.get("contains_case_terms"):
        return "INVESTIGATIVE_WORK"
    if ln == "PROMPT_NOTE" or row.get("contains_instruction_law"):
        return "PROMPTING"
    if ln in {"SAVED_FILE", "PROOF_CANDIDATE", "MODEL_ARTIFACT", "RUNTIME_LOG", "RECEIPT"}:
        return "FILE_ORGANIZATION"
    if ln == "GRAPH_STAGING":
        return "DEV_WORK" if row.get("contains_diff_terms") else "FILE_ORGANIZATION"
    return "UNKNOWN"


def labels(row: dict[str, Any], features: dict[str, Any]) -> dict[str, str]:
    lane = training_lane(row)
    n_lane = normalized_lane(row)
    if lane == "DEV_WORK":
        mode = "bugchase" if features["has_error_terms"] or features["has_fix_terms"] or features["has_diff_terms"] else "build"
        work = "code"
    elif lane == "INVESTIGATIVE_WORK":
        mode = "investigation"
        work = "case"
    elif lane == "PROMPTING":
        mode = "review" if features["has_receipt_terms"] else "writing"
        work = "prompt"
    elif n_lane == "GRAPH_STAGING" or features["has_graph_terms"]:
        mode = "review"
        work = "graph"
    elif n_lane == "RECEIPT":
        mode = "review"
        work = "receipt"
    else:
        mode = "storage_cleanup" if lane == "FILE_ORGANIZATION" else "unknown"
        work = "evidence" if features["has_case_terms"] or features["has_receipt_terms"] else "unknown"
    outcome = "failure" if features["has_failure_terms"] else "success" if features["has_success_terms"] else "unknown"
    return {"operator_mode": mode, "outcome_guess": outcome, "work_type": work}


def load_graph_node_map(graph_packet: Path | None) -> dict[str, str]:
    if graph_packet is None or not graph_packet.exists():
        return {}
    try:
        packet = json.loads(graph_packet.read_text(encoding="utf-8"))
        nodes = ROOT / packet["graph_nodes"] if not Path(packet["graph_nodes"]).is_absolute() else Path(packet["graph_nodes"])
    except Exception:
        return {}
    out: dict[str, str] = {}
    try:
        with nodes.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                n = json.loads(line)
                sp = n.get("source_path")
                if sp:
                    out[str(sp)] = str(n.get("node_id"))
    except Exception:
        return out
    return out


def build_rows(index: Path, output: Path | None, summary_output: Path | None = None, graph_packet: Path | None = None) -> tuple[Path, Path, dict[str, Any]]:
    out_base = OUT_RIVER if graph_packet else OUT
    out_base.mkdir(parents=True, exist_ok=True)
    graph_nodes_by_path = load_graph_node_map(graph_packet)
    out = output or out_base / f"krampuschewing_river_training_candidates_{stamp()}.jsonl"
    summary_path = summary_output or out_base / f"krampuschewing_river_training_summary_{stamp()}.json"
    if not out.is_absolute():
        out = ROOT / out
    if not summary_path.is_absolute():
        summary_path = ROOT / summary_path
    out.parent.mkdir(parents=True, exist_ok=True)
    emitted = 0
    rows_seen = 0
    skipped_active_db = 0
    by_lane: Counter[str] = Counter()
    with out.open("w", encoding="utf-8") as fh:
        for row in load_rows(index):
            rows_seen += 1
            rp = row_path(row)
            if row.get("active_runtime_db_risk") or "CHROMADB" in rp or "chroma.sqlite3" in rp:
                skipped_active_db += 1
                continue
            n_lane = normalized_lane(row)
            path_depth = len(Path(str(row.get("relative_path") or "")).parts)
            reasons = row.get("classification_reason_codes") or []
            features = {
                "has_diff_terms": bool(row.get("contains_diff_terms")) or "dev_terms_or_project_guess" in reasons,
                "has_error_terms": False,
                "has_fix_terms": False,
                "has_receipt_terms": n_lane == "RECEIPT" or bool(row.get("kind_guess") == "receipt"),
                "has_case_terms": bool(row.get("contains_case_terms")) or n_lane == "CASE_WORK",
                "has_graph_terms": bool(row.get("contains_graph_terms")) or n_lane == "GRAPH_STAGING",
                "has_prompt_terms": bool(row.get("contains_instruction_law")) or n_lane == "PROMPT_NOTE",
                "has_success_terms": False,
                "has_failure_terms": bool(row.get("blockers")),
                "file_size_bucket": size_bucket(row.get("size_bytes")),
                "extension": row.get("extension"),
                "path_depth": path_depth,
                "large_file_class": row.get("large_file_class"),
                "recommended_next_action": row.get("recommended_next_action"),
            }
            lane = training_lane(row)
            item = {
                "schema": "lucidota.krampuschewing.river_training_candidate.v1",
                "row_id": row_id(row),
                "source_path": rp,
                "source_sha256": row.get("sha256"),
                "source_sha256_status": row.get("sha256_status"),
                "graph_node_id": graph_nodes_by_path.get(rp),
                "graph_packet": rel(graph_packet) if graph_packet else None,
                "lane": lane,
                "normalized_lane_guess": n_lane,
                "event_time_guess": row.get("modified_time_utc"),
                "features": features,
                "labels_if_obvious": labels(row, features),
                "truth_status": "training_candidate_only",
                "not_for_accepted_truth": True,
            }
            fh.write(json.dumps(item, sort_keys=False, ensure_ascii=False) + "\n")
            emitted += 1
            by_lane[lane] += 1
    summary = {
        "schema": "lucidota.krampuschewing.river_training.summary.v1",
        "generated_at_utc": now(),
        "index": rel(index),
        "graph_packet": rel(graph_packet) if graph_packet else None,
        "output_path": rel(out),
        "summary_path": rel(summary_path),
        "rows_seen": rows_seen,
        "rows_emitted": emitted,
        "by_lane": dict(sorted(by_lane.items())),
        "active_runtime_db_excluded": skipped_active_db > 0,
        "active_runtime_db_skipped_count": skipped_active_db,
        "river_training_performed": False,
        "verdict": "PASS" if emitted else "PARTIAL_FAIL",
        "blockers": [] if emitted else ["no_rows_emitted"],
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=False, ensure_ascii=False), encoding="utf-8")
    return out, summary_path, summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Emit River training candidate rows; do not train")
    ap.add_argument("--index", required=True)
    ap.add_argument("--graph-packet")
    ap.add_argument("--output")
    ap.add_argument("--summary-output")
    args = ap.parse_args()
    index = Path(args.index)
    if not index.is_absolute():
        index = ROOT / index
    graph_packet = Path(args.graph_packet) if args.graph_packet else None
    if graph_packet is not None and not graph_packet.is_absolute():
        graph_packet = ROOT / graph_packet
    out, summary_path, summary = build_rows(index, Path(args.output) if args.output else None, Path(args.summary_output) if args.summary_output else None, graph_packet)
    print("RIVER_TRAINING_CANDIDATES_PATH=" + rel(out))
    print("RIVER_TRAINING_SUMMARY_PATH=" + rel(summary_path))
    print("RIVER_TRAINING_CANDIDATES=" + str(summary["rows_emitted"]))
    print("ACTIVE_RUNTIME_DB_SKIPPED=" + str(summary["active_runtime_db_skipped_count"]))
    print("KRAMPUSCHEWING_RIVER_ROWS=" + summary["verdict"])
    return 0 if summary["verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
