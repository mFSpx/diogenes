#!/usr/bin/env python3
"""Deterministic hunch/hypertimeline audit packetizer.

This does not decide truth. It extracts hunch records, preserves source hashes,
compares parsed counts against stated ledgers, and emits observation-center
state for downstream graph/Indy/Percyphon lanes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "hunch_hypertimeline"
BINDING = ROOT / "04_RUNTIME" / "indy_percyphon_hunch_subtleknife_binding.json"

DEFAULT_AUDIT_SOURCES = [
    ROOT / "09_STORAGE/krampuschewing_unpacked/retry_mfspx_dot_claude_key_state.zip_605505274/.claude/projects/C--Users-mfspx-Documents-Luci-Lucidota/memory/user_hunch_accuracy.md",
    ROOT / "09_STORAGE/krampuschewing_unpacked/retry_mfspx_dot_claude_key_state.zip_605505274/.claude/projects/C--Users-mfspx-Documents-Luci-Lucidota/memory/project_luci_personhood_hunch.md",
    ROOT / "09_STORAGE/krampuschewing_unpacked/docs_Luci-010.zip_10711561291/Luci/RandomReading/Lucidota/LOG/PONYBOY_HUNCH_ACCURACY_AUDIT.md",
    ROOT / "09_STORAGE/krampuschewing_unpacked/docs_Luci-010.zip_10711561291/Luci/RandomReading/Lucidota/LOG/PONYBOY_HUNCH_AUDIT_DEEP_SCAN.md",
    ROOT / "09_STORAGE/krampuschewing_unpacked/docs_Luci-010.zip_10711561291/Luci/RandomReading/Lucidota/LOG/SSP-006_HUNCH_CORRECTIONS_20260405.md",
]

TEXT_SUFFIXES = {".md", ".txt", ".json", ".jsonl", ".csv", ".log"}
EXCLUDE_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    "node_modules",
    "target",
    "build",
    "dist",
    ".pytest_cache",
    ".mypy_cache",
}

HUNCH_HEADER_RE = re.compile(
    r"^###\s+(?:NEW\s+)?(?:HUNCH\s+)?(?P<hunch_id>[A-Z]{2,8}-\d{2,3}|DS-\d{3})\s*:\s*(?P<title>.*)\s*$"
)
RATING_RE = re.compile(r"(?:^|\n)\s*-?\s*\*\*Rating:\s*(?P<rating>[^*\n.]+)", re.I)
COUNT_RE = re.compile(
    r"(?P<count>\d{1,5})\s+(?:tracked\s+)?hunches|hunch\s+count\s+updated:\s*(?P<count2>\d{1,5})\s+tracked",
    re.I,
)
CONFIRMED_RE = re.compile(r"(?P<count>\d{1,5})\s+confirmed\s+right", re.I)
SCORE_TOTAL_RE = re.compile(r"\|\s*\*\*TOTAL\*\*\s*\|\s*\*\*(?P<count>\d+)\*\*", re.I)

RATING_CANON = {
    "FUCKIN RIGHT": "FUCKIN RIGHT",
    "RIGHT ISH": "RIGHT ISH",
    "SORTA RIGHT": "SORTA RIGHT",
    "WRONG ISH": "WRONG ISH",
    "WRONG": "WRONG",
    "DAMN FOOL": "DAMN FOOL",
    "OPEN": "OPEN",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path | str, *, root: Path = ROOT) -> str:
    try:
        return str(Path(path).resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def normalize_rating(raw: str | None) -> str:
    if not raw:
        return "UNRATED"
    cleaned = re.sub(r"[^A-Za-z ]+", "", raw).upper().strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return RATING_CANON.get(cleaned, cleaned or "UNRATED")


def extract_explicit_counts(text: str, source: Path, *, root: Path = ROOT) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for m in COUNT_RE.finditer(text):
        value = int(m.group("count") or m.group("count2"))
        rows.append({"kind": "tracked_hunches", "count": value, "source_path": rel(source, root=root), "match": m.group(0)[:160]})
    for m in CONFIRMED_RE.finditer(text):
        rows.append({"kind": "confirmed_right", "count": int(m.group("count")), "source_path": rel(source, root=root), "match": m.group(0)[:160]})
    for m in SCORE_TOTAL_RE.finditer(text):
        rows.append({"kind": "score_card_total", "count": int(m.group("count")), "source_path": rel(source, root=root), "match": m.group(0)[:160]})
    return rows


def parse_hunch_file(path: Path, *, root: Path = ROOT) -> list[dict[str, Any]]:
    path = path.resolve()
    text = read_text(path)
    lines = text.splitlines()
    source_hash = sha256_path(path)
    starts: list[tuple[int, re.Match[str]]] = []
    for i, line in enumerate(lines, start=1):
        m = HUNCH_HEADER_RE.match(line)
        if m:
            starts.append((i, m))
    records: list[dict[str, Any]] = []
    for idx, (line_start, match) in enumerate(starts):
        line_end = (starts[idx + 1][0] - 1) if idx + 1 < len(starts) else len(lines)
        body = "\n".join(lines[line_start - 1 : line_end])
        rating_match = RATING_RE.search(body)
        rating = normalize_rating(rating_match.group("rating") if rating_match else None)
        title = (match.group("title") or "").strip().strip('"')
        hunch_id = match.group("hunch_id")
        records.append(
            {
                "kind": "HUNCH",
                "hunch_id": hunch_id,
                "title": title,
                "rating": rating,
                "line_start": line_start,
                "line_end": line_end,
                "source_path": rel(path, root=root),
                "source_sha256": source_hash,
                "source_bytes": path.stat().st_size,
                "source_mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat().replace("+00:00", "Z"),
                "body_sha256": sha256_bytes(body.encode("utf-8", errors="replace")),
                "evidence_state": "rated_from_local_audit" if rating != "UNRATED" else "candidate_unrated",
                "truth_promotion": "blocked_until_evidence_paths_reviewed",
            }
        )
    return records


def candidate_files(roots: Iterable[Path], *, max_files: int, include_repo_scan: bool) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []

    def add(path: Path) -> None:
        try:
            p = path.resolve()
        except Exception:
            return
        if p in seen or not p.exists() or not p.is_file():
            return
        if p.suffix.lower() not in TEXT_SUFFIXES:
            return
        seen.add(p)
        out.append(p)

    for root in roots:
        if root.is_file():
            add(root)
        elif root.is_dir():
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
                for name in filenames:
                    low = name.lower()
                    if "hunch" in low or "audit" in low or "chrono" in low or "timeline" in low:
                        add(Path(dirpath) / name)
                    if len(out) >= max_files:
                        return out

    if include_repo_scan and len(out) < max_files:
        for dirpath, dirnames, filenames in os.walk(ROOT):
            dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
            for name in filenames:
                low = name.lower()
                if "hunch" not in low and "timeline" not in low and "chrono" not in low:
                    continue
                add(Path(dirpath) / name)
                if len(out) >= max_files:
                    return out
    return out


def load_binding(root: Path = ROOT) -> dict[str, Any]:
    path = root / "04_RUNTIME" / "indy_percyphon_hunch_subtleknife_binding.json"
    if not path.exists():
        return {"rules": [], "sources": [], "missing": True}
    return json.loads(path.read_text(encoding="utf-8"))


def build_audit(
    *,
    roots: list[Path] | None = None,
    root: Path = ROOT,
    max_files: int = 2000,
    include_repo_scan: bool = False,
) -> dict[str, Any]:
    roots = roots or DEFAULT_AUDIT_SOURCES
    files = candidate_files(roots, max_files=max_files, include_repo_scan=include_repo_scan)
    binding = load_binding(root)
    records: list[dict[str, Any]] = []
    explicit_counts: list[dict[str, Any]] = []
    file_summaries: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for path in files:
        try:
            text = read_text(path)
            hunches = parse_hunch_file(path, root=root)
            records.extend(hunches)
            explicit_counts.extend(extract_explicit_counts(text, path, root=root))
            file_summaries.append(
                {
                    "path": rel(path, root=root),
                    "sha256": sha256_path(path),
                    "bytes": path.stat().st_size,
                    "hunch_headings": len(hunches),
                    "explicit_counts": len(explicit_counts),
                }
            )
        except Exception as exc:
            errors.append({"path": rel(path, root=root), "error": f"{type(exc).__name__}:{exc}"})

    rating_counts = Counter(r["rating"] for r in records)
    unique_ids = sorted({r["hunch_id"] for r in records})
    tracked_counts = [row["count"] for row in explicit_counts if row["kind"] == "tracked_hunches"]
    canonical_known = max(tracked_counts) if tracked_counts else None
    parsed_total = len(records)
    count_status = "NO_EXPLICIT_TOTAL_FOUND"
    if canonical_known is not None:
        count_status = "MATCH" if canonical_known == parsed_total else "DISCREPANCY_REQUIRES_REVIEW"
    materially_correct = sum(rating_counts[k] for k in ("FUCKIN RIGHT", "RIGHT ISH", "SORTA RIGHT"))
    materially_wrong = sum(rating_counts[k] for k in ("WRONG ISH", "WRONG", "DAMN FOOL"))
    resolved = parsed_total - rating_counts["OPEN"] - rating_counts["UNRATED"]
    resolved_accuracy = round((materially_correct / resolved) * 100.0, 3) if resolved else None

    comparison_anchors = [
        {
            "name": "Harold Ross",
            "role": "New Yorker co-founder/first editor",
            "local_evaluation_axis": "editorial pattern-sense plus execution discipline; useful comparator for taste, compression, and institutional voice, not proof of factual accuracy.",
            "authority": "external_history_anchor_requires_citation_when_published",
        },
        {
            "name": "Charles Sanders Peirce",
            "role": "abduction/retroduction logic anchor",
            "local_evaluation_axis": "hunch -> hypothesis -> consequences -> test; best fit for the LUCIDOTA hunch promotion gate.",
            "authority": "established_abduction_anchor",
        },
        {
            "name": "John Dewey",
            "role": "Chicago-linked pragmatist/inquiry anchor candidate",
            "local_evaluation_axis": "inquiry as iterative organism-environment correction; useful for execution feedback, not mystical certainty.",
            "authority": "probable_operator_reference_labeled_as_candidate",
        },
    ]

    return {
        "schema": "lucidota.hunch_hypertimeline_audit.v1",
        "generated_at": now(),
        "source_roots": [rel(p, root=root) for p in roots],
        "files_scanned": len(files),
        "files": file_summaries,
        "errors": errors,
        "hunch_records": records,
        "parsed_hunch_headings_total": parsed_total,
        "unique_hunch_ids_total": len(unique_ids),
        "unique_hunch_ids_sample": unique_ids[:40],
        "explicit_count_statements": explicit_counts,
        "canonical_known_tracked_total": canonical_known,
        "count_status": count_status,
        "rating_counts": dict(sorted(rating_counts.items())),
        "resolved_hunches": resolved,
        "materially_correct_count": materially_correct,
        "materially_wrong_count": materially_wrong,
        "resolved_directional_accuracy_pct": resolved_accuracy,
        "binding_rules": binding.get("rules", []),
        "applies_to": binding.get("applies_to", ["INDY_READs", "PercyphonAI"]),
        "comparison_anchors": comparison_anchors,
        "machine_law": {
            "hunch_state": "high_priority_signal_not_canonical_truth",
            "promotion_required": ["source_hash", "evidence_path", "contradiction_check", "graph_promotion_gate"],
            "subtleknife_required": "receipt_or_log_seal_for_every_cut",
        },
        "graph_writes_performed": False,
        "canonical_db_writes_performed": False,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def write_observation_center(audit: dict[str, Any], *, root: Path = ROOT) -> dict[str, Any]:
    runtime_path = root / "04_RUNTIME" / "observation_center" / "hunch_hypertimeline_latest.json"
    out_dir = root / "05_OUTPUTS" / "hunch_hypertimeline"
    report_path = out_dir / f"hunch_hypertimeline_{stamp()}.json"
    candidates_path = out_dir / f"hunch_candidates_{stamp()}.jsonl"
    latest_candidates_path = out_dir / "hunch_candidates_latest.jsonl"

    summary = {
        "schema": "lucidota.observation_center.hunch_hypertimeline.v1",
        "generated_at": audit["generated_at"],
        "canonical_known_tracked_total": audit["canonical_known_tracked_total"],
        "parsed_hunch_headings_total": audit["parsed_hunch_headings_total"],
        "unique_hunch_ids_total": audit["unique_hunch_ids_total"],
        "count_status": audit["count_status"],
        "rating_counts": audit["rating_counts"],
        "resolved_directional_accuracy_pct": audit["resolved_directional_accuracy_pct"],
        "applies_to": audit["applies_to"],
        "graph_writes_performed": False,
        "report_path": rel(report_path, root=root),
        "candidates_path": rel(candidates_path, root=root),
    }
    audit = dict(audit)
    audit["report_path"] = rel(report_path, root=root)
    audit["candidates_path"] = rel(candidates_path, root=root)
    write_json(report_path, audit)
    write_json(runtime_path, summary)
    write_jsonl(candidates_path, audit["hunch_records"])
    write_jsonl(latest_candidates_path, audit["hunch_records"])

    big_path = root / "05_OUTPUTS" / "big_board.json"
    if big_path.exists():
        try:
            big = json.loads(big_path.read_text(encoding="utf-8"))
        except Exception:
            big = {}
    else:
        big = {}
    big.setdefault("observation_center", {})["hunch_hypertimeline"] = summary
    counters = big.setdefault("counters", {})
    counters["hunch_hypertimeline_known_total"] = audit["canonical_known_tracked_total"]
    counters["hunch_hypertimeline_parsed_headings"] = audit["parsed_hunch_headings_total"]
    counters["hunch_hypertimeline_open"] = audit["rating_counts"].get("OPEN", 0)
    write_json(big_path, big)

    return {
        "report_path": rel(report_path, root=root),
        "candidates_path": rel(candidates_path, root=root),
        "latest_candidates_path": rel(latest_candidates_path, root=root),
        "observation_center_path": rel(runtime_path, root=root),
        "big_board_path": rel(big_path, root=root),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit local hunch records and publish observation-center state")
    ap.add_argument("--root", action="append", dest="roots", help="file or directory to scan; repeatable")
    ap.add_argument("--max-files", type=int, default=2000)
    ap.add_argument("--repo-scan", dest="include_repo_scan", action="store_true", help="also scan bounded repo hunch/chrono/timeline filenames")
    ap.add_argument("--no-repo-scan", dest="include_repo_scan", action="store_false")
    ap.set_defaults(include_repo_scan=False)
    ap.add_argument("--execute", action="store_true", help="write observation-center and big-board data packet")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    roots = [Path(p) for p in args.roots] if args.roots else DEFAULT_AUDIT_SOURCES
    audit = build_audit(roots=roots, max_files=args.max_files, include_repo_scan=args.include_repo_scan)
    if args.execute:
        audit["write_result"] = write_observation_center(audit)
    payload = {
        "status": "PASS" if not audit["errors"] else "REVIEW",
        "schema": audit["schema"],
        "canonical_known_tracked_total": audit["canonical_known_tracked_total"],
        "parsed_hunch_headings_total": audit["parsed_hunch_headings_total"],
        "count_status": audit["count_status"],
        "rating_counts": audit["rating_counts"],
        "resolved_directional_accuracy_pct": audit["resolved_directional_accuracy_pct"],
        "files_scanned": audit["files_scanned"],
        "write_result": audit.get("write_result"),
        "graph_writes_performed": False,
    }
    if args.json:
        print(json.dumps(payload, sort_keys=True, default=str))
    else:
        print("HUNCH_HYPERTIMELINE_AUDIT=" + payload["status"])
        print("KNOWN_TRACKED_TOTAL=" + str(payload["canonical_known_tracked_total"]))
        print("PARSED_HEADINGS=" + str(payload["parsed_hunch_headings_total"]))
        print("COUNT_STATUS=" + payload["count_status"])
        if payload.get("write_result"):
            print("REPORT_PATH=" + payload["write_result"]["report_path"])
            print("OBSERVATION_CENTER=" + payload["write_result"]["observation_center_path"])
    return 0 if payload["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
