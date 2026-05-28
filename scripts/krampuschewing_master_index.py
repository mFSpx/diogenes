#!/usr/bin/env python3
"""Build a non-mutating KRAMPUSCHEWING master index."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "krampuschewing"
TEXT_EXTS = {".md", ".txt", ".json", ".jsonl", ".ndjson", ".yaml", ".yml", ".toml", ".py", ".rs", ".sql", ".sh", ".ps1", ".cmd", ".bat", ".csv", ".log", ".html", ".htm", ".xml", ".svg"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".tiff", ".tif", ".bmp", ".svg"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
ARCHIVE_EXTS = {".zip", ".7z", ".tar", ".gz", ".bz2", ".xz", ".rar"}
CODE_EXTS = {".py", ".rs", ".sql", ".sh", ".ps1", ".cmd", ".bat", ".js", ".ts", ".toml"}

GRAPH_RE = re.compile(r"graph[_ -]?(candidate|promotion|staging|item|edge|journal)|lucidota_go\.|node\b|edge\b", re.I)
CLAIM_EVIDENCE_RE = re.compile(r"\b(claim|evidence|custody|proof|source artifact|screenshot|exhibit|affidavit|receipt)\b", re.I)
INSTRUCTION_RE = re.compile(r"CURRENT MODE|HARD LAW|DO NOT|MUST|one[- ]shot|operator instruction|assistant|Claude|Codex|ChatGPT|prompt", re.I)
RIVER_RE = re.compile(r"\bRiver\b|DBSTREAM|river_model|vibe_telemetry|online learning|training row", re.I)
DIFF_RE = re.compile(r"\bdiff\b|patch|changed files|bug|bugchase|fix|failure|regression|dev cycle|delta|commit", re.I)
CASE_RE = re.compile(r"\b(case|counsel|RTB|BCFSA|complaint|witness|landlord|tenant|property|investigation|forensic|chronology|timeline|claim|affidavit|exhibit|lease|evidence)\b", re.I)
DEV_RE = re.compile(r"\b(repo|src|scripts|cargo|pytest|bug|build|work order|phase|implementation|diff|patch|TODO|dev|rust|python|compile|test)\b", re.I)
PROMPT_RE = INSTRUCTION_RE
SUCCESS_RE = re.compile(r"\b(PASS|PASSED|success|succeeded|verified|digested|complete)\b", re.I)
FAIL_RE = re.compile(r"\b(FAIL|FAILED|error|traceback|blocked|blocker|partial|regression)\b", re.I)


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def utc_from_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sample_text(path: Path, max_bytes: int) -> str:
    if path.suffix.lower() not in TEXT_EXTS and path.name.lower() not in {"makefile", "dockerfile"}:
        return ""
    try:
        with path.open("rb") as fh:
            data = fh.read(max_bytes)
        return data.decode("utf-8", errors="replace")
    except Exception:
        return ""


def kind_guess(path: Path, text: str) -> str:
    ext = path.suffix.lower()
    low_path = str(path).lower()
    if ".obsidian" in low_path:
        return "obsidian"
    if ext == ".md":
        return "markdown"
    if ext == ".json":
        if '"schema"' in text and ('"report_path"' in text or '"verdict"' in text or '"status"' in text):
            return "receipt"
        return "json"
    if ext in {".jsonl", ".ndjson"}:
        return "jsonl"
    if ext == ".py":
        return "python"
    if ext == ".rs":
        return "rust"
    if ext == ".sql":
        return "sql"
    if ext in {".sh", ".bash", ".ps1", ".cmd", ".bat"}:
        return "shell"
    if ext == ".pdf":
        return "pdf"
    if ext in IMAGE_EXTS:
        return "image"
    if ext in AUDIO_EXTS:
        return "audio"
    if ext in VIDEO_EXTS:
        return "video"
    if ext in ARCHIVE_EXTS:
        return "archive"
    if ext in CODE_EXTS:
        return "source"
    if "receipt" in low_path or "report" in low_path:
        return "receipt"
    return "text" if text else "unknown"


def guess_case(path_text: str) -> str | None:
    # Do not let the organizer root name ("KRAMPUSCHEWING") classify every
    # artifact as the KRAMPUS case. Only explicit case names/identifiers count.
    checks = [
        ("KRAMPUS_HOLIDAY_POGROMS", r"KRAMPUS_HOLIDAY_POGROMS|holiday pogrom|5490|Ash|PropertySage|Rakhra|RTB|BCFSA"),
        ("RICKSHAW_ROBBERY", r"RICKSHAW|Rickshaw|RR-|RTB910237063"),
        ("NORTHERN_STRIKE", r"NorthernStrike|NORTHERN_STRIKE"),
        ("CCTC", r"CCTC|Cute City|tricycle"),
        ("META_SUPPORT", r"Meta|Facebook|Messenger"),
    ]
    for name, pat in checks:
        if re.search(pat, path_text, re.I):
            return name
    return None


def guess_dev_project(path: Path) -> str | None:
    parts = list(path.parts)
    for marker in ["PROJECTS", "01_REPOS", "CORE", "scripts", "ALGOS"]:
        if marker in parts:
            idx = parts.index(marker)
            if idx + 1 < len(parts):
                return parts[idx + 1]
            return marker
    text = str(path)
    for name in ["KRAMPUS_EXPRESS", "Lucidota", "Luci", "claudecode", "lucidota_etl", "XGB_PIPELINE"]:
        if name.lower() in text.lower():
            return name
    return None


def lane_guess(path: Path, kind: str, hay: str, flags: dict[str, bool], case_guess_value: str | None = None) -> str:
    low_path = str(path).lower()
    if kind == "receipt" or "05_outputs" in low_path or "receipt" in low_path or "report" in low_path:
        return "RECEIPT"
    if kind in {"python", "rust", "sql", "shell", "source"} or path.suffix.lower() in CODE_EXTS:
        return "SOURCE_CODE"
    if kind == "obsidian":
        return "PROMPT_NOTE" if flags["contains_instruction_law"] else "SAVED_FILE"
    if kind == "unknown" and path.suffix.lower() in {".pkl", ".model", ".bin"}:
        return "MODEL_ARTIFACT"
    if kind == "text" and path.suffix.lower() == ".log" or "runtime" in low_path or "journal" in low_path:
        return "RUNTIME_LOG"
    if flags["contains_graph_terms"]:
        return "GRAPH_STAGING"
    if flags["contains_instruction_law"] or PROMPT_RE.search(hay):
        return "PROMPT_NOTE"
    if flags["contains_case_terms"] or case_guess_value:
        return "CASE_WORK"
    if DEV_RE.search(hay) or flags["contains_diff_terms"]:
        return "DEV_WORK"
    if flags["contains_claim_evidence_terms"]:
        return "PROOF_CANDIDATE"
    if kind in {"pdf", "image", "audio", "video", "archive", "markdown", "json", "jsonl"}:
        return "SAVED_FILE"
    return "UNKNOWN"


def row_for(path: Path, root: Path, seen_hashes: dict[str, str], text_sample_bytes: int, max_hash_bytes: int, source_label: str | None = None) -> dict[str, Any]:
    blockers: list[str] = []
    sha256_status = "computed"
    try:
        st = path.stat()
        readable = True
        if max_hash_bytes >= 0 and st.st_size > max_hash_bytes:
            digest = None
            sha256_status = "skipped_large_file"
            blockers.append(f"sha256_skipped_large_file:{st.st_size}>{max_hash_bytes}")
        else:
            digest = sha256_file(path)
    except Exception as exc:
        st = None
        digest = None
        readable = False
        sha256_status = "unreadable"
        blockers.append(f"unreadable:{exc}")
    text = sample_text(path, text_sample_bytes) if readable else ""
    rp = rel(path)
    try:
        relative_path = str(path.resolve().relative_to(root.resolve()))
    except Exception:
        relative_path = rp
    hay = f"{rp}\n{relative_path}\n{text[:text_sample_bytes]}"
    case_hay = f"{relative_path}\n{text[:text_sample_bytes]}"
    flags = {
        "contains_graph_terms": bool(GRAPH_RE.search(hay)),
        "contains_claim_evidence_terms": bool(CLAIM_EVIDENCE_RE.search(hay)),
        "contains_instruction_law": bool(INSTRUCTION_RE.search(hay)),
        "contains_river_terms": bool(RIVER_RE.search(hay)),
        "contains_diff_terms": bool(DIFF_RE.search(hay)),
        "contains_case_terms": bool(CASE_RE.search(hay)),
    }
    kind = kind_guess(path, text)
    duplicate_of = seen_hashes.get(digest or "") if digest else None
    if digest and not duplicate_of:
        seen_hashes[digest] = rp
    case_guess_value = guess_case(case_hay)
    lane = lane_guess(path, kind, hay, flags, case_guess_value)
    return {
        "schema": "lucidota.krampuschewing.master_index.row.v1",
        "source_label": source_label,
        "path": str(path.resolve()),
        "relative_path": relative_path,
        "repo_relative_path": rp,
        "sha256": digest,
        "sha256_status": sha256_status,
        "size_bytes": int(st.st_size) if st else None,
        "modified_time_utc": utc_from_ts(st.st_mtime) if st else None,
        "extension": path.suffix.lower() or "[none]",
        "kind_guess": kind,
        "lane_guess": lane,
        "case_guess": case_guess_value,
        "dev_project_guess": guess_dev_project(path),
        **flags,
        "duplicate_of": duplicate_of,
        "readable": readable,
        "blockers": blockers,
    }


def build_index(root_arg: str, max_files: int, text_sample_bytes: int, max_hash_bytes: int, index_out: Path | None, summary_out: Path | None, source_label: str | None = None) -> tuple[dict[str, Any], Path, Path]:
    root = (ROOT / root_arg).resolve() if not Path(root_arg).is_absolute() else Path(root_arg).resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    idx_path = index_out or OUT / f"krampuschewing_master_index_{stamp()}.jsonl"
    sum_path = summary_out or OUT / f"krampuschewing_master_summary_{stamp()}.json"
    if not idx_path.is_absolute():
        idx_path = ROOT / idx_path
    if not sum_path.is_absolute():
        sum_path = ROOT / sum_path
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    seen_hashes: dict[str, str] = {}
    blockers: list[str] = []
    files_seen = 0
    bytes_total = 0
    if not root.exists():
        blockers.append(f"root_missing:{root}")
    with idx_path.open("w", encoding="utf-8") as out:
        if root.exists():
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames.sort()
                for name in sorted(filenames):
                    if files_seen >= max_files:
                        blockers.append(f"max_files_reached:{max_files}")
                        break
                    path = Path(dirpath) / name
                    files_seen += 1
                    row = row_for(path, root, seen_hashes, text_sample_bytes, max_hash_bytes, source_label)
                    rows.append(row)
                    if isinstance(row.get("size_bytes"), int):
                        bytes_total += int(row["size_bytes"])
                    out.write(json.dumps(row, sort_keys=False, ensure_ascii=False) + "\n")
                if files_seen >= max_files:
                    break
    by_lane = Counter(str(r.get("lane_guess")) for r in rows)
    by_kind = Counter(str(r.get("kind_guess")) for r in rows)
    dups = sum(1 for r in rows if r.get("duplicate_of"))
    hash_skipped_large_files = sum(1 for r in rows if r.get("sha256_status") == "skipped_large_file")
    files_hashed = sum(1 for r in rows if r.get("sha256_status") == "computed")
    hash_status_counts = Counter(str(r.get("sha256_status")) for r in rows)
    largest = sorted([r for r in rows if isinstance(r.get("size_bytes"), int)], key=lambda r: int(r["size_bytes"]), reverse=True)[:20]
    dated = [r for r in rows if r.get("modified_time_utc")]
    oldest = sorted(dated, key=lambda r: str(r["modified_time_utc"]))[:20]
    newest = sorted(dated, key=lambda r: str(r["modified_time_utc"]), reverse=True)[:20]
    case_counts = Counter(str(r.get("case_guess")) for r in rows if r.get("case_guess"))
    project_counts = Counter(str(r.get("dev_project_guess")) for r in rows if r.get("dev_project_guess"))
    summary = {
        "schema": "lucidota.krampuschewing.master_summary.v1",
        "generated_at_utc": now(),
        "root": str(root),
        "source_label": source_label,
        "index_path": rel(idx_path),
        "summary_path": rel(sum_path),
        "files_seen": files_seen,
        "files_indexed": len(rows),
        "files_hashed": files_hashed,
        "hash_status_counts": dict(sorted(hash_status_counts.items())),
        "bytes_total": bytes_total,
        "duplicates": dups,
        "hash_skipped_large_files": hash_skipped_large_files,
        "max_hash_bytes": max_hash_bytes,
        "by_lane_guess": dict(sorted(by_lane.items())),
        "by_kind_guess": dict(sorted(by_kind.items())),
        "largest_files": [{"path": r.get("repo_relative_path"), "size_bytes": r.get("size_bytes"), "lane_guess": r.get("lane_guess")} for r in largest],
        "oldest_files": [{"path": r.get("repo_relative_path"), "modified_time_utc": r.get("modified_time_utc")} for r in oldest],
        "newest_files": [{"path": r.get("repo_relative_path"), "modified_time_utc": r.get("modified_time_utc")} for r in newest],
        "top_case_guesses": case_counts.most_common(30),
        "top_dev_project_guesses": project_counts.most_common(30),
        "blockers": blockers + ([f"sha256_skipped_large_files:{hash_skipped_large_files}"] if hash_skipped_large_files else []) + [b for r in rows for b in r.get("blockers", [])][:50],
        "canonical_graph_materialization": False,
        "canonical_graph_writes": False,
        "files_moved": [],
        "files_deleted": [],
        "verdict": "PASS" if not blockers and hash_skipped_large_files == 0 else "PARTIAL_FAIL",
    }
    sum_path.write_text(json.dumps(summary, indent=2, sort_keys=False, ensure_ascii=False), encoding="utf-8")
    return summary, idx_path, sum_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Build KRAMPUSCHEWING master index without moving/mutating files")
    ap.add_argument("--root", default="KRAMPUSCHEWING")
    ap.add_argument("--max-files", type=int, default=1_000_000)
    ap.add_argument("--text-sample-bytes", type=int, default=64_000)
    ap.add_argument("--max-hash-bytes", type=int, default=268_435_456, help="Skip full SHA256 for files larger than this byte count; use -1 to hash all files.")
    ap.add_argument("--index-output")
    ap.add_argument("--summary-output")
    ap.add_argument("--source-label")
    args = ap.parse_args()
    summary, idx, summ = build_index(
        args.root,
        args.max_files,
        args.text_sample_bytes,
        args.max_hash_bytes,
        Path(args.index_output) if args.index_output else None,
        Path(args.summary_output) if args.summary_output else None,
        args.source_label,
    )
    print("INDEX_PATH=" + rel(idx))
    print("SUMMARY_PATH=" + rel(summ))
    print("KRAMPUSCHEWING_MASTER_INDEX=" + summary["verdict"])
    print("FILES_INDEXED=" + str(summary["files_indexed"]))
    return 0 if summary["verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
