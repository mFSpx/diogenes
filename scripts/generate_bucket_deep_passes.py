#!/usr/bin/env python3
"""Generate per-bucket deep-pass artifacts from existing bucket pass markdown reports."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "05_OUTPUTS" / "tool_function_buckets"
MANIFEST_PATH = OUTPUT_DIR / "tool_function_bucket_manifest_latest.json"

ACTION_MAP = {
    "KEEP_CORE": "KEEP",
    "KEEP_SECONDARY": "KEEP",
    "KEEP": "KEEP",
    "KEEP_DOC": "KEEP_DOC",
    "ARCHIVE_OR_DOC": "KEEP_DOC",
    "COMPARE_FOR_SIMILARITY": "SIMILARITY_REVIEW",
    "REVIEW_NO_USE": "REVIEW_NO_USE",
    "REPAIR_NOW": "REPAIR_NOW",
    "UNKNOWN": "UNKNOWN",
}

SECTION_KEY_MAP = {
    "keep candidates": "keep",
    "review/no-use now": "review",
    "repair_now candidates": "repair",
}

STATUS_ORDER = [
    "KEEP_CORE",
    "KEEP_SECONDARY",
    "KEEP",
    "KEEP_DOC",
    "KEEP_OR_DOC",
    "ARCHIVE_OR_DOC",
    "COMPARE_FOR_SIMILARITY",
    "REPAIR_NOW",
    "REVIEW_NO_USE",
    "UNKNOWN",
]


DOC_EXTS = {
    ".md",
    ".markdown",
    ".txt",
    ".rst",
    ".pdf",
    ".html",
    ".htm",
    ".org",
    ".log",
    ".json",
    ".yaml",
    ".yml",
}

CODE_EXTS = {
    ".py",
    ".sh",
    ".bash",
    ".zsh",
    ".sql",
    ".rs",
    ".js",
    ".tsx",
    ".ts",
    ".go",
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".java",
    ".kt",
    ".r",
    ".jl",
    ".rb",
    ".pl",
}


def normalize_path(value: str) -> str:
    return str(value or "").strip("` \t")


def load_manifest_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        payload = json.load(fh)
    items = payload.get("items", []) if isinstance(payload, dict) else []
    out: dict[str, dict[str, Any]] = {}
    for row in items:
        p = normalize_path(str(row.get("path", "")).strip())
        if p:
            out[p] = row
    return out


def split_md_row(line: str, expected_cols: int | None = None) -> list[str]:
    text = line.strip()
    if not text.startswith("|"):
        return []
    text = text.strip("|")
    parts = [p.strip() for p in text.split("|")]
    if expected_cols is not None and len(parts) > expected_cols:
        head = parts[: expected_cols - 1]
        tail = "|".join(parts[expected_cols - 1 :]).strip()
        parts = head + [tail]
    return parts


def parse_table_section(section: str, lines: list[str], start: int):
    headers: list[str] | None = None
    i = start
    rows: list[dict[str, str]] = []

    while i < len(lines):
        line = lines[i].rstrip("\n")
        if line.startswith("## "):
            break
        if not line.strip():
            i += 1
            continue
        if not line.startswith("|"):
            i += 1
            continue

        if headers is None:
            if "|" not in line:
                i += 1
                continue
            headers = split_md_row(line)
            if not headers:
                i += 1
                continue
            # skip markdown separator row
            j = i + 1
            if j < len(lines) and lines[j].lstrip().startswith("|---"):
                i = j + 1
                continue
            i += 1
            continue

        if line.lstrip().startswith("|---"):
            i += 1
            continue

        row_parts = split_md_row(line, expected_cols=len(headers))
        if len(row_parts) != len(headers):
            i += 1
            continue
        row = {headers[idx]: val.strip() for idx, val in enumerate(row_parts)}
        rows.append(row)
        i += 1

    return rows, i


def parse_pass_markdown(pass_path: Path) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    lines = [ln.rstrip("\n") for ln in pass_path.read_text(encoding="utf-8", errors="ignore").splitlines()]

    status_counts: dict[str, int] = {}
    for line in lines:
        m = re.match(r"^-\s*([A-Z0-9_]+):\s*(\d+)", line)
        if m:
            status_counts[m.group(1)] = int(m.group(2))

    sections: dict[str, list[dict[str, str]]] = defaultdict(list)

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("## "):
            heading = line[3:].strip().lower()
            section_key = SECTION_KEY_MAP.get(heading)
            if section_key:
                rows, next_i = parse_table_section(section_key, lines, i + 1)
                sections[section_key] = rows
                i = next_i
                continue
        i += 1

    keep_rows = sections.get("keep", [])
    review_rows = sections.get("review", [])
    repair_rows = sections.get("repair", [])

    return status_counts, keep_rows, review_rows, repair_rows


def infer_status_scores(path: str, row: dict[str, str], manifest_row: dict[str, Any] | None, sig_count: int, is_truncated: bool = False) -> dict[str, float]:
    """Score row membership for each status class.

    This is a deterministic fallback when pass markdown rows are intentionally truncated.
    """

    p = path.lower()
    use = parse_int(row.get("use", manifest_row.get("known_use_count", 0) if manifest_row else 0))
    ext = Path(path).suffix.lower()
    is_doc = ext in DOC_EXTS or p.endswith(".ipynb") or "/books/" in p
    is_code = ext in CODE_EXTS or "/scripts/" in p or "/src/" in p or "/algos/" in p.lower()
    is_legacy = "/legacy/" in p
    is_test = "/tests/" in p or p.endswith("_test.py") or "/test_" in p
    is_tmp = "/tmp/" in p or "/temp/" in p or ".tmp." in p
    repeats = str(manifest_row.get("repeatability", "")) if manifest_row else ""
    base = Path(path).name.lower()

    # Conservative defaults that still behave the same for previously parsed rows.
    scores: dict[str, float] = {
        "KEEP_CORE": 0.0,
        "KEEP_SECONDARY": 0.0,
        "KEEP": 0.0,
        "ARCHIVE_OR_DOC": 0.0,
        "COMPARE_FOR_SIMILARITY": 0.0,
        "REPAIR_NOW": 0.0,
        "REVIEW_NO_USE": 0.0,
        "UNKNOWN": 0.0,
    }

    scores["KEEP_CORE"] = 0
    if use >= 10:
        scores["KEEP_CORE"] += 90
    elif use >= 7:
        scores["KEEP_CORE"] += 65
    elif use >= 5:
        scores["KEEP_CORE"] += 35
    if repeats == "REPEATABLE_STANDARD":
        scores["KEEP_CORE"] += 45
    if is_code and use > 0:
        scores["KEEP_CORE"] += 20
    if sig_count > 1:
        scores["KEEP_CORE"] -= 25

    scores["KEEP_SECONDARY"] = 0
    if 2 <= use <= 6:
        scores["KEEP_SECONDARY"] += 80
    elif use == 1:
        scores["KEEP_SECONDARY"] += 30
    if repeats in {"REPEATABLE_LIMITED", "REPEATABLE_STANDARD"}:
        scores["KEEP_SECONDARY"] += 20
    if is_code:
        scores["KEEP_SECONDARY"] += 15
    if is_legacy:
        scores["KEEP_SECONDARY"] -= 20

    scores["KEEP"] = 0
    if 1 <= use <= 4:
        scores["KEEP"] += 70
    elif use == 0 and is_code and not is_legacy:
        scores["KEEP"] += 15
    if repeats in {"REPEATABLE_LIMITED", "REPEATABLE_STANDARD"}:
        scores["KEEP"] += 25
    if is_code and sig_count == 1:
        scores["KEEP"] += 10
    if is_test and use >= 1:
        scores["KEEP"] -= 10

    scores["ARCHIVE_OR_DOC"] = 0
    if is_doc:
        scores["ARCHIVE_OR_DOC"] += 110
    if use <= 1:
        scores["ARCHIVE_OR_DOC"] += 20
    if ext in {".md", ".txt", ".rst", ".html", ".htm"}:
        scores["ARCHIVE_OR_DOC"] += 20
    if is_legacy:
        scores["ARCHIVE_OR_DOC"] += 15

    scores["COMPARE_FOR_SIMILARITY"] = 0
    if sig_count > 1:
        scores["COMPARE_FOR_SIMILARITY"] += 90
        scores["COMPARE_FOR_SIMILARITY"] += min(sig_count - 1, 20) * 1.8
    if is_code and use <= 2:
        scores["COMPARE_FOR_SIMILARITY"] += 25
    if is_test:
        scores["COMPARE_FOR_SIMILARITY"] += 15
    if repeats == "UNKNOWN_PRACTICAL_SCOPE" and use <= 2:
        scores["COMPARE_FOR_SIMILARITY"] += 10
    if is_doc:
        scores["COMPARE_FOR_SIMILARITY"] -= 20

    scores["REPAIR_NOW"] = 0
    if is_tmp or "oneoff" in p:
        scores["REPAIR_NOW"] += 55
    if "repair" in base or "repair" in p:
        scores["REPAIR_NOW"] += 25
    if is_legacy and use <= 1:
        scores["REPAIR_NOW"] += 25
    if repeats == "UNKNOWN_PRACTICAL_SCOPE":
        scores["REPAIR_NOW"] += 15
    if use == 0:
        scores["REPAIR_NOW"] -= 10

    scores["REVIEW_NO_USE"] = 0
    if use == 0:
        scores["REVIEW_NO_USE"] += 100
    if is_tmp:
        scores["REVIEW_NO_USE"] += 30
    if is_legacy:
        scores["REVIEW_NO_USE"] += 20
    if repeats in {"UNKNOWN_PRACTICAL_SCOPE", "ONE_TIME_LIKELY"}:
        scores["REVIEW_NO_USE"] += 15
    if is_code:
        scores["REVIEW_NO_USE"] += 10
    if is_doc and ext in {".json", ".js"}:
        scores["REVIEW_NO_USE"] -= 8

    if is_truncated:
        # avoid overconfident guesses when source source rows were truncated
        for key in scores:
            scores[key] = scores[key] * 0.97

    return scores


def allocate_statuses_to_bucket_rows(
    rows: list[dict[str, Any]],
    status_targets: dict[str, int],
    explicit_assignments: dict[str, str],
    manifest_rows: dict[str, dict[str, Any]],
    sig_counts: dict[str, int],
    is_truncated: bool = False,
) -> dict[str, str]:
    """Assign status to rows not covered in explicit pass tables, honoring target counts."""
    assigned = dict(explicit_assignments)
    bucket_counts = {k: 0 for k in STATUS_ORDER}
    for value in assigned.values():
        if value in bucket_counts:
            bucket_counts[value] += 1
    remaining = {k: status_targets.get(k, 0) - bucket_counts.get(k, 0) for k in status_targets}

    # clamp negative leftovers if explicit rows exceed parsed counts
    for k in remaining:
        if remaining[k] < 0:
            remaining[k] = 0

    unassigned = [row for row in rows if normalize_path(row["path"]) not in assigned]
    if not unassigned:
        return assigned

    ordered_statuses = list(remaining.keys())
    if not ordered_statuses:
        ordered_statuses = ["KEEP_CORE", "KEEP_SECONDARY", "KEEP", "ARCHIVE_OR_DOC", "COMPARE_FOR_SIMILARITY", "REPAIR_NOW", "REVIEW_NO_USE", "UNKNOWN"]

    # First pass: allocate explicit buckets to the strongest candidates.
    for status in ordered_statuses:
        need = remaining.get(status, 0)
        if need <= 0:
            continue
        candidates = []
        for row in unassigned:
            p = normalize_path(row["path"])
            if p in assigned:
                continue
            score = infer_status_scores(p, row, manifest_rows.get(p), sig_counts.get(p, 1), is_truncated=is_truncated).get(status, 0.0)
            candidates.append((score, p))
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        selected = [p for _, p in candidates[:need]]
        for p in selected:
            if p in assigned:
                continue
            assigned[p] = status
            remaining[status] -= 1

    # Secondary pass: anything still unassigned gets REVIEW_NO_USE first, then REVIEW fallback.
    # This can happen if status totals don't sum exactly.
    final_status = "REVIEW_NO_USE"
    if sum(remaining.values()) > 0:
        for row in list(unassigned):
            p = normalize_path(row["path"])
            if p not in assigned:
                assigned[p] = final_status if sum(remaining.values()) > 0 else "REVIEW_NO_USE"
                # drain remaining on a best-effort basis
                if sum(remaining.values()) > 0:
                    chosen = final_status if remaining.get(final_status, 0) > 0 else next((s for s, v in remaining.items() if v > 0), final_status)
                    if chosen:
                        remaining[chosen] = max(0, remaining.get(chosen, 0) - 1)

    # Last-pass clean-up for underfilled explicit targets: leave assigned status as-is and set UNKNOWN.
    return assigned


def parse_int(value: str, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return default


def infer_signature(row: dict[str, str], manifest_row: dict[str, Any], path: str) -> str:
    explicit_sig = (row.get("signature") or "").strip()
    if explicit_sig:
        return explicit_sig
    role = (manifest_row.get("role") or "").strip() if manifest_row else ""
    if role:
        return f"role::{role}"
    base = Path(path).name
    if base:
        stem = Path(base).stem
        return f"name::{stem}"
    return "name::unknown"


def infer_necessity(row: dict[str, str], status: str, manifest_row: dict[str, Any]) -> str:
    raw = row.get("necessity", "").strip().lower()
    if raw:
        return raw
    if status in {"KEEP_CORE", "KEEP_SECONDARY", "KEEP"}:
        return "high"
    if status == "ARCHIVE_OR_DOC":
        return "high"
    if status == "COMPARE_FOR_SIMILARITY":
        return "med"
    if status == "REPAIR_NOW":
        return "med"
    return "low"


def infer_speed(row: dict[str, str], status: str) -> str:
    raw = row.get("speed", "").strip()
    if raw:
        return raw
    if status == "REPAIR_NOW":
        return "Likely"
    if status == "COMPARE_FOR_SIMILARITY":
        return "Likely"
    if status in {"ARCHIVE_OR_DOC", "KEEP_CORE", "KEEP_SECONDARY", "KEEP"}:
        return row.get("speed", "") or "Likely"
    return "unknown"


def infer_quality(row: dict[str, str], status: str) -> str:
    raw = row.get("quality", "").strip()
    if raw:
        return raw
    if status == "REPAIR_NOW":
        return "Medium"
    if status == "COMPARE_FOR_SIMILARITY":
        return "Medium"
    if status in {"KEEP_CORE", "KEEP_SECONDARY", "KEEP", "ARCHIVE_OR_DOC"}:
        return "unknown"
    return "unknown"


def format_last_ref_list(values: list[str], limit: int = 6) -> str:
    if not values:
        return ""
    return str(values[:limit])


def get_mtime_iso(path_str: str) -> str:
    p = ROOT / path_str
    if p.exists() and p.is_file():
        dt = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return ""


def infer_last_refs(manifest_row: dict[str, Any] | None) -> tuple[list[str], str]:
    known_uses = manifest_row.get("known_uses", []) if isinstance(manifest_row, dict) else []
    if not isinstance(known_uses, list):
        return [], ""

    if not known_uses:
        return known_uses, ""

    mtimes = [get_mtime_iso(v) for v in known_uses]
    mtimes = [v for v in mtimes if v]
    if mtimes:
        return known_uses, max(mtimes)
    return known_uses, ""


def build_deep_row(
    bucket: str,
    path: str,
    status: str,
    pass_row: dict[str, str],
    manifest_row: dict[str, Any] | None,
    raw_mtime_fallback: str = "",
    uniq_like: str = "",
) -> dict[str, Any]:
    manifest = manifest_row or {}
    use = parse_int(pass_row.get("use", manifest.get("known_use_count", 0)))
    sig = infer_signature(pass_row, manifest, path)
    purpose_inferred = (
        (pass_row.get("purpose") or pass_row.get("rationale") or "")
        .strip()
    )
    if not purpose_inferred:
        purpose_inferred = manifest.get("what_it_does", "") or ""

    role = manifest.get("role", "") or pass_row.get("role", "") or pass_row.get("signature", "") or ""

    last_ref, inferred_last_used = infer_last_refs(manifest)
    raw_last_used = inferred_last_used
    if not raw_last_used and raw_mtime_fallback:
        raw_last_used = raw_mtime_fallback

    return {
        "bucket": bucket,
        "path": path,
        "status": status,
        "action": ACTION_MAP.get(status, status),
        "necessity_tier": infer_necessity(pass_row, status, manifest),
        "speed_tier": infer_speed(pass_row, status),
        "quality_tier": infer_quality(pass_row, status),
        "sig": sig,
        "known_use_count": use,
        "repeatability": manifest.get("repeatability", "UNKNOWN_PRACTICAL_SCOPE"),
        "role": role,
        "purpose_inferred": purpose_inferred,
        "raw_mtime": raw_mtime_fallback,
        "raw_last_used": raw_last_used,
        "last_ref": last_ref,
        "known_ref_count": len(last_ref),
        "uniq_like": uniq_like,
        "what_it_does": purpose_inferred,
    }


def generate_for_bucket(bucket: str, pass_path: Path, manifest_rows: dict[str, dict[str, Any]], date_tag: str):
    status_counts, keep_rows, review_rows, repair_rows = parse_pass_markdown(pass_path)

    bucket_manifest_rows = [row for row in manifest_rows.values() if str(row.get("function_bucket", "")).upper() == bucket]
    bucket_manifest_paths: list[str] = []
    bucket_manifest_lookup: dict[str, dict[str, Any]] = {}
    for row in bucket_manifest_rows:
        p = normalize_path(str(row.get("path", "")))
        if not p or p in bucket_manifest_lookup:
            continue
        bucket_manifest_lookup[p] = row
        bucket_manifest_paths.append(p)

    # Preserve all explicit pass-table guidance, including any statuses not represented
    # in the bucket status split header.
    explicit_rows: dict[str, dict[str, Any]] = {}
    explicit_assignments: dict[str, str] = {}

    for r in keep_rows:
        path = normalize_path(r.get("path", ""))
        if not path:
            continue
        status = r.get("status", "UNKNOWN").strip() or "UNKNOWN"
        manifest = manifest_rows.get(path, {})
        explicit_row = build_deep_row(bucket, path, status, r, manifest, raw_mtime_fallback=r.get("mtime", "").strip(), uniq_like=r.get("uniqueness", ""))
        if "purpose_inferred" in explicit_row and not explicit_row["purpose_inferred"]:
            explicit_row["purpose_inferred"] = r.get("purpose", "") or r.get("rationale", "") or manifest.get("what_it_does", "") or ""
            explicit_row["what_it_does"] = explicit_row["purpose_inferred"]
        explicit_rows[path] = explicit_row
        explicit_assignments[path] = status

    for r in review_rows:
        path = normalize_path(r.get("path", ""))
        if not path:
            continue
        status = "REVIEW_NO_USE"
        manifest = manifest_rows.get(path, {})
        explicit_row = build_deep_row(bucket, path, status, r, manifest, raw_mtime_fallback=r.get("mtime", "").strip(), uniq_like=r.get("signature", ""))
        explicit_rows[path] = explicit_row
        explicit_assignments[path] = status

    for r in repair_rows:
        path = normalize_path(r.get("path", ""))
        if not path:
            continue
        status = "REPAIR_NOW"
        manifest = manifest_rows.get(path, {})
        explicit_row = build_deep_row(bucket, path, status, r, manifest, raw_mtime_fallback="", uniq_like=r.get("role", ""))
        explicit_rows[path] = explicit_row
        explicit_assignments[path] = status

    # If this pass markdown table was truncated, backfill the remaining manifest rows.
    parsed_count = len(keep_rows) + len(review_rows) + len(repair_rows)
    pass_status_total = sum(status_counts.values())
    is_truncated = bool(status_counts) and parsed_count < pass_status_total

    status_targets = Counter({k: int(v) for k, v in status_counts.items()})
    explicit_status_counts = Counter(explicit_assignments.values())
    for status, explicit_count in explicit_status_counts.items():
        if status_targets.get(status, 0) < explicit_count:
            status_targets[status] = explicit_count

    # If header targets are narrower than manifest cardinality, fill missing with UNKNOWN.
    manifest_count = len(bucket_manifest_paths)
    status_target_total = sum(status_targets.values())
    if manifest_count > status_target_total:
        status_targets["UNKNOWN"] += manifest_count - status_target_total
    if manifest_count > 0:
        status_targets = Counter({k: int(v) for k, v in status_targets.items() if v > 0})

    manifest_alloc_rows = []
    for path in bucket_manifest_paths:
        manifest_row = bucket_manifest_lookup.get(path, {})
        manifest_alloc_rows.append({"path": path, "use": str(manifest_row.get("known_use_count", 0))})
    # Also include explicit rows not in this manifest slice so they stay represented.
    for path in explicit_assignments:
        if path not in bucket_manifest_lookup:
            manifest_alloc_rows.append({"path": path})

    # signature frequencies for allocation
    sig_counts = Counter()
    for item in bucket_manifest_rows:
        p = normalize_path(item.get("path", ""))
        if not p:
            continue
        manifest = bucket_manifest_lookup.get(p, {})
        sig = infer_signature({}, manifest, p)
        sig_counts[sig] += 1
    explicit_not_in_manifest = set(explicit_assignments) - set(bucket_manifest_lookup)
    for path in explicit_assignments:
        if path in explicit_not_in_manifest:
            sig_counts[infer_signature({}, manifest_rows.get(path, {}), path)] += 1

    assigned_statuses = allocate_statuses_to_bucket_rows(
        rows=manifest_alloc_rows,
        status_targets=dict(status_targets),
        explicit_assignments=explicit_assignments,
        manifest_rows=bucket_manifest_lookup,
        sig_counts=sig_counts,
        is_truncated=is_truncated,
    )

    # Build final rows from manifest with assigned status, preferring pass-provided explicit fields.
    rows: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for path in bucket_manifest_paths:
        normalized_path = normalize_path(path)
        if not normalized_path:
            continue
        manifest = bucket_manifest_lookup.get(normalized_path, {})
        if normalized_path in seen_paths:
            continue
        status = assigned_statuses.get(normalized_path, "UNKNOWN")
        if normalized_path in explicit_rows:
            row = explicit_rows[normalized_path]
            row["status"] = status
            row["action"] = ACTION_MAP.get(status, status)
        else:
            row = build_deep_row(bucket, normalized_path, status, {}, manifest, raw_mtime_fallback="")
        row["path"] = normalized_path
        row["role"] = manifest.get("role", row.get("role", "")) or row.get("role", "")
        row["what_it_does"] = row.get("purpose_inferred", row.get("what_it_does", ""))
        rows.append(row)
        seen_paths.add(normalized_path)

    # Keep any explicit rows not present in manifest as-is (best effort for stale manifests).
    for path, explicit_row in explicit_rows.items():
        normalized_path = normalize_path(path)
        if normalized_path in seen_paths:
            continue
        rows.append(explicit_row)

    # Signature duplicate accounting
    sig_counts = Counter(r["sig"] for r in rows if r.get("sig"))
    for r in rows:
        r["sig_count"] = int(sig_counts.get(r.get("sig", ""), 0))
        r["dup_like_count"] = max(0, r["sig_count"] - 1)
        r["unique_signal"] = "YES" if r["sig_count"] <= 1 else "NO"
        r["last_ref_count"] = int(r.get("known_ref_count", 0))
        r["last_ref"] = format_last_ref_list(r.get("last_ref", []))
        r["last_ref_mtime"] = r.get("raw_last_used", "")

    # Build deep CSV/MD
    bucket_safe = bucket.lower()
    deep_csv = OUTPUT_DIR / f"{bucket_safe}_function_deep_pass_{date_tag}.csv"
    deep_md = OUTPUT_DIR / f"{bucket_safe}_function_deep_pass_{date_tag}.md"

    csv_rows = []
    for r in rows:
        csv_rows.append(
            {
                "path": r["path"],
                "status": r["status"],
                "action": r["action"],
                "necessity_tier": r["necessity_tier"],
                "speed_tier": r["speed_tier"],
                "quality_tier": r["quality_tier"],
                "unique_signal": r["unique_signal"],
                "sig": r["sig"],
                "sig_count": r["sig_count"],
                "dup_like_count": r["dup_like_count"],
                "known_use_count": r["known_use_count"],
                "repeatability": r["repeatability"],
                "role": r["role"],
                "purpose_inferred": r["purpose_inferred"],
                "last_ref_count": r["last_ref_count"],
                "last_ref": r["last_ref"],
                "last_ref_mtime": r["last_ref_mtime"],
                "git_last_commit": r.get("git_last_commit", ""),
                "what_it_does": r["what_it_does"],
            }
        )

    with deep_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "path",
                "status",
                "action",
                "necessity_tier",
                "speed_tier",
                "quality_tier",
                "unique_signal",
                "sig",
                "sig_count",
                "dup_like_count",
                "known_use_count",
                "repeatability",
                "role",
                "purpose_inferred",
                "last_ref_count",
                "last_ref",
                "last_ref_mtime",
                "git_last_commit",
                "what_it_does",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    top_clusters = [
        (sig, count) for sig, count in sig_counts.items() if sig and count > 1
    ]
    top_clusters.sort(key=lambda kv: kv[1], reverse=True)

    row_status_counts = Counter(r["status"] for r in rows)
    lines: list[str] = [
        f"# {bucket} deep pass (operational review)",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "",
        "## Bucket-wide findings",
        f"- Total entries: {len(rows)}",
    ]
    ordered_status_keys = [k for k in STATUS_ORDER if row_status_counts.get(k, 0)]
    ordered_status_keys.extend(
        [k for k in sorted(row_status_counts.keys()) if k not in set(STATUS_ORDER)]
    )
    for k in ordered_status_keys:
        if row_status_counts.get(k, 0):
            lines.append(f"- {k}: {row_status_counts[k]}")

    unique_count = sum(1 for c in sig_counts.values() if c == 1)
    non_unique = len(rows) - unique_count
    lines.extend([
        "",
        "## Key quality/speed/uniqueness signals",
        f"- Unique-function entries: {unique_count}",
        f"- Non-unique (duplicate signal) entries: {non_unique} (largest cluster: {top_clusters[0][0] if top_clusters else 'n/a'})",
        f"- Duplicate clusters count: {len(top_clusters)}",
    ])
    if top_clusters:
        lines.append("- Top duplicate clusters:")
        for sig, count in top_clusters[:8]:
            lines.append(f"  - {sig}: {count}")

    for status in ordered_status_keys:
        subset = [r for r in rows if r["status"] == status]
        if not subset:
            continue
        lines.extend([
            "",
            f"## {status} ({len(subset)})",
            "",
            "|path|use|uniq_like|similar_count|needs/quality/speed|action|last_used_hint|what_it_does|",
            "|---|---:|---|---:|---|---|---|---|",
        ])
        for r in subset:
            use = r["known_use_count"]
            uniq_like = r["unique_signal"]
            similar = r["sig_count"] - 1
            nq = f"{r['necessity_tier']}/{r['quality_tier']}/{r['speed_tier']}"
            what = (r["what_it_does"] or "").replace("|", "/")
            last = r["last_ref_mtime"] if r["last_ref_mtime"] else "-"
            lines.append(f"| {r['path']} | {use} | {uniq_like} | {similar} | {nq} | {r['action']} | {last} | {what} |")

    lines.extend([
        "",
        "## Similarity clusters (signal frequency >1)",
    ])
    if top_clusters:
        for sig, count in top_clusters:
            lines.append(f"- {sig}: {count}")
    else:
        lines.append("- (none)")

    deep_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    row_count = len(rows)
    return {
        "bucket": bucket,
        "status_counts": dict(row_status_counts),
        "total": row_count,
        "deep_csv": str(deep_csv.relative_to(ROOT)),
        "deep_md": str(deep_md.relative_to(ROOT)),
        "row_count": row_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deep pass files from existing bucket pass markdowns")
    parser.add_argument("--bucket-dir", default=str(OUTPUT_DIR), help="Directory containing *_bucket_pass_*.md files")
    parser.add_argument("--manifest", default=str(MANIFEST_PATH), help="Path to tool_function_bucket_manifest_latest.json")
    parser.add_argument("--only", nargs="*", default=[], help="Optional bucket names to process")
    args = parser.parse_args()

    manifest_rows = load_manifest_rows(Path(args.manifest))
    bucket_dir = Path(args.bucket_dir)

    pass_paths = sorted(bucket_dir.glob("*_bucket_pass_*.md"))
    if args.only:
        only_set = {x.upper() for x in args.only}
        pass_paths = [p for p in pass_paths if p.name.split("_bucket_pass_")[0].upper() in only_set]

    summary = []
    for p in pass_paths:
        m = re.search(r"_bucket_pass_(\d{8}T\d{6}\d+Z)\.md$", p.name)
        date_tag = m.group(1) if m else datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        bucket = p.name.split("_bucket_pass_")[0].upper()
        result = generate_for_bucket(bucket, p, manifest_rows, date_tag)
        summary.append(result)

    for item in summary:
        print(f"BUCKET_PASS_GENERATED={item['bucket']} csv={item['deep_csv']} md={item['deep_md']} rows={item['row_count']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
