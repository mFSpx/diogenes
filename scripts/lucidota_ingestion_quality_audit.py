#!/usr/bin/env python3
"""Audit corpus chunks before BGE embedding.

This is the pre-embed sieve for LUCIDOTA's corpus backlog.  It answers three
questions with receipts:

1. What is actually queued for embedding?
2. Does a stratified >=1/300 sample look like readable material?
3. Which chunks must be blocked/reparsed before any embedding worker touches them?
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import psycopg2
import psycopg2.extras

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "05_OUTPUTS" / "ingestion_audit"
DEFAULT_STORAGE_DSN = "postgresql:///lucidota_storage"
DEFAULT_STATE_DSN = "postgresql:///lucidota_state"

RAW_EMAIL_RE = re.compile(
    r"(Content-Transfer-Encoding:|Content-Type:\s*text/html|Mime-Version:|"
    r"protonmail_signature_block|class=3D|</?div\b|=\r?\n|=C2=A0|=3D)",
    re.I,
)
LONG_BASE64_RE = re.compile(r"[A-Za-z0-9+/]{180,}={0,2}")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_'-]{2,}")
ARCHIVE_EXT_RE = re.compile(r"\.(zip|tar|tgz|tar\.gz|7z|rar|gz|bz2|xz)$", re.I)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def embedding_quality_sql_where() -> str:
    """SQL predicate shared by audit/enqueuer/worker for eligible backlog rows."""
    return (
        "embedding IS NULL "
        "AND COALESCE(go25->>'embedding_quality_status', '') <> 'block'"
    )


def _printable_ratio(text: str) -> float:
    if not text:
        return 0.0
    printable = 0
    for ch in text:
        if ch.isprintable() or ch in "\n\r\t":
            printable += 1
    return printable / len(text)


def _word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def _base64_hit(text: str) -> bool:
    return bool(LONG_BASE64_RE.search(text))


def classify_readability(
    content: str,
    *,
    mime: str | None = "",
    source_path: str | None = "",
) -> dict[str, Any]:
    """Return pass/block + reasons for whether text is safe to embed as-is."""
    text = content or ""
    stripped = text.strip()
    mime = mime or ""
    source_path = source_path or ""
    length = len(text)
    printable_ratio = _printable_ratio(text)
    words = _word_count(text)
    alpha_ratio = (sum(1 for c in text if c.isalpha()) / length) if length else 0.0
    qp_email_artifacts = len(RAW_EMAIL_RE.findall(text))
    base64_like = _base64_hit(text)

    reasons: list[str] = []
    warnings: list[str] = []

    if length < 50:
        reasons.append("too_short")
    if "\x00" in text or printable_ratio < 0.95:
        reasons.append("nonprintable_text")
    if base64_like and alpha_ratio < 0.55:
        reasons.append("base64_or_encoded_blob")
    if (mime == "text/eml" or source_path.lower().endswith(".eml") or ".eml" in source_path.lower()) and qp_email_artifacts:
        reasons.append("raw_email_headers_or_html")
    elif qp_email_artifacts >= 4:
        warnings.append("quoted_printable_or_html_artifacts")
    if ARCHIVE_EXT_RE.search(source_path.split("!")[-1] or source_path):
        reasons.append("archive_member_not_unpacked")
    if words < 5 and not any(mime.endswith(x) for x in ("json", "py", "rs", "sql", "csv", "toml", "yml", "yaml")):
        reasons.append("too_few_words")

    status = "block" if reasons else "pass"
    return {
        "status": status,
        "reasons": sorted(set(reasons)),
        "warnings": sorted(set(warnings)),
        "metrics": {
            "length": length,
            "printable_ratio": round(printable_ratio, 4),
            "alpha_ratio": round(alpha_ratio, 4),
            "word_count": words,
            "qp_email_artifacts": qp_email_artifacts,
            "base64_like": base64_like,
        },
    }


def archive_inventory() -> dict[str, Any]:
    krampus = ROOT / "KRAMPUSCHEWING"
    archives: list[dict[str, Any]] = []
    if krampus.exists():
        for p in sorted(krampus.iterdir()):
            if p.is_file() and ARCHIVE_EXT_RE.search(p.name):
                archives.append(
                    {
                        "path": rel(p),
                        "size_bytes": p.stat().st_size,
                        "size_gib": round(p.stat().st_size / (1024**3), 3),
                        "mtime_utc": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc)
                        .isoformat()
                        .replace("+00:00", "Z"),
                    }
                )
    unpacked_root = ROOT / "09_STORAGE" / "krampuschewing_unpacked"
    unpacked_files = 0
    unpacked_dirs = 0
    if unpacked_root.exists():
        for p in unpacked_root.rglob("*"):
            if p.is_dir():
                unpacked_dirs += 1
            elif p.is_file():
                unpacked_files += 1
    receipts = sorted((ROOT / "05_OUTPUTS").glob("**/*archive_unpack*"), key=lambda p: p.stat().st_mtime if p.exists() else 0)
    disk = shutil.disk_usage(ROOT)
    return {
        "archives_present": archives,
        "archives_present_count": len(archives),
        "archives_present_total_gib": round(sum(a["size_bytes"] for a in archives) / (1024**3), 3),
        "persistent_unpacked_dirs": unpacked_dirs,
        "persistent_unpacked_files": unpacked_files,
        "archive_unpack_receipts_count": len([p for p in receipts if p.is_file()]),
        "recent_archive_unpack_receipts": [rel(p) for p in receipts[-10:] if p.is_file()],
        "disk": {
            "root": rel(ROOT),
            "total_gib": round(disk.total / (1024**3), 2),
            "used_gib": round(disk.used / (1024**3), 2),
            "free_gib": round(disk.free / (1024**3), 2),
            "used_pct": round(disk.used / disk.total * 100, 2),
        },
    }


def queue_snapshot(state_dsn: str) -> dict[str, Any]:
    try:
        conn = psycopg2.connect(state_dsn)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT status, count(*) AS count
                FROM lucidota_control.absurd_queue_job
                WHERE job_kind='embed_fill_batch'
                GROUP BY status
                ORDER BY status
                """
            )
            status_counts = [dict(r) for r in cur.fetchall()]
            cur.execute(
                """
                SELECT job_uuid::text, status, idempotency_key, payload, created_at::text
                FROM lucidota_control.absurd_queue_job
                WHERE job_kind='embed_fill_batch'
                ORDER BY created_at DESC
                LIMIT 10
                """
            )
            recent = [dict(r) for r in cur.fetchall()]
        return {"status_counts": status_counts, "recent": recent}
    finally:
        conn.close()


def iter_null_rows(conn, only_eligible: bool) -> Iterable[dict[str, Any]]:
    where = embedding_quality_sql_where() if only_eligible else "embedding IS NULL"
    with conn.cursor(name="lucidota_ingestion_audit_cursor", cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.itersize = 1000
        cur.execute(
            f"""
            SELECT chunk_uuid::text, source_path, mime, extractor, embedding_model,
                   chunk_index, content, created_at::text, go25
            FROM lucidota_korpus.corpus_chunk
            WHERE {where}
            ORDER BY created_at, chunk_uuid
            """
        )
        for row in cur:
            yield dict(row)


def collect_archive_prefix_counts(conn) -> list[dict[str, Any]]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT split_part(source_path, '!', 1) AS archive_prefix,
                   count(*) AS chunks,
                   count(DISTINCT source_path) AS sources,
                   count(*) FILTER (WHERE embedding IS NULL) AS nulls
            FROM lucidota_korpus.corpus_chunk
            WHERE source_path LIKE '%.zip!%' OR source_path LIKE '%.7z!%' OR source_path LIKE '%.tar!%'
            GROUP BY 1
            ORDER BY chunks DESC
            LIMIT 60
            """
        )
        return [dict(r) for r in cur.fetchall()]


def run_audit(
    *,
    storage_dsn: str,
    state_dsn: str,
    sample_stride: int,
    seed: int,
    only_eligible: bool,
    quarantine_bad: bool,
) -> dict[str, Any]:
    conn = psycopg2.connect(storage_dsn)
    rng = random.Random(seed)
    sample_by_bucket: dict[int, dict[str, Any]] = {}
    bad_updates: list[tuple[str, str]] = []
    counts = Counter()
    extractor_counts: Counter[str] = Counter()
    mime_counts: Counter[str] = Counter()
    source_mode_counts = Counter()
    reason_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    lengths: list[int] = []
    examples_by_reason: dict[str, list[dict[str, Any]]] = defaultdict(list)

    try:
        archive_prefix_counts = collect_archive_prefix_counts(conn)
        for idx, row in enumerate(iter_null_rows(conn, only_eligible=only_eligible)):
            source_path = str(row.get("source_path") or "")
            mime = str(row.get("mime") or "")
            extractor = str(row.get("extractor") or "")
            quality = classify_readability(str(row.get("content") or ""), mime=mime, source_path=source_path)
            status = quality["status"]
            counts["total_scanned"] += 1
            counts[f"quality_{status}"] += 1
            extractor_counts[extractor] += 1
            mime_counts[mime] += 1
            if "!" in source_path:
                source_mode_counts["virtual_archive_member"] += 1
            elif source_path.startswith("KRAMPUSCHEWING/") or "/KRAMPUSCHEWING/" in source_path:
                source_mode_counts["krampus_file_path"] += 1
            elif source_path.startswith("cas://"):
                source_mode_counts["cas"] += 1
            elif not source_path:
                source_mode_counts["blank"] += 1
            else:
                source_mode_counts["other"] += 1
            length = int(quality["metrics"]["length"])
            lengths.append(length)
            for reason in quality["reasons"]:
                reason_counts[reason] += 1
                if len(examples_by_reason[reason]) < 5:
                    examples_by_reason[reason].append(
                        {
                            "chunk_uuid": row["chunk_uuid"],
                            "source_path": source_path,
                            "mime": mime,
                            "extractor": extractor,
                            "chunk_index": row["chunk_index"],
                            "preview": str(row.get("content") or "")[:700].replace("\r", "\\r").replace("\n", " ⏎ "),
                            "metrics": quality["metrics"],
                        }
                    )
            for warning in quality["warnings"]:
                warning_counts[warning] += 1
            if status == "block":
                bad_updates.append((row["chunk_uuid"], json.dumps({
                    "embedding_quality_status": "block",
                    "embedding_quality_reasons": quality["reasons"],
                    "embedding_quality_audited_at": now_iso(),
                    "embedding_quality_audit": "scripts/lucidota_ingestion_quality_audit.py",
                }, sort_keys=True)))

            bucket = idx // sample_stride
            seen_in_bucket = counts[f"bucket_{bucket}_seen"] + 1
            counts[f"bucket_{bucket}_seen"] = seen_in_bucket
            if bucket not in sample_by_bucket or rng.randrange(seen_in_bucket) == 0:
                sample_by_bucket[bucket] = {
                    "sample_bucket": bucket,
                    "ordinal_floor": bucket * sample_stride,
                    "chunk_uuid": row["chunk_uuid"],
                    "source_path": source_path,
                    "mime": mime,
                    "extractor": extractor,
                    "embedding_model": row.get("embedding_model"),
                    "chunk_index": row.get("chunk_index"),
                    "quality": quality,
                    "preview": str(row.get("content") or "")[:900].replace("\r", "\\r").replace("\n", " ⏎ "),
                }

        quarantined = 0
        if quarantine_bad and bad_updates:
            with conn.cursor() as cur:
                psycopg2.extras.execute_batch(
                    cur,
                    """
                    UPDATE lucidota_korpus.corpus_chunk
                    SET go25 = go25 || %s::jsonb
                    WHERE chunk_uuid = %s::uuid
                    """,
                    [(payload, uid) for uid, payload in bad_updates],
                    page_size=1000,
                )
                # psycopg2 execute_batch rowcount can report only the last batch.
                # The intended receipt value is the attempted blocked-row count;
                # a follow-up SQL count is used by verification.
                quarantined = len(bad_updates)
            conn.commit()

        samples = [sample_by_bucket[k] for k in sorted(sample_by_bucket)]
        length_stats = {}
        if lengths:
            length_stats = {
                "min": min(lengths),
                "median": int(statistics.median(lengths)),
                "max": max(lengths),
            }
        verdict = "PASS"
        blockers: list[str] = []
        if counts["quality_block"]:
            verdict = "BLOCK_EMBED_DRAIN"
            blockers.append("blocked_quality_rows_present")
        archives = archive_inventory()
        if archives["archives_present_count"] and archives["persistent_unpacked_files"] == 0:
            blockers.append("archives_present_without_persistent_unpacked_manifest")
            verdict = "BLOCK_EMBED_DRAIN"

        report = {
            "schema": "lucidota.ingestion_quality_audit.v1",
            "generated_at": now_iso(),
            "sample_policy": {
                "stride": sample_stride,
                "requirement": "one random sample per stride bucket; default satisfies >=1/300 chunks",
                "seed": seed,
                "sample_count": len(samples),
            },
            "db": {"storage_dsn": storage_dsn, "state_dsn": state_dsn},
            "scan": {
                "only_eligible_input": only_eligible,
                "counts": {k: v for k, v in counts.items() if not k.startswith("bucket_")},
                "length_stats": length_stats,
                "extractor_counts": dict(extractor_counts.most_common()),
                "mime_counts": dict(mime_counts.most_common()),
                "source_mode_counts": dict(source_mode_counts.most_common()),
                "reason_counts": dict(reason_counts.most_common()),
                "warning_counts": dict(warning_counts.most_common()),
                "quarantined_bad_rows": quarantined,
            },
            "archive_inventory": archives,
            "archive_prefix_counts": archive_prefix_counts,
            "queue_snapshot": queue_snapshot(state_dsn),
            "examples_by_reason": examples_by_reason,
            "samples": samples,
            "verdict": verdict,
            "blockers": blockers,
            "next_actions": [
                "Do not run the embed worker against raw NULL backlog until blocked rows are excluded.",
                "Reparse C_ARCHIVE email chunks with real MIME/quoted-printable/html cleanup.",
                "Use controlled archive inventory/unpack manifests for remaining KRAMPUSCHEWING archives before embedding.",
            ],
        }
        return report
    finally:
        conn.close()


def write_reports(report: dict[str, Any]) -> tuple[Path, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    s = stamp()
    json_path = OUT_DIR / f"ingestion_quality_audit_{s}.json"
    md_path = OUT_DIR / f"ingestion_quality_audit_{s}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=False, ensure_ascii=False, default=str), encoding="utf-8")

    scan = report["scan"]
    archives = report["archive_inventory"]
    lines = [
        "# LUCIDOTA Ingestion Quality Audit",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Verdict: **{report['verdict']}**",
        f"- Sample policy: one random sample per `{report['sample_policy']['stride']}` chunks",
        f"- Sample count: `{report['sample_policy']['sample_count']}`",
        f"- Total scanned: `{scan['counts'].get('total_scanned', 0)}`",
        f"- Quality pass: `{scan['counts'].get('quality_pass', 0)}`",
        f"- Quality block: `{scan['counts'].get('quality_block', 0)}`",
        f"- Quarantined this run: `{scan.get('quarantined_bad_rows', 0)}`",
        "",
        "## Top block reasons",
    ]
    for reason, count in list(scan["reason_counts"].items())[:20]:
        lines.append(f"- `{reason}`: {count}")
    lines.extend([
        "",
        "## Source modes",
    ])
    for mode, count in scan["source_mode_counts"].items():
        lines.append(f"- `{mode}`: {count}")
    lines.extend([
        "",
        "## Archives",
        f"- Archives still present: `{archives['archives_present_count']}` / `{archives['archives_present_total_gib']} GiB`",
        f"- Persistent unpacked files: `{archives['persistent_unpacked_files']}`",
        f"- Archive unpack receipts found: `{archives['archive_unpack_receipts_count']}`",
        f"- Disk free: `{archives['disk']['free_gib']} GiB`",
        "",
        "## Queue snapshot",
    ])
    for row in report.get("queue_snapshot", {}).get("status_counts", []):
        lines.append(f"- `{row['status']}`: {row['count']}")
    lines.extend([
        "",
        "## Representative bad examples",
    ])
    for reason, examples in list(report.get("examples_by_reason", {}).items())[:8]:
        lines.append(f"### {reason}")
        for ex in examples[:2]:
            lines.append(f"- `{ex['source_path']}` chunk `{ex['chunk_index']}`: {ex['preview'][:250]}")
    lines.extend([
        "",
        "## Files",
        f"- JSON receipt: `{rel(json_path)}`",
        f"- Markdown report: `{rel(md_path)}`",
    ])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Audit corpus chunks before BGE embedding")
    p.add_argument("--storage-dsn", default=os.environ.get("LUCIDOTA_GO_STORAGE_DSN", DEFAULT_STORAGE_DSN))
    p.add_argument("--state-dsn", default=os.environ.get("LUCIDOTA_GO_STATE_DSN") or os.environ.get("DATABASE_URL", DEFAULT_STATE_DSN))
    p.add_argument("--sample-stride", type=int, default=300)
    p.add_argument("--seed", type=int, default=260601)
    p.add_argument("--only-eligible", action="store_true", help="Scan only rows not already audit-blocked")
    p.add_argument("--quarantine-bad", action="store_true", help="Mark blocked rows in go25 so workers/enqueuers exclude them")
    return p


def main() -> int:
    args = build_parser().parse_args()
    report = run_audit(
        storage_dsn=args.storage_dsn,
        state_dsn=args.state_dsn,
        sample_stride=args.sample_stride,
        seed=args.seed,
        only_eligible=args.only_eligible,
        quarantine_bad=args.quarantine_bad,
    )
    json_path, md_path = write_reports(report)
    print(f"INGESTION_AUDIT_JSON={rel(json_path)}")
    print(f"INGESTION_AUDIT_MD={rel(md_path)}")
    print(f"VERDICT={report['verdict']}")
    print(f"SAMPLED={report['sample_policy']['sample_count']}")
    print(f"SCANNED={report['scan']['counts'].get('total_scanned', 0)}")
    print(f"QUALITY_PASS={report['scan']['counts'].get('quality_pass', 0)}")
    print(f"QUALITY_BLOCK={report['scan']['counts'].get('quality_block', 0)}")
    print(f"QUARANTINED={report['scan'].get('quarantined_bad_rows', 0)}")
    return 0 if report["verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
