#!/usr/bin/env python3
"""Ledger-only duplicate reclamation planner for LUCIDOTA.

This script does not hash source files and does not read source file contents.
It trusts the already-built Chrono ledger's source_sha256 fields as the hash oracle,
then resolves one master path per hash and emits a dry-run purge manifest.

Deletion is gated behind an exact confirmation phrase and, even then, verifies every
candidate remains inside the requested scope. Dry-run is the intended default.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    import psycopg
    from psycopg.rows import dict_row
except Exception:  # pragma: no cover - dry-run can still work without size lookup
    psycopg = None  # type: ignore[assignment]
    dict_row = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCOPE = ROOT
DEFAULT_LEDGER_DIR = ROOT / "05_OUTPUTS" / "CHRONO_PARTS"
OUT_DIR = ROOT / "05_OUTPUTS" / "dedupe"
FORBIDDEN_SCOPE_PREFIXES = tuple(Path(p).resolve() for p in (
    "/bin", "/usr", "/etc", "/boot", "/lib", "/lib64", "/sbin", "/var",
    "/opt", "/proc", "/sys", "/dev", "/run",
))
EXECUTE_PHRASE = "EXECUTE PURGE"


@dataclass(frozen=True)
class LedgerOccurrence:
    source_sha256: str
    source_path: str
    absolute_path: str
    file_uuid: str | None
    claim_uuid: str | None
    evidence_source: str | None
    candidate_timestamp: str | None
    first_part: str
    first_line_no: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def fmt_bytes(n: int | float | None) -> str:
    if n is None:
        return "UNKNOWN"
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TB"


def validate_scope(scope: Path) -> Path:
    resolved = scope.expanduser().resolve()
    if not resolved.exists() or not resolved.is_dir():
        raise SystemExit(f"invalid scope: {resolved}")
    for forbidden in FORBIDDEN_SCOPE_PREFIXES:
        try:
            resolved.relative_to(forbidden)
            raise SystemExit(f"forbidden cleanup scope: {resolved} is under {forbidden}")
        except ValueError:
            pass
    return resolved


def inside_scope(path: Path, scope: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(scope)
        return True
    except ValueError:
        return False


def absolute_from_source_path(source_path: str, scope: Path) -> str:
    raw = Path(source_path)
    if raw.is_absolute():
        candidate = raw
    else:
        candidate = scope / raw
    return str(candidate.resolve(strict=False))


def path_depth(abs_path: str, scope: Path) -> int:
    try:
        return len(Path(abs_path).resolve(strict=False).relative_to(scope).parts)
    except Exception:
        return len(Path(abs_path).parts)


def master_sort_key(abs_path: str, scope: Path) -> tuple[int, int, str]:
    # User law: shortest absolute path OR highest in tree. Apply shortest first,
    # then depth, then lexical for deterministic ties.
    return (len(abs_path), path_depth(abs_path, scope), abs_path)


def iter_json_objects_from_parts(ledger_dir: Path) -> Iterable[tuple[Path, int, dict[str, Any]]]:
    parts = sorted(ledger_dir.glob("CHRONO_PART_*.txt"))
    if not parts:
        raise SystemExit(f"no CHRONO_PART_*.txt files found in {ledger_dir}")
    for part in parts:
        with part.open("r", encoding="utf-8", errors="replace") as fh:
            for line_no, line in enumerate(fh, 1):
                stripped = line.strip()
                if not stripped or not stripped.startswith("{"):
                    continue
                try:
                    obj = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    yield part, line_no, obj


def extract_occurrences(ledger_dir: Path, scope: Path) -> tuple[list[LedgerOccurrence], dict[str, Any]]:
    seen_path_by_hash: dict[tuple[str, str], LedgerOccurrence] = {}
    claim_rows = 0
    skipped = defaultdict(int)
    for part, line_no, obj in iter_json_objects_from_parts(ledger_dir):
        if "source_sha256" not in obj or "source_path" not in obj:
            continue
        claim_rows += 1
        source_sha256 = str(obj.get("source_sha256") or "").strip().lower()
        source_path = str(obj.get("source_path") or "").strip()
        if len(source_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in source_sha256):
            skipped["invalid_source_sha256"] += 1
            continue
        if not source_path:
            skipped["missing_source_path"] += 1
            continue
        abs_path = absolute_from_source_path(source_path, scope)
        if not inside_scope(Path(abs_path), scope):
            skipped["outside_scope"] += 1
            continue
        key = (source_sha256, abs_path)
        if key in seen_path_by_hash:
            skipped["repeat_claim_same_hash_path"] += 1
            continue
        seen_path_by_hash[key] = LedgerOccurrence(
            source_sha256=source_sha256,
            source_path=source_path,
            absolute_path=abs_path,
            file_uuid=str(obj.get("file_uuid")) if obj.get("file_uuid") else None,
            claim_uuid=str(obj.get("claim_uuid")) if obj.get("claim_uuid") else None,
            evidence_source=str(obj.get("evidence_source")) if obj.get("evidence_source") else None,
            candidate_timestamp=str(obj.get("candidate_timestamp")) if obj.get("candidate_timestamp") else None,
            first_part=rel(part),
            first_line_no=line_no,
        )
    stats = {
        "ledger_claim_rows_with_source_fields": claim_rows,
        "unique_hash_path_occurrences": len(seen_path_by_hash),
        "skipped": dict(skipped),
    }
    return list(seen_path_by_hash.values()), stats


def load_sizes_from_korpus(source_hashes: set[str], database_url: str) -> dict[str, int]:
    if not source_hashes or psycopg is None:
        return {}
    sizes: dict[str, int] = {}
    hashes = sorted(source_hashes)
    with psycopg.connect(database_url, row_factory=dict_row) as conn:  # type: ignore[arg-type]
        with conn.cursor() as cur:
            for idx in range(0, len(hashes), 1000):
                chunk = hashes[idx: idx + 1000]
                cur.execute(
                    """
                    SELECT sha256, max(size_bytes)::bigint AS size_bytes
                    FROM lucidota_korpus.file_object
                    WHERE sha256 = ANY(%s)
                    GROUP BY sha256
                    """,
                    (chunk,),
                )
                for row in cur.fetchall():
                    if row["sha256"] and row["size_bytes"] is not None:
                        sizes[str(row["sha256"]).lower()] = int(row["size_bytes"])
    return sizes


def build_plan(occurrences: list[LedgerOccurrence], scope: Path, sizes: dict[str, int]) -> dict[str, Any]:
    grouped: dict[str, list[LedgerOccurrence]] = defaultdict(list)
    for occ in occurrences:
        grouped[occ.source_sha256].append(occ)

    duplicate_groups: list[dict[str, Any]] = []
    deletes: list[dict[str, Any]] = []
    unknown_size_deletes = 0
    reclaim_known = 0

    for sha, group in grouped.items():
        # Only distinct paths are eligible. Multiple claims on one path are not duplicates.
        if len(group) < 2:
            continue
        ordered = sorted(group, key=lambda occ: master_sort_key(occ.absolute_path, scope))
        master = ordered[0]
        delete_occs = ordered[1:]
        size = sizes.get(sha)
        group_payload = {
            "source_sha256": sha,
            "size_bytes": size,
            "size_human": fmt_bytes(size),
            "master": master.__dict__,
            "duplicate_count": len(delete_occs),
            "delete_paths": [occ.__dict__ for occ in delete_occs],
        }
        duplicate_groups.append(group_payload)
        for occ in delete_occs:
            item = {
                "delete_path": occ.absolute_path,
                "keep_master_path": master.absolute_path,
                "source_sha256": sha,
                "size_bytes": size,
                "size_human": fmt_bytes(size),
                "claim_uuid": occ.claim_uuid,
                "file_uuid": occ.file_uuid,
                "evidence_source": occ.evidence_source,
                "ledger_ref": f"{occ.first_part}:{occ.first_line_no}",
            }
            deletes.append(item)
            if size is None:
                unknown_size_deletes += 1
            else:
                reclaim_known += size

    deletes_sorted = sorted(
        deletes,
        key=lambda d: (-(d["size_bytes"] or -1), len(d["delete_path"]), d["delete_path"]),
    )
    return {
        "duplicate_hash_groups": len(duplicate_groups),
        "duplicate_files_marked_for_deletion": len(deletes),
        "known_reclaim_bytes": reclaim_known,
        "known_reclaim_human": fmt_bytes(reclaim_known),
        "unknown_size_duplicate_deletions": unknown_size_deletes,
        "duplicate_groups": duplicate_groups,
        "delete_manifest": deletes_sorted,
        "sample_deletions_top5": deletes_sorted[:5],
    }


def write_report(payload: dict[str, Any]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"chrono_ledger_duplicate_reclaimer_dry_run_{stamp()}.json"
    payload["report_path"] = rel(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path


def dry_run(args: argparse.Namespace) -> int:
    scope = validate_scope(Path(args.scope))
    ledger_dir = Path(args.ledger_dir).expanduser().resolve()
    occurrences, extraction_stats = extract_occurrences(ledger_dir, scope)
    hashes = {occ.source_sha256 for occ in occurrences}
    sizes = load_sizes_from_korpus(hashes, args.database_url) if args.lookup_sizes else {}
    plan = build_plan(occurrences, scope, sizes)
    payload = {
        "schema": "lucidota.chrono_ledger_duplicate_reclaimer.dry_run.v1",
        "generated_at": utc_now(),
        "mode": "DRY_RUN_ONLY",
        "scope": str(scope),
        "ledger_dir": str(ledger_dir),
        "physical_source_file_content_reads_performed": False,
        "hash_oracle": "CHRONO_PART_*.txt source_sha256 fields",
        "size_oracle": "lucidota_korpus.file_object.size_bytes keyed by source_sha256" if args.lookup_sizes else "disabled",
        "empty_directory_cleanup_evaluated": False,
        "junk_file_cleanup_evaluated": False,
        "purge_confirmation_required": EXECUTE_PHRASE,
        "extraction_stats": extraction_stats,
        **{k: v for k, v in plan.items() if k not in {"duplicate_groups", "delete_manifest"}},
        "delete_manifest_count": len(plan["delete_manifest"]),
        "duplicate_groups": plan["duplicate_groups"],
        "delete_manifest": plan["delete_manifest"],
    }
    report = write_report(payload)
    print(f"REPORT_PATH={rel(report)}")
    print(f"TOTAL_DUPLICATE_FILES={plan['duplicate_files_marked_for_deletion']}")
    print(f"TOTAL_SPACE_RECLAIMABLE_KNOWN={plan['known_reclaim_human']}")
    print(f"UNKNOWN_SIZE_DUPLICATE_DELETIONS={plan['unknown_size_duplicate_deletions']}")
    print("SAMPLE_DELETE_PATHS_JSON=" + json.dumps(plan["sample_deletions_top5"], ensure_ascii=False, default=str))
    print(f"WAITING_FOR={EXECUTE_PHRASE}")
    return 0


def purge(args: argparse.Namespace) -> int:
    if args.confirm != EXECUTE_PHRASE:
        print("PURGE_ABORTED_CONFIRMATION_MISMATCH")
        return 2
    manifest_path = Path(args.manifest).expanduser().resolve()
    scope = validate_scope(Path(args.scope))
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    deleted: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    for item in data.get("delete_manifest", []):
        target = Path(str(item["delete_path"])).resolve(strict=False)
        master = Path(str(item["keep_master_path"])).resolve(strict=False)
        if not inside_scope(target, scope):
            blocked.append({"path": str(target), "reason": "outside_scope"})
            continue
        if target == master:
            blocked.append({"path": str(target), "reason": "target_is_master"})
            continue
        if target.is_symlink():
            blocked.append({"path": str(target), "reason": "symlink_not_deleted_by_reclaimer"})
            continue
        if not target.exists():
            blocked.append({"path": str(target), "reason": "missing_at_purge_time"})
            continue
        if not target.is_file():
            blocked.append({"path": str(target), "reason": "not_regular_file"})
            continue
        # No content re-hashing here: the user explicitly selected the ledger oracle.
        size = target.stat().st_size
        target.unlink()
        deleted.append({"path": str(target), "size_bytes": size})
    purge_report = {
        "schema": "lucidota.chrono_ledger_duplicate_reclaimer.purge.v1",
        "generated_at": utc_now(),
        "scope": str(scope),
        "manifest": rel(manifest_path),
        "deleted_count": len(deleted),
        "deleted_bytes": sum(d["size_bytes"] for d in deleted),
        "deleted_human": fmt_bytes(sum(d["size_bytes"] for d in deleted)),
        "blocked_count": len(blocked),
        "blocked": blocked[:200],
        "deleted_sample": deleted[:20],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"chrono_ledger_duplicate_reclaimer_purge_{stamp()}.json"
    purge_report["report_path"] = rel(out)
    out.write_text(json.dumps(purge_report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"REPORT_PATH={rel(out)}")
    print(f"PURGED_FILES={len(deleted)}")
    print(f"PURGED_SPACE={purge_report['deleted_human']}")
    print(f"BLOCKED={len(blocked)}")
    return 0 if not blocked else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Chrono ledger-only duplicate reclamation planner")
    parser.add_argument("--scope", default=str(DEFAULT_SCOPE))
    parser.add_argument("--ledger-dir", default=str(DEFAULT_LEDGER_DIR))
    parser.add_argument("--database-url", default=os.environ.get("KORPUS_DATABASE_URL") or os.environ.get("DATABASE_URL") or "postgresql:///lucidota_storage")
    sub = parser.add_subparsers(dest="cmd", required=True)
    dry = sub.add_parser("dry-run")
    dry.add_argument("--lookup-sizes", action=argparse.BooleanOptionalAction, default=True)
    dry.set_defaults(func=dry_run)
    pur = sub.add_parser("purge")
    pur.add_argument("--manifest", required=True)
    pur.add_argument("--confirm", required=True)
    pur.set_defaults(func=purge)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
