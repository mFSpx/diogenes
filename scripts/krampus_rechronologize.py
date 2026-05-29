#!/usr/bin/env python3
"""Forensic re-chronologizer and River replay for KORPUS KRAMPII.

Purpose:
  - Do NOT require the whole corpus to arrive before ingestion.
  - Preserve raw/CAS first, then improve original-file dates as more metadata lands.
  - Rebuild DBSTREAM/brain-map state later in corrected chronological order.

No LLM calls. Deterministic date evidence + replay only.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parents[1]
STORAGE_DSN = os.environ.get("LUCIDOTA_GO_STORAGE_DSN", "postgresql:///lucidota_storage")
DEFAULT_REPLAY_STATE = ROOT / "03_VAULT" / "krampus_dbstream_brain.replayed.pkl"
DEFAULT_REPLAY_MAP = ROOT / "05_OUTPUTS" / "korpus_krampii" / "krampus_brain_map.replayed.jsonl"
LIVE_STATE = ROOT / "03_VAULT" / "krampus_dbstream_brain.pkl"
LIVE_MAP = ROOT / "05_OUTPUTS" / "korpus_krampii" / "krampus_brain_map.jsonl"
MODEL_KEY = "korpus_component_riverml_stats_v1"
TEXTISH_SUFFIXES = {
    ".md", ".mdx", ".txt", ".log", ".json", ".jsonl", ".ndjson", ".yaml", ".yml",
    ".csv", ".tsv", ".html", ".htm", ".xml", ".py", ".ps1", ".sh", ".js", ".ts",
    ".css", ".sql", ".toml", ".ini", ".rst",
}

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ALGOS.krampus_chrono import archive_chrono_candidates, chrono_candidates_for_path, read_text  # type: ignore  # noqa: E402
from scripts.korpus_krampii import ensure_storage  # type: ignore  # noqa: E402
from scripts.lucidota_brain_ingest import (  # type: ignore  # noqa: E402
    KrampusDBStream,
    jdump,
    load_brain,
    save_brain,
)


def utcnow_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def iso(ts: dt.datetime | None) -> str | None:
    if not ts:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.timezone.utc)
    return ts.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def parse_jsonish(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def resolved_path(path_text: str) -> Path:
    p = Path(path_text)
    if p.is_absolute():
        return p
    return ROOT / p


def is_wrapper_path(path_text: str) -> bool:
    s = path_text.replace("\\", "/")
    wrappers = (
        "/KRAMPUSCHEWING/",
        "/03_VAULT/korpus_krampii/DIGESTED/",
        "/03_VAULT/cas/",
        "/03_VAULT/korpus_krampii/DROP/",
    )
    return any(w in s for w in wrappers)


def strip_digest_move_prefix(path_text: str) -> str:
    """Remove watcher-added DIGESTED prefixes without touching real paths.

    move_to_digested writes: YYYYMMDDTHHMMSS.<original-name>
    That timestamp is ingestion bookkeeping, not original chronology.
    """
    s = str(path_text or "")
    if not s:
        return s
    slash = max(s.rfind("/"), s.rfind("\\"))
    head, name = (s[: slash + 1], s[slash + 1 :]) if slash >= 0 else ("", s)
    name = re.sub(r"^(?:20|19)\d{6}T\d{6}Z?\.(.+)$", r"\1", name)
    return head + name


def raw_text_sample(row: dict[str, Any], max_sample_chars: int) -> str:
    suffix = str(row.get("suffix") or "").lower()
    mime = str(row.get("mime") or "").lower()
    kind = str(row.get("file_kind") or "").lower()
    if not (suffix in TEXTISH_SUFFIXES or mime.startswith("text/") or kind in {"text", "code", "document"}):
        return ""
    locked = str(row.get("locked_relative_path") or "")
    if not locked:
        return ""
    p = resolved_path(locked)
    try:
        if p.exists() and p.is_file():
            return read_text(p, max_sample_chars)
    except OSError:
        return ""
    return ""


def add_candidates_for_path(candidates: list[dict[str, Any]], path_text: str, text_sample: str, wrapper: bool) -> None:
    if not path_text:
        return
    try:
        found = chrono_candidates_for_path(resolved_path(path_text), text_sample)
    except (OSError, TypeError, ValueError):
        found = []
    for cand in found:
        source = str(cand.get("source", ""))
        # CAS/drop/digested wrapper paths and copied mtimes are provenance, not
        # necessarily original chronology. Keep content-derived dates from the
        # same sample, but don't let a drop timestamp poison the time machine.
        if wrapper and (source.startswith("path_") or source.startswith("os_")):
            continue
        c2 = dict(cand)
        c2["path_evidence"] = path_text
        candidates.append(c2)


def best_chrono(row: dict[str, Any], max_sample_chars: int, max_archive_members: int) -> dict[str, Any]:
    component_sample = str(row.get("component_sample") or "")[:max_sample_chars]
    raw_sample = raw_text_sample(row, max_sample_chars)
    text_sample = (raw_sample + "\n" + component_sample).strip()[:max_sample_chars] if raw_sample else component_sample
    candidates: list[dict[str, Any]] = []
    seen_paths: set[str] = set()

    def add_path(path_text: Any, *, wrapper: bool | None = None) -> None:
        if not path_text:
            return
        s = strip_digest_move_prefix(str(path_text))
        if not s or s in seen_paths:
            return
        seen_paths.add(s)
        add_candidates_for_path(candidates, s, text_sample, is_wrapper_path(s) if wrapper is None else wrapper)

    # Original relative paths are usually the cleanest forensic path evidence.
    occurrences = parse_jsonish(row.get("occurrences"), [])
    if isinstance(occurrences, list):
        for occ in occurrences:
            if not isinstance(occ, dict):
                continue
            add_path(occ.get("relative_path"), wrapper=False)
            # Absolute paths under KRAMPUSCHEWING/DIGESTED are wrapper evidence;
            # absolute paths elsewhere can still contain original source folders.
            add_path(occ.get("absolute_path"))

    add_path(row.get("first_seen_path"))
    locked = row.get("locked_relative_path") or ""
    if locked:
        add_path(locked, wrapper=True)
        locked_path = resolved_path(str(locked))
        if str(row.get("suffix") or "").lower() == ".zip":
            for cand in archive_chrono_candidates(locked_path, max_members=max_archive_members):
                candidates.append(cand)

    # Deduplicate repeated content candidates produced across path probes.
    unique: dict[tuple[str, str, str], dict[str, Any]] = {}
    for cand in candidates:
        ts = cand.get("timestamp")
        if isinstance(ts, dt.datetime):
            tsi = iso(ts) or ""
        else:
            tsi = str(ts)
        key = (tsi, str(cand.get("source", "")), str(cand.get("raw", "")))
        current = unique.get(key)
        if current is None or int(cand.get("priority", 999)) < int(current.get("priority", 999)):
            unique[key] = cand
    candidates = list(unique.values())

    if candidates:
        best = sorted(candidates, key=lambda c: (int(c.get("priority", 999)), c.get("timestamp"), str(c.get("path_evidence", ""))))[0]
        ts = best["timestamp"].astimezone(dt.timezone.utc)
        return {
            "iso": iso(ts),
            "source": str(best.get("source", "")),
            "raw": str(best.get("raw", "")),
            "confidence_bps": int(best.get("confidence_bps", 0)),
            "candidate_count": len(candidates),
            "path_evidence": str(best.get("path_evidence", "")),
        }

    first_seen = row.get("first_seen_at")
    if isinstance(first_seen, dt.datetime):
        return {
            "iso": iso(first_seen),
            "source": "first_seen_at_fallback",
            "raw": iso(first_seen) or "",
            "confidence_bps": 1000,
            "candidate_count": 0,
            "path_evidence": str(row.get("first_seen_path") or ""),
        }
    return {
        "iso": None,
        "source": "missing",
        "raw": "",
        "confidence_bps": 0,
        "candidate_count": 0,
        "path_evidence": "",
    }


def fetch_file_rows(conn: psycopg.Connection, limit: int | None, recent_hours: float | None, max_sample_chars: int) -> list[dict[str, Any]]:
    params: list[Any] = [max_sample_chars, recent_hours, recent_hours]
    sql = """
    SELECT
        fo.file_uuid::text,
        fo.first_seen_path,
        fo.locked_relative_path,
        fo.file_kind,
        fo.mime,
        fo.suffix,
        fo.size_bytes,
        fo.first_seen_at,
        fo.detail,
        COALESCE((
            SELECT jsonb_agg(DISTINCT jsonb_build_object(
                'absolute_path', o.absolute_path,
                'relative_path', o.relative_path,
                'root_path', o.root_path,
                'mtime', o.mtime
            ))
            FROM lucidota_korpus.file_occurrence o
            WHERE o.file_uuid = fo.file_uuid
        ), '[]'::jsonb) AS occurrences,
        COALESCE((
            SELECT left(string_agg(left(concat_ws(E'\n', c.title, c.content), 8000), E'\n' ORDER BY c.component_index), %s)
            FROM lucidota_korpus.component c
            WHERE c.file_uuid = fo.file_uuid
        ), '') AS component_sample
    FROM lucidota_korpus.file_object fo
    WHERE (%s::double precision IS NULL OR fo.first_seen_at >= now() - make_interval(secs => (%s::double precision * 3600)::integer))
    ORDER BY fo.first_seen_at ASC, fo.file_uuid ASC
    """
    if limit and limit > 0:
        sql += " LIMIT %s"
        params.append(limit)
    return list(conn.execute(sql, params).fetchall())


def refresh_chrono(args: argparse.Namespace) -> dict[str, Any]:
    updated = 0
    unchanged = 0
    source_counts: dict[str, int] = {}
    examples: list[dict[str, Any]] = []
    with psycopg.connect(STORAGE_DSN, row_factory=dict_row) as conn:
        ensure_storage(conn)
        rows = fetch_file_rows(conn, args.limit, args.recent_hours, args.max_sample_chars)
        for row in rows:
            chrono = best_chrono(row, args.max_sample_chars, args.max_archive_members)
            source = chrono["source"]
            source_counts[source] = source_counts.get(source, 0) + 1
            patch = {
                "chrono_original_date": chrono["iso"],
                "chrono_date_source": source,
                "chrono_confidence_bps": chrono["confidence_bps"],
                "chrono_raw": chrono["raw"],
                "chrono_candidate_count": chrono["candidate_count"],
                "chrono_path_evidence": chrono["path_evidence"],
                "chrono_refreshed_at": utcnow_iso(),
                "chrono_refreshed_by": "krampus_rechronologize.refresh.v1",
            }
            before = parse_jsonish(row.get("detail"), {}) if row.get("detail") is not None else {}
            changed = any(before.get(k) != v for k, v in patch.items() if k != "chrono_refreshed_at")
            conn.execute(
                """
                UPDATE lucidota_korpus.file_object
                SET detail = detail || %s::jsonb,
                    updated_at = now()
                WHERE file_uuid = %s::uuid
                """,
                (json.dumps(patch, ensure_ascii=False), row["file_uuid"]),
            )
            if chrono["iso"]:
                conn.execute(
                    """
                    UPDATE lucidota_korpus.vibe_telemetry
                    SET original_file_date = %s::timestamptz,
                        original_file_date_source = %s,
                        original_file_date_confidence_bps = %s,
                        detail = detail || %s::jsonb
                    WHERE file_uuid = %s::uuid
                    """,
                    (
                        chrono["iso"],
                        source,
                        chrono["confidence_bps"],
                        json.dumps({"chrono_refreshed_at": patch["chrono_refreshed_at"], "chrono_path_evidence": chrono["path_evidence"]}, ensure_ascii=False),
                        row["file_uuid"],
                    ),
                )
            if changed:
                updated += 1
                if len(examples) < 10:
                    examples.append({"file_uuid": row["file_uuid"], "date": chrono["iso"], "source": source, "path": row.get("first_seen_path")})
            else:
                unchanged += 1
        conn.commit()
    return {
        "ok": True,
        "action": "refresh",
        "files_seen": updated + unchanged,
        "updated": updated,
        "unchanged": unchanged,
        "source_counts": dict(sorted(source_counts.items())),
        "examples": examples,
    }


def replay_rows(conn: psycopg.Connection, limit: int | None) -> list[dict[str, Any]]:
    params: list[Any] = []
    sql = """
    SELECT
        c.component_uuid::text,
        c.file_uuid::text,
        c.component_index,
        c.component_kind,
        c.title,
        c.content,
        fo.first_seen_path,
        COALESCE(
            vt.original_file_date,
            NULLIF(fo.detail->>'chrono_original_date','')::timestamptz,
            fo.first_seen_at
        ) AS original_file_date,
        COALESCE(
            NULLIF(vt.original_file_date_source,''),
            NULLIF(fo.detail->>'chrono_date_source',''),
            'first_seen_at_fallback'
        ) AS original_file_date_source,
        COALESCE(
            vt.original_file_date_confidence_bps,
            CASE WHEN (fo.detail->>'chrono_confidence_bps') ~ '^\\d+$' THEN (fo.detail->>'chrono_confidence_bps')::integer ELSE 0 END
        ) AS original_file_date_confidence_bps
    FROM lucidota_korpus.component c
    JOIN lucidota_korpus.file_object fo ON fo.file_uuid = c.file_uuid
    LEFT JOIN lucidota_korpus.vibe_telemetry vt
        ON vt.component_uuid = c.component_uuid
       AND vt.model_key = %s
    WHERE COALESCE(c.content, '') <> ''
    ORDER BY original_file_date ASC, fo.first_seen_path ASC, c.component_index ASC, c.component_uuid ASC
    """
    params.append(MODEL_KEY)
    if limit and limit > 0:
        sql += " LIMIT %s"
        params.append(limit)
    return list(conn.execute(sql, params).fetchall())


def replay_brain(args: argparse.Namespace) -> dict[str, Any]:
    if args.replace_live:
        state_path = LIVE_STATE
        map_path = LIVE_MAP
    else:
        state_path = Path(args.state_path).expanduser() if args.state_path else DEFAULT_REPLAY_STATE
        map_path = Path(args.map_jsonl).expanduser() if args.map_jsonl else DEFAULT_REPLAY_MAP
    if not state_path.is_absolute():
        state_path = ROOT / state_path
    if not map_path.is_absolute():
        map_path = ROOT / map_path

    if not args.append:
        if state_path.exists():
            state_path.unlink()
        if map_path.exists():
            map_path.unlink()
    brain: KrampusDBStream = load_brain(state_path)
    map_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    skipped = 0
    first_date: str | None = None
    last_date: str | None = None
    with psycopg.connect(STORAGE_DSN, row_factory=dict_row) as conn, map_path.open("a", encoding="utf-8") as out:
        ensure_storage(conn)
        rows = replay_rows(conn, args.limit)
        for row in rows:
            text = f"{row.get('title') or ''}\n{row.get('content') or ''}".strip()
            if not text:
                skipped += 1
                continue
            file_time = iso(row.get("original_file_date"))
            if first_date is None:
                first_date = file_time
            last_date = file_time
            metadata = {
                "path": row.get("first_seen_path") or "",
                "file_uuid": row["file_uuid"],
                "component_uuid": row["component_uuid"],
                "component_index": row["component_index"],
                "component_kind": row["component_kind"],
                "file_time": file_time,
                "file_time_source": row.get("original_file_date_source") or "",
                "file_time_confidence_bps": int(row.get("original_file_date_confidence_bps") or 0),
                "replay_run_at": utcnow_iso(),
                "replay_source": "krampus_rechronologize.replay.v1",
            }
            result = brain.ingest(text, metadata)
            out.write(jdump(result) + "\n")
            count += 1
            if count % max(1, int(args.save_every)) == 0:
                save_brain(state_path, brain)
        save_brain(state_path, brain)
    return {
        "ok": True,
        "action": "replay",
        "components_replayed": count,
        "skipped": skipped,
        "state_path": str(state_path),
        "map_jsonl": str(map_path),
        "replace_live": bool(args.replace_live),
        "append": bool(args.append),
        "first_date": first_date,
        "last_date": last_date,
        "clusters": brain.centers_count(),
        "files_ingested_in_brain_state": brain.files_ingested,
    }


def add_common_refresh(p: argparse.ArgumentParser) -> None:
    p.add_argument("--limit", type=int, default=0, help="Max file rows to refresh; 0 = all.")
    p.add_argument("--recent-hours", type=float, default=None, help="Only refresh files first seen in the last N hours.")
    p.add_argument("--max-sample-chars", type=int, default=256_000)
    p.add_argument("--max-archive-members", type=int, default=20_000)


def add_common_replay(p: argparse.ArgumentParser) -> None:
    p.add_argument("--limit", type=int, default=0, help="Max components to replay; 0 = all.")
    p.add_argument("--state-path", default=str(DEFAULT_REPLAY_STATE))
    p.add_argument("--map-jsonl", default=str(DEFAULT_REPLAY_MAP))
    p.add_argument("--replace-live", action="store_true", help="Overwrite the live DBSTREAM state/map with corrected chronological replay.")
    p.add_argument("--append", action="store_true", help="Append to existing replay state/map instead of rebuilding from zero.")
    p.add_argument("--save-every", type=int, default=100)


def main() -> int:
    ap = argparse.ArgumentParser(prog="krampus-rechronologize")
    ap.add_argument("--json", action="store_true")
    sub = ap.add_subparsers(dest="command", required=True)

    p_refresh = sub.add_parser("refresh", help="Recompute original_file_date evidence in DB.")
    add_common_refresh(p_refresh)

    p_replay = sub.add_parser("replay", help="Replay River/brain-map in corrected chronological order.")
    add_common_replay(p_replay)

    p_run = sub.add_parser("run", help="Refresh dates, then replay River/brain-map.")
    p_run.add_argument("--refresh-limit", type=int, default=0)
    p_run.add_argument("--recent-hours", type=float, default=None)
    p_run.add_argument("--max-sample-chars", type=int, default=256_000)
    p_run.add_argument("--max-archive-members", type=int, default=20_000)
    p_run.add_argument("--replay-limit", type=int, default=0)
    p_run.add_argument("--state-path", default=str(DEFAULT_REPLAY_STATE))
    p_run.add_argument("--map-jsonl", default=str(DEFAULT_REPLAY_MAP))
    p_run.add_argument("--replace-live", action="store_true")
    p_run.add_argument("--append", action="store_true")
    p_run.add_argument("--save-every", type=int, default=100)

    args = ap.parse_args()
    if args.command == "refresh":
        args.limit = args.limit or None
        result = refresh_chrono(args)
    elif args.command == "replay":
        args.limit = args.limit or None
        result = replay_brain(args)
    else:
        refresh_args = argparse.Namespace(
            limit=args.refresh_limit or None,
            recent_hours=args.recent_hours,
            max_sample_chars=args.max_sample_chars,
            max_archive_members=args.max_archive_members,
        )
        replay_args = argparse.Namespace(
            limit=args.replay_limit or None,
            state_path=args.state_path,
            map_jsonl=args.map_jsonl,
            replace_live=args.replace_live,
            append=args.append,
            save_every=args.save_every,
        )
        result = {"ok": True, "action": "run", "refresh": refresh_chrono(refresh_args), "replay": replay_brain(replay_args)}
    print(jdump(result) if args.json else json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
