#!/usr/bin/env python3
"""Ingest hunch audit records into Postgres without promoting graph truth."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "06_SCHEMA" / "116_hunch_postgres_ingest.sql"
OUT = ROOT / "05_OUTPUTS" / "hunch_hypertimeline"
OBS = ROOT / "04_RUNTIME" / "observation_center" / "hunch_hypertimeline_latest.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path | str, *, root: Path = ROOT) -> str:
    try:
        return str(Path(path).resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def db_url(args: argparse.Namespace | None = None) -> str:
    return (
        (getattr(args, "database_url", None) if args is not None else None)
        or os.environ.get("LUCIDOTA_HUNCH_DATABASE_URL")
        or os.environ.get("LUCIDOTA_CONTROL_DATABASE_URL")
        or os.environ.get("DATABASE_URL")
        or "postgresql:///lucidota_state"
    )


def resolve_path(path: Path | str, *, root: Path = ROOT) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = root / p
    return p.resolve()


def read_json(path: Path | str, *, root: Path = ROOT) -> dict[str, Any]:
    p = resolve_path(path, root=root)
    return json.loads(p.read_text(encoding="utf-8"))


def build_manual_hunch_record(*, text: str, source: str, root: Path = ROOT) -> dict[str, Any]:
    text = text.strip()
    digest = sha256_text(f"{source}:{text}")
    title = text[:180]
    return {
        "kind": "HUNCH",
        "hunch_id": "OP-" + digest[:12].upper(),
        "title": title,
        "rating": "OPEN",
        "line_start": None,
        "line_end": None,
        "source_path": source,
        "source_sha256": digest,
        "source_bytes": len(text.encode("utf-8")),
        "source_mtime_utc": now(),
        "body_sha256": sha256_text(text),
        "evidence_state": "operator_fresh_hunch",
        "truth_promotion": "blocked_until_evidence_paths_reviewed",
        "operator_hunch_text": text,
        "canonical_graph_writes_performed": False,
    }


def load_hunch_records(path: Path | str = OBS, *, manual_hunch: str = "", root: Path = ROOT) -> list[dict[str, Any]]:
    data = read_json(path, root=root)
    records: list[dict[str, Any]] = []
    candidate_path = data.get("candidates_path")
    report_path = data.get("report_path")
    if candidate_path:
        p = resolve_path(candidate_path, root=root)
        if p.exists():
            with p.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            records.append(obj)
    elif report_path:
        report = read_json(report_path, root=root)
        records = [r for r in report.get("hunch_records", []) if isinstance(r, dict)]
    if manual_hunch.strip():
        records.append(build_manual_hunch_record(text=manual_hunch, source="operator_chat", root=root))
    return records


def build_upsert_rows(records: list[dict[str, Any]], *, root: Path = ROOT) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        hunch_id = str(record.get("hunch_id") or "").strip()
        if not hunch_id:
            continue
        title = str(record.get("title") or hunch_id).strip()[:1000]
        detail = dict(record)
        rows.append(
            {
                "hunch_id": hunch_id,
                "title": title,
                "rating": str(record.get("rating") or "OPEN"),
                "authority_class": "operator_hunch_signal_not_truth",
                "evidence_state": str(record.get("evidence_state") or "candidate"),
                "truth_promotion": str(record.get("truth_promotion") or "blocked_until_evidence_paths_reviewed"),
                "source_path": str(record.get("source_path") or ""),
                "source_sha256": str(record.get("source_sha256") or ""),
                "source_line_start": record.get("line_start"),
                "source_line_end": record.get("line_end"),
                "body_sha256": str(record.get("body_sha256") or sha256_text(json.dumps(record, sort_keys=True, default=str))),
                "detail": detail,
                "canonical_graph_writes_performed": False,
            }
        )
    return rows


def build_graph_stage_packets(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    packets: list[dict[str, Any]] = []
    for row in rows:
        packets.append(
            {
                "kind": "OBJECT",
                "term": "HUNCH",
                "id": "hunch:" + row["hunch_id"],
                "label": row["title"],
                "rating": row["rating"],
                "authority_class": row["authority_class"],
                "source_sha256": row["source_sha256"],
                "body_sha256": row["body_sha256"],
                "promotion_state": "staged_candidate_only",
                "graph_writes_performed": False,
            }
        )
    return packets


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def write_ingest_observation(receipt: dict[str, Any], rows: list[dict[str, Any]], *, root: Path = ROOT) -> dict[str, str]:
    latest_op = next((r for r in reversed(rows) if str(r.get("hunch_id", "")).startswith("OP-")), None)
    summary = {
        "schema": "lucidota.observation_center.hunch_postgres_ingest.v1",
        "generated_at": receipt.get("generated_at") or now(),
        "records_seen": receipt.get("records_seen", len(rows)),
        "records_upserted": receipt.get("records_upserted", 0),
        "graph_candidates_written": receipt.get("graph_candidates_written", 0),
        "manual_hunch_present": bool(receipt.get("manual_hunch_present")),
        "latest_operator_hunch": latest_op,
        "receipt_path": receipt.get("receipt_path"),
        "graph_stage_path": receipt.get("graph_stage_path"),
        "canonical_graph_writes_performed": False,
    }
    obs_path = root / "04_RUNTIME" / "observation_center" / "hunch_postgres_ingest_latest.json"
    write_json(obs_path, summary)

    big_path = root / "05_OUTPUTS" / "big_board.json"
    if big_path.exists():
        try:
            big = json.loads(big_path.read_text(encoding="utf-8"))
        except Exception:
            big = {}
    else:
        big = {}
    big.setdefault("observation_center", {})["hunch_postgres_ingest"] = summary
    counters = big.setdefault("counters", {})
    counters["hunch_postgres_records_seen"] = summary["records_seen"]
    counters["hunch_postgres_records_upserted"] = summary["records_upserted"]
    counters["hunch_postgres_manual_hunch_present"] = summary["manual_hunch_present"]
    if latest_op:
        counters["hunch_postgres_latest_op_hunch_id"] = latest_op.get("hunch_id")
    write_json(big_path, big)
    return {"observation_center_path": rel(obs_path, root=root), "big_board_path": rel(big_path, root=root)}


def upsert_rows(conn: psycopg.Connection[Any], rows: list[dict[str, Any]]) -> int:
    with conn.cursor() as cur:
        cur.execute(SCHEMA.read_text(encoding="utf-8"))
        count = 0
        for row in rows:
            cur.execute(
                """
                INSERT INTO lucidota_hunch.hunch_record(
                  hunch_id, title, rating, authority_class, evidence_state, truth_promotion,
                  source_path, source_sha256, source_line_start, source_line_end, body_sha256,
                  detail, canonical_graph_writes_performed
                )
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,false)
                ON CONFLICT(hunch_id) DO UPDATE SET
                  title=EXCLUDED.title,
                  rating=EXCLUDED.rating,
                  authority_class=EXCLUDED.authority_class,
                  evidence_state=EXCLUDED.evidence_state,
                  truth_promotion=EXCLUDED.truth_promotion,
                  source_path=EXCLUDED.source_path,
                  source_sha256=EXCLUDED.source_sha256,
                  source_line_start=EXCLUDED.source_line_start,
                  source_line_end=EXCLUDED.source_line_end,
                  body_sha256=EXCLUDED.body_sha256,
                  detail=EXCLUDED.detail,
                  canonical_graph_writes_performed=false,
                  updated_at=now()
                """,
                (
                    row["hunch_id"],
                    row["title"],
                    row["rating"],
                    row["authority_class"],
                    row["evidence_state"],
                    row["truth_promotion"],
                    row["source_path"],
                    row["source_sha256"],
                    row["source_line_start"],
                    row["source_line_end"],
                    row["body_sha256"],
                    json.dumps(row["detail"], default=str),
                ),
            )
            count += 1
    return count


def insert_run(
    conn: psycopg.Connection[Any],
    *,
    source_report_path: str,
    records_seen: int,
    records_upserted: int,
    graph_candidates_written: int,
    receipt_path: str,
    detail: dict[str, Any],
) -> str:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO lucidota_hunch.hunch_ingest_run(
              source_report_path, records_seen, records_upserted, graph_candidates_written,
              receipt_path, detail, canonical_graph_writes_performed
            )
            VALUES (%s,%s,%s,%s,%s,%s::jsonb,false)
            RETURNING run_uuid::text
            """,
            (source_report_path, records_seen, records_upserted, graph_candidates_written, receipt_path, json.dumps(detail, default=str)),
        )
        return str(cur.fetchone()[0])


def main() -> int:
    ap = argparse.ArgumentParser(description="Load hunch audit records into Postgres hunch tables and graph-stage JSONL.")
    ap.add_argument("--input", default=rel(OBS), help="observation summary or full hunch audit report")
    ap.add_argument("--manual-hunch", default="")
    ap.add_argument("--database-url")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    records = load_hunch_records(args.input, manual_hunch=args.manual_hunch)
    rows = build_upsert_rows(records)
    packets = build_graph_stage_packets(rows)
    run_id = None
    upserted = 0
    graph_path = OUT / f"hunch_graph_stage_{stamp()}.jsonl"
    receipt_path = OUT / f"hunch_postgres_ingest_{stamp()}.json"

    if args.execute:
        with psycopg.connect(db_url(args)) as conn:
            upserted = upsert_rows(conn, rows)
            write_jsonl(graph_path, packets)
            detail = {"input": args.input, "manual_hunch_present": bool(args.manual_hunch.strip()), "graph_stage_path": rel(graph_path)}
            run_id = insert_run(
                conn,
                source_report_path=args.input,
                records_seen=len(rows),
                records_upserted=upserted,
                graph_candidates_written=len(packets),
                receipt_path=rel(receipt_path),
                detail=detail,
            )
            conn.commit()
    else:
        write_jsonl(graph_path, packets)

    receipt = {
        "schema": "lucidota.hunch_postgres_ingest.receipt.v1",
        "generated_at": now(),
        "execute_performed": bool(args.execute),
        "input": args.input,
        "manual_hunch_present": bool(args.manual_hunch.strip()),
        "records_seen": len(rows),
        "records_upserted": upserted,
        "graph_candidates_written": len(packets),
        "graph_stage_path": rel(graph_path),
        "run_uuid": run_id,
        "db_url_redacted": "postgresql://<redacted>",
        "canonical_graph_writes_performed": False,
        "receipt_path": rel(receipt_path),
    }
    receipt["observation"] = write_ingest_observation(receipt, rows)
    write_json(receipt_path, receipt)
    print("REPORT_PATH=" + rel(receipt_path))
    print("GRAPH_STAGE_PATH=" + rel(graph_path))
    print("HUNCH_POSTGRES_INGEST=" + ("PASS" if args.execute else "DRY_RUN_PASS"))
    if args.json:
        print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
