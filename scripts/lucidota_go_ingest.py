#!/usr/bin/env python3
"""GO ingest/promote lane for DIOGENES/LUCIDOTA.

Postgres-first. Uses psql when present, with psycopg fallback in venv runs.

Env:
  LUCIDOTA_GO_STORAGE_DSN  default: postgresql:///lucidota_storage
  LUCIDOTA_GO_STATE_DSN    default: postgresql:///lucidota_state

Commands:
  bootstrap-dbs
  init
  validate-item item.json
  validate-edge edge.json
  validate-packet packet_or_batch.json
  stage packet_or_batch.json [--dry-run]
  promote <packet_uuid> [<packet_uuid> ...] --operator <operator_uuid>
  batch-approve --operator <operator_uuid> [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import psycopg  # type: ignore
except Exception:  # pragma: no cover - exercised through monkeypatch in tests
    psycopg = None

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_SQL = ROOT / "06_SCHEMA" / "016_go_graph_core.sql"
TERMS_PATH = ROOT / "BOOKS" / "GO_ACTIVE_TERMS.json"
ONTOLOGY_PATH = ROOT / "OFFICIAL_ONTOLOGY.json"

STORAGE_DSN = os.environ.get("LUCIDOTA_GO_STORAGE_DSN", "postgresql:///lucidota_storage")
STATE_DSN = os.environ.get("LUCIDOTA_GO_STATE_DSN", "postgresql:///lucidota_state")

STATUSES = {
    "located",
    "staged",
    "approved",
    "rejected",
    "superseded",
    "archived",
    "error_corrected",
    "lost",
    "collapsed",
}
APPROVAL_SCOPES = {"contemporaneous", "current", "historical", "unknown_current", "superseded"}
CURRENT_STATUSES = {"yes", "no", "unknown"}
BPS = {0, 2, 4, 6, 10, 50, 69, 150}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sql_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, (dict, list)):
        s = json.dumps(value, ensure_ascii=False, sort_keys=True)
    else:
        s = str(value)
    return "'" + s.replace("'", "''") + "'"


def psql(args: list[str], *, dsn: str = STORAGE_DSN, input_sql: str | None = None) -> str:
    cmd = ["psql", "-X", "-v", "ON_ERROR_STOP=1", dsn, *args]
    try:
        proc = subprocess.run(
            cmd,
            input=input_sql,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError:
        raise SystemExit("psql not found. Install PostgreSQL client tools.")
    if proc.returncode != 0:
        raise SystemExit(proc.stderr.strip() or proc.stdout.strip())
    return proc.stdout


def psycopg_execute(sql: str, *, dsn: str = STORAGE_DSN, fetch: bool = False) -> str | list[list[str]]:
    if psycopg is None:
        raise SystemExit("psql not found and psycopg is not installed. Install PostgreSQL client tools or run inside .venv.")
    try:
        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                if fetch:
                    return [["" if value is None else str(value) for value in row] for row in cur.fetchall()]
            conn.commit()
    except Exception as exc:
        raise SystemExit(str(exc)) from exc
    return ""


def run_sql(sql: str, *, dsn: str = STORAGE_DSN) -> str:
    try:
        return psql(["-q"], dsn=dsn, input_sql=sql)
    except SystemExit as exc:
        if "psql not found" not in str(exc):
            raise
        return str(psycopg_execute(sql, dsn=dsn))


def query_tsv(sql: str, *, dsn: str = STORAGE_DSN) -> list[list[str]]:
    try:
        out = psql(["-A", "-t", "-F", "\t", "-c", sql], dsn=dsn)
    except SystemExit as exc:
        if "psql not found" not in str(exc):
            raise
        return psycopg_execute(sql, dsn=dsn, fetch=True)  # type: ignore[return-value]
    rows = []
    for line in out.splitlines():
        if line.strip():
            rows.append(line.split("\t"))
    return rows


def active_terms() -> set[str]:
    data = load_json(TERMS_PATH)
    return {t["term"] for t in data.get("terms", [])}


def active_layers() -> set[str]:
    data = load_json(ONTOLOGY_PATH)
    return set(data.get("initial_graph_layers", []))


def non_empty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def has_proof_note(item: dict[str, Any]) -> bool:
    payload = item.get("payload") or {}
    if not isinstance(payload, dict):
        return False
    for key in ("evidence_note", "proof_note"):
        if non_empty_text(payload.get(key)):
            return True
    return False


def normalized_layers(obj: dict[str, Any]) -> list[dict[str, Any]]:
    payload = obj.get("payload") if isinstance(obj.get("payload"), dict) else {}
    raw_layers = obj.get("layers", payload.get("layers", []))
    if raw_layers is None:
        return []
    if not isinstance(raw_layers, list):
        return [{"_error": "layers must be a list"}]
    layers: list[dict[str, Any]] = []
    for entry in raw_layers:
        if isinstance(entry, str):
            layers.append({"layer": entry, "role": "", "confidence_bps": 0, "detail": {}})
        elif isinstance(entry, dict):
            layers.append({
                "layer": entry.get("layer"),
                "role": entry.get("role", ""),
                "confidence_bps": entry.get("confidence_bps", 0),
                "detail": entry.get("detail", {}),
            })
        else:
            layers.append({"_error": "layer entries must be strings or objects"})
    return layers


def validate_layers(obj: dict[str, Any], prefix: str) -> list[str]:
    errors: list[str] = []
    allowed = active_layers()
    for layer in normalized_layers(obj):
        if layer.get("_error"):
            errors.append(f"{prefix}.{layer['_error']}")
            continue
        if layer.get("layer") not in allowed:
            errors.append(f"{prefix}.unknown layer: {layer.get('layer')}")
        if layer.get("confidence_bps", 0) not in BPS:
            errors.append(f"{prefix}.bad layer confidence_bps: {layer.get('confidence_bps')}")
        if not isinstance(layer.get("role", ""), str):
            errors.append(f"{prefix}.layer role must be string")
        if not isinstance(layer.get("detail", {}), dict):
            errors.append(f"{prefix}.layer detail must be object")
    return errors


def validate_item(item: dict[str, Any], *, require_approved_fields: bool = False) -> list[str]:
    errors: list[str] = []
    terms = active_terms()
    required = ["uuid", "term", "status", "time_on_graph", "location_at_on_graph", "location_real_at_added"]
    for key in required:
        if key not in item:
            errors.append(f"missing item.{key}")
    if "uuid" in item:
        try:
            uuid.UUID(str(item["uuid"]))
        except (AttributeError, TypeError, ValueError):
            errors.append("item.uuid is not a UUID")
    if item.get("term") not in terms:
        errors.append(f"unknown term: {item.get('term')}")
    if item.get("status") not in STATUSES:
        errors.append(f"bad status: {item.get('status')}")
    if not item.get("location_at_on_graph"):
        errors.append("location_at_on_graph cannot be empty")
    if not isinstance(item.get("location_real_at_added", {}), dict):
        errors.append("location_real_at_added must be object")
    if not isinstance(item.get("payload", {}), dict):
        errors.append("payload must be object")
    errors.extend(validate_layers(item, "item"))
    if item.get("status") == "approved" or require_approved_fields:
        for key in ["time_approved", "approval_scope", "operator_uuid"]:
            if not item.get(key):
                errors.append(f"approved item missing {key}")
        if item.get("approval_scope") not in APPROVAL_SCOPES:
            errors.append(f"bad approval_scope: {item.get('approval_scope')}")
        if item.get("term") == "CLAIM" and not has_proof_note(item):
            errors.append("approved CLAIM requires payload.evidence_note or payload.proof_note")
    payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
    if item.get("term") == "HYPOTHESIS" and (item.get("inferred") is True or payload.get("inferred") is True):
        confidence = item.get("confidence_bps", payload.get("confidence_bps", 0))
        if confidence not in BPS or confidence <= 0:
            errors.append("inferred HYPOTHESIS requires positive approved confidence_bps")
    return errors


def validate_edge(edge: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = ["edge_uuid", "source_uuid", "target_uuid", "edge_type", "status", "current_status", "current_unknown"]
    for key in required:
        if key not in edge:
            errors.append(f"missing edge.{key}")
    for key in ["edge_uuid", "source_uuid", "target_uuid"]:
        if key in edge:
            try:
                uuid.UUID(str(edge[key]))
            except (AttributeError, TypeError, ValueError):
                errors.append(f"edge.{key} is not a UUID")
    if edge.get("status") not in STATUSES:
        errors.append(f"bad edge status: {edge.get('status')}")
    if edge.get("current_status") not in CURRENT_STATUSES:
        errors.append(f"bad current_status: {edge.get('current_status')}")
    if not isinstance(edge.get("current_unknown", True), bool):
        errors.append("current_unknown must be boolean")
    if not edge.get("edge_type"):
        errors.append("edge_type cannot be empty")
    if edge.get("confidence_bps", 0) not in BPS:
        errors.append(f"bad confidence_bps: {edge.get('confidence_bps')}")
    errors.extend(validate_layers(edge, "edge"))
    return errors


def cmd_init(_: argparse.Namespace) -> None:
    if not SCHEMA_SQL.exists():
        raise SystemExit(f"missing schema: {SCHEMA_SQL}")
    psql(["-f", str(SCHEMA_SQL)], dsn=STORAGE_DSN)
    print(json.dumps({"ok": True, "storage_dsn": STORAGE_DSN, "schema": str(SCHEMA_SQL)}, indent=2))


def cmd_bootstrap_dbs(_: argparse.Namespace) -> None:
    """Create state/storage databases when the local Postgres user can do it."""
    admin_dsn = os.environ.get("LUCIDOTA_GO_ADMIN_DSN", "postgresql:///postgres")
    dbs = [
        os.environ.get("LUCIDOTA_GO_STATE_DB", "lucidota_state"),
        os.environ.get("LUCIDOTA_GO_STORAGE_DB", "lucidota_storage"),
    ]
    created: list[str] = []
    for db in dbs:
        rows = query_tsv(f"SELECT 1 FROM pg_database WHERE datname={sql_literal(db)};", dsn=admin_dsn)
        if not rows:
            # CREATE DATABASE cannot run inside a transaction block. psql -c is fine.
            psql(["-c", f'CREATE DATABASE "{db}"'], dsn=admin_dsn)
            created.append(db)
    print(json.dumps({"ok": True, "admin_dsn": admin_dsn, "databases": dbs, "created": created}, indent=2))


def normalize_packet(packet: dict[str, Any]) -> dict[str, Any]:
    packet = dict(packet)
    packet.setdefault("packet_uuid", str(uuid.uuid4()))
    packet.setdefault("source_id", "")
    packet.setdefault("parser_name", "go_ingest_manual")
    packet.setdefault("proposed_term", packet.get("proposed_item", {}).get("term"))
    packet.setdefault("raw_anchor", "")
    packet.setdefault("claim", "")
    packet.setdefault("proposed_item", {})
    packet.setdefault("proposed_edges", [])
    packet.setdefault("status", "pending")
    packet.setdefault("confidence_bps", 0)
    return packet


def validate_packet(packet: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    packet = normalize_packet(packet)
    errors: list[str] = []
    if packet.get("confidence_bps", 0) not in BPS:
        errors.append(f"bad packet confidence_bps: {packet.get('confidence_bps')}")
    item = packet.get("proposed_item") or {}
    if item:
        item.setdefault("uuid", str(uuid.uuid4()))
        item.setdefault("status", "staged")
        item.setdefault("time_on_graph", now_iso())
        item.setdefault("location_at_on_graph", f"staging_packet:{packet['packet_uuid']}")
        item.setdefault("location_real_at_added", {})
        packet["proposed_item"] = item
        payload = item.setdefault("payload", {})
        if isinstance(payload, dict) and item.get("term") == "HYPOTHESIS":
            payload.setdefault("confidence_bps", packet.get("confidence_bps", 0))
        errors.extend(validate_item(item))
    for edge in packet.get("proposed_edges", []):
        edge.setdefault("edge_uuid", str(uuid.uuid4()))
        edge.setdefault("status", "staged")
        edge.setdefault("current_status", "unknown")
        edge.setdefault("current_unknown", True)
        errors.extend(validate_edge(edge))
    return packet, errors


def iter_packets(path: str | Path) -> list[dict[str, Any]]:
    packet_path = Path(path)
    if packet_path.suffix.lower() in {".jsonl", ".ndjson"}:
        packets: list[dict[str, Any]] = []
        for lineno, line in enumerate(packet_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise SystemExit(f"packet JSONL line {lineno} must contain an object")
            packets.append(obj)
        return packets
    data = load_json(packet_path)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("packets"), list):
        return data["packets"]
    if isinstance(data, dict):
        return [data]
    raise SystemExit("packet file must contain an object, a list, or {packets:[...]}")


def stage_one(packet: dict[str, Any], operator: str | None, dry_run: bool = False) -> dict[str, Any]:
    packet, errors = validate_packet(packet)
    if errors:
        raise SystemExit(json.dumps({"ok": False, "errors": errors}, indent=2))
    item = packet.get("proposed_item") or {}
    if dry_run:
        return {"packet_uuid": packet["packet_uuid"], "item_uuid": item.get("uuid"), "validated": True, "dry_run": True}
    sql = f"""
INSERT INTO lucidota_go.staging_packet (
  packet_uuid, source_id, parser_name, proposed_term, raw_anchor, claim,
  proposed_item, proposed_edges, status, confidence_bps, operator_uuid
) VALUES (
  {sql_literal(packet['packet_uuid'])}::uuid,
  {sql_literal(packet['source_id'])},
  {sql_literal(packet['parser_name'])},
  {sql_literal(packet.get('proposed_term'))},
  {sql_literal(packet['raw_anchor'])},
  {sql_literal(packet['claim'])},
  {sql_literal(packet['proposed_item'])}::jsonb,
  {sql_literal(packet['proposed_edges'])}::jsonb,
  {sql_literal(packet['status'])},
  {int(packet.get('confidence_bps', 0))},
  {sql_literal(operator)}::uuid
)
ON CONFLICT (packet_uuid) DO NOTHING;
"""
    run_sql(sql)
    return {"packet_uuid": packet["packet_uuid"], "item_uuid": item.get("uuid"), "staged": True}


def cmd_stage(ns: argparse.Namespace) -> None:
    results = [stage_one(packet, ns.operator, ns.dry_run) for packet in iter_packets(ns.packet)]
    print(json.dumps({"ok": True, "count": len(results), "results": results}, indent=2, sort_keys=True))


def fetch_packet(packet_uuid: str) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    rows = query_tsv(f"""
SELECT proposed_item::text, proposed_edges::text, status
FROM lucidota_go.staging_packet
WHERE packet_uuid = {sql_literal(packet_uuid)}::uuid
LIMIT 1;
""")
    if not rows:
        raise SystemExit(f"packet not found: {packet_uuid}")
    item_s, edges_s, status = rows[0]
    return json.loads(item_s or "{}"), json.loads(edges_s or "[]"), status


def insert_item_sql(item: dict[str, Any], operator: str, approval_scope: str) -> str:
    item = dict(item)
    item["status"] = "approved"
    item["time_approved"] = now_iso()
    item["approval_scope"] = approval_scope
    item["operator_uuid"] = operator
    item.setdefault("location_real_at_approved", item.get("location_real_at_added", {}))
    errors = validate_item(item)
    if errors:
        raise SystemExit(json.dumps({"ok": False, "errors": errors}, indent=2))
    return f"""
INSERT INTO lucidota_go.graph_item (
  uuid, term, label, status, canonical_uuid, time_on_graph, location_at_on_graph,
  location_real_at_added, time_approved, location_real_at_approved, approval_scope,
  operator_uuid, payload
) VALUES (
  {sql_literal(item['uuid'])}::uuid,
  {sql_literal(item['term'])},
  {sql_literal(item.get('label',''))},
  'approved',
  {sql_literal(item.get('canonical_uuid'))}::uuid,
  {sql_literal(item['time_on_graph'])}::timestamptz,
  {sql_literal(item['location_at_on_graph'])},
  {sql_literal(item.get('location_real_at_added',{}))}::jsonb,
  {sql_literal(item['time_approved'])}::timestamptz,
  {sql_literal(item.get('location_real_at_approved',{}))}::jsonb,
  {sql_literal(item['approval_scope'])},
  {sql_literal(operator)}::uuid,
  {sql_literal(item.get('payload',{}))}::jsonb
)
ON CONFLICT (uuid) DO UPDATE SET
  status='approved',
  time_approved=EXCLUDED.time_approved,
  location_real_at_approved=EXCLUDED.location_real_at_approved,
  approval_scope=EXCLUDED.approval_scope,
  operator_uuid=EXCLUDED.operator_uuid,
  payload=EXCLUDED.payload,
  updated_at=now();
"""


def item_layer_sql(item: dict[str, Any], operator: str) -> str:
    rows = []
    for layer in normalized_layers(item):
        rows.append(f"""
INSERT INTO lucidota_go.graph_item_layer (uuid, layer, role, confidence_bps, operator_uuid, detail)
VALUES (
  {sql_literal(item['uuid'])}::uuid,
  {sql_literal(layer['layer'])},
  {sql_literal(layer.get('role', ''))},
  {int(layer.get('confidence_bps', 0))},
  {sql_literal(operator)}::uuid,
  {sql_literal(layer.get('detail', {}))}::jsonb
)
ON CONFLICT (uuid, layer, role) DO UPDATE SET
  confidence_bps=EXCLUDED.confidence_bps,
  operator_uuid=EXCLUDED.operator_uuid,
  detail=EXCLUDED.detail;
""")
    return "\n".join(rows)


def edge_layer_sql(edge: dict[str, Any], operator: str) -> str:
    rows = []
    for layer in normalized_layers(edge):
        rows.append(f"""
INSERT INTO lucidota_go.graph_edge_layer (edge_uuid, layer, role, confidence_bps, operator_uuid, detail)
VALUES (
  {sql_literal(edge['edge_uuid'])}::uuid,
  {sql_literal(layer['layer'])},
  {sql_literal(layer.get('role', ''))},
  {int(layer.get('confidence_bps', 0))},
  {sql_literal(operator)}::uuid,
  {sql_literal(layer.get('detail', {}))}::jsonb
)
ON CONFLICT (edge_uuid, layer, role) DO UPDATE SET
  confidence_bps=EXCLUDED.confidence_bps,
  operator_uuid=EXCLUDED.operator_uuid,
  detail=EXCLUDED.detail;
""")
    return "\n".join(rows)


def insert_edge_sql(edge: dict[str, Any], operator: str) -> str:
    errors = validate_edge(edge)
    if errors:
        raise SystemExit(json.dumps({"ok": False, "errors": errors}, indent=2))
    return f"""
INSERT INTO lucidota_go.graph_edge (
  edge_uuid, source_uuid, target_uuid, edge_type, term, relationship_family, status,
  valid_from, valid_to, current_status, current_unknown, will_be, location_vector,
  location_uuid, evidence_uuid, operator_uuid, detail
) VALUES (
  {sql_literal(edge['edge_uuid'])}::uuid,
  {sql_literal(edge['source_uuid'])}::uuid,
  {sql_literal(edge['target_uuid'])}::uuid,
  {sql_literal(edge['edge_type'])},
  {sql_literal(edge.get('term'))},
  {sql_literal(edge.get('relationship_family'))},
  {sql_literal(edge.get('status','approved'))},
  {sql_literal(edge.get('valid_from'))}::timestamptz,
  {sql_literal(edge.get('valid_to'))}::timestamptz,
  {sql_literal(edge.get('current_status','unknown'))},
  {'true' if edge.get('current_unknown', True) else 'false'},
  {sql_literal(edge.get('will_be',''))},
  {sql_literal(edge.get('location_vector'))},
  {sql_literal(edge.get('location_uuid'))}::uuid,
  {sql_literal(edge.get('evidence_uuid'))}::uuid,
  {sql_literal(operator)}::uuid,
  {sql_literal(edge.get('detail',{}))}::jsonb
)
ON CONFLICT (edge_uuid) DO NOTHING;
"""


def promote_packet(packet_uuid: str, operator: str, approval_scope: str) -> dict[str, Any]:
    item, edges, status = fetch_packet(packet_uuid)
    if status == "approved":
        return {"packet_uuid": packet_uuid, "already_approved": True}
    if not item:
        raise SystemExit(f"packet has no proposed_item: {packet_uuid}")
    sql_parts = ["BEGIN;"]
    sql_parts.append(insert_item_sql(item, operator, approval_scope))
    sql_parts.append(item_layer_sql(item, operator))
    for edge in edges:
        edge["status"] = "approved"
        edge["operator_uuid"] = operator
        sql_parts.append(insert_edge_sql(edge, operator))
        sql_parts.append(edge_layer_sql(edge, operator))
    sql_parts.append(f"""
UPDATE lucidota_go.staging_packet
SET status='approved', operator_uuid={sql_literal(operator)}::uuid, decided_at=now()
WHERE packet_uuid={sql_literal(packet_uuid)}::uuid;
INSERT INTO lucidota_go.graph_journal (item_uuid, operator_uuid, action, reason, after_state)
VALUES (
  {sql_literal(item['uuid'])}::uuid,
  {sql_literal(operator)}::uuid,
  'approve',
  {sql_literal('promoted from staging_packet ' + packet_uuid)},
  {sql_literal({'packet_uuid': packet_uuid, 'edges': len(edges)})}::jsonb
);
COMMIT;
""")
    run_sql("\n".join(sql_parts))
    return {"packet_uuid": packet_uuid, "item_uuid": item["uuid"], "edges": len(edges), "approved": True}


def cmd_promote(ns: argparse.Namespace) -> None:
    results = [promote_packet(packet_uuid, ns.operator, ns.approval_scope) for packet_uuid in ns.packet_uuid]
    print(json.dumps({"ok": True, "count": len(results), "results": results}, indent=2, sort_keys=True))


def cmd_batch_approve(ns: argparse.Namespace) -> None:
    rows = query_tsv(f"""
SELECT packet_uuid::text
FROM lucidota_go.staging_packet
WHERE status='pending'
ORDER BY created_at ASC
LIMIT {int(ns.limit)};
""")
    results = []
    for (packet_uuid,) in rows:
        results.append(promote_packet(packet_uuid, ns.operator, ns.approval_scope))
    print(json.dumps({"ok": True, "count": len(results), "results": results}, indent=2, sort_keys=True))


def cmd_validate_packet(ns: argparse.Namespace) -> None:
    results = []
    ok = True
    for raw in iter_packets(ns.packet):
        packet, errors = validate_packet(raw)
        ok = ok and not errors
        results.append({"packet_uuid": packet.get("packet_uuid"), "ok": not errors, "errors": errors})
    print(json.dumps({"ok": ok, "count": len(results), "results": results}, indent=2, sort_keys=True))
    if not ok:
        raise SystemExit(1)


def cmd_validate_item(ns: argparse.Namespace) -> None:
    item = load_json(ns.item)
    errors = validate_item(item)
    print(json.dumps({"ok": not errors, "errors": errors}, indent=2))
    if errors:
        raise SystemExit(1)


def cmd_validate_edge(ns: argparse.Namespace) -> None:
    edge = load_json(ns.edge)
    errors = validate_edge(edge)
    print(json.dumps({"ok": not errors, "errors": errors}, indent=2))
    if errors:
        raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="GO Postgres ingest/promote lane")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("bootstrap-dbs")
    p.set_defaults(func=cmd_bootstrap_dbs)

    p = sub.add_parser("init")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("validate-item")
    p.add_argument("item")
    p.set_defaults(func=cmd_validate_item)

    p = sub.add_parser("validate-edge")
    p.add_argument("edge")
    p.set_defaults(func=cmd_validate_edge)

    p = sub.add_parser("validate-packet")
    p.add_argument("packet")
    p.set_defaults(func=cmd_validate_packet)

    p = sub.add_parser("stage")
    p.add_argument("packet")
    p.add_argument("--operator", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_stage)

    p = sub.add_parser("promote")
    p.add_argument("packet_uuid", nargs="+")
    p.add_argument("--operator", required=True)
    p.add_argument("--approval-scope", default="current", choices=sorted(APPROVAL_SCOPES))
    p.set_defaults(func=cmd_promote)

    p = sub.add_parser("batch-approve")
    p.add_argument("--operator", required=True)
    p.add_argument("--approval-scope", default="current", choices=sorted(APPROVAL_SCOPES))
    p.add_argument("--limit", type=int, default=25)
    p.set_defaults(func=cmd_batch_approve)

    return ap


def main(argv: list[str] | None = None) -> int:
    ns = build_parser().parse_args(argv)
    ns.func(ns)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
