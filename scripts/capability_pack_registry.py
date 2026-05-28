#!/usr/bin/env python3
"""Register ontology capability packs into the durable capability registry.

This is intentionally small: read pack registry JSON, map it into the existing
`lucidota_investigation.capability_registry` table, and emit a receipt.
No new schema island, no agent sprawl.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:  # Optional in tests; present in the live venv.
    import psycopg  # type: ignore
except Exception:  # pragma: no cover - exercised only without psycopg installed.
    psycopg = None  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
PACK_ROOT = ROOT / "BOOKS" / "ontology_packs"
OUT_DIR = ROOT / "05_OUTPUTS" / "runtime"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path | str, root: Path = ROOT) -> str:
    try:
        return str(Path(path).resolve().relative_to(root))
    except Exception:
        return str(path)


def discover_pack_registries(root: Path = PACK_ROOT) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.glob("*/registry.json") if p.is_file())


def load_pack_registry(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pack_to_capability_row(pack: dict[str, Any], source_path: Path) -> dict[str, Any]:
    pack_id = str(pack.get("pack_id") or source_path.parent.name).strip()
    pack_name = str(pack.get("pack_name") or pack_id).strip()
    version = str(pack.get("version") or "").strip()
    status = str(pack.get("status") or "staged_not_promoted").strip()
    first_test = pack.get("recommended_first_test") or {}
    primary_chain = pack.get("primary_chain") or []
    row = {
        "capability_key": f"ontology-pack-{pack_id}",
        "capability_group": "Ontology Packs",
        "capability_name": f"{pack_name} [{version or 'unversioned'}]",
        "lifecycle_status": "prototype" if status != "promoted" else "active",
        "run_state": "planned",
        "workflow_name": "ontology-pack-registry",
        "command": "scripts/capability_pack_registry.py register",
        "detail": {
            "pack_id": pack_id,
            "pack_name": pack_name,
            "version": version,
            "status": status,
            "primary_chain": primary_chain,
            "graph_safety_rule": pack.get("graph_safety_rule", ""),
            "recommended_first_test": first_test,
            "pillar_count": len(pack.get("pillars") or []),
            "source_registry": rel(source_path),
        },
    }
    return row


def capability_rows_from_packs(pack_paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in pack_paths:
        pack = load_pack_registry(path)
        rows.append(pack_to_capability_row(pack, path))
    return rows


def upsert_capability_rows(conn: Any, rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    sql = """
    INSERT INTO lucidota_investigation.capability_registry
    (capability_key, capability_group, capability_name, lifecycle_status, run_state, workflow_name, command, detail)
    VALUES (%(capability_key)s, %(capability_group)s, %(capability_name)s, %(lifecycle_status)s, %(run_state)s,
            %(workflow_name)s, %(command)s, %(detail)s::jsonb)
    ON CONFLICT (capability_key) DO UPDATE SET
        capability_group = EXCLUDED.capability_group,
        capability_name = EXCLUDED.capability_name,
        lifecycle_status = EXCLUDED.lifecycle_status,
        run_state = EXCLUDED.run_state,
        workflow_name = EXCLUDED.workflow_name,
        command = EXCLUDED.command,
        detail = EXCLUDED.detail,
        updated_at = now()
    """
    with conn.cursor() as cur:
        for row in rows:
            params = dict(row)
            params["detail"] = json.dumps(row["detail"], sort_keys=True)
            cur.execute(sql, params)
    conn.commit()
    return len(rows)


def write_receipt(payload: dict[str, Any]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"capability_pack_registry_{stamp()}.json"
    payload = dict(payload)
    payload.setdefault("generated_at", now())
    payload["receipt_path"] = rel(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return path


def run_register(*, execute: bool, database_url: str | None, pack_root: Path = PACK_ROOT) -> dict[str, Any]:
    pack_paths = discover_pack_registries(pack_root)
    rows = capability_rows_from_packs(pack_paths)
    db_result: dict[str, Any] = {"attempted": bool(execute), "upserted": 0}
    if execute:
        if psycopg is None:
            raise SystemExit("psycopg is unavailable; cannot execute registry upsert")
        if not database_url:
            raise SystemExit("database url required for --execute")
        with psycopg.connect(database_url, connect_timeout=5) as conn:
            db_result["upserted"] = upsert_capability_rows(conn, rows)
    payload = {
        "schema": "lucidota.capability_pack_registry.v1",
        "ok": True,
        "pack_count": len(rows),
        "pack_paths": [rel(p) for p in pack_paths],
        "rows": rows,
        "db_result": db_result,
    }
    receipt = write_receipt(payload)
    payload["receipt_path"] = rel(receipt)
    return payload


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Register ontology pack registries into the capability registry.")
    p.add_argument("--database-url", default=None)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--json", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("register")
    r.set_defaults(func=_cmd_register)
    s = sub.add_parser("scan")
    s.set_defaults(func=_cmd_scan)
    return p


def _cmd_register(args: argparse.Namespace) -> dict[str, Any]:
    return run_register(execute=bool(args.execute), database_url=args.database_url)


def _cmd_scan(args: argparse.Namespace) -> dict[str, Any]:
    rows = capability_rows_from_packs(discover_pack_registries())
    return {
        "schema": "lucidota.capability_pack_registry.scan.v1",
        "ok": True,
        "pack_count": len(rows),
        "rows": rows,
    }


def main() -> int:
    args = build_parser().parse_args()
    result = args.func(args)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
    else:
        print(result)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
