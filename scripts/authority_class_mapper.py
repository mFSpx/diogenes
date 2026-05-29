#!/usr/bin/env python3
"""Map and enforce authority_class on extraction outputs.

Hardening role: every model/extractor output must carry an explicit authority
class before it can move toward command, graph, or evidence workflows. This tool
is intentionally bounded: dry validation is default, input JSON is size-capped,
and outputs are processed in a fixed batch.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "06_SCHEMA/090_authority_class_mapper.sql"
OUT = ROOT / "05_OUTPUTS/contracts"
MAX_OUTPUTS_DEFAULT = 2000
MAX_INPUT_BYTES = 2 * 1024 * 1024


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: str | Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def db(args: argparse.Namespace) -> str:
    return args.database_url or os.environ.get("KORPUS_DATABASE_URL") or os.environ.get("DATABASE_URL") or "postgresql:///lucidota_storage"


def write(name: str, payload: dict[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"authority_class_mapper_{name}_{stamp()}.json"
    payload.setdefault("generated_at", now())
    payload["report_path"] = rel(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"REPORT_PATH={rel(path)}")
    return path


def init(args: argparse.Namespace) -> int:
    blockers: list[str] = []
    if not SCHEMA.exists():
        blockers.append("schema_missing")
    if args.execute and not blockers:
        with psycopg.connect(db(args)) as conn:
            with conn.cursor() as cur:
                cur.execute(SCHEMA.read_text(encoding="utf-8"))
            conn.commit()
    write(
        "init_schema_execute" if args.execute and not blockers else "init_schema_dry_run",
        {"action": "init_schema", "execute_performed": bool(args.execute and not blockers), "schema": rel(SCHEMA), "blockers": blockers, "status": "PASS" if not blockers else "FAIL"},
    )
    return 0 if not blockers else 4


def resolve_json_input(raw: str) -> Any:
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    if candidate.exists():
        if not candidate.is_file():
            raise ValueError(f"input_not_file:{candidate}")
        size = candidate.stat().st_size
        if size > MAX_INPUT_BYTES:
            raise ValueError(f"input_too_large:{size}>{MAX_INPUT_BYTES}")
        return json.loads(candidate.read_text(encoding="utf-8"))
    raw_size = len(raw.encode("utf-8"))
    if raw_size > MAX_INPUT_BYTES:
        raise ValueError(f"inline_json_too_large:{raw_size}>{MAX_INPUT_BYTES}")
    return json.loads(raw)


def load_outputs(raw: str | None, max_outputs: int) -> list[dict[str, Any]]:
    if raw:
        data = resolve_json_input(raw)
    else:
        data = [
            {"output_kind": "sticker_feature_vector", "extractor_name": "sticker_feature_extractor_v1", "payload": {"x": 1}},
            {"output_kind": "operator_label", "extractor_name": "operator", "authority_class": "operator_defined_label", "payload": {"label": "Operator"}},
        ]
    if isinstance(data, dict):
        data = data.get("outputs", [data])
    if not isinstance(data, list):
        raise ValueError("outputs_must_be_list")
    if len(data) > max_outputs:
        raise ValueError(f"outputs_over_limit:{len(data)}>{max_outputs}")
    for i, out in enumerate(data):
        if not isinstance(out, dict):
            raise ValueError(f"output_not_object:{i}")
    return data


def lookup(cur, kind: str, extractor: str) -> str | None:
    cur.execute(
        """
        SELECT authority_class::text
        FROM lucidota_archaeology.authority_class_mapping
        WHERE active AND output_kind=%s AND extractor_name IN (%s,'*')
        ORDER BY (extractor_name=%s) DESC
        LIMIT 1
        """,
        (kind, extractor, extractor),
    )
    row = cur.fetchone()
    return row["authority_class"] if row else None


def validate(args: argparse.Namespace) -> int:
    max_outputs = max(1, min(int(args.max_outputs), MAX_OUTPUTS_DEFAULT))
    outputs = load_outputs(args.outputs_json, max_outputs)
    results: list[dict[str, Any]] = []
    blockers: list[str] = []
    mapped: list[dict[str, Any]] = []
    with psycopg.connect(db(args), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            for i, out in enumerate(outputs):
                kind = str(out.get("output_kind", "")).strip()
                extractor = str(out.get("extractor_name", "*")).strip() or "*"
                if not kind:
                    blockers.append(f"missing_output_kind:{i}")
                if not isinstance(out.get("payload", {}), dict):
                    blockers.append(f"payload_must_be_object:{i}")
                mapped_class = lookup(cur, kind, extractor) if kind else None
                have = out.get("authority_class")
                if not mapped_class:
                    blockers.append(f"no_mapping:{i}:{kind}")
                if have and mapped_class and have != mapped_class:
                    blockers.append(f"authority_mismatch:{i}:{have}!={mapped_class}")
                if not have and not args.apply:
                    blockers.append(f"missing_authority_class:{i}")
                new = dict(out)
                if args.apply and mapped_class and not have:
                    new["authority_class"] = mapped_class
                mapped.append(new)
                results.append({"index": i, "output_kind": kind, "extractor_name": extractor, "input_authority_class": have, "mapped_authority_class": mapped_class, "passed": bool(mapped_class and (have == mapped_class or (args.apply and not have)))})
    report = {"action": "validate", "execute_performed": False, "apply": bool(args.apply), "max_outputs": max_outputs, "outputs_checked": len(outputs), "results": results, "mapped_outputs": mapped, "blockers": blockers, "status": "PASS" if not blockers else "FAIL"}
    write("validate_pass" if not blockers else "validate_fail", report)
    print("AUTHORITY_CLASS_MAPPER=" + report["status"])
    return 0 if not blockers else 4


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-url")
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("init-schema")
    s.add_argument("--execute", action="store_true")
    v = sub.add_parser("validate")
    v.add_argument("--outputs-json")
    v.add_argument("--apply", action="store_true")
    v.add_argument("--max-outputs", type=int, default=MAX_OUTPUTS_DEFAULT)
    args = parser.parse_args()
    return init(args) if args.cmd == "init-schema" else validate(args)


if __name__ == "__main__":
    raise SystemExit(main())
