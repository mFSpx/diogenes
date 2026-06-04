#!/usr/bin/env python3
"""Render live RPC helper packets from PostgREST."""
from __future__ import annotations

import argparse
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_BASE_URL = "http://127.0.0.1:3000"
ROOT = Path(__file__).resolve().parents[1]
DB_URL = os.environ.get("ABSURD_SYSTEM_DATABASE_URL") or os.environ.get("DATABASE_URL") or "postgresql:///lucidota_state"


def get_node_row(node_id: str) -> dict[str, Any] | None:
    try:
        import psycopg
    except Exception:
        return None
    try:
        with psycopg.connect(DB_URL, connect_timeout=2) as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    node_id, parent_id, node_sort_key, manual_id, source_refs, node_kind,
                    payload_format, payload, ontology_tags, dependencies, affects_nodes,
                    status, version, hash_current, previous_hash, created_at, updated_at
                FROM lucidota_canon.bible_nodes
                WHERE node_id = %s
                LIMIT 1
                """,
                (node_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            cols = [d.name for d in cur.description]
            return dict(zip(cols, row, strict=True))
    except Exception:
        return None


def post_json(base_url: str, path: str, payload: dict[str, Any]) -> tuple[str, Any]:
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
    data = json.dumps(payload, default=str, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"content-type": "application/json", "accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        return url, json.loads(resp.read().decode("utf-8") or "null")


def get_json(base_url: str, path: str) -> tuple[str, Any]:
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
    with urllib.request.urlopen(url, timeout=5) as resp:
        return url, json.loads(resp.read().decode("utf-8") or "null")


def decode_payload_json(text: str) -> dict[str, Any]:
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("payload_json must decode to an object")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description="Show live RPC helper packets from PostgREST.")
    sub = ap.add_subparsers(dest="action", required=True)

    sp = sub.add_parser("subtree")
    sp.add_argument("--root-id", required=True)
    sp.add_argument("--base-url", default=DEFAULT_BASE_URL)
    sp.add_argument("--json", action="store_true")

    sp = sub.add_parser("sort-key")
    sp.add_argument("--node-id", required=True)
    sp.add_argument("--base-url", default=DEFAULT_BASE_URL)
    sp.add_argument("--json", action="store_true")

    sp = sub.add_parser("material")
    sp.add_argument("--node-id", required=True)
    sp.add_argument("--base-url", default=DEFAULT_BASE_URL)
    sp.add_argument("--json", action="store_true")

    sp = sub.add_parser("cloud-packet")
    sp.add_argument("--work-order-id", required=True)
    sp.add_argument("--base-url", default=DEFAULT_BASE_URL)
    sp.add_argument("--max-chars", type=int, default=8_000)
    sp.add_argument("--max-items", type=int, default=12)
    sp.add_argument("--task-type", default="")
    sp.add_argument("--target-model", default="")
    sp.add_argument("--include-raw-bodies", action="store_true")
    sp.add_argument("--json", action="store_true")

    sp = sub.add_parser("file-prompt")
    sp.add_argument("--payload-json", required=True)
    sp.add_argument("--base-url", default=DEFAULT_BASE_URL)
    sp.add_argument("--json", action="store_true")

    sp = sub.add_parser("decompose-prompt")
    sp.add_argument("--payload-json", required=True)
    sp.add_argument("--base-url", default=DEFAULT_BASE_URL)
    sp.add_argument("--json", action="store_true")

    sp = sub.add_parser("link-prompt")
    sp.add_argument("--payload-json", required=True)
    sp.add_argument("--base-url", default=DEFAULT_BASE_URL)
    sp.add_argument("--json", action="store_true")

    args = ap.parse_args()
    if args.action == "subtree":
        url, payload = get_json(args.base_url, f"rpc/get_subtree?root_id={urllib.parse.quote(args.root_id)}")
    elif args.action == "sort-key":
        url, payload = get_json(args.base_url, f"rpc/fn_bible_node_sort_key?p_node_id={urllib.parse.quote(args.node_id)}")
    elif args.action == "material":
        node = get_node_row(args.node_id)
        if node is None:
            raise SystemExit(f"node_not_found:{args.node_id}")
        url, payload = post_json(args.base_url, "rpc/fn_bible_node_material", {"node_row": node})
    elif args.action == "cloud-packet":
        url, payload = post_json(
            args.base_url,
            "rpc/cloud_packet",
            {
                "work_order_id": args.work_order_id,
                "max_chars": args.max_chars,
                "max_items": args.max_items,
                "task_type": args.task_type,
                "target_model": args.target_model,
                "include_raw_bodies": args.include_raw_bodies,
            },
        )
    elif args.action == "file-prompt":
        url, payload = post_json(args.base_url, "rpc/file_prompt", decode_payload_json(args.payload_json))
    elif args.action == "decompose-prompt":
        url, payload = post_json(args.base_url, "rpc/decompose_prompt_to_work_orders", decode_payload_json(args.payload_json))
    else:
        url, payload = post_json(args.base_url, "rpc/link_prompt_work_order", decode_payload_json(args.payload_json))

    if args.json:
        print(json.dumps({"source_url": url, "payload": payload}, sort_keys=True, ensure_ascii=False))
    else:
        print(f"URL={url}")
        print(json.dumps(payload, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
