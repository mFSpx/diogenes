#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import psycopg

ROOT = Path(__file__).resolve().parents[1]
LIVE_DSN = "postgresql:///lucidota_state"


def test_luci_cloud_packet_shell_requests_bounded_prompt_packet() -> None:
    work_order_uuid = None
    with psycopg.connect(LIVE_DSN, connect_timeout=3) as conn, conn.cursor() as cur:
        cur.execute("select work_order_uuid::text from lucidota_control.work_order order by created_at desc limit 1")
        row = cur.fetchone()
        work_order_uuid = row[0] if row else None
    assert work_order_uuid

    proc = subprocess.run(
        [str(ROOT / "luci"), "cloud", "packet", "--work-order-id", work_order_uuid, "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["contract_name"] == "prompt_api.cloud_packet.v1"
    assert payload["work_order_id"] == work_order_uuid
