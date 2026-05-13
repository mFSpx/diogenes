#!/usr/bin/env python3
"""Print the current LUCIDOTA model runtime registry."""

from __future__ import annotations

import os

import psycopg


def main() -> int:
    db_url = os.environ.get(
        "LUCIDOTA_CONTROL_DATABASE_URL",
        "postgresql://mfspx@/lucidota_state",
    )
    sql = """
    SELECT
        l.loadout_id,
        s.slot_name,
        s.instance_count,
        m.model_id,
        m.role,
        m.benchmark_status,
        coalesce(m.quantization, '') AS quantization
    FROM lucidota_runtime.resident_loadout l
    JOIN lucidota_runtime.resident_loadout_slot s ON s.loadout_id = l.loadout_id
    JOIN lucidota_runtime.model_candidate m ON m.model_id = s.model_id
    WHERE l.active
    ORDER BY s.priority, s.slot_name;
    """
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()

    print("LUCIDOTA Model Registry")
    print("=======================")
    for loadout_id, slot, count, model, role, status, quant in rows:
        print(f"{loadout_id:<28} {slot:<24} x{count:<2} {model:<28} {role:<12} {status:<8} {quant}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
