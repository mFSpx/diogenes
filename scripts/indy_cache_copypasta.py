#!/usr/bin/env python3
"""Generate deterministic cache copypasta from telemetry. No LLM. Pure SQL."""
import json, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def generate_cache_copypasta(conn) -> str:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT fact_value->>'mode', fact_value->'telemetry'->>'ram_avail_mb'
            FROM lucidota_control.runtime_status_fact
            WHERE subsystem='hardware' AND fact_key='hw_state'
        """)
        hw = cur.fetchone() or ('unknown', '?')

        cur.execute("""
            SELECT fact_value->>'recommended_batch_size', fact_value->>'n_samples'
            FROM lucidota_control.runtime_status_fact
            WHERE subsystem='river_governor' AND fact_key='batch_recommendation'
        """)
        river = cur.fetchone() or ('?', '0')

        cur.execute("""
            SELECT count(*) FILTER (WHERE status='COMPLETE'), count(*)
            FROM lucidota_control.phase_ledger
        """)
        phases = cur.fetchone() or (0, 0)

        cur.execute("""
            SELECT correction_text FROM lucidota_control.correction_observation
            ORDER BY observed_at DESC LIMIT 3
        """)
        corrections = [r[0][:60] for r in cur.fetchall()]

    parts = [
        f"HW:{hw[0]}/RAM:{hw[1]}MB",
        f"RIVER_BATCH:{river[0]}/n={river[1]}",
        f"PHASES:{phases[0]}/{phases[1]}",
    ] + [f"CORR: {c}" for c in corrections]

    return "CONTEXT_BLOCK | " + " | ".join(parts)


if __name__ == "__main__":
    from core.runtime_dsns import resolve_state_dsn
    import psycopg2
    conn = psycopg2.connect(resolve_state_dsn("postgresql://mfspx@/lucidota_state"))
    result = generate_cache_copypasta(conn)
    conn.close()
    print(result)
