#!/usr/bin/env python3
"""diogenes_routing_loop.py — ABSURD staging_packet enrichment worker.

Mutation class: candidate_writer
- Reads from lucidota_storage.lucidota_go.staging_packet (FOR UPDATE SKIP LOCKED)
- Updates staging_packet.status to 'processed' (NOT canonical graph)
- Writes runtime_status_fact to lucidota_state
- Writes receipts to 05_OUTPUTS/diogenes/
- Calls Percyphon scaffold, comms filter, bandit router, Needle, Forge
"""
import os
import sys
import json
import time
import logging
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import psycopg2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ALGOS import percyphon, percyphon_comms_filter, bandit_router
from scripts.absurd_worker_contracts import validate_worker_contract

logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

STORAGE_DSN = os.environ.get("LUCIDOTA_GO_STORAGE_DSN", "postgresql:///lucidota_storage")
STATE_DSN = os.environ.get("LUCIDOTA_GO_STATE_DSN", "postgresql:///lucidota_state")
OUTPUT_DIR = ROOT / "05_OUTPUTS" / "diogenes"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GO25_VALID = {
    "ENTITY", "GROUP", "EVENT", "EVIDENCE", "CLAIM", "SOURCE",
    "GRIP", "SNARE", "LIE", "GHOST", "WITNESS",
}


def dequeue_staging_packet(conn) -> dict | None:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT uuid, claim, confidence_bps, proposed_term
            FROM lucidota_go.staging_packet
            WHERE status = 'pending'
            ORDER BY created_at
            LIMIT 1
            FOR UPDATE SKIP LOCKED
        """)
        row = cur.fetchone()
    if not row:
        return None
    return {"uuid": str(row[0]), "claim": row[1], "confidence_bps": row[2], "proposed_term": row[3]}


def call_needle(claim_text: str) -> str | None:
    try:
        req = urllib.request.Request(
            "http://127.0.0.1:8090/",
            json.dumps({"input": claim_text[:500]}).encode(),
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req, timeout=5)
        if resp.getcode() == 200:
            return resp.read().decode()
    except Exception as e:
        logging.warning("Needle unavailable: %s", e)
    return None


def call_forge(claim: str) -> str | None:
    """Call Forge (local Falcon/Mamba) for GO-25 term classification. Returns valid term or None."""
    try:
        req = urllib.request.Request(
            "http://127.0.0.1:9000/v1/chat/completions",
            json.dumps({
                "model": "Falcon3-Mamba-7B-Instruct-Q2_K.gguf",
                "system": (
                    "Extract the primary GO-25 entity term from this text. "
                    "Reply with ONLY one term from: ENTITY GROUP EVENT EVIDENCE CLAIM SOURCE GRIP SNARE LIE GHOST WITNESS"
                ),
                "user": claim[:300],
            }).encode(),
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req, timeout=10)
        if resp.getcode() == 200:
            data = json.loads(resp.read().decode())
            # Forge returns either {"term": "ENTITY"} or OpenAI chat format
            if isinstance(data, dict):
                term = (
                    data.get("term")
                    or (data.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()
                    or ""
                )
                term = term.strip().upper()
                if term in GO25_VALID:
                    return term
    except Exception as e:
        logging.warning("Forge unavailable: %s", e)
    return None


def process_packet(packet: dict, storage_conn, state_conn) -> dict:
    """Enrich one staging packet. Returns detail dict. Raises on unrecoverable error."""
    claim = packet["claim"]

    scaffold = percyphon.procedural_entity_generator([claim], fluid_slots=12)
    comms_result = percyphon_comms_filter.classify_comms_pattern(claim)
    context_vector = {
        "confidence_bps": packet["confidence_bps"] / 150.0,
        "comms_signal": 1.0 if comms_result["confidence_bps"] > 0 else 0.0,
    }
    action = bandit_router.select_action(
        context_vector, actions=list(GO25_VALID), algorithm="linucb"
    )
    needle_response = call_needle(claim)
    forge_term = call_forge(claim)

    proposed_term = forge_term if (forge_term and forge_term != packet["proposed_term"]) else packet["proposed_term"]

    detail = {
        "percyphon_scaffold": scaffold,
        "comms_pattern": comms_result,
        "bandit_choice": action,
        "needle_response": needle_response,
        "forge_term": forge_term,
    }

    ts = datetime.now(timezone.utc).isoformat()

    # Update staging_packet (candidate_writer — NOT canonical graph)
    with storage_conn.cursor() as cur:
        cur.execute("""
            UPDATE lucidota_go.staging_packet
            SET status = 'processed', proposed_term = %s, detail = %s
            WHERE uuid = %s
        """, (proposed_term, json.dumps(detail), packet["uuid"]))
    storage_conn.commit()

    # Write runtime_status_fact to STATE db
    with state_conn.cursor() as cur:
        cur.execute("""
            INSERT INTO lucidota_control.runtime_status_fact (subsystem, fact_key, fact_value, derived_at)
            VALUES ('diogenes_loop', 'last_tick', %s::jsonb, now())
            ON CONFLICT (subsystem, fact_key) DO UPDATE
              SET fact_value = EXCLUDED.fact_value, derived_at = now()
        """, (json.dumps({"packet_uuid": packet["uuid"], "ts": ts}),))
    state_conn.commit()

    # File receipt
    receipt_path = OUTPUT_DIR / f"diogenes_{packet['uuid']}.json"
    receipt_path.write_text(json.dumps({
        "schema": "lucidota.diogenes.routing_loop.v2",
        "packet_uuid": packet["uuid"],
        "ts": ts,
        "proposed_term": proposed_term,
        "detail": detail,
        "mutation_class": "candidate_writer",
    }, indent=2))

    return detail


def main() -> None:
    storage_conn = psycopg2.connect(STORAGE_DSN)
    state_conn = psycopg2.connect(STATE_DSN)

    # Validate worker contract before processing
    with storage_conn.cursor() as cur:
        decision = validate_worker_contract(
            cur, queue_name="diogenes", job_kind="staging_packet_enrich", worker_key=None
        )
    if decision.blocked:
        logging.error("Worker contract blocked: %s", decision.reason)
        storage_conn.close()
        state_conn.close()
        return

    logging.info("diogenes_routing_loop started (contract validated)")

    try:
        while True:
            try:
                packet = dequeue_staging_packet(storage_conn)
                if packet:
                    process_packet(packet, storage_conn, state_conn)
                    logging.info("processed %s → %s", packet["uuid"][:8], packet["proposed_term"])
            except Exception as e:
                logging.error("tick error: %s", e)
                try:
                    storage_conn.rollback()
                    state_conn.rollback()
                except Exception:
                    pass
            time.sleep(0.1)
    finally:
        storage_conn.close()
        state_conn.close()


if __name__ == "__main__":
    main()
