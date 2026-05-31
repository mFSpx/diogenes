#!/usr/bin/env python3
"""
Diogenes Governor Loop — watches embedding queue depth and writes phase decisions
to lucidota_runtime.load_governor_decision.

PHASE_EMBED_CRUSH  : embedding_queue has rows pending  → local LLMs locked out
PHASE_ORCHESTRATOR_FREE : queue empty → DeepSeek/Mamba/Bonsai available to claw
"""
from __future__ import annotations
import os, sys, time, json, argparse
import psycopg2

STATE_DSN     = os.environ.get("LUCIDOTA_GO_STATE_DSN",   "postgresql:///lucidota_state")
STORAGE_DSN   = os.environ.get("LUCIDOTA_GO_STORAGE_DSN", "postgresql:///lucidota_storage")
POLL_SECS     = int(os.environ.get("LUCIDOTA_GOVERNOR_POLL_SECS", "15"))


def _embed_pending(storage_conn) -> int:
    try:
        cur = storage_conn.cursor()
        # Try embedding_queue first, fall back to corpus_chunk.embedded flag
        try:
            cur.execute("SELECT COUNT(*) FROM lucidota_korpus.embedding_queue")
        except Exception:
            storage_conn.rollback()
            cur.execute("SELECT COUNT(*) FROM lucidota_korpus.corpus_chunk WHERE embedded IS NOT TRUE")
        row = cur.fetchone()
        return int(row[0]) if row else 0
    except Exception:
        return -1  # unknown


def _write_decision(state_conn, phase: str, pending: int) -> None:
    detail = json.dumps({"phase": phase, "embedding_queue_depth": pending})
    state_conn.cursor().execute("""
        INSERT INTO lucidota_runtime.load_governor_decision
            (loadout_id, target_gpu, budget_vram_mb, observed_used_mb, observed_free_mb,
             estimated_required_mb, headroom_mb, decision, rationale, detail)
        VALUES
            ('embed_governor', 'GTX1650', 4096, 0, 4096, 0, 4096,
             %s, %s, %s::jsonb)
    """, (phase, f"embed_queue_depth={pending}", detail))
    state_conn.commit()


def run_loop(poll_secs: int, once: bool) -> None:
    state_conn   = psycopg2.connect(STATE_DSN)
    storage_conn = psycopg2.connect(STORAGE_DSN)
    state_conn.autocommit   = False
    storage_conn.autocommit = True

    last_phase = None
    while True:
        pending = _embed_pending(storage_conn)
        phase = "PHASE_EMBED_CRUSH" if pending > 0 else "PHASE_ORCHESTRATOR_FREE"

        if phase != last_phase:
            try:
                _write_decision(state_conn, phase, pending)
                print(f"[governor] phase={phase} queue_depth={pending}", flush=True)
                last_phase = phase
            except Exception as e:
                print(f"[governor] write error: {e}", file=sys.stderr, flush=True)
                try: state_conn.rollback()
                except Exception: pass

        if once:
            break
        time.sleep(poll_secs)


def main() -> None:
    ap = argparse.ArgumentParser(description="Diogenes embed-phase governor loop")
    ap.add_argument("--once",  action="store_true", help="Single tick then exit")
    ap.add_argument("--poll",  type=int, default=POLL_SECS, help="Poll interval in seconds")
    args = ap.parse_args()
    run_loop(args.poll, args.once)


if __name__ == "__main__":
    main()
