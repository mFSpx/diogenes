#!/usr/bin/env python3
"""
diogenes_chat.py — DIOGENES operator chat surface.

Port of SANTA (Krampus Express) adapted for LUCIDOTA:
  - Groq HTTP (openai-compat) as primary model hand, not Anthropic SDK direct.
  - Manifest injected from lucidota_state + lucidota_storage facts (not Krampus schema).
  - Log path: 04_RUNTIME/diogenes_sessions/.
  - Persona: Indy_READs / Diogenes resident analyst.

Mutation class: read_only (DB reads) + external_effect (Groq API call).
Not a pipeline orchestrator. No canonical graph writes. No queue writes.

Usage:
    python3 scripts/diogenes_chat.py [--model MODEL] [--dry-run]
    echo "what is the current staging count?" | python3 scripts/diogenes_chat.py --once
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "04_RUNTIME" / "diogenes_sessions"

try:
    import psycopg2
    _HAS_PG = True
except ImportError:
    _HAS_PG = False

_STATE_DSN   = os.environ.get("LUCIDOTA_GO_STATE_DSN",   "postgresql:///lucidota_state")
_STORAGE_DSN = os.environ.get("LUCIDOTA_GO_STORAGE_DSN", "postgresql:///lucidota_storage")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = os.environ.get("DIOGENES_MODEL", "llama-3.1-8b-instant")
GROQ_HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "groq-python/0.28.0",
}


_SYSTEM_PROMPT_BASE = """
You are Diogenes, the Resident Analyst and operating mind of the LUCIDOTA intelligence system.
You are NOT a generative assistant — you are a deterministic investigator and builder.

Doctrine:
  - LUCIDOTA = GO-25 ternary graph + ABSURD Postgres queue + Groq hands.
  - Truth lives in the DB. Claims require evidence refs. Model output is not authority.
  - Graph: OBJECT/EVENT/EDGE → staging_packet → promotion gate → canonical graph_item.
  - Three logical stores: lucidota_state (workflow/control), lucidota_storage (graph/corpus).
  - Never fake a PASS. Never prose-complete a receipt. Always cite a fresh DB fact or file.

When asked about system state, refer to the SESSION MANIFEST below.
When asked to run something, tell the operator the exact command.
When asked to explain something, cite the relevant schema, script, or ALGOS file.
Never claim completion without a receipt path or live DB fact.
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _build_manifest() -> str:
    """Query live DB facts for the session manifest injected into system prompt."""
    if not _HAS_PG:
        return "[manifest unavailable: psycopg2 not installed]"

    parts = ["=== SESSION MANIFEST ===", f"Generated: {_now_iso()}"]

    try:
        conn = psycopg2.connect(_STORAGE_DSN, connect_timeout=3)
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM lucidota_korpus.corpus_chunk")
            cc = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM lucidota_go.staging_packet")
            sp = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM lucidota_go.staging_packet WHERE status='pending'")
            sp_pending = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM lucidota_go.graph_item")
            gi = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM lucidota_go.term_registry")
            tr = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM lucidota_go.percyphon_village")
            pv = cur.fetchone()[0]
        conn.close()
        parts.append(f"corpus_chunk: {cc}")
        parts.append(f"staging_packet: {sp} (pending: {sp_pending})")
        parts.append(f"graph_item: {gi}")
        parts.append(f"term_registry: {tr} (GO-25 + CO + IO ontology)")
        parts.append(f"percyphon_village: {pv} procedural scaffolds")
    except Exception as e:
        parts.append(f"[storage read error: {e}]")

    try:
        conn = psycopg2.connect(_STATE_DSN, connect_timeout=3)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT fact_key, fact_value FROM lucidota_control.runtime_status_fact "
                "WHERE subsystem IN ('governor','system') ORDER BY subsystem, fact_key"
            )
            for row in cur.fetchall():
                parts.append(f"runtime.{row[0]}: {row[1]}")
            cur.execute(
                "SELECT rung, action_key, ts FROM lucidota_control.governor_action "
                "ORDER BY ts DESC LIMIT 1"
            )
            row = cur.fetchone()
            if row:
                parts.append(f"last_governor_tick: rung={row[0]} action={row[1]} at {row[2]}")
        conn.close()
    except Exception as e:
        parts.append(f"[state read error: {e}]")

    return "\n".join(parts)


def _groq_call(messages: list[dict], model: str = GROQ_MODEL, max_tokens: int = 2048) -> str:
    """POST to Groq openai-compat endpoint. Returns reply text or error string."""
    if not GROQ_API_KEY:
        return "[ERROR: GROQ_API_KEY not set — source scripts/lucidota_safe_ops_env.sh]"
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }).encode("utf-8")
    headers = {**GROQ_HEADERS, "Authorization": f"Bearer {GROQ_API_KEY}"}
    req = urllib.request.Request(GROQ_URL, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")[:200]
        return f"[Groq HTTP error {e.code}: {body}]"
    except Exception as e:
        return f"[Groq call error: {e}]"


class DiogenesAgent:
    def __init__(self, model: str = GROQ_MODEL):
        self.model = model
        self.history: list[dict] = []
        self.session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.log_path = LOG_DIR / f"diogenes_{self.session_id}.jsonl"
        manifest = _build_manifest()
        self.system_prompt = _SYSTEM_PROMPT_BASE + "\n\n" + manifest
        print(f"[DIOGENES] Online. Session: {self.session_id} | Model: {self.model}")
        print(f"[DIOGENES] Manifest injected ({len(manifest)} chars). Resident Analyst mode.")

    def query(self, user_input: str) -> str:
        self.history.append({"role": "user", "content": user_input})
        messages = [{"role": "system", "content": self.system_prompt}] + self.history
        reply = _groq_call(messages, model=self.model)
        self.history.append({"role": "assistant", "content": reply})
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": _now_iso(), "q": user_input, "r": reply}) + "\n")
        return reply


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DIOGENES operator chat shell")
    parser.add_argument("--model", default=GROQ_MODEL, help="Groq model ID")
    parser.add_argument("--once", action="store_true", help="Read one query from stdin, reply, exit")
    parser.add_argument("--dry-run", action="store_true", help="Show manifest only; no Groq call")
    args = parser.parse_args()

    if args.dry_run:
        print(_build_manifest())
        sys.exit(0)

    agent = DiogenesAgent(model=args.model)

    if args.once:
        q = sys.stdin.read().strip()
        if q:
            print(agent.query(q))
        sys.exit(0)

    print("[DIOGENES] Type 'exit' to quit. Receipts → 04_RUNTIME/diogenes_sessions/\n")
    while True:
        try:
            q = input("operator > ").strip()
            if q.lower() in ("exit", "quit", "q"):
                break
            if not q:
                continue
            print(f"\n[DIOGENES]\n{agent.query(q)}\n")
        except (KeyboardInterrupt, EOFError):
            break
