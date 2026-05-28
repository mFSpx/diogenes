#!/usr/bin/env python3
"""Local graph-first runtime for the hard-modded Claw CLI.

This is intentionally small:
- no Anthropic/OpenAI/xAI keys
- route the user message through CKDOG1's baked-in deterministic embedding path
- reduce concepts to GO-25 terms
- write the request to the GO graph and the ABSURD/control event surface
- print a compact response that is derived from the graph payload, not a prompt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import psycopg

ROOT = Path(__file__).resolve().parents[1]
GO_SCHEMA = ROOT / "06_SCHEMA" / "016_go_graph_core.sql"
PROMOTION_SCHEMA = ROOT / "06_SCHEMA" / "034_graph_promotion_pipeline.sql"
CONTROL_SCHEMA = ROOT / "06_SCHEMA" / "001_lucidota_control.sql"
WORKFLOW_SCHEMA = ROOT / "06_SCHEMA" / "006_workflow_registry.sql"
TERMS_PATH = ROOT / "BOOKS" / "GO_ACTIVE_TERMS.json"
STORAGE_DSN = os.environ.get("LUCIDOTA_GO_STORAGE_DSN", "postgresql:///lucidota_storage")
STATE_DSN = os.environ.get("LUCIDOTA_GO_STATE_DSN", os.environ.get("ABSURD_SYSTEM_DATABASE_URL", "postgresql:///lucidota_state"))
OPERATOR_UUID = os.environ.get("LUCIDOTA_OPERATOR_UUID", "00000000-0000-4000-8000-000000000414")
INDY_SUUID = int(os.environ.get("LUCIDOTA_INDY_READS_SUUID", "414"))
EMBED_MODE = "ckdog1.kernel.hash_quantized_embedding.v1"
GRAPH_APPROVAL_MODE = (
    "approved"
    if os.environ.get("LUCIDOTA_GRAPH_APPROVAL_MODE", "").strip().lower() == "approved"
    and os.environ.get("LUCIDOTA_ALLOW_DIRECT_GRAPH_APPROVAL", "").strip().lower() in {"1", "true", "yes", "on"}
    else "staged"
)

sys.path.insert(0, str(ROOT / "01_REPOS" / "doggystyle"))
try:
    from kernel.mini_embeddings import dot_i64, hash_quantized_embedding  # type: ignore
except Exception:  # pragma: no cover - only used if the kernel package is broken
    import hashlib

    def hash_quantized_embedding(text: str, *, dim: int = 384) -> tuple[int, ...]:
        out: list[int] = []
        counter = 0
        while len(out) < dim:
            digest = hashlib.blake2b(f"{counter}\0{text}".encode(), digest_size=64).digest()
            for i in range(0, len(digest), 2):
                out.append(int.from_bytes(digest[i : i + 2], "big") - 32768)
                if len(out) == dim:
                    break
            counter += 1
        return tuple(out)

    def dot_i64(left: tuple[int, ...], right: tuple[int, ...]) -> int:
        return sum(a * b for a, b in zip(left, right, strict=True))


GO_KEYWORDS: dict[str, tuple[str, ...]] = {
    "ENTITY": ("person", "system", "book", "model", "agent", "persona", "node", "thing", "who"),
    "ATTRIBUTE": ("property", "metadata", "quality", "field", "feature", "status"),
    "RELATIONSHIP": ("link", "edge", "between", "relation", "connect", "with"),
    "FRICTION": ("issue", "bug", "risk", "block", "stuck", "problem", "fail"),
    "LEVERAGE": ("priority", "force", "capacity", "pressure", "advantage", "routing"),
    "VISIBILITY": ("report", "show", "print", "display", "public", "visible"),
    "ACTION": ("run", "make", "build", "fix", "do", "need", "implement", "work", "train"),
    "EVENT": ("happens", "dropped", "started", "finished", "trigger", "automatic"),
    "TIME": ("now", "today", "when", "schedule", "automatic", "watch", "loop"),
    "PATTERN": ("logic", "understanding", "pattern", "structure", "core"),
    "HYPOTHESIS": ("maybe", "guess", "model", "theory", "hunch"),
    "CLAIM": ("claim", "supposed", "should", "must", "is", "be"),
    "EVIDENCE": ("source", "chunk", "read", "citation", "proof", "book", "page"),
    "ATOMIC_ID": ("uuid", "id", "hash", "sha", "identifier"),
    "SIGNAL": ("message", "request", "input", "signal", "picked", "ingest"),
    "GLOW": ("persona", "indy", "salience", "voice", "glow"),
    "TERM": ("ontology", "go", "word", "term", "primitive"),
    "TOOL": ("cli", "claw", "clawd", "script", "adapter", "cartridge", "watcher"),
    "ALGORITHM": ("embedding", "lora", "chunk", "train", "router", "math"),
    "NAUGHTY": ("bad", "broken", "not listening", "hard", "fail"),
    "NICE": ("neat", "nice", "works", "ok", "ready"),
    "GROUP": ("agents", "suite", "swarm", "models"),
    "OPERATOR": ("operator", "route", "absurd", "workflow"),
    "MODE": ("mode", "local", "stateless", "deterministic", "kernel"),
    "COMMENT": ("note", "comment", "message", "reply", "report"),
}


def approx_tokens(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def load_go25() -> list[dict[str, str]]:
    data = json.loads(TERMS_PATH.read_text(encoding="utf-8"))
    return [
        {"id": str(t["id"]), "term": str(t["term"]), "definition": str(t.get("definition", ""))}
        for t in data.get("terms", [])
        if str(t.get("id", "")).startswith("@") and int(str(t["id"])[1:]) <= 25
    ]


def tokenize_for_route(text: str) -> set[str]:
    return {t.lower() for t in re.findall(r"[A-Za-z0-9_]+", text)}


def route_go25(text: str, top_n: int = 7) -> list[dict[str, Any]]:
    terms = load_go25()
    text_vec = hash_quantized_embedding(text)
    words = tokenize_for_route(text)
    ranked: list[dict[str, Any]] = []
    for term in terms:
        name = term["term"]
        definition = term["definition"]
        anchor = f"{name} {definition}"
        anchor_words = tokenize_for_route(anchor)
        direct = 1 if name.lower() in words else 0
        overlap = len(words & {w for w in anchor_words if len(w) > 3})
        keyword_hits = sum(1 for kw in GO_KEYWORDS.get(name, ()) if kw in words or kw.lower() in text.lower())
        lexical_score = direct * 5000 + overlap * 80 + keyword_hits * 700
        embedding_score = dot_i64(text_vec, hash_quantized_embedding(anchor))
        ranked.append(
            {
                "id": term["id"],
                "term": name,
                "definition": definition,
                "lexical_score": lexical_score,
                "embedding_score": embedding_score,
                "score": lexical_score * 1_000_000_000_000 + embedding_score,
            }
        )
    ranked.sort(key=lambda r: (r["score"], r["lexical_score"], r["embedding_score"]), reverse=True)
    return ranked[:top_n]


def ensure_storage(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(GO_SCHEMA.read_text(encoding="utf-8"))
        cur.execute(
            """
            INSERT INTO lucidota_go.soul_registry(suuid, soul_kind, label, status, detail)
            VALUES (%s, 'VILLAGER_SOUL', 'INDY_READs', 'active', %s::jsonb)
            ON CONFLICT(suuid) DO UPDATE SET
              soul_kind=EXCLUDED.soul_kind,
              label=EXCLUDED.label,
              status=EXCLUDED.status,
              detail=EXCLUDED.detail,
              updated_at=now()
            """,
            (INDY_SUUID, json.dumps({"persona": "main_ai", "pronouns": "she/her", "role": "reader_on_graph"}, sort_keys=True)),
        )
    conn.commit()


def ensure_state(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(CONTROL_SCHEMA.read_text(encoding="utf-8"))
        cur.execute(WORKFLOW_SCHEMA.read_text(encoding="utf-8"))
    conn.commit()


def graph_direct_approved() -> bool:
    return GRAPH_APPROVAL_MODE == "approved"


def insert_graph_request(text: str, routes: list[dict[str, Any]], meta: dict[str, Any]) -> dict[str, Any]:
    """Stage a graph-promotion packet; never write canonical graph tables."""
    primary = routes[0]["term"] if routes else "COMMENT"
    packet_uuid = str(uuid.uuid4())
    location = f"clawd://promotion-packet/{packet_uuid}"
    evidence_refs = [
        {
            "kind": "operator_message",
            "location": location,
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "route_terms": [r["term"] for r in routes[:7]],
        }
    ]
    payload = {
        "kind": "clawd_user_message",
        "raw_message": text,
        "route_terms": routes,
        "embedding_mode": EMBED_MODE,
        "source": "hard_modded_clawd",
        "canonical_mutation_allowed": False,
        "graph_write_mode": "promotion_packet_only",
        "approval_required": True,
        "meta": meta,
    }
    detail = {
        "primary_term": primary,
        "go_route": [r["term"] for r in routes[:7]],
        "location": location,
        "absurd_boundary": "promotion_packet_only",
        "direct_canonical_graph_write": False,
    }
    with psycopg.connect(STORAGE_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(PROMOTION_SCHEMA.read_text(encoding="utf-8"))
            cur.execute(
                """
                INSERT INTO lucidota_go.graph_promotion_packet(
                  packet_uuid, source_system, candidate_kind, candidate_payload,
                  evidence_refs, authority_class, promotion_status, detail
                ) VALUES (%s,'luci_clawd_absurd_runtime','node',%s::jsonb,%s::jsonb,
                          'operator_authored_assertion','candidate',%s::jsonb)
                RETURNING packet_uuid::text
                """,
                (packet_uuid, json.dumps(payload, ensure_ascii=False, sort_keys=True), json.dumps(evidence_refs, sort_keys=True), json.dumps(detail, sort_keys=True)),
            )
            packet_uuid = cur.fetchone()[0]
        conn.commit()
    return {"item_uuid": packet_uuid, "promotion_packet_uuid": packet_uuid, "primary_term": primary, "location": location, "payload": payload}

def deterministic_model_lanes(routes: list[dict[str, Any]], graph_item_uuid: str) -> list[dict[str, Any]]:
    terms = [str(r["term"]) for r in routes]
    lanes = [
        {
            "lane": "mamba_watch",
            "model_role": "listener_manager",
            "target_model_id": "mamba-7b-ternary-ram-watch",
            "payload": {"graph_item_uuid": graph_item_uuid, "go_terms": terms[:7]},
        }
    ]
    if any(t in terms for t in ("ACTION", "ALGORITHM", "HYPOTHESIS", "CLAIM", "PATTERN")):
        lanes.append(
            {
                "lane": "deepseek_think",
                "model_role": "deep_reasoner",
                "target_model_id": "deepseek-r1-qwen-distill-1.5b-gpu",
                "payload": {"graph_item_uuid": graph_item_uuid, "go_terms": terms[:7], "instruction": "reason_over_go_payload_only"},
            }
        )
    lanes.append(
        {
            "lane": "needle_report",
            "model_role": "fast_token_reporter",
            "target_model_id": "needle-26m",
            "payload": {"graph_item_uuid": graph_item_uuid, "go_terms": terms[:5], "instruction": "stream_status_tokens_only"},
        }
    )
    return lanes


def insert_workflow_event(graph_result: dict[str, Any], routes: list[dict[str, Any]]) -> dict[str, Any]:
    model_lanes = deterministic_model_lanes(routes, graph_result["item_uuid"])
    detail = {
        "graph_item_uuid": graph_result["item_uuid"],
        "primary_term": graph_result["primary_term"],
        "go_route": [r["term"] for r in routes[:7]],
        "absurd_surface": "workflow_event",
        "model_lanes": model_lanes,
        "payload_rule": "model lanes receive GO terms + graph refs only; raw message stays on graph",
        "next": "route/dispatch/report",
    }
    with psycopg.connect(STATE_DSN) as conn:
        ensure_state(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO lucidota_control.workflow_event(workflow_id, run_id, phase, status, source, detail)
                VALUES ('clawd-go-runtime', %s, 'route', 'succeeded', 'hard_modded_clawd', %s::jsonb)
                RETURNING event_id::text
                """,
                (graph_result["item_uuid"], json.dumps(detail, sort_keys=True)),
            )
            event_id = cur.fetchone()[0]
        conn.commit()
    return {"event_id": event_id, "detail": detail}


def indy_counts() -> dict[str, int]:
    counts = {"books": 0, "chunks": 0, "lora_cartridges": 0}
    try:
        with psycopg.connect(STORAGE_DSN) as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM information_schema.tables WHERE table_schema='lucidota_indy' AND table_name='book_source'")
            if cur.fetchone()[0] == 0:
                return counts
            cur.execute("SELECT count(*) FROM lucidota_indy.book_source")
            counts["books"] = int(cur.fetchone()[0])
            cur.execute("SELECT count(*) FROM lucidota_indy.book_chunk")
            counts["chunks"] = int(cur.fetchone()[0])
    except Exception as exc:
        print(f"warning: indy storage count failed: {exc}", file=sys.stderr)
    try:
        with psycopg.connect(STATE_DSN) as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM information_schema.tables WHERE table_schema='lucidota_runtime' AND table_name='adapter_cartridge'")
            if cur.fetchone()[0]:
                cur.execute("SELECT count(*) FROM lucidota_runtime.adapter_cartridge WHERE adapter_id LIKE 'indy_reads__%'")
                counts["lora_cartridges"] = int(cur.fetchone()[0])
    except Exception as exc:
        print(f"warning: indy runtime count failed: {exc}", file=sys.stderr)
    return counts


def extract_message(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("message"), str) and payload["message"].strip():
        return payload["message"]
    messages = payload.get("messages") or []
    for msg in reversed(messages if isinstance(messages, list) else []):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") not in (None, "user", "system"):
            continue
        text = str(msg.get("text") or msg.get("content") or "").strip()
        if text:
            return text
    return ""


def render_response(graph_result: dict[str, Any] | None, workflow: dict[str, Any] | None, routes: list[dict[str, Any]], text: str, error: str = "") -> str:
    route_line = " > ".join(f"{r['id']} {r['term']}" for r in routes[:5]) or "@25 COMMENT"
    counts = indy_counts()
    if graph_result and workflow:
        lane_line = ",".join(l["lane"] for l in workflow["detail"].get("model_lanes", []))
        return (
            "DIOGENES GO_ROUTE_OK\n"
            f"- promotion_packet: {graph_result['item_uuid']}\n"
            f"- GO: {route_line}\n"
            f"- ABSURD: workflow_event {workflow['event_id']}\n"
            f"- lanes: {lane_line}\n"
            f"- Indy_READs: persona=active books={counts['books']} chunks={counts['chunks']} lora_cartridges={counts['lora_cartridges']}\n"
            "- auth: external Anthropic keys not used\n"
            f"- next: {routes[0]['term'] if routes else 'COMMENT'} payload staged for graph promotion"
        )
    return (
        "DIOGENES LOCAL_ROUTE_DEGRADED\n"
        f"- GO: {route_line}\n"
        f"- message_tokens: {len(approx_tokens(text))}\n"
        f"- graph_write_error: {error[:240] if error else 'unknown'}\n"
        "- auth: external Anthropic keys not used"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--message", default="")
    args = ap.parse_args()
    payload: dict[str, Any] = {}
    raw = sys.stdin.read().strip()
    if raw:
        payload = json.loads(raw)
    if args.message:
        payload["message"] = args.message

    message = extract_message(payload).strip() or "STATUS"
    routes = route_go25(message)
    graph_result: dict[str, Any] | None = None
    workflow: dict[str, Any] | None = None
    error = ""
    started = time.time()
    try:
        graph_result = insert_graph_request(
            message,
            routes,
            {"cwd": payload.get("cwd"), "model": payload.get("model"), "token_budget": len(approx_tokens(message))},
        )
        workflow = insert_workflow_event(graph_result, routes)
    except Exception as exc:
        error = str(exc)

    text = render_response(graph_result, workflow, routes, message, error)
    out = {
        "ok": bool(graph_result and workflow),
        "text": text,
        "routes": routes,
        "graph": graph_result,
        "workflow": workflow,
        "error": error,
        "input_tokens": len(approx_tokens(message)),
        "output_tokens": len(approx_tokens(text)),
        "elapsed_ms": int((time.time() - started) * 1000),
    }
    print(json.dumps(out, ensure_ascii=False, sort_keys=True) if args.json else text)
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
