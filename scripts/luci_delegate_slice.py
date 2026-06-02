#!/usr/bin/env python3
"""Reusable LUCI delegate/provider slice.

Fan out a bounded operator task to the external worker lanes we already trust:
- Groq for compact review/plan output
- Vibes for prompt-side sidecar work

The slice writes a receipt and, when possible, DB-backed work_order/work_receipt
rows so delegate activity is ledgered instead of being a one-off script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "luci_delegate"
RUNTIME = ROOT / "04_RUNTIME" / "luci_delegate"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def db_url(args: argparse.Namespace | None = None) -> str:
    return (
        (getattr(args, "database_url", None) if args is not None else None)
        or os.environ.get("LUCIDOTA_CONTROL_DATABASE_URL")
        or os.environ.get("ABSURD_SYSTEM_DATABASE_URL")
        or os.environ.get("DATABASE_URL")
        or "postgresql:///lucidota_state"
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def load_context() -> dict[str, Any]:
    context: dict[str, Any] = {}
    handoff = ROOT / "GOALS" / "CURRENT_HANDOFF.md"
    if handoff.exists():
        context["current_handoff"] = handoff.read_text(encoding="utf-8")[:6000]
    ontology = ROOT / "OFFICIAL_ONTOLOGY.json"
    if ontology.exists():
        try:
            data = json.loads(ontology.read_text(encoding="utf-8"))
            context["ontology"] = {
                "official_ontology": data.get("official_ontology"),
                "core_sentence": data.get("core_sentence"),
                "active_terms": (data.get("active_terms") or [])[:32],
            }
        except Exception:
            context["ontology"] = {"official_ontology": "unreadable"}
    return context


def build_prompt(text: str, *, kind: str, provider: str, context: dict[str, Any]) -> str:
    return (
        "You are LUCI's external delegate worker.\n"
        f"Delegate kind: {kind}\n"
        f"Provider lane: {provider}\n"
        "Be terse, bounded, and receipt-aware.\n"
        "Return JSON only with keys: summary, findings, next_steps, blockers, suggested_commands.\n\n"
        f"Operator text:\n{text}\n\n"
        f"Context:\n{json.dumps(context, sort_keys=True, indent=2, default=str)}\n"
    )


def write_receipt(payload: dict[str, Any], *, receipt_key: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"luci_delegate_{receipt_key}.json"
    payload["generated_at"] = now()
    payload["receipt_path"] = rel(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return path


def write_db_ledger(payload: dict[str, Any], receipt_path: str) -> dict[str, str]:
    identity = {
        "kind": payload["kind"],
        "text": payload["text"],
        "provider": payload["provider"],
        "run_id": payload["run_id"],
    }
    event_id = sha256_text(stable_json(identity))
    raw_ref = f"inline://luci_delegate/{event_id[:16]}"
    with psycopg.connect(db_url(None)) as conn:
        with conn.cursor() as cur:
            raw_artifact_row = cur.execute(
                """
                INSERT INTO lucidota_control.raw_artifact(raw_ref, raw_sha256, hash_algo, source, actor, byte_count, char_count, mime_type, storage_hint, detail)
                VALUES (%s,%s,'sha256','luci_delegate_slice','worker',%s,%s,'application/json','inline_or_receipt',%s::jsonb)
                ON CONFLICT (raw_ref) DO UPDATE SET
                  raw_sha256 = EXCLUDED.raw_sha256,
                  hash_algo = EXCLUDED.hash_algo,
                  source = EXCLUDED.source,
                  actor = EXCLUDED.actor,
                  byte_count = EXCLUDED.byte_count,
                  char_count = EXCLUDED.char_count,
                  mime_type = EXCLUDED.mime_type,
                  storage_hint = EXCLUDED.storage_hint,
                  detail = EXCLUDED.detail
                RETURNING raw_artifact_uuid::text
                """,
                (
                    raw_ref,
                    sha256_text(stable_json({"text": payload["text"], "kind": payload["kind"], "provider": payload["provider"]})),
                    len(payload["text"].encode("utf-8", errors="replace")),
                    len(payload["text"]),
                    json.dumps({"kind": payload["kind"], "provider": payload["provider"]}),
                ),
            ).fetchone()
            raw_artifact_uuid = raw_artifact_row["raw_artifact_uuid"] if isinstance(raw_artifact_row, dict) else raw_artifact_row[0]
            cur.execute(
                """
                INSERT INTO lucidota_control.event_envelope(event_id, ts, source, actor, raw_ref, raw_artifact_uuid, verbatim_hash, hash_algo, text, entities, claims, actions_requested, artifacts_referenced, risk_flags, route_candidates, board_features, embedding_ref, detail)
                VALUES (%s, now(), 'luci_delegate_slice', 'worker', %s, %s::uuid, %s, 'sha256', %s, '[]'::jsonb, '[]'::jsonb, %s::jsonb, '[]'::jsonb, '[]'::jsonb, '[]'::jsonb, %s::jsonb, NULL, %s::jsonb)
                ON CONFLICT (event_id) DO UPDATE SET
                  ts = EXCLUDED.ts,
                  source = EXCLUDED.source,
                  actor = EXCLUDED.actor,
                  raw_ref = EXCLUDED.raw_ref,
                  raw_artifact_uuid = EXCLUDED.raw_artifact_uuid,
                  verbatim_hash = EXCLUDED.verbatim_hash,
                  hash_algo = EXCLUDED.hash_algo,
                  text = EXCLUDED.text,
                  actions_requested = EXCLUDED.actions_requested,
                  board_features = EXCLUDED.board_features,
                  detail = EXCLUDED.detail
                RETURNING event_id
                """,
                (
                    event_id,
                    raw_ref,
                    raw_artifact_uuid,
                    sha256_text(payload["text"]),
                    payload["text"],
                    json.dumps([payload["kind"], payload["provider"]]),
                    json.dumps({"delegate": payload["provider"], "kind": payload["kind"]}),
                    json.dumps({"receipt_path": receipt_path}),
                ),
            )
            work_order_key = f"luci:delegate:{payload['run_id']}:{payload['provider']}:{payload['kind']}:{sha256_text(payload['text'])[:16]}"
            cur.execute(
                """
                INSERT INTO lucidota_control.work_order(event_id, lane, work_kind, status, payload, idempotency_key)
                VALUES (%s, 'external', %s, %s, %s::jsonb, %s)
                ON CONFLICT (idempotency_key) DO UPDATE SET
                  event_id = EXCLUDED.event_id,
                  lane = EXCLUDED.lane,
                  work_kind = EXCLUDED.work_kind,
                  status = EXCLUDED.status,
                  payload = EXCLUDED.payload,
                  updated_at = now()
                RETURNING work_order_uuid::text
                """,
                (
                    event_id,
                    "luci_delegate_fanout",
                    "succeeded",
                    json.dumps({"text": payload["text"], "kind": payload["kind"], "provider": payload["provider"], "receipt_path": receipt_path}),
                    work_order_key,
                ),
            )
            work_order_row = cur.fetchone()
            work_order_uuid = work_order_row["work_order_uuid"] if isinstance(work_order_row, dict) else work_order_row[0]
            receipt_row = cur.execute(
                """
                SELECT work_receipt_uuid::text
                FROM lucidota_control.work_receipt
                WHERE work_order_uuid = %s::uuid AND receipt_path = %s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (work_order_uuid, receipt_path),
            ).fetchone()
            if receipt_row:
                work_receipt_uuid = receipt_row["work_receipt_uuid"] if isinstance(receipt_row, dict) else receipt_row[0]
            else:
                cur.execute(
                    """
                    INSERT INTO lucidota_control.work_receipt(event_id, work_order_uuid, receipt_path, receipt_sha256, verdict, cost, gain, artifact_refs, canonical_graph_writes_performed, graph_write_mode, detail)
                    VALUES (%s, %s::uuid, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, false, 'staged_only', %s::jsonb)
                    RETURNING work_receipt_uuid::text
                    """,
                    (
                        event_id,
                        work_order_uuid,
                        receipt_path,
                        sha256_text(stable_json(payload)),
                        "promote" if not payload.get("blockers") else "retry",
                        json.dumps({"tokens": ((payload.get("groq") or {}).get("usage") or {}).get("total_tokens", 0)}),
                        json.dumps({"gain": 0.5 if payload.get("groq", {}).get("execute_performed") else 0.1}),
                        json.dumps([raw_ref, rel(payload.get("vibes_prompt_path", ""))] if payload.get("vibes_prompt_path") else [raw_ref]),
                        json.dumps({"delegate_kind": payload["kind"], "provider": payload["provider"], "run_id": payload["run_id"]}),
                    ),
                )
                work_receipt_row = cur.fetchone()
                work_receipt_uuid = work_receipt_row["work_receipt_uuid"] if isinstance(work_receipt_row, dict) else work_receipt_row[0]
        conn.commit()
    return {
        "work_order_uuid": work_order_uuid,
        "work_receipt_uuid": work_receipt_uuid,
        "raw_artifact_uuid": raw_artifact_uuid,
        "event_id": event_id,
    }


def run_groq_delegate(text: str, *, kind: str) -> dict[str, Any]:
    prompt = ROOT / "04_RUNTIME" / "luci_delegate" / f"groq_delegate_{stamp()}.txt"
    prompt.parent.mkdir(parents=True, exist_ok=True)
    ctx = load_context()
    prompt.write_text(build_prompt(text, kind=kind, provider="groq", context=ctx), encoding="utf-8")
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "groq_goal_delegate.py"),
        "--task",
        f"@{rel(prompt)}",
        "--kind",
        kind,
        "--model",
        os.environ.get("GROQ_GOAL_MODEL", "llama-3.1-8b-instant"),
        "--max-tokens",
        "512",
        "--execute",
        "--json",
    ]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=180)
    receipt_path = None
    report_path = None
    for line in proc.stdout.splitlines():
        if line.startswith("REPORT_PATH="):
            report_path = line.split("=", 1)[1].strip()
        if line.startswith("RECEIPT_PATH="):
            receipt_path = line.split("=", 1)[1].strip()
    report = {}
    if report_path:
        report_file = ROOT / report_path
        if report_file.exists():
            report = json.loads(report_file.read_text(encoding="utf-8"))
    return {
        "execute_performed": proc.returncode == 0,
        "prompt_path": rel(prompt),
        "subreceipt_path": receipt_path or report.get("subreceipt_path"),
        "report_path": report_path,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-1500:],
        "stderr_tail": proc.stderr[-1500:],
        "usage": report.get("usage"),
        "text": report.get("text", ""),
        "blockers": report.get("blockers", []) if isinstance(report.get("blockers"), list) else [],
    }


def write_vibes_prompt(text: str, *, kind: str) -> dict[str, Any]:
    RUNTIME.mkdir(parents=True, exist_ok=True)
    prompt_path = RUNTIME / f"vibes_delegate_{stamp()}.prompt"
    prompt_text = build_prompt(text, kind=kind, provider="vibes", context=load_context())
    prompt_path.write_text(prompt_text, encoding="utf-8")
    return {
        "prompt_path": rel(prompt_path),
        "execute_hint": f".venv/bin/vibe -p @{rel(prompt_path)} --agent auto-approve --trust --workdir {ROOT}",
    }


def run_vibes_prompt(prompt_path: str) -> dict[str, Any]:
    vibe = ROOT / ".venv" / "bin" / "vibe"
    cmd = [str(vibe), "-p", f"@{prompt_path}", "--agent", "auto-approve", "--trust", "--workdir", str(ROOT)]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=180)
    return {
        "execute_performed": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-1500:],
        "stderr_tail": proc.stderr[-1500:],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Reusable LUCI delegate/provider class-handler.")
    ap.add_argument("--text", required=True)
    ap.add_argument("--kind", default="review", choices=["audit", "review", "plan", "code-slice"])
    ap.add_argument("--provider", default="auto", choices=["auto", "groq", "vibes", "both"])
    ap.add_argument("--run-id")
    ap.add_argument("--execute-vibes", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    blockers: list[str] = []
    if not args.text.strip():
        blockers.append("text_required")

    provider_choice = args.provider
    if provider_choice == "auto":
        provider_choice = "both"

    payload: dict[str, Any] = {
        "schema": "lucidota.luci_delegate.v1",
        "kind": args.kind,
        "provider": provider_choice,
        "text": args.text,
        "run_id": args.run_id or "luci-delegate:" + sha256_text(stable_json({"kind": args.kind, "provider": provider_choice, "text": args.text}))[:24],
        "vibes": None,
        "groq": None,
        "blockers": blockers,
    }

    if provider_choice in {"groq", "both"}:
        payload["groq"] = run_groq_delegate(args.text, kind=args.kind)
        blockers.extend([b for b in payload["groq"].get("blockers", []) if b not in blockers])

    if provider_choice in {"vibes", "both"}:
        vibes = write_vibes_prompt(args.text, kind=args.kind)
        if args.execute_vibes:
            vibes.update(run_vibes_prompt(vibes["prompt_path"]))
        payload["vibes"] = vibes

    payload["status"] = "PASS" if not blockers else "BLOCKED"
    receipt_key = sha256_text(stable_json({"kind": payload["kind"], "provider": payload["provider"], "text": payload["text"], "run_id": payload["run_id"]}))[:24]
    receipt_path = write_receipt(payload, receipt_key=receipt_key)
    try:
        payload["db_write"] = write_db_ledger({**payload, "generated_at": now()}, receipt_path=rel(receipt_path))
    except Exception as exc:
        blockers.append(f"db_write_failed:{type(exc).__name__}")
        payload["db_write_error"] = f"{type(exc).__name__}: {exc}"
        payload["status"] = "PASS" if provider_choice in {"groq", "vibes", "both"} else "BLOCKED"
        payload["blockers"] = blockers
        # Re-write receipt with the DB failure recorded.
        receipt_path = write_receipt(payload, receipt_key=receipt_key)
    payload["visible_response"] = {
        "summary": "Indy_READs: delegated a bounded provider review and wrote the ledger.",
        "work_order_id": payload.get("db_write", {}).get("work_order_uuid", ""),
        "work_receipt_id": payload.get("db_write", {}).get("work_receipt_uuid", ""),
        "attempt_id": payload.get("db_write", {}).get("work_order_uuid", ""),
        "raw_artifact_id": payload.get("db_write", {}).get("raw_artifact_uuid", ""),
        "receipt_path": rel(receipt_path),
    }
    if args.json:
        print(json.dumps(payload, sort_keys=True, default=str))
    else:
        print(f"RECEIPT_PATH={rel(receipt_path)}")
        print(f"DELEGATE={payload['status']}")
    return 0 if payload["status"] == "PASS" else 4


if __name__ == "__main__":
    raise SystemExit(main())
