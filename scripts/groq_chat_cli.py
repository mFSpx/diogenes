#!/usr/bin/env python3
"""Groq Chat Completions bridge with receipts and dry-run default.

Uses Groq's OpenAI-compatible Chat Completions endpoint. API keys are read
from environment only and are never printed or written to receipts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import uuid
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import psycopg
from psycopg.rows import dict_row
from groq_env import load_groq_env
from model_invocation_trace import build_generation_trace, spawn_generation_event_bridge
from project2501_admin_prompt import compose_system_prompt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "model_invocations"
DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "llama-3.1-8b-instant"
load_groq_env()


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def read_text_arg(value: str | None) -> str:
    if not value:
        return ""
    if value.startswith("@"):
        path = Path(value[1:])
        if not path.is_absolute():
            path = ROOT / path
        return path.read_text(encoding="utf-8")
    return value


def api_key(env_names: list[str]) -> tuple[str | None, str | None]:
    for name in env_names:
        value = os.environ.get(name)
        if value:
            return value, name
    return None, None


def build_messages(system: str, prompt: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system.strip():
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages


def message_receipts(messages: list[dict[str, str]], *, log_prompts: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content", "")
        row: dict[str, Any] = {
            "role": msg.get("role", ""),
            "content_chars": len(content),
            "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        }
        if log_prompts:
            row["content_text"] = content
        rows.append(row)
    return rows


def groq_text(response: dict[str, Any]) -> str:
    try:
        message = response["choices"][0]["message"]
        for key in ("content", "reasoning", "reasoning_content"):
            value = message.get(key)
            if isinstance(value, str) and value:
                return value
        return ""
    except Exception:
        return ""


def write_receipt(payload: dict[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"groq_chat_{payload['mode']}_{payload['receipt_key']}.json"
    payload["generated_at"] = now()
    payload["report_path"] = rel(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
    try:
        spawn_generation_event_bridge(path, root=ROOT)
    except Exception:
        pass
    return path


def write_db_ledger(payload: dict[str, Any], receipt_path: str) -> dict[str, str]:
    identity = {
        "provider": payload["provider"],
        "model": payload["model"],
        "prompt_hash": payload["prompt_hash"],
        "system_hash": payload["system_hash"],
        "mode": payload["mode"],
        "run_id": payload["run_id"],
    }
    event_id = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    model_invocation_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, json.dumps(identity, sort_keys=True, separators=(",", ":"))))
    raw_ref = f"inline://groq_chat/{event_id[:16]}"
    work_order_key = f"groq-chat:{payload['run_id']}:{payload['provider']}:{payload['model']}:{payload['mode']}:{payload['prompt_hash']}:{payload['system_hash']}"
    with psycopg.connect(os.environ.get("LUCIDOTA_CONTROL_DATABASE_URL") or os.environ.get("ABSURD_SYSTEM_DATABASE_URL") or os.environ.get("DATABASE_URL") or "postgresql:///lucidota_state", row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            raw_row = cur.execute(
                """
                INSERT INTO lucidota_control.raw_artifact(raw_ref, raw_sha256, hash_algo, source, actor, byte_count, char_count, mime_type, storage_hint, detail)
                VALUES (%s, %s, 'sha256', 'groq_chat_cli', 'groq', %s, %s, 'application/json', 'inline_or_receipt', %s::jsonb)
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
                    hashlib.sha256(json.dumps({"prompt": payload["prompt_text"], "system": payload["system_text"]}, sort_keys=True).encode("utf-8")).hexdigest(),
                    len((payload["prompt_text"] + "\n" + payload["system_text"]).encode("utf-8")),
                    len(payload["prompt_text"] + "\n" + payload["system_text"]),
                    json.dumps({"provider": payload["provider"], "model": payload["model"], "run_id": payload["run_id"]}),
                ),
            ).fetchone()
            raw_artifact_uuid = raw_row["raw_artifact_uuid"] if isinstance(raw_row, dict) else raw_row[0]
            cur.execute(
                """
                INSERT INTO lucidota_control.event_envelope(event_id, ts, source, actor, raw_ref, raw_artifact_uuid, verbatim_hash, hash_algo, text, entities, claims, actions_requested, artifacts_referenced, risk_flags, route_candidates, board_features, embedding_ref, detail)
                VALUES (%s, now(), 'groq_chat_cli', 'groq', %s, %s::uuid, %s, 'sha256', %s, '[]'::jsonb, '[]'::jsonb, %s::jsonb, '[]'::jsonb, '[]'::jsonb, '[]'::jsonb, %s::jsonb, NULL, %s::jsonb)
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
                    payload["prompt_hash"],
                    payload.get("prompt_text", ""),
                    json.dumps([payload["provider"], payload["model"]]),
                    json.dumps({"provider": payload["provider"], "model": payload["model"], "run_id": payload["run_id"], "mode": payload["mode"]}),
                    json.dumps({"receipt_path": receipt_path, "run_id": payload["run_id"], "mode": payload["mode"]}),
                ),
            )
            work_order_row = cur.execute(
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
                    "groq_chat_provider",
                    "succeeded" if payload.get("status") == "PASS" else "failed",
                    json.dumps(
                        {
                            "provider": payload["provider"],
                            "model": payload["model"],
                            "mode": payload["mode"],
                            "run_id": payload["run_id"],
                            "prompt_hash": payload["prompt_hash"],
                            "system_hash": payload["system_hash"],
                            "receipt_path": receipt_path,
                            "execute_performed": payload.get("execute_performed", False),
                        }
                    ),
                    work_order_key,
                ),
            ).fetchone()
            work_order_uuid = work_order_row["work_order_uuid"] if isinstance(work_order_row, dict) else work_order_row[0]
            receipt_lookup = cur.execute(
                """
                SELECT work_receipt_uuid::text
                FROM lucidota_control.work_receipt
                WHERE work_order_uuid = %s::uuid AND receipt_path = %s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (work_order_uuid, receipt_path),
            ).fetchone()
            if receipt_lookup:
                work_receipt_uuid = receipt_lookup["work_receipt_uuid"] if isinstance(receipt_lookup, dict) else receipt_lookup[0]
            else:
                receipt_row = cur.execute(
                    """
                    INSERT INTO lucidota_control.work_receipt(event_id, work_order_uuid, receipt_path, receipt_sha256, verdict, cost, gain, artifact_refs, canonical_graph_writes_performed, graph_write_mode, detail)
                    VALUES (%s, %s::uuid, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, false, 'staged_only', %s::jsonb)
                    RETURNING work_receipt_uuid::text
                    """,
                    (
                        event_id,
                        work_order_uuid,
                        receipt_path,
                        hashlib.sha256(
                            json.dumps(
                                {
                                    "provider": payload["provider"],
                                    "model": payload["model"],
                                    "mode": payload["mode"],
                                    "run_id": payload["run_id"],
                                    "status": payload.get("status"),
                                    "output_hash": payload.get("output_hash"),
                                },
                                sort_keys=True,
                                separators=(",", ":"),
                            ).encode("utf-8")
                        ).hexdigest(),
                        "promote" if payload.get("status") == "PASS" else "retry",
                        json.dumps({"latency_ms": payload.get("latency_ms", 0), "payload_size_bytes": payload.get("payload_size_bytes", 0)}),
                        json.dumps({"gain": 1.0 if payload.get("status") == "PASS" else 0.0, "output_hash": payload.get("output_hash")}),
                        json.dumps([raw_ref, receipt_path]),
                        json.dumps(
                            {
                                "provider": payload["provider"],
                                "model": payload["model"],
                                "run_id": payload["run_id"],
                                "mode": payload["mode"],
                                "model_invocation_uuid": model_invocation_uuid,
                            }
                        ),
                    ),
                ).fetchone()
                work_receipt_uuid = receipt_row["work_receipt_uuid"] if isinstance(receipt_row, dict) else receipt_row[0]
            cur.execute(
                """
                INSERT INTO lucidota_control.model_invocation(
                  model_invocation_uuid, event_id, target, model_name, prompt_hash, output_hash, payload_size_bytes,
                  latency_ms, token_counts, verdict, receipt_path, raw_output, detail
                )
                VALUES (%s::uuid, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s::jsonb)
                ON CONFLICT (model_invocation_uuid) DO UPDATE SET
                  event_id = EXCLUDED.event_id,
                  target = EXCLUDED.target,
                  model_name = EXCLUDED.model_name,
                  prompt_hash = EXCLUDED.prompt_hash,
                  output_hash = EXCLUDED.output_hash,
                  payload_size_bytes = EXCLUDED.payload_size_bytes,
                  latency_ms = EXCLUDED.latency_ms,
                  token_counts = EXCLUDED.token_counts,
                  verdict = EXCLUDED.verdict,
                  receipt_path = EXCLUDED.receipt_path,
                  raw_output = EXCLUDED.raw_output,
                  detail = EXCLUDED.detail
                RETURNING model_invocation_uuid::text
                """,
                (
                    model_invocation_uuid,
                    event_id,
                    payload["provider"],
                    payload["model"],
                    payload["prompt_hash"],
                    payload["output_hash"],
                    payload["payload_size_bytes"],
                    payload["latency_ms"],
                    json.dumps(payload.get("usage") or {}),
                    "promote" if payload.get("status") == "PASS" else "retry",
                    receipt_path,
                    payload.get("text", ""),
                    json.dumps({"provider": payload["provider"], "model": payload["model"], "run_id": payload["run_id"], "mode": payload["mode"]}),
                ),
            )
    return {
        "model_invocation_uuid": model_invocation_uuid,
        "work_order_uuid": work_order_uuid,
        "work_receipt_uuid": work_receipt_uuid,
        "raw_artifact_uuid": raw_artifact_uuid,
        "event_id": event_id,
    }


def call_groq(base_url: str, key: str, request_payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    endpoint = base_url.rstrip("/") + "/chat/completions"
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(request_payload).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "lucidota-groq-chat-cli/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Call Groq Chat Completions with receipt-backed dry-run/execute modes.")
    ap.add_argument("--prompt", required=True, help="Prompt text, or @path to read prompt from a file.")
    ap.add_argument("--system", default="", help="Optional system message text, or @path.")
    ap.add_argument("--model", default=os.environ.get("GROQ_MODEL", DEFAULT_MODEL))
    ap.add_argument("--base-url", default=os.environ.get("GROQ_BASE_URL", DEFAULT_BASE_URL))
    ap.add_argument("--api-key-env", action="append", default=["GROQ_API_KEY"])
    ap.add_argument("--run-id")
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--max-tokens", type=int)
    ap.add_argument("--timeout-sec", type=float, default=60.0)
    ap.add_argument("--execute", action="store_true", help="Actually call Groq. Omit for dry-run receipt only.")
    ap.add_argument("--no-log-prompts", dest="log_prompts", action="store_false", help="Do not store exact request text in the receipt.")
    ap.add_argument("--json", action="store_true", help="Print full receipt JSON.")
    args = ap.parse_args()

    prompt = read_text_arg(args.prompt)
    system, admin_prompt_policy = compose_system_prompt(read_text_arg(args.system))
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    system_hash = hashlib.sha256(system.encode("utf-8")).hexdigest()
    blockers: list[str] = []
    if not prompt.strip():
        blockers.append("prompt_required")
    if args.timeout_sec <= 0:
        blockers.append("timeout_must_be_positive")
    if args.max_tokens is not None and args.max_tokens <= 0:
        blockers.append("max_tokens_must_be_positive")
    if args.temperature is not None and not 0 <= args.temperature <= 2:
        blockers.append("temperature_outside_0_to_2")
    key, key_env = api_key(args.api_key_env)
    if args.execute and not key:
        blockers.append("missing_api_key_env:" + ",".join(args.api_key_env))

    request_payload: dict[str, Any] = {
        "model": args.model,
        "messages": build_messages(system, prompt),
    }
    if args.temperature is not None:
        request_payload["temperature"] = args.temperature
    if args.max_tokens is not None:
        request_payload["max_tokens"] = args.max_tokens

    payload: dict[str, Any] = {
        "schema": "lucidota.model_invocation.groq_chat.v1",
        "mode": "execute" if args.execute else "dry_run",
        "provider": "groq",
        "endpoint": args.base_url.rstrip("/") + "/chat/completions",
        "model": args.model,
        "run_id": args.run_id or "groq-chat:" + hashlib.sha256(json.dumps({"prompt_hash": prompt_hash, "system_hash": system_hash, "model": args.model, "mode": "execute" if args.execute else "dry_run"}, sort_keys=True).encode("utf-8")).hexdigest()[:24],
        "prompt_text": prompt,
        "system_text": system,
        "prompt_hash": prompt_hash,
        "system_hash": system_hash,
        "api_key_env_used": key_env,
        "api_key_redacted": bool(key),
        "request": {
            **request_payload,
            "messages": message_receipts(request_payload["messages"], log_prompts=args.log_prompts),
        },
        "wire_request": request_payload if args.log_prompts else {
            **request_payload,
            "messages": message_receipts(request_payload["messages"], log_prompts=False),
        },
        "admin_prompt_policy": admin_prompt_policy,
        "generation_trace": build_generation_trace(
            target="groq",
            model_name=args.model,
            request_payload=request_payload,
            latency_ms=0,
            raw_output="",
            raw_response=None,
            execute_performed=False,
        ),
        "execute_performed": False,
        "blockers": blockers,
    }

    if args.execute and not blockers and key:
        started = time.perf_counter()
        try:
            response = call_groq(args.base_url, key, request_payload, args.timeout_sec)
            latency_ms = (time.perf_counter() - started) * 1000
            text = groq_text(response)
            payload.update(
                {
                    "execute_performed": True,
                    "response_id": response.get("id"),
                    "finish_reason": ((response.get("choices") or [{}])[0] or {}).get("finish_reason"),
                    "text": text,
                    "usage": response.get("usage"),
                    "raw_response": response,
                    "latency_ms": round(latency_ms, 3),
                    "generation_trace": build_generation_trace(
                        target="groq",
                        model_name=args.model,
                        request_payload=request_payload,
                        latency_ms=latency_ms,
                        raw_output=text,
                        raw_response=response,
                        execute_performed=True,
                    ),
                }
            )
        except urllib.error.HTTPError as exc:
            latency_ms = (time.perf_counter() - started) * 1000
            body = exc.read().decode("utf-8", errors="replace")
            retry_after = None
            try:
                retry_after = exc.headers.get("Retry-After")
            except Exception:
                retry_after = None
            payload["blockers"].append(f"groq_http_error:{exc.code}")
            payload["error_body"] = body[:4000]
            if retry_after is not None:
                payload["retry_after_header"] = str(retry_after)
                try:
                    payload["retry_after_seconds"] = float(retry_after)
                except Exception:
                    pass
            payload["latency_ms"] = round(latency_ms, 3)
            payload["generation_trace"] = build_generation_trace(target="groq", model_name=args.model, request_payload=request_payload, latency_ms=latency_ms, raw_output=body[:4000], raw_response=None, execute_performed=False)
        except Exception as exc:
            latency_ms = (time.perf_counter() - started) * 1000
            payload["blockers"].append(f"groq_call_failed:{type(exc).__name__}:{exc}")
            payload["latency_ms"] = round(latency_ms, 3)
            payload["generation_trace"] = build_generation_trace(target="groq", model_name=args.model, request_payload=request_payload, latency_ms=latency_ms, raw_output=f"{type(exc).__name__}:{exc}", raw_response=None, execute_performed=False)

    payload["status"] = "PASS" if not payload["blockers"] else "BLOCKED"
    payload["output_hash"] = hashlib.sha256((payload.get("text") or "").encode("utf-8")).hexdigest()
    payload["payload_size_bytes"] = len(json.dumps(request_payload).encode("utf-8"))
    payload["receipt_key"] = hashlib.sha256(json.dumps({"provider": payload["provider"], "model": payload["model"], "prompt_hash": payload["prompt_hash"], "system_hash": payload["system_hash"], "mode": payload["mode"], "run_id": payload["run_id"]}, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:24]
    path = write_receipt(payload)
    try:
        payload["db_write"] = write_db_ledger(payload, receipt_path=rel(path))
    except Exception as exc:
        payload["db_write_error"] = f"{type(exc).__name__}: {exc}"
    payload["visible_response"] = {
        "summary": f"Indy_READs: Groq {args.model} {'executed' if payload['execute_performed'] else 'dry-ran'} and wrote the ledger.",
        "model_invocation_id": payload.get("db_write", {}).get("model_invocation_uuid", ""),
        "work_order_id": payload.get("db_write", {}).get("work_order_uuid", ""),
        "work_receipt_id": payload.get("db_write", {}).get("work_receipt_uuid", ""),
        "attempt_id": payload.get("db_write", {}).get("work_order_uuid", ""),
        "raw_artifact_id": payload.get("db_write", {}).get("raw_artifact_uuid", ""),
        "receipt_path": rel(path),
    }
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        if payload.get("text"):
            print(payload["text"])
        print("RECEIPT_PATH=" + rel(path))
        print("GROQ_CHAT=" + payload["status"])
    return 0 if payload["status"] == "PASS" else 4


if __name__ == "__main__":
    raise SystemExit(main())
