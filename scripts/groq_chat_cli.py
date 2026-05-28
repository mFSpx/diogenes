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
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
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
    path = OUT / f"groq_chat_{payload['mode']}_{stamp()}.json"
    payload["generated_at"] = now()
    payload["report_path"] = rel(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
    try:
        spawn_generation_event_bridge(path, root=ROOT)
    except Exception:
        pass
    return path


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
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--max-tokens", type=int)
    ap.add_argument("--timeout-sec", type=float, default=60.0)
    ap.add_argument("--execute", action="store_true", help="Actually call Groq. Omit for dry-run receipt only.")
    ap.add_argument("--no-log-prompts", dest="log_prompts", action="store_false", help="Do not store exact request text in the receipt.")
    ap.add_argument("--json", action="store_true", help="Print full receipt JSON.")
    args = ap.parse_args()

    prompt = read_text_arg(args.prompt)
    system, admin_prompt_policy = compose_system_prompt(read_text_arg(args.system))
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
            payload["blockers"].append(f"groq_http_error:{exc.code}")
            payload["error_body"] = body[:4000]
            payload["latency_ms"] = round(latency_ms, 3)
            payload["generation_trace"] = build_generation_trace(target="groq", model_name=args.model, request_payload=request_payload, latency_ms=latency_ms, raw_output=body[:4000], raw_response=None, execute_performed=False)
        except Exception as exc:
            latency_ms = (time.perf_counter() - started) * 1000
            payload["blockers"].append(f"groq_call_failed:{type(exc).__name__}:{exc}")
            payload["latency_ms"] = round(latency_ms, 3)
            payload["generation_trace"] = build_generation_trace(target="groq", model_name=args.model, request_payload=request_payload, latency_ms=latency_ms, raw_output=f"{type(exc).__name__}:{exc}", raw_response=None, execute_performed=False)

    payload["status"] = "PASS" if not payload["blockers"] else "BLOCKED"
    path = write_receipt(payload)
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    if payload.get("text"):
        print(payload["text"])
    print("RECEIPT_PATH=" + rel(path))
    print("GROQ_CHAT=" + payload["status"])
    return 0 if payload["status"] == "PASS" else 4


if __name__ == "__main__":
    raise SystemExit(main())
