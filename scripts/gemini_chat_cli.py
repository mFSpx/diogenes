#!/usr/bin/env python3
"""Gemini chat bridge with receipts and dry-run default.

Uses the Gemini REST generateContent endpoint with an API key from the
environment only. Keys are never printed or written to receipts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from provider_secret_quarantine import load_provider_secret_env

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "model_invocations"
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MODEL = "gemini-2.5-flash"


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


def api_key_candidates(env_names: list[str]) -> list[tuple[str, str]]:
    return [(name, os.environ.get(name, "").strip()) for name in env_names if os.environ.get(name, "").strip()]


def build_request(system: str, prompt: str, *, model: str, temperature: float | None, max_tokens: int | None) -> dict[str, Any]:
    request: dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
    }
    if system.strip():
        request["systemInstruction"] = {"parts": [{"text": system}]}
    generation_config: dict[str, Any] = {}
    if temperature is not None:
        generation_config["temperature"] = temperature
    if max_tokens is not None:
        generation_config["maxOutputTokens"] = max_tokens
    if generation_config:
        request["generationConfig"] = generation_config
    request["model"] = model
    return request


def call_gemini(base_url: str, key: str, model: str, request_payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    endpoint = base_url.rstrip("/") + f"/models/{urllib.parse.quote(model, safe='')}:generateContent"
    req = urllib.request.Request(
        endpoint,
        data=json.dumps({k: v for k, v in request_payload.items() if k != "model"}).encode("utf-8"),
        method="POST",
        headers={
            "x-goog-api-key": key,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "lucidota-gemini-chat-cli/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def execute_gemini_with_key_fallback(
    *,
    base_url: str,
    model: str,
    request_payload: dict[str, Any],
    timeout: float,
    key_candidates: list[tuple[str, str]],
    call_fn=call_gemini,
) -> tuple[dict[str, Any] | None, str | None, list[str], str, list[str]]:
    blockers: list[str] = []
    attempt_blockers: list[str] = []
    last_error_body = ""
    for idx, (key_env, key) in enumerate(key_candidates):
        try:
            response = call_fn(base_url, key, model, request_payload, timeout)
            return response, key_env, blockers, "", attempt_blockers
        except urllib.error.HTTPError as exc:
            last_error_body = exc.read().decode("utf-8", errors="replace") if getattr(exc, "fp", None) else ""
            attempt_blockers.append(f"gemini_http_error:{exc.code}")
            if exc.code in {401, 429} and idx + 1 < len(key_candidates):
                continue
            blockers.append(f"gemini_http_error:{exc.code}")
            return None, key_env, blockers, last_error_body, attempt_blockers
        except Exception as exc:
            blockers.append(f"gemini_call_failed:{type(exc).__name__}:{exc}")
            return None, key_env, blockers, last_error_body, attempt_blockers
    return None, None, blockers, last_error_body, attempt_blockers


def extract_text(response: dict[str, Any]) -> str:
    try:
        candidates = response.get("candidates") or []
        if not candidates:
            return ""
        content = (candidates[0] or {}).get("content") or {}
        parts = content.get("parts") or []
        texts: list[str] = []
        for part in parts:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                texts.append(part["text"])
        return "".join(texts).strip()
    except Exception:
        return ""


def write_receipt(payload: dict[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"gemini_chat_{payload['mode']}_{payload['receipt_key']}.json"
    payload["generated_at"] = now()
    payload["report_path"] = rel(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
    return path


def main() -> int:
    load_provider_secret_env()
    ap = argparse.ArgumentParser(description="Call Gemini generateContent with receipt-backed dry-run/execute modes.")
    ap.add_argument("--prompt", required=True, help="Prompt text, or @path to read prompt from a file.")
    ap.add_argument("--system", default="", help="Optional system message text, or @path.")
    ap.add_argument("--model", default=os.environ.get("GEMINI_MODEL", DEFAULT_MODEL))
    ap.add_argument("--base-url", default=os.environ.get("GEMINI_BASE_URL", DEFAULT_BASE_URL))
    ap.add_argument("--api-key-env", action="append", default=["GEMINI_API_KEY", "GOOGLE_API_KEY"])
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--max-tokens", type=int)
    ap.add_argument("--timeout-sec", type=float, default=60.0)
    ap.add_argument("--execute", action="store_true", help="Actually call Gemini. Omit for dry-run receipt only.")
    ap.add_argument("--json", action="store_true", help="Print full receipt JSON.")
    args = ap.parse_args()

    prompt = read_text_arg(args.prompt)
    system = read_text_arg(args.system)
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

    key_candidates = api_key_candidates(args.api_key_env)
    if args.execute and not key_candidates:
        blockers.append("missing_api_key_env:" + ",".join(args.api_key_env))

    request_payload = build_request(system, prompt, model=args.model, temperature=args.temperature, max_tokens=args.max_tokens)

    payload: dict[str, Any] = {
        "schema": "lucidota.model_invocation.gemini_chat.v1",
        "mode": "execute" if args.execute else "dry_run",
        "provider": "gemini",
        "endpoint": args.base_url.rstrip("/") + f"/models/{args.model}:generateContent",
        "model": args.model,
        "prompt_text": prompt,
        "system_text": system,
        "prompt_hash": prompt_hash,
        "system_hash": system_hash,
        "api_key_env_used": key_candidates[0][0] if key_candidates else None,
        "api_key_redacted": bool(key_candidates),
        "request": request_payload,
        "wire_request": request_payload,
        "execute_performed": False,
        "blockers": blockers,
    }

    started = time.perf_counter()
    if args.execute and not blockers and key_candidates:
        try:
            response, key_env_used, call_blockers, error_body, attempt_blockers = execute_gemini_with_key_fallback(
                base_url=args.base_url,
                model=args.model,
                request_payload=request_payload,
                timeout=args.timeout_sec,
                key_candidates=key_candidates,
            )
            payload["blockers"].extend(call_blockers)
            if attempt_blockers:
                payload["fallback_blockers"] = attempt_blockers
            if response is not None:
                payload.update(
                    {
                        "execute_performed": True,
                        "response": response,
                        "response_id": response.get("responseId") or response.get("id"),
                        "text": extract_text(response),
                    }
                )
            if error_body:
                payload["error_body"] = error_body[:4000]
            if key_env_used:
                payload["api_key_env_used"] = key_env_used
        finally:
            payload["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)

    payload["status"] = "PASS" if not payload["blockers"] else "BLOCKED"
    payload["output_hash"] = hashlib.sha256((payload.get("text") or "").encode("utf-8")).hexdigest()
    payload["payload_size_bytes"] = len(json.dumps(request_payload).encode("utf-8"))
    payload["receipt_key"] = hashlib.sha256(
        json.dumps(
            {
                "provider": payload["provider"],
                "model": payload["model"],
                "prompt_hash": payload["prompt_hash"],
                "system_hash": payload["system_hash"],
                "mode": payload["mode"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:24]
    path = write_receipt(payload)
    payload["visible_response"] = {
        "summary": f"Indy_READs: Gemini {args.model} {'executed' if payload['execute_performed'] else 'dry-ran'} and wrote the ledger.",
        "receipt_path": rel(path),
    }
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        if payload.get("text"):
            print(payload["text"])
        print("RECEIPT_PATH=" + rel(path))
        print("GEMINI_CHAT=" + payload["status"])
    return 0 if payload["status"] == "PASS" else 4


if __name__ == "__main__":
    raise SystemExit(main())
