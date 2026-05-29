#!/usr/bin/env python3
"""LUCIDOTA Groq delegation tool.

Routes work to the right Groq model based on task type.
Groq never makes final repo truth. Groq returns candidates. Opus judges.

MODEL POLICY:
  scout/json/code : qwen/qwen3-32b  → openai/gpt-oss-120b → llama-3.3-70b-versatile
  reason/critic   : openai/gpt-oss-120b → qwen/qwen3-32b → llama-3.3-70b-versatile
  fallback        : llama-3.3-70b-versatile (always stable)

--allow-preview (default ON for local dev): allows Qwen3-32B even if preview-flagged.
--no-preview: forces GPT-OSS-120B / Llama-70B only — use for CI / legal authority / Opus briefing.
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

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "groq_tool"
GROQ_BASE = "https://api.groq.com/openai/v1"

# Ordered preference chains per route
ROUTE_CHAINS: dict[str, list[str]] = {
    "scout":    ["qwen/qwen3-32b", "openai/gpt-oss-120b", "llama-3.3-70b-versatile"],
    "json":     ["qwen/qwen3-32b", "openai/gpt-oss-120b", "llama-3.3-70b-versatile"],
    "code":     ["qwen/qwen3-32b", "openai/gpt-oss-120b", "llama-3.3-70b-versatile"],
    "reason":   ["openai/gpt-oss-120b", "qwen/qwen3-32b", "llama-3.3-70b-versatile"],
    "critic":   ["openai/gpt-oss-120b", "qwen/qwen3-32b", "llama-3.3-70b-versatile"],
    "fallback": ["llama-3.3-70b-versatile"],
}

PREVIEW_MODELS = {"qwen/qwen3-32b"}
STABLE_MODELS  = {"openai/gpt-oss-120b", "llama-3.3-70b-versatile"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()

def load_key() -> str:
    # Try env first, then secrets file
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        secrets = Path.home() / ".config" / "lucidota" / "secrets.env"
        if secrets.exists():
            for line in secrets.read_text().splitlines():
                if line.startswith("GROQ_API_KEY="):
                    key = line.split("=", 1)[1].strip()
                    break
    if not key:
        key_file = Path("/tmp/lucidota_groq_key")
        if key_file.exists():
            key = key_file.read_text().strip()
    return key

def fetch_active_models(key: str) -> list[str]:
    req = urllib.request.Request(
        f"{GROQ_BASE}/models",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
        return [m["id"] for m in data.get("data", [])]
    except Exception:
        return []

def write_inventory(active: list[str]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"model_inventory_{stamp()}.json"
    p.write_text(json.dumps({"fetched_at": now_iso(), "active_models": sorted(active)}, indent=2))
    return p

def select_model(
    route: str,
    active: list[str],
    allow_preview: bool,
    model_override: str | None,
) -> tuple[str, list[str]]:
    chain = ROUTE_CHAINS[route]

    if model_override:
        if model_override not in active:
            raise ValueError(f"--model {model_override!r} is not active on Groq right now")
        if not allow_preview and model_override in PREVIEW_MODELS:
            raise ValueError(f"--model {model_override!r} is a preview model; use --allow-preview or pick a stable model")
        return model_override, chain

    for candidate in chain:
        if candidate not in active:
            continue
        if not allow_preview and candidate in PREVIEW_MODELS:
            continue
        return candidate, chain

    raise RuntimeError(f"No available model for route={route!r} allow_preview={allow_preview}. Active: {active}")

def call_groq(
    key: str,
    model: str,
    prompt: str,
    system: str,
    max_tokens: int,
    temperature: float,
    json_mode: bool,
    timeout: int,
    retries: int,
) -> tuple[str, dict[str, Any]]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{GROQ_BASE}/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )

    last_err = ""
    for attempt in range(retries + 1):
        try:
            t0 = time.monotonic()
            with urllib.request.urlopen(req, timeout=timeout) as r:
                resp = json.loads(r.read())
            latency_ms = int((time.monotonic() - t0) * 1000)
            content = resp["choices"][0]["message"]["content"]
            usage = resp.get("usage", {})
            return content, {"latency_ms": latency_ms, "usage": usage}
        except urllib.error.HTTPError as e:
            last_err = f"HTTP {e.code}: {e.read().decode()[:200]}"
        except Exception as e:
            last_err = str(e)
        if attempt < retries:
            time.sleep(2 ** attempt)

    raise RuntimeError(f"Groq call failed after {retries+1} attempts: {last_err}")

def write_receipt(data: dict[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"receipt_{stamp()}.json"
    p.write_text(json.dumps(data, indent=2, default=str))
    return p

def read_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    if args.prompt:
        return args.prompt
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise ValueError("No prompt: use --prompt, --prompt-file, or stdin")

SYSTEM_PROMPTS = {
    "scout":    "You are a fast scout. Return terse, structured findings. Be direct.",
    "json":     "You return valid JSON only. No prose. No markdown fences.",
    "code":     "You are a senior systems engineer. Return working code with brief explanation.",
    "reason":   "You are a deep reasoning judge. Think step by step. Be adversarial.",
    "critic":   "You are an adversarial critic. Find flaws, edge cases, and hidden assumptions.",
    "fallback": "You are a helpful assistant. Return clear, concise answers.",
}

def main() -> int:
    ap = argparse.ArgumentParser(description="LUCIDOTA Groq delegation tool")
    ap.add_argument("--route", choices=list(ROUTE_CHAINS), default="scout")
    ap.add_argument("--prompt", help="Prompt string")
    ap.add_argument("--prompt-file", help="Path to prompt file")
    ap.add_argument("--model", help="Override model (must be active)")
    ap.add_argument("--allow-preview", action="store_true", default=True, help="Allow preview models like Qwen3 (default ON)")
    ap.add_argument("--no-preview", dest="allow_preview", action="store_false", help="Disable preview models; force stable only")
    ap.add_argument("--json", dest="json_mode", action="store_true", help="Request JSON output mode")
    ap.add_argument("--out", help="Write output text to this path")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--inventory-only", action="store_true", help="Just fetch and print active models")
    ap.add_argument("--dry-run", action="store_true", help="Show what would run, no API call")
    args = ap.parse_args()

    key = load_key()
    if not key:
        print("ERROR: no GROQ_API_KEY found", file=sys.stderr)
        return 1

    # Fetch active models
    active = fetch_active_models(key)
    inv_path = write_inventory(active)

    if args.inventory_only:
        print(json.dumps({"active_models": sorted(active), "inventory": str(inv_path)}, indent=2))
        return 0

    # Select model
    try:
        model, chain = select_model(args.route, active, args.allow_preview, args.model)
    except (ValueError, RuntimeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    # Read prompt
    try:
        prompt_text = read_prompt(args)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    prompt_hash = sha256(prompt_text)

    if args.dry_run:
        print(json.dumps({
            "dry_run": True, "route": args.route, "selected_model": model,
            "fallback_chain": chain, "allow_preview": args.allow_preview,
            "prompt_hash": prompt_hash, "active_models": sorted(active),
        }, indent=2))
        return 0

    # Call
    receipt: dict[str, Any] = {
        "schema": "lucidota.groq_tool.receipt.v1",
        "timestamp": now_iso(),
        "route": args.route,
        "selected_model": model,
        "fallback_chain": chain,
        "preview_allowed": args.allow_preview,
        "prompt_hash": prompt_hash,
        "json_mode": args.json_mode,
        "max_tokens": args.max_tokens,
    }

    try:
        content, meta = call_groq(
            key=key,
            model=model,
            prompt=prompt_text,
            system=SYSTEM_PROMPTS[args.route],
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            json_mode=args.json_mode,
            timeout=args.timeout,
            retries=args.retries,
        )
        receipt.update({
            "status": "ok",
            "output_hash": sha256(content),
            "latency_ms": meta["latency_ms"],
            "usage": meta["usage"],
        })
    except Exception as e:
        receipt["status"] = "error"
        receipt["error"] = str(e)
        rp = write_receipt(receipt)
        print(f"ERROR: {e}\nReceipt: {rp}", file=sys.stderr)
        return 1

    # Validate JSON if requested
    if args.json_mode:
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            # One repair attempt
            try:
                start = content.index("{")
                end = content.rindex("}") + 1
                parsed = json.loads(content[start:end])
                content = json.dumps(parsed)
            except Exception:
                receipt["status"] = "error"
                receipt["error"] = "malformed JSON output after repair attempt"
                write_receipt(receipt)
                print("ERROR: model returned malformed JSON", file=sys.stderr)
                return 1

    receipt_path = write_receipt(receipt)

    # Output
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(content, encoding="utf-8")

    print(content)
    print(f"\n# Receipt: {receipt_path}", file=sys.stderr)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
