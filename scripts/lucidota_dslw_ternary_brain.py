#!/usr/bin/env python3
"""
DSLW — Double LoRA Single Weighting — Ternary Brain Controller.

Architecture:
  Shared base: Bonsai 8B Q1_0 (1-bit weights, 1.1 GB)
  Two personalities layered on the shared base via the llama.cpp server's
  parallel-slot + unified-KV topology:
    Slot 0 (Proposer) — expansive, generative, reaches
    Slot 1 (Critic)   — contractive, analytical, checks
  Ternary synthesis: merge Proposer reach + Critic check → single judgment.

The "double LoRA" is currently realized through divergent system prompts and
sampling parameters per slot. When LoRA GGUF adapters are trained, they
slot in via --lora / --lora-scaled flags on the server command line.

Usage:
  source scripts/lucidota_safe_ops_env.sh

  # Start the server (runs Bonsai 8B Q1_0 with 2 slots, unified KV)
  bash scripts/lucidota_start_bonsai_ternary_llama.sh

  # Run the ternary brain on a prompt
  python3 scripts/lucidota_dslw_ternary_brain.py --prompt "your prompt"

  # Run with JSON output and receipt
  python3 scripts/lucidota_dslw_ternary_brain.py --prompt "..." --json --receipt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

# ─── Slot personalities ───────────────────────────────────────────
# These are the "Double LoRA" in prompt space until real LoRA GGUF
# adapters are trained. Each slot gets its own temperature, system
# prompt, and response framing.

PROPOSER = {
    "slot": 0,
    "name": "proposer",
    "temperature": 0.7,
    "top_p": 0.92,
    "max_tokens": 256,
    "system": (
        "You are the PROPOSER hemisphere of a ternary reasoning engine. "
        "Your role is GENERATIVE and EXPANSIVE. Reach outward: generate "
        "hypotheses, surface possibilities, propose actions, explore edges. "
        "Do not self-censor. Do not evaluate your own output — that is the "
        "Critic's job. Output raw proposals with confidence markers.\n\n"
        "Confidence markers: [HI] = high confidence, [MD] = medium, [LO] = speculative."
    ),
}

CRITIC = {
    "slot": 1,
    "name": "critic",
    "temperature": 0.3,
    "top_p": 0.85,
    "max_tokens": 256,
    "system": (
        "You are the CRITIC hemisphere of a ternary reasoning engine. "
        "Your role is CONTRACTIVE and ANALYTICAL. You receive a proposal "
        "and must: 1) identify the strongest claim, 2) identify the weakest "
        "claim or risk, 3) assign a GO/NOGO judgment with reasoning. "
        "Be terse. Use bullet points. End with a single line: "
        "'SYNTHESIS: <one-sentence integrated judgment>'."
    ),
}

# ─── HTTP helpers ──────────────────────────────────────────────────

def _base_url(host: str = "127.0.0.1", port: int = 8082) -> str:
    return f"http://{host}:{port}/v1"


def _chat_completion(
    *,
    messages: list[dict[str, str]],
    temperature: float,
    top_p: float,
    max_tokens: int,
    host: str = "127.0.0.1",
    port: int = 8082,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """Send a chat completion request to the Bonsai llama-server."""
    import urllib.request
    import urllib.error

    url = f"{_base_url(host, port)}/chat/completions"
    body = json.dumps({
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": False,
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('LLXPRT_LOCAL_API_KEY', 'not-needed')}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        return {"error": str(exc), "ok": False}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "ok": False}


def _health(host: str = "127.0.0.1", port: int = 8082, timeout: float = 2.0) -> dict[str, Any]:
    import urllib.request
    import urllib.error
    url = f"http://{host}:{port}/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return {"ok": True, "status": resp.status, "body": resp.read(2048).decode("utf-8", "replace")}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


# ─── Ternary brain ─────────────────────────────────────────────────

def run_ternary_brain(
    *,
    prompt: str,
    host: str = "127.0.0.1",
    port: int = 8082,
    execute: bool = False,
) -> dict[str, Any]:
    """Run the full DSLW ternary reasoning loop.

    Proposer generates → Critic evaluates → Synthesis merges.
    """

    t0 = time.time()

    # Check server health first
    health = _health(host, port)
    if not health.get("ok"):
        return {
            "schema": "lucidota.dslw_ternary_brain.v1",
            "status": "SERVER_DOWN",
            "health": health,
            "elapsed_ms": int((time.time() - t0) * 1000),
        }

    if not execute:
        return {
            "schema": "lucidota.dslw_ternary_brain.v1",
            "status": "DRY_RUN",
            "mode": "dry_run",
            "health": health,
            "prompt": prompt,
            "proposer_config": PROPOSER["name"],
            "critic_config": CRITIC["name"],
            "elapsed_ms": int((time.time() - t0) * 1000),
        }

    # ── Proposer pass ──
    proposer_msgs = [
        {"role": "system", "content": PROPOSER["system"]},
        {"role": "user", "content": prompt},
    ]
    t_proposer = time.time()
    proposer_result = _chat_completion(
        messages=proposer_msgs,
        temperature=PROPOSER["temperature"],
        top_p=PROPOSER["top_p"],
        max_tokens=PROPOSER["max_tokens"],
        host=host,
        port=port,
    )
    proposer_ms = int((time.time() - t_proposer) * 1000)

    proposer_text = ""
    if proposer_result.get("choices"):
        proposer_text = proposer_result["choices"][0]["message"]["content"]
    elif proposer_result.get("error"):
        return {
            "schema": "lucidota.dslw_ternary_brain.v1",
            "status": "PROPOSER_FAILED",
            "proposer_error": proposer_result["error"],
            "elapsed_ms": int((time.time() - t0) * 1000),
        }

    # ── Critic pass ──
    critic_prompt = (
        f"Evaluate this proposal:\n\n"
        f"PROPOSAL:\n{proposer_text}\n\n"
        f"ORIGINAL PROMPT:\n{prompt}"
    )
    critic_msgs = [
        {"role": "system", "content": CRITIC["system"]},
        {"role": "user", "content": critic_prompt},
    ]
    t_critic = time.time()
    critic_result = _chat_completion(
        messages=critic_msgs,
        temperature=CRITIC["temperature"],
        top_p=CRITIC["top_p"],
        max_tokens=CRITIC["max_tokens"],
        host=host,
        port=port,
    )
    critic_ms = int((time.time() - t_critic) * 1000)

    critic_text = ""
    if critic_result.get("choices"):
        critic_text = critic_result["choices"][0]["message"]["content"]
    elif critic_result.get("error"):
        return {
            "schema": "lucidota.dslw_ternary_brain.v1",
            "status": "CRITIC_FAILED",
            "proposer_text": proposer_text,
            "critic_error": critic_result["error"],
            "elapsed_ms": int((time.time() - t0) * 1000),
        }

    # ── Synthesis ──
    # Extract any SYNTHESIS line from critic output, otherwise use
    # the full critic text as the synthesis.
    synthesis = critic_text
    for line in critic_text.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("SYNTHESIS:"):
            synthesis = stripped.split(":", 1)[1].strip()
            break

    # Determine verdict
    critic_upper = critic_text.upper()
    if "NOGO" in critic_upper:
        verdict = "NOGO"
    elif "GO" in critic_upper and "NOGO" not in critic_upper:
        verdict = "GO"
    else:
        verdict = "UNCLEAR"

    elapsed_ms = int((time.time() - t0) * 1000)

    payload: dict[str, Any] = {
        "schema": "lucidota.dslw_ternary_brain.v1",
        "status": "COMPLETE",
        "verdict": verdict,
        "prompt": prompt,
        "proposer": {
            "text": proposer_text,
            "latency_ms": proposer_ms,
            "config": {
                "temperature": PROPOSER["temperature"],
                "top_p": PROPOSER["top_p"],
                "max_tokens": PROPOSER["max_tokens"],
            },
        },
        "critic": {
            "text": critic_text,
            "latency_ms": critic_ms,
            "config": {
                "temperature": CRITIC["temperature"],
                "top_p": CRITIC["top_p"],
                "max_tokens": CRITIC["max_tokens"],
            },
        },
        "synthesis": synthesis,
        "total_latency_ms": elapsed_ms,
        "host": f"{host}:{port}",
        "model": "Bonsai-8B-Q1_0 (dual-slot, unified KV, virtual DSLW)",
    }

    return payload


# ─── CLI ────────────────────────────────────────────────────────────

def _stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _write_receipt(payload: dict[str, Any]) -> Path:
    out = ROOT / "05_OUTPUTS" / "dslw_ternary"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"dslw_{_stamp()}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="DSLW Ternary Brain — Proposer/Critic over Bonsai 8B Q1_0"
    )
    parser.add_argument("--prompt", "-p", required=True, help="Prompt text or @file")
    parser.add_argument("--host", default=os.environ.get("LUCIDOTA_BONSAI_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("LUCIDOTA_BONSAI_PORT", "8082")))
    parser.add_argument("--execute", "-x", action="store_true", help="Actually call the server")
    parser.add_argument("--json", "-j", action="store_true", help="Output JSON only")
    parser.add_argument("--receipt", "-r", action="store_true", help="Write receipt to 05_OUTPUTS/")
    args = parser.parse_args(argv)

    prompt = args.prompt
    if prompt.startswith("@"):
        prompt_path = Path(prompt[1:])
        if not prompt_path.exists():
            print(f"ERROR: prompt file not found: {prompt_path}", file=sys.stderr)
            return 2
        prompt = prompt_path.read_text(encoding="utf-8").strip()

    result = run_ternary_brain(
        prompt=prompt,
        host=args.host,
        port=args.port,
        execute=args.execute,
    )

    if args.receipt:
        receipt_path = _write_receipt(result)
        result["receipt_path"] = str(receipt_path.relative_to(ROOT))

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        status = result.get("status", "UNKNOWN")
        print(f"DSLW TERNARY BRAIN — {status}")
        print(f"  Model: {result.get('model', '?')}")
        print(f"  Latency: {result.get('total_latency_ms', 0)}ms")
        if result.get("verdict"):
            print(f"  Verdict: {result['verdict']}")
        if result.get("proposer", {}).get("text"):
            print(f"\n── PROPOSER ──")
            print(result["proposer"]["text"][:500])
        if result.get("critic", {}).get("text"):
            print(f"\n── CRITIC ──")
            print(result["critic"]["text"][:500])
        if result.get("synthesis"):
            print(f"\n── SYNTHESIS ──")
            print(result["synthesis"])
        if result.get("receipt_path"):
            print(f"\nReceipt: {result['receipt_path']}")

    if result.get("status") == "COMPLETE":
        return 0
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
