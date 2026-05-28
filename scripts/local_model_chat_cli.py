#!/usr/bin/env python3
"""Receipt-backed local model invocation probe for llama.cpp and Needle lanes."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from model_invocation_trace import build_generation_trace, spawn_generation_event_bridge
from project2501_admin_prompt import compose_system_prompt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "model_invocations"

LLAMA_LANES = {
    "deepseek": (8080, "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf", "03_VAULT/models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf"),
    "mamba_cpu": (8081, "Falcon3-Mamba-7B-Instruct-Q2_K.gguf", "03_VAULT/models/tensorblock/Falcon3-Mamba-7B-Instruct-GGUF/Falcon3-Mamba-7B-Instruct-Q2_K.gguf"),
    "bonsai": (8082, "Ternary-Bonsai-4B-Q2_0.gguf", "03_VAULT/models/prism-ml/Ternary-Bonsai-4B-gguf/Ternary-Bonsai-4B-Q2_0.gguf"),
    "mamba_gpu": (8083, "Falcon3-Mamba-7B-Instruct-Q2_K.gguf", "03_VAULT/models/tensorblock/Falcon3-Mamba-7B-Instruct-GGUF/Falcon3-Mamba-7B-Instruct-Q2_K.gguf"),
}
NEEDLE_LANES = {f"needle_{i}": (8090 + i, "needle-26m", "03_VAULT/models/needle/needle.pkl") for i in range(6)}
NEEDLE_LANES["spark"] = NEEDLE_LANES["needle_0"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def read_text_arg(value: str) -> str:
    if value.startswith("@"):
        path = Path(value[1:])
        return (path if path.is_absolute() else ROOT / path).read_text(encoding="utf-8")
    return value


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"content-type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def lane_spec(lane: str) -> dict[str, Any]:
    if lane in LLAMA_LANES:
        port, model, model_path = LLAMA_LANES[lane]
        return {"kind": "llama.cpp", "port": port, "model": model, "model_path": model_path, "endpoint_path": "/v1/chat/completions"}
    if lane in NEEDLE_LANES:
        port, model, model_path = NEEDLE_LANES[lane]
        return {"kind": "needle", "port": port, "model": model, "model_path": model_path, "endpoint_path": "/generate"}
    raise KeyError(lane)


def build_payload(spec: dict[str, Any], prompt: str, system: str, max_tokens: int, temperature: float) -> dict[str, Any]:
    if spec["kind"] == "needle":
        query = (system.strip() + "\n" + prompt).strip()
        return {"query": query, "tools": "[]", "max_gen_len": max_tokens}
    messages = ([{"role": "system", "content": system}] if system.strip() else []) + [{"role": "user", "content": prompt}]
    return {"model": spec["model"], "messages": messages, "max_tokens": max_tokens, "temperature": temperature}


def response_text(spec: dict[str, Any], response: dict[str, Any]) -> str:
    if spec["kind"] == "needle":
        return str(response.get("output", ""))
    try:
        message = response["choices"][0]["message"]
        for key in ("content", "reasoning_content", "reasoning"):
            value = message.get(key)
            if isinstance(value, str) and value:
                return value
        return ""
    except Exception:
        return ""


def token_accounting(prompt: str, system: str, text: str, response: dict[str, Any] | None, dry: bool, max_tokens: int) -> dict[str, Any]:
    usage = (response or {}).get("usage") if isinstance(response, dict) else None
    if isinstance(usage, dict) and isinstance(usage.get("total_tokens"), int):
        return {"source": "provider_usage", **usage}
    prompt_tokens = estimate_tokens((system + "\n" + prompt).strip())
    completion_tokens = max_tokens if dry else estimate_tokens(text)
    return {"source": "dry_run_estimate" if dry else "char_estimate", "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens}


def write_receipt(payload: dict[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"local_model_chat_{payload['lane']}_{payload['mode']}_{stamp()}.json"
    payload["report_path"] = rel(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    try:
        spawn_generation_event_bridge(path, root=ROOT)
    except Exception:
        pass
    return path


def probe_lane(*, lane: str, prompt: str, system: str = "", max_tokens: int = 16, temperature: float = 0.0, timeout: float = 60.0, execute: bool = False, log_prompts: bool = True, clear_history: bool = False) -> dict[str, Any]:
    blockers: list[str] = []
    try:
        spec = lane_spec(lane)
    except KeyError:
        spec = {"kind": "unknown", "port": None, "model": lane, "model_path": None, "endpoint_path": ""}
        blockers.append("unknown_lane")
    system, admin_prompt_policy = compose_system_prompt(system)
    if not prompt.strip():
        blockers.append("prompt_required")
    endpoint = f"http://127.0.0.1:{spec['port']}{spec['endpoint_path']}" if spec.get("port") else ""
    request_payload = build_payload(spec, prompt, system, max_tokens, temperature) if not blockers else {}
    response: dict[str, Any] | None = None
    text = ""
    latency_ms = 0.0
    if execute and not blockers:
        started = time.perf_counter()
        try:
            response = post_json(endpoint, request_payload, timeout)
            text = response_text(spec, response)
        except Exception as exc:
            blockers.append(f"local_call_failed:{type(exc).__name__}:{str(exc)[:160]}")
            text = f"{type(exc).__name__}:{str(exc)[:160]}"
        finally:
            latency_ms = (time.perf_counter() - started) * 1000
    dry = not execute
    receipt: dict[str, Any] = {
        "schema": "lucidota.model_invocation.local_chat.v1",
        "generated_at": now(),
        "mode": "execute" if execute else "dry_run",
        "provider": "local",
        "lane": lane,
        "clear_history_requested": bool(clear_history),
        "backend": spec["kind"],
        "endpoint": endpoint,
        "model": spec["model"],
        "model_path": spec.get("model_path"),
        "request": {"max_tokens": max_tokens, "temperature": temperature, "prompt_chars": len(prompt), "system_chars": len(system), **({"prompt_text": prompt, "system_text": system} if log_prompts else {})},
        "admin_prompt_policy": admin_prompt_policy,
        "input_sha256": sha({"prompt": prompt, "system": system, "lane": lane}),
        "output_sha256": sha({"output": text}),
        "text": text,
        "usage": (response or {}).get("usage") if isinstance(response, dict) else None,
        "token_accounting": token_accounting(prompt, system, text, response, dry, max_tokens),
        "raw_response": response,
        "latency_ms": round(latency_ms, 3),
        "generation_trace": build_generation_trace(
            target="local",
            model_name=spec["model"],
            request_payload=request_payload,
            latency_ms=latency_ms,
            raw_output=text,
            raw_response=response,
            execute_performed=bool(execute and response is not None and not blockers),
        ),
        "execute_performed": bool(execute and not blockers),
        "model_calls_performed": bool(execute and not blockers),
        "real_inference_performed": bool(execute and not blockers),
        "canonical_graph_writes_performed": False,
        "blockers": blockers,
        "status": "PASS" if not blockers else "BLOCKED",
    }
    write_receipt(receipt)
    return receipt


def main() -> int:
    ap = argparse.ArgumentParser(description="Call one local LUCIDOTA model lane and write a prompt/output/token receipt.")
    ap.add_argument("--lane", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--system", default="")
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout-sec", type=float, default=60.0)
    ap.add_argument("--clear-history", action="store_true")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--no-log-prompts", dest="log_prompts", action="store_false")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    if args.clear_history:
        # Local lanes are stateless for probe mode; clear-history is a compatibility no-op.
        pass
    receipt = probe_lane(
        lane=args.lane,
        prompt=read_text_arg(args.prompt),
        system=read_text_arg(args.system),
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout=args.timeout_sec,
        execute=args.execute,
        log_prompts=args.log_prompts,
        clear_history=args.clear_history,
    )
    if args.json:
        print(json.dumps(receipt, sort_keys=True))
    if receipt.get("text"):
        print(receipt["text"])
    print("RECEIPT_PATH=" + receipt["report_path"])
    print("LOCAL_MODEL_CHAT=" + receipt["status"])
    return 0 if receipt["status"] == "PASS" else 4


if __name__ == "__main__":
    raise SystemExit(main())
