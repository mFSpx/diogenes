#!/usr/bin/env python3
"""Normalized model-generation telemetry for LUCIDOTA model runners."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def stable_payload_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def payload_size_bytes(payload: dict[str, Any]) -> int:
    return len(stable_payload_json(payload).encode("utf-8"))


def build_generation_trace(
    *,
    target: str,
    model_name: str,
    request_payload: dict[str, Any],
    latency_ms: float | int | None,
    raw_output: str,
    raw_response: Any | None = None,
    execute_performed: bool = False,
) -> dict[str, Any]:
    if not target.strip():
        raise ValueError("target_required")
    if not model_name.strip():
        raise ValueError("model_name_required")
    if latency_ms is not None and float(latency_ms) < 0:
        raise ValueError("latency_ms_must_be_non_negative")
    payload_json = stable_payload_json(request_payload)
    output = raw_output if isinstance(raw_output, str) else str(raw_output)
    response_present = raw_response is not None
    return {
        "schema": "lucidota.model_generation_trace.v1",
        "target": target,
        "model_name": model_name,
        "payload_size_bytes": len(payload_json.encode("utf-8")),
        "payload_size_chars": len(payload_json),
        "latency_ms": round(float(latency_ms or 0), 3),
        "raw_output": output,
        "raw_output_chars": len(output),
        "raw_response_present": response_present,
        "raw_response_keys": sorted(raw_response.keys()) if isinstance(raw_response, dict) else [],
        "execute_performed": bool(execute_performed),
    }


def spawn_generation_event_bridge(receipt_path: Path | str, *, root: Path | None = None) -> dict[str, Any]:
    """Launch non-blocking PG/ABSURD event staging for a written receipt.

    This is deliberately best-effort and async: model runners must not block on
    global DB reads/writes. The bridge itself performs the targeted read of the
    single receipt path and writes durable state.
    """
    if os.environ.get("LUCIDOTA_DISABLE_ASYNC_MODEL_EVENT_BRIDGE") or os.environ.get("PYTEST_CURRENT_TEST"):
        return {"launched": False, "reason": "disabled"}
    base = root or Path(__file__).resolve().parents[1]
    script = base / "scripts" / "model_generation_event_bridge.py"
    if not script.exists():
        return {"launched": False, "reason": "bridge_script_missing"}
    out_dir = base / "05_OUTPUTS" / "model_generation_events"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    log_path = out_dir / f"model_generation_event_bridge_async_{stamp}.log"
    receipt = Path(receipt_path)
    try:
        receipt_arg = str(receipt.resolve().relative_to(base.resolve()))
    except Exception:
        receipt_arg = str(receipt)
    log = log_path.open("ab")
    proc = subprocess.Popen(
        [sys.executable, str(script), "stage", "--receipt", receipt_arg, "--execute"],
        cwd=base,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
    )
    log.close()
    return {"launched": True, "pid": proc.pid, "log_path": str(log_path.relative_to(base))}
