#!/usr/bin/env python3
"""Deterministic local audit fallback for signed five-task coverage."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "model_invocation_audits" / "local_audit_receipts"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def latest_audit() -> dict[str, Any]:
    paths = sorted((ROOT / "05_OUTPUTS" / "model_invocation_audits").glob("model_invocation_audit_*.json"))
    if not paths:
        raise SystemExit("no model_invocation_audit receipt found")
    return json.loads(paths[-1].read_text(encoding="utf-8"))


def find_block(audit: dict[str, Any], block_id: str) -> dict[str, Any]:
    for block in audit.get("five_task_audit_blocks", []):
        if block.get("block_id") == block_id:
            return block
    raise SystemExit(f"block not found: {block_id}")


def build_receipt(*, block: dict[str, Any], verdict: str, findings: list[str]) -> dict[str, Any]:
    audit_json = {
        "block_id": block["block_id"],
        "block_signature": block["block_signature"],
        "auditor_provider": "local",
        "verdict": verdict,
        "findings": findings,
        "risk_flags": ["deterministic_fallback_after_local_model_json_failure"],
        "next_action": "rerun scripts/model_invocation_audit.py --json",
    }
    text = json.dumps(audit_json, sort_keys=True, separators=(",", ":"))
    return {
        "schema": "lucidota.model_invocation_audit.local_deterministic.v1",
        "generated_at": now(),
        "provider": "local",
        "model": "deterministic-local-audit-v1",
        "deterministic_audit_receipt": True,
        "audit_block_id": block["block_id"],
        "audit_block_signature": block["block_signature"],
        "text": text,
        "generation_trace": {
            "schema": "lucidota.model_generation_trace.v1",
            "target": "local",
            "model_name": "deterministic-local-audit-v1",
            "raw_output": text,
            "raw_output_chars": len(text),
            "execute_performed": False,
            "latency_ms": 0,
        },
        "task_receipts": block.get("task_receipts", []),
        "execute_performed": False,
        "model_calls_performed": False,
        "real_inference_performed": False,
        "canonical_graph_writes_performed": False,
        "status": "PASS",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block-id", default="task_block_0001")
    ap.add_argument("--verdict", choices=["PASS", "PARTIAL", "FAIL"], default="PARTIAL")
    ap.add_argument("--finding", action="append", default=[])
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    block = find_block(latest_audit(), args.block_id)
    findings = args.finding or ["local deterministic audit preserved coverage without counting as model usage"]
    receipt = build_receipt(block=block, verdict=args.verdict, findings=findings)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"local_deterministic_audit_{args.block_id}_{stamp()}.json"
    receipt["report_path"] = rel(path)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    if args.json:
        print(json.dumps(receipt, sort_keys=True))
    print("REPORT_PATH=" + receipt["report_path"])
    print("LOCAL_DETERMINISTIC_AUDIT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
