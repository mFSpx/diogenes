#!/usr/bin/env python3
"""Compile subsystem-quality repair rows into bounded ABSURD work orders."""
from __future__ import annotations

import argparse, hashlib, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "quality_work_orders"
DEFAULT_SOURCE = ROOT / "05_OUTPUTS/subsystem_quality_audit/repair_queue_latest.jsonl"


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: str | Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def stable_id(row: dict[str, Any]) -> str:
    raw = json.dumps({"path": row.get("path"), "hash": row.get("hash") or row.get("sha256"), "verdict": row.get("verdict")}, sort_keys=True)
    return "qwo:" + hashlib.sha256(raw.encode()).hexdigest()[:16]


def build_orders(rows: list[dict[str, Any]], *, limit: int, receipt_root: str = "05_OUTPUTS/quality_work_orders/ouroboros") -> list[dict[str, Any]]:
    orders = []
    for row in rows[: max(0, limit)]:
        target = row.get("path") or row.get("target_path")
        reason = row.get("reason") or row.get("last_known_purpose") or "quality repair"
        command = [
            sys.executable,
            "scripts/lucidota_ouroboros_loop.py",
            "--loops",
            "1",
            "--target",
            str(target),
            "--receipt-root",
            receipt_root,
        ]
        orders.append({
            "work_order_id": stable_id(row),
            "handler": "external_command",
            "target_path": str(target),
            "verdict": row.get("verdict", "REPAIR"),
            "reason": reason,
            "command": command,
            "timeout_seconds": 120,
            "canonical_graph_writes_performed": False,
        })
    return orders


def enqueue(orders: list[dict[str, Any]], *, queue: str, workflow: str, timeout: int = 120) -> list[dict[str, Any]]:
    receipts = []
    for idx, order in enumerate(orders, 1):
        payload = {"handler": order["handler"], "command": order["command"], "quality_work_order": order}
        cmd = [
            sys.executable,
            "scripts/absurd_queue_spine.py",
            "--action",
            "enqueue",
            "--execute",
            "--queue",
            queue,
            "--workflow",
            workflow,
            "--job-kind",
            "external_command",
            "--payload-json",
            json.dumps(payload, separators=(",", ":"), sort_keys=True),
            "--idempotency-key",
            order["work_order_id"],
            "--priority",
            str(idx),
        ]
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=timeout)
        report = next((line.split("=", 1)[1] for line in proc.stdout.splitlines() if line.startswith("REPORT_PATH=")), "")
        receipts.append({"work_order_id": order["work_order_id"], "target_path": order.get("target_path", ""), "rc": proc.returncode, "report_path": report, "stderr_tail": proc.stderr[-1000:]})
    return receipts


def write_report(payload: dict[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"quality_work_order_compiler_{stamp()}.json"
    payload["report_path"] = rel(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print("REPORT_PATH=" + rel(path))
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=str(DEFAULT_SOURCE)); ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--queue", default="goal_swarm"); ap.add_argument("--workflow", default="quality_repair_queue_v1")
    ap.add_argument("--execute", action="store_true"); ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    rows = load_jsonl(args.source); orders = build_orders(rows, limit=args.limit)
    receipts = enqueue(orders, queue=args.queue, workflow=args.workflow) if args.execute else []
    blockers = [r for r in receipts if r["rc"] != 0]
    payload = {"schema": "lucidota.quality_work_order_compiler.v1", "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "source": rel(args.source), "execute_performed": bool(args.execute), "order_count": len(orders), "orders": orders, "enqueue_receipts": receipts, "blockers": blockers, "status": "PASS" if not blockers else "FAIL"}
    write_report(payload)
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    print("QUALITY_WORK_ORDER_COMPILER=" + payload["status"])
    return 0 if not blockers else 3


if __name__ == "__main__":
    raise SystemExit(main())
