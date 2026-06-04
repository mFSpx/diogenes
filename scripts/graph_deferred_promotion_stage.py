#!/usr/bin/env python3
"""Stage deferred graph-promotion packets through the existing graph gate.

This script intentionally does **not** materialize canonical graph rows. It only
turns deferred packet JSONL into `graph_promotion_gate.py gate --execute`
packet/decision/audit writes, preserving the approval fence.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "lucidota.graph.deferred_promotion_stage.receipt.v1"


def now_z() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def iter_jsonl(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            obj = json.loads(line)
            if isinstance(obj, dict):
                yield obj


def _report_path(stdout: str) -> str:
    for line in stdout.splitlines():
        if line.startswith("REPORT_PATH="):
            return line.split("=", 1)[1].strip()
    return ""


def default_python_bin() -> str:
    """Prefer the project venv for graph-gate subprocesses when available."""
    venv_python = ROOT / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def _gate_command(packet: dict[str, Any], payload_path: Path, python_bin: str) -> list[str]:
    cmd = [
        python_bin,
        str(ROOT / "scripts" / "graph_promotion_gate.py"),
        "gate",
        "--execute",
        "--candidate-kind",
        str(packet.get("candidate_kind") or "node"),
        "--candidate-payload-json",
        str(payload_path),
        "--authority-class",
        str(packet.get("authority_class") or "operator_authored_assertion"),
        "--decision",
        str(packet.get("decision") or "defer"),
        "--rationale",
        str(packet.get("rationale") or "Deferred packet staging; no canonical materialization."),
    ]
    for ref in packet.get("evidence_refs") or []:
        if ref:
            cmd.extend(["--evidence-ref", str(ref)])
    return cmd


def stage_deferred_packets(
    *,
    packets_path: Path,
    receipt_path: Path,
    start_index: int = 1,
    max_packets: int = 25,
    dry_run: bool = False,
    python_bin: str | None = None,
    run_cmd: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    seen = 0
    staged = 0
    failed = 0
    resolved_python_bin = python_bin or default_python_bin()

    with tempfile.TemporaryDirectory(prefix="luci-deferred-graph-") as td:
        tmpdir = Path(td)
        for idx, packet in enumerate(iter_jsonl(packets_path), 1):
            if idx < start_index:
                continue
            if max_packets > 0 and seen >= max_packets:
                break
            seen += 1
            payload = packet.get("candidate_payload") if isinstance(packet.get("candidate_payload"), dict) else {}
            if not payload:
                failed += 1
                reports.append({"index": idx, "status": "FAIL", "error": "missing_candidate_payload"})
                continue
            if dry_run:
                reports.append({"index": idx, "status": "DRY_RUN", "chunk_ref": packet.get("chunk_ref", "")})
                continue
            payload_path = tmpdir / f"candidate_{idx}.json"
            payload_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            cmd = _gate_command(packet, payload_path, resolved_python_bin)
            proc = run_cmd(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
            report_path = _report_path(getattr(proc, "stdout", "") or "")
            ok = int(getattr(proc, "returncode", 1) or 0) == 0
            if ok:
                staged += 1
            else:
                failed += 1
            reports.append(
                {
                    "index": idx,
                    "status": "PASS" if ok else "FAIL",
                    "returncode": int(getattr(proc, "returncode", 1) or 0),
                    "report_path": report_path,
                    "stdout_tail": (getattr(proc, "stdout", "") or "")[-1000:],
                    "stderr_tail": (getattr(proc, "stderr", "") or "")[-1000:],
                }
            )

    receipt = {
        "schema": SCHEMA,
        "status": "PASS" if failed == 0 and seen > 0 else "FAIL",
        "created_at": now_z(),
        "packets_path": str(packets_path),
        "dry_run": dry_run,
        "python_bin": resolved_python_bin,
        "start_index": start_index,
        "max_packets": max_packets,
        "packets_seen": seen,
        "packets_staged": staged,
        "packets_failed": failed,
        "db_writes_performed": bool(staged and not dry_run),
        "graph_writes_performed": False,
        "canonical_graph_writes_performed": False,
        "materialize_requested": False,
        "reports": reports,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="graph-deferred-promotion-stage")
    p.add_argument("--packets", type=Path, default=Path("05_OUTPUTS/graph/deferred_promotion_packets_1128.jsonl"))
    p.add_argument("--receipt", type=Path, default=Path("05_OUTPUTS/graph/deferred_promotion_stage_latest.json"))
    p.add_argument("--start-index", type=int, default=1, help="1-based JSONL row index to start staging from.")
    p.add_argument("--max-packets", type=int, default=25)
    p.add_argument("--python-bin", default=None, help="Python executable for graph_promotion_gate.py; defaults to .venv/bin/python when present.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--json", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    receipt = stage_deferred_packets(
        packets_path=args.packets,
        receipt_path=args.receipt,
        start_index=args.start_index,
        max_packets=args.max_packets,
        dry_run=args.dry_run,
        python_bin=args.python_bin,
    )
    print(json.dumps(receipt, sort_keys=True) if args.json else json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
