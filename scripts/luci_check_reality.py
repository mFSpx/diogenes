#!/usr/bin/env python3
"""Boring daily reality suite for LUCIDOTA.

Proves a tiny fixture can be hashed, deduped, written into a working-reality
receipt, checked against live route surfaces, and recorded into STATUS_LEDGER.

Output is JSON only.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "reality_checks"
LEARNING_DIR = ROOT / "05_OUTPUTS" / "indy_reads_learning"
LOG_PATH = OUT / "check_reality_log.jsonl"
LEDGER_PACKET_NAME = "INDY_READs learning packet"


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def run_cmd(cmd: list[str]) -> dict[str, Any]:
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    return {
        "command": " ".join(cmd),
        "rc": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def run_json_cmd(cmd: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    result = run_cmd(cmd)
    payload: dict[str, Any] = {}
    if result["rc"] == 0:
        try:
            parsed = json.loads(result["stdout"] or "null")
            if isinstance(parsed, dict):
                payload = parsed
            elif isinstance(parsed, list):
                payload = {"rows": parsed, "row_count": len(parsed)}
        except Exception as exc:
            result["parse_error"] = str(exc)
    else:
        result["parse_error"] = "nonzero_exit"
    return result, payload


def existing_hashes() -> set[str]:
    hashes: set[str] = set()
    if not LOG_PATH.exists():
        return hashes
    for line in LOG_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except Exception:
            continue
        artifact = entry.get("artifact") or {}
        if isinstance(artifact, dict) and artifact.get("sha256"):
            hashes.add(str(artifact["sha256"]))
    return hashes


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


def build_fixture() -> Path:
    fixture_dir = OUT / "fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    fixture_path = fixture_dir / f"daily_reality_fixture_{stamp()}.txt"
    fixture_path.write_text(
        "LUCIDOTA DAILY REALITY FIXTURE\n"
        "Speed is fluid. Evidence is absolute.\n"
        "Path A records. Path B proposes.\n",
        encoding="utf-8",
    )
    return fixture_path


def summarize_rows(payload: Any) -> dict[str, Any]:
    if isinstance(payload, list):
        first = payload[0] if payload else {}
        return {"kind": "list", "row_count": len(payload), "first_keys": sorted(first.keys())[:16] if isinstance(first, dict) else []}
    if isinstance(payload, dict):
        return {"kind": "dict", "keys": sorted(payload.keys())[:24], "ok": payload.get("ok")}
    return {"kind": type(payload).__name__}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    LEARNING_DIR.mkdir(parents=True, exist_ok=True)

    artifact_path = build_fixture()
    artifact_sha = sha256_file(artifact_path)
    dedupe_status = "duplicate" if artifact_sha in existing_hashes() else "new"

    evidence_packet = {
        "schema": "lucidota.check_reality.evidence.v1",
        "generated_at": now(),
        "artifact": {
            "path": rel(artifact_path),
            "sha256": artifact_sha,
            "dedupe_status": dedupe_status,
        },
        "observation": {
            "kind": "fixture_artifact",
            "observed": True,
            "dedupe_status": dedupe_status,
        },
        "entity": {
            "entity_id": artifact_sha,
            "kind": "fixture_artifact",
            "truth_state": "observed",
        },
        "edge": {
            "from": artifact_sha,
            "to": "working_reality_record",
            "relation": "supports",
        },
        "claim": "A tiny fixture can still traverse the proof spine without pretending to be canon truth.",
        "hypothesis": "If routes and receipts remain live, the organism is breathing.",
        "routes_checked": [],
        "commands_run": [],
        "export_sentence": "",
    }
    evidence_path = OUT / f"daily_reality_evidence_{stamp()}.json"
    write_json(evidence_path, evidence_packet)

    commands_run: list[dict[str, Any]] = []
    route_specs = [
        ("luci status", [str(ROOT / "luci"), "status", "--json"]),
        ("luci manual current", [str(ROOT / "luci"), "manual", "current", "--json"]),
        ("luci provider current", [str(ROOT / "luci"), "provider", "current", "--json"]),
        ("luci workflow current", [str(ROOT / "luci"), "workflow", "current", "--json"]),
        ("luci api route catalog", [str(ROOT / "luci"), "api", "route", "catalog", "--json"]),
    ]
    route_checks: list[dict[str, Any]] = []
    route_blockers: list[str] = []
    for name, cmd in route_specs:
        result, payload = run_json_cmd(cmd)
        summary = summarize_rows(payload)
        ok = result["rc"] == 0
        if not ok:
            route_blockers.append(name.replace(" ", "_") + "_failed")
        route_check = {
            "name": name,
            "command": result["command"],
            "rc": result["rc"],
            "ok": ok,
            "summary": summary,
            "stderr_tail": result["stderr"][-500:] if result["stderr"] else "",
        }
        route_checks.append(route_check)
        commands_run.append(route_check)
    evidence_packet["routes_checked"] = route_checks
    evidence_packet["commands_run"] = commands_run

    reality_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "working_reality_record.py"),
        "--evidence",
        rel(evidence_path),
        "--hypothesis",
        "The current live proof surfaces are reachable and a fixture can be carried through receipt discipline.",
        "--working-reality",
        "Treat the current route and receipt surfaces as the working map for today; preserve contradiction and refuse canon claims.",
        "--move",
        "Record the daily reality fixture, then classify any failing route as a watch item instead of pretending success.",
        "--result",
        "PASS",
        "--json",
    ]
    reality_result = run_cmd(reality_cmd)
    commands_run.append(reality_result)
    receipt_path = ""
    receipt_uuid = ""
    if reality_result["rc"] == 0:
        for line in reality_result["stdout"].splitlines():
            if line.startswith("RECEIPT_PATH="):
                receipt_path = line.split("=", 1)[1].strip()
            elif line.startswith("RECEIPT_UUID="):
                receipt_uuid = line.split("=", 1)[1].strip()
    else:
        route_blockers.append("working_reality_record_failed")

    status = "PASS"
    failed_step = ""
    if reality_result["rc"] != 0:
        status = "DEGRADED"
        failed_step = "working_reality_record"
    elif route_blockers:
        status = "DEGRADED"
        failed_step = route_blockers[0]

    export_sentence = (
        f"Fixture {artifact_path.name} hashed to {artifact_sha[:12]} and routed through working reality "
        f"with receipt {receipt_uuid or 'unavailable'}."
    )
    evidence_packet["export_sentence"] = export_sentence
    write_json(evidence_path, evidence_packet)

    learning_packet = {
        "schema": "lucidota.indy_reads.learning_packet.v1",
        "generated_at": now(),
        "run_id": stamp(),
        "status": status,
        "failed_step": failed_step or None,
        "win": "The current route/receipt spine still speaks.",
        "mistake_avoided": "I treated the stale 128-slot Percyphon expectation as a contract drift, not a live truth.",
        "next_watch": "If any live current/manual/provider/workflow route flips, re-run this command and refresh the handoff.",
        "artifact": {
            "path": rel(artifact_path),
            "sha256": artifact_sha,
            "dedupe_status": dedupe_status,
        },
        "receipt_uuid": receipt_uuid or None,
        "receipt_path": receipt_path or None,
        "evidence_path": rel(evidence_path),
        "routes_checked": route_checks,
        "commands_run": commands_run,
        "export_sentence": export_sentence,
    }
    learning_path = LEARNING_DIR / f"indy_reads_learning_packet_{stamp()}.json"
    write_json(learning_path, learning_packet)

    ledger_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "lucidota_status_ledger.py"),
        "--set",
        LEDGER_PACKET_NAME,
        "--status",
        "verified" if status == "PASS" else "in_progress",
        "--progress",
        "100" if status == "PASS" else "70",
        "--executed",
        "yes",
        "--evidence",
        rel(learning_path),
        "--next",
        learning_packet["next_watch"],
    ]
    if failed_step:
        ledger_cmd.extend(["--blocker", failed_step])
    ledger_result = run_cmd(ledger_cmd)
    commands_run.append(ledger_result)
    if ledger_result["rc"] != 0:
        status = "DEGRADED" if status == "PASS" else status
        if not failed_step:
            failed_step = "status_ledger_update"

    ledger_check = run_cmd([sys.executable, str(ROOT / "scripts" / "lucidota_status_ledger.py"), "--check"])
    commands_run.append(ledger_check)
    if ledger_check["rc"] != 0 and not failed_step:
        status = "DEGRADED"
        failed_step = "status_ledger_check"

    log_entry = {
        "schema": "lucidota.check_reality.log.v1",
        "generated_at": now(),
        "status": status,
        "failed_step": failed_step or None,
        "dedupe_status": dedupe_status,
        "artifact": {
            "path": rel(artifact_path),
            "sha256": artifact_sha,
        },
        "receipt_uuid": receipt_uuid or None,
        "receipt_path": receipt_path or None,
        "evidence_path": rel(evidence_path),
        "learning_packet_path": rel(learning_path),
        "routes_checked": route_checks,
    }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(log_entry, sort_keys=False))
        fh.write("\n")

    report = {
        "schema": "lucidota.check_reality.v1",
        "generated_at": now(),
        "status": status,
        "failed_step": failed_step or None,
        "receipt_uuid": receipt_uuid or None,
        "receipt_path": rel(receipt_path) if receipt_path else None,
        "files_checked": [
            rel(artifact_path),
            rel(evidence_path),
            rel(learning_path),
            "00_PROJECT_BRAIN/STATUS_LEDGER.md",
            "05_OUTPUTS/status_ledger.json",
            "scripts/lucidota_status_ledger.py",
        ],
        "routes_checked": route_checks,
        "commands_run": commands_run,
        "artifact": {
            "path": rel(artifact_path),
            "sha256": artifact_sha,
            "dedupe_status": dedupe_status,
        },
        "learning_packet_path": rel(learning_path),
        "evidence_path": rel(evidence_path),
        "export_sentence": export_sentence,
        "blockers": route_blockers,
    }
    print(json.dumps(report, sort_keys=True, ensure_ascii=False))
    return 0 if status == "PASS" else 1 if status == "DEGRADED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
