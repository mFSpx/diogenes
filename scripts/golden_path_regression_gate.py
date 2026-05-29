#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "golden_path_regression"
DEFAULT_RECEIPT = ROOT / "tests" / "fixtures" / "golden_path" / "valid_receipt_bundle.json"
BUNDLE_SCHEMA = "lucidota.lineage.receipt_bundle.v1"
EXPECTED_TYPES = ["operator_instruction", "conversation_command", "absurd_queue", "worker_execution", "worker_receipt", "graph_staging", "audit"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def run(cmd: list[str]) -> dict[str, Any]:
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    refs = []
    for line in (proc.stdout + "\n" + proc.stderr).splitlines():
        if line.startswith("REPORT_PATH="):
            refs.append(line.split("=", 1)[1].strip())
    return {
        "command": " ".join(cmd),
        "rc": proc.returncode,
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
        "report_paths": refs,
    }


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def evidence_paths(data: dict[str, Any]) -> list[str]:
    paths: set[str] = set()
    for ev in data.get("evidence_refs") or []:
        if isinstance(ev, dict) and ev.get("path"):
            paths.add(str(ev["path"]))
    for receipt in data.get("receipts") or []:
        if isinstance(receipt, dict):
            if receipt.get("receipt_path"):
                paths.add(str(receipt["receipt_path"]))
            for ev in receipt.get("evidence_refs") or []:
                if isinstance(ev, dict) and ev.get("path"):
                    paths.add(str(ev["path"]))
    return sorted(paths)


def required_bundle_steps(data: dict[str, Any]) -> dict[str, bool]:
    types = [r.get("receipt_type") for r in data.get("receipts") or [] if isinstance(r, dict)]
    return {f"lineage_step:{name}": name in types for name in EXPECTED_TYPES}


def check_contract(data: dict[str, Any]) -> tuple[list[str], dict[str, bool], dict[str, Any]]:
    blockers: list[str] = []
    if data.get("schema") != BUNDLE_SCHEMA:
        return ["unsupported_receipt_contract_schema"], {}, {}
    for key in ("lineage_id", "command_uuid", "idempotency_key", "job_uuid", "worker_receipt_uuid", "audit_uuid"):
        if not data.get(key):
            blockers.append(f"missing_{key}")
    steps = required_bundle_steps(data)
    blockers.extend([f"missing_required_step:{k}" for k, ok in steps.items() if not ok])
    if data.get("materialization_performed") is True or data.get("canonical_graph_writes_performed") is True:
        blockers.append("canonical_graph_materialization_performed")
    counts = data.get("canonical_graph_counts") or {}
    before = counts.get("before")
    after = counts.get("after")
    if before is None or after is None:
        blockers.append("missing_canonical_graph_counts")
    elif before != after:
        blockers.append("canonical_graph_counts_changed_fixture")
    return blockers, steps, counts


def main() -> int:
    ap = argparse.ArgumentParser(description="Golden path regression gate: fixed receipt contract, same lineage, no canonical graph mutation")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    ap.add_argument("--no-materialize", action="store_true")
    ap.add_argument("--receipt", default=str(DEFAULT_RECEIPT))
    args = ap.parse_args()

    blockers: list[str] = []
    commands_run: list[dict[str, Any]] = []
    receipts_created: list[str] = []
    files_read: list[str] = []
    if args.execute and not args.no_materialize:
        blockers.append("execute_requires_no_materialize_for_this_gate")

    receipt_path = ROOT / args.receipt if not Path(args.receipt).is_absolute() else Path(args.receipt)
    files_read.append(rel(receipt_path))
    try:
        agg = load(receipt_path)
    except Exception as exc:
        agg = {}
        blockers.append(f"aggregate_unreadable:{exc}")

    contract_blockers, steps, graph_counts = check_contract(agg) if agg else (["aggregate_missing"], {}, {})
    blockers.extend(contract_blockers)

    lineage_cmd = [sys.executable, "scripts/same_lineage_validator.py", "--receipt", rel(receipt_path)]
    lineage_run = run(lineage_cmd)
    commands_run.append(lineage_run)
    receipts_created.extend(lineage_run["report_paths"])
    if lineage_run["rc"] != 0:
        blockers.append("same_lineage_validator_failed")

    ledger_run = run([sys.executable, "scripts/status_ledger_evidence_gate.py", "--audit-ledger"])
    commands_run.append(ledger_run)
    receipts_created.extend(ledger_run["report_paths"])
    if ledger_run["rc"] != 0:
        blockers.append("status_ledger_evidence_audit_failed")

    barrier_run = run([sys.executable, "scripts/graph_canonical_mutation_detector.py"])
    commands_run.append(barrier_run)
    receipts_created.extend(barrier_run["report_paths"])
    if barrier_run["rc"] != 0:
        blockers.append("canonical_mutation_detector_failed")
    for rp in barrier_run["report_paths"]:
        p = ROOT / rp
        if p.exists():
            files_read.append(rel(p))
            try:
                bd = load(p)
                unchanged = bd.get("unchanged")
                if unchanged is None:
                    unchanged = bd.get("before_counts") == bd.get("after_counts")
                if unchanged is not True:
                    blockers.append("canonical_graph_counts_changed_current_detector")
            except Exception as exc:
                blockers.append(f"canonical_detector_report_unreadable:{exc}")

    failing_invariant = blockers[0] if blockers else None
    payload = {
        "schema": "lucidota.golden_path.regression_gate.v1",
        "generated_at": now(),
        "verdict": "PASS" if not blockers else "FAIL",
        "mode": "execute_no_materialize" if args.execute else "dry_run",
        "receipt_contract": BUNDLE_SCHEMA,
        "lineage_id": agg.get("lineage_id"),
        "command_uuid": agg.get("command_uuid"),
        "idempotency_key": agg.get("idempotency_key"),
        "job_uuid": agg.get("job_uuid"),
        "worker_receipt_uuid": agg.get("worker_receipt_uuid"),
        "audit_uuid": agg.get("audit_uuid"),
        "same_lineage": "same_lineage_validator_failed" not in blockers,
        "canonical_graph_materialization": False,
        "canonical_graph_writes_performed": False,
        "canonical_graph_counts": graph_counts,
        "steps": steps,
        "blockers": blockers,
        "failing_invariant": failing_invariant,
        "commands_run": commands_run,
        "files_read": sorted(set(files_read)),
        "evidence_paths": evidence_paths(agg),
        "receipts_created": sorted(set(receipts_created)),
        "files_touched": [
            "scripts/golden_path_regression_gate.py",
            "scripts/same_lineage_validator.py",
            "tests/fixtures/golden_path/valid_receipt_bundle.json",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"golden_path_regression_{stamp()}.json"
    payload["report_path"] = rel(out)
    out.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
    print("REPORT_PATH=" + rel(out))
    print("GOLDEN_PATH_REGRESSION_GATE=" + payload["verdict"])
    if failing_invariant:
        print("FAILING_INVARIANT=" + failing_invariant)
    return 0 if payload["verdict"] == "PASS" else 4


if __name__ == "__main__":
    raise SystemExit(main())
