#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "06_SCHEMA" / "dev_order_matrix_policy.v1.json"
OUT_DIR = ROOT / "05_OUTPUTS" / "dev_order_matrix"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_policy(path: Path = POLICY_PATH) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    return json.loads(raw.decode("utf-8")), sha256_bytes(raw)


def load_receipt(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("receipt_json_not_object")
    return data


def nonempty_list(value: Any) -> bool:
    return isinstance(value, list) and any(str(item).strip() for item in value)


def concrete_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def task_by_id(matrix_trace: list[Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for item in matrix_trace:
        if isinstance(item, dict) and item.get("task_id"):
            out[str(item["task_id"])] = item
    return out


def validate_matrix_trace(receipt: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    missing_tasks: list[str] = []
    bad_orders: list[dict[str, Any]] = []
    missing_constraints: list[dict[str, Any]] = []
    constraints_without_evidence: list[dict[str, Any]] = []
    blocked_without_reason: list[dict[str, Any]] = []
    unknown_constraints: list[dict[str, Any]] = []
    duplicate_constraints: list[dict[str, Any]] = []
    t7_integration_blockers: list[str] = []

    trace = receipt.get("matrix_trace")
    if not isinstance(trace, list):
        blockers.append("matrix_trace_missing_or_not_array")
        if isinstance(trace, str):
            blockers.append("matrix_trace_is_prose_only")
        return {
            "blockers": blockers,
            "missing_tasks": list(policy["canonical_task_orders"].keys()),
            "bad_orders": bad_orders,
            "missing_constraints": missing_constraints,
            "constraints_without_evidence": constraints_without_evidence,
            "blocked_without_reason": blocked_without_reason,
            "unknown_constraints": unknown_constraints,
            "duplicate_constraints": duplicate_constraints,
            "t7_integration_blockers": t7_integration_blockers,
        }

    tasks = task_by_id(trace)
    if len(tasks) != len(trace):
        blockers.append("duplicate_or_malformed_task_ids")
    expected_constraints = set(policy["constraint_dictionary"].keys())
    evidence_fields = policy["matrix_trace_schema"]["evidence_link_fields"]
    allowed_statuses = set(policy["matrix_trace_schema"]["allowed_statuses"])

    for task_id, spec in policy["canonical_task_orders"].items():
        task = tasks.get(task_id)
        if task is None:
            missing_tasks.append(task_id)
            continue
        actual_order = task.get("constraint_order")
        expected_order = spec["constraint_order"]
        if actual_order != expected_order:
            bad_orders.append({"task_id": task_id, "expected": expected_order, "actual": actual_order})
        trace_rows = task.get("constraint_trace")
        if not isinstance(trace_rows, list):
            missing_constraints.append({"task_id": task_id, "missing": sorted(expected_constraints), "reason": "constraint_trace_not_array"})
            continue
        letters = [row.get("constraint") for row in trace_rows if isinstance(row, dict)]
        if letters != expected_order:
            bad_orders.append({"task_id": task_id, "expected_trace_order": expected_order, "actual_trace_order": letters})
        for letter in letters:
            if letter not in expected_constraints:
                unknown_constraints.append({"task_id": task_id, "constraint": letter})
        for letter in sorted(expected_constraints - set(letters)):
            missing_constraints.append({"task_id": task_id, "missing": letter})
        for letter in sorted({x for x in letters if letters.count(x) > 1}):
            duplicate_constraints.append({"task_id": task_id, "constraint": letter})
        for row in trace_rows:
            if not isinstance(row, dict):
                blockers.append(f"constraint_trace_row_not_object:{task_id}")
                continue
            letter = row.get("constraint")
            status = row.get("status")
            if status not in allowed_statuses:
                blockers.append(f"invalid_constraint_status:{task_id}:{letter}:{status}")
                continue
            if status == "applied":
                if not any(nonempty_list(row.get(field)) for field in evidence_fields):
                    constraints_without_evidence.append({"task_id": task_id, "constraint": letter})
            else:
                if not (concrete_text(row.get("blocker")) or concrete_text(row.get("evidence"))):
                    blocked_without_reason.append({"task_id": task_id, "constraint": letter, "status": status})

    extra_tasks = sorted(set(tasks) - set(policy["canonical_task_orders"]))
    if extra_tasks:
        blockers.append("unknown_task_ids:" + ",".join(extra_tasks))

    t7 = tasks.get("T7")
    if t7:
        linked: set[str] = set()
        for row in t7.get("constraint_trace") or []:
            if not isinstance(row, dict):
                continue
            for field in ("files_changed", "tests", "commands", "receipt_fields", "failure_conditions"):
                for item in row.get(field) or []:
                    linked.add(str(item))
        for required_path in policy.get("t7_required_paths") or []:
            exists = (ROOT / required_path).exists()
            mentioned = any(required_path in item for item in linked)
            if not exists:
                t7_integration_blockers.append(f"t7_required_path_missing:{required_path}")
            if not mentioned:
                t7_integration_blockers.append(f"t7_required_path_not_referenced:{required_path}")

    blockers.extend(["missing_task:" + task_id for task_id in missing_tasks])
    blockers.extend(["bad_constraint_order:" + item["task_id"] for item in bad_orders])
    blockers.extend(["missing_constraint:" + item["task_id"] + ":" + str(item["missing"]) for item in missing_constraints])
    blockers.extend(["unknown_constraint:" + item["task_id"] + ":" + str(item["constraint"]) for item in unknown_constraints])
    blockers.extend(["duplicate_constraint:" + item["task_id"] + ":" + str(item["constraint"]) for item in duplicate_constraints])
    blockers.extend(["constraint_without_evidence:" + item["task_id"] + ":" + str(item["constraint"]) for item in constraints_without_evidence])
    blockers.extend(["blocked_without_reason:" + item["task_id"] + ":" + str(item["constraint"]) for item in blocked_without_reason])
    blockers.extend(t7_integration_blockers)
    if receipt.get("verdict") == "PASS" and blockers:
        blockers.append("final_pass_with_incomplete_matrix_trace")
    return {
        "blockers": blockers,
        "missing_tasks": missing_tasks,
        "bad_orders": bad_orders,
        "missing_constraints": missing_constraints,
        "constraints_without_evidence": constraints_without_evidence,
        "blocked_without_reason": blocked_without_reason,
        "unknown_constraints": unknown_constraints,
        "duplicate_constraints": duplicate_constraints,
        "t7_integration_blockers": t7_integration_blockers,
    }


def check_receipt(receipt_path: Path) -> dict[str, Any]:
    policy, policy_sha = load_policy()
    blockers: list[str] = []
    receipt: dict[str, Any] = {}
    try:
        receipt = load_receipt(receipt_path)
    except Exception as exc:
        blockers.append(f"receipt_unreadable:{exc}")
    detail = validate_matrix_trace(receipt, policy) if receipt else {
        "blockers": ["receipt_missing"],
        "missing_tasks": list(policy["canonical_task_orders"].keys()),
        "bad_orders": [],
        "missing_constraints": [],
        "constraints_without_evidence": [],
        "blocked_without_reason": [],
        "unknown_constraints": [],
        "duplicate_constraints": [],
        "t7_integration_blockers": [],
    }
    blockers.extend(detail["blockers"])
    final_pass_allowed = not blockers
    return {
        "schema": "lucidota.dev_order_matrix.check_receipt.v1",
        "generated_at": utc_now(),
        "input_receipt": rel(receipt_path),
        "policy_version": policy.get("policy_version"),
        "policy_sha256": policy_sha,
        "verdict": "PASS" if final_pass_allowed else "FAIL",
        "missing_tasks": detail["missing_tasks"],
        "bad_orders": detail["bad_orders"],
        "missing_constraints": detail["missing_constraints"],
        "unknown_constraints": detail["unknown_constraints"],
        "duplicate_constraints": detail["duplicate_constraints"],
        "constraints_without_evidence": detail["constraints_without_evidence"],
        "blocked_without_reason": detail["blocked_without_reason"],
        "t7_integration_blockers": detail["t7_integration_blockers"],
        "final_pass_allowed": final_pass_allowed,
        "blockers": blockers,
        "canonical_graph_materialization": False,
        "canonical_graph_writes": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate LUCIDOTA A-H matrix_trace receipts")
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    receipt_path = Path(args.receipt)
    if not receipt_path.is_absolute():
        receipt_path = ROOT / receipt_path
    payload = check_receipt(receipt_path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.output) if args.output else OUT_DIR / f"matrix_trace_check_{stamp()}.json"
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    payload["report_path"] = rel(out)
    out.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
    print("REPORT_PATH=" + rel(out))
    print("MATRIX_TRACE_CHECK=" + payload["verdict"])
    return 0 if payload["verdict"] == "PASS" else 4


if __name__ == "__main__":
    raise SystemExit(main())
