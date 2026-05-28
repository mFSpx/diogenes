#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "06_SCHEMA" / "dev_order_matrix_policy.v1.json"
ORIGINAL_DIR = ROOT / "04_RUNTIME" / "dev_orders" / "original"
WRAPPED_DIR = ROOT / "04_RUNTIME" / "dev_orders" / "wrapped"
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


def slugify(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    text = text.strip("._-")
    return text[:120] or "dev_order"


def load_policy() -> tuple[dict[str, Any], bytes, str]:
    data = POLICY_PATH.read_bytes()
    policy = json.loads(data.decode("utf-8"))
    return policy, data, sha256_bytes(data)


def make_read_only(path: Path) -> None:
    try:
        path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    except PermissionError:
        pass


def preserve_original(name: str, original_bytes: bytes) -> Path:
    digest = sha256_bytes(original_bytes)
    ORIGINAL_DIR.mkdir(parents=True, exist_ok=True)
    path = ORIGINAL_DIR / f"{slugify(name)}_{digest}.txt"
    if path.exists():
        if path.read_bytes() != original_bytes:
            raise RuntimeError(f"immutable_original_collision:{rel(path)}")
        make_read_only(path)
        return path
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(original_bytes)
    make_read_only(tmp)
    tmp.replace(path)
    make_read_only(path)
    return path


def methodology_block(policy: dict[str, Any], policy_sha256: str, original_sha256: str) -> str:
    lines = [
        "CURRENT MODE: BUILD + BUGCHASE + GROUP 7 MATRIX ENFORCEMENT",
        "",
        "POLICY:",
        f"- policy_version: {policy['policy_version']}",
        f"- policy_sha256: {policy_sha256}",
        f"- original_sha256: {original_sha256}",
        "- matrix_required: true",
        "- canonical_graph_materialization: false",
        "- canonical_graph_writes: false",
        "",
        "A-H CONSTRAINT DICTIONARY:",
    ]
    for key in sorted(policy["constraint_dictionary"]):
        item = policy["constraint_dictionary"][key]
        lines.append(f"[{key}] {item['name']}: {item['description']}")
    lines += ["", "FORCED TASK ORDERS:"]
    for task_id, task in policy["canonical_task_orders"].items():
        lines.append(f"{task_id} {task['task_name']}: {', '.join(task['constraint_order'])}")
    lines += [
        "",
        "REQUIRED FINAL RECEIPT CONTRACT:",
        "- Include matrix_trace as JSON, not prose.",
        "- Include commands_run, files_changed, receipts_written, blockers, verdict.",
        "- Include canonical_graph_materialization=false and canonical_graph_writes=false unless explicitly allowed by a newer policy.",
        "- PASS is forbidden unless scripts/matrix_trace_checker.py and scripts/dev_order_gate.py pass.",
        "- Every applied constraint must link to at least one changed file, test, command, receipt field, or failure condition.",
        "- Every blocked/not_applicable constraint must include a concrete blocker/reason.",
        "",
        "REFUSAL / BLOCKER CONDITIONS:",
        "- Do not silently rewrite operator intent.",
        "- Do not delete history.",
        "- Do not make markdown authority.",
        "- Do not perform canonical graph materialization.",
        "- Do not write lucidota_go.graph_item, lucidota_go.graph_edge, or lucidota_go.graph_journal.",
        "- Do not claim PASS from prompt self-report.",
    ]
    return "\n".join(lines)


def build_wrapped_text(original_text: str, policy: dict[str, Any], policy_sha256: str, original_sha256: str) -> str:
    return (
        methodology_block(policy, policy_sha256, original_sha256)
        + "\n\n----- ORIGINAL DEV ORDER: BEGIN -----\n"
        + original_text
        + ("" if original_text.endswith("\n") else "\n")
        + "----- ORIGINAL DEV ORDER: END -----\n\n"
        + "OUTPUT CONTRACT:\n"
        + json.dumps(
            {
                "matrix_trace": [
                    {
                        "task_id": "T7",
                        "task_name": "Dev Order Matrix Enforcement",
                        "constraint_order": policy["canonical_task_orders"]["T7"]["constraint_order"],
                        "constraint_trace": [
                            {
                                "constraint": "F",
                                "name": policy["constraint_dictionary"]["F"]["name"],
                                "status": "applied|blocked|not_applicable",
                                "files_changed": [],
                                "tests": [],
                                "commands": [],
                                "receipt_fields": [],
                                "failure_conditions": [],
                                "evidence": "machine-checkable link required",
                                "blocker": None,
                            }
                        ],
                    }
                ]
            },
            indent=2,
        )
        + "\n"
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Wrap dev orders with the LUCIDOTA A-H permuted matrix policy")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input")
    src.add_argument("--stdin", action="store_true")
    parser.add_argument("--name")
    parser.add_argument("--output")
    parser.add_argument("--receipt-out")
    args = parser.parse_args()

    blockers: list[str] = []
    try:
        if args.stdin:
            original_bytes = sys.stdin.buffer.read()
            source_path = f"stdin:{args.name or 'dev_order'}"
            name = args.name or "stdin_dev_order"
        else:
            input_path = Path(args.input)
            if not input_path.is_absolute():
                input_path = ROOT / input_path
            original_bytes = input_path.read_bytes()
            source_path = rel(input_path)
            name = args.name or input_path.stem
        original_sha = sha256_bytes(original_bytes)
        original_text = original_bytes.decode("utf-8")
        policy, policy_bytes, policy_sha = load_policy()
        original_copy = preserve_original(name, original_bytes)
        wrapped_text = build_wrapped_text(original_text, policy, policy_sha, original_sha)
        wrapped_bytes = wrapped_text.encode("utf-8")
        wrapped_path = Path(args.output) if args.output else WRAPPED_DIR / f"{slugify(name)}.wrapped.txt"
        if not wrapped_path.is_absolute():
            wrapped_path = ROOT / wrapped_path
        wrapped_path.parent.mkdir(parents=True, exist_ok=True)
        wrapped_path.write_bytes(wrapped_bytes)
        wrapped_sha = sha256_bytes(wrapped_bytes)
    except Exception as exc:
        source_path = args.input or (f"stdin:{args.name or 'dev_order'}" if args.stdin else "")
        policy_sha = ""
        original_sha = ""
        wrapped_sha = ""
        original_copy = Path("")
        wrapped_path = Path(args.output) if args.output else WRAPPED_DIR / "failed.wrapped.txt"
        blockers.append(str(exc))

    receipt = {
        "schema": "lucidota.dev_order_matrix.wrap_receipt.v1",
        "generated_at": utc_now(),
        "policy_version": (json.loads(POLICY_PATH.read_text(encoding="utf-8")).get("policy_version") if POLICY_PATH.exists() else None),
        "policy_path": rel(POLICY_PATH),
        "policy_sha256": policy_sha,
        "source_input_path": source_path,
        "original_path": rel(original_copy) if str(original_copy) else None,
        "original_sha256": original_sha,
        "wrapped_path": rel(wrapped_path),
        "wrapped_sha256": wrapped_sha,
        "matrix_required": True,
        "canonical_graph_materialization": False,
        "canonical_graph_writes": False,
        "canonical_graph_writes_performed": False,
        "verdict": "PASS" if not blockers else "FAIL",
        "blockers": blockers,
    }
    receipt_path = Path(args.receipt_out) if args.receipt_out else OUT_DIR / f"wrap_{stamp()}.json"
    if not receipt_path.is_absolute():
        receipt_path = ROOT / receipt_path
    receipt["report_path"] = rel(receipt_path)
    write_json(receipt_path, receipt)
    print("REPORT_PATH=" + rel(receipt_path))
    print("DEV_ORDER_MATRIX_WRAP=" + receipt["verdict"])
    return 0 if receipt["verdict"] == "PASS" else 4


if __name__ == "__main__":
    raise SystemExit(main())
