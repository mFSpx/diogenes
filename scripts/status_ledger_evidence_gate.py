#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "status_ledger"
DEFAULT_LEDGER = ROOT / "05_OUTPUTS" / "status_ledger.json"
RECOGNIZED_SCHEMAS = {
    "lucidota.golden_path.single_instruction.v1",
    "lucidota.golden_path.regression_gate.v1",
    "lucidota.same_lineage_validator.v1",
    "lucidota.canonical_graph_write_scanner.v1",
    "lucidota.spine_authority_check.v1",
    "lucidota.golden_path_hardening.v1",
    "lucidota.instruction_hygiene_receipt.v1",
    "lucidota.status_ledger_evidence_gate.v1",
    "lucidota.proof_kernel.receipt.v1",
    "lucidota.slop_audit_law.v1",
}
ORACLE_SCHEMAS = {
    "lucidota.golden_path.single_instruction.v1",
    "lucidota.golden_path.regression_gate.v1",
    "lucidota.same_lineage_validator.v1",
    "lucidota.golden_path_hardening.v1",
    "lucidota.status_ledger_evidence_gate.v1",
}
BROAD_MYTH_RE = re.compile(r"\b(whole|entire|all\s+of|complete|completed|finished|100%|production\s+ready|system\s+complete|crowned\s+whole)\b", re.I)
SCOPED_OK_RE = re.compile(r"\b(golden path|single[- ]instruction|no[- ]materialization|regression gate|same[- ]lineage|canonical graph scanner|spine authority|proof kernel|status ledger|graph write fence)\b", re.I)
PASS_STATUSES = {"PASS", "REAL", "PASS_DRY_RUN", "VERIFIED", "SIGNED_OFF"}
UPGRADE_STATUSES = {"verified", "executed", "dry_run_passed", "ready", "complete"}
PARTIAL_STATUSES = {"blocked", "in_progress", "scaffolded"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def read_json(path: Path, blockers: list[str], prefix: str = "evidence") -> dict[str, Any]:
    if not path.exists():
        blockers.append(f"{prefix}_path_missing")
        return {}
    if not path.is_file():
        blockers.append(f"{prefix}_not_file")
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            blockers.append(f"{prefix}_not_object")
            return {}
        return data
    except Exception as exc:
        blockers.append(f"{prefix}_unreadable:{exc}")
        return {}


def receipt_status(data: dict[str, Any]) -> str | None:
    for key in ("verdict", "status", "readiness_status"):
        value = data.get(key)
        if isinstance(value, str):
            return value
    if data.get("release_ready") is True and not data.get("blockers"):
        return "PASS"
    return None


def status_supported(data: dict[str, Any], status: str) -> bool:
    verdict = receipt_status(data)
    if verdict in PASS_STATUSES and status in {"verified", "executed", "dry_run_passed", "ready", "complete"}:
        return True
    if verdict == "FAIL" and status in {"blocked", "in_progress"}:
        return True
    return False


def is_critical_claim(claim: str, status: str) -> bool:
    return status in UPGRADE_STATUSES and (SCOPED_OK_RE.search(claim) is not None or BROAD_MYTH_RE.search(claim) is not None)


def validate(evidence: str, claim: str, proposed_status: str, *, blocker: str = "", next_action: str = "", oracle_receipt: str = "") -> dict[str, Any]:
    blockers: list[str] = []
    path = ROOT / evidence if not Path(evidence).is_absolute() else Path(evidence)
    data = read_json(path, blockers, "evidence")
    schema = data.get("schema")
    if data and schema not in RECOGNIZED_SCHEMAS:
        blockers.append(f"unrecognized_receipt_schema:{schema}")
    if data and (data.get("blockers") or data.get("non_spine_blockers")):
        blockers.append("receipt_has_unresolved_blockers")
    if proposed_status in UPGRADE_STATUSES:
        if BROAD_MYTH_RE.search(claim) and not SCOPED_OK_RE.search(claim):
            blockers.append("claim_is_global_mythology_not_scoped")
        if not SCOPED_OK_RE.search(claim):
            blockers.append("claim_not_scoped_to_known_evidence_lane")
        if data and not status_supported(data, proposed_status):
            blockers.append("receipt_verdict_does_not_support_status")
        if is_critical_claim(claim, proposed_status) and schema not in ORACLE_SCHEMAS and not oracle_receipt:
            blockers.append("critical_status_upgrade_requires_oracle_or_audit_receipt")
    if proposed_status in PARTIAL_STATUSES:
        if proposed_status == "blocked" and not blocker:
            blockers.append("blocked_status_requires_blocker")
        if not next_action:
            blockers.append("partial_status_requires_next_action")
    return {
        "schema": "lucidota.status_ledger_evidence_gate.v1",
        "generated_at": now(),
        "claim": claim,
        "proposed_status": proposed_status,
        "evidence": rel(path),
        "evidence_schema": schema,
        "oracle_receipt": oracle_receipt or None,
        "blockers": blockers,
        "canonical_graph_materialization": False,
        "canonical_graph_writes_performed": False,
        "verdict": "PASS" if not blockers else "FAIL",
    }


def verify_command_syntax(command: str) -> bool:
    return bool(command and command.strip())


def validate_claim_record(record: dict[str, Any], idx: int) -> list[str]:
    blockers: list[str] = []
    name = str(record.get("name") or f"claim[{idx}]")
    status = str(record.get("scoped_status") or record.get("status") or "")
    evidence_path = record.get("evidence_path")
    evidence_type = record.get("evidence_type")
    verification_command = record.get("verification_command")
    last_verified_time = record.get("last_verified_time")
    claim_text = str(record.get("claim") or name)
    blocker = str(record.get("blocker") or "")
    next_action = str(record.get("next_action") or "")
    if not name:
        blockers.append(f"claim_record_missing_name:{idx}")
    if not status:
        blockers.append(f"claim_record_missing_status:{name}")
    if not evidence_type:
        blockers.append(f"claim_record_missing_evidence_type:{name}")
    if not verify_command_syntax(str(verification_command or "")):
        blockers.append(f"claim_record_missing_verification_command:{name}")
    if not last_verified_time:
        blockers.append(f"claim_record_missing_last_verified_time:{name}")
    if status in UPGRADE_STATUSES:
        if not evidence_path:
            blockers.append(f"claim_record_missing_evidence_path:{name}")
        else:
            p = ROOT / str(evidence_path) if not Path(str(evidence_path)).is_absolute() else Path(str(evidence_path))
            data = read_json(p, blockers, f"claim_record_evidence:{name}")
            if data and not status_supported(data, "verified"):
                blockers.append(f"claim_record_evidence_not_pass:{name}")
            schema = data.get("schema") if data else None
            if is_critical_claim(claim_text, status) and schema not in ORACLE_SCHEMAS and not record.get("oracle_receipt_path"):
                blockers.append(f"claim_record_critical_upgrade_missing_oracle:{name}")
    if status in PARTIAL_STATUSES:
        if status == "blocked" and not blocker:
            blockers.append(f"claim_record_blocked_missing_blocker:{name}")
        if not next_action:
            blockers.append(f"claim_record_partial_missing_next_action:{name}")
    return blockers


def audit_ledger(ledger_path: str | None = None) -> dict[str, Any]:
    blockers: list[str] = []
    path = Path(ledger_path) if ledger_path else DEFAULT_LEDGER
    if not path.is_absolute():
        path = ROOT / path
    ledger = read_json(path, blockers, "ledger")
    claims = ledger.get("status_claims") if isinstance(ledger, dict) else None
    if ledger.get("markdown_claims") and not claims:
        blockers.append("markdown_only_claims_are_not_authority")
    if claims is None:
        claims = []
    if not isinstance(claims, list):
        blockers.append("status_claims_not_list")
        claims = []
    for idx, claim in enumerate(claims):
        if not isinstance(claim, dict):
            blockers.append(f"claim_record_not_object:{idx}")
            continue
        blockers.extend(validate_claim_record(claim, idx))
    return {
        "schema": "lucidota.status_ledger_evidence_gate.v1",
        "generated_at": now(),
        "action": "audit_ledger",
        "ledger_path": rel(path),
        "claims_checked": len(claims),
        "blockers": blockers,
        "canonical_graph_materialization": False,
        "canonical_graph_writes_performed": False,
        "verdict": "PASS" if not blockers else "FAIL",
    }


def write_payload(payload: dict[str, Any], output: str | None) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    out = Path(output) if output else OUT / f"status_ledger_evidence_gate_{stamp()}.json"
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    payload["report_path"] = rel(out)
    out.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
    print("REPORT_PATH=" + rel(out))
    print("STATUS_LEDGER_EVIDENCE_GATE=" + payload["verdict"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Machine-check status ledger evidence claims")
    ap.add_argument("--evidence")
    ap.add_argument("--claim")
    ap.add_argument("--proposed-status", choices=["verified", "executed", "dry_run_passed", "in_progress", "blocked", "scaffolded", "ready", "complete"])
    ap.add_argument("--blocker", default="")
    ap.add_argument("--next-action", default="")
    ap.add_argument("--oracle-receipt", default="")
    ap.add_argument("--audit-ledger", nargs="?", const=str(DEFAULT_LEDGER), default=None)
    ap.add_argument("--output")
    args = ap.parse_args()
    if args.audit_ledger is not None:
        payload = audit_ledger(args.audit_ledger)
    else:
        missing = [name for name in ("evidence", "claim", "proposed_status") if getattr(args, name) is None]
        if missing:
            ap.error("missing required arguments for evidence validation: " + ", ".join("--" + m.replace("_", "-") for m in missing))
        payload = validate(args.evidence, args.claim, args.proposed_status, blocker=args.blocker, next_action=args.next_action, oracle_receipt=args.oracle_receipt)
    write_payload(payload, args.output)
    return 0 if payload["verdict"] == "PASS" else 4


if __name__ == "__main__":
    raise SystemExit(main())
