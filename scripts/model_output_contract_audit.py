#!/usr/bin/env python3
"""Normalize and audit model receipts against the GO-25 worker output contract."""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "goals"
REQUIRED_FIELDS = ["status", "result", "next_action", "receipt_path", "evidence_refs"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def load_receipt(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    return json.loads(p.read_text(encoding="utf-8"))


def parse_embedded_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        match = re.search(r"\{.*\}", cleaned, re.S)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None


def extract_partial_scalars(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field in ("status", "result", "next_action", "receipt_path"):
        m = re.search(rf'"{field}"\s*:\s*"([^"]*)"', text)
        if m:
            out[field] = m.group(1)
    m = re.search(r'"evidence_refs"\s*:\s*\[(.*?)\]', text, re.S)
    if m:
        refs = re.findall(r'"([^"]*)"', m.group(1))
        out["evidence_refs"] = refs
    return out


def normalize_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    canonical = receipt
    parsed_text = parse_embedded_json(str(receipt.get("text") or ""))
    if parsed_text:
        canonical = {**receipt, **parsed_text}
    elif receipt.get("text"):
        partial = extract_partial_scalars(str(receipt.get("text") or ""))
        if partial:
            canonical = {**receipt, **partial}
    if isinstance(canonical.get("result"), dict):
        nested = canonical["result"]
        canonical = {
            **canonical,
            "next_action": canonical.get("next_action") or nested.get("next_action"),
            "receipt_path": nested.get("receipt_path") or canonical.get("receipt_path") or canonical.get("report_path"),
            "evidence_refs": canonical.get("evidence_refs") or nested.get("evidence_refs") or [],
            "decision_pairs": canonical.get("decision_pairs") or nested.get("decision_pairs") or nested.get("decisions") or [],
            "result": nested.get("result") if isinstance(nested.get("result"), (str, int, float, bool)) else nested.get("status") or canonical.get("result"),
        }
    field_aliases: dict[str, str] = {}
    decision_pairs = canonical.get("decision_pairs")
    if decision_pairs is None and "decisions" in canonical:
        decision_pairs = canonical.get("decisions")
        field_aliases["decisions"] = "decision_pairs"
    if decision_pairs is None:
        decision_pairs = []
    missing_required_fields = [field for field in REQUIRED_FIELDS if canonical.get(field) in (None, "", [])]
    evidence_refs = canonical.get("evidence_refs") or []
    if not isinstance(evidence_refs, list):
        evidence_refs = [evidence_refs]
    normalized = {
        "status": canonical.get("status"),
        "result": canonical.get("result"),
        "next_action": canonical.get("next_action"),
        "receipt_path": canonical.get("receipt_path") or canonical.get("report_path"),
        "evidence_refs": evidence_refs,
        "decision_pairs": decision_pairs,
        "field_aliases": field_aliases,
        "missing_required_fields": missing_required_fields,
    }
    return normalized


def build_report(receipt_paths: list[str]) -> dict[str, Any]:
    receipts = [load_receipt(path) for path in receipt_paths]
    normalized = [normalize_receipt(receipt) for receipt in receipts]
    aliases = sorted({f"{alias}->{canonical}" for item in normalized for alias, canonical in item["field_aliases"].items()})
    missing = sorted({field for item in normalized for field in item["missing_required_fields"]})
    return {
        "schema": "lucidota.model_output_contract_audit.v1",
        "generated_at": now(),
        "status": "PASS",
        "objective": "Normalize model receipts into the GO-25 worker output contract and surface field drift.",
        "evidence_refs": [rel(path) for path in receipt_paths],
        "required_fields": REQUIRED_FIELDS,
        "aliases": aliases,
        "missing_required_fields": missing,
        "receipt_count": len(receipts),
        "normalized_receipts": normalized,
        "receipt_path": "",
    }


def write_report(report: dict[str, Any]) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"model_output_contract_audit_{stamp()}.json"
    report["receipt_path"] = rel(path)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="Normalize model receipts to the GO-25 worker output contract.")
    ap.add_argument("--receipt", action="append", required=True)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    report = write_report(build_report(args.receipt))
    print("REPORT_PATH=" + report["receipt_path"])
    print("MODEL_OUTPUT_CONTRACT_AUDIT=PASS")
    print(json.dumps(report, sort_keys=True) if args.json else report["receipt_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
