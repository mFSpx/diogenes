#!/usr/bin/env python3
"""Validate one receipt lineage by hash-linked parent/child receipts.

Accepted primary input: lucidota.lineage.receipt_bundle.v1.  A bundle is a
compact, deterministic proof that an operator instruction moved through the
command, queue, worker, worker receipt, graph-staging, and audit lanes without
loose report-collage references.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "golden_path"
UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
BUNDLE_SCHEMA = "lucidota.lineage.receipt_bundle.v1"
EXPECTED_TYPES = [
    "operator_instruction",
    "conversation_command",
    "absurd_queue",
    "worker_execution",
    "worker_receipt",
    "graph_staging",
    "audit",
]
REQUIRED_TOP_LEVEL_IDS = [
    "lineage_id",
    "command_uuid",
    "idempotency_key",
    "job_uuid",
    "worker_receipt_uuid",
    "audit_uuid",
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def receipt_hash(receipt: dict[str, Any]) -> str:
    stripped = copy.deepcopy(receipt)
    stripped.pop("receipt_hash", None)
    return sha256_text(canonical_json(stripped))


def registry_hash(data: dict[str, Any]) -> str:
    stripped = copy.deepcopy(data)
    stripped.pop("bundle_hash", None)
    return sha256_text(canonical_json(stripped))


def parse_ts(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def load_json(path: str | Path, files_read: list[str], blockers: list[str]) -> dict[str, Any]:
    p = ROOT / path if not Path(path).is_absolute() else Path(path)
    files_read.append(rel(p))
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            blockers.append(f"receipt_not_object:{rel(p)}")
            return {}
        return data
    except Exception as exc:
        blockers.append(f"receipt_unreadable:{rel(p)}:{exc}")
        return {}


def require_uuid(name: str, value: Any, blockers: list[str]) -> None:
    if not isinstance(value, str) or not UUID_RE.match(value):
        blockers.append(f"invalid_or_missing_{name}")
        return
    try:
        uuid.UUID(value)
    except Exception:
        blockers.append(f"invalid_or_missing_{name}")


def evidence_objects(receipt: dict[str, Any], blockers: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, ev in enumerate(receipt.get("evidence_refs") or []):
        if not isinstance(ev, dict):
            blockers.append(f"loose_evidence_ref_without_hash:{receipt.get('receipt_type')}:{idx}")
            continue
        path = ev.get("path")
        ev_hash = ev.get("receipt_hash") or ev.get("sha256")
        if not path:
            blockers.append(f"evidence_ref_missing_path:{receipt.get('receipt_type')}:{idx}")
        if not ev_hash:
            blockers.append(f"evidence_ref_missing_hash:{receipt.get('receipt_type')}:{idx}")
        out.append(ev)
    return out


def validate_bundle(bundle: dict[str, Any], *, receipt_label: str) -> dict[str, Any]:
    blockers: list[str] = []
    failing_edge: str | None = None
    files_read: list[str] = []

    if bundle.get("schema") != BUNDLE_SCHEMA:
        blockers.append("missing_receipt_bundle_hash_chain")
        failing_edge = "bundle.schema"

    for key in REQUIRED_TOP_LEVEL_IDS:
        if key not in bundle or bundle.get(key) in (None, ""):
            blockers.append(f"missing_{key}")
    for key in ("lineage_id", "command_uuid", "job_uuid", "worker_receipt_uuid", "audit_uuid"):
        if key in bundle and bundle.get(key):
            require_uuid(key, bundle.get(key), blockers)
    if not isinstance(bundle.get("idempotency_key"), str) or not bundle.get("idempotency_key"):
        blockers.append("invalid_or_missing_idempotency_key")

    receipts = bundle.get("receipts")
    if not isinstance(receipts, list) or not receipts:
        blockers.append("missing_receipts")
        receipts = []
    types = [r.get("receipt_type") for r in receipts if isinstance(r, dict)]
    if types != EXPECTED_TYPES:
        blockers.append("receipt_type_order_mismatch")
        failing_edge = failing_edge or "receipts.order"

    top_id_locations = {key: [] for key in REQUIRED_TOP_LEVEL_IDS}
    receipt_uuids: list[str] = []
    receipt_hashes: list[str] = []
    previous_hash: str | None = None
    previous_ts: datetime | None = None
    audit_ts: datetime | None = None
    known_evidence: dict[str, dict[str, Any]] = {}
    audit_refs: list[dict[str, Any]] = []

    for idx, receipt in enumerate(receipts):
        if not isinstance(receipt, dict):
            blockers.append(f"receipt_not_object_at:{idx}")
            continue
        rtype = str(receipt.get("receipt_type") or f"idx_{idx}")
        r_uuid = receipt.get("receipt_uuid")
        if not r_uuid:
            blockers.append(f"missing_receipt_uuid:{rtype}")
        else:
            require_uuid("receipt_uuid", r_uuid, blockers)
            receipt_uuids.append(str(r_uuid))

        ts = parse_ts(receipt.get("created_at"))
        if ts is None:
            blockers.append(f"missing_or_invalid_timestamp:{rtype}")
        elif previous_ts is not None and ts.timestamp() + 5 < previous_ts.timestamp():
            blockers.append(f"timestamp_non_monotonic:{types[idx-1] if idx else 'start'}->{rtype}")
            failing_edge = failing_edge or f"{types[idx-1] if idx else 'start'}->{rtype}"
        if ts is not None:
            previous_ts = ts
        if rtype == "audit":
            audit_ts = ts

        actual_hash = receipt_hash(receipt)
        declared_hash = receipt.get("receipt_hash")
        if declared_hash != actual_hash:
            blockers.append(f"receipt_hash_mismatch:{rtype}")
            failing_edge = failing_edge or f"{rtype}.receipt_hash"
        receipt_hashes.append(str(declared_hash or actual_hash))

        parent = receipt.get("parent_receipt_hash")
        if idx == 0:
            if parent not in (None, ""):
                blockers.append("first_receipt_parent_hash_must_be_null")
                failing_edge = failing_edge or "start->operator_instruction"
        elif parent != previous_hash:
            blockers.append(f"parent_receipt_hash_mismatch:{types[idx-1] if idx else 'start'}->{rtype}")
            failing_edge = failing_edge or f"{types[idx-1] if idx else 'start'}->{rtype}"
        previous_hash = str(declared_hash or actual_hash)

        for key in REQUIRED_TOP_LEVEL_IDS:
            if receipt.get(key):
                top_id_locations[key].append(rtype)
                if bundle.get(key) and receipt.get(key) != bundle.get(key):
                    blockers.append(f"{key}_regenerated_or_mismatched:{rtype}")
                    failing_edge = failing_edge or f"{rtype}.{key}"

        if receipt.get("receipt_path"):
            known_evidence[str(receipt["receipt_path"])] = {
                "path": str(receipt["receipt_path"]),
                "receipt_hash": str(declared_hash or actual_hash),
                "created_at": receipt.get("created_at"),
                "lineage_id": bundle.get("lineage_id"),
            }
        evs = evidence_objects(receipt, blockers)
        if rtype == "audit":
            audit_refs = evs
        else:
            for ev in evs:
                if ev.get("path"):
                    known_evidence[str(ev["path"])] = ev

    duplicated_receipt_ids = sorted({x for x in receipt_uuids if receipt_uuids.count(x) > 1})
    if duplicated_receipt_ids:
        blockers.append("duplicate_receipt_uuid:" + ",".join(duplicated_receipt_ids))
        failing_edge = failing_edge or "receipts.receipt_uuid"
    duplicated_hashes = sorted({x for x in receipt_hashes if receipt_hashes.count(x) > 1})
    if duplicated_hashes:
        blockers.append("duplicate_receipt_hash:" + ",".join(duplicated_hashes))
        failing_edge = failing_edge or "receipts.receipt_hash"

    for key in REQUIRED_TOP_LEVEL_IDS:
        if not bundle.get(key) and top_id_locations[key]:
            blockers.append(f"{key}_only_present_in_child_report")
            failing_edge = failing_edge or key

    absurd_events = bundle.get("absurd_event_rows") or []
    if not isinstance(absurd_events, list):
        blockers.append("absurd_event_rows_not_list")
        absurd_events = []
    for idx, event in enumerate(absurd_events):
        if not isinstance(event, dict):
            blockers.append(f"absurd_event_row_not_object:{idx}")
            continue
        for key in ("command_uuid", "idempotency_key", "job_uuid"):
            if event.get(key) != bundle.get(key):
                blockers.append(f"absurd_event_{key}_mismatch:{idx}")
                failing_edge = failing_edge or f"absurd_event_rows[{idx}].{key}"

    graph_counts = bundle.get("canonical_graph_counts") or {}
    before = graph_counts.get("before")
    after = graph_counts.get("after")
    if before is None or after is None:
        blockers.append("missing_canonical_graph_counts")
    elif before != after:
        blockers.append("canonical_graph_counts_changed")
        failing_edge = failing_edge or "canonical_graph_counts"
    if bundle.get("materialization_performed") is True or bundle.get("canonical_graph_writes_performed") is True:
        blockers.append("canonical_graph_materialization_performed")
        failing_edge = failing_edge or "canonical_graph_materialization"

    if audit_refs:
        if audit_ts is None:
            blockers.append("audit_timestamp_required_for_evidence_check")
        for ev in audit_refs:
            path = str(ev.get("path") or "")
            if path not in known_evidence:
                blockers.append(f"audit_references_evidence_outside_lineage:{path}")
                failing_edge = failing_edge or "audit.evidence_refs"
            ev_ts = parse_ts(ev.get("created_at"))
            if audit_ts is not None and ev_ts is not None and ev_ts > audit_ts:
                blockers.append(f"audit_references_future_evidence:{path}")
                failing_edge = failing_edge or "audit.evidence_refs.created_at"

    expected_bundle_hash = bundle.get("bundle_hash")
    if expected_bundle_hash:
        actual_bundle_hash = registry_hash(bundle)
        if expected_bundle_hash != actual_bundle_hash:
            blockers.append("bundle_hash_mismatch")
            failing_edge = failing_edge or "bundle.bundle_hash"
    else:
        actual_bundle_hash = registry_hash(bundle)
    verdict = "PASS" if not blockers else "FAIL"
    return {
        "schema": "lucidota.same_lineage_validator.v1",
        "generated_at": now(),
        "receipt": receipt_label,
        "accepted_input_schema": bundle.get("schema"),
        "verdict": verdict,
        "same_lineage": verdict == "PASS",
        "failing_edge": failing_edge,
        "blockers": blockers,
        "lineage_id": bundle.get("lineage_id"),
        "command_uuid": bundle.get("command_uuid"),
        "idempotency_key": bundle.get("idempotency_key"),
        "job_uuid": bundle.get("job_uuid"),
        "worker_receipt_uuid": bundle.get("worker_receipt_uuid"),
        "audit_uuid": bundle.get("audit_uuid"),
        "receipt_hashes": receipt_hashes,
        "bundle_hash": expected_bundle_hash or actual_bundle_hash,
        "files_read": sorted(set(files_read)),
        "canonical_graph_materialization": False,
        "canonical_graph_writes_performed": False,
    }

def validate(receipt_path: str | Path) -> dict[str, Any]:
    blockers: list[str] = []
    files_read: list[str] = []
    data = load_json(receipt_path, files_read, blockers)
    if data.get("schema") == BUNDLE_SCHEMA:
        payload = validate_bundle(data, receipt_label=rel(receipt_path))
        payload["files_read"] = sorted(set(payload.get("files_read", []) + files_read))
        if blockers:
            payload["blockers"] = blockers + payload.get("blockers", [])
            payload["verdict"] = "FAIL"
            payload["same_lineage"] = False
        return payload
    if isinstance(data.get("receipt_bundle"), dict):
        payload = validate_bundle(data["receipt_bundle"], receipt_label=rel(receipt_path))
        payload["files_read"] = sorted(set(payload.get("files_read", []) + files_read))
        if blockers:
            payload["blockers"] = blockers + payload.get("blockers", [])
            payload["verdict"] = "FAIL"
            payload["same_lineage"] = False
        return payload
    return {
        "schema": "lucidota.same_lineage_validator.v1",
        "generated_at": now(),
        "receipt": rel(receipt_path),
        "accepted_input_schema": data.get("schema"),
        "verdict": "FAIL",
        "same_lineage": False,
        "failing_edge": "bundle.schema",
        "blockers": blockers + ["missing_receipt_bundle_hash_chain"],
        "lineage_id": data.get("lineage_id"),
        "command_uuid": data.get("command_uuid"),
        "idempotency_key": data.get("idempotency_key"),
        "job_uuid": data.get("job_uuid"),
        "worker_receipt_uuid": data.get("worker_receipt_uuid"),
        "audit_uuid": data.get("audit_uuid"),
        "receipt_hashes": [],
        "files_read": sorted(set(files_read)),
        "canonical_graph_materialization": False,
        "canonical_graph_writes_performed": False,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate hash-linked same-lineage receipt bundle")
    ap.add_argument("--receipt", required=True)
    ap.add_argument("--output")
    args = ap.parse_args()
    payload = validate(args.receipt)
    OUT.mkdir(parents=True, exist_ok=True)
    out = Path(args.output) if args.output else OUT / f"same_lineage_validator_{stamp()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload["report_path"] = rel(out)
    out.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
    print("REPORT_PATH=" + rel(out))
    print("SAME_LINEAGE_VALIDATOR=" + payload["verdict"])
    if payload["verdict"] != "PASS" and payload.get("failing_edge"):
        print("FAILING_EDGE=" + str(payload["failing_edge"]))
    return 0 if payload["verdict"] == "PASS" else 4


if __name__ == "__main__":
    raise SystemExit(main())
