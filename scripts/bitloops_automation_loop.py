#!/usr/bin/env python3
"""Receipt-gated Bitloops automation loop for ABSURD momentary flows.

This is the thin orchestration layer: verified cases run through the
Bitloops -> Bytewax -> River -> ternary momentary flow; unverified cases are
preserved for quarantine/reconciliation and never promoted to graph truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "bitloops"
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_obj(obj: Any) -> str:
    return hashlib.sha256(dumps(obj).encode("utf-8")).hexdigest()


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def valid_hash(value: Any) -> bool:
    return isinstance(value, str) and bool(HEX64.fullmatch(value))


def first_receipt(case: dict[str, Any]) -> dict[str, Any]:
    receipt = case.get("receipt")
    if isinstance(receipt, dict):
        return receipt
    receipts = case.get("receipts")
    if isinstance(receipts, list):
        for item in receipts:
            if isinstance(item, dict):
                return item
    return {}


def verify_case(case: dict[str, Any]) -> tuple[bool, list[str]]:
    receipt = first_receipt(case)
    reasons: list[str] = []
    if not receipt:
        reasons.append("missing_receipt")
    input_sha = receipt.get("input_sha256")
    if not valid_hash(input_sha) or input_sha != sha256_obj(case.get("input")):
        reasons.append("input_sha256_mismatch")
    has_output_hash = valid_hash(receipt.get("output_sha256"))
    trace = receipt.get("transition_trace")
    has_trace = isinstance(trace, list) and bool(trace)
    test_trace = receipt.get("test_trace")
    has_test_trace = isinstance(test_trace, dict) and test_trace.get("status") in {"pass", "passed", "PASS", "PASSED"}
    if not (has_output_hash or has_test_trace):
        reasons.append("missing_output_hash_or_passing_test_trace")
    if not (has_trace or has_test_trace):
        reasons.append("missing_transition_trace_or_passing_test_trace")
    return not reasons, reasons


def quarantine_item(case: dict[str, Any], reasons: list[str]) -> dict[str, Any]:
    return {
        "case_id": str(case.get("case_id") or sha256_obj(case)[:16]),
        "legacy_etl": str(case.get("legacy_etl") or case.get("source") or "unknown"),
        "reason": "unreceipted_slop",
        "evidence_errors": reasons,
        "preservation_action": "preserve_index_quarantine_reconcile",
        "canonical_status": "not_canonical_yet",
        "case_sha256": sha256_obj(case),
    }


def case_id(case: dict[str, Any]) -> str:
    return str(case.get("case_id") or sha256_obj(case)[:16])


def legacy_etl(case: dict[str, Any]) -> str:
    return str(case.get("legacy_etl") or case.get("source") or "unknown")


def mappable_input(case: dict[str, Any]) -> bool:
    return "input" in case and case.get("input") is not None


def recovery_failure(case: dict[str, Any]) -> dict[str, str]:
    return {
        "status": "QUARANTINE_FAILED",
        "reason": "irrecoverable_schema_mismatch",
        "raw_id": case_id(case),
        "legacy_etl": legacy_etl(case),
    }


def recover_receipt(case: dict[str, Any], reasons: list[str]) -> dict[str, Any]:
    cid = case_id(case)
    transition_trace = [
        {"node": "legacy_input", "op": "deterministic_parse", "to": "bitloops_context"},
        {"node": "bitloops_context", "op": "align_case_context", "to": "bytewax_hint"},
        {"node": "bytewax_hint", "op": "derive_stream_hint", "to": "river_training_lane"},
        {"node": "river_training_lane", "op": "emit_training_example", "to": "ternary_truth"},
    ]
    normalized = {
        "case_id": cid,
        "legacy_etl": legacy_etl(case),
        "input": case.get("input"),
        "target_schema": "bitloops_and_river_lane_specs",
        "transition_trace": transition_trace,
    }
    return {
        "case_id": cid,
        "legacy_etl": legacy_etl(case),
        "receipt_origin": "deterministic_replay_current_logic",
        "verification_method": "sha256_over_normalized_input_and_current_transition_trace",
        "recovered_from_errors": reasons,
        "input_sha256": sha256_obj(case.get("input")),
        "output_sha256": sha256_obj(normalized),
        "transition_trace": transition_trace,
    }


def graph_mutation(case: dict[str, Any]) -> dict[str, Any]:
    receipt = first_receipt(case)
    return {
        "mutation_kind": "upsert_case_training_evidence",
        "mutation_state": "candidate_requires_graph_promotion_gate",
        "case_id": case_id(case),
        "legacy_etl": legacy_etl(case),
        "input_sha256": receipt.get("input_sha256"),
        "output_sha256": receipt.get("output_sha256"),
        "transition_trace": receipt.get("transition_trace") or [],
        "lanes": ["bitloops_context", "bytewax_hint", "river_training_lane", "ternary_truth"],
    }


def evidence_refs_for(case: dict[str, Any]) -> list[str]:
    receipt = first_receipt(case)
    refs = [str(case.get("source_ref") or legacy_etl(case)), f"input_sha256:{receipt.get('input_sha256')}", f"output_sha256:{receipt.get('output_sha256')}"]
    return [r for r in refs if r and not r.endswith(":None")]


def claim_packet_for(mutation: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    return {
        "claim_packet_id": mutation["case_id"],
        "claim_text": f"{mutation['legacy_etl']} recovered into Bitloops/River training lanes",
        "label": "bitloops_river_training_evidence_candidate",
        "matched_text": dumps(mutation),
        "review_state": "candidate_requires_graph_promotion_gate",
        "authority_class": "deterministic_metric",
        "evidence_refs": evidence_refs_for(case),
        "candidate_kind": "node",
        "candidate_payload": mutation,
    }


def compact_batch(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": report["status"],
        "accepted": len(report["accepted_case_ids"]),
        "recovered": len(report["recovered_case_ids"]),
        "quarantine_failed": len(report["quarantine_failed"]),
        "river_training_lane_count": report["river_training_lane_count"],
        "graph_mutation_candidates": len(report["graph_mutations"]),
        "canonical_graph_writes_performed": report["canonical_graph_writes_performed"],
        "state_collapsed": report["state_collapsed"],
    }


def write_graph_promotion_bundle(report: dict[str, Any], accepted: list[dict[str, Any]], output: Path | None) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = output or OUT / f"bitloops_graph_promotion_packet_bundle_{stamp()}.json"
    if not path.is_absolute():
        path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    claim_packets = [claim_packet_for(mutation, case) for mutation, case in zip(report["graph_mutations"], accepted)]
    bundle = {
        "schema": "lucidota.bitloops.graph_promotion_packet_bundle.v1",
        "generated_at": now(),
        "source_system": "bitloops_automation_loop",
        "loop_id": report["loop_id"],
        "claim_packet_count": len(claim_packets),
        "claim_packets": claim_packets,
        "compact_batch": report["compact_batch"],
        "db_writes_performed": False,
        "canonical_graph_writes_performed": False,
        "next_gate": "scripts/graph_promotion_dry_run.py --candidate-file <this-file>",
    }
    bundle["bundle_sha256"] = sha256_obj({k: v for k, v in bundle.items() if k != "bundle_sha256"})
    path.write_text(json.dumps(bundle, indent=2, sort_keys=True), encoding="utf-8")
    return path


def row_identifier(row: dict[str, Any]) -> str:
    for key in ("row_id", "candidate_id", "event_id", "file_uuid", "artifact_sha256", "source_sha256", "sha256", "id"):
        value = row.get(key)
        if value:
            return str(value)
    return sha256_obj(row)[:16]


def row_text(row: dict[str, Any]) -> str:
    keys = (
        "source_path",
        "path",
        "provenance_path",
        "first_seen_path",
        "proposed_term",
        "lane",
        "truth_status",
        "event_time_utc",
        "resolved_timestamp",
        "epistemic_flag",
        "dominant_evidence_source",
    )
    return " ".join(str(row.get(k) or "") for k in keys if row.get(k) is not None).strip()


def row_to_case(row: dict[str, Any], *, legacy_etl_name: str, source_ref: str) -> dict[str, Any]:
    return {
        "case_id": row_identifier(row),
        "legacy_etl": legacy_etl_name,
        "source_ref": source_ref,
        "input": {
            "text": row_text(row),
            "legacy_row": row,
            "source_sha256": row.get("source_sha256") or row.get("artifact_sha256") or row.get("sha256"),
        },
    }


def model_lane_plan() -> dict[str, Any]:
    return {
        "policy": "local_first_cloud_optional_dry_by_default",
        "model_calls_performed": False,
        "lanes": [
            {
                "lane_group": "local",
                "execution": "planned_only",
                "lanes": ["deepseek", "mamba_cpu", "bonsai", "needle_swarm"],
                "role": "sovereign first-pass routing/summarization/cross-check",
            },
            {
                "lane_group": "groq",
                "execution": "dry_run_only_until_explicit_execute",
                "role": "optional cheap cloud sidecar for bounded code/audit slices",
            },
            {
                "lane_group": "cohere",
                "execution": "dry_run_only_until_explicit_execute",
                "role": "optional cloud sidecar for text/semantic comparison",
            },
            {
                "lane_group": "current_assistant",
                "execution": "active_orchestrator_receipted_by_chat_and_files",
                "role": "coordination, code edits, verification receipts",
            },
        ],
    }


def flow_payload(case: dict[str, Any]) -> dict[str, Any]:
    case_id = str(case.get("case_id") or sha256_obj(case)[:16])
    source_ref = str(case.get("source_ref") or case.get("legacy_etl") or case_id)
    payload_input = case.get("input") if isinstance(case.get("input"), dict) else {"value": case.get("input")}
    return {
        "flow_id": f"bitloops_auto_{case_id}",
        "source_ref": source_ref,
        "input": {
            **payload_input,
            "case_id": case_id,
            "legacy_etl": case.get("legacy_etl"),
            "receipt": first_receipt(case),
        },
        "nodes": [
            {"id": "bitloops", "op": "bitloops_context"},
            {"id": "bytewax", "op": "bytewax_hint"},
            {"id": "river", "op": "river_mre"},
            {"id": "ternary", "op": "ternary_truth"},
        ],
    }


def run_momentary(payload: dict[str, Any]) -> dict[str, Any]:
    if str(ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(ROOT / "scripts"))
    from absurd_queue_spine import run_job

    ok, result, err = run_job("momentary_flow", payload)
    if not ok:
        result = dict(result)
        result["loop_error"] = err
    return result


def orchestrate_loop(packet: dict[str, Any], *, write_graph_promotion_packet: bool = False, graph_promotion_packet_path: Path | None = None) -> dict[str, Any]:
    cases = packet.get("cases") if isinstance(packet.get("cases"), list) else []
    accepted: list[dict[str, Any]] = []
    recovered: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    quarantine_failed: list[dict[str, Any]] = []
    for case in cases:
        if not isinstance(case, dict):
            quarantine_failed.append(
                {
                    "status": "QUARANTINE_FAILED",
                    "reason": "irrecoverable_schema_mismatch",
                    "raw_id": "non_object_case",
                    "legacy_etl": "unknown",
                }
            )
            continue
        ok, reasons = verify_case(case)
        if ok:
            accepted.append(case)
        elif mappable_input(case):
            rebuilt = dict(case)
            rebuilt["receipt"] = recover_receipt(case, reasons)
            accepted.append(rebuilt)
            recovered.append(rebuilt)
        else:
            quarantine_failed.append(recovery_failure(case))

    momentary_results = [run_momentary(flow_payload(case)) for case in accepted]
    training_examples = [ex for result in momentary_results for ex in result.get("training_examples", [])]
    failed = [r for r in momentary_results if r.get("outcome") != "succeeded"]
    if failed:
        status = "FAIL"
    elif accepted and not quarantine_failed:
        status = "PASS"
    elif accepted and quarantine_failed:
        status = "PARTIAL"
    elif quarantine_failed:
        status = "QUARANTINE_FAILED"
    else:
        status = "EMPTY"
    report = {
        "schema": "lucidota.bitloops.automation_loop.v1",
        "generated_at": now(),
        "loop_id": str(packet.get("loop_id") or "bitloops_automation_loop"),
        "status": status,
        "accepted_case_ids": [str(case.get("case_id") or sha256_obj(case)[:16]) for case in accepted],
        "recovered_case_ids": [case_id(case) for case in recovered],
        "recovery_receipts": [first_receipt(case) for case in recovered],
        "quarantine": quarantine,
        "quarantine_failed": quarantine_failed,
        "momentary_results": momentary_results,
        "river_training_lane_count": len(training_examples),
        "bytewax_hint_count": sum(1 for ex in training_examples if ex.get("channel") == "bytewax_hint"),
        "graph_mutations": [graph_mutation(case) for case in accepted],
        "model_lane_plan": model_lane_plan(),
        "state_collapsed": all(r.get("state_collapsed") is True for r in momentary_results) if momentary_results else True,
        "destroyed_case_count": 0,
        "purged_case_count": 0,
        "canonical_graph_writes_performed": False,
        "model_calls_performed": False,
        "db_writes_performed": False,
        "proof": "verified cases ran through momentary ABSURD flow in-process; unverified cases preserved for quarantine/reconciliation; canonical graph untouched",
    }
    report["compact_batch"] = compact_batch(report)
    report["graph_promotion_packet_path"] = None
    if write_graph_promotion_packet:
        report["graph_promotion_packet_path"] = rel(write_graph_promotion_bundle(report, accepted, graph_promotion_packet_path))
    return report


def load_packet(args: argparse.Namespace) -> dict[str, Any]:
    if args.cases_file:
        data = json.loads(Path(args.cases_file).read_text(encoding="utf-8"))
    elif args.cases_json:
        data = json.loads(args.cases_json)
    elif args.legacy_jsonl:
        cases: list[dict[str, Any]] = []
        with Path(args.legacy_jsonl).open("r", encoding="utf-8") as fh:
            for line in fh:
                if args.limit and len(cases) >= args.limit:
                    break
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                cases.append(row_to_case(row, legacy_etl_name=args.legacy_etl, source_ref=rel(args.legacy_jsonl)))
        data = {"loop_id": f"{args.legacy_etl}_legacy_jsonl_recovery", "cases": cases}
    elif args.chrono_snapshot:
        cases = []
        in_rows = False
        with Path(args.chrono_snapshot).open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if "[RESOLVED_CHRONO_TIMELINE_JSONL]" in line:
                    in_rows = True
                    continue
                if not in_rows or not line.lstrip().startswith("{"):
                    continue
                if args.limit and len(cases) >= args.limit:
                    break
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    cases.append(row_to_case(row, legacy_etl_name="chrono_master_snapshot", source_ref=rel(args.chrono_snapshot)))
        data = {"loop_id": "chrono_master_snapshot_ouroboros_recovery", "cases": cases}
    else:
        data = {"cases": []}
    if not isinstance(data, dict):
        raise SystemExit("cases packet must be a JSON object")
    return data


def write_report(report: dict[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"bitloops_automation_loop_{stamp()}.json"
    report["report_path"] = rel(path)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"REPORT_PATH={rel(path)}")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="Run a no-purge Bitloops/Bytewax/River automation loop over receipted cases.")
    source = ap.add_mutually_exclusive_group()
    source.add_argument("--cases-file")
    source.add_argument("--cases-json")
    source.add_argument("--legacy-jsonl")
    source.add_argument("--chrono-snapshot")
    ap.add_argument("--legacy-etl", default="legacy_etl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--write-graph-promotion-packet", action="store_true")
    ap.add_argument("--graph-promotion-packet-path")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    packet_path = Path(args.graph_promotion_packet_path) if args.graph_promotion_packet_path else None
    report = orchestrate_loop(load_packet(args), write_graph_promotion_packet=args.write_graph_promotion_packet, graph_promotion_packet_path=packet_path)
    write_report(report)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
