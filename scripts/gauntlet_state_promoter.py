#!/usr/bin/env python3
"""Promote RGAUNTLET work-order states from validated proof packets.

Streaming rules:
- proof packet ingest is bounded by --max-proofs
- master gauntlet is streamed line-by-line
- source gauntlet is never mutated; a timestamped replacement JSONL is written
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GAUNTLET = ROOT / "05_OUTPUTS/work_orders/lucidota_600_work_order_gauntlet_20260517T211943051910Z.jsonl"
DEFAULT_PROOFS = ROOT / "05_OUTPUTS/matrix_execution/work_order_proof_packets_20260520T074817Z.jsonl"
OUT_DIR = ROOT / "05_OUTPUTS/work_orders"
RECEIPT_DIR = ROOT / "05_OUTPUTS/matrix_execution"
ACCEPTED_BEFORE = {"BLOCKED", "PENDING", "PASSED_FOR_REVIEW"}
ACCEPTED_AFTER = {"PASSED_FOR_REVIEW", "PASSED"}


def now_z() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def load_validated_proofs(path: Path, *, max_proofs: int) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    proofs: dict[str, dict[str, Any]] = {}
    stats: dict[str, Any] = {
        "lines_read": 0,
        "proofs_loaded": 0,
        "duplicates_seen": 0,
        "invalid_packets": 0,
        "ignored_packets": collections.Counter(),
    }
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            stats["lines_read"] += 1
            if stats["lines_read"] > max_proofs:
                raise SystemExit(f"proof_limit_exceeded:{max_proofs}")
            try:
                packet = json.loads(line)
            except json.JSONDecodeError:
                stats["invalid_packets"] += 1
                stats["ignored_packets"]["json_decode_error"] += 1
                continue
            wid = str(packet.get("work_order_id") or "")
            before = str(packet.get("status_before") or "").upper()
            after = str(packet.get("status_after") or "").upper()
            return_codes = packet.get("command_return_codes") or []
            blocker = packet.get("remaining_blocker")
            if not wid:
                stats["invalid_packets"] += 1
                stats["ignored_packets"]["missing_work_order_id"] += 1
                continue
            if before not in ACCEPTED_BEFORE:
                stats["ignored_packets"]["status_before_not_promotable"] += 1
                continue
            if after not in ACCEPTED_AFTER:
                stats["ignored_packets"]["status_after_not_accepted"] += 1
                continue
            if blocker:
                stats["ignored_packets"]["remaining_blocker_present"] += 1
                continue
            if not return_codes or any(int(c) != 0 for c in return_codes):
                stats["ignored_packets"]["nonzero_or_missing_returncode"] += 1
                continue
            if wid in proofs:
                stats["duplicates_seen"] += 1
            packet["_proof_line_no"] = line_no
            packet["status_after"] = after
            packet["status_before"] = before
            proofs[wid] = packet
            stats["proofs_loaded"] = len(proofs)
    stats["ignored_packets"] = dict(stats["ignored_packets"])
    return proofs, stats


def promote_gauntlet(source: Path, proofs: dict[str, dict[str, Any]], *, target_status: str) -> tuple[Path, dict[str, Any]]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"lucidota_600_work_order_gauntlet_promoted_{stamp()}.jsonl"
    stats: dict[str, Any] = {
        "source_rows_read": 0,
        "rows_written": 0,
        "matched_proofs": 0,
        "promoted_count": 0,
        "skipped_matches": collections.Counter(),
        "status_before_counts": collections.Counter(),
        "status_after_counts": collections.Counter(),
        "promoted_work_order_ids": [],
    }
    seen_ids: set[str] = set()
    with source.open("r", encoding="utf-8") as src, out.open("w", encoding="utf-8") as dst:
        for line_no, line in enumerate(src, 1):
            if not line.strip():
                continue
            stats["source_rows_read"] += 1
            row = json.loads(line)
            wid = str(row.get("work_order_id") or "")
            old_status = str(row.get("status") or "").upper()
            stats["status_before_counts"][old_status] += 1
            proof = proofs.get(wid)
            if proof:
                stats["matched_proofs"] += 1
                seen_ids.add(wid)
                if old_status not in ACCEPTED_BEFORE:
                    stats["skipped_matches"]["gauntlet_status_not_promotable"] += 1
                else:
                    new_status = proof.get("status_after") if target_status == "proof" else target_status
                    if new_status not in ACCEPTED_AFTER:
                        raise SystemExit(f"invalid_target_status:{new_status}")
                    row["status"] = new_status
                    row["previous_status"] = old_status
                    row["status_promoted_at"] = now_z()
                    row["status_promotion_source"] = "proof_packet"
                    row["status_promotion_source_file"] = rel(proof.get("receipt_path") or "")
                    row["proof_packet_line_no"] = proof.get("_proof_line_no")
                    row["proof_name"] = proof.get("proof_name")
                    row["proof_commands"] = proof.get("exact_commands_run") or []
                    row["proof_return_codes"] = proof.get("command_return_codes") or []
                    row["proof_reason"] = proof.get("reason_closure_is_justified")
                    row["receipt_path"] = proof.get("receipt_path") or row.get("receipt_path")
                    stats["promoted_count"] += 1
                    stats["promoted_work_order_ids"].append(wid)
                    old_status = str(row.get("status") or "").upper()
            stats["status_after_counts"][old_status] += 1
            dst.write(json.dumps(row, sort_keys=True, ensure_ascii=False, default=str) + "\n")
            stats["rows_written"] += 1
    missing = sorted(set(proofs) - seen_ids)
    stats["proofs_without_gauntlet_match"] = missing[:100]
    stats["proofs_without_gauntlet_match_count"] = len(missing)
    stats["skipped_matches"] = dict(stats["skipped_matches"])
    stats["status_before_counts"] = dict(stats["status_before_counts"])
    stats["status_after_counts"] = dict(stats["status_after_counts"])
    return out, stats


def write_receipt(payload: dict[str, Any], path: Path | None = None) -> Path:
    RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
    path = path or (RECEIPT_DIR / f"gauntlet_state_promotion_receipt_{stamp()}.json")
    payload["receipt_path"] = rel(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str), encoding="utf-8")
    return path


def update_status_ledger(updated_gauntlet: Path, receipt_path: Path) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "scripts/lucidota_status_ledger.py",
        "--set",
        "RGAUNTLET Master Gauntlet",
        "--status",
        "verified",
        "--progress",
        "100",
        "--executed",
        "yes",
        "--evidence",
        f"{rel(updated_gauntlet)}; {rel(receipt_path)}",
        "--next",
        f"Use {rel(updated_gauntlet)} for the next bounded matrix executor wave.",
        "--blocker",
        "",
    ]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60, check=False)
    return {"cmd": cmd, "returncode": proc.returncode, "stdout_tail": proc.stdout[-2000:], "stderr_tail": proc.stderr[-2000:]}


def main() -> int:
    ap = argparse.ArgumentParser(description="Promote RGAUNTLET states from validated proof packets")
    ap.add_argument("--gauntlet", default=str(DEFAULT_GAUNTLET))
    ap.add_argument("--proof-packets", default=str(DEFAULT_PROOFS))
    ap.add_argument("--max-proofs", type=int, default=10000)
    ap.add_argument("--target-status", choices=["proof", "PASSED_FOR_REVIEW", "PASSED"], default="proof")
    ap.add_argument("--update-status-ledger", action="store_true")
    args = ap.parse_args()

    source = Path(args.gauntlet)
    packets = Path(args.proof_packets)
    proofs, proof_stats = load_validated_proofs(packets, max_proofs=args.max_proofs)
    updated, promotion_stats = promote_gauntlet(source, proofs, target_status=args.target_status)
    receipt = {
        "schema": "lucidota.matrix.gauntlet_state_promotion_receipt.v1",
        "generated_at": now_z(),
        "source_gauntlet": rel(source),
        "updated_gauntlet": rel(updated),
        "proof_packets": rel(packets),
        "source_gauntlet_mutated": False,
        "proof_stats": proof_stats,
        "promotion_stats": promotion_stats,
        "accepted_before_statuses": sorted(ACCEPTED_BEFORE),
        "accepted_after_statuses": sorted(ACCEPTED_AFTER),
        "status_ledger_update": None,
        "blockers": [],
    }
    receipt_path = write_receipt(receipt)
    if args.update_status_ledger:
        ledger = update_status_ledger(updated, receipt_path)
        receipt["status_ledger_update"] = ledger
        if ledger["returncode"] != 0:
            receipt["blockers"].append("status_ledger_update_failed")
        write_receipt(receipt, receipt_path)
    print(f"UPDATED_GAUNTLET={rel(updated)}")
    print(f"PROMOTION_RECEIPT={rel(receipt_path)}")
    print(f"PROMOTED_COUNT={promotion_stats['promoted_count']}")
    return 0 if not receipt["blockers"] and promotion_stats["promoted_count"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
