#!/usr/bin/env python3
"""Emit the Edge Grail rolling KV/cache policy receipt.

This answers the practical eviction question: rolling 500-token chunk ring with
receipt-backed eviction. It deliberately separates current proof from target
claims: Needle shared server batching exists; exact tensor K/V pointer-sharing is
still a runner-extension target until benchmarked.
"""
from __future__ import annotations

import argparse
import json
import time
from hashlib import sha256
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "04_RUNTIME/edge_grail_kv_cache_policy.json"
RECEIPT = ROOT / "05_OUTPUTS/runtime/edge_grail_kv_cache_policy_latest.json"


def now_z() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def sha_text(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def load_policy(path: Path = POLICY) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate(policy: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if policy.get("policy") != "rolling_window_shared_prefix":
        errors.append("policy_not_rolling_window_shared_prefix")
    if int(policy.get("chunk_tokens") or 0) != 500:
        errors.append("chunk_tokens_not_500")
    if policy.get("flush_strategy") == "blind_flush_every_500":
        errors.append("blind_flush_forbidden")
    if not policy.get("eviction", {}).get("drop_raw_text_from_prompt"):
        errors.append("raw_text_not_reaped")
    if not policy.get("eviction", {}).get("never_keep_unbounded_history"):
        errors.append("unbounded_history_not_forbidden")
    ledger = policy.get("ledger_mib", {})
    total = int(ledger.get("target_total_allocated_mib") or 0)
    budget = int(ledger.get("gtx1650_budget_mib") or 0)
    remaining = int(ledger.get("target_remaining_mib") or 0)
    if total > budget:
        errors.append("target_total_exceeds_budget")
    if remaining <= 0:
        errors.append("no_remaining_vram")
    truth = policy.get("truth_law", {})
    if truth.get("exact_needle_tensor_kv_dedup_unproven_until_runner_extension") is not True:
        errors.append("needle_tensor_dedup_truth_not_explicit")
    return errors


def build_receipt(policy: dict[str, Any], *, policy_path: Path = POLICY, receipt_path: Path = RECEIPT) -> dict[str, Any]:
    raw = json.dumps(policy, sort_keys=True)
    errors = validate(policy)
    truth = policy.get("truth_law", {})
    payload = {
        "schema": "lucidota.edge_grail.kv_cache_policy.receipt.v1",
        "status": "PASS" if not errors else "FAIL",
        "generated_at": now_z(),
        "policy_path": rel(policy_path),
        "policy_hash": sha_text(raw),
        "policy": policy.get("policy"),
        "chunk_tokens": policy.get("chunk_tokens"),
        "chunk_overlap_tokens": policy.get("chunk_overlap_tokens"),
        "flush_strategy": policy.get("flush_strategy"),
        "ledger_mib": policy.get("ledger_mib", {}),
        "truth_flags": {
            "exact_needle_tensor_kv_dedup_proven": False,
            "shared_needle_server_batched_prefix_available": bool(truth.get("shared_needle_server_batched_prefix_available")),
            "bonsai_kv_unified_configured_in_start_script": bool(truth.get("bonsai_kv_unified_configured_in_start_script")),
            "mamba_iq1_s_one_bit_class_downloaded": bool(truth.get("mamba_iq1_s_one_bit_class_downloaded")),
            "mamba_literal_q1_0_downloaded": not bool(truth.get("mamba_literal_q1_0_absent")),
            "treelite_gpu_residency_proven": bool(truth.get("treelite_gpu_residency_proven_by_runpod_fil_receipt")),
        },
        "eviction_answer": policy.get("answer_to_eviction_question"),
        "errors": errors,
        "receipt_path": rel(receipt_path),
    }
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(prog="edge-grail-kv-cache-policy")
    ap.add_argument("--policy", default=str(POLICY))
    ap.add_argument("--receipt", default=str(RECEIPT))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    policy_path = Path(args.policy)
    receipt_path = Path(args.receipt)
    policy = load_policy(policy_path)
    receipt = build_receipt(policy, policy_path=policy_path, receipt_path=receipt_path)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True) if args.json else json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
