#!/usr/bin/env python3
"""Receipt-backed local GTX admission check for the Edge Grail loadout.

This script does not start or kill models. It reads the verified KV/VRAM ledger,
reads current `nvidia-smi` telemetry, and distinguishes:

- ADMISSIBLE_NOW: current free VRAM can admit the verified target plus reserve.
- ADMISSIBLE_AFTER_SPINDOWN: total card can fit it, but current free VRAM cannot.
- BLOCKED_BY_CARD_BUDGET: even total card budget cannot fit it with reserve.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "04_RUNTIME" / "edge_grail_kv_cache_policy.json"
DEFAULT_RECEIPT = ROOT / "05_OUTPUTS" / "runtime" / "edge_grail_local_admission_latest.json"


def now_z() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def load_policy(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_gpu_csv(text: str) -> dict[str, Any]:
    parts = [p.strip() for p in text.strip().split(",")]
    if len(parts) < 5:
        raise ValueError("expected GPU CSV: name,total,used,free,temp")
    return {
        "name": parts[0],
        "total_mib": int(float(parts[1])),
        "used_mib": int(float(parts[2])),
        "free_mib": int(float(parts[3])),
        "temperature_c": int(float(parts[4])),
    }


def read_gpu() -> dict[str, Any]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,memory.free,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    cp = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=10)
    if cp.returncode != 0:
        return {"available": False, "error": (cp.stderr or cp.stdout).strip()[:500]}
    first = next((line for line in cp.stdout.splitlines() if line.strip()), "")
    gpu = parse_gpu_csv(first)
    gpu["available"] = True
    return gpu


def evaluate_admission(policy: dict[str, Any], *, gpu: dict[str, Any], reserve_mib: int = 450) -> dict[str, Any]:
    ledger = policy.get("ledger_mib") or {}
    target = int(ledger.get("target_total_allocated_mib") or 0)
    card_budget = int(ledger.get("gtx1650_budget_mib") or 0)
    total = int(gpu.get("total_mib") or 0)
    free = int(gpu.get("free_mib") or 0)
    used = int(gpu.get("used_mib") or 0)
    reserve = int(reserve_mib)
    needed_free = target + reserve
    usable_total = min(total, card_budget or total)
    fits_card = usable_total >= needed_free
    admissible_now = bool(fits_card and free >= needed_free)
    requires_spindown = bool(fits_card and not admissible_now)
    if admissible_now:
        status = "ADMISSIBLE_NOW"
    elif requires_spindown:
        status = "ADMISSIBLE_AFTER_SPINDOWN"
    else:
        status = "BLOCKED_BY_CARD_BUDGET"
    return {
        "schema": "lucidota.edge_grail.local_admission.v1",
        "status": status,
        "generated_at": now_z(),
        "gpu": gpu,
        "ledger_mib": ledger,
        "reserve_mib": reserve,
        "needed_free_mib": needed_free,
        "current_free_mib": free,
        "current_used_mib": used,
        "usable_total_mib": usable_total,
        "models_started": False,
        "models_killed": False,
        "truth_flags": {
            "fits_card_with_reserve": bool(fits_card),
            "admissible_now": bool(admissible_now),
            "requires_spindown": bool(requires_spindown),
            "local_full_runtime_currently_admitted": bool(admissible_now),
        },
        "next_required_action": (
            "No spindown required for this ledger." if admissible_now else
            "Spin down unrelated/resident VRAM users, then re-run admission before starting full Edge Grail loadout." if requires_spindown else
            "Reduce ledger, lower reserve, or move a lane off local GTX before attempting admission."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(prog="edge-grail-local-admission")
    ap.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    ap.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    ap.add_argument("--reserve-mib", type=int, default=450)
    ap.add_argument("--mock-gpu", help="Testing CSV: name,total,used,free,temp")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    policy = load_policy(args.policy)
    gpu = parse_gpu_csv(args.mock_gpu) if args.mock_gpu else read_gpu()
    gpu.setdefault("available", True if args.mock_gpu else bool(gpu.get("available")))
    receipt = evaluate_admission(policy, gpu=gpu, reserve_mib=args.reserve_mib)
    receipt["policy"] = {"path": rel(args.policy), "schema": policy.get("schema"), "policy": policy.get("policy")}
    receipt["receipt_path"] = rel(args.receipt)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True) if args.json else json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"] in {"ADMISSIBLE_NOW", "ADMISSIBLE_AFTER_SPINDOWN"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
