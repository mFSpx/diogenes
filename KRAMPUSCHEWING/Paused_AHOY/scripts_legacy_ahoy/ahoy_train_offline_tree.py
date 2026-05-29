#!/usr/bin/env python3
"""Ahoy train offline tree CLI wrapper for Ahoy receipts."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ahoy_sim.engine.receipts import OUT, stamp, utc_now, write_json_receipt
from ahoy_sim.rules.rule_gaps import blocking_gaps
from ahoy_sim.training.offline_train import train_offline_tree

def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=False)
    ap.add_argument("--allow-degraded-dataset", action="store_true")
    args = ap.parse_args()
    candidates = sorted((OUT / "training").glob("*rows*.jsonl"))
    dataset = args.dataset or (candidates[-1] if candidates else OUT / "training" / "missing.jsonl")
    model_path = OUT / "models" / f"offline_tree_{stamp()}.pkl"
    gaps = blocking_gaps()
    if gaps and not args.allow_degraded_dataset:
        result = {"verdict": "BLOCKED", "blockers": [g["gap_id"] for g in gaps] + ["clean_rule_complete_dataset_required"], "model_path": str(model_path), "rows": 0}
    else:
        result = train_offline_tree(dataset, model_path)
    receipt = {"schema": "lucidota.ahoy.offline_tree_training.v1", "created_at": utc_now(), "dataset_path": str(dataset), "blocking_rule_gaps": gaps, **result}
    path = OUT / "models" / f"model_{stamp()}.json"
    write_json_receipt(path, receipt)
    print(json.dumps({"verdict": receipt["verdict"], "receipt": display_path(path), "model_path": receipt.get("model_path")}, sort_keys=True))
    return 0 if receipt["verdict"] == "PASS" else 4

if __name__ == "__main__":
    raise SystemExit(main())
