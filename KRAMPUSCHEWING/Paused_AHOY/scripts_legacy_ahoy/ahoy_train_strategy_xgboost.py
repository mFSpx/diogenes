#!/usr/bin/env python3
"""Ahoy train strategy xgboost CLI wrapper for Ahoy receipts."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ahoy_sim.training.train_strategy_xgboost import train_strategy_xgboost

def positive_int(value: str) -> int:
    try:
        out = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if out < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return out

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--model-out", type=Path, required=True)
    ap.add_argument("--receipt-out", type=Path, required=True)
    ap.add_argument("--n-estimators", type=positive_int, default=16)
    ap.add_argument("--max-depth", type=positive_int, default=3)
    args = ap.parse_args()
    receipt = train_strategy_xgboost(args.rows, args.target, args.model_out, args.receipt_out, n_estimators=args.n_estimators, max_depth=args.max_depth)
    print(json.dumps({"verdict": receipt["verdict"], "target": args.target, "model": str(args.model_out), "accuracy": receipt.get("accuracy"), "receipt": str(args.receipt_out)}, sort_keys=True))
    return 0 if receipt["verdict"] == "PASS" else 4

if __name__ == "__main__":
    raise SystemExit(main())
