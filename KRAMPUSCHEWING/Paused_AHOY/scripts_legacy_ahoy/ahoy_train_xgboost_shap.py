#!/usr/bin/env python3
"""Ahoy train xgboost shap CLI wrapper for Ahoy receipts."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ahoy_sim.engine.receipts import OUT, stamp
from ahoy_sim.training.xgb_shap_pipeline import train_xgb_shap

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
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--target", default="resolved_effect")
    ap.add_argument("--n-estimators", type=positive_int, default=96)
    ap.add_argument("--shap-sample", type=positive_int, default=512)
    ap.add_argument("--max-depth", type=positive_int, default=3)
    args = ap.parse_args()
    out_dir = args.out or (OUT / "models" / f"xgb_shap_{stamp()}")
    receipt = train_xgb_shap(args.dataset, out_dir, target=args.target, n_estimators=args.n_estimators, max_depth=args.max_depth, shap_sample=args.shap_sample)
    print(json.dumps({"verdict": receipt["verdict"], "out_dir": str(out_dir), "accuracy": receipt.get("xgboost", {}).get("accuracy"), "macro_f1": receipt.get("xgboost", {}).get("macro_f1"), "treelite_model": receipt.get("artifacts", {}).get("treelite_model")}, sort_keys=True))
    return 0 if receipt["verdict"] == "PASS" else 4


if __name__ == "__main__":
    raise SystemExit(main())
