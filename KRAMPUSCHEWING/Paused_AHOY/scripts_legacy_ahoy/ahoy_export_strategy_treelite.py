#!/usr/bin/env python3
"""Export Ahoy XGBoost strategy models to Treelite artifacts with receipts."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ahoy_sim.training.export_strategy_treelite import export_strategy_treelite

def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)

def main() -> int:
    ap = argparse.ArgumentParser(description="Export Ahoy XGBoost strategy models to Treelite artifacts.")
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--receipt-out", type=Path, required=True)
    args = ap.parse_args()
    receipt = export_strategy_treelite(args.model, args.out, args.receipt_out)
    actual_out = Path(receipt.get("treelite_path", args.out))
    print(json.dumps({"verdict": receipt["verdict"], "treelite_exported": receipt.get("treelite_exported"), "out": display_path(actual_out), "receipt": display_path(args.receipt_out)}, sort_keys=True))
    return 0 if receipt["verdict"] == "PASS" else 4

if __name__ == "__main__":
    raise SystemExit(main())
