#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ahoy_sim.engine.receipts import OUT, stamp, utc_now, write_json_receipt
from ahoy_sim.training.treelite_export import export_treelite


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=False)
    args = ap.parse_args()
    models = sorted((OUT / "models").glob("offline_tree_*.pkl"))
    model = args.model or (models[-1] if models else OUT / "models" / "missing.pkl")
    result = export_treelite(model, OUT / "models" / f"treelite_{stamp()}")
    receipt = {"schema": "lucidota.ahoy.treelite_export.v1", "created_at": utc_now(), "model_path": str(model), **result}
    path = OUT / "models" / f"treelite_{stamp()}.json"
    write_json_receipt(path, receipt)
    print(json.dumps({"verdict": receipt["verdict"], "receipt": str(path), "treelite_status": receipt.get("treelite_status")}, sort_keys=True))
    return 0 if receipt["verdict"] == "PASS" else 4

if __name__ == "__main__":
    raise SystemExit(main())
