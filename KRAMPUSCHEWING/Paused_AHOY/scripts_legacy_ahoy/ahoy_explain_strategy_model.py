#!/usr/bin/env python3
"""Build an Ahoy strategy model explanation ontology packet."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ahoy_sim.training.explain_strategy_model import compute_prediction_and_shap, explain_strategy_model, read_json_object, write_json_atomic

def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)

def main() -> int:
    ap = argparse.ArgumentParser(description="Build an Ahoy strategy model explanation ontology packet.")
    ap.add_argument("--training-row", type=Path, required=True)
    ap.add_argument("--prediction", type=Path)
    ap.add_argument("--shap", type=Path)
    ap.add_argument("--model", type=Path)
    ap.add_argument("--model-metadata", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    prediction_path = args.prediction
    shap_path = args.shap
    try:
        if not args.model and not prediction_path:
            raise ValueError("prediction_or_model_required")
        if args.model and not prediction_path and not shap_path:
            row = read_json_object(args.training_row)
            pred, shap = compute_prediction_and_shap(row, args.model)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            prediction_path = args.out.parent / (args.out.stem + "_prediction.json")
            shap_path = args.out.parent / (args.out.stem + "_shap.json")
            write_json_atomic(prediction_path, pred)
            write_json_atomic(shap_path, shap)
        packet = explain_strategy_model(args.training_row, prediction_path, shap_path, args.out, model_metadata_path=args.model_metadata)
    except Exception as exc:
        print(json.dumps({"verdict": "FAIL", "blockers": [f"{type(exc).__name__}:{exc}"]}, sort_keys=True))
        return 4
    print(json.dumps({"verdict": "PASS", "out": display_path(args.out), "evidence": len(packet["ontology"]["EVIDENCE"])}, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
