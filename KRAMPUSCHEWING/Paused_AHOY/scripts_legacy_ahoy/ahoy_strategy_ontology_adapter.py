#!/usr/bin/env python3
"""Ahoy strategy ontology adapter CLI wrapper for Ahoy receipts."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ahoy_sim.ontology.strategy_adapter import explain_model_packet, ontology_packet_from_training_row
from ahoy_sim.training.strategy_dataset import raw_row_to_strategy_training_row, default_feature_schema

MAX_JSON_BYTES = 1024 * 1024

def read_json_object(path: Path) -> dict:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"json_missing:{path}")
    size = path.stat().st_size
    if size > MAX_JSON_BYTES:
        raise ValueError(f"json_too_large:{size}>{MAX_JSON_BYTES}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"json_must_be_object:{path}")
    return data


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("packet")
    p.add_argument("--state-row", type=Path, required=True)
    p.add_argument("--feature-schema", type=Path)
    p.add_argument("--out", type=Path, required=True)
    e = sub.add_parser("explain-model")
    e.add_argument("--training-row", type=Path, required=True)
    e.add_argument("--prediction", type=Path, required=True)
    e.add_argument("--shap", type=Path)
    e.add_argument("--out", type=Path, required=True)
    ap.add_argument("--state-row", type=Path)
    ap.add_argument("--feature-schema", type=Path)
    ap.add_argument("--prediction", type=Path)
    ap.add_argument("--shap", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    try:
        if args.cmd == "explain-model":
            row = read_json_object(args.training_row)
            pred = read_json_object(args.prediction)
            shap = read_json_object(args.shap) if args.shap else None
            packet = explain_model_packet(row, pred, shap)
            _write(args.out, packet)
            out = args.out
        else:
            state_row = args.state_row
            out = args.out
            if not state_row or not out:
                ap.error("packet mode requires --state-row and --out")
            raw = read_json_object(state_row)
            schema = read_json_object(args.feature_schema) if args.feature_schema else default_feature_schema()
            training_row = raw if raw.get("schema") == "lucidota.ahoy.strategy_training_row.v1" else raw_row_to_strategy_training_row(raw, schema, future_raw_row=raw, future_window=0)
            packet = ontology_packet_from_training_row(training_row)
            _write(out, packet)
    except Exception as exc:
        print(json.dumps({"verdict": "FAIL", "blockers": [f"{type(exc).__name__}:{exc}"]}, sort_keys=True))
        return 4
    print(json.dumps({"verdict": "PASS", "out": str(out)}, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
