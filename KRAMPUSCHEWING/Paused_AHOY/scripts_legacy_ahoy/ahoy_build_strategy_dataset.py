#!/usr/bin/env python3
"""Build Ahoy abstract strategy training rows and ontology packets."""
from __future__ import annotations
import argparse, json, shlex, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ahoy_sim.training.strategy_dataset import build_strategy_dataset, default_feature_schema

MAX_FEATURE_SCHEMA_BYTES = 256 * 1024

def write_failure_receipt(path: Path, blocker: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": "lucidota.ahoy.strategy_dataset_receipt.v1", "verdict": "FAIL", "blockers": [blocker]}
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)

def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)

def load_feature_schema(path: Path | None) -> dict:
    if path is None:
        return default_feature_schema()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"feature_schema_missing:{path}")
    size = path.stat().st_size
    if size > MAX_FEATURE_SCHEMA_BYTES:
        raise ValueError(f"feature_schema_too_large:{size}>{MAX_FEATURE_SCHEMA_BYTES}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("feature_schema_must_be_object")
    return data

def main() -> int:
    ap = argparse.ArgumentParser(description="Build Ahoy abstract strategy training rows and ontology packets.")
    ap.add_argument("--games", type=Path, required=True, help="Directory or JSONL file containing raw Ahoy telemetry rows")
    ap.add_argument("--out-rows", type=Path, required=True)
    ap.add_argument("--out-packets", type=Path, required=True)
    ap.add_argument("--receipt-out", type=Path, required=True)
    ap.add_argument("--feature-schema", type=Path)
    ap.add_argument("--future-window", type=int, default=3)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    try:
        if not args.games.exists():
            raise FileNotFoundError(f"games_source_missing:{args.games}")
        schema = load_feature_schema(args.feature_schema)
        receipt = build_strategy_dataset(args.games, args.out_rows, args.out_packets, args.receipt_out, feature_schema=schema, future_window=args.future_window, limit=args.limit, command=shlex.join(sys.argv))
    except Exception as exc:
        write_failure_receipt(args.receipt_out, f"{type(exc).__name__}:{exc}")
        print(json.dumps({"verdict": "FAIL", "blockers": [f"{type(exc).__name__}:{exc}"]}, sort_keys=True))
        return 4
    print(json.dumps({"verdict": receipt["verdict"], "rows": receipt["training_rows_written"], "packets": receipt["ontology_packets_written"], "receipt": display_path(args.receipt_out)}, sort_keys=True))
    return 0 if receipt["verdict"] in {"PASS", "PARTIAL_FAIL"} else 4

if __name__ == "__main__":
    raise SystemExit(main())
