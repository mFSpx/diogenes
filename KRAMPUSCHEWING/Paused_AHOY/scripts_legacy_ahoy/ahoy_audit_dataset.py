#!/usr/bin/env python3
"""Audit Ahoy training JSONL quality and optionally persist the receipt."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ahoy_sim.training.dataset_audit import audit_dataset

def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)

def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)

def main() -> int:
    ap = argparse.ArgumentParser(description="Audit Ahoy training JSONL quality.")
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--min-distinct-labels-per-faction", type=int, default=5)
    ap.add_argument("--out", type=Path, help="Optional JSON receipt path.")
    ap.add_argument("--pretty", action="store_true", help="Pretty-print stdout JSON.")
    args = ap.parse_args()
    result = audit_dataset(args.dataset, min_distinct_labels_per_faction=args.min_distinct_labels_per_faction)
    if args.out:
        out = args.out if args.out.is_absolute() else ROOT / args.out
        result = dict(result, report_path=display_path(out))
        write_json_atomic(out, result)
    print(json.dumps(result, indent=2 if args.pretty else None, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 4


if __name__ == "__main__":
    raise SystemExit(main())
