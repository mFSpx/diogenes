#!/usr/bin/env python3
"""Apply legacy Ahoy XGB/SHAP artifact bundles and emit a receipt."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ahoy_sim.engine.receipts import OUT, stamp, write_json_receipt

SCHEMA = "lucidota.ahoy.xgboost_treelite_apply.v1"
REQUIRED_PICKLE_ARTIFACTS = ("vectorizer.pkl", "label_encoder.pkl")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Apply an Ahoy XGBoost/Treelite artifact bundle to a JSONL dataset and write a receipt."
    )
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--artifacts", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--target", default="resolved_effect")
    ap.add_argument("--no-treelite", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    return ap


def preflight_blockers(dataset: Path, artifacts: Path, *, use_treelite: bool) -> list[str]:
    blockers: list[str] = []
    if not dataset.exists():
        blockers.append("dataset_missing")
    elif not dataset.is_file():
        blockers.append("dataset_not_file")

    if not artifacts.exists():
        blockers.append("artifact_dir_missing")
        return blockers
    if not artifacts.is_dir():
        blockers.append("artifact_dir_not_directory")
        return blockers

    for name in REQUIRED_PICKLE_ARTIFACTS:
        if not (artifacts / name).is_file():
            blockers.append(f"artifact_missing:{name}")

    has_xgboost = (artifacts / "xgboost_model.json").is_file()
    has_treelite = use_treelite and (artifacts / "treelite_model.tl").is_file()
    if not has_xgboost and not has_treelite:
        blockers.append("artifact_missing:xgboost_model.json_or_treelite_model.tl")
    return blockers


def blocked_receipt(args: argparse.Namespace, out_path: Path, blockers: list[str]) -> dict[str, Any]:
    receipt = {
        "schema": SCHEMA,
        "verdict": "BLOCKED",
        "dataset_path": str(args.dataset),
        "artifact_dir": str(args.artifacts),
        "target": args.target,
        "use_treelite": not args.no_treelite,
        "engine": "not_started",
        "rows": 0,
        "accuracy": None,
        "macro_f1": None,
        "blockers": blockers,
    }
    write_json_receipt(out_path, receipt)
    return receipt


def fail_receipt(args: argparse.Namespace, out_path: Path, exc: Exception) -> dict[str, Any]:
    receipt = {
        "schema": SCHEMA,
        "verdict": "FAIL",
        "dataset_path": str(args.dataset),
        "artifact_dir": str(args.artifacts),
        "target": args.target,
        "use_treelite": not args.no_treelite,
        "engine": "error",
        "rows": 0,
        "accuracy": None,
        "macro_f1": None,
        "blockers": [f"{type(exc).__name__}:{exc}"],
    }
    write_json_receipt(out_path, receipt)
    return receipt


def apply_xgb_or_treelite(
    dataset: Path,
    artifacts: Path,
    out_path: Path,
    *,
    target: str,
    use_treelite: bool,
    limit: int | None,
) -> dict[str, Any]:
    from ahoy_sim.training.xgb_shap_pipeline import apply_xgb_or_treelite as apply_model

    return apply_model(dataset, artifacts, out_path, target=target, use_treelite=use_treelite, limit=limit)


def summary(receipt: dict[str, Any], out_path: Path) -> dict[str, Any]:
    compact = {
        "verdict": receipt.get("verdict", "UNKNOWN"),
        "engine": receipt.get("engine"),
        "rows": receipt.get("rows", 0),
        "accuracy": receipt.get("accuracy"),
        "macro_f1": receipt.get("macro_f1"),
        "receipt": str(out_path),
    }
    if receipt.get("blockers"):
        compact["blockers"] = receipt["blockers"]
    return compact


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_path = args.out or (OUT / "models" / f"treelite_apply_{stamp()}.json")

    blockers = preflight_blockers(args.dataset, args.artifacts, use_treelite=not args.no_treelite)
    if blockers:
        receipt = blocked_receipt(args, out_path, blockers)
    else:
        try:
            receipt = apply_xgb_or_treelite(
                args.dataset,
                args.artifacts,
                out_path,
                target=args.target,
                use_treelite=not args.no_treelite,
                limit=args.limit,
            )
        except Exception as exc:
            receipt = fail_receipt(args, out_path, exc)
    print(json.dumps(summary(receipt, out_path), sort_keys=True))
    return 0 if receipt.get("verdict") == "PASS" else 4


if __name__ == "__main__":
    raise SystemExit(main())
