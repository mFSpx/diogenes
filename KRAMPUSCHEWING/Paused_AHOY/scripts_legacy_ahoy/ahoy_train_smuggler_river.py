#!/usr/bin/env python3
"""Ahoy train smuggler river CLI wrapper for Ahoy receipts."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ahoy_sim.cli import build_policies
from ahoy_sim.engine.game import run_game
from ahoy_sim.engine.receipts import OUT, stamp, utc_now, write_json_receipt
from ahoy_sim.rules.enums import Faction
from ahoy_sim.rules.rule_gaps import blocking_gaps
from ahoy_sim.training.dataset_writer import DatasetWriter

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
    ap.add_argument("--games", type=positive_int, default=10)
    ap.add_argument("--seed", type=int, default=414)
    ap.add_argument("--allow-partial-rules", action="store_true")
    args = ap.parse_args()
    gaps = blocking_gaps()
    dataset = OUT / "training" / f"river_smuggler_rows_{stamp()}.jsonl"
    receipt = {"schema": "lucidota.ahoy.river_training.v1", "created_at": utc_now(), "verdict": "PASS", "rules_verdict": "PASS", "blocking_rule_gaps": gaps, "games_requested": args.games, "games_completed": 0, "rows_written": 0, "model": None, "blockers": []}
    if gaps and not args.allow_partial_rules:
        receipt.update({"verdict": "BLOCKED", "rules_verdict": "BLOCKED", "blockers": [g["gap_id"] for g in gaps]})
        path = OUT / "training" / f"training_{stamp()}.json"
        write_json_receipt(path, receipt)
        print(json.dumps({"verdict": "BLOCKED", "receipt": str(path)}))
        return 4
    if gaps:
        receipt.update({"verdict": "DEGRADED", "rules_verdict": "BLOCKED", "blockers": [g["gap_id"] for g in gaps]})
    policies = build_policies("river", "strong", "strong", args.seed)
    writer = DatasetWriter(dataset)
    smuggler = policies[Faction.SMUGGLER.value]
    for i in range(args.games):
        state, rows, _ = run_game(args.seed + i, policies, max_rounds=4, training_writer=writer)
        for row in rows:
            if row.faction == Faction.SMUGGLER.value and hasattr(smuggler, "learn"):
                smuggler.learn(row)
        receipt["games_completed"] += 1
    receipt["rows_written"] = writer.rows_written
    receipt["dataset_path"] = str(dataset)
    receipt["model"] = smuggler.metadata()
    path = OUT / "training" / f"training_{stamp()}.json"
    write_json_receipt(path, receipt)
    print(json.dumps({"verdict": receipt["verdict"], "receipt": str(path), "rows_written": writer.rows_written}, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
