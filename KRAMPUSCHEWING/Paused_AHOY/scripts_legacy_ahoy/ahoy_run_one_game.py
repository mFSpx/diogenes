#!/usr/bin/env python3
"""Ahoy run one game CLI wrapper for Ahoy receipts."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ahoy_sim.cli import build_policies
from ahoy_sim.engine.game import run_and_write_game
from ahoy_sim.engine.receipts import OUT

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
    ap.add_argument("--seed", type=int, default=414)
    ap.add_argument("--smuggler", default="random", choices=["random", "river"])
    ap.add_argument("--bluefin", default="heuristic", choices=["random", "heuristic", "strong"])
    ap.add_argument("--mollusk", default="heuristic", choices=["random", "heuristic", "strong"])
    ap.add_argument("--max-rounds", type=positive_int, default=6)
    ap.add_argument("--out", type=Path, default=OUT / "games")
    args = ap.parse_args()
    path, receipt = run_and_write_game(args.seed, build_policies(args.smuggler, args.bluefin, args.mollusk, args.seed), max_rounds=args.max_rounds, out_dir=args.out)
    print(json.dumps({"verdict": receipt["verdict"], "rules_verdict": receipt["rules_verdict"], "receipt": str(path), "illegal_actions": receipt["illegal_action_count"], "engine_exceptions": receipt["engine_exceptions"]}, sort_keys=True))
    return 0 if receipt["verdict"] in {"PASS", "DEGRADED"} else 2

if __name__ == "__main__":
    raise SystemExit(main())
