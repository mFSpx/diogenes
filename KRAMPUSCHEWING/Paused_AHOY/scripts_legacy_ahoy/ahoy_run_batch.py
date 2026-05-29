#!/usr/bin/env python3
"""Ahoy run batch CLI wrapper for Ahoy receipts."""
from __future__ import annotations
import argparse, json, sys, os, shutil, shlex
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ahoy_sim.cli import build_policies
from ahoy_sim.engine.game import run_game, winner_summary
from ahoy_sim.engine.receipts import OUT, stamp, utc_now, write_json_receipt
from ahoy_sim.rules.rule_gaps import blocking_gaps
from ahoy_sim.training.dataset_writer import DatasetWriter
from ahoy_sim.training.dataset_audit import audit_dataset
from ahoy_sim.rules.validators import validate_training_row

def positive_int(value: str) -> int:
    try:
        out = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if out < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return out

def worker_arg(value: str) -> str:
    if value == "auto":
        return value
    positive_int(value)
    return value

def _run_chunk(payload):
    start, stop, base_seed, smuggler, bluefin, mollusk, max_rounds, shard_path = payload
    policies = build_policies(smuggler, bluefin, mollusk, base_seed + start)
    wins, scores = Counter(), {}
    rows = illegal = exceptions = done = 0
    shard = Path(shard_path)
    shard.parent.mkdir(parents=True, exist_ok=True)
    with shard.open("w", encoding="utf-8") as fh:
        for i in range(start, stop):
            state, game_rows, metrics = run_game(base_seed + i, policies, max_rounds=max_rounds, training_writer=None)
            summary = winner_summary(state)
            final_scores = summary["scores"]
            wins[summary["winner_class"]] += 1
            for f, score in final_scores.items(): scores.setdefault(f, []).append(score)
            for row in game_rows:
                data = row.as_dict()
                blockers = validate_training_row(data)
                if blockers:
                    raise ValueError("invalid_training_row:" + ",".join(blockers))
                fh.write(json.dumps(data, sort_keys=True, default=str) + "\n")
            rows += len(game_rows); illegal += metrics["illegal_action_count"]; exceptions += metrics["engine_exceptions"]; done += 1
    return {"wins": dict(wins), "scores": scores, "rows": rows, "illegal": illegal, "exceptions": exceptions, "done": done, "shard_path": str(shard)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=positive_int, required=True)
    ap.add_argument("--workers", type=worker_arg, default="1")
    ap.add_argument("--seed", type=int, default=414)
    ap.add_argument("--smuggler", default="river", choices=["random", "river"])
    ap.add_argument("--bluefin", default="strong", choices=["random", "heuristic", "strong"])
    ap.add_argument("--mollusk", default="strong", choices=["random", "heuristic", "strong"])
    ap.add_argument("--out", type=Path, default=OUT / "batches")
    ap.add_argument("--allow-partial-rules", action="store_true")
    ap.add_argument("--toy", action="store_true")
    ap.add_argument("--max-rounds", type=positive_int, default=6)
    ap.add_argument("--skip-dataset-audit", action="store_true")
    args = ap.parse_args()
    gaps = blocking_gaps()
    dataset_path = OUT / "training" / f"ahoy_training_rows_{stamp()}.jsonl"
    receipt = {
        "schema": "lucidota.ahoy.batch.v1",
        "created_at": utc_now(),
        "verdict": "PASS",
        "rules_verdict": "PASS",
        "blocking_rule_gaps": gaps,
        "games_requested": args.games,
        "games_completed": 0,
        "illegal_actions": 0,
        "engine_exceptions": 0,
        "policy_fallbacks": 0,
        "seed": args.seed,
        "policies": {},
        "win_rates": {},
        "score_stats": {},
        "training_rows_written": 0,
        "dataset_path": str(dataset_path),
        "commands_run": [shlex.join(sys.argv)],
        "files_written": [],
        "tests_run": [],
        "blockers": [],
    }
    if gaps and not (args.allow_partial_rules or args.toy):
        receipt.update({"verdict": "BLOCKED", "rules_verdict": "BLOCKED", "blockers": [g["gap_id"] for g in gaps] + ["rerun_with_--allow-partial-rules_for_toy_data_only"]})
        path = args.out / f"batch_{stamp()}.json"
        write_json_receipt(path, receipt)
        print(json.dumps({"verdict": "BLOCKED", "receipt": str(path), "games_completed": 0}, sort_keys=True))
        return 4
    if gaps:
        receipt.update({"verdict": "DEGRADED", "rules_verdict": "BLOCKED", "blockers": [g["gap_id"] for g in gaps]})
    wins: Counter[str] = Counter()
    scores: dict[str, list[int]] = {}
    policies = build_policies(args.smuggler, args.bluefin, args.mollusk, args.seed)
    receipt["policies"] = {f: p.metadata() for f, p in policies.items()}
    batch_max_rounds = args.max_rounds
    if args.workers == "auto" or int(args.workers or "1") > 1:
        n = os.cpu_count() if args.workers == "auto" else int(args.workers)
        n = max(1, min(n or 1, args.games))
        chunk = (args.games + n - 1) // n
        shard_dir = dataset_path.with_suffix("").parent / f"{dataset_path.stem}_shards"
        jobs = [(i, min(args.games, i + chunk), args.seed, args.smuggler, args.bluefin, args.mollusk, batch_max_rounds, str(shard_dir / f"part_{i:07d}_{min(args.games, i + chunk):07d}.jsonl")) for i in range(0, args.games, chunk)]
        shard_paths = []
        with ProcessPoolExecutor(max_workers=n) as ex:
            for result in ex.map(_run_chunk, jobs):
                receipt["games_completed"] += result["done"]
                receipt["training_rows_written"] += result["rows"]
                receipt["illegal_actions"] += result["illegal"]
                receipt["engine_exceptions"] += result["exceptions"]
                shard_paths.append(result["shard_path"])
                wins.update(result["wins"])
                for f, vals in result["scores"].items(): scores.setdefault(f, []).extend(vals)
        with dataset_path.open("wb") as out_fh:
            for shard in sorted(shard_paths):
                with Path(shard).open("rb") as in_fh:
                    shutil.copyfileobj(in_fh, out_fh, length=1024 * 1024)
        receipt["shard_dir"] = str(shard_dir)
        receipt["shard_count"] = len(shard_paths)
    else:
        writer = DatasetWriter(dataset_path)
        for i in range(args.games):
            state, rows, metrics = run_game(args.seed + i, policies, max_rounds=batch_max_rounds, training_writer=writer)
            summary = winner_summary(state)
            final_scores = summary["scores"]
            wins[summary["winner_class"]] += 1
            for f, score in final_scores.items(): scores.setdefault(f, []).append(score)
            receipt["games_completed"] += 1
            receipt["illegal_actions"] += metrics["illegal_action_count"]
            receipt["engine_exceptions"] += metrics["engine_exceptions"]
        receipt["training_rows_written"] = writer.rows_written
    receipt["files_written"] = [str(dataset_path)]
    if not args.skip_dataset_audit:
        audit = audit_dataset(dataset_path, min_distinct_labels_per_faction=3 if args.games < 1000 else 5)
        receipt["dataset_audit"] = audit
        receipt["tests_run"].append("dataset_quality_audit")
        if audit["verdict"] != "PASS":
            receipt["verdict"] = "FAIL"
            receipt["blockers"].extend(audit.get("blockers", []))
    receipt["win_rates"] = {f: wins[f] / max(1, args.games) for f in sorted(wins)}
    receipt["score_stats"] = {f: {"mean": sum(v) / len(v), "min": min(v), "max": max(v)} for f, v in scores.items() if v}
    path = args.out / f"batch_{stamp()}.json"
    write_json_receipt(path, receipt)
    print(json.dumps({"verdict": receipt["verdict"], "receipt": str(path), "games_completed": receipt["games_completed"], "training_rows_written": receipt["training_rows_written"]}, sort_keys=True))
    return 0 if receipt["verdict"] in {"PASS", "DEGRADED"} else 3

if __name__ == "__main__":
    raise SystemExit(main())
