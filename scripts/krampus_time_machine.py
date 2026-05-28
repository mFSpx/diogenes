#!/usr/bin/env python3
"""KORPUS KRAMPII Time Machine orchestrator.

One command for the historical corpus run:
  1. build chrono-ledger from content/path/frontmatter dates;
  2. feed brain map in exact chronological order;
  3. run durable KORPUS ingest sequentially so River learns concept drift;
  4. leave vibe_telemetry in Postgres.

No LLM calls.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "bin" / "python"
if not PY.exists():
    PY = Path(sys.executable)
OUTPUT_ROOT = ROOT / "05_OUTPUTS" / "korpus_krampii"


def jdump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)


def run_cmd(args: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"[{dt.datetime.now().isoformat()}] $ {' '.join(args)}\n")
        fh.flush()
        subprocess.run(args, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT, check=True)


def build_ledger(root: Path, out: Path, args: argparse.Namespace, log: Path) -> dict[str, Any]:
    cmd = [str(PY), str(ROOT / "scripts" / "krampus_chrono_ledger.py"), "--json", "build", str(root), "--out", str(out)]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if args.include_hidden:
        cmd.append("--include-hidden")
    if args.no_default_excludes:
        cmd.append("--no-default-excludes")
    result = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True, check=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as fh:
        fh.write(result.stdout)
        if result.stderr:
            fh.write(result.stderr)
    return json.loads(result.stdout.strip().splitlines()[-1])


def brain_pass(ledger: Path, args: argparse.Namespace, log: Path) -> int:
    count = 0
    brain_state = Path(args.brain_state).expanduser()
    brain_map = Path(args.brain_map).expanduser()
    if args.reset_brain:
        brain_state.unlink(missing_ok=True)
        brain_map.unlink(missing_ok=True)
    with ledger.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            count += 1
            path = row["absolute_path"]
            cmd = [
                str(PY), str(ROOT / "scripts" / "lucidota_brain_ingest.py"),
                "--json", "--state-path", str(brain_state), "--map-jsonl", str(brain_map), path,
            ]
            try:
                run_cmd(cmd, log)
            except subprocess.CalledProcessError:
                if not args.keep_going:
                    raise
            if count % args.save_every == 0:
                with log.open("a", encoding="utf-8") as out:
                    out.write(f"[{dt.datetime.now().isoformat()}] brain checkpoint files={count}\n")
    return count


def korpus_pass(root: Path, args: argparse.Namespace, log: Path) -> None:
    cmd = [
        str(PY), str(ROOT / "scripts" / "korpus_krampii.py"), "--json", "ingest", str(root),
        "--case", args.case, "--label", args.label, "--chronological", "--workers", "1",
        "--max-file-mb", str(args.max_file_mb), "--max-text-mb", str(args.max_text_mb),
        "--commit-every", str(args.commit_every),
    ]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if args.include_hidden:
        cmd.append("--include-hidden")
    if args.no_default_excludes:
        cmd.append("--no-default-excludes")
    run_cmd(cmd, log)


def cmd_run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).expanduser().resolve()
    stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    ledger = Path(args.ledger).expanduser() if args.ledger else OUTPUT_ROOT / f"{stamp}.time_machine.chrono_ledger.csv"
    log = Path(args.log).expanduser()
    started = dt.datetime.now()
    ledger_result = build_ledger(root, ledger, args, log)
    brain_count = 0
    if args.mode in {"both", "brain-only"}:
        brain_count = brain_pass(ledger, args, log)
    if args.mode in {"both", "korpus-only"}:
        korpus_pass(root, args, log)
    return {
        "ok": True,
        "root": str(root),
        "mode": args.mode,
        "ledger": str(ledger),
        "ledger_rows": ledger_result.get("rows", 0),
        "brain_files": brain_count,
        "log": str(log),
        "elapsed_seconds": round((dt.datetime.now() - started).total_seconds(), 3),
    }


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="krampus-time-machine")
    ap.add_argument("--json", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("root")
    p.add_argument("--mode", choices=["both", "brain-only", "korpus-only"], default="both")
    p.add_argument("--case", default="KORPUS KRAMPII")
    p.add_argument("--label", default="time-machine-chronological")
    p.add_argument("--ledger", default="")
    p.add_argument("--log", default=str(ROOT / "04_RUNTIME" / "krampus_time_machine.log"))
    p.add_argument("--brain-state", default=str(ROOT / "03_VAULT" / "krampus_dbstream_brain.pkl"))
    p.add_argument("--brain-map", default=str(OUTPUT_ROOT / "krampus_brain_map.jsonl"))
    p.add_argument("--reset-brain", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--max-file-mb", type=int, default=64)
    p.add_argument("--max-text-mb", type=int, default=8)
    p.add_argument("--commit-every", type=int, default=25)
    p.add_argument("--include-hidden", action="store_true")
    p.add_argument("--no-default-excludes", action="store_true")
    p.add_argument("--keep-going", action="store_true", default=True)
    p.set_defaults(func=cmd_run)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    result = args.func(args)
    print(jdump(result) if args.json else json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True, default=str))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
