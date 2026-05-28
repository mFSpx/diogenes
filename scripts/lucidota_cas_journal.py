#!/usr/bin/env python3
"""Append durable local CAS ingest journal records.

This is the filesystem half of dual-write recovery: if bytes land but the DB
metadata commit dies, the ignored journal gives the reconciler source context.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_JOURNAL=ROOT/'03_VAULT'/'journal'/'cas_ingest.jsonl'


def append_record(record: dict, journal: Path = DEFAULT_JOURNAL) -> None:
    journal.parent.mkdir(parents=True, exist_ok=True)
    payload={**record, 'ts': time.time()}
    with journal.open('a', encoding='utf-8') as fh:
        fh.write(json.dumps(payload, sort_keys=True)+'\n')
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError as exc:
            print(f"cas journal fsync warning: {exc}", file=sys.stderr)


def load_records(journal: Path = DEFAULT_JOURNAL) -> list[dict]:
    if not journal.exists(): return []
    out=[]
    for line in journal.read_text(errors='ignore').splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:
            print(f"cas journal corrupt line skipped: {exc}", file=sys.stderr)
    return out


def main()->int:
    ap=argparse.ArgumentParser(prog='lucidota-cas-journal')
    ap.add_argument('--json', action='store_true')
    ap.add_argument('--journal', type=Path, default=DEFAULT_JOURNAL)
    args=ap.parse_args()
    records=load_records(args.journal)
    report={'ok':True,'journal':str(args.journal),'records':len(records)}
    print(json.dumps(report, sort_keys=True) if args.json else report)
    return 0
if __name__=='__main__': raise SystemExit(main())
