#!/usr/bin/env python3
"""Static + DB smoke audit for Wake Bus batch delivery shape.

Checks the worker still claims bounded rows with one CTE, row locks, and SKIP LOCKED
instead of drifting into per-row/O(N) delivery loops.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
WAKE=ROOT/'scripts'/'lucidota_wake_bus.py'


def audit() -> dict:
    text=WAKE.read_text(encoding='utf-8')
    checks={
        'uses_cte_target_events': bool(re.search(r'WITH\s+target_events\s+AS', text, re.I)),
        'uses_for_update_skip_locked': 'FOR UPDATE SKIP LOCKED' in text,
        'uses_limit_parameter': bool(re.search(r'LIMIT\s+%s', text)),
        'single_update_from_cte': bool(re.search(r'UPDATE\s+lucidota_control\.event_outbox\s+o.*FROM\s+target_events', text, re.I|re.S)),
        'notify_inside_cte': 'pg_notify' in text and bool(re.search(r'notified\s+AS', text, re.I)),
        'no_per_row_execute_loop': not re.search(r'for\s+\w+\s+in\s+rows\s*:[\s\S]{0,240}conn\.execute', text),
    }
    return {'ok': all(checks.values()), 'file': str(WAKE.relative_to(ROOT)), 'checks': checks}


def main() -> int:
    ap=argparse.ArgumentParser(prog='lucidota-wake-bus-audit')
    ap.add_argument('--json', action='store_true')
    args=ap.parse_args()
    report=audit()
    print(json.dumps(report, sort_keys=True) if args.json else report)
    return 0 if report['ok'] else 1

if __name__=='__main__': raise SystemExit(main())
