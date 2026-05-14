#!/usr/bin/env python3
"""Render LOCAL_READS runtime contract v0 from project brain."""
from pathlib import Path
import json, argparse
ROOT=Path(__file__).resolve().parents[1]
CONTRACT=ROOT/'00_PROJECT_BRAIN'/'LOCAL_READS_RUNTIME_CONTRACT.md'
STATUS=ROOT/'00_PROJECT_BRAIN'/'STATUS.md'

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--json', action='store_true')
    args=ap.parse_args()
    data={'ok': CONTRACT.exists(), 'contract_path': str(CONTRACT.relative_to(ROOT)), 'status_path': str(STATUS.relative_to(ROOT)), 'contract_head': CONTRACT.read_text().splitlines()[:24] if CONTRACT.exists() else []}
    print(json.dumps(data, sort_keys=True) if args.json else '\n'.join(data['contract_head']))
    return 0 if data['ok'] else 1
if __name__=='__main__': raise SystemExit(main())
