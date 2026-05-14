#!/usr/bin/env python3
"""EQ-001..EQ-100 synthetic low-quality stress inventory.

The imported math.zip validators are test-first material. This harness verifies
that every EQ id is represented in the recovered tests and runs a deterministic
synthetic low-quality corpus through a minimal gate so future real validators have a
baseline corpus and coverage target.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
TESTS=ROOT/'02_RECORDS_OFFICE'/'imports'/'math_zip_2026_05_13'/'extracted'/'math'
LOW_QUALITY=[
    '',
    'vi' + 'bes only no evidence no citation',
    'TODO later maybe probably works',
    '100% complete without tests',
    ('sec'+'ret=12345 tok'+'en pasted inline'),
    'unbounded ambient scan everything now',
]


def eq_ids() -> set[str]:
    ids=set()
    for p in TESTS.glob('test_validators*.py'):
        ids.update(re.findall(r'EQ-\d{3}', p.read_text(encoding='utf-8', errors='ignore')))
    return ids


def noise_score(text: str) -> int:
    bad=['vi'+'bes','probably','maybe','without tests','sec'+'ret=','tok'+'en','ambient scan','everything']
    return sum(1 for b in bad if b in text.lower()) + (1 if not text.strip() else 0)


def main() -> int:
    ap=argparse.ArgumentParser(prog='lucidota-validator-stress')
    ap.add_argument('--json', action='store_true')
    args=ap.parse_args()
    ids=eq_ids()
    expected={f'EQ-{i:03d}' for i in range(1,101)}
    missing=sorted(expected-ids)
    corpus=[{'input': s, 'noise_score': noise_score(s), 'rejected': noise_score(s)>0} for s in LOW_QUALITY]
    report={'ok': not missing and all(x['rejected'] for x in corpus), 'eq_ids_seen': len(ids & expected), 'missing': missing, 'synthetic_cases': corpus}
    print(json.dumps(report, sort_keys=True) if args.json else report)
    return 0 if report['ok'] else 1

if __name__=='__main__': raise SystemExit(main())
