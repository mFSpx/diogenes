#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, glob
from datetime import datetime, timezone
from pathlib import Path
import psycopg

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '05_OUTPUTS/production'
SCHEMA = ROOT / '06_SCHEMA/106_production_readiness.sql'


def ts() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')


def rel(p):
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def latest(pattern):
    xs = glob.glob(str(ROOT / pattern))
    return rel(max(xs, key=os.path.getmtime)) if xs else None


def read_status(path):
    if not path:
        return None
    try:
        data = json.loads((ROOT / path).read_text(encoding='utf-8'))
        return data.get('status') or data.get('verdict') or data.get('readiness_status')
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--sign-off', action='store_true')
    a = ap.parse_args()
    evidence = {
        'mega_gate': latest('05_OUTPUTS/mega_gate/lucidota_mega_gate_*.json'),
        'ci': latest('05_OUTPUTS/ci/lucidota_ci_gate_*.json'),
        'dev_order_methodology': latest('05_OUTPUTS/dev_order_matrix/dev_order_methodology_*.json'),
        'chrono_service': 'scripts/check_chrono_ledger_service.sh',
    }
    blockers = [k + '_missing' for k, v in evidence.items() if not v]
    for k in ('mega_gate', 'ci', 'dev_order_methodology'):
        if evidence.get(k) and read_status(evidence[k]) not in {'PASS', 'PASSED'}:
            blockers.append(k + '_not_pass')
    status = 'signed_off' if a.sign_off and not blockers else 'pass' if not blockers else 'blocked'
    if a.execute:
        with psycopg.connect(os.environ.get('DBOS_SYSTEM_DATABASE_URL') or 'postgresql:///lucidota_state') as c:
            with c.cursor() as cur:
                cur.execute(SCHEMA.read_text())
                for k, v in evidence.items():
                    cur.execute("INSERT INTO lucidota_control.production_readiness_item(readiness_key,subsystem,required_evidence,current_status,blocker,evidence_refs) VALUES (%s,%s,%s,%s,%s,%s::jsonb) ON CONFLICT(readiness_key) DO UPDATE SET current_status=EXCLUDED.current_status, blocker=EXCLUDED.blocker, evidence_refs=EXCLUDED.evidence_refs, updated_at=now()", (k, k, v or 'missing', status, ';'.join(blockers), json.dumps([v] if v else [])))
            c.commit()
    payload = {'action': 'production_readiness_eval', 'execute_performed': a.execute, 'sign_off_requested': a.sign_off, 'evidence': evidence, 'blockers': blockers, 'status': 'PASS' if not blockers else 'FAIL', 'readiness_status': status, 'canonical_graph_materialization': False, 'canonical_graph_writes': False}
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f'production_readiness_eval_{ts()}.json'
    payload['report_path'] = rel(p)
    p.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print('REPORT_PATH=' + rel(p))
    print('PRODUCTION_READINESS=' + payload['status'])
    return 0 if not blockers else 4


if __name__ == '__main__':
    raise SystemExit(main())
