#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MATRIX_FIXTURE = Path(__file__).resolve().parents[1] / 'tests/fixtures/matrix_trace/valid_receipt.json'

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '05_OUTPUTS/release'
REQUIRED = [
    'tests/test_lucidota_acceptance.py',
    'tests/test_demo_product_snapshot.py',
    'tests/test_mutation_safety_oracle.py',
    'tests/test_source_linked_claims.py',
    'tests/test_case_workspace_isolation.py',
    'tests/test_dev_order_matrix_wrapper.py',
    'tests/test_matrix_trace_checker.py',
    'tests/test_dev_order_gate.py',
    'tests/test_strict_model_stack_admission.py',
    'tests/test_graph_materialization_command_policy.py',
    'tests/test_graph_promotion_gate_safety.py',
]


def ts() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')


def rel(p: str | Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)



def load_matrix_trace() -> list[dict[str, Any]]:
    return json.loads(MATRIX_FIXTURE.read_text(encoding='utf-8'))['matrix_trace']

def run(cmd: list[str]) -> dict[str, Any]:
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    refs = []
    for line in (proc.stdout + '\n' + proc.stderr).splitlines():
        if line.startswith('REPORT_PATH='):
            refs.append(line.split('=', 1)[1].strip())
    return {'command': ' '.join(cmd), 'returncode': proc.returncode, 'stdout': proc.stdout, 'stderr': proc.stderr, 'report_paths': refs, 'status': 'PASSED' if proc.returncode == 0 else 'FAILED'}


def run_gate(extra: list[str] | None = None) -> dict[str, Any]:
    tests = REQUIRED + (extra or [])
    steps = [
        run([sys.executable, '-m', 'pytest', *tests, '-q']),
        run([sys.executable, 'scripts/run_dev_order_methodology_checks.py']),
    ]
    blockers = [s['command'] for s in steps if s['returncode'] != 0]
    combined_stdout = ''.join(s.get('stdout', '') for s in steps)
    combined_stderr = ''.join(s.get('stderr', '') for s in steps)
    return {'status': 'PASSED' if not blockers else 'FAILED', 'steps': steps, 'blockers': blockers, 'required_tests': tests, 'receipts_written': sorted({p for s in steps for p in s.get('report_paths', [])}), 'stdout': combined_stdout, 'stderr': combined_stderr}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', action='store_true')
    args = ap.parse_args()
    res = run_gate()
    payload = {
        'schema': 'lucidota.release_gate.v2',
        'generated_at': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        **res,
        'matrix_trace': load_matrix_trace(),
        'commands_run': list(res['steps']),
        'files_changed': ['scripts/lucidota_release_gate.py', 'scripts/run_dev_order_methodology_checks.py', 'scripts/dev_order_gate.py'],
        'canonical_graph_materialization': False,
        'canonical_graph_writes': False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f'lucidota_release_gate_{ts()}.json'
    payload['report_path'] = rel(out)
    out.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding='utf-8')
    if payload['status'] == 'PASSED':
        gate = run([sys.executable, 'scripts/dev_order_gate.py', '--receipt', rel(out)])
        payload['commands_run'].append(gate)
        payload['self_gate'] = gate
        payload['receipts_written'] = sorted(set(payload['receipts_written'] + gate.get('report_paths', [])))
        if gate['returncode'] != 0:
            payload['blockers'].append('self_dev_order_gate_failed')
            payload['status'] = 'FAILED'
        out.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding='utf-8')
    print(json.dumps(payload, sort_keys=True) if args.json else payload['steps'][0]['stdout'])
    print('REPORT_PATH=' + rel(out))
    print('LUCIDOTA_RELEASE_GATE=' + payload['status'])
    return 0 if payload['status'] == 'PASSED' else 2


if __name__ == '__main__':
    raise SystemExit(main())
