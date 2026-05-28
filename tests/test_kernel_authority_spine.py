#!/usr/bin/env python3
from __future__ import annotations
import json, subprocess, sys, tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'scripts'))
from kernel_control_packet import absurd_enqueue_packet, verify_control_packet
from spine_kernel_authorization import validate_job_kernel_authorization
from ckdog_kernel_route_plan import route_plan_from_packet

def test_missing_expired_tampered_wrong_lane_wrong_path_replay_and_valid_route():
    with tempfile.TemporaryDirectory() as td:
        ledger=Path(td)/'ledger.jsonl'; events=Path(td)/'events.jsonl'; idem='kernel-spine-fixture'
        missing=validate_job_kernel_authorization(queue_name='korpus', job_kind='korpus_lane_job', payload={'lane':'intake','source_path':'README.md','bridge_version':'v2'})
        assert not missing.ok and missing.error_kind=='missing_kernel_authorization'
        expired=absurd_enqueue_packet(queue_name='korpus', lane='intake', source_path='README.md', idempotency_key=idem, authorized_by='operator_cli', expires_at=(datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat().replace('+00:00','Z'))
        ok,err=verify_control_packet(expired); assert not ok and 'expired' in err
        packet=absurd_enqueue_packet(queue_name='korpus', lane='intake', source_path='README.md', idempotency_key=idem, authorized_by='operator_cli')
        tampered=json.loads(json.dumps(packet)); tampered['payload']['lane']='other'; ok,err=verify_control_packet(tampered); assert not ok and 'hash mismatch' in err
        wrong_lane=validate_job_kernel_authorization(queue_name='korpus', job_kind='korpus_lane_job', payload={'lane':'other','source_path':'README.md','bridge_version':'v2','kernel_authorization':packet}); assert not wrong_lane.ok and wrong_lane.error_kind=='kernel_authorization_lane_mismatch'
        wrong_path=validate_job_kernel_authorization(queue_name='korpus', job_kind='korpus_lane_job', payload={'lane':'intake','source_path':'OTHER.md','bridge_version':'v2','kernel_authorization':packet}); assert not wrong_path.ok and wrong_path.error_kind=='kernel_authorization_source_path_mismatch'
        valid=validate_job_kernel_authorization(queue_name='korpus', job_kind='korpus_lane_job', payload={'lane':'intake','source_path':'README.md','bridge_version':'v2','kernel_authorization':packet}); assert valid.ok
        plan=route_plan_from_packet(packet, ledger_path=ledger, event_log=events); assert plan['status']=='ROUTED'; assert events.exists()
        replay=route_plan_from_packet(packet, ledger_path=ledger, event_log=events); assert replay['status']=='DENIED' and replay['error']=='packet_replay'

def test_kernel_packet_cli_routes_and_denies_replay():
    with tempfile.TemporaryDirectory() as td:
        packet=Path(td)/'packet.json'; ledger=Path(td)/'ledger.jsonl'; events=Path(td)/'events.jsonl'
        p=absurd_enqueue_packet(queue_name='korpus', lane='intake', source_path='README.md', idempotency_key='cli-fixture', authorized_by='operator_cli')
        packet.write_text(json.dumps(p))
        ok=subprocess.run([sys.executable,'scripts/kernel_packet_cli.py','route','--packet-file',str(packet),'--ledger-path',str(ledger),'--event-log',str(events)],cwd=ROOT,text=True,capture_output=True)
        assert ok.returncode==0, ok.stderr+ok.stdout
        denied=subprocess.run([sys.executable,'scripts/kernel_packet_cli.py','route','--packet-file',str(packet),'--ledger-path',str(ledger),'--event-log',str(events)],cwd=ROOT,text=True,capture_output=True)
        assert denied.returncode==2
        assert 'packet_replay' in denied.stdout


def test_kernel_packet_cli_make_absurd_alias_and_legacy_dbos_shape():
    for alias in ['make-absurd', 'make-dbos']:
        proc = subprocess.run([
            sys.executable,
            'scripts/kernel_packet_cli.py',
            alias,
            '--queue-name',
            'korpus',
            '--lane',
            'intake',
            '--source-path',
            'README.md',
            '--idempotency-key',
            f'alias-{alias}',
            '--authorized-by',
            'operator_cli',
        ], cwd=ROOT, text=True, capture_output=True)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        packet = json.loads(proc.stdout)
        assert packet['lane'] == 'absurd:korpus:intake'
        assert packet['packet_hash']
