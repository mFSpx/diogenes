#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from kernel_control_packet import absurd_enqueue_packet, verify_control_packet
from ckdog_kernel_route_plan import route_plan_from_packet
from spine_common import receipt

def main() -> int:
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest='cmd', required=True)
    mk=sub.add_parser('make-absurd', aliases=['make-dbos']); mk.add_argument('--queue-name',required=True); mk.add_argument('--lane',required=True); mk.add_argument('--source-path',required=True); mk.add_argument('--idempotency-key',required=True); mk.add_argument('--authorized-by',required=True); mk.add_argument('--expires-at')
    vf=sub.add_parser('validate'); vf.add_argument('--packet-file',required=True)
    rt=sub.add_parser('route'); rt.add_argument('--packet-file',required=True); rt.add_argument('--ledger-path'); rt.add_argument('--event-log')
    a=ap.parse_args()
    if a.cmd in {'make-absurd','make-dbos'}:
        print(json.dumps(absurd_enqueue_packet(queue_name=a.queue_name,lane=a.lane,source_path=a.source_path,idempotency_key=a.idempotency_key,authorized_by=a.authorized_by,expires_at=a.expires_at),sort_keys=True)); return 0
    packet=json.loads(Path(a.packet_file).read_text())
    if a.cmd=='validate':
        ok,err=verify_control_packet(packet); print(json.dumps({'authorized':ok,'error':err,'packet_hash':packet.get('packet_hash')},sort_keys=True)); return 0 if ok else 2
    plan=route_plan_from_packet(packet, ledger_path=a.ledger_path, event_log=a.event_log); receipt('kernel_packet_route', plan, root='05_OUTPUTS/kernel'); print(json.dumps(plan,sort_keys=True)); return 0 if plan['status']=='ROUTED' else 2
if __name__=='__main__': raise SystemExit(main())
