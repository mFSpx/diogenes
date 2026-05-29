#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from kernel_control_packet import require_control_packet, verify_control_packet
from control_packet_ledger import append_packet, packet_seen
from ckdog_kernel_events import append_event
from spine_common import sha256_json, now

def route_plan_from_packet(packet: dict[str,Any], *, purpose: str='mutation', ledger_path: str|Path|None=None, event_log: str|Path|None=None) -> dict[str,Any]:
    ok, err = verify_control_packet(packet)
    if not ok:
        append_packet(packet, purpose=purpose, accepted=False, denial_reason=err, ledger_path=ledger_path or Path('05_OUTPUTS/kernel/control_packet_ledger.jsonl'))
        return {'status':'DENIED','error':err,'packet_hash':packet.get('packet_hash')}
    if packet_seen(packet['packet_hash'], ledger_path or Path('05_OUTPUTS/kernel/control_packet_ledger.jsonl')):
        append_packet(packet, purpose=purpose, accepted=False, denial_reason='packet_replay', ledger_path=ledger_path or Path('05_OUTPUTS/kernel/control_packet_ledger.jsonl'))
        return {'status':'DENIED','error':'packet_replay','packet_hash':packet.get('packet_hash')}
    payload=packet.get('payload',{})
    plan={'status':'ROUTED','packet_hash':packet['packet_hash'],'route_id':'route-'+sha256_json(packet)[:16],'lane':packet.get('lane'),'queue_name':payload.get('queue_name'),'source_path':payload.get('source_path'),'idempotency_key':payload.get('idempotency_key'),'created_at':now(),'canonical_mutation_allowed':bool(payload.get('canonical_mutation_allowed') is True)}
    append_packet(packet, purpose=purpose, accepted=True, ledger_path=ledger_path or Path('05_OUTPUTS/kernel/control_packet_ledger.jsonl'))
    plan['event']=append_event('kernel.route_plan.created', plan, event_log=str(event_log) if event_log else None)
    return plan
