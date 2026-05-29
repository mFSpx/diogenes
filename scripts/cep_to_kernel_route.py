#!/usr/bin/env python3
from __future__ import annotations
from typing import Any
from kernel_control_packet import absurd_enqueue_packet
from ckdog_kernel_route_plan import route_plan_from_packet

def route_cep_to_pipeline(cep: dict[str,Any], *, source_folder: str, case_id: str, ledger_path=None, event_log=None) -> dict[str,Any]:
    if cep.get('authority_class') != 'operator_authored_assertion':
        return {'status':'DENIED','error':'authority_class_not_operator'}
    packet=absurd_enqueue_packet(queue_name='pipeline', lane='product_pipeline', source_path=source_folder, idempotency_key=case_id, authorized_by=cep.get('source','operator_cli'))
    plan=route_plan_from_packet(packet, purpose='pipeline_submission', ledger_path=ledger_path, event_log=event_log)
    return {'status':plan['status'],'cep_id':cep['command_id'],'control_packet':packet,'route_plan':plan}
