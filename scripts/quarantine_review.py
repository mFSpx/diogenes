#!/usr/bin/env python3
from __future__ import annotations
from typing import Any
from spine_common import sha256_json

def review_tasks_from_quarantine(rows: list[dict[str,Any]]) -> list[dict[str,Any]]:
    tasks=[]
    for r in rows:
        reason=r.get('quarantine_reason') or 'QUARANTINED'
        tasks.append({'work_order_id':'quarantine:'+sha256_json({'path':r.get('relative_path'),'reason':reason})[:24], 'kind':'quarantine_review', 'relative_path':r.get('relative_path'), 'custody_id':r.get('occurrence_id'), 'reason':reason, 'safe_options':['keep_quarantined','operator_approve_manual_extract','ignore_for_case'], 'operator_approval_required':True, 'status':'CREATED'})
    return tasks
