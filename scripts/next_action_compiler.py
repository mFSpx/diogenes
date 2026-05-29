#!/usr/bin/env python3
from __future__ import annotations
from typing import Any
from spine_common import sha256_json

def next_actions_from_gaps(items: list[dict[str,Any]]) -> list[dict[str,Any]]:
    return [{'action_id':'action:'+sha256_json(i)[:24], 'action_type':'ADD_EVIDENCE', 'target_ref':i.get('claim_cluster_id') or i.get('claim_id'), 'plain_language_instruction':f"Add or reject evidence for claim {i.get('claim_id')}", 'closure_gate':i['closure_gate'], 'source_missing_evidence_id':i['missing_evidence_id']} for i in items]
