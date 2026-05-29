#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'scripts'))
from missing_evidence_checklist import missing_evidence_items
from next_action_compiler import next_actions_from_gaps
from work_order_importer import work_orders_from_next_actions

def test_unsupported_claim_becomes_work_order_with_closure_gate():
    claims=[{'claim_id':'c1','claim_text':'Maybe Alice signed','modality':'HYPOTHESIS','cluster_id':'cl1','source_ref':{'custody_id':'o1'}}]
    gaps=missing_evidence_items(claims)
    actions=next_actions_from_gaps(gaps)
    work=work_orders_from_next_actions(actions)
    assert gaps and actions and work
    assert work[0]['target_ref']=='cl1'
    assert 'closure_gate' in work[0]
    assert work[0]['status']=='CREATED'
