#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'scripts'))
from operator_decisions import make_decision, decision_allows

def test_operator_decision_receipt_records_actor_target_reason_and_effect():
    d=make_decision(actor='operator',decision_type='APPROVE',target_ref='graphcand:1',reason='source verified',downstream_effect='graph_promotion_allowed')
    assert d['decision_id'].startswith('decision:')
    assert d['actor']=='operator'
    assert decision_allows(d,target_ref='graphcand:1',required='APPROVE')
    try: make_decision(actor='operator',decision_type='YES',target_ref='x',reason='r')
    except ValueError: pass
    else: raise AssertionError('invalid decision accepted')
