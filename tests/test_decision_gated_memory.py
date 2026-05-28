#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'scripts'))
from memory_candidates import candidates_from_claims, apply_memory_decision
from operator_decisions import make_decision

def test_memory_candidate_requires_operator_decision_to_accept():
    claims=[{'claim_id':'c1','claim_text':'Alice prefers tea','modality':'FACT','source_ref':{'custody_id':'o1'}}]
    cand=candidates_from_claims(claims)[0]
    assert cand['status']=='CANDIDATE'
    try: apply_memory_decision(cand, make_decision(actor='operator',decision_type='APPROVE',target_ref=cand['memory_id'],reason='x'))
    except ValueError: pass
    else: raise AssertionError('approved memory without ACCEPT')
    acc=apply_memory_decision(cand, make_decision(actor='operator',decision_type='ACCEPT',target_ref=cand['memory_id'],reason='true preference'))
    assert acc['status']=='ACCEPTED'
    rej=apply_memory_decision(cand, make_decision(actor='operator',decision_type='REJECT',target_ref=cand['memory_id'],reason='bad evidence'))
    assert rej['status']=='REJECTED'
