#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'scripts'))
from graph_promotion import promote_candidate
from graph_store_adapter import GraphStoreAdapter
from operator_decisions import make_decision
from kernel_control_packet import absurd_enqueue_packet

def test_graph_candidate_requires_decision_and_kernel_authorization(tmp_path):
    cand={'candidate_id':'graphcand:1','label':'Alice saw Evidence','evidence_refs':[{'custody_id':'o1'}]}
    assert promote_candidate(cand,dry_run=True)['status']=='DRY_RUN'
    assert promote_candidate(cand,dry_run=False)['error']=='operator_decision_required'
    dec=make_decision(actor='operator',decision_type='APPROVE',target_ref='graphcand:1',reason='verified')
    assert promote_candidate(cand,decision=dec,dry_run=False)['error']=='kernel_authorization_required'
    packet=absurd_enqueue_packet(queue_name='graph',lane='promotion',source_path='case',idempotency_key='graphcand:1',authorized_by='operator')
    store=GraphStoreAdapter(tmp_path/'graph')
    res=promote_candidate(cand,decision=dec,kernel_packet=packet,graph_store=store,dry_run=False)
    assert res['status']=='MATERIALIZED'
    assert (tmp_path/'graph'/'journal.jsonl').exists()
