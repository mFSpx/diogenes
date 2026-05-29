#!/usr/bin/env python3
from __future__ import annotations
from typing import Any
from operator_decisions import decision_allows
from graph_store_adapter import GraphStoreAdapter

def promote_candidate(candidate: dict[str,Any], *, decision: dict[str,Any]|None=None, kernel_packet: dict[str,Any]|None=None, graph_store: GraphStoreAdapter|None=None, dry_run: bool=True) -> dict[str,Any]:
    if not candidate.get('candidate_id') or not candidate.get('evidence_refs'):
        return {'status':'REJECTED','error':'invalid_candidate'}
    if dry_run:
        return {'status':'DRY_RUN','candidate_id':candidate['candidate_id'],'canonical_mutation_allowed':False}
    if not decision or not decision_allows(decision,target_ref=candidate['candidate_id'],required='APPROVE'):
        return {'status':'REJECTED','error':'operator_decision_required'}
    if not kernel_packet:
        return {'status':'REJECTED','error':'kernel_authorization_required'}
    item=(graph_store or GraphStoreAdapter('05_OUTPUTS/product_graph/store')).materialize_candidate(candidate, decision=decision, kernel_packet=kernel_packet)
    return {'status':'MATERIALIZED','item':item}
