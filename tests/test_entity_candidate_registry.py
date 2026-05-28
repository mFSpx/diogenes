#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'scripts'))
from entity_candidate_registry import build_entity_registry

def test_entity_registry_dedupes_exact_names_with_aliases_and_refs():
    rows=[{'staging_id':'s1','entity_candidates':['Alice','Evidence'],'source_ref':{'custody_id':'o1'},'source_span':{'start':0,'end':10},'confidence_bps':7000}, {'staging_id':'s2','entity_candidates':['Alice'],'source_ref':{'custody_id':'o2'},'source_span':{'start':2,'end':12},'confidence_bps':5000}]
    reg=build_entity_registry(rows)
    alice=next(e for e in reg if e['canonical_name']=='Alice')
    assert alice['status']=='CANDIDATE'
    assert len(alice['source_refs'])==2
    assert alice['confidence_bps']==7000
