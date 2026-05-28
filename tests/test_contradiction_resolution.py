#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'scripts'))
from contradiction_report import detect_contradictions, set_resolution

def test_contradictions_have_severity_refs_and_resolution_status():
    claims=[{'claim_id':'c1','claim_text':'Alice signed contract','source_ref':{'custody_id':'o1'},'entity_candidates':['Alice']},{'claim_id':'c2','claim_text':'Alice did not sign contract','source_ref':{'custody_id':'o2'},'entity_candidates':['Alice']}]
    recs=detect_contradictions(claims)
    assert len(recs)==1
    r=recs[0]
    assert r['severity']=='HIGH'
    assert r['resolution_status']=='OPEN'
    assert r['claim_ids']==['c1','c2']
    resolved=set_resolution(r,'RESOLVED',reason='operator supplied source')
    assert resolved['resolution_status']=='RESOLVED'
