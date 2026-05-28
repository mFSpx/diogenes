#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'scripts'))
from claim_support_score import score_claims

def test_claim_support_score_uses_sources_quotes_confidence_and_contradiction():
    claims=[{'claim_id':'1','claim_text':'A happened','confidence_bps':5000,'quote_text':'A happened','source_ref':{'custody_id':'a'},'staging_ref':'s1'}, {'claim_id':'2','claim_text':'A happened','confidence_bps':5000,'source_ref':{'custody_id':'b'},'staging_ref':'s2'}, {'claim_id':'3','claim_text':'B happened','confidence_bps':5000,'source_ref':{'custody_id':'c'},'contradiction_status':'OPEN'}]
    out=score_claims(claims)
    a=out[0]; b=out[2]
    assert a['support_score'] > b['support_score']
    assert a['support_reason']['supporting_source_count']==2
    assert a['support_reason']['quote_bonus']>0
    assert b['support_reason']['contradiction_penalty']>0
