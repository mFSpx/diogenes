#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'scripts'))
from claim_clusterer import cluster_claims

def test_duplicate_claims_share_cluster_id():
    claims=[{'claim_id':'1','claim_text':'Alice saw Evidence.'},{'claim_id':'2','claim_text':'alice saw evidence'},{'claim_id':'3','claim_text':'Bob disagrees'}]
    out=cluster_claims(claims)
    a=[c for c in out if c['claim_text'].lower().startswith('alice')]
    assert len({c['cluster_id'] for c in a})==1
    assert a[0]['cluster_size']==2
    assert next(c for c in out if c['claim_id']=='3')['cluster_size']==1
