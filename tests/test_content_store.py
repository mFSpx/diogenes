#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'scripts'))
from content_store import ContentStore

def test_content_store_dedupes_identical_content_and_changes_refs(tmp_path):
    store=ContentStore(tmp_path/'cas', case_id='case1')
    a=store.put_json({'x':1,'y':[2]})
    b=store.put_json({'y':[2],'x':1})
    c=store.put_json({'x':2})
    assert a['content_ref']==b['content_ref']
    assert a['content_ref']!=c['content_ref']
    assert store.get(a['content_ref'])
