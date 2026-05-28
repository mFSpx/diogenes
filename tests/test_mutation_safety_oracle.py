#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'scripts'))
from mutation_safety_oracle import enforce_write_scope

def test_mutation_safety_oracle_blocks_out_of_scope_write(tmp_path):
    root=tmp_path/'root'; allowed=root/'allowed'; root.mkdir(); allowed.mkdir()
    ok=enforce_write_scope(repo_root=root, allowed_roots=[allowed], fn=lambda: (allowed/'x.txt').write_text('ok'))
    assert ok['status']=='PASSED'
    try: enforce_write_scope(repo_root=root, allowed_roots=[allowed], fn=lambda: (root/'bad.txt').write_text('bad'))
    except RuntimeError as e: assert 'write_scope_violation' in str(e)
    else: raise AssertionError('out of scope write accepted')
