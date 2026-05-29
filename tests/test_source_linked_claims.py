#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'scripts'))
from text_chunker import chunk_text
from chunk_to_staging import chunks_to_staging
from claim_table_compiler import compile_claim_table

def test_claim_table_requires_file_occurrence_chunk_and_source_span():
    chunks=chunk_text('Alice saw Evidence.', source_ref={'custody_id':'occ-1','content_id':'sha256:x','source_path':'a.md','source_sha256':'x'}, max_chars=100)
    staging=chunks_to_staging(chunks)
    table=compile_claim_table(staging)
    claim=table['claims'][0]
    assert claim['source_ref']['custody_id']=='occ-1'
    assert claim['source_ref']['chunk_id']==chunks[0]['chunk_id']
    assert claim['source_span']['start']==0
    assert claim['source_span']['end']>0
    bad=dict(staging[0]); bad['source_span']={}
    try: compile_claim_table([bad])
    except ValueError as e: assert 'source span' in str(e)
    else: raise AssertionError('bad claim accepted')
