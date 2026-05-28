#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'scripts'))
from text_chunker import chunk_text
from chunk_to_staging import chunks_to_staging
from claim_table_compiler import compile_claim_table
from source_quote_extractor import attach_quote_context

def test_claim_rows_include_quote_and_context_when_chunk_text_available():
    text='Before context. Alice saw Evidence. After context.'
    chunks=chunk_text(text, source_ref={'custody_id':'occ-1','content_id':'sha256:x','source_path':'a.md','source_sha256':'x'}, max_chars=200)
    staging=chunks_to_staging(chunks)
    chunk_texts={chunks[0]['chunk_id']:chunks[0]['text']}
    table=compile_claim_table(staging, chunk_texts=chunk_texts)
    claim=next(c for c in table['claims'] if 'Alice saw Evidence' in c['claim_text'])
    assert 'Alice saw Evidence' in claim['quote_text']
    assert claim['context_before'] is not None
    assert claim['context_after'] is not None
