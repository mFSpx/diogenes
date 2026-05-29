#!/usr/bin/env python3
from __future__ import annotations
import json, sys, zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'scripts'))
from product_intake import intake_folder
from product_parse_pipeline import parse_custody_package

def test_custody_package_becomes_parse_records_chunks_and_blockers(tmp_path):
    src=tmp_path/'messy'; src.mkdir()
    (src/'note.md').write_text('Alpha claim. Beta follows.')
    (src/'events.log').write_text('2026-05-17 event happened')
    (src/'data.json').write_text('{"fact":"json works"}')
    (src/'paper.pdf').write_bytes(b'%PDF-1.4')
    (src/'image.png').write_bytes(b'PNG')
    (src/'movie.mp4').write_bytes(b'MP4')
    (src/'local.db').write_bytes(b'SQLite format 3')
    with zipfile.ZipFile(src/'archive.zip','w') as z: z.writestr('x.txt','x')
    intake=intake_folder(src, output_dir=tmp_path/'custody', max_files=50)
    parse=parse_custody_package(tmp_path/'custody'/'custody_package.json', source_root=src, output_dir=tmp_path/'parse', max_chunk_chars=12)
    statuses={r['source_path'].split('/')[-1]:r['status'] for r in parse['parse_records']}
    assert statuses['note.md']=='PARSED'
    assert statuses['events.log']=='PARSED'
    assert statuses['data.json']=='PARSED'
    assert statuses['paper.pdf']=='OCR_BLOCKED'
    assert statuses['image.png']=='OCR_BLOCKED'
    assert statuses['movie.mp4']=='PARSED'
    assert statuses['local.db']=='QUARANTINED'
    assert statuses['archive.zip']=='QUARANTINED'
    assert parse['chunks']
    assert all('custody_id' in c['source_ref'] for c in parse['chunks'])
    assert Path(ROOT/parse['package']['parse_records_path']).exists()
    assert Path(ROOT/parse['package']['chunks_path']).exists()
    assert parse['receipt_path']
