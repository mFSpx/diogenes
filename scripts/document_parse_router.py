#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from spine_common import rel, source_hash, sha256_json
from ocr_routing import classify_ocr
from media_metadata import extract_media_metadata, MEDIA_EXT
TEXT_EXT={'.md','.txt','.log','.py','.rs','.sql','.toml','.yaml','.yml','.html','.csv'}
JSON_EXT={'.json','.jsonl'}
ARCHIVE_EXT={'.zip','.tar','.gz','.tgz','.7z','.rar'}
DB_EXT={'.db','.sqlite','.sqlite3'}

def parse_file(path: str|Path, *, custody_ref: dict[str,Any]|None=None, ocr_available: bool=False) -> dict[str,Any]:
    p=Path(path); ext=p.suffix.lower(); ref=custody_ref or {}
    base={'source_path':rel(p),'source_sha256':source_hash(p),'custody_id':ref.get('occurrence_id'),'content_id':ref.get('content_id'),'blocks':[]}
    if ext in TEXT_EXT:
        text=p.read_text(encoding='utf-8', errors='replace')
        base.update({'status':'PARSED','text':text,'parse_method':'plain_text','blocks':[{'block_id':'block:'+sha256_json({'path':rel(p),'text':text})[:24],'start':0,'end':len(text),'text':text,'kind':'text'}]})
        return base
    if ext in JSON_EXT:
        raw=p.read_text(encoding='utf-8', errors='replace')
        try:
            obj=json.loads(raw)
            text=json.dumps(obj, sort_keys=True, indent=2)
            status='PARSED'; method='json'
        except Exception:
            text=raw; status='PARSED'; method='json_text_fallback'
        base.update({'status':status,'text':text,'parse_method':method,'blocks':[{'block_id':'block:'+sha256_json({'path':rel(p),'text':text})[:24],'start':0,'end':len(text),'text':text,'kind':'json'}]})
        return base
    if ext in MEDIA_EXT:
        meta=extract_media_metadata(p)
        base.update({'status':'PARSED','text':'','parse_method':'media_metadata','media_metadata':meta,'blocks':[{'block_id':'media:'+meta['source_sha256'][:24],'start':0,'end':0,'text':'','kind':'media_metadata','metadata':meta}]})
        return base
    if ext in ARCHIVE_EXT:
        base.update({'status':'QUARANTINED','text':'','parse_method':'archive_quarantine','quarantine_reason':'ARCHIVE_QUARANTINE'})
        return base
    if ext in DB_EXT:
        base.update({'status':'QUARANTINED','text':'','parse_method':'database_quarantine','quarantine_reason':'DATABASE_QUARANTINE'})
        return base
    ocr=classify_ocr(str(p), ocr_available=ocr_available)
    if ocr['ocr_required']:
        base.update({'status':'OCR_REQUIRED' if ocr_available else 'OCR_BLOCKED','text':'','parse_method':'ocr_route','ocr':ocr})
        return base
    base.update({'status':'UNSUPPORTED','text':'','parse_method':'unsupported','error':'UNSUPPORTED_FILE_TYPE'})
    return base
