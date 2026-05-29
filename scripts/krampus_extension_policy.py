#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
TEXT_EXT={'.md','.txt','.log','.py','.rs','.sql','.toml','.yaml','.yml','.html','.csv'}
JSON_EXT={'.json','.jsonl'}
PDF_EXT={'.pdf'}
IMAGE_EXT={'.png','.jpg','.jpeg','.webp','.gif','.bmp'}
MEDIA_EXT={'.mp4','.mov','.mkv','.opus','.mp3','.wav','.m4a'}
ARCHIVE_EXT={'.zip','.tar','.gz','.tgz','.7z','.rar'}
DB_EXT={'.db','.sqlite','.sqlite3'}

def classify_path(path: str|Path) -> dict:
    p=Path(path); ext=p.suffix.lower()
    if ext in TEXT_EXT: return {'kind':'text','lane':'parse_text','quarantine':False,'reason':''}
    if ext in JSON_EXT: return {'kind':'json','lane':'parse_json','quarantine':False,'reason':''}
    if ext in PDF_EXT: return {'kind':'pdf','lane':'ocr_or_pdf_parse','quarantine':False,'reason':'OCR_REQUIRED'}
    if ext in IMAGE_EXT: return {'kind':'image','lane':'ocr_image','quarantine':False,'reason':'OCR_REQUIRED'}
    if ext in MEDIA_EXT: return {'kind':'media','lane':'media_metadata','quarantine':False,'reason':'METADATA_ONLY'}
    if ext in ARCHIVE_EXT: return {'kind':'archive','lane':'quarantine_archive','quarantine':True,'reason':'ARCHIVE_QUARANTINE'}
    if ext in DB_EXT: return {'kind':'database','lane':'quarantine_database','quarantine':True,'reason':'DATABASE_QUARANTINE'}
    return {'kind':'binary','lane':'quarantine_unknown','quarantine':True,'reason':'UNKNOWN_BINARY_QUARANTINE'}
