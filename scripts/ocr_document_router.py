#!/usr/bin/env python3
"""Route files into OCR/document parsing states without pretending OCR ran."""
from __future__ import annotations
import argparse, importlib.util, json, mimetypes
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS/ocr'
OCR_EXT={'.pdf','.png','.jpg','.jpeg','.tif','.tiff','.bmp','.webp','.docx','.pptx','.xlsx'}
NO_OCR_EXT={'.txt','.md','.json','.jsonl','.py','.rs','.sql','.csv','.html','.xml','.yaml','.yml','.toml'}
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
    try: return str(Path(p).resolve().relative_to(ROOT))
    except Exception: return str(p)
def classify(path:Path):
    ext=path.suffix.lower(); mime=mimetypes.guess_type(str(path))[0]
    docling=bool(importlib.util.find_spec('docling'))
    if ext in NO_OCR_EXT: status='OCR_NOT_REQUIRED'
    elif ext in OCR_EXT: status='OCR_READY' if docling else 'OCR_BLOCKED'
    else: status='OCR_SKIPPED'
    return {'path':rel(path),'exists':path.exists(),'suffix':ext or '[none]','mime':mime,'ocr_required':ext in OCR_EXT,'docling_importable':docling,'status':status,'ocr_executed':False,'route_only':True}
def main():
    p=argparse.ArgumentParser(); p.add_argument('--path',required=True); p.add_argument('--execute',action='store_true'); p.add_argument('--dry-run',action='store_true')
    a=p.parse_args(); r={'schema':'diogenes.ocr_document_router.v1','generated_at':now(),'execute_performed':False,'classification':classify(Path(a.path))}
    OUT.mkdir(parents=True,exist_ok=True); out=OUT/f'ocr_document_router_{stamp()}.json'; r['receipt_path']=rel(out); out.write_text(json.dumps(r,indent=2),encoding='utf-8'); print('RECEIPT_PATH='+rel(out)); print('OCR_ROUTE_STATUS='+r['classification']['status']); return 0
if __name__=='__main__': raise SystemExit(main())
