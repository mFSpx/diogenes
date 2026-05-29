#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'scripts'))
from ocr_backlog import ocr_jobs_from_parse_records

def test_pdf_and_image_parse_records_become_ocr_backlog_jobs():
    records=[{'status':'OCR_BLOCKED','source_path':'a.pdf','custody_id':'o1'}, {'status':'OCR_REQUIRED','source_path':'b.png','custody_id':'o2'}, {'status':'PARSED','source_path':'c.md','custody_id':'o3'}]
    jobs=ocr_jobs_from_parse_records(records, ocr_allowed=False, ocr_available=False)
    assert len(jobs)==2
    assert all(j['status']=='BLOCKED' for j in jobs)
    assert all(j['blocker']=='OCR_NOT_ALLOWED_OR_BACKEND_MISSING' for j in jobs)
    ready=ocr_jobs_from_parse_records(records, ocr_allowed=True, ocr_available=True)
    assert all(j['status']=='CREATED' for j in ready)
