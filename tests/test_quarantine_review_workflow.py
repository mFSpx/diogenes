#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'scripts'))
from quarantine_review import review_tasks_from_quarantine

def test_archive_and_db_quarantine_rows_create_operator_review_tasks():
    rows=[{'relative_path':'a.zip','occurrence_id':'o1','quarantine_reason':'ARCHIVE_QUARANTINE'}, {'relative_path':'b.db','occurrence_id':'o2','quarantine_reason':'DATABASE_QUARANTINE'}]
    tasks=review_tasks_from_quarantine(rows)
    assert len(tasks)==2
    assert all(t['operator_approval_required'] is True for t in tasks)
    assert all('operator_approve_manual_extract' in t['safe_options'] for t in tasks)
    assert {t['reason'] for t in tasks}=={'ARCHIVE_QUARANTINE','DATABASE_QUARANTINE'}
