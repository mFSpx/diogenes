#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
from typing import Any
from spine_common import sha256_json, rel, now

def source_bundle_from_intake(intake_receipt: dict[str,Any], *, safety_policy: dict[str,Any]|None=None) -> dict[str,Any]:
    normal=intake_receipt.get('normal',[]); quarantine=intake_receipt.get('quarantine',[])
    rows=normal+quarantine
    bundle={'schema':'lucidota.source_bundle.v1','bundle_id':'bundle:'+sha256_json({'source':intake_receipt['package']['source_root'],'rows':[(r.get('relative_path'),r.get('sha256')) for r in rows]})[:24], 'root_path':intake_receipt['package']['source_root'], 'file_count':len(rows), 'total_bytes':sum(int(r.get('size_bytes') or 0) for r in rows), 'normal_count':len(normal), 'quarantine_count':len(quarantine), 'duplicate_groups':len(intake_receipt.get('duplicates',{})), 'content_ids':[r.get('content_id') for r in rows if r.get('content_id')], 'occurrence_ids':[r.get('occurrence_id') for r in rows], 'safety_policy':{**(safety_policy or {'max_files':intake_receipt.get('cursor',{}).get('max_files')}), 'destructive_actions_allowed': False}, 'created_at':now()}
    return bundle
