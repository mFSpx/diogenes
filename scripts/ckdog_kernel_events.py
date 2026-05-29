#!/usr/bin/env python3
from __future__ import annotations
from typing import Any
import copy
from spine_common import ROOT, append_jsonl, now, rel, sha256_json
EVENT_LOG = ROOT / '05_OUTPUTS/kernel/kernel_events.jsonl'

def append_event(event_type: str, payload: dict[str,Any], *, event_log: str|None=None) -> dict[str,Any]:
    path = ROOT/event_log if event_log else EVENT_LOG
    safe_payload=copy.deepcopy(payload)
    row={'event_type':event_type,'payload':safe_payload,'created_at':now()}
    row['event_hash']=sha256_json(row)
    append_jsonl(path,row); row['event_log']=rel(path); return row
