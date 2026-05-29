#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from spine_common import sha256_json, append_jsonl, write_json, now

class GraphStoreAdapter:
    def __init__(self, root: str|Path):
        self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True); self.items=self.root/'items.jsonl'; self.journal=self.root/'journal.jsonl'; self.state=self.root/'state.json'
    def _state(self): return json.loads(self.state.read_text()) if self.state.exists() else {'items':{},'edges':{}}
    def _save(self,s): write_json(self.state,s)
    def materialize_candidate(self, candidate: dict[str,Any], *, decision: dict[str,Any], kernel_packet: dict[str,Any]|None=None) -> dict[str,Any]:
        if decision.get('decision_type')!='APPROVE' or decision.get('target_ref') != candidate['candidate_id']:
            raise ValueError('operator decision required for graph materialization')
        if not kernel_packet: raise ValueError('kernel authorization required')
        s=self._state(); item_id='graphitem:'+sha256_json(candidate)[:24]
        if item_id in s['items']: raise ValueError('duplicate graph item')
        item={'item_id':item_id,'label':candidate.get('label'),'candidate_ref':candidate['candidate_id'],'evidence_refs':candidate.get('evidence_refs',[]),'created_at':now()}
        s['items'][item_id]=item; self._save(s); append_jsonl(self.items,item); append_jsonl(self.journal,{'event':'MATERIALIZED','item_id':item_id,'candidate_id':candidate['candidate_id'],'decision_id':decision['decision_id'],'rollback_id':'rollback:'+item_id.split(':',1)[1]})
        return item
    def rollback(self, rollback_id: str) -> dict[str,Any]:
        suffix=rollback_id.split(':',1)[1]; item_id='graphitem:'+suffix; s=self._state()
        if item_id in s['items']:
            removed=s['items'].pop(item_id); self._save(s); append_jsonl(self.journal,{'event':'ROLLBACK','item_id':item_id,'rollback_id':rollback_id}); return {'status':'ROLLED_BACK','removed':removed}
        return {'status':'NOOP'}
