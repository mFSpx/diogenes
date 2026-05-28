#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from spine_common import now, sha256_json, append_jsonl, write_json
LEGAL = {None:{'CREATED'}, 'CREATED':{'QUEUED','CANCELLED'}, 'QUEUED':{'RUNNING','CANCELLED'}, 'RUNNING':{'COMPLETED','FAILED'}, 'FAILED':{'QUEUED','DEAD_LETTER'}, 'COMPLETED':set(), 'DEAD_LETTER':set(), 'CANCELLED':set()}

class ABSURDJobAdapter:
    def __init__(self, root: str|Path):
        self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True); self.jobs_path=self.root/'jobs.jsonl'; self.state_path=self.root/'jobs_state.json'
    def _state(self):
        return json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
    def _save(self,s): write_json(self.state_path,s)
    def create_job(self, *, lane: str, payload: dict[str,Any], idempotency_key: str, depends_on: list[str]|None=None) -> dict[str,Any]:
        state=self._state()
        for j in state.values():
            if j['idempotency_key']==idempotency_key: return j
        job={'job_id':'job:'+sha256_json({'lane':lane,'idempotency_key':idempotency_key})[:24], 'lane':lane, 'payload':payload, 'idempotency_key':idempotency_key, 'depends_on':depends_on or [], 'state':'CREATED', 'attempt_count':0, 'max_attempts':3, 'created_at':now(), 'updated_at':now(), 'result':{}, 'error':None}
        state[job['job_id']]=job; self._save(state); append_jsonl(self.jobs_path, {'event':'CREATED','job':job,'at':now()}); return job
    def transition(self, job_id: str, new_state: str, *, result: dict[str,Any]|None=None, error: str|None=None) -> dict[str,Any]:
        state=self._state(); job=state[job_id]; cur=job['state']
        if new_state not in LEGAL.get(cur,set()): raise ValueError(f'illegal transition {cur}->{new_state}')
        job['state']=new_state; job['updated_at']=now(); job['result']=result or job.get('result',{}); job['error']=error
        if new_state=='RUNNING': job['attempt_count']+=1
        state[job_id]=job; self._save(state); append_jsonl(self.jobs_path, {'event':new_state,'job_id':job_id,'result':result,'error':error,'at':now()}); return job
    def ready_jobs(self) -> list[dict[str,Any]]:
        s=self._state(); completed={jid for jid,j in s.items() if j['state']=='COMPLETED'}
        return [j for j in s.values() if j['state'] in {'CREATED','QUEUED'} and all(d in completed for d in j.get('depends_on',[]))]
