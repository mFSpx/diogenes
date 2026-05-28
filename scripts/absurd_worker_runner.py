#!/usr/bin/env python3
"""Drain local file-backed ABSURD pipeline jobs with legal state transitions."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from spine_job_adapter import ABSURDJobAdapter
from case_pipeline_dispatch import dispatch

def drain(adapter: ABSURDJobAdapter, *, base_dir: str|Path|None=None, max_jobs: int=100) -> dict:
    processed=[]
    for _ in range(max_jobs):
        ready=adapter.ready_jobs()
        if not ready: break
        job=sorted(ready, key=lambda j:j['created_at'])[0]
        if job['state']=='CREATED':
            job=adapter.transition(job['job_id'],'QUEUED')
        adapter.transition(job['job_id'],'RUNNING')
        try:
            result=dispatch(job['lane'], job['payload'], base_dir=base_dir)
            adapter.transition(job['job_id'],'COMPLETED',result={'receipt_path':result.get('receipt_path'), 'status':result.get('status')})
            processed.append({'job_id':job['job_id'],'lane':job['lane'],'status':'COMPLETED'})
        except Exception as e:
            failed=adapter.transition(job['job_id'],'FAILED',error=repr(e))
            if failed['attempt_count'] >= failed.get('max_attempts',3):
                adapter.transition(job['job_id'],'DEAD_LETTER',error=repr(e))
            processed.append({'job_id':job['job_id'],'lane':job['lane'],'status':'FAILED','error':repr(e)})
    return {'status':'PASSED','processed':processed,'processed_count':len(processed)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('adapter_root'); ap.add_argument('--base-dir'); a=ap.parse_args(); res=drain(ABSURDJobAdapter(a.adapter_root), base_dir=a.base_dir); print(json.dumps(res,sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
