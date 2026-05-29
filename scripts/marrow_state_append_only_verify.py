#!/usr/bin/env python3
"""Verify Marrow state is append-only using a local hash-chain receipt."""
from __future__ import annotations
import argparse,hashlib,json
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS/marrow_loop'; CHAIN=OUT/'marrow_state_hash_chain.jsonl'
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def sha(b:bytes): return hashlib.sha256(b).hexdigest()
def rel(p):
 try: return str(Path(p).resolve().relative_to(ROOT))
 except Exception: return str(p)
def write(payload):
 OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'marrow_state_append_only_verify_{stamp()}.json'; payload.setdefault('generated_at',now()); payload['report_path']=rel(p); p.write_text(json.dumps(payload,indent=2),encoding='utf-8'); print(f'REPORT_PATH={rel(p)}')
def load_last():
 if not CHAIN.exists(): return None
 lines=[l for l in CHAIN.read_text(encoding='utf-8').splitlines() if l.strip()]
 return json.loads(lines[-1]) if lines else None
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--state',default='05_OUTPUTS/marrow_loop/marrow_state.md'); ap.add_argument('--record',action='store_true'); a=ap.parse_args()
 path=ROOT/a.state; blockers=[]; entries=0; cur_text=''
 if not path.exists(): blockers.append('state_missing')
 else:
  cur_text=path.read_text(encoding='utf-8'); entries=cur_text.count('command_uuid') + cur_text.count('## Command') + cur_text.count('## Marrow Command')
  if entries < 1: blockers.append('no_command_entries_detected')
 cur_hash=sha(cur_text.encode()) if path.exists() else None; size=len(cur_text.encode()) if path.exists() else 0; last=load_last(); append_ok=True
 if last and path.exists():
  prev_size=int(last.get('state_size_bytes',0)); prev_hash=last.get('state_sha256')
  if size < prev_size: append_ok=False; blockers.append('state_shrank_since_last_receipt')
  elif size==prev_size and cur_hash!=prev_hash: append_ok=False; blockers.append('same_size_content_changed')
 if a.record and path.exists() and append_ok:
  rec={'recorded_at':now(),'state_path':rel(path),'state_sha256':cur_hash,'state_size_bytes':size,'entry_count':entries,'previous_state_sha256':last.get('state_sha256') if last else None}
  OUT.mkdir(parents=True,exist_ok=True); CHAIN.open('a',encoding='utf-8').write(json.dumps(rec)+'\n')
 payload={'action':'append_only_verify','record_performed':bool(a.record),'state_path':rel(path),'state_sha256':cur_hash,'state_size_bytes':size,'entry_count':entries,'previous_receipt':last,'append_only_ok':append_ok,'blockers':blockers,'status':'PASS' if not blockers else 'FAIL'}
 write(payload); print('MARROW_APPEND_ONLY='+payload['status']); return 0 if not blockers else 4
if __name__=='__main__': raise SystemExit(main())
