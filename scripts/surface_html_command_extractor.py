#!/usr/bin/env python3
"""Extract command-envelope JSON candidates from generated surface HTML."""
from __future__ import annotations
import argparse,json,re,html
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS/surfaces'
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
 try: return str(Path(p).resolve().relative_to(ROOT))
 except Exception: return str(p)
def write(payload):
 OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'surface_html_command_extractor_{payload["status"].lower()}_{stamp()}.json'; payload.setdefault('generated_at',now()); payload['report_path']=rel(p); p.write_text(json.dumps(payload,indent=2,default=str),encoding='utf-8'); print(f'REPORT_PATH={rel(p)}'); return p
def candidates(text):
 out=[]
 # data-command-envelope='json' or double quoted
 for m in re.finditer(r'data-command-envelope\s*=\s*([\"\'])(.*?)\1', text, re.S|re.I): out.append(html.unescape(m.group(2)))
 # script/json or code blocks containing protocol
 for m in re.finditer(r'\{[^{}]*(?:protocol|command_envelope|plain_language_instruction)[^{}]*\}', text, re.S|re.I): out.append(html.unescape(m.group(0)))
 return out
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--html',required=True); a=ap.parse_args(); p=Path(a.html); p=p if p.is_absolute() else ROOT/p; blockers=[]; envelopes=[]
 if not p.exists(): blockers.append('html_missing')
 else:
  text=p.read_text(encoding='utf-8',errors='replace')
  for raw in candidates(text):
   try:
    data=json.loads(raw)
    if isinstance(data,dict): envelopes.append(data)
   except Exception: pass
  if not envelopes and 'command envelope' not in text.lower() and 'command-envelope' not in text.lower(): blockers.append('no_command_envelope_language_or_json')
  for e in envelopes:
   if e.get('canonical_mutation_allowed') is True or e.get('direct_canonical_mutation') is True: blockers.append('envelope_allows_direct_mutation')
 report={'action':'extract','html_path':rel(p),'envelopes_found':len(envelopes),'envelopes':envelopes[:20],'blockers':blockers,'db_writes_performed':False,'graph_writes_performed':False,'status':'PASS' if not blockers else 'FAIL'}
 write(report); print('SURFACE_HTML_COMMAND_EXTRACTOR='+report['status']); return 0 if not blockers else 4
if __name__=='__main__': raise SystemExit(main())
