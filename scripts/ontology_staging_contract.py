#!/usr/bin/env python3
"""Create ontology staging candidates; never promotes truth directly."""
from __future__ import annotations
import argparse, hashlib, json, re
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS/ontology'; SCHEMA=ROOT/'06_SCHEMA/ontology_staging.schema.json'
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
    try: return str(Path(p).resolve().relative_to(ROOT))
    except Exception: return str(p)
def sha(path:Path): return hashlib.sha256(path.read_bytes()).hexdigest()
def load_terms():
    terms=set()
    for p in [ROOT/'OFFICIAL_ONTOLOGY.json',ROOT/'BOOKS/GO_ACTIVE_TERMS.json',ROOT/'BOOKS/GO_EXTENSIONS.json']:
        if not p.exists(): continue
        try:
            data=json.loads(p.read_text())
            blob=json.dumps(data)
            terms.update(re.findall(r'\b[A-Z][A-Z0-9_]{2,}\b', blob))
        except Exception: pass
    return sorted(terms)
def stage(path:Path, method:str):
    text=path.read_text(encoding='utf-8',errors='ignore')[:50000]
    terms=load_terms(); found=[t for t in terms if re.search(r'\b'+re.escape(t)+r'\b', text)]
    sentences=[s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip())>20][:50]
    claim=[s for s in sentences if re.search(r'\b(is|are|has|must|should|requires|means)\b',s,re.I)][:20]
    events=[s for s in sentences if re.search(r'\b(20\d{2}|happens|event|when|after|before)\b',s,re.I)][:20]
    ents=sorted(set(re.findall(r'\b[A-Z][A-Za-z0-9_-]{2,}\b', text)))[:100]
    return {'schema':'diogenes.ontology_staging_contract.v1','source_file':rel(path),'source_hash':sha(path),'extraction_method':method,'entity_candidates':ents,'claim_candidates':claim,'event_candidates':events,'ontology_terms_found':found,'confidence':3500 if claim or ents else 1000,'status':'STAGED','promotion_blocker':'operator_review_and_graph_promotion_required','direct_truth_promotion_performed':False}
def validate_packet(packet:dict)->list[str]:
    errors=[]
    required=['source_file','source_hash','extraction_method','entity_candidates','claim_candidates','event_candidates','confidence','status','promotion_blocker']
    for key in required:
        if key not in packet: errors.append(f'missing:{key}')
    if not isinstance(packet.get('confidence'),int) or not (0 <= int(packet.get('confidence',-1)) <= 10000): errors.append('confidence_not_basis_points_integer_0_10000')
    if packet.get('status') not in {'STAGED','REJECTED'}: errors.append('status_not_schema_enum')
    for key in ['entity_candidates','claim_candidates','event_candidates']:
        if not isinstance(packet.get(key),list): errors.append(f'{key}_not_list')
    return errors
def main():
    p=argparse.ArgumentParser(); p.add_argument('--source-file',required=True); p.add_argument('--extraction-method',default='regex_static_contract_v1'); p.add_argument('--execute',action='store_true'); p.add_argument('--dry-run',action='store_true'); p.add_argument('--json',action='store_true')
    a=p.parse_args(); src=Path(a.source_file); src=src if src.is_absolute() else ROOT/src
    if not src.exists(): print('ONTOLOGY_STAGE=BLOCKED'); print('ERROR=SOURCE_FILE_MISSING'); return 3
    r=stage(src,a.extraction_method); r['generated_at']=now(); r['execute_performed']=False
    errors=validate_packet(r); r['schema_path']=rel(SCHEMA); r['schema_valid']=not errors; r['schema_errors']=errors
    OUT.mkdir(parents=True,exist_ok=True); out=OUT/f'ontology_staging_contract_{stamp()}.json'; r['receipt_path']=rel(out); out.write_text(json.dumps(r,indent=2),encoding='utf-8'); print('RECEIPT_PATH='+rel(out));
    if a.json: print(json.dumps(r,sort_keys=True))
    print('ONTOLOGY_STAGE='+('SCHEMA_VALID' if r['schema_valid'] else 'SCHEMA_INVALID')); return 0 if r['schema_valid'] else 4
if __name__=='__main__': raise SystemExit(main())
