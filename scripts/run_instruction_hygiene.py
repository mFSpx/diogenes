#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, re, shutil, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "instruction_hygiene"
REGISTRY = ROOT / "00_PROJECT_BRAIN" / "instruction_authority_registry.json"
ACTIVE_INDEX = ROOT / "00_PROJECT_BRAIN" / "ACTIVE_INSTRUCTION_INDEX.md"
STORAGE = ROOT / "09_STORAGE"
ARCHIVE = STORAGE / "instruction_archive"
QUARANTINE = STORAGE / "instruction_quarantine"
LEGACY_PROMPTS = STORAGE / "legacy_prompts"
GRAPH_CANDIDATES_DIR = STORAGE / "graph_staging_candidates"
TREE_MAP = ROOT / "05_OUTPUTS" / "repo_tree_3level_20260517T224732Z.txt"

INCLUDE_SUFFIXES = {".md", ".txt", ".json", ".jsonl", ".yaml", ".yml", ".toml", ".py", ".ps1", ".cmd", ".bat", ".sql", ".sh"}
ROOT_NAME_INCLUDE = {"README", "AGENTS", "CLAUDE", "SECURITY", "CONTRIBUTING", "Makefile", "AIDUMB", "wordsofpower"}
MARKERS = ["CURRENT MODE", "HARD LAW", "MUST", "NEVER", "DO NOT", "canonical", "queue", "worker", "auditor", "oracle", "graph", "materialize", "proof", "custody", "accepted truth", "STATUS_LEDGER", "work order", "prompt", "Claude", "Codex", "ChatGPT", "Deep Research", "one-shot", "golden path", "spine", "membrane", "translator", "operator instruction", "system prompt", "role prompt", "daemon", "launcher"]
CERTAIN_LEGACY_PROMPTS = [Path("00_PROJECT_BRAIN/AGENTSI_SELF_SOVEREIGN_JOB_FAIR/FANTASY_INTERVIEW_100.md")]
CANONICAL_FILES = [
    "00_PROJECT_BRAIN/STATUS_LEDGER.md",
    "00_PROJECT_BRAIN/CANONICAL_FINISHED_PRODUCT_MAP.md",
    "00_PROJECT_BRAIN/spine_authority_registry.json",
    "00_PROJECT_BRAIN/canonical_graph_write_allowlist.json",
    "00_PROJECT_BRAIN/instruction_authority_registry.json",
    "00_PROJECT_BRAIN/ACTIVE_INSTRUCTION_INDEX.md",
    "00_PROJECT_BRAIN/BLUEPRINT_FIRST_MODEL_SECOND_PSEUDOLAW.md",
    "GOALS/AGENT_ORCHESTRATION_POLICY.md",
]


def now() -> str: return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
def stamp() -> str: return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
def rel(path: Path | str) -> str:
    try: return str(Path(path).resolve().relative_to(ROOT))
    except Exception: return str(path)
def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()
def text_hash(text: str) -> str: return hashlib.sha256(text.encode('utf-8', errors='ignore')).hexdigest()

def hardening_pass() -> tuple[bool, str | None, list[str]]:
    receipts=sorted((ROOT/'05_OUTPUTS/golden_path_hardening').glob('golden_path_hardening_*.json'), key=lambda p:p.stat().st_mtime)
    if not receipts: return False, None, ["missing_hardening_receipt"]
    latest=receipts[-1]
    try: data=json.loads(latest.read_text(encoding='utf-8'))
    except Exception as exc: return False, rel(latest), [f"hardening_receipt_unreadable:{exc}"]
    return data.get('verdict') == 'PASS', rel(latest), [] if data.get('verdict') == 'PASS' else [f"hardening_not_pass:{data.get('verdict')}"]

def ensure_tree_map(ts: str) -> str:
    if TREE_MAP.exists(): return rel(TREE_MAP)
    OUT.mkdir(parents=True, exist_ok=True)
    out=OUT/f"repo_tree_3level_{ts}.txt"
    cmd="find . -maxdepth 3 -path './.git/objects' -prune -o -path './03_VAULT/cas' -prune -o -path './01_REPOS/llama.cpp/build-cuda' -prune -o -path './node_modules' -prune -o -print"
    proc=subprocess.run(cmd, cwd=ROOT, shell=True, text=True, capture_output=True)
    out.write_text(proc.stdout, encoding='utf-8')
    return rel(out)

def file_type(path: Path) -> str:
    if path.name in ROOT_NAME_INCLUDE: return path.name
    if path.suffix: return path.suffix.lower().lstrip('.')
    if os.access(path, os.X_OK): return 'executable_text'
    return 'unknown_text'

def should_skip_path(path: Path) -> bool:
    parts=set(path.parts)
    s=str(path)
    if any(x in parts for x in {'.git','__pycache__','node_modules','.venv','venv','dist','build'}): return True
    if path.suffix == '.pyc': return True
    if '01_REPOS/llama.cpp/build-cuda' in s or '03_VAULT/cas' in s: return True
    return False

def include_file(path: Path) -> bool:
    if should_skip_path(path) or not path.is_file(): return False
    rp=rel(path)
    if path.stat().st_size > 5_000_000: return False
    if rp.startswith('01_REPOS/'):
        parts=Path(rp).parts
        if len(parts) >= 3:
            name=path.name
            if name in {'README.md','CLAUDE.md','AGENTS.md','CONTRIBUTING.md','SECURITY.md','pyproject.toml','Cargo.toml','Makefile'}: return True
            if '.github' in parts and 'workflows' in parts and path.suffix in {'.yml','.yaml'}: return True
        return False
    if rp.startswith('03_VAULT/') or rp.startswith('KRAMPUSCHEWING/'):
        return path.suffix.lower() in {'.md','.txt','.json','.jsonl'} and path.stat().st_size <= 1_000_000
    if rp.startswith('04_RUNTIME/'):
        return path.name == 'README.md' or (path.suffix.lower() == '.json' and path.stat().st_size <= 1_000_000)
    if rp.startswith('05_OUTPUTS/'):
        if path.name == 'README.md' or rp == '05_OUTPUTS/status_ledger.json': return True
        if 'golden_path_hardening/' in rp and path.name.startswith('golden_path_hardening_'): return True
        if 'golden_path/' in rp and path.name.startswith('golden_path_single_instruction_'): return True
        if 'instruction_hygiene/' in rp and path.suffix == '.json': return True
        if path.suffix == '.md' and path.stat().st_size <= 500_000: return True
        return False
    if rp.startswith('.claw/') or rp.startswith('.pytest_cache/'): return False
    if path.suffix.lower() in INCLUDE_SUFFIXES: return True
    if path.name in ROOT_NAME_INCLUDE: return True
    return False

def collect_files() -> list[Path]:
    files: set[Path] = set()
    def add(p: Path) -> None:
        if include_file(p):
            files.add(p)
    def walk_dir(d: Path) -> None:
        if not d.exists():
            return
        for base, dirs, names in os.walk(d):
            dirs[:] = [x for x in dirs if x not in {'.git','__pycache__','node_modules','.venv','venv','dist','build','build-cuda','cas'}]
            for name in names:
                add(Path(base) / name)
    # Root misc instruction surfaces.
    for p in ROOT.iterdir():
        if p.is_file():
            add(p)
    # Zone A: active authority/law candidates.
    for d in ['00_PROJECT_BRAIN','06_SCHEMA','scripts','tests','automation']:
        walk_dir(ROOT / d)
    walk_dir(ROOT / '.github' / 'workflows')
    # Zone B: nested repos: top-level instruction files plus workflow files only.
    repos = ROOT / '01_REPOS'
    if repos.exists():
        for repo in repos.iterdir():
            if not repo.is_dir():
                continue
            for name in ['README.md','CLAUDE.md','AGENTS.md','CONTRIBUTING.md','SECURITY.md','pyproject.toml','Cargo.toml','Makefile']:
                add(repo / name)
            wf = repo / '.github' / 'workflows'
            if wf.exists():
                for p in wf.glob('*'):
                    add(p)
    # Zone C: evidence/vault text-only instruction-like files only. Do not walk
    # raw evidence trees recursively; they are storage/proof, not active law.
    for d in ['KRAMPUSCHEWING','03_VAULT']:
        base = ROOT / d
        if not base.exists():
            continue
        for p in base.iterdir():
            if p.is_file():
                add(p)
        for pat in ['LUCIDOTA*.md', '*CONTRACT*.md', '*POLICY*.md', '*README*.md', '*STATUS*.md']:
            for p in base.glob(pat):
                add(p)
    # Zone D: runtime configs only.
    walk_dir(ROOT / '04_RUNTIME')
    # Zone E: selected output receipts/indexes, not every receipt exhaust file.
    for p in [ROOT/'05_OUTPUTS/README.md', ROOT/'05_OUTPUTS/status_ledger.json']:
        add(p)
    for pat in [
        '05_OUTPUTS/*.md',
        '05_OUTPUTS/golden_path/golden_path_single_instruction_*.json',
        '05_OUTPUTS/golden_path_hardening/golden_path_hardening_*.json',
        '05_OUTPUTS/instruction_hygiene/*.json',
    ]:
        for p in ROOT.glob(pat):
            add(p)
    return sorted(files)

def flags(text: str) -> dict[str, bool]:
    low=text.lower()
    return {
        'contains_runtime_law': any(x in low for x in ['hard law','must','never','do not','required']),
        'contains_operator_prompt': any(x in low for x in ['operator instruction','current mode','prompt','system prompt','role prompt','codex','chatgpt','claude']),
        'contains_architecture_claim': any(x in low for x in ['architecture','canonical','finished product','design','contract']),
        'contains_graph_instruction': any(x in low for x in ['graph','materialize','lucidota_go.graph_item','graph promotion']),
        'contains_queue_instruction': any(x in low for x in ['queue','absurd','work order','worker-once']),
        'contains_worker_instruction': 'worker' in low or 'daemon' in low or 'launcher' in low,
        'contains_auditor_instruction': 'audit' in low or 'auditor' in low or 'oracle' in low,
        'contains_status_claim': 'status_ledger' in low or 'verified' in low or 'complete' in low,
        'contains_legacy_danger': any(x in low for x in ['materialize without','bypass','legacy','receipt collage','self-report sufficient','accepted truth without']),
    }

def likelihood(text: str, fl: dict[str,bool], path: str) -> str:
    count=sum(text.lower().count(m.lower()) for m in MARKERS)
    if path in CANONICAL_FILES or 'HARD LAW' in text or 'CURRENT MODE' in text: return 'critical'
    if fl['contains_legacy_danger'] or count >= 12: return 'high'
    if count >= 5 or sum(fl.values()) >= 4: return 'medium'
    if count >= 1 or sum(fl.values()) >= 1: return 'low'
    return 'none'

def inventory(files: list[Path]) -> list[dict[str,Any]]:
    rows=[]
    for p in files:
        try: text=p.read_text(encoding='utf-8', errors='ignore')
        except Exception: text=''
        st=p.stat(); rp=rel(p); fl=flags(text)
        rows.append({'path':rp,'sha256':sha256_file(p),'size_bytes':st.st_size,'modified_time':datetime.fromtimestamp(st.st_mtime,timezone.utc).isoformat().replace('+00:00','Z'),'file_type':file_type(p),'instruction_likelihood':likelihood(text,fl,rp),**fl})
    return rows

def classify_packet(path: str, text: str, inv: dict[str,Any]) -> tuple[str,str,str,str]:
    low=text.lower(); p=path.lower()
    if inv.get('contains_legacy_danger'): itype='dangerous_instruction'
    elif 'hard law' in low or 'must' in low or 'never' in low or 'do not' in low: itype='law'
    elif 'prompt' in low or 'current mode' in low or 'operator instruction' in low: itype='prompt'
    elif 'architecture' in low or 'contract' in low: itype='architecture'
    elif 'status_ledger' in low or 'verified' in low: itype='status_claim'
    elif 'worker' in low or 'daemon' in low: itype='workflow'
    else: itype='unknown'
    if path in CANONICAL_FILES: scope='project'; auth='canonical'; action='keep_active'
    elif p.startswith('01_repos/'): scope='subsystem'; auth='commentary'; action='stage_for_graph_later'
    elif p.startswith('05_outputs/'): scope='historical'; auth='commentary'; action='convert_to_reference'
    elif 'fantasy' in p or 'one-shot' in low or 'old prompt' in low: scope='historical'; auth='legacy'; action='archive'
    elif inv.get('contains_legacy_danger'): scope='unknown'; auth='dangerous'; action='quarantine'
    elif inv.get('instruction_likelihood') in {'critical','high'} and p.startswith(('00_project_brain/','06_schema/','scripts/','tests/','.github/workflows/')): scope='project'; auth='active_candidate'; action='manual_review'
    else: scope='unknown'; auth='commentary'; action='manual_review'
    return itype, scope, auth, action

def extract_packets(inv_rows: list[dict[str,Any]]) -> list[dict[str,Any]]:
    packets=[]
    by_path={r['path']:r for r in inv_rows}
    for row in inv_rows:
        if row['instruction_likelihood'] in {'none','low'}: continue
        p=ROOT/row['path']
        try: lines=p.read_text(encoding='utf-8', errors='ignore').splitlines()
        except Exception: lines=[]
        hits=[i for i,l in enumerate(lines,1) if any(m.lower() in l.lower() for m in MARKERS)]
        if not hits: hits=[1]
        start=max(1,min(hits)-2); end=min(len(lines), start+24) if lines else 0
        excerpt='\n'.join(lines[start-1:end])[:2000]
        itype,scope,auth,action=classify_packet(row['path'], excerpt or '\n'.join(lines[:30]), row)
        pid='ip_'+hashlib.sha256(f"{row['path']}:{start}:{end}:{row['sha256']}".encode()).hexdigest()[:24]
        packets.append({'instruction_packet_id':pid,'source_path':row['path'],'line_start':start,'line_end':end,'sha256':text_hash(excerpt),'text_excerpt':excerpt,'instruction_type':itype,'scope':scope,'authority_candidate':auth,'jurisdiction':'nested_repo_local' if row['path'].startswith('01_REPOS/') else 'lucidota','conflicts_with':[],'supports':[],'recommended_action':action})
    return packets

def write_registry(manifest_ref: str, manual_review: list[str], moved: list[dict[str,Any]], quarantined: list[dict[str,Any]]) -> dict[str,Any]:
    canonical=[p for p in CANONICAL_FILES if (ROOT/p).exists() or p.endswith('instruction_authority_registry.json') or p.endswith('ACTIVE_INSTRUCTION_INDEX.md')]
    data={'schema':'lucidota.instruction_authority_registry.v1','updated_at':now(),'authority_order':['operator_current_instruction','00_PROJECT_BRAIN/STATUS_LEDGER.md','00_PROJECT_BRAIN/CANONICAL_FINISHED_PRODUCT_MAP.md','00_PROJECT_BRAIN/spine_authority_registry.json','00_PROJECT_BRAIN/canonical_graph_write_allowlist.json','06_SCHEMA','scripts_with_tests','archived_reference','legacy_storage'],'canonical_files':canonical,'active_laws':[{'law_key':'golden_path_v1_no_materialization','domain':'golden_path','source':'05_OUTPUTS/golden_path/golden_path_single_instruction_20260517T233140Z.json'},{'law_key':'canonical_graph_sacred_tables','domain':'canonical_graph_writes','source':'00_PROJECT_BRAIN/canonical_graph_write_allowlist.json'},{'law_key':'spine_authority_singletons','domain':'spine_authority','source':'00_PROJECT_BRAIN/spine_authority_registry.json'},{'law_key':'blueprint_first_model_second_pocketflow_hygiene','domain':'workflow_hygiene','source':'00_PROJECT_BRAIN/BLUEPRINT_FIRST_MODEL_SECOND_PSEUDOLAW.md'}],'superseded_files':[],'quarantined_files':[m['to'] for m in quarantined],'legacy_reference_files':[m['to'] for m in moved],'manual_review_files':sorted(set(manual_review))[:300],'archive_manifest':manifest_ref,'hard_rule':'Random old markdown/prompt files cannot override STATUS_LEDGER, canonical finished product map, spine registry, graph write allowlist, golden path regression receipts, or current operator instruction.'}
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(json.dumps(data,indent=2,sort_keys=False),encoding='utf-8')
    return data

def write_active_index(registry: dict[str,Any], manifest_ref: str) -> None:
    lines=['# Active Canonical Instruction Sources','','## STATUS_LEDGER','- `00_PROJECT_BRAIN/STATUS_LEDGER.md`','','## CANONICAL_FINISHED_PRODUCT_MAP','- `00_PROJECT_BRAIN/CANONICAL_FINISHED_PRODUCT_MAP.md`','','## spine authority registry','- `00_PROJECT_BRAIN/spine_authority_registry.json`','','## graph write allowlist','- `00_PROJECT_BRAIN/canonical_graph_write_allowlist.json`','','## proof/kernel laws','- Golden Path V1 receipt: `05_OUTPUTS/golden_path/golden_path_single_instruction_20260517T233140Z.json`','- Golden Path hardening receipt: latest under `05_OUTPUTS/golden_path_hardening/`','','## workflow hygiene pseudolaw','- `00_PROJECT_BRAIN/BLUEPRINT_FIRST_MODEL_SECOND_PSEUDOLAW.md`','','## GOALS dev-mode law','- `GOALS/AGENT_ORCHESTRATION_POLICY.md`','','## queue runtime laws','- `06_SCHEMA/035_absurd_queue_spine.sql`','- `scripts/absurd_queue_spine.py`','- `scripts/conversation_command_accept_worker.py`','','## oracle/auditor laws','- `scripts/canonical_graph_write_scanner.py`','- `scripts/spine_authority_checker.py`','- `scripts/instruction_conflict_scanner.py`','','## current regression gates','- `scripts/run_golden_path_hardening_checks.py`','- `scripts/run_instruction_hygiene.py`','','## Archived / Historical Instruction Sources',f'- Storage/archive manifest: `{manifest_ref}`','','## Manual Review Required']
    for p in registry.get('manual_review_files',[])[:80]: lines.append(f'- `{p}`')
    ACTIVE_INDEX.write_text('\n'.join(lines).rstrip()+'\n',encoding='utf-8')

def graph_candidates(packets: list[dict[str,Any]], ts: str) -> tuple[str,int]:
    GRAPH_CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    out=GRAPH_CANDIDATES_DIR/f'instruction_graph_candidates_{ts}.jsonl'
    count=0
    with out.open('w',encoding='utf-8') as f:
        for p in packets:
            if p['authority_candidate']=='canonical': kind='authority_edge'
            elif p['authority_candidate'] in {'dangerous'}: kind='legacy_danger'
            elif p['authority_candidate'] in {'legacy','superseded'}: kind='supersession'
            elif p['conflicts_with']: kind='conflict'
            else: kind='instruction'
            rec={'candidate_kind':kind,'source_instruction_packet_id':p['instruction_packet_id'],'source_path':p['source_path'],'claim':p['text_excerpt'][:500].replace('\n',' '),'evidence_ref':p['source_path'],'recommended_graph_action':'stage_only','materialization_allowed_now':False}
            f.write(json.dumps(rec,sort_keys=True,ensure_ascii=False)+'\n'); count+=1
    return rel(out), count

def maybe_move(mode: str, ts: str) -> tuple[list[dict[str,Any]], list[dict[str,Any]]]:
    moved=[]; quarantined=[]
    for d in [ARCHIVE,QUARANTINE,LEGACY_PROMPTS,GRAPH_CANDIDATES_DIR]: d.mkdir(parents=True, exist_ok=True)
    for src_rel in CERTAIN_LEGACY_PROMPTS:
        dest = LEGACY_PROMPTS / src_rel
        if dest.exists() and not (ROOT / src_rel).exists():
            sha = sha256_file(dest)
            moved.append({'from':str(src_rel),'to':rel(dest),'sha256_before':sha,'sha256_after':sha,'reason':'already archived legacy fantasy/prompt artifact; preserved outside active authority field','restore_command':f"mkdir -p {str(src_rel.parent)!r} && mv {rel(dest)!r} {str(src_rel)!r}", 'already_archived': True})
    if mode != 'execute': return moved, quarantined
    for src_rel in CERTAIN_LEGACY_PROMPTS:
        src=ROOT/src_rel
        if not src.exists(): continue
        dest=LEGACY_PROMPTS/src_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        sha_before=sha256_file(src)
        shutil.move(str(src), str(dest))
        sha_after=sha256_file(dest)
        moved.append({'from':str(src_rel),'to':rel(dest),'sha256_before':sha_before,'sha256_after':sha_after,'reason':'certain legacy fantasy/prompt artifact; preserved outside active authority field','restore_command':f"mkdir -p {str(src_rel.parent)!r} && mv {rel(dest)!r} {str(src_rel)!r}"})
    return moved, quarantined

def run_cmd(cmd: list[str]) -> dict[str,Any]:
    p=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True)
    refs=[line.split('=',1)[1] for line in (p.stdout+'\n'+p.stderr).splitlines() if line.startswith('REPORT_PATH=')]
    return {'command':' '.join(cmd),'rc':p.returncode,'result':'PASS' if p.returncode==0 else 'FAIL','report_paths':refs,'stdout_tail':p.stdout[-2000:],'stderr_tail':p.stderr[-2000:]}

def main() -> int:
    ap=argparse.ArgumentParser(); mode=ap.add_mutually_exclusive_group(required=True); mode.add_argument('--dry-run',action='store_true'); mode.add_argument('--execute',action='store_true'); a=ap.parse_args(); mode_name='execute' if a.execute else 'dry_run'; ts=stamp(); OUT.mkdir(parents=True, exist_ok=True)
    ok, hardening_ref, blockers = hardening_pass()
    if not ok:
        payload={'schema':'lucidota.instruction_hygiene_receipt.v1','verdict':'BLOCKED','mode':mode_name,'repo_root':str(ROOT),'files_scanned':0,'instruction_packets_found':0,'files_archived':[],'files_quarantined':[],'manual_review':[],'active_canonical_files':[],'conflicts_found':[],'conflicts_resolved':[],'graph_candidates_written':[],'canonical_graph_materialization':False,'canonical_graph_writes':False,'tests_run':[],'receipts_written':[hardening_ref] if hardening_ref else [],'restore_manifest':'','blockers':blockers}
        out=OUT/f'instruction_hygiene_{ts}.json'; payload['report_path']=rel(out); out.write_text(json.dumps(payload,indent=2),encoding='utf-8'); print('REPORT_PATH='+rel(out)); print('INSTRUCTION_HYGIENE=BLOCKED'); return 6
    tree_ref=ensure_tree_map(ts)
    files=collect_files(); inv=inventory(files)
    inv_path=OUT/f'instruction_inventory_{ts}.json'; inv_path.write_text(json.dumps({'schema':'lucidota.instruction_inventory.v1','generated_at':now(),'repo_root':str(ROOT),'tree_map':tree_ref,'files':inv},indent=2),encoding='utf-8')
    packets=extract_packets(inv)
    packets_path=OUT/f'instruction_packets_{ts}.jsonl'
    with packets_path.open('w',encoding='utf-8') as f:
        for p in packets: f.write(json.dumps(p,ensure_ascii=False,sort_keys=True)+'\n')
    moved, quarantined=maybe_move(mode_name, ts)
    manifest={'schema':'lucidota.instruction_archive_manifest.v1','generated_at':now(),'mode':mode_name,'files_moved':moved,'files_quarantined':quarantined,'proposed_moves':[] if mode_name=='execute' else [{'from':str(p),'to':rel(LEGACY_PROMPTS/p),'reason':'certain legacy fantasy/prompt artifact; execute mode will preserve and move if still present','restore_command':f"mkdir -p {str(p.parent)!r} && mv {rel(LEGACY_PROMPTS/p)!r} {str(p)!r}"} for p in CERTAIN_LEGACY_PROMPTS if (ROOT/p).exists()],'restore_policy':'Every moved/quarantined/proposed storage record must include from, to, sha or reason, and restore_command; no deletes are allowed.','restore_manifest_has_restore_commands':True,'canonical_graph_materialization':False,'canonical_graph_writes':False}
    manifest_path=OUT/f'instruction_archive_manifest_{ts}.json'; manifest['report_path']=rel(manifest_path); manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    manual=sorted({p['source_path'] for p in packets if p['recommended_action']=='manual_review'})
    registry=write_registry(rel(manifest_path), manual, moved, quarantined)
    write_active_index(registry, rel(manifest_path))
    graph_path, graph_count=graph_candidates(packets, ts)
    tests=[]
    conflict=run_cmd([sys.executable,'scripts/instruction_conflict_scanner.py']); tests.append(conflict)
    tests.append(run_cmd([sys.executable,'scripts/language_router.py','--text','instruction hygiene recovery proof','--verbosity','terse','--json']))
    pytest_cmd=[sys.executable,'-m','pytest','tests/test_instruction_conflict_scanner.py','tests/test_instruction_authority_registry.py','-q']
    if os.environ.get('INSTRUCTION_HYGIENE_SKIP_TESTS') != '1' and (ROOT/'tests/test_instruction_conflict_scanner.py').exists() and (ROOT/'tests/test_instruction_authority_registry.py').exists():
        tests.append(run_cmd(pytest_cmd))
    conflicts=[]
    for rp in conflict.get('report_paths',[]):
        try: conflicts=json.loads((ROOT/rp).read_text()).get('blockers',[])
        except Exception: pass
    receipts=[rel(inv_path),rel(packets_path),rel(manifest_path),graph_path,rel(REGISTRY),rel(ACTIVE_INDEX)] + [r for t in tests for r in t.get('report_paths',[])]
    blockers_final=[*blockers]
    if conflict['rc']!=0: blockers_final.append('instruction_conflict_scan_failed')
    for t in tests:
        if t['rc']!=0: blockers_final.append(t['command'])
    payload={'schema':'lucidota.instruction_hygiene_receipt.v1','verdict':'PASS' if not blockers_final else 'PARTIAL_FAIL','mode':mode_name,'repo_root':str(ROOT),'files_scanned':len(inv),'instruction_packets_found':len(packets),'files_archived':moved,'files_quarantined':quarantined,'manual_review':manual[:300],'active_canonical_files':registry.get('canonical_files',[]),'conflicts_found':conflicts,'conflicts_resolved':[],'graph_candidates_written':[{'path':graph_path,'count':graph_count}],'canonical_graph_materialization':False,'canonical_graph_writes':False,'tests_run':tests,'receipts_written':receipts,'restore_manifest':rel(manifest_path),'blockers':blockers_final}
    out=OUT/f'instruction_hygiene_{ts}.json'; payload['report_path']=rel(out); out.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print('REPORT_PATH='+rel(out)); print('INSTRUCTION_HYGIENE='+payload['verdict']); return 0 if payload['verdict']=='PASS' else 5
if __name__=='__main__': raise SystemExit(main())
