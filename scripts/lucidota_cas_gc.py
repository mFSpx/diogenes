#!/usr/bin/env python3
"""Report-first CAS GC for LUCIDOTA.

Default mode never deletes and never moves bytes. It scans local CAS files,
compares them to Postgres references, writes a durable report, and prints counts.
Optional quarantine mode moves only orphan candidates into an ignored quarantine
folder with a manifest trail. There is intentionally no delete mode here.
"""
from __future__ import annotations

import argparse, hashlib, json, os, shutil, sys
from pathlib import Path
import psycopg

ROOT=Path(__file__).resolve().parents[1]
SCHEMA=ROOT/'06_SCHEMA'/'005_cas_manifest.sql'
DEFAULT_DB='postgresql://mfspx@/lucidota_graph'
DEFAULT_VAULT=ROOT/'03_VAULT'/'cas'
DEFAULT_QUARANTINE=ROOT/'03_VAULT'/'quarantine'/'cas_orphans'


def iter_cas_files(vault: Path):
    if not vault.exists():
        return
    for path in vault.rglob('*'):
        if path.is_file() and not path.name.endswith('.tmp'):
            yield path


def sha256_file(path: Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_references(conn)->set[str]:
    # Authoritative semantic references only. cas_manifest is an index of bytes,
    # not proof that a source/record metadata commit succeeded.
    refs=set()
    queries=[
        "SELECT sha256 FROM lucidota_survey.artifact WHERE sha256 IS NOT NULL AND sha256 <> ''",
    ]
    for sql in queries:
        try:
            refs.update(r[0] for r in conn.execute(sql).fetchall())
        except Exception as exc:
            print(f"warning: CAS reference query failed: {exc}", file=sys.stderr)
    return refs


def main()->int:
    ap=argparse.ArgumentParser(prog='lucidota-cas-gc')
    ap.add_argument('--vault', type=Path, default=Path(os.environ.get('LUCIDOTA_CAS_VAULT', DEFAULT_VAULT)))
    ap.add_argument('--quarantine-root', type=Path, default=DEFAULT_QUARANTINE)
    ap.add_argument('--db-url', default=os.environ.get('LUCIDOTA_GRAPH_DATABASE_URL', DEFAULT_DB))
    ap.add_argument('--apply-quarantine', action='store_true', help='Move orphan candidates to ignored quarantine. Never deletes.')
    ap.add_argument('--recover-journal', action='store_true', help='Classify orphan candidates with local CAS journal context when present.')
    ap.add_argument('--json', action='store_true')
    args=ap.parse_args()

    mode='quarantine' if args.apply_quarantine else 'report'
    candidates=[]; referenced=orphans=corrupt=quarantined=0; total=0
    with psycopg.connect(args.db_url) as conn:
        conn.execute(SCHEMA.read_text())
        refs=load_references(conn)
        journal={}
        if args.recover_journal:
            try:
                from lucidota_cas_journal import load_records
                for rec in load_records():
                    if rec.get('sha256') and rec.get('stage') == 'written':
                        journal[rec['sha256']] = rec
            except (OSError, json.JSONDecodeError):
                journal={}
        row=conn.execute("""
            INSERT INTO lucidota_vault.cas_gc_run (mode,status,detail)
            VALUES (%s,'succeeded',%s::jsonb) RETURNING run_id
        """, (mode, json.dumps({'delete_mode': False}))).fetchone()
        run_id=row[0]
        qroot=args.quarantine_root / str(run_id)
        for path in iter_cas_files(args.vault) or []:
            total+=1
            digest=sha256_file(path)
            rel=path.relative_to(args.vault).as_posix()
            status='referenced' if digest in refs else 'orphan_candidate'
            reason='referenced in authoritative Postgres metadata' if status == 'referenced' else 'no authoritative Postgres metadata reference found'
            if status == 'orphan_candidate' and digest in journal:
                reason='filesystem journal exists but metadata commit is absent; recover or quarantine'
            qrel=''
            if len(path.name)==64 and path.name != digest:
                status='corrupt'; reason='path digest does not match content digest'; corrupt+=1
            elif status == 'referenced':
                referenced+=1
            else:
                orphans+=1
                if args.apply_quarantine:
                    dest=qroot / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(path), str(dest))
                    status='quarantined'; qrel=dest.relative_to(args.quarantine_root).as_posix(); quarantined+=1
            candidates.append({'sha256':digest,'path':rel,'status':status,'size':path.stat().st_size if path.exists() else 0,'reason':reason})
            conn.execute("""
                INSERT INTO lucidota_vault.cas_gc_candidate
                  (run_id, sha256, original_relative_path, size_bytes, status, quarantine_relative_path, reason)
                VALUES (%s,%s,%s,%s,%s,%s,%s)
            """, (run_id,digest,rel,candidates[-1]['size'],status,qrel,reason))
        conn.execute("""
            UPDATE lucidota_vault.cas_gc_run SET
              total_files=%s, referenced_files=%s, orphan_candidates=%s,
              quarantined_files=%s, corrupt_files=%s, detail=%s::jsonb
            WHERE run_id=%s
        """, (total,referenced,orphans,quarantined,corrupt,json.dumps({'sample':candidates[:25],'delete_mode':False}),run_id))
        conn.commit()
    report={'ok':corrupt==0,'mode':mode,'run_id':str(run_id),'total_files':total,'referenced_files':referenced,'orphan_candidates':orphans,'quarantined_files':quarantined,'corrupt_files':corrupt,'delete_mode':False}
    print(json.dumps(report, sort_keys=True) if args.json else report)
    return 0 if corrupt == 0 else 1
if __name__=='__main__':
    raise SystemExit(main())
