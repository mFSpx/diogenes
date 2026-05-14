#!/usr/bin/env python3
"""Hydra browser capture contract.

Browser rendering is policy-gated fallback, not default extraction. If no browser
binary is available, this reports skipped instead of fake success.
"""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, tempfile
from pathlib import Path
import psycopg

ROOT=Path(__file__).resolve().parents[1]
GRAPH_DB=os.environ.get('LUCIDOTA_GRAPH_DATABASE_URL','postgresql://mfspx@/lucidota_graph')
SCHEMA=ROOT/'06_SCHEMA'/'011_hydra_capture.sql'
DEFAULT_VAULT=ROOT/'03_VAULT'/'cas'

import sys
sys.path.insert(0, str(ROOT/'scripts'))
import lucidota_scout as scout  # noqa: E402


def browser_bin()->str|None:
    for name in ('chromium','chromium-browser','google-chrome','google-chrome-stable'):
        p=shutil.which(name)
        if p: return p
    return None


def visual_hash(data: bytes)->str:
    # v0: byte hash placeholder. Later: pHash/SSIM features after stable browser path.
    return hashlib.sha256(data).hexdigest()


def main()->int:
    ap=argparse.ArgumentParser(prog='lucidota-hydra-browser-capture')
    ap.add_argument('source')
    ap.add_argument('--db-url', default=GRAPH_DB)
    ap.add_argument('--vault', type=Path, default=DEFAULT_VAULT)
    ap.add_argument('--width', type=int, default=1365)
    ap.add_argument('--height', type=int, default=900)
    ap.add_argument('--timeout', type=int, default=20)
    ap.add_argument('--json', action='store_true')
    args=ap.parse_args()
    b=browser_bin()
    if not b:
        report={'ok':True,'skipped':True,'reason':'no_browser_binary','browser_default':False}
        print(json.dumps(report, sort_keys=True) if args.json else report)
        return 0
    with tempfile.TemporaryDirectory(prefix='lucidota-hydra-browser.') as td:
        out=Path(td)/'shot.png'
        profile=Path(td)/'profile'
        cmd=[b,'--headless=new','--disable-gpu','--no-first-run','--disable-dev-shm-usage','--hide-scrollbars',f'--window-size={args.width},{args.height}',f'--user-data-dir={profile}',f'--screenshot={out}',args.source]
        r=subprocess.run(cmd, text=True, capture_output=True, timeout=args.timeout)
        if not out.exists():
            err=r.stderr[-500:] or r.stdout[-500:]
            if 'snap command-chain' in err or 'slot is connected' in err or 'Permission denied' in err or 'bytes written to file' in err:
                report={'ok':True,'skipped':True,'reason':'browser_output_unavailable','browser_default':False,'detail':err.strip()[:180]}
                print(json.dumps(report, sort_keys=True) if args.json else report)
                return 0
            report={'ok':False,'skipped':False,'error':err}
            print(json.dumps(report, sort_keys=True) if args.json else report)
            return 1
        data=out.read_bytes()
    digest, cas_uri, _=scout.store_cas(args.vault,data)
    vhash=visual_hash(data)
    with psycopg.connect(args.db_url) as conn:
        conn.execute(SCHEMA.read_text())
        row=conn.execute("""
          INSERT INTO lucidota_hydra.capture
            (source,capture_kind,status,sha256,cas_uri,size_bytes,mime,title,visual_hash,detail)
          VALUES (%s,'screenshot_placeholder','succeeded',%s,%s,%s,'image/png','',%s,%s::jsonb)
          RETURNING capture_id
        """, (args.source,digest,cas_uri,len(data),vhash,json.dumps({'browser':b,'viewport':[args.width,args.height],'stabilization':'cli_v0_hide_scrollbars','returncode':r.returncode,'stderr_tail':r.stderr[-240:]}))).fetchone()
        conn.commit()
    report={'ok':True,'skipped':False,'capture_id':str(row[0]),'sha256':digest,'cas_uri':cas_uri,'visual_hash':vhash,'size_bytes':len(data)}
    print(json.dumps(report, sort_keys=True) if args.json else report)
    return 0
if __name__=='__main__': raise SystemExit(main())
