#!/usr/bin/env python3
"""Body Capture capture v0: operator-supervised HTTP/body snapshot into local CAS.

This is the first non-browser capture slice. Browser screenshots/DOM via
Playwright come next; this establishes the same evidence path: fetch, hash,
CAS, metadata row, delta row.
"""
from __future__ import annotations

import argparse, hashlib, html.parser, json, os, re, unicodedata
from pathlib import Path
from urllib.parse import urlparse
import psycopg

THIS_FILE=Path(__file__).resolve()
ROOT=THIS_FILE.parents[2] if THIS_FILE.parent.name == 'legacy' else THIS_FILE.parents[1]
DEFAULT_DB='postgresql://mfspx@/lucidota_graph'
DEFAULT_STATE_DB='postgresql://mfspx@/lucidota_state'
SCHEMA=ROOT/'06_SCHEMA'/'011_body_capture.sql'
SIGNAL_SCHEMA=ROOT/'06_SCHEMA'/'013_signal_ingress.sql'

import sys
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))
import lucidota_survey as survey  # noqa: E402
from ALGOS import tri_algo_conduit  # noqa: E402




class CanonicalHTML(html.parser.HTMLParser):
    SKIP = {'script', 'style', 'noscript', 'template'}
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.skip_depth=0
        self.text=[]
        self.skeleton=[]
    def handle_starttag(self, tag, attrs):
        tag=tag.lower()
        if tag in self.SKIP:
            self.skip_depth += 1
            return
        if self.skip_depth:
            return
        self.skeleton.append(('start', tag))
    def handle_endtag(self, tag):
        tag=tag.lower()
        if tag in self.SKIP and self.skip_depth:
            self.skip_depth -= 1
            return
        if self.skip_depth:
            return
        self.skeleton.append(('end', tag))
    def handle_data(self, data):
        if not self.skip_depth:
            self.text.append(data)


def canonical_hashes(data: bytes, mime: str)->tuple[str,str]:
    raw=data[:2_000_000].decode('utf-8', errors='ignore')
    if 'html' in (mime or '') or '<html' in raw[:2000].lower():
        parser=CanonicalHTML()
        parser.feed(raw)
        text=' '.join(parser.text)
        skeleton=' '.join(f'{kind}:{tag}' for kind,tag in parser.skeleton)
    else:
        text=raw
        skeleton='non_html_body'
    text=unicodedata.normalize('NFKC', text)
    text=re.sub(r'\s+', ' ', text).strip()
    return hashlib.sha256(text.encode('utf-8')).hexdigest(), hashlib.sha256(skeleton.encode('utf-8')).hexdigest()


def title_from_html(data: bytes)->str:
    text=data[:200000].decode('utf-8', errors='ignore')
    m=re.search(r'<title[^>]*>(.*?)</title>', text, re.I|re.S)
    return re.sub(r'\s+', ' ', m.group(1)).strip()[:240] if m else ''


def previous_capture(conn, source:str):
    return conn.execute("""
      SELECT capture_id, sha256 FROM lucidota_body_capture.capture
      WHERE source=%s AND status='succeeded' AND sha256 IS NOT NULL
      ORDER BY created_at DESC LIMIT 1
    """, (source,)).fetchone()


def main()->int:
    ap=argparse.ArgumentParser(prog='lucidota-body_capture-capture')
    ap.add_argument('source')
    ap.add_argument('--db-url', default=os.environ.get('LUCIDOTA_GRAPH_DATABASE_URL', DEFAULT_DB))
    ap.add_argument('--state-db-url', default=os.environ.get('DBOS_SYSTEM_DATABASE_URL', DEFAULT_STATE_DB))
    ap.add_argument('--vault', type=Path, default=survey.DEFAULT_VAULT)
    ap.add_argument('--max-bytes', type=int, default=survey.DEFAULT_MAX_BYTES)
    ap.add_argument('--timeout', type=float, default=10.0)
    ap.add_argument('--allow-local-addresses', action='store_true')
    ap.add_argument('--disable-signal-gate', action='store_true',
                    help='Bypass tri-algo signal gate and always capture fetched bytes.')
    ap.add_argument('--json', action='store_true')
    args=ap.parse_args()

    parsed=urlparse(args.source)
    if parsed.scheme not in ('http','https'):
        raise SystemExit('Body Capture v0 captures http/https sources only')
    result, data=survey.fetch_target(args.source,args.max_bytes,args.timeout,True,args.allow_local_addresses)
    if data is None:
        raise SystemExit(json.dumps({'ok':False,'error':'no capture bytes','decision':result.decision}))

    # Tri-algo conduit: passive monitor -> Hoeffding gate -> Serpentina recovery.
    # This gates expensive capture work; canonical truth still lives in CAS/DB rows.
    structural_links=len(re.findall(rb'<a\s+[^>]*href=', data[:500000], re.I))
    observations=max(512, min(5000, len(data)//256 + structural_links + 1))
    signal_decision=tri_algo_conduit.decide(
        data,
        observations=observations,
        status_code=result.status_code,
        mime=result.mime,
        structural_links=structural_links,
        max_bytes=args.max_bytes,
    )
    with psycopg.connect(args.db_url) as conn:
        conn.execute(SCHEMA.read_text())
        conn.execute(SIGNAL_SCHEMA.read_text())
        conn.execute("""
          INSERT INTO lucidota_signal.ingress_decision
            (source,subsystem,action,confidence_gap,epsilon,signal_score,noise_score,
             dormancy_probability,recovery_priority,reason,detail)
          VALUES (%s,'body_capture',%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
        """, (
            args.source,
            signal_decision.action,
            signal_decision.confidence_gap,
            signal_decision.epsilon,
            signal_decision.signal_score,
            signal_decision.noise_score,
            signal_decision.dormancy_probability,
            signal_decision.recovery_priority,
            signal_decision.reason,
            json.dumps({'observations':observations,'structural_links':structural_links,'disabled':args.disable_signal_gate}),
        ))
        if signal_decision.action == 'standby' and not args.disable_signal_gate:
            conn.commit()
            report={'ok':True,'skipped':True,'reason':'tri_algo_standby','signal_gate':signal_decision.__dict__}
            print(json.dumps(report, sort_keys=True) if args.json else report)
            return 0
        if signal_decision.action == 'recover' and not args.disable_signal_gate:
            conn.commit()
            report={'ok':False,'skipped':True,'reason':'tri_algo_recovery','signal_gate':signal_decision.__dict__}
            print(json.dumps(report, sort_keys=True) if args.json else report)
            return 1

        digest, cas_uri, _path=survey.store_cas(args.vault,data)
        title=title_from_html(data)
        content_hash, structure_hash = canonical_hashes(data, result.mime)
        prev=previous_capture(conn,args.source)
        row=conn.execute("""
          INSERT INTO lucidota_body_capture.capture
            (source,capture_kind,status,sha256,cas_uri,size_bytes,mime,title,content_hash,structure_hash,detail)
          VALUES (%s,'http_body','succeeded',%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
          RETURNING capture_id
        """, (args.source,digest,cas_uri,len(data),result.mime,title,content_hash,structure_hash,json.dumps({'method':result.method,'status_code':result.status_code,'final_url':result.final_url,'signal_gate':signal_decision.__dict__}))).fetchone()
        capture_id=row[0]
        old_id=prev[0] if prev else None; old_sha=prev[1] if prev else ''
        changed=(old_sha != '' and old_sha != digest)
        conn.execute("""
          INSERT INTO lucidota_body_capture.delta
            (source,old_capture_id,new_capture_id,old_sha256,new_sha256,changed,detail)
          VALUES (%s,%s,%s,%s,%s,%s,%s::jsonb)
        """, (args.source,old_id,capture_id,old_sha,digest,changed,json.dumps({'first_capture':prev is None})))
        conn.execute("""
          INSERT INTO lucidota_body_capture.workflow_event_outbox
            (capture_id, run_id, detail)
          VALUES (%s, %s, %s::jsonb)
        """, (capture_id, str(capture_id), json.dumps({'source':args.source,'sha256':digest,'decision':'captured','signal_gate':signal_decision.__dict__})))
        conn.commit()
    try:
        with psycopg.connect(args.state_db_url) as sconn:
            sconn.execute("""
              INSERT INTO lucidota_control.workflow_event (workflow_id, run_id, phase, status, source, detail)
              VALUES ('body_capture-capture', %s, 'capture', 'succeeded', 'lucidota_body_capture', %s::jsonb)
            """, (str(capture_id), json.dumps({'source':args.source,'sha256':digest,'decision':'captured'})))
            sconn.commit()
        with psycopg.connect(args.db_url) as gconn:
            gconn.execute("""
              UPDATE lucidota_body_capture.workflow_event_outbox
              SET delivery_status='delivered', attempts=attempts+1, delivered_at=now(), last_error=''
              WHERE capture_id=%s AND delivery_status='pending'
            """, (capture_id,))
            gconn.commit()
    except Exception as state_exc:
        try:
            with psycopg.connect(args.db_url) as gconn:
                gconn.execute("""
                  UPDATE lucidota_body_capture.workflow_event_outbox
                  SET delivery_status='failed', attempts=attempts+1, last_error=%s
                  WHERE capture_id=%s AND delivery_status='pending'
                """, (f'state workflow_event delivery failed; graph-local outbox retained: {state_exc}', capture_id))
                gconn.commit()
        except Exception as outbox_exc:
            print(f"warning: workflow outbox update failed for capture {capture_id}: {outbox_exc}", file=sys.stderr)
    report={'ok':True,'capture_id':str(capture_id),'sha256':digest,'cas_uri':cas_uri,'size_bytes':len(data),'changed':changed,'content_hash':content_hash,'structure_hash':structure_hash,'title':title,'signal_gate':signal_decision.__dict__}
    print(json.dumps(report, sort_keys=True) if args.json else report)
    return 0
if __name__=='__main__': raise SystemExit(main())
