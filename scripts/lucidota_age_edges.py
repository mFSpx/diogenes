#!/usr/bin/env python3
"""Write Survey/Hop/CAS provenance edges into Apache AGE."""
from __future__ import annotations
import argparse, json, os
import psycopg
DB=os.environ.get('LUCIDOTA_GRAPH_DATABASE_URL','postgresql://mfspx@/lucidota_graph')
GRAPH='lucidota_provenance'

def q(s:str)->str:
    return json.dumps(s)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--limit',type=int,default=200)
    ap.add_argument('--json',action='store_true')
    args=ap.parse_args()
    with psycopg.connect(DB) as conn:
        cur=conn.cursor()
        cur.execute("LOAD 'age'"); cur.execute('SET search_path = ag_catalog, "$user", public')
        exists=cur.execute("SELECT count(*) FROM ag_catalog.ag_graph WHERE name=%s", (GRAPH,)).fetchone()[0]
        if not exists: cur.execute(f"SELECT create_graph('{GRAPH}')")
        rows=cur.execute("""
          SELECT source_target,candidate,candidate_kind,score,reason
          FROM lucidota_survey.pivot_candidate
          ORDER BY created_at DESC LIMIT %s
        """, (args.limit,)).fetchall()
        edges=0
        for source,cand,kind,score,reason in rows:
            cypher=f"""
            MERGE (s:Target {{id: {q(source)}}})
            MERGE (c:Target {{id: {q(cand)}}})
            MERGE (s)-[r:PIVOTS_TO]->(c)
            SET r.kind={q(kind)}, r.score={int(score)}, r.reason={q(reason)}
            RETURN 1
            """
            cur.execute(f"SELECT * FROM cypher('{GRAPH}', $$ {cypher} $$) AS (ok agtype)")
            edges+=1
        hrows=cur.execute("""
          SELECT source_target,candidate,candidate_kind,score,reason
          FROM lucidota_pivot.promotion
          ORDER BY created_at DESC LIMIT %s
        """, (args.limit,)).fetchall()
        for source,cand,kind,score,reason in hrows:
            cypher=f"""
            MERGE (s:Target {{id: {q(source)}}})
            MERGE (c:Target {{id: {q(cand)}}})
            MERGE (s)-[r:PROMOTES]->(c)
            SET r.kind={q(kind)}, r.score={int(score)}, r.reason={q(reason)}
            RETURN 1
            """
            cur.execute(f"SELECT * FROM cypher('{GRAPH}', $$ {cypher} $$) AS (ok agtype)")
            edges+=1
        arows=cur.execute("""
          SELECT source_url, cas_uri, sha256, size_bytes, mime
          FROM lucidota_survey.artifact
          WHERE source_url IS NOT NULL AND source_url <> ''
          ORDER BY created_at DESC LIMIT %s
        """, (args.limit,)).fetchall()
        for source,cas_uri,sha256,size_bytes,mime in arows:
            cypher=f"""
            MERGE (s:Target {{id: {q(source)}}})
            MERGE (a:Artifact {{id: {q(cas_uri)}}})
            SET a.sha256={q(sha256)}, a.size_bytes={int(size_bytes)}, a.mime={q(mime or '')}
            MERGE (s)-[r:STORED_AS]->(a)
            SET r.storage='local_cas'
            RETURN 1
            """
            cur.execute(f"SELECT * FROM cypher('{GRAPH}', $$ {cypher} $$) AS (ok agtype)")
            edges+=1
        try:
            crows=cur.execute("""
              SELECT source, cas_uri, sha256, size_bytes, mime, capture_kind
              FROM lucidota_body_capture.capture
              WHERE cas_uri IS NOT NULL AND cas_uri <> ''
              ORDER BY created_at DESC LIMIT %s
            """, (args.limit,)).fetchall()
        except psycopg.Error:
            crows=[]
        for source,cas_uri,sha256,size_bytes,mime,kind in crows:
            cypher=f"""
            MERGE (s:Target {{id: {q(source)}}})
            MERGE (a:Artifact {{id: {q(cas_uri)}}})
            SET a.sha256={q(sha256 or '')}, a.size_bytes={int(size_bytes or 0)}, a.mime={q(mime or '')}
            MERGE (s)-[r:CAPTURED_AS]->(a)
            SET r.kind={q(kind or '')}, r.storage='local_cas'
            RETURN 1
            """
            cur.execute(f"SELECT * FROM cypher('{GRAPH}', $$ {cypher} $$) AS (ok agtype)")
            edges+=1
        conn.commit()
    print(json.dumps({'ok':True,'graph':GRAPH,'edges_written':edges}, sort_keys=True) if args.json else {'ok':True,'edges_written':edges})
if __name__=='__main__': raise SystemExit(main())
