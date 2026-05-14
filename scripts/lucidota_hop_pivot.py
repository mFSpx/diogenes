#!/usr/bin/env python3
"""Queue-backed bounded Hop Pivot v1.

Uses Scout as the sensor; persists job/node/promotion state; emits workflow events.
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path
from urllib.parse import urlparse
import psycopg

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import lucidota_scout as scout  # noqa: E402
from ALGOS import pheromone  # noqa: E402

GRAPH_DB=os.environ.get('LUCIDOTA_GRAPH_DATABASE_URL','postgresql://mfspx@/lucidota_graph')
STATE_DB=os.environ.get('DBOS_SYSTEM_DATABASE_URL','postgresql://mfspx@/lucidota_state')
SCHEMA=ROOT/'06_SCHEMA'/'008_hop_pivot.sql'


def emit_event(job_id, phase, status, detail):
    with psycopg.connect(STATE_DB) as conn:
        conn.execute("""
            INSERT INTO lucidota_control.workflow_event (workflow_id, run_id, phase, status, source, detail)
            VALUES ('hop-pivot', %s, %s, %s, 'lucidota_hop_pivot', %s::jsonb)
        """, (str(job_id), phase, status, json.dumps(detail)))
        conn.commit()


def insert_job(conn, root, max_depth, max_pivots, threshold):
    conn.execute(SCHEMA.read_text())
    row=conn.execute("""
        INSERT INTO lucidota_pivot.hop_job (root_target,max_depth,max_pivots,promote_threshold,status,started_at)
        VALUES (%s,%s,%s,%s,'running',now()) RETURNING job_id
    """, (root,max_depth,max_pivots,threshold)).fetchone()
    conn.commit(); return row[0]


def upsert_node(conn, job_id, target, depth, status='queued', **kw):
    conn.execute("""
        INSERT INTO lucidota_pivot.hop_node (job_id,target,depth,status,decision,score,sha256,parent_target,detail,pheromone,utility,maintenance_cost,finished_at)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s, CASE WHEN %s IN ('succeeded','failed','skipped') THEN now() ELSE NULL END)
        ON CONFLICT (job_id,target) DO UPDATE SET
          status=EXCLUDED.status, decision=EXCLUDED.decision, score=EXCLUDED.score, sha256=EXCLUDED.sha256,
          parent_target=COALESCE(EXCLUDED.parent_target,lucidota_pivot.hop_node.parent_target), detail=EXCLUDED.detail,
          pheromone=EXCLUDED.pheromone, utility=EXCLUDED.utility, maintenance_cost=EXCLUDED.maintenance_cost,
          finished_at=EXCLUDED.finished_at
    """, (job_id,target,depth,status,kw.get('decision',''),kw.get('score',0),kw.get('sha256'),kw.get('parent'),json.dumps(kw.get('detail',{})),kw.get('pheromone',0.0),kw.get('utility',0.0),kw.get('maintenance_cost',0.0),status))


def add_promotion(conn, job_id, source, cand):
    conn.execute("""
        INSERT INTO lucidota_pivot.promotion (job_id,source_target,candidate,candidate_kind,score,reason)
        VALUES (%s,%s,%s,%s,%s,%s)
    """, (job_id, source, cand.get('candidate',''), cand.get('kind',''), int(cand.get('score',0)), cand.get('reason','')))



def select_links(conn, job_id, source, candidates, max_pivots, curiosity, gamma):
    edges=[]
    for cand in candidates:
        score=float(cand.get('score', 0))
        utility=score / 100.0
        cost=0.05 if cand.get('kind') == 'link' else 0.10
        ph=pheromone.evaporate(0.0, reinforcement=utility, phi=0.05)
        edges.append(pheromone.EdgeSignal(cand.get('candidate',''), ph, utility=utility, cost=cost))
    selected=pheromone.top_k(edges, max_pivots, curiosity=curiosity, gamma=gamma)
    selected_set={item for item,_ in selected}
    probs=dict(pheromone.probabilities(edges, curiosity=curiosity, gamma=gamma))
    for e in edges:
        conn.execute("""
            INSERT INTO lucidota_pivot.edge_signal
              (job_id,source_target,candidate,pheromone,utility,maintenance_cost,probability,selected,detail)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
        """, (job_id,source,e.item,e.pheromone,e.utility,e.cost,probs.get(e.item,0.0),e.item in selected_set,json.dumps({'curiosity':curiosity,'gamma':gamma})))
    return [item for item,_ in selected]


def run_scout(target, args):
    class A: pass
    a=A()
    a.max_bytes=args.max_bytes; a.timeout=args.timeout; a.fetch=args.fetch; a.vault=args.vault
    a.keyword=args.keyword; a.keywords=args.keywords; a.wayback=args.wayback; a.no_db=args.no_db
    a.db_url=args.db_url; a.state_db_url=args.state_db_url
    a.allow_local_addresses=args.allow_local_addresses
    a.hop_depth=0; a.max_pivots=args.max_pivots; a.promote_threshold=args.promote_threshold
    return scout.run_one(a, target)


def allowed_target(t):
    p=urlparse(t)
    return p.scheme in ('http','https') or p.scheme == ''


def main():
    ap=argparse.ArgumentParser(prog='lucidota-hop-pivot')
    ap.add_argument('target')
    ap.add_argument('--max-depth', type=int, default=1)
    ap.add_argument('--max-pivots', type=int, default=8)
    ap.add_argument('--promote-threshold', type=int, default=35)
    ap.add_argument('--keyword', action='append', default=[])
    ap.add_argument('--keywords', default='')
    ap.add_argument('--fetch', action='store_true')
    ap.add_argument('--wayback', action='store_true')
    ap.add_argument('--allow-local-addresses', action='store_true')
    ap.add_argument('--max-bytes', type=int, default=scout.DEFAULT_MAX_BYTES)
    ap.add_argument('--timeout', type=float, default=10.0)
    ap.add_argument('--vault', type=Path, default=scout.DEFAULT_VAULT)
    ap.add_argument('--db-url', default=GRAPH_DB)
    ap.add_argument('--state-db-url', default=STATE_DB)
    ap.add_argument('--no-db', action='store_true')
    ap.add_argument('--curiosity', type=float, default=0.05)
    ap.add_argument('--gamma', type=float, default=1.5)
    ap.add_argument('--json', action='store_true')
    args=ap.parse_args()

    with psycopg.connect(args.db_url) as conn:
        scout.apply_schema(args.db_url)
        job_id=insert_job(conn,args.target,args.max_depth,args.max_pivots,args.promote_threshold)
        emit_event(job_id,'hop_start','running',{'target':args.target,'max_depth':args.max_depth})
        frontier=[(args.target,0,None)]
        seen=set(); promotions=[]; nodes=0
        while frontier:
            target,depth,parent=frontier.pop(0)
            if target in seen: continue
            seen.add(target); nodes+=1
            if not allowed_target(target):
                upsert_node(conn,job_id,target,depth,'skipped',parent=parent,detail={'reason':'unsupported_scheme'}); continue
            try:
                upsert_node(conn,job_id,target,depth,'running',parent=parent); conn.commit()
                res=run_scout(target,args)
                max_score=max([int(c.get('score',0)) for c in res.pivot_candidates] or [0])
                utility=max_score/100.0
                ph=pheromone.evaporate(0.0, reinforcement=utility, phi=0.05)
                upsert_node(conn,job_id,target,depth,'succeeded',decision=res.decision,score=max_score,sha256=res.sha256,parent=parent,detail={'pivots':len(res.pivot_candidates),'mime':res.mime},pheromone=ph,utility=utility,maintenance_cost=0.05)
                for cand in res.pivot_candidates:
                    if int(cand.get('score',0)) >= args.promote_threshold:
                        add_promotion(conn,job_id,target,cand); promotions.append({'source':target,**cand})
                if depth < args.max_depth:
                    link_cands=[c for c in res.pivot_candidates if c.get('kind')=='link']
                    links=select_links(conn,job_id,target,link_cands,args.max_pivots,args.curiosity,args.gamma)
                    frontier.extend((link, depth+1, target) for link in links if link not in seen)
                conn.commit()
                emit_event(job_id,'hop_node','succeeded',{'target':target,'depth':depth,'decision':res.decision,'pivots':len(res.pivot_candidates),'max_score':max_score})
            except Exception as exc:
                upsert_node(conn,job_id,target,depth,'failed',parent=parent,detail={'error':str(exc)}); conn.commit()
                emit_event(job_id,'hop_node','failed',{'target':target,'depth':depth,'error':str(exc)})
        conn.execute("UPDATE lucidota_pivot.hop_job SET status='succeeded', finished_at=now(), detail=%s::jsonb WHERE job_id=%s", (json.dumps({'nodes':nodes,'promotions':len(promotions)}), job_id))
        conn.commit()
        emit_event(job_id,'hop_finish','succeeded',{'nodes':nodes,'promotions':len(promotions)})
    report={'ok':True,'job_id':str(job_id),'nodes':nodes,'promotions':len(promotions),'top_promotions':sorted(promotions,key=lambda c:-int(c.get('score',0)))[:10]}
    print(json.dumps(report, indent=None if args.json else 2, sort_keys=True))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
