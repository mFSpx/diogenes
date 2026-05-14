#!/usr/bin/env python3
"""Tiny Treelite routing-hint proof.

Builds a minimal Treelite tree with no XGBoost runtime dependency, stores the
compiled/lightweight artifact in the ignored vault, runs a prediction, and
persists the advisory result. DBOS policy remains the authority gate.
"""
from __future__ import annotations

import argparse, json, os
from pathlib import Path
import numpy as np
import psycopg
from treelite import model_builder as mb
import treelite.gtil as gtil

ROOT=Path(__file__).resolve().parents[1]
DB=os.environ.get('DBOS_SYSTEM_DATABASE_URL','postgresql://mfspx@/lucidota_state')
SCHEMA=ROOT/'06_SCHEMA'/'009_treelite_router.sql'
DEFAULT_ARTIFACT=ROOT/'03_VAULT'/'router'/'treelite_router_v0.tl'


def build_model():
    metadata=mb.Metadata(num_feature=3, task_type='kRegressor', average_tree_output=False, num_target=1, num_class=[1], leaf_vector_shape=(1,1))
    ann=mb.TreeAnnotation(num_tree=1, target_id=[0], class_id=[0])
    post=mb.PostProcessorFunc(name='identity')
    b=mb.ModelBuilder(threshold_type='float32', leaf_output_type='float32', metadata=metadata, tree_annotation=ann, postprocessor=post, base_scores=[0.0])
    b.start_tree()
    b.start_node(0); b.numerical_test(0, 0.5, default_left=True, opname='<', left_child_key=1, right_child_key=2); b.end_node()
    b.start_node(1); b.leaf(0.25); b.end_node()
    b.start_node(2); b.leaf(0.90); b.end_node()
    b.end_tree()
    return b.commit()


def main() -> int:
    ap=argparse.ArgumentParser(prog='lucidota-treelite-router')
    ap.add_argument('--artifact', type=Path, default=DEFAULT_ARTIFACT)
    ap.add_argument('--db-url', default=DB)
    ap.add_argument('--json', action='store_true')
    args=ap.parse_args()

    model=build_model()
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.artifact.write_bytes(model.serialize_bytes())
    features=np.array([[0.80, 1.0, 1.0]], dtype=np.float32)
    score=float(gtil.predict(model, features).reshape(-1)[0])
    route='scout-hop-promote' if score >= 0.5 else 'metadata-only'
    detail={'runtime':'treelite-gtil','xgboost_runtime':False,'features':['source_confidence','has_keyword','has_link']}
    with psycopg.connect(args.db_url) as conn:
        conn.execute(SCHEMA.read_text())
        conn.execute("""
            INSERT INTO lucidota_learning.treelite_router_run (status, artifact_uri, examples, route, score, detail)
            VALUES ('succeeded', %s, %s, %s, %s, %s::jsonb)
        """, (f'file://{args.artifact}', 1, route, score, json.dumps(detail)))
        conn.commit()
    report={'ok':True,'route':route,'score':score,'artifact':str(args.artifact),'xgboost_runtime':False}
    print(json.dumps(report, sort_keys=True) if args.json else report)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
