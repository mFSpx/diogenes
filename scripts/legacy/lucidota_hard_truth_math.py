#!/usr/bin/env python3
"""Hard-math telemetry for LUCIDOTA.

Implements four factual-ish signal families:
  1) Linguistic Style Matching (LSM) over chat turns.
  2) Observed-state Markov/HMM-proxy transitions over brain-map states.
  3) Stylometry with deterministic hashed features + hinge classifier.
  4) Semantic isolation via diagonal Mahalanobis distance over pgvector embeddings.

Important: these are document/communication metrics, not psychological diagnosis and
not proof of dominance, identity, or intent.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import psycopg

ROOT = Path(__file__).resolve().parents[1]
STORAGE_DSN = os.environ.get("LUCIDOTA_GO_STORAGE_DSN", "postgresql:///lucidota_storage")
STATE_DSN = os.environ.get("LUCIDOTA_GO_STATE_DSN", os.environ.get("DBOS_SYSTEM_DATABASE_URL", "postgresql:///lucidota_state"))
SCHEMA = ROOT / "06_SCHEMA" / "021_hard_truth_math.sql"
WORKFLOW_SCHEMA = ROOT / "06_SCHEMA" / "006_workflow_registry.sql"
BRAIN_MAP = ROOT / "05_OUTPUTS" / "korpus_krampii" / "krampus_brain_map.jsonl"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from ALGOS.hard_truth_math import (  # type: ignore  # noqa: E402
    FUNCTION_CATS,
    lsm_score,
    lsm_vector,
    parse_iso,
    parse_vector,
    persona_label,
    softmax,
    stylometry_features,
    train_hinge_classifier,
)
from lucidota_artifact_ingest import ensure_state, normalize_ws  # type: ignore  # noqa: E402

def jdump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)


def ensure_schema(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA.read_text(encoding="utf-8"))
    conn.commit()


def emit_event(run_uuid: str, phase: str, status: str, detail: dict[str, Any]) -> str:
    with psycopg.connect(STATE_DSN) as conn:
        ensure_state(conn)
        conn.execute(WORKFLOW_SCHEMA.read_text(encoding="utf-8"))
        row = conn.execute(
            """
            INSERT INTO lucidota_control.workflow_event(workflow_id, run_id, phase, status, source, detail)
            VALUES ('hard-truth-math',%s,%s,%s,'hard_truth_math',%s::jsonb)
            RETURNING event_id::text
            """,
            (run_uuid, phase, status, jdump(detail)),
        ).fetchone()
        conn.commit()
        return str(row[0])


def start_run(conn: psycopg.Connection, kind: str, detail: dict[str, Any]) -> str:
    row = conn.execute(
        """
        INSERT INTO lucidota_hardmath.analysis_run(run_kind, status, detail)
        VALUES (%s,'running',%s::jsonb)
        RETURNING run_uuid::text
        """,
        (kind, jdump(detail)),
    ).fetchone()
    conn.commit()
    return str(row[0])


def finish_run(conn: psycopg.Connection, run_uuid: str, status: str, detail: dict[str, Any]) -> None:
    conn.execute(
        """
        UPDATE lucidota_hardmath.analysis_run
        SET status=%s, finished_at=now(), detail=detail || %s::jsonb
        WHERE run_uuid=%s::uuid
        """,
        (status, jdump(detail), run_uuid),
    )
    conn.commit()


def run_lsm(conn: psycopg.Connection, run_uuid: str, limit: int) -> dict[str, Any]:
    """Compute style-alignment over adjacent turns in chat + comm dumps.

    Chat dumps and universal communications dumps intentionally share the same
    internal row shape here:
      conversation_uuid, provider, title, role, speaker, create_time,
      sequence_index, content_text
    """
    rows = list(conn.execute(
        """
        SELECT c.conversation_uuid::text, m.provider, c.title, m.role, COALESCE(NULLIF(m.author_name,''), m.role) AS speaker,
               m.create_time, m.sequence_index, m.content_text
        FROM lucidota_chatdump.message m
        JOIN lucidota_chatdump.conversation c ON c.conversation_uuid=m.conversation_uuid
        WHERE m.content_text <> ''
        ORDER BY c.conversation_uuid, m.create_time ASC NULLS LAST, m.sequence_index ASC
        LIMIT %s
        """,
        (limit,),
    ).fetchall())
    commdump_available = bool(conn.execute("SELECT to_regclass('lucidota_commdump.message') IS NOT NULL").fetchone()[0])
    comm_rows_seen = 0
    if commdump_available:
        comm_rows = conn.execute(
            """
            SELECT t.thread_uuid::text,
                   'commdump:' || m.source_kind AS provider,
                   COALESCE(NULLIF(t.title,''), NULLIF(m.subject,''), t.provider_thread_id, 'comm thread') AS title,
                   m.source_kind AS role,
                   COALESCE(NULLIF(m.sender,''), m.source_kind, 'unknown') AS speaker,
                   m.occurred_at AS create_time,
                   m.sequence_index,
                   COALESCE(NULLIF(m.subject,''),'') || CASE WHEN m.subject <> '' AND m.content_text <> '' THEN E'\n' ELSE '' END || m.content_text AS content_text
            FROM lucidota_commdump.message m
            JOIN lucidota_commdump.thread t ON t.thread_uuid=m.thread_uuid
            WHERE m.content_text <> '' OR m.subject <> ''
            ORDER BY t.thread_uuid, m.occurred_at ASC NULLS LAST, m.sequence_index ASC
            LIMIT %s
            """,
            (limit,),
        ).fetchall()
        comm_rows_seen = len(comm_rows)
        rows.extend(comm_rows)
    by_conv: dict[str, list[Any]] = defaultdict(list)
    for r in rows:
        by_conv[str(r[0])].append(r)
    agg: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for conv_uuid, msgs in by_conv.items():
        for prev, cur in zip(msgs, msgs[1:]):
            role_a = str(prev[4] or prev[3] or "")
            role_b = str(cur[4] or cur[3] or "")
            if not role_a or not role_b or role_a == role_b:
                continue
            score, detail = lsm_score(lsm_vector(prev[7] or ""), lsm_vector(cur[7] or ""))
            key = (conv_uuid, str(prev[1]), str(prev[2] or ""), " -> ".join([role_a, role_b]))
            bucket = agg.setdefault(key, {"scores": [], "details": [], "first": prev[5], "last": cur[5], "a": role_a, "b": role_b})
            bucket["scores"].append(score)
            bucket["details"].append(detail)
            if prev[5] and (bucket["first"] is None or prev[5] < bucket["first"]):
                bucket["first"] = prev[5]
            if cur[5] and (bucket["last"] is None or cur[5] > bucket["last"]):
                bucket["last"] = cur[5]
    inserted = 0
    for (conv_uuid, provider, title, _pair), b in agg.items():
        avg = float(sum(b["scores"]) / len(b["scores"])) if b["scores"] else 0.0
        cats: dict[str, float] = {}
        for cat in FUNCTION_CATS:
            vals = [d.get(cat, 0.0) for d in b["details"]]
            cats[cat] = round(sum(vals) / len(vals), 6) if vals else 0.0
        conn.execute(
            """
            INSERT INTO lucidota_hardmath.lsm_pair(run_uuid, provider, conversation_uuid, conversation_title, speaker_a, speaker_b, turn_pairs, lsm_score, function_word_detail, first_at, last_at)
            VALUES (%s::uuid,%s,%s::uuid,%s,%s,%s,%s,%s,%s::jsonb,%s,%s)
            """,
            (run_uuid, provider, conv_uuid, title[:1000], b["a"], b["b"], len(b["scores"]), avg, jdump(cats), b["first"], b["last"]),
        )
        inserted += 1
    conn.commit()
    return {"lsm_pairs": inserted, "messages_seen": len(rows), "comm_messages_seen": comm_rows_seen, "conversations_seen": len(by_conv)}


def load_brain_states(limit: int) -> list[tuple[dt.datetime, str, str]]:
    points: list[tuple[dt.datetime, str, str]] = []
    if BRAIN_MAP.exists():
        with BRAIN_MAP.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                state = str(obj.get("operator_cluster_hint") or obj.get("assigned_cluster_id") or "unknown")
                t = parse_iso((obj.get("metadata") or {}).get("file_time"))
                path = str((obj.get("metadata") or {}).get("path") or "")
                if t and state:
                    points.append((t, state, path))
    points.sort(key=lambda x: (x[0], x[2]))
    return points[:limit] if limit else points


def run_state_transitions(conn: psycopg.Connection, run_uuid: str, limit: int, stress: float) -> dict[str, Any]:
    seq = load_brain_states(limit)
    counts: dict[tuple[str, str], int] = Counter()
    gaps: dict[tuple[str, str], list[float]] = defaultdict(list)
    outgoing: Counter[str] = Counter()
    for (t1, s1, _), (t2, s2, _) in zip(seq, seq[1:]):
        key = (s1, s2)
        counts[key] += 1
        outgoing[s1] += 1
        gaps[key].append(max(0.0, (t2 - t1).total_seconds()))
    inserted = 0
    states = sorted({s for _, s, _ in seq})
    stress_targets = {"poetic_purge", "forensic_shield", "resource_exhaustion_cluster", "third_person_dissociation", "tactical_wrath", "paladin_protocol_cluster"}
    for (s1, s2), n in counts.items():
        prob = n / max(1, outgoing[s1])
        stress_prob = prob
        if any(tok in s2 for tok in stress_targets) or s2 in stress_targets:
            stress_prob = max(0.0, min(1.0, prob * (1.0 + stress)))
        conn.execute(
            """
            INSERT INTO lucidota_hardmath.state_transition(run_uuid, model_kind, from_state, to_state, transition_count, transition_probability, avg_gap_seconds, stress_adjusted_probability, detail)
            VALUES (%s::uuid,'observed_markov_hmm_proxy',%s,%s,%s,%s,%s,%s,%s::jsonb)
            """,
            (run_uuid, s1, s2, n, prob, statistics.mean(gaps[(s1, s2)]) if gaps[(s1, s2)] else None, stress_prob, jdump({"states": len(states), "stress": stress, "caveat": "Observed-state transition model; hidden-state HMM training requires more labeled sequence depth."})),
        )
        inserted += 1
    conn.commit()
    return {"state_points": len(seq), "states": len(states), "transitions": inserted}


def load_stylometry_samples(conn: psycopg.Connection, limit: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT c.component_uuid::text, fo.first_seen_at, c.title, c.content
        FROM lucidota_korpus.component c
        JOIN lucidota_korpus.file_object fo ON fo.file_uuid=c.file_uuid
        WHERE c.content <> ''
        ORDER BY fo.first_seen_at ASC, c.created_at ASC
        LIMIT %s
        """,
        (limit,),
    ).fetchall()
    out = []
    for uuid, occurred_at, title, content in rows:
        text = f"{title or ''}\n{content or ''}"
        out.append({"uuid": str(uuid), "occurred_at": occurred_at, "text": text, "label": persona_label(text)})
    return out


def run_stylometry(conn: psycopg.Connection, run_uuid: str, limit: int) -> dict[str, Any]:
    samples = load_stylometry_samples(conn, limit)
    labels, W, packed = train_hinge_classifier(samples)
    if not labels:
        return {"stylometry_samples": len(samples), "labels": 0, "attributions": 0}
    mu, sigma = packed[:96], packed[96:]
    inserted = 0
    for s in samples:
        x = (stylometry_features(s["text"]) - mu) / sigma
        scores = W.dot(x)
        probs = softmax(scores)
        order = list(np.argsort(-probs))
        pred = labels[order[0]] if order else ""
        conf = float(probs[order[0]]) if order else 0.0
        top = {labels[i]: round(float(probs[i]), 6) for i in order[:5]}
        conn.execute(
            """
            INSERT INTO lucidota_hardmath.stylometry_attribution(run_uuid, source_kind, source_uuid, occurred_at, observed_label, predicted_label, confidence, top_scores, feature_detail, excerpt)
            VALUES (%s::uuid,'korpus_component',%s::uuid,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s)
            ON CONFLICT(run_uuid, source_kind, source_uuid) DO NOTHING
            """,
            (run_uuid, s["uuid"], s["occurred_at"], s.get("label") or "", pred, conf, jdump(top), jdump({"model": "linear_hinge_stylometry", "labels": labels}), normalize_ws(s["text"])[:300]),
        )
        inserted += 1
    conn.commit()
    return {"stylometry_samples": len(samples), "labels": len(labels), "attributions": inserted, "label_set": labels}


def run_mahalanobis(conn: psycopg.Connection, run_uuid: str, limit: int, outliers: int) -> dict[str, Any]:
    rows = conn.execute(
        """
        SELECT c.component_uuid::text, c.file_uuid::text, fo.first_seen_at, c.title, fo.first_seen_path, left(c.content, 500), c.embedding::text
        FROM lucidota_korpus.component c
        JOIN lucidota_korpus.file_object fo ON fo.file_uuid=c.file_uuid
        WHERE c.embedding IS NOT NULL
        ORDER BY fo.first_seen_at ASC, c.created_at ASC
        LIMIT %s
        """,
        (limit,),
    ).fetchall()
    vecs: list[np.ndarray] = []
    metas: list[Any] = []
    for r in rows:
        v = parse_vector(r[6])
        if v is not None and v.size > 0 and np.isfinite(v).all():
            vecs.append(v)
            metas.append(r)
    if len(vecs) < 3:
        return {"embedded_components": len(vecs), "outliers": 0, "reason": "need at least 3 embeddings"}
    X = np.vstack(vecs)
    mu = X.mean(axis=0)
    var = X.var(axis=0) + 1e-6
    diff = X - mu
    distances = np.sqrt(np.sum((diff * diff) / var, axis=1))
    order = list(np.argsort(-distances))[:outliers]
    for rank, idx in enumerate(order, 1):
        r = metas[idx]
        conn.execute(
            """
            INSERT INTO lucidota_hardmath.semantic_outlier(run_uuid, component_uuid, file_uuid, occurred_at, mahalanobis_distance, z_rank, title, path, excerpt, detail)
            VALUES (%s::uuid,%s::uuid,%s::uuid,%s,%s,%s,%s,%s,%s,%s::jsonb)
            ON CONFLICT(run_uuid, component_uuid) DO NOTHING
            """,
            (run_uuid, r[0], r[1], r[2], float(distances[idx]), rank, str(r[3] or "")[:500], str(r[4] or ""), normalize_ws(str(r[5] or ""))[:500], jdump({"method": "diagonal_mahalanobis", "embedding_count": len(vecs), "dimension": int(X.shape[1])})),
        )
    conn.commit()
    return {"embedded_components": len(vecs), "dimension": int(X.shape[1]), "outliers": len(order), "max_distance": float(distances[order[0]]) if order else 0.0}


def cmd_run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    with psycopg.connect(STORAGE_DSN) as conn:
        ensure_schema(conn)
        run_uuid = start_run(conn, "hard_truth_math_all", {"limit": args.limit, "outliers": args.outliers, "stress": args.stress})
        detail: dict[str, Any] = {"run_uuid": run_uuid}
        status = "succeeded"
        try:
            if args.only in {"all", "lsm"}:
                detail["lsm"] = run_lsm(conn, run_uuid, args.limit)
            if args.only in {"all", "states"}:
                detail["states"] = run_state_transitions(conn, run_uuid, args.limit, args.stress)
            if args.only in {"all", "stylometry"}:
                detail["stylometry"] = run_stylometry(conn, run_uuid, args.limit)
            if args.only in {"all", "mahalanobis"}:
                detail["mahalanobis"] = run_mahalanobis(conn, run_uuid, args.limit, args.outliers)
        except Exception as exc:
            status = "failed"
            detail["error"] = str(exc)
            finish_run(conn, run_uuid, status, {**detail, "elapsed_seconds": round(time.time() - started, 3)})
            emit_event(run_uuid, "hard_math", status, detail)
            raise
        detail["elapsed_seconds"] = round(time.time() - started, 3)
        detail["llm_calls"] = 0
        finish_run(conn, run_uuid, status, detail)
    event_id = emit_event(run_uuid, "hard_math", status, detail)
    return {"ok": status == "succeeded", **detail, "workflow_event": event_id}


def cmd_status(args: argparse.Namespace) -> dict[str, Any]:
    with psycopg.connect(STORAGE_DSN) as conn:
        ensure_schema(conn)
        row = conn.execute(
            """
            SELECT
              (SELECT count(*) FROM lucidota_hardmath.analysis_run),
              (SELECT count(*) FROM lucidota_hardmath.lsm_pair),
              (SELECT count(*) FROM lucidota_hardmath.state_transition),
              (SELECT count(*) FROM lucidota_hardmath.stylometry_attribution),
              (SELECT count(*) FROM lucidota_hardmath.semantic_outlier)
            """
        ).fetchone()
        top_outliers = conn.execute(
            """
            SELECT mahalanobis_distance, title, path, excerpt
            FROM lucidota_hardmath.semantic_outlier
            ORDER BY created_at DESC, z_rank ASC
            LIMIT %s
            """,
            (args.limit,),
        ).fetchall()
        transitions = conn.execute(
            """
            SELECT from_state, to_state, transition_probability, transition_count
            FROM lucidota_hardmath.state_transition
            ORDER BY created_at DESC, transition_probability DESC, transition_count DESC
            LIMIT %s
            """,
            (args.limit,),
        ).fetchall()
    return {
        "ok": True,
        "counts": dict(zip(["runs", "lsm_pairs", "state_transitions", "stylometry_attributions", "semantic_outliers"], row)),
        "top_recent_outliers": [dict(zip(["distance", "title", "path", "excerpt"], r)) for r in top_outliers],
        "top_recent_transitions": [dict(zip(["from_state", "to_state", "probability", "count"], r)) for r in transitions],
    }


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="lucidota-hard-truth-math")
    ap.add_argument("--json", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--only", choices=["all", "lsm", "states", "stylometry", "mahalanobis"], default="all")
    p.add_argument("--limit", type=int, default=50000)
    p.add_argument("--outliers", type=int, default=100)
    p.add_argument("--stress", type=float, default=0.0)
    p.set_defaults(func=cmd_run)
    p = sub.add_parser("status")
    p.add_argument("--limit", type=int, default=10)
    p.set_defaults(func=cmd_status)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    result = args.func(args)
    print(jdump(result) if args.json else json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True, default=str))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
