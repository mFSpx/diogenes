#!/usr/bin/env python3
"""Confidence-scored condensation of symbol streams into evidence-linked claims."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import psycopg  # noqa: E402
from ALGOS.bayes_update import bayes_marginal, bayes_update  # noqa: E402
from ALGOS.decision_hygiene import counts, score_features  # noqa: E402
from scripts.spine_common import sha256_json  # noqa: E402

OUT = ROOT / "05_OUTPUTS" / "goals"
GO25 = ("OBJECT", "EVENT", "EDGE")
DEFAULT_SYMBOLS = ["OBJECT", "EVENT", "EDGE", "CLAIM", "EVIDENCE", "TERM"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def storage_db_url(args: argparse.Namespace) -> str:
    return args.storage_database_url or os.environ.get("KORPUS_DATABASE_URL") or os.environ.get("DATABASE_URL") or "postgresql:///lucidota_storage"


def state_db_url(args: argparse.Namespace) -> str:
    return args.state_database_url or os.environ.get("LUCIDOTA_ABSURD_STATE_DSN") or os.environ.get("DATABASE_URL") or "postgresql:///lucidota_state"


def clean_symbols(raw: list[str] | None) -> list[str]:
    items = [s.strip().upper() for s in (raw or []) if s and s.strip()]
    return items or list(DEFAULT_SYMBOLS)


def fetch_db_evidence(storage_conn: psycopg.Connection[Any], state_conn: psycopg.Connection[Any], *, term_limit: int) -> dict[str, Any]:
    evidence: dict[str, Any] = {"active_terms": [], "runtime_facts": [], "graph_counts": {}}
    try:
        evidence["active_terms"] = [r[0] for r in storage_conn.execute(
            "SELECT term FROM lucidota_go.term_registry WHERE status='active' ORDER BY term_number NULLS LAST, term LIMIT %s",
            (term_limit,),
        ).fetchall()]
    except Exception as exc:
        evidence["active_terms_error"] = type(exc).__name__
    try:
        rows = state_conn.execute(
            "SELECT fact_key, fact_value FROM lucidota_control.runtime_status_fact WHERE subsystem='system' ORDER BY fact_key"
        ).fetchall()
        evidence["runtime_facts"] = [{"fact_key": r[0], "fact_value": r[1]} for r in rows]
    except Exception as exc:
        evidence["runtime_facts_error"] = type(exc).__name__
    for key, sql in {
        "graph_item_count": "SELECT count(*)::bigint FROM lucidota_go.graph_item",
        "graph_edge_count": "SELECT count(*)::bigint FROM lucidota_go.graph_edge",
        "graph_journal_count": "SELECT count(*)::bigint FROM lucidota_go.graph_journal",
    }.items():
        try:
            evidence["graph_counts"][key] = int(state_conn.execute(sql).fetchone()[0])
        except Exception as exc:
            try:
                state_conn.rollback()
            except Exception:
                pass
            evidence["graph_counts"][key] = None
            evidence.setdefault("graph_count_errors", {})[key] = type(exc).__name__
    return evidence


def pair_claim(a: str, b: str, *, pair_count: int, evidence_hits: int, evidence_text: str) -> dict[str, Any]:
    prior = min(0.95, 0.30 + 0.10 * min(5, evidence_hits))
    likelihood = min(0.95, 0.55 + 0.10 * min(4, pair_count))
    false_positive = max(0.02, 0.35 - 0.05 * min(4, evidence_hits))
    marginal = bayes_marginal(prior, likelihood, false_positive)
    posterior = bayes_update(prior, likelihood, marginal)
    confidence = max(0, min(10000, int(round(posterior * 10000))))
    status = "backed" if confidence >= 8000 and evidence_hits > 0 else ("candidate" if confidence >= 5000 else "hypothesis")
    return {
        "pair": [a, b],
        "claim": f"{a} -> {b}",
        "confidence_bps": confidence,
        "status": status,
        "mutable": status != "backed",
        "evidence_hits": evidence_hits,
        "pair_count": pair_count,
        "evidence_anchor": sha256_json({"pair": [a, b], "evidence": evidence_text[:3000]})[:24],
        "evidence_refs": ["db:lucidota_go.term_registry", "db:lucidota_control.runtime_status_fact"],
    }


def compare_with_baseline(current: list[dict[str, Any]], baseline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    base = {tuple(item["pair"]): item for item in baseline}
    out = []
    for item in current:
        prev = base.get(tuple(item["pair"]))
        delta = item["confidence_bps"] - int(prev["confidence_bps"]) if prev else item["confidence_bps"]
        out.append({**item, "confidence_delta_bps": delta, "baseline_present": bool(prev)})
    return out


def build_report(*, symbols: list[str], storage_database_url: str | None, state_database_url: str | None, term_limit: int, baseline_report: str | None, objective: str) -> dict[str, Any]:
    symbol_stream = clean_symbols(symbols)
    with psycopg.connect(storage_database_url or os.environ.get("KORPUS_DATABASE_URL") or os.environ.get("DATABASE_URL") or "postgresql:///lucidota_storage") as storage_conn:
        with psycopg.connect(state_database_url or os.environ.get("LUCIDOTA_ABSURD_STATE_DSN") or os.environ.get("DATABASE_URL") or "postgresql:///lucidota_state") as state_conn:
            evidence = fetch_db_evidence(storage_conn, state_conn, term_limit=term_limit)
    evidence_terms = [str(t).upper() for t in evidence.get("active_terms", []) if t]
    evidence_text = json.dumps(evidence, sort_keys=True, default=str)
    stream = symbol_stream + [s for s in evidence_terms if s not in symbol_stream]
    pair_counts = Counter(tuple(p) for p in pairwise(stream))
    claims = []
    for a, b in pair_counts:
        evidence_hits = int(a in evidence_terms) + int(b in evidence_terms) + sum(1 for row in evidence.get("runtime_facts", []) if a in json.dumps(row, sort_keys=True, default=str).upper()) + sum(1 for row in evidence.get("runtime_facts", []) if b in json.dumps(row, sort_keys=True, default=str).upper())
        claims.append(pair_claim(a, b, pair_count=pair_counts[(a, b)], evidence_hits=evidence_hits, evidence_text=evidence_text))
    claims = sorted(claims, key=lambda c: (-c["confidence_bps"], c["pair"][0], c["pair"][1]))
    if baseline_report:
        baseline = json.loads(Path(baseline_report).read_text(encoding="utf-8")) if Path(baseline_report).exists() else {}
        baseline_claims = baseline.get("claims") or []
        claims = compare_with_baseline(claims, baseline_claims)
    features = counts(" ".join(symbol_stream) + " " + evidence_text[:5000])
    hygiene_score, hygiene_label = score_features(features)
    backed = [c for c in claims if c["status"] == "backed"]
    morphing = [c for c in claims if c["mutable"]]
    language = " / ".join(symbol_stream[:8])
    return {
        "schema": "lucidota.graph_symbol_condensation.v1",
        "generated_at": now(),
        "objective": objective or "Condense graph and Postgres evidence into confidence-scored symbolic claims.",
        "intent": "symbol_condensation",
        "ontology_mode": "GO25_STRICT",
        "ontology_terms": list(GO25),
        "symbol_lexicon": symbol_stream,
        "language_seed": language,
        "evidence_refs": ["db:lucidota_go.term_registry", "db:lucidota_control.runtime_status_fact", "db:lucidota_go.graph_item", "db:lucidota_go.graph_edge", "db:lucidota_go.graph_journal"],
        "graph_snapshot": evidence,
        "claims": claims,
        "rule_candidates": [
            {"rule": f"if {a} then {b}", "confidence_bps": c["confidence_bps"], "status": c["status"], "mutable": c["mutable"]}
            for c in claims
            for a, b in [c["pair"]]
        ],
        "comparison_summary": {"backed_claims": len(backed), "morphing_claims": len(morphing), "hygiene_label": hygiene_label, "hygiene_score": hygiene_score},
        "status": "PASS",
        "next_action": "compare against a later seed or dispatch the high-confidence rules to a worker lane",
        "model_calls_performed": False,
        "canonical_graph_writes_performed": False,
    }


def write_report(report: dict[str, Any]) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"graph_symbol_condensation_{stamp()}.json"
    report["receipt_path"] = rel(path)
    report["report_path"] = report["receipt_path"]
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="")
    ap.add_argument("--storage-database-url")
    ap.add_argument("--state-database-url")
    ap.add_argument("--term-limit", type=int, default=24)
    ap.add_argument("--baseline-report")
    ap.add_argument("--objective", default="")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    symbols = [s for s in a.symbols.split(",") if s.strip()] if a.symbols else list(DEFAULT_SYMBOLS)
    report = write_report(build_report(symbols=symbols, storage_database_url=a.storage_database_url, state_database_url=a.state_database_url, term_limit=a.term_limit, baseline_report=a.baseline_report, objective=a.objective))
    print("REPORT_PATH=" + report["report_path"])
    print("GRAPH_SYMBOL_CONDENSATION=PASS")
    print(json.dumps(report, sort_keys=True) if a.json else report["report_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
