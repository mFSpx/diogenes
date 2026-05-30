#!/usr/bin/env python3
"""Sheaf cohomology consistency check over staged graph-promotion candidates.

mutation_class: read_only

Applies cellular sheaf cohomology (from ALGOS/sheaf_cohomology.py) to the
staged graph-promotion candidates in lucidota_storage.  Nodes are candidate
items; edges are claimed relationships between them; section vectors are
mean confidence/evidence score vectors per node.  The consistency residual
flags contradictory promotion packets.

Usage
-----
    python3 scripts/staging_sheaf_consistency.py --dry-run
    python3 scripts/staging_sheaf_consistency.py --execute
    python3 scripts/staging_sheaf_consistency.py --execute --limit 500
    python3 scripts/staging_sheaf_consistency.py --execute --output-dir 05_OUTPUTS/graph
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path bootstrap — add ALGOS/ so sheaf_cohomology imports cleanly
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
ALGOS_DIR = REPO_ROOT / "ALGOS"
if str(ALGOS_DIR) not in sys.path:
    sys.path.insert(0, str(ALGOS_DIR))

import numpy as np
from sheaf_cohomology import Sheaf, identity_sheaf  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_DB = os.environ.get("LUCIDOTA_GO_STORAGE_DSN") or "postgresql:///lucidota_storage"
DEFAULT_OUT_DIR = REPO_ROOT / "05_OUTPUTS" / "graph"
SCHEMA = "lucidota_go"
STAGING_TABLES = [
    "staging_packet",
    "graph_promotion_packet",
]
INCONSISTENCY_TOL = 1e-6


# ---------------------------------------------------------------------------
# Timestamp helper
# ---------------------------------------------------------------------------
def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------
def _connect(dsn: str):
    """Return a psycopg2 connection or raise ImportError/OperationalError."""
    import psycopg2  # type: ignore
    return psycopg2.connect(dsn)


def _discover_tables(conn, schema: str) -> list[str]:
    """Return list of table names in the given schema."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = %s
              AND table_type = 'BASE TABLE'
            ORDER BY table_name
            """,
            (schema,),
        )
        return [row[0] for row in cur.fetchall()]


def _find_staging_tables(available: list[str]) -> list[str]:
    """Return the intersection of known staging table names and available tables."""
    return [t for t in STAGING_TABLES if t in available]


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------
def _extract_staging_packet(conn, limit: int) -> list[dict[str, Any]]:
    """Extract rows from lucidota_go.staging_packet."""
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT
                packet_uuid::text          AS id,
                proposed_term              AS term,
                confidence_bps             AS confidence_bps,
                proposed_edges::text       AS proposed_edges_json,
                status
            FROM {SCHEMA}.staging_packet
            WHERE status IN ('pending', 'needs_repair')
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (limit,),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def _extract_graph_promotion_packet(conn, limit: int) -> list[dict[str, Any]]:
    """Extract rows from lucidota_go.graph_promotion_packet."""
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT
                packet_uuid::text          AS id,
                candidate_kind             AS term,
                candidate_payload::text    AS payload_json,
                promotion_status           AS status,
                authority_class
            FROM {SCHEMA}.graph_promotion_packet
            WHERE promotion_status IN ('candidate', 'defer')
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (limit,),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


# ---------------------------------------------------------------------------
# Sheaf construction
# ---------------------------------------------------------------------------
TERM_TO_INT: dict[str, int] = {}


def _term_vec(term: str | None, confidence_bps: int) -> np.ndarray:
    """Return a 2-D section vector [term_code, normalised_confidence]."""
    if term not in TERM_TO_INT:
        TERM_TO_INT[term] = len(TERM_TO_INT) + 1
    code = float(TERM_TO_INT[term])
    conf = confidence_bps / 150.0  # max valid bps is 150
    return np.array([code, conf], dtype=float)


def _build_sheaf_from_staging(rows: list[dict[str, Any]]) -> tuple[Sheaf, dict]:
    """Convert staging_packet rows into a Sheaf and return analysis metadata."""
    if not rows:
        return None, {"source": "staging_packet", "row_count": 0}

    # Each row is a node.  Proposed_edges links to other packet UUIDs by id.
    # Build adjacency from proposed_edges jsonb array.
    id_set = {r["id"] for r in rows}
    adj: dict[str, list[str]] = {r["id"]: [] for r in rows}
    sections: dict[str, np.ndarray] = {}

    for r in rows:
        conf = r.get("confidence_bps") or 0
        term = r.get("term") or "unknown"
        sections[r["id"]] = _term_vec(term, int(conf))

        try:
            edges_raw = json.loads(r.get("proposed_edges_json") or "[]")
        except (json.JSONDecodeError, TypeError):
            edges_raw = []

        for edge_item in edges_raw:
            if not isinstance(edge_item, dict):
                continue
            target = edge_item.get("target_uuid") or edge_item.get("uuid")
            if target and target in id_set:
                adj[r["id"]].append(str(target))

    # Build identity sheaf (dim=2 section vectors)
    dim = 2
    sh = identity_sheaf(adj, dim)
    for node_id, vec in sections.items():
        sh.set_section(node_id, vec)

    meta = {
        "source": "staging_packet",
        "row_count": len(rows),
        "node_count": len(adj),
        "edge_count": len(sh.edges),
    }
    return sh, meta


def _build_sheaf_from_promotion(rows: list[dict[str, Any]]) -> tuple[Sheaf, dict]:
    """Convert graph_promotion_packet rows into a Sheaf."""
    if not rows:
        return None, {"source": "graph_promotion_packet", "row_count": 0}

    id_set = {r["id"] for r in rows}
    adj: dict[str, list[str]] = {r["id"]: [] for r in rows}
    sections: dict[str, np.ndarray] = {}

    authority_rank = {
        "raw_evidence": 1,
        "operator_authored_assertion": 2,
        "operator_defined_label": 3,
        "deterministic_metric": 4,
        "statistical_finding": 5,
        "model_computed_finding": 6,
        "stream_ml_finding": 7,
        "graph_inferred_relation": 8,
        "operator_confirmed_finding": 9,
        "canonical_doctrine": 10,
        "external_action_authorized": 11,
    }

    for r in rows:
        term = r.get("term") or "other"
        auth = r.get("authority_class") or "raw_evidence"
        auth_score = authority_rank.get(auth, 0) / 11.0
        if term not in TERM_TO_INT:
            TERM_TO_INT[term] = len(TERM_TO_INT) + 1
        code = float(TERM_TO_INT[term])
        sections[r["id"]] = np.array([code, auth_score], dtype=float)

        try:
            payload = json.loads(r.get("payload_json") or "{}")
        except (json.JSONDecodeError, TypeError):
            payload = {}

        # Look for edge references inside the candidate_payload
        for key in ("target_uuid", "source_uuid", "related_uuid"):
            target = payload.get(key)
            if target and str(target) in id_set:
                adj[r["id"]].append(str(target))

    dim = 2
    sh = identity_sheaf(adj, dim)
    for node_id, vec in sections.items():
        sh.set_section(node_id, vec)

    meta = {
        "source": "graph_promotion_packet",
        "row_count": len(rows),
        "node_count": len(adj),
        "edge_count": len(sh.edges),
    }
    return sh, meta


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def _analyse(sh: Sheaf, meta: dict) -> dict:
    """Run sheaf analysis and return a result dict."""
    if sh is None:
        return {**meta, "status": "no_data", "global_inconsistency": None,
                "inconsistent_edges": [], "consistent": True}

    global_incon = sh.global_inconsistency()
    bad_edges = sh.inconsistent_edges(tol=INCONSISTENCY_TOL)

    return {
        **meta,
        "status": "ok",
        "global_inconsistency": global_incon,
        "inconsistent_edge_count": len(bad_edges),
        "inconsistent_edges": [
            {"u": u, "v": v, "residual_norm": norm}
            for u, v, norm in bad_edges[:50]  # cap report at 50
        ],
        "consistent": len(bad_edges) == 0,
    }


# ---------------------------------------------------------------------------
# Receipt writer
# ---------------------------------------------------------------------------
def _write_receipt(report: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = report.get("generated_at", _ts()).replace(":", "").replace("-", "").replace("+00:00", "Z")
    fname = out_dir / f"sheaf_consistency_{ts}.json"
    fname.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return fname


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply sheaf cohomology to staged graph-promotion candidates."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true",
                      help="Print what would be done without writing output.")
    mode.add_argument("--execute", action="store_true",
                      help="Run analysis and write receipt.")
    parser.add_argument("--limit", type=int, default=2000,
                        help="Maximum rows per staging table (default 2000).")
    parser.add_argument("--output-dir", type=str,
                        default=str(DEFAULT_OUT_DIR),
                        help="Directory to write receipt JSON.")
    parser.add_argument("--db", type=str, default=DEFAULT_DB,
                        help="Postgres DSN for lucidota_storage.")
    args = parser.parse_args()

    if not args.dry_run and not args.execute:
        parser.print_help()
        return 1

    out_dir = Path(args.output_dir)
    ts_now = _ts()
    report: dict[str, Any] = {
        "script": "scripts/staging_sheaf_consistency.py",
        "mutation_class": "read_only",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "mode": "dry_run" if args.dry_run else "execute",
        "limit": args.limit,
        "db_dsn_used": args.db,
        "analyses": [],
        "receipt_scope": "LOCAL_FILE_PRODUCT",
    }

    # --- DB connection ---
    conn = None
    try:
        conn = _connect(args.db)
    except ImportError as exc:
        msg = f"psycopg2 not available: {exc}"
        print(f"[WARN] {msg}", file=sys.stderr)
        report["db_status"] = "unavailable"
        report["db_error"] = msg
    except Exception as exc:  # noqa: BLE001
        msg = f"DB connection failed: {exc}"
        print(f"[WARN] {msg}", file=sys.stderr)
        report["db_status"] = "unavailable"
        report["db_error"] = msg

    if conn is None:
        report["analyses"] = [{"status": "db_unavailable", "consistent": None}]
        report["consistent"] = None
    else:
        # Discover tables
        try:
            available = _discover_tables(conn, SCHEMA)
        except Exception as exc:  # noqa: BLE001
            available = []
            report["schema_discovery_error"] = str(exc)

        report["schema_discovery"] = {
            "schema": SCHEMA,
            "available_table_count": len(available),
            "available_tables_sample": available[:40],
        }

        found = _find_staging_tables(available)
        report["staging_tables_found"] = found

        analyses = []

        # staging_packet
        if "staging_packet" in found:
            try:
                rows = _extract_staging_packet(conn, args.limit)
                sh, meta = _build_sheaf_from_staging(rows)
                analyses.append(_analyse(sh, meta))
            except Exception as exc:  # noqa: BLE001
                analyses.append({"source": "staging_packet", "status": "error", "error": str(exc)})
        else:
            analyses.append({"source": "staging_packet", "status": "table_not_found"})

        # graph_promotion_packet
        if "graph_promotion_packet" in found:
            try:
                rows = _extract_graph_promotion_packet(conn, args.limit)
                sh, meta = _build_sheaf_from_promotion(rows)
                analyses.append(_analyse(sh, meta))
            except Exception as exc:  # noqa: BLE001
                analyses.append({"source": "graph_promotion_packet", "status": "error", "error": str(exc)})
        else:
            analyses.append({"source": "graph_promotion_packet", "status": "table_not_found"})

        conn.close()
        report["analyses"] = analyses

        # Roll-up consistency
        consistent_vals = [a.get("consistent") for a in analyses if a.get("consistent") is not None]
        if consistent_vals:
            report["consistent"] = all(consistent_vals)
        else:
            report["consistent"] = None

    # --- Print summary ---
    print(json.dumps({
        "mode": report["mode"],
        "db_status": report.get("db_status", "connected"),
        "staging_tables_found": report.get("staging_tables_found", []),
        "analyses_count": len(report["analyses"]),
        "consistent": report.get("consistent"),
        "analyses_summary": [
            {k: v for k, v in a.items() if k not in ("inconsistent_edges",)}
            for a in report["analyses"]
        ],
    }, indent=2))

    # --- Write receipt ---
    if args.execute:
        out_path = _write_receipt(report, out_dir)
        print(f"\nReceipt written: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
