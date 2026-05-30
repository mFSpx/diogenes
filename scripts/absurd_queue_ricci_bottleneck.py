#!/usr/bin/env python3
"""Apply Ollivier-Ricci curvature to the ABSURD queue job dependency graph.

mutation class = read_only

Reads job records from lucidota_control.absurd_queue_job and builds an
adjacency graph where nodes are (queue_name, job_kind) pairs and edges come
from:
  - source_queue links in payload_json
  - shared worker_key co-membership

Runs ollivier_ricci() from ALGOS/ and reports bottleneck edges (kappa < 0).
Writes a receipt to 05_OUTPUTS/absurd/ricci_bottleneck_{ts}.json.

No canonical graph tables are read or written.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALGOS = ROOT / "ALGOS"
OUT_DIR = ROOT / "05_OUTPUTS" / "absurd"

# Add ALGOS to path so ollivier_ricci_curvature is importable
if str(ALGOS) not in sys.path:
    sys.path.insert(0, str(ALGOS))

from ollivier_ricci_curvature import ollivier_ricci, bottleneck_edges  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def now_iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def dsn(args):
    return (
        getattr(args, "database_url", None)
        or os.environ.get("LUCIDOTA_GO_STATE_DSN")
        or os.environ.get("ABSURD_SYSTEM_DATABASE_URL")
        or os.environ.get("DATABASE_URL")
        or "postgresql:///lucidota_state"
    )


def redact(url):
    if url.startswith("postgresql:///"):
        return "postgresql:///<db>"
    if "@" in url:
        return "postgresql://<redacted>@" + url.split("@", 1)[1]
    return "<redacted>"


# ---------------------------------------------------------------------------
# Synthetic graph fallback
# ---------------------------------------------------------------------------

SYNTHETIC_JOB_KINDS = [
    ("river_queue",      "river",          "river_worker"),
    ("chrono_queue",     "chrono",         "chrono_worker"),
    ("corpus_queue",     "corpus",         "corpus_worker"),
    ("intake_queue",     "intake",         "intake_worker"),
    ("graph_promo_queue","graph_promotion","graph_promo_worker"),
    ("river_queue",      "river",          "intake_worker"),   # shared worker_key link
    ("intake_queue",     "intake",         "corpus_worker"),   # shared worker_key link
]

SYNTHETIC_SOURCE_LINKS = [
    # (src_queue, src_kind) -> (dst_queue, dst_kind)
    ("intake_queue",     "intake",     "river_queue",       "river"),
    ("river_queue",      "river",      "corpus_queue",      "corpus"),
    ("corpus_queue",     "corpus",     "graph_promo_queue", "graph_promotion"),
    ("graph_promo_queue","graph_promotion", "chrono_queue", "chrono"),
]


def build_synthetic_graph():
    """Return (rows, source_links, note)."""
    rows = [(q, k, w) for q, k, w in SYNTHETIC_JOB_KINDS]
    note = "synthetic_graph_fallback"
    return rows, SYNTHETIC_SOURCE_LINKS, note


# ---------------------------------------------------------------------------
# DB queries
# ---------------------------------------------------------------------------

def query_db(url):
    """Return (rows, source_links, note) from live DB or raise."""
    import psycopg  # noqa: PLC0415 — optional import

    with psycopg.connect(url) as conn:
        with conn.cursor() as cur:
            # Distinct (queue_name, job_kind, worker_key) nodes
            cur.execute(
                "SELECT DISTINCT queue_name, job_kind, worker_key "
                "FROM lucidota_control.absurd_queue_job LIMIT 500"
            )
            rows = cur.fetchall()

            # source_queue links in payload
            cur.execute(
                "SELECT queue_name, job_kind, "
                "payload_json->>'source_queue' AS src "
                "FROM lucidota_control.absurd_queue_job "
                "WHERE payload_json ? 'source_queue' LIMIT 500"
            )
            raw_links = cur.fetchall()

    # Build source_queue link tuples: (src_queue, None, dst_queue, dst_kind)
    # We only know dst_queue + dst_kind from the row; src is the value in payload.
    # Represent as (src_queue, src_kind_unknown, dst_queue, dst_kind).
    source_links = []
    for dst_queue, dst_kind, src_queue in raw_links:
        if src_queue:
            source_links.append((src_queue, None, dst_queue, dst_kind))

    return rows, source_links, "live_db"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_adjacency(rows, source_links):
    """Build undirected adjacency dict from job rows and source_queue links.

    Nodes are (queue_name, job_kind) tuples.
    Edges:
      1. source_queue links: payload says this job came from src_queue.
      2. Shared worker_key: two (queue, kind) nodes that share a worker_key
         are connected — they compete for the same worker resource.
    """
    adj = {}

    def add_edge(a, b):
        if a == b:
            return
        adj.setdefault(a, [])
        adj.setdefault(b, [])
        if b not in adj[a]:
            adj[a].append(b)
        if a not in adj[b]:
            adj[b].append(a)

    # Register all nodes
    for queue_name, job_kind, worker_key in rows:
        node = (queue_name, job_kind)
        adj.setdefault(node, [])

    # Edges from source_queue links
    # src_queue may map to multiple (queue, kind) combos; use the first match or
    # create a synthetic node for it.
    known_queues = {}
    for queue_name, job_kind, worker_key in rows:
        known_queues.setdefault(queue_name, []).append((queue_name, job_kind))

    for src_queue, src_kind, dst_queue, dst_kind in source_links:
        dst_node = (dst_queue, dst_kind)
        adj.setdefault(dst_node, [])
        if src_kind is not None:
            src_node = (src_queue, src_kind)
        else:
            # Best-effort: pick first known node for src_queue, else synthetic
            candidates = known_queues.get(src_queue)
            if candidates:
                src_node = candidates[0]
            else:
                src_node = (src_queue, "unknown")
                adj.setdefault(src_node, [])
                known_queues.setdefault(src_queue, []).append(src_node)
        add_edge(src_node, dst_node)

    # Edges from shared worker_key
    worker_map = {}
    for queue_name, job_kind, worker_key in rows:
        if worker_key:
            worker_map.setdefault(worker_key, []).append((queue_name, job_kind))

    for worker_key, nodes in worker_map.items():
        for i, a in enumerate(nodes):
            for b in nodes[i + 1:]:
                add_edge(a, b)

    return adj


def build_adjacency_synthetic(rows, source_links):
    """Same as build_adjacency but rows have no worker_key column."""
    adj = {}

    def add_edge(a, b):
        if a == b:
            return
        adj.setdefault(a, [])
        adj.setdefault(b, [])
        if b not in adj[a]:
            adj[a].append(b)
        if a not in adj[b]:
            adj[b].append(a)

    known_queues = {}
    for queue_name, job_kind, worker_key in rows:
        node = (queue_name, job_kind)
        adj.setdefault(node, [])
        known_queues.setdefault(queue_name, []).append(node)

    for src_queue, src_kind, dst_queue, dst_kind in source_links:
        src_node = (src_queue, src_kind) if src_kind else None
        dst_node = (dst_queue, dst_kind)
        adj.setdefault(dst_node, [])
        if src_node:
            adj.setdefault(src_node, [])
            add_edge(src_node, dst_node)

    # shared worker_key edges
    worker_map = {}
    for queue_name, job_kind, worker_key in rows:
        if worker_key:
            worker_map.setdefault(worker_key, []).append((queue_name, job_kind))
    for wk, nodes in worker_map.items():
        for i, a in enumerate(nodes):
            for b in nodes[i + 1:]:
                add_edge(a, b)

    return adj


# ---------------------------------------------------------------------------
# Receipt
# ---------------------------------------------------------------------------

def write_receipt(report, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = stamp()
    path = out / f"ricci_bottleneck_{ts}.json"
    report["receipt_path"] = str(path.relative_to(ROOT))
    path.write_text(json.dumps(report, indent=2, sort_keys=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def run(args):
    execute = getattr(args, "execute", False)
    dry_run = getattr(args, "dry_run", False)
    output_dir = getattr(args, "output_dir", None) or str(OUT_DIR)

    url = dsn(args)
    data_source = "unknown"
    rows = []
    source_links = []
    db_error = None

    # Attempt live DB
    try:
        import psycopg  # noqa: F401
        rows, source_links, data_source = query_db(url)
    except Exception as exc:
        db_error = str(exc)
        print(f"[ricci] DB unavailable ({exc}); using synthetic graph")
        rows, source_links, data_source = build_synthetic_graph()

    # Build adjacency
    adj = build_adjacency_synthetic(rows, source_links) if db_error else build_adjacency(rows, source_links)

    node_count = len(adj)
    edge_pairs = set()
    for node, nbrs in adj.items():
        for nb in nbrs:
            edge_pairs.add((min(node, nb), max(node, nb)))
    edge_count = len(edge_pairs)

    print(f"[ricci] Graph: {node_count} nodes, {edge_count} edges  (source={data_source})")

    curvatures = {}
    bottlenecks = []

    if node_count < 2:
        print("[ricci] Graph too small for curvature computation (need >= 2 nodes with edges)")
    else:
        curvatures_raw = ollivier_ricci(adj, alpha=0.5)
        # Convert tuple keys to string for JSON serialisation
        curvatures = {f"{a} -- {b}": round(k, 6) for (a, b), k in curvatures_raw.items()}
        bottlenecks_raw = bottleneck_edges(curvatures_raw, threshold=0.0)
        bottlenecks = [
            {"edge": f"{a} -- {b}", "kappa": round(k, 6)}
            for a, b, k in bottlenecks_raw
        ]

    # Human-readable summary
    print(f"\n[ricci] Ollivier-Ricci curvature results ({data_source})")
    print(f"  Nodes:      {node_count}")
    print(f"  Edges:      {edge_count}")
    print(f"  Curvatures: {len(curvatures)}")
    print(f"  Bottlenecks (kappa < 0): {len(bottlenecks)}")
    if bottlenecks:
        print("\n  Bottleneck edges (most negative first):")
        for item in bottlenecks:
            print(f"    kappa={item['kappa']:+.4f}  {item['edge']}")
    else:
        print("  No bottleneck edges detected.")

    if not curvatures and node_count >= 2:
        print("  (No edges in graph — add source_queue links or shared worker_key entries)")

    receipt = {
        "script": "scripts/absurd_queue_ricci_bottleneck.py",
        "mutation_class": "read_only",
        "receipt_mode": "ABSURD_POSTGRES_RUNTIME",
        "generated_at": now_iso(),
        "data_source": data_source,
        "db_url_redacted": redact(url),
        "db_error": db_error,
        "dry_run": dry_run,
        "execute": execute,
        "graph": {
            "node_count": node_count,
            "edge_count": edge_count,
            "nodes": [{"queue_name": n[0], "job_kind": n[1]} for n in sorted(adj.keys())],
        },
        "curvatures": curvatures,
        "bottleneck_count": len(bottlenecks),
        "bottlenecks": bottlenecks,
    }

    if dry_run:
        print("\n[ricci] dry-run: receipt not written to disk")
        print(json.dumps(receipt, indent=2))
        return 0

    path = write_receipt(receipt, output_dir)
    print(f"\n[ricci] receipt written -> {path.relative_to(ROOT)}")
    print(f"REPORT_PATH={path.relative_to(ROOT)}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Ollivier-Ricci curvature on the ABSURD queue job dependency graph."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print results but do not write receipt to disk.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Write receipt to 05_OUTPUTS/absurd/. (Default when neither flag set.)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUT_DIR),
        help="Directory to write the receipt JSON (default: 05_OUTPUTS/absurd/).",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        dest="database_url",
        help="Postgres DSN (overrides LUCIDOTA_GO_STATE_DSN env var).",
    )
    args = parser.parse_args()

    # Default behaviour: execute if neither flag given
    if not args.dry_run and not args.execute:
        args.execute = True

    sys.exit(run(args))


if __name__ == "__main__":
    main()
