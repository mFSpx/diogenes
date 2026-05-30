"""
ALGOS/mistletoe_graph.py  -- WO-6 MISTLETOE REVIVAL

READ-ONLY graph scoring primitives ported from KRAMPUSCHEWING
MISTLETOE (pagerank.py + centrality.py).  Zero DB writes.
Zero canonical mutation.  mutation_class = read_only.

Source provenance:
  KRAMPUSCHEWING/Lucidota/Lucidota/PROJECTS/KRAMPUS_EXPRESS/MISTLETOE/pagerank.py
  KRAMPUSCHEWING/Lucidota/Lucidota/PROJECTS/KRAMPUS_EXPRESS/MISTLETOE/centrality.py

Algorithms (all pure stdlib, no networkx):
  - pagerank()          -- iterative PageRank, damping 0.85
  - degree_centrality() -- raw undirected degree per node
  - connected_components() -- BFS union-find component labelling

Edge sources (in priority order):
  1. lucidota_go.graph_edge  (canonical, if rows exist)
  2. lucidota_go.staging_packet proposed_term co-occurrence per source_id
     (used when canonical graph is sparse, i.e. < 2 edges)

Output shape fed to run_mistletoe_scores():
  {
    "generated_at": "<ISO8601>",
    "source": "graph_edge | staging_cooccurrence",
    "node_count": N,
    "edge_count": M,
    "pagerank": {node_id: float, ...},
    "degree_centrality": {node_id: int, ...},
    "connected_components": [["node", ...], ...],
    "top10_pagerank": [{node, score}, ...],
    "top10_degree": [{node, degree}, ...],
  }
"""

from __future__ import annotations

import os
import json
import datetime
from collections import defaultdict, deque
from typing import Any

# ---------------------------------------------------------------------------
# Pure-stdlib graph algorithms
# ---------------------------------------------------------------------------

AdjList = dict[str, list[str]]


def _normalize(scores: dict[str, float]) -> dict[str, float]:
    vals = [v for v in scores.values() if v > 0]
    if not vals:
        return {k: 0.0 for k in scores}
    mx = max(vals)
    if mx == 0:
        return {k: 0.0 for k in scores}
    return {k: v / mx for k, v in scores.items()}


def pagerank(
    nodes: set[str],
    adj: AdjList,
    damping: float = 0.85,
    iterations: int = 100,
    tolerance: float = 1e-6,
) -> dict[str, float]:
    """
    Iterative PageRank on an undirected adjacency list.
    Handles dangling nodes (zero out-degree) via dangling mass redistribution.
    Returns dict node_id -> rank, sorted descending.  Scores sum to ~1.0.
    """
    n = len(nodes)
    if n == 0:
        return {}
    rank: dict[str, float] = {nid: 1.0 / n for nid in nodes}

    for _ in range(iterations):
        dangling_sum = sum(rank[nid] for nid in nodes if not adj.get(nid))
        new_rank: dict[str, float] = {}
        for nid in nodes:
            incoming = 0.0
            for nbr in adj.get(nid, []):
                out_deg = len(adj.get(nbr, []))
                if out_deg > 0:
                    incoming += rank[nbr] / out_deg
            new_rank[nid] = (
                (1.0 - damping) / n
                + damping * (incoming + dangling_sum / n)
            )
        diff = sum(abs(new_rank[nid] - rank[nid]) for nid in nodes)
        rank = new_rank
        if diff < tolerance:
            break

    return dict(sorted(rank.items(), key=lambda x: -x[1]))


def degree_centrality(
    nodes: set[str],
    adj: AdjList,
) -> dict[str, int]:
    """Raw undirected degree for each node, sorted descending."""
    degrees = {nid: len(adj.get(nid, [])) for nid in nodes}
    return dict(sorted(degrees.items(), key=lambda x: -x[1]))


def connected_components(
    nodes: set[str],
    adj: AdjList,
) -> list[list[str]]:
    """BFS connected-component labelling.  Returns list of component lists."""
    visited: set[str] = set()
    components: list[list[str]] = []
    for start in nodes:
        if start in visited:
            continue
        component: list[str] = []
        queue: deque[str] = deque([start])
        visited.add(start)
        while queue:
            v = queue.popleft()
            component.append(v)
            for nbr in adj.get(v, []):
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append(nbr)
        components.append(sorted(component))
    return sorted(components, key=lambda c: -len(c))


# ---------------------------------------------------------------------------
# Data loaders (DB reads, no writes)
# ---------------------------------------------------------------------------

_DSN_STORAGE = os.environ.get("LUCIDOTA_GO_STORAGE_DSN", "postgresql:///lucidota_storage")


def _psql_csv(query: str) -> list[list[str]]:
    """
    Execute a read-only query via psql subprocess, return rows as list of lists.
    Uses --csv mode for clean field splitting; header row is skipped.
    """
    import subprocess
    import csv
    import io

    result = subprocess.run(
        ["psql", _DSN_STORAGE, "--csv", "-c", query],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"psql error: {result.stderr.strip()}")
    reader = csv.reader(io.StringIO(result.stdout))
    rows = list(reader)
    # First row is header — skip it
    return rows[1:] if rows else []


def _load_from_graph_edge() -> tuple[set[str], AdjList, int]:
    """Load canonical edges from lucidota_go.graph_edge."""
    rows = _psql_csv("SELECT source_uuid::text, target_uuid::text FROM lucidota_go.graph_edge")

    nodes: set[str] = set()
    adj: AdjList = defaultdict(list)
    for row in rows:
        if len(row) < 2:
            continue
        src, tgt = row[0], row[1]
        nodes.add(src)
        nodes.add(tgt)
        adj[src].append(tgt)
        adj[tgt].append(src)
    return nodes, dict(adj), len(rows)


def _load_from_staging_cooccurrence() -> tuple[set[str], AdjList, int]:
    """
    Build an edge list from proposed_term co-occurrence within the same source_id.
    Nodes = distinct proposed_term values.
    Edge = two terms appear in the same source_id (undirected, deduped).
    """
    rows = _psql_csv(
        "SELECT source_id, proposed_term FROM lucidota_go.staging_packet "
        "WHERE proposed_term IS NOT NULL ORDER BY source_id, proposed_term"
    )

    # Group terms by source_id
    by_source: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if len(row) < 2:
            continue
        by_source[row[0]].append(row[1])

    nodes: set[str] = set()
    edge_set: set[tuple[str, str]] = set()
    for terms in by_source.values():
        unique_terms = sorted(set(terms))
        for i, t1 in enumerate(unique_terms):
            nodes.add(t1)
            for t2 in unique_terms[i + 1:]:
                nodes.add(t2)
                key = (min(t1, t2), max(t1, t2))
                edge_set.add(key)

    adj: AdjList = defaultdict(list)
    for src, tgt in edge_set:
        adj[src].append(tgt)
        adj[tgt].append(src)

    return nodes, dict(adj), len(edge_set)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_mistletoe_scores() -> dict[str, Any]:
    """
    Compute PageRank, degree centrality, and connected components on live data.
    Falls back to staging_packet co-occurrence if canonical graph has < 2 edges.
    Returns the score dict AND writes it to 05_OUTPUTS/runtime/mistletoe_scores_<ts>.json.
    mutation_class = read_only (no DB writes, no canonical mutation).
    """
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # Try canonical graph first
    nodes, adj, edge_count = _load_from_graph_edge()
    if edge_count < 2:
        nodes, adj, edge_count = _load_from_staging_cooccurrence()
        source_label = "staging_cooccurrence"
    else:
        source_label = "graph_edge"

    pr = pagerank(nodes, adj)
    dc = degree_centrality(nodes, adj)
    cc = connected_components(nodes, adj)

    top10_pr = [{"node": n, "score": round(s, 8)} for n, s in list(pr.items())[:10]]
    top10_dc = [{"node": n, "degree": d} for n, d in list(dc.items())[:10]]

    result: dict[str, Any] = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source": source_label,
        "node_count": len(nodes),
        "edge_count": edge_count,
        "component_count": len(cc),
        "pagerank": pr,
        "degree_centrality": dc,
        "connected_components": cc,
        "top10_pagerank": top10_pr,
        "top10_degree": top10_dc,
    }

    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "05_OUTPUTS", "runtime"
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"mistletoe_scores_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    return result, out_path


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result, out_path = run_mistletoe_scores()
    print(f"source        : {result['source']}")
    print(f"nodes         : {result['node_count']}")
    print(f"edges         : {result['edge_count']}")
    print(f"components    : {result['component_count']}")
    print(f"top10 pagerank: {result['top10_pagerank']}")
    print(f"top10 degree  : {result['top10_degree']}")
    print(f"output        : {out_path}")
