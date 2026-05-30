# DARWIN HAMMER — match 36, survivor 3
# gen: 2
# parent_a: ternary_router.py (gen0)
# parent_b: hybrid_minimum_cost_tree_bayes_update_m6_s1.py (gen1)
# born: 2026-05-29T23:23:31Z

from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Compatibility shim for the original FairyFuse backend.
# If the real package is unavailable (e.g. during the smoke test) we fall
# back to lightweight stubs that mimic the required interface.
# ----------------------------------------------------------------------
try:
    from services.fairyfuse.fairyfuse_backend import resident_engine_from_env, route_command  # type: ignore
except Exception:  # pragma: no cover
    class _DummyEngine:
        def status(self) -> Dict[str, Any]:
            return {"state": "dummy", "uptime_s": 0}

    def resident_engine_from_env() -> _DummyEngine:  # noqa: D401
        """Return a dummy engine with a ``status`` method."""
        return _DummyEngine()

    def route_command(text: str, intent: str, context: Dict[str, Any]) -> Any:
        """Return a simple namespace mimicking the real route result."""
        class _Result:
            def to_dict(self) -> Dict[str, Any]:
                return {
                    "text": text[:100],
                    "intent": intent,
                    "context": context,
                    "confidence": random.random(),
                }

        return _Result()


# ----------------------------------------------------------------------
# Core utilities shared by both parent algorithms.
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUNTIME_DIR = ROOT / "04_RUNTIME" / "fairyfuse"
DEFAULT_HEARTBEAT = RUNTIME_DIR / "ternary_router_heartbeat.jsonl"


def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 Zulu format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def emit_json(obj: Any) -> None:
    """Print a JSON object in a deterministic order."""
    print(json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str))


# ----------------------------------------------------------------------
# Minimum‑cost tree with Bayesian edge uncertainty (parent B).
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]


def _euclidean_length(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def hybrid_tree_cost(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    edge_priors: Dict[Edge, float],
    path_weight: float = 0.2,
    likelihood: float = 0.8,
    false_positive: float = 0.1,
) -> float:
    """
    Compute the expected material cost of a tree rooted at ``root``.
    Edge lengths are weighted by their *posterior* probabilities obtained
    via a single Bayesian update step.
    """
    # Build adjacency list.
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    # Breadth‑first traversal to obtain distances from the root.
    dist: Dict[str, float] = {root: 0.0}
    stack = [root]
    while stack:
        cur = stack.pop()
        for nxt in adj[cur]:
            if nxt not in dist:
                dist[nxt] = dist[cur] + _euclidean_length(nodes[cur], nodes[nxt])
                stack.append(nxt)

    # Bayesian‑updated material cost.
    updated_material = 0.0
    for a, b in edges:
        prior = edge_priors[(a, b)]
        marginal = likelihood * prior + false_positive * (1.0 - prior)
        posterior = prior * likelihood / marginal if marginal > 0 else 0.0
        updated_material += _euclidean_length(nodes[a], nodes[b]) * posterior

    return updated_material + path_weight * sum(dist.values())


def bayes_update_edge_priors(
    edge_priors: Dict[Edge, float],
    evidence: Dict[Edge, float],
    likelihood: float = 0.8,
    false_positive: float = 0.1,
) -> Dict[Edge, float]:
    """
    Apply a Bayesian update to each edge prior using per‑edge evidence.
    ``evidence[e]`` is interpreted as the observed frequency of successful
    traversal for edge ``e`` (value in [0,1]).
    """
    updated = {}
    for edge, prior in edge_priors.items():
        eff_likelihood = likelihood * evidence.get(edge, 1.0)
        marginal = eff_likelihood * prior + false_positive * (1.0 - prior)
        posterior = prior * eff_likelihood / marginal if marginal > 0 else 0.0
        updated[edge] = posterior
    return updated


# ----------------------------------------------------------------------
# Hybrid router that scores each candidate engine channel with the tree cost.
# ----------------------------------------------------------------------
ENGINE_CHANNELS = ["cpu_fairyfuse_ternary", "gpu_fairyfuse_binary", "tpu_fairyfuse_quantized"]


def _build_graph() -> Tuple[Dict[str, Point], List[Edge], Dict[Edge, float]]:
    """
    Construct a fully connected graph over ``ENGINE_CHANNELS``.
    Nodes are placed on a unit circle for deterministic distances.
    Edge priors start at 0.5 (maximum uncertainty).
    """
    n = len(ENGINE_CHANNELS)
    angle_step = 2 * math.pi / n
    nodes: Dict[str, Point] = {}
    for i, ch in enumerate(ENGINE_CHANNELS):
        theta = i * angle_step
        nodes[ch] = (math.cos(theta), math.sin(theta))

    edges: List[Edge] = []
    edge_priors: Dict[Edge, float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            a, b = ENGINE_CHANNELS[i], ENGINE_CHANNELS[j]
            edges.append((a, b))
            edge_priors[(a, b)] = 0.5
            edge_priors[(b, a)] = 0.5  # Include reverse edges
    return nodes, edges, edge_priors


def hybrid_route_packet(packet: Dict[str, Any]) -> Dict[str, Any]:
    """
    Route ``packet`` to the engine channel that yields the lowest expected
    hybrid tree cost.  The function:
      1. Extracts ``text`` and ``intent`` (mirroring parent A).
      2. Scores every candidate channel using ``hybrid_tree_cost``.
      3. Selects the best channel, invokes the underlying ``route_command``,
         and augments the result with static metadata.
    """
    text = packet.get("text", "")
    intent = packet.get("intent", "")
    context = packet.get("context", {})

    # Build graph and get initial edge priors
    nodes, edges, edge_priors = _build_graph()

    # Score each channel
    channel_scores = {}
    for channel in ENGINE_CHANNELS:
        score = hybrid_tree_cost(nodes, edges, channel, edge_priors)
        channel_scores[channel] = score

    # Select best channel
    best_channel = min(channel_scores, key=channel_scores.get)

    # Get route command result
    route_result = route_command(text, intent, context)

    # Update edge priors with routing decision evidence
    evidence = {}
    for edge in edges:
        if best_channel in edge:
            evidence[edge] = 1.0  # Successful traversal
        else:
            evidence[edge] = 0.0  # Unsuccessful traversal
    edge_priors = bayes_update_edge_priors(edge_priors, evidence)

    # Augment route result with static metadata
    route_result_dict = route_result.to_dict()
    route_result_dict["engine_channel"] = best_channel
    route_result_dict["outbound_state"] = "routed"

    return route_result_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=json.loads)
    args = parser.parse_args()

    result = hybrid_route_packet(args.packet)
    emit_json(result)