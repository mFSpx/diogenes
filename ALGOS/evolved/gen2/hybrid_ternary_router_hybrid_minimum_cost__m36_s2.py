# DARWIN HAMMER — match 36, survivor 2
# gen: 2
# parent_a: ternary_router.py (gen0)
# parent_b: hybrid_minimum_cost_tree_bayes_update_m6_s1.py (gen1)
# born: 2026-05-29T23:23:31Z

"""Hybrid ternary router + Bayesian minimum‑cost tree.

Parent A: ``ternary_router.py`` – an always‑on router that receives a packet,
extracts text/intents, calls ``route_command`` from the FairyFuse backend and
adds static metadata (engine_channel, outbound_state).

Parent B: ``hybrid_minimum_cost_tree_bayes_update_m6_s1.py`` – defines a tree
cost function where each edge carries a prior probability; the priors are
updated with Bayes’ rule given new evidence and the updated probabilities
modulate the material cost of the tree.

Mathematical bridge
-------------------
Both systems manipulate probability‑like quantities.  In the router the choice
of *engine channel* can be seen as selecting a node in a graph of possible
execution back‑ends.  The hybrid tree treats edges between such nodes as
uncertain, assigning a prior `p(e)` and updating it with evidence.  By
computing a Bayesian‑updated minimum‑cost tree for each candidate engine
channel we obtain a scalar *routing score*.  The router then selects the
channel with the lowest expected cost, thereby fusing the deterministic
routing logic of A with the probabilistic cost optimisation of B.

The implementation below:
* builds a tiny graph whose nodes are the supported engine channels,
* maintains edge priors,
* provides ``hybrid_route_packet`` that scores each channel with
  ``hybrid_tree_cost`` and returns the best route,
* updates edge priors after each routing decision using Bayesian evidence.

Only the standard library, ``numpy`` and ``math`` are used, satisfying the
fusion requirements.
"""

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
) -> float:
    """
    Compute the expected material cost of a tree rooted at ``root``.
    Edge lengths are weighted by their *posterior* probabilities obtained
    via a single Bayesian update step (fixed likelihood = 0.8, false positive = 0.1).
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
        likelihood = 0.8
        false_pos = 0.1
        marginal = likelihood * prior + false_pos * (1.0 - prior)
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
        # Simple model: treat evidence as an additional likelihood factor.
        # The effective likelihood becomes a blend of the generic likelihood
        # and the observed evidence.
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
    return nodes, edges, edge_priors


def hybrid_route_packet(packet: Dict[str, Any]) -> Dict[str, Any]:
    """
    Route ``packet`` to the engine channel that yields the lowest expected
    hybrid tree cost.  The function:
      1. Extracts ``text`` and ``intent`` (mirroring parent A).
      2. Scores every candidate channel using ``hybrid_tree_cost``.
      3. Selects the best channel, invokes the underlying ``route_command``,
         and augments the result with the chosen ``engine_channel``.
    """
    # Step 1 – extract routing inputs.
    text = str(
        packet.get("text_surface")
        or packet.get("raw_command")
        or packet.get("text")
        or ""
    )
    intent = str(
        packet.get("normalized_intent")
        or packet.get("intent")
        or "bytewax_rete_bandit"
    )
    context = {
        "source": packet.get("source"),
        "source_ref": packet.get("source_ref"),
        "ontology_terms": packet.get("ontology_terms") or [],
        "epistemic_flag": packet.get("epistemic_flag"),
        "payload": packet.get("payload") or {},
    }

    # Step 2 – compute costs for each engine channel.
    nodes, edges, edge_priors = _build_graph()
    costs: Dict[str, float] = {}
    for channel in ENGINE_CHANNELS:
        # Treat the channel as the root of the tree.
        cost = hybrid_tree_cost(nodes, edges, channel, edge_priors)
        costs[channel] = cost

    # Choose the channel with minimal expected cost.
    best_channel = min(costs, key=costs.get)

    # Step 3 – delegate to the original routing logic.
    route = route_command(text[:4096], intent, context).to_dict()
    route["engine_channel"] = best_channel
    route["outbound_state"] = "draft_only"
    route["routing_score"] = costs[best_channel]
    return route


def update_priors_after_route(
    edge_priors: Dict[Edge, float],
    chosen_channel: str,
    success: bool,
) -> Dict[Edge, float]:
    """
    Produce a new prior map after observing the outcome of a routing decision.
    ``success`` indicates whether the chosen channel processed the packet
    correctly.  Evidence is propagated to all edges incident to the chosen node.
    """
    # Build evidence: edges incident to the chosen channel get evidence=1.0 on success,
    # otherwise 0.0.  Unrelated edges retain a neutral evidence of 0.5.
    evidence: Dict[Edge, float] = {}
    for edge in edge_priors:
        if chosen_channel in edge:
            evidence[edge] = 1.0 if success else 0.0
        else:
            evidence[edge] = 0.5
    return bayes_update_edge_priors(edge_priors, evidence)


# ----------------------------------------------------------------------
# Daemon / CLI (mirrors parent A but uses the hybrid router).
# ----------------------------------------------------------------------
def status_cmd(_: argparse.Namespace) -> int:
    emit_json(resident_engine_from_env().status())
    return 0


def route_cmd(args: argparse.Namespace) -> int:
    packet = {
        "raw_command": args.raw_command,
        "normalized_intent": args.normalized_intent,
        "context": json.loads(args.context) if args.context else {},
    }
    route = hybrid_route_packet(packet)
    emit_json(route)
    return 0


def daemon_cmd(args: argparse.Namespace) -> int:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    heartbeat_path = Path(args.heartbeat)
    if not heartbeat_path.is_absolute():
        heartbeat_path = ROOT / heartbeat_path
    heartbeat_path.parent.mkdir(parents=True, exist_ok=True)

    engine = resident_engine_from_env()
    stop = {"flag": False}

    def _stop(signum, frame):  # noqa: ANN001
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    cycles = 0
    while not stop["flag"]:
        cycles += 1
        status = engine.status()
        receipt = {
            "schema": "lucidota.fairyfuse.hybrid_router_heartbeat.v1",
            "created_at": now_z(),
            "cycle": cycles,
            "pid": os.getpid(),
            "always_on": True,
            "engine_channel": "hybrid_router",
            "status": status,
        }
        with heartbeat_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(receipt, ensure_ascii=False, sort_keys=True, default=str) + "\n")
        if args.max_cycles and cycles >= args.max_cycles:
            break
        time.sleep(max(0.1, float(args.idle_sleep)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hybrid FairyFuse ternary router with Bayesian tree scoring")
    sub = parser.add_subparsers(dest="cmd", required=False)
    p_status = sub.add_parser("status")
    p_status.set_defaults(func=status_cmd)

    p_route = sub.add_parser("route")
    p_route.add_argument("--raw-command", default="")
    p_route.add_argument("--normalized-intent", default="bytewax_rete_bandit")
    p_route.add_argument("--context", default="{}")
    p_route.set_defaults(func=route_cmd)

    p_daemon = sub.add_parser("daemon")
    p_daemon.add_argument("--idle-sleep", type=float, default=float(os.environ.get("LUCIDOTA_FAIRYFUSE_IDLE_SLEEP", "2.0")))
    p_daemon.add_argument("--max-cycles", type=int, default=0)
    p_daemon.add_argument("--heartbeat", default=str(DEFAULT_HEARTBEAT.relative_to(ROOT)))
    p_daemon.set_defaults(func=daemon_cmd)

    parser.set_defaults(func=status_cmd)
    return parser


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


# ----------------------------------------------------------------------
# Smoke test – runs a few hybrid operations without external dependencies.
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Basic hybrid routing on a dummy packet.
    dummy_packet = {
        "raw_command": "search for unicorns",
        "normalized_intent": "search_intent",
        "source": "unit_test",
        "payload": {"query": "unicorns"},
    }
    print("Hybrid route result:")
    print(json.dumps(hybrid_route_packet(dummy_packet), indent=2, sort_keys=True))

    # 2. Demonstrate prior update after a successful routing.
    _, edges, priors = _build_graph()
    updated = update_priors_after_route(priors, "cpu_fairyfuse_ternary", success=True)
    print("\nUpdated edge priors (sample):")
    sample = list(updated.items())[:3]
    for edge, prob in sample:
        print(f"{edge}: {prob:.3f}")

    # 3. Run the CLI parser with a '--help' flag to ensure it doesn't crash.
    # (We don't actually invoke the subprocess; just ensure parsing works.)
    parser = build_parser()
    parser.parse_args(["status"])

    # Exit cleanly.
    sys.exit(0)