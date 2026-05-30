# DARWIN HAMMER — match 41, survivor 2
# gen: 2
# parent_a: pheromone.py (gen0)
# parent_b: hybrid_distributed_leader_e_perceptual_dedupe_m16_s1.py (gen1)
# born: 2026-05-29T23:23:29Z

"""Hybrid algorithm integrating Darwinian surface pheromone dynamics (Algorithm A)
and distributed leader election with perceptual hashing (Algorithm B).

Mathematical bridge:
- Algorithm A defines an exponential decay of a scalar signal:
      v(t) = v₀ * 0.5^{Δt / τ}
  where τ = half_life_seconds / 3600 (hours) and Δt is elapsed hours.
- Algorithm B clusters graph nodes by computing a perceptual hash of a
  numeric feature vector and then selecting leaders via a maximal
  independent set.

The hybrid treats each graph node as a “surface” that carries a pheromone
signal.  Before hashing, the node’s feature vector is re‑scaled by the
current decayed pheromone value, i.e.  v_i ← v_i * v_pheromone(t).  This
injects the exponential decay dynamics directly into the hash‑based
clustering, yielding clusters that respect both similarity (hash) and
temporal relevance (pheromone decay)."""

import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Hashable, Set, Dict, List

# ----------------------------------------------------------------------
# Pheromone subsystem (derived from Algorithm A)
# ----------------------------------------------------------------------
Node = Hashable
Graph = Mapping[Node, Set[Node]]

class PheromoneStore:
    """In‑memory store mimicking the surface_pheromone table."""
    def __init__(self) -> None:
        # surface_key → list of records
        self._store: Dict[str, List[Dict]] = {}

    def signal(
        self,
        surface_key: str,
        signal_kind: str,
        signal_value: float,
        half_life_seconds: int,
        detail: str | None = None,
    ) -> str:
        """Insert a new pheromone record and return its UUID‑like id."""
        rec = {
            "pheromone_id": f"{surface_key}:{len(self._store.get(surface_key, []))}",
            "surface_key": surface_key,
            "signal_kind": signal_kind,
            "signal_value": float(signal_value),
            "half_life_seconds": int(half_life_seconds),
            "created_at": datetime.now(timezone.utc),
            "detail": detail or "",
            "active": True,
        }
        self._store.setdefault(surface_key, []).append(rec)
        return rec["pheromone_id"]

    def decay(self, surface_key: str) -> List[Dict]:
        """Apply exponential decay to all active records of a surface."""
        now = datetime.now(timezone.utc)
        updated: List[Dict] = []
        for rec in self._store.get(surface_key, []):
            if not rec["active"]:
                continue
            elapsed_h = max(
                0.0,
                (now - rec["created_at"]).total_seconds() / 3600.0,
            )
            half_h = max(1.0, rec["half_life_seconds"] / 3600.0)
            decayed = rec["signal_value"] * math.pow(0.5, elapsed_h / half_h)
            rec["signal_value"] = decayed
            rec["detail"] += f" | decayed @ {now.isoformat()}"
            updated.append(rec)
        return updated

    def current_strength(self, surface_key: str) -> float:
        """Sum of decayed signal values for a surface (treated as node weight)."""
        self.decay(surface_key)  # ensure up‑to‑date
        return sum(rec["signal_value"] for rec in self._store.get(surface_key, []))

# ----------------------------------------------------------------------
# Perceptual hashing & leader election (derived from Algorithm B)
# ----------------------------------------------------------------------
def compute_phash(values: List[float]) -> int:
    """Perceptual hash: compare each of the first 64 values to the mean."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def broadcast_probability(phases: int, phase: int) -> float:
    """p = 1 / 2^{phases‑phase}, clamped to [0,1]."""
    if phases < 1 or phase < 1:
        raise ValueError("phases and phase must be positive")
    return min(1.0, 1.0 / (2 ** max(0, phases - phase)))


def maximal_independent_set(
    graph: Graph, phases: int = 8, seed: int | str | None = None
) -> Set[Node]:
    """Approximate a maximal independent set via probabilistic broadcasts."""
    rng = random.Random(seed)
    undecided: Set[Node] = set(graph)
    leaders: Set[Node] = set()
    blocked: Set[Node] = set()

    for phase in range(1, phases + 1):
        if not undecided:
            break
        p = broadcast_probability(phases, phase)
        broadcasts = {n for n in undecided if rng.random() < p}
        new_leaders = {
            n for n in broadcasts if not (graph.get(n, set()) & broadcasts)
        }
        leaders.update(new_leaders)
        newly_blocked = set().union(
            *(graph.get(n, set()) for n in new_leaders), new_leaders
        )
        blocked.update(newly_blocked)
        undecided.difference_update(blocked)

    # deterministic sweep for any leftovers
    for n in sorted(undecided, key=str):
        if not (graph.get(n, set()) & leaders):
            leaders.add(n)
    return leaders


def cluster_by_phash(hashes: Dict[Node, int], max_distance: int = 4) -> List[List[Node]]:
    """Group nodes whose hashes are within max_distance Hamming distance."""
    clusters: List[List[Node]] = []
    for node, h in hashes.items():
        placed = False
        for c in clusters:
            if hamming_distance(h, hashes[c[0]]) <= max_distance:
                c.append(node)
                placed = True
                break
        if not placed:
            clusters.append([node])
    return clusters


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def hybrid_node_weights(
    graph: Graph,
    node_values: Dict[Node, List[float]],
    pheromone_store: PheromoneStore,
    surface_key_prefix: str = "node",
) -> Dict[Node, List[float]]:
    """
    Scale each node's feature vector by its current pheromone strength.
    The surface key for a node is ``f"{surface_key_prefix}:{node}"``.
    """
    weighted: Dict[Node, List[float]] = {}
    for n, vals in node_values.items():
        key = f"{surface_key_prefix}:{n}"
        strength = pheromone_store.current_strength(key)
        # Avoid zero‑weight collapse; fall back to 1.0 if no pheromone yet.
        factor = strength if strength > 0 else 1.0
        weighted[n] = [v * factor for v in vals]
    return weighted


def hybrid_clustering(
    graph: Graph,
    node_values: Dict[Node, List[float]],
    pheromone_store: PheromoneStore,
    phases: int = 8,
    seed: int | str | None = None,
) -> List[List[Node]]:
    """
    1. Select leaders via maximal independent set.
    2. Weight leaders' feature vectors by decayed pheromone strength.
    3. Compute perceptual hashes of the weighted vectors.
    4. Cluster leaders by hash proximity.
    """
    leaders = maximal_independent_set(graph, phases, seed)
    # Restrict to leaders for hashing
    leader_vals = {n: node_values[n] for n in leaders}
    weighted_vals = hybrid_node_weights(
        graph, leader_vals, pheromone_store, surface_key_prefix="node"
    )
    hashes = {n: compute_phash(weighted_vals[n]) for n in leaders}
    return cluster_by_phash(hashes)


def simulate_pheromone_signal(
    pheromone_store: PheromoneStore,
    node_ids: List[Node],
    base_value: float = 1.0,
    half_life_seconds: int = 86400,
) -> None:
    """
    Emit a single pheromone signal for each node.  The surface key is
    ``node:{node_id}``.  This helper populates the store so that subsequent
    hybrid clustering has non‑trivial weights.
    """
    for nid in node_ids:
        key = f"node:{nid}"
        pheromone_store.signal(
            surface_key=key,
            signal_kind="used",
            signal_value=base_value,
            half_life_seconds=half_life_seconds,
            detail="initial signal",
        )


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple undirected graph
    graph: Graph = {
        0: {1, 2},
        1: {0, 2},
        2: {0, 1, 3},
        3: {2, 4},
        4: {3},
    }

    # Raw feature vectors per node
    node_vals: Dict[Node, List[float]] = {
        0: [1.0, 2.0, 3.0],
        1: [1.5, 2.5, 3.5],
        2: [2.0, 3.0, 4.0],
        3: [5.0, 5.5, 6.0],
        4: [5.2, 5.7, 6.2],
    }

    # Initialise pheromone subsystem
    store = PheromoneStore()
    simulate_pheromone_signal(store, list(graph.keys()), base_value=2.0, half_life_seconds=43200)

    # Perform hybrid clustering
    clusters = hybrid_clustering(graph, node_vals, store, phases=5, seed=42)

    print("Hybrid clusters (leaders grouped by weighted perceptual hash):")
    for idx, cl in enumerate(clusters, 1):
        print(f"  Cluster {idx}: {cl}")

    # Demonstrate decay over simulated time by manually adjusting timestamps
    # (for the smoke test we simply invoke decay to show it runs)
    for key in list(store._store.keys()):
        store.decay(key)

    # Re‑run clustering after decay – weights will have changed
    clusters_after = hybrid_clustering(graph, node_vals, store, phases=5, seed=42)
    print("\nClusters after one decay step:")
    for idx, cl in enumerate(clusters_after, 1):
        print(f"  Cluster {idx}: {cl}")

    sys.exit(0)