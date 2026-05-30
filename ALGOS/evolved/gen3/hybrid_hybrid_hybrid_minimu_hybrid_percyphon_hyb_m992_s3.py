# DARWIN HAMMER — match 992, survivor 3
# gen: 3
# parent_a: hybrid_hybrid_minimum_cost__hybrid_infotaxis_min_m178_s4.py (gen2)
# parent_b: hybrid_percyphon_hybrid_endpoint_circ_m45_s2.py (gen2)
# born: 2026-05-29T23:32:13Z

import math
import random
import hashlib
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict, Any

import numpy as np

# -------------------------- Types --------------------------

Point = Tuple[float, float]
Edge = Tuple[str, str]

# -------------------------- Bayesian helpers --------------------------

def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """Compute the marginal probability P(evidence)."""
    if not (0.0 <= prior <= 1.0 and 0.0 <= likelihood <= 1.0 and 0.0 <= false_positive <= 1.0):
        raise ValueError("All probabilities must be in [0, 1]")
    return likelihood * prior + false_positive * (1.0 - prior)


def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Return the posterior P(H|E) = P(E|H)P(H)/P(E)."""
    if marginal <= 0.0:
        raise ValueError("Marginal probability must be > 0")
    return prior * likelihood / marginal


def bayes_edge_posterior(
    node: str,
    edge: Edge,
    priors: Dict[str, float],
    likelihoods: Dict[Edge, float],
    false_positive: float = 0.1,
) -> float:
    """Posterior that an edge is useful given the node prior and edge likelihood."""
    prior = priors.get(node, 0.5)
    likelihood = likelihoods.get(edge, 0.5)
    marginal = bayes_marginal(prior, likelihood, false_positive)
    return bayes_update(prior, likelihood, marginal)


# -------------------------- Procedural entity --------------------------

@dataclass(frozen=True)
class ProceduralSlot:
    slot_index: int
    name: str
    alias: str
    persona: str
    uuid: str
    ternary_offset: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "slot_index": self.slot_index,
            "name": self.name,
            "alias": self.alias,
            "persona": self.persona,
            "uuid": self.uuid,
            "ternary_offset": self.ternary_offset,
        }


def _uuid_from_sha256(seed: str) -> str:
    h = hashlib.sha256(seed.encode("utf-8", errors="ignore")).hexdigest()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def _slot_name(seed: str, idx: int) -> Tuple[str, str, str]:
    h = hashlib.sha256(f"{seed}:{idx}".encode()).hexdigest()
    name = f"Villager-{int(h[:6], 16) % 5000:04d}"
    alias = f"Alias-{h[6:10]}"
    persona = ["ledger", "runner", "witness", "archivist", "carrier", "scribe"][
        int(h[10:12], 16) % 6
    ]
    return name, alias, persona


def procedural_entity_generator(
    villagers: Iterable[str] | None = None,
    psyche_wrath_velocity: float = 0.0,
    psyche_forensic_shield_ratio: float = 0.0,
    fluid_slots: int = 88,
    rng_seed: int | None = None,
) -> Dict[str, Any]:
    """
    Produce a deterministic collection of procedural slots.
    The random component (ternary_offset) is seeded for reproducibility.
    """
    rng = random.Random(rng_seed)
    villagers = list(vvillagers or [])
    seed = "|".join(vvillagers[:5000]) if vvillagers else "lucidota-villager-baseline"

    slots: List[ProceduralSlot] = []
    ternary_offs: List[int] = []

    for idx in range(fluid_slots):
        name, alias, persona = _slot_name(seed, idx)
        uuid = _uuid_from_sha256(name)
        ternary_offset = rng.choice([-1, 0, 1])
        slot = ProceduralSlot(idx, name, alias, persona, uuid, ternary_offset)
        slots.append(slot)
        ternary_offs.append(ternary_offset)

    return {"slots": slots, "ternary_offs": ternary_offs}


# -------------------------- Union‑Find for MST --------------------------

class UnionFind:
    def __init__(self, elements: Iterable[str]):
        self.parent = {e: e for e in elements}
        self.rank = {e: 0 for e in elements}

    def find(self, x: str) -> str:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: str, y: str) -> bool:
        xr, yr = self.find(x), self.find(y)
        if xr == yr:
            return False
        if self.rank[xr] < self.rank[yr]:
            self.parent[xr] = yr
        elif self.rank[xr] > self.rank[yr]:
            self.parent[yr] = xr
        else:
            self.parent[yr] = xr
            self.rank[xr] += 1
        return True


# -------------------------- Hybrid tree construction --------------------------

def adaptive_threshold(
    ternary_offs: List[int],
    base: float = 0.5,
    scale: float = 0.3,
) -> float:
    """
    Compute a threshold in [0,1] that adapts to the morphology.
    Higher variance (std) of ternary offsets yields a more permissive threshold.
    """
    if not ternary_offs:
        return base
    std = np.std(ternary_offs)
    # std is at most 1 (since values are -1,0,1); map to [0,scale]
    adj = scale * std
    thresh = base + adj
    return min(1.0, max(0.0, thresh))


def hybrid_tree_construction(
    nodes: List[str],
    edges: List[Edge],
    priors: Dict[str, float],
    likelihoods: Dict[Edge, float],
    edge_costs: Dict[Edge, float],
    entity_generator: Dict[str, Any],
) -> List[Edge]:
    """
    Build a minimum‑cost spanning tree where an edge is eligible only if its
    Bayesian posterior exceeds an adaptive threshold derived from procedural morphology.
    """
    # 1. Compute posterior for every edge
    posteriors: Dict[Edge, float] = {}
    for edge in edges:
        node = edge[0]  # use source node as the conditioning node
        post = bayes_edge_posterior(node, edge, priors, likelihoods)
        posteriors[edge] = post

    # 2. Determine adaptive threshold
    thresh = adaptive_threshold(entity_generator.get("ternary_offs", []))

    # 3. Filter edges by posterior > threshold
    eligible = [e for e in edges if posteriors.get(e, 0.0) > thresh]

    # 4. Sort eligible edges by cost (Kruskal)
    eligible.sort(key=lambda e: edge_costs.get(e, math.inf))

    uf = UnionFind(nodes)
    tree: List[Edge] = []

    for e in eligible:
        if uf.union(e[0], e[1]):
            tree.append(e)
            if len(tree) == len(nodes) - 1:
                break

    return tree


# -------------------------- Example usage --------------------------

def main() -> None:
    nodes = ["A", "B", "C", "D"]
    edges = [
        ("A", "B"),
        ("A", "C"),
        ("B", "C"),
        ("B", "D"),
        ("C", "D"),
    ]

    # Example priors per node
    priors = {"A": 0.6, "B": 0.7, "C": 0.8, "D": 0.55}

    # Likelihood per directed edge (could be asymmetric)
    likelihoods = {
        ("A", "B"): 0.9,
        ("A", "C"): 0.4,
        ("B", "C"): 0.8,
        ("B", "D"): 0.6,
        ("C", "D"): 0.7,
    }

    # Simple cost model (lower is better)
    edge_costs = {
        ("A", "B"): 3.0,
        ("A", "C"): 2.5,
        ("B", "C"): 1.2,
        ("B", "D"): 4.1,
        ("C", "D"): 2.0,
    }

    # Deterministic procedural generation
    entity_generator = procedural_entity_generator(
        villagers=["alice", "bob", "carol"],
        fluid_slots=20,
        rng_seed=42,
    )

    tree = hybrid_tree_construction(
        nodes,
        edges,
        priors,
        likelihoods,
        edge_costs,
        entity_generator,
    )

    print("Selected edges in hybrid minimum‑cost tree:")
    for e in tree:
        print(f"  {e}  cost={edge_costs.get(e)}  posterior={bayes_edge_posterior(e[0], e, priors, likelihoods):.3f}")


if __name__ == "__main__":
    main()