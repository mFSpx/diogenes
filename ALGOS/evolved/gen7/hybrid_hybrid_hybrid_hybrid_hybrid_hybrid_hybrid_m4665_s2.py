# DARWIN HAMMER — match 4665, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_minimum_cost__m539_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1887_s0.py (gen6)
# born: 2026-05-29T23:57:27Z

"""
Hybrid Algorithm: Fusion of LSM‑Bayesian Topology (Parent A) with Sheaf‑Associative Memory
=====================================================================================

Parent A contributes:
    * LSM vectors for each model tier.
    * Bayesian marginal and update equations.
    * Reconstruction‑risk scoring.

Parent B contributes:
    * A cellular sheaf defined on a directed graph with restriction maps.
    * A dense associative memory (DAM) whose energy guides inference.
    * A regret‑weighted learning‑rate strategy.

Mathematical Bridge
-------------------
For every directed edge ``(u, v)`` of the sheaf we construct a *hybrid weight*  

``w_uv = bayes_marginal(prior_uv, likelihood_uv, false_pos_uv) *
        compute_lsm_similarity(lsm_u, lsm_v)``  

where ``lsm_u`` and ``lsm_v`` are the LSM vectors of the model tiers attached to
nodes ``u`` and ``v``.  This scalar modulates the restriction maps of the sheaf,
thereby allowing the sheaf’s section consistency to be influenced by both
probabilistic relevance (Bayesian term) and linguistic similarity (LSM term).

The hybrid weight also enters the DAM energy term, scaling the contribution of
each edge to the total energy.  The regret‑weighted learning rate adapts the
update magnitude of the sheaf sections based on reconstruction‑risk scores.

The resulting system jointly optimises:
    * Sheaf section coherence (via restriction maps),
    * Memory recall quality (via DAM energy),
    * Probabilistic and linguistic relevance (via the hybrid edge weight).

"""

import math
import random
import sys
import pathlib
import numpy as np
from dataclasses import dataclass

# ----------------------------------------------------------------------
# Parent A building blocks
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ModelTier:
    """Lightweight descriptor for a model tier."""
    name: str
    ram_mb: int
    tier: str

    def lsm_vector(self) -> np.ndarray:
        # Simple deterministic LSM vector based on tier name hash
        rng = np.random.default_rng(abs(hash(self.name)) % (2**32))
        return rng.random(3)


def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    """Sigmoid‑shaped risk score."""
    if total_records == 0:
        return 1.0
    return 1.0 / (1.0 + math.exp(-unique_quasi_identifiers / total_records))


def compute_lsm_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Euclidean distance turned into a similarity (smaller distance → higher similarity)."""
    dist = np.linalg.norm(v1 - v2)
    return 1.0 / (1.0 + dist)


def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """Marginal probability for Bayesian update."""
    if not all(0.0 <= x <= 1.0 for x in (prior, likelihood, false_positive)):
        raise ValueError("probabilities must be in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)


def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Posterior probability."""
    if marginal <= 0.0:
        raise ValueError("marginal must be > 0")
    return (likelihood * prior) / marginal


# ----------------------------------------------------------------------
# Parent B building blocks (Sheaf + Dense Associative Memory)
# ----------------------------------------------------------------------
class Sheaf:
    """
    Cellular sheaf on a directed graph.

    Nodes carry a vector space of dimension given by ``node_dims``.
    Each directed edge ``(u, v)`` carries a linear restriction map
    ``src_map : ℝ^{dim(u)} → ℝ^{dim(e)}`` and ``dst_map : ℝ^{dim(v)} → ℝ^{dim(e)}``.
    A *section* assigns a vector to every node.
    """

    def __init__(self, node_dims: dict, edges: list):
        self.node_dims = node_dims          # {node: dim}
        self.edges = edges                  # list of (u, v)
        self._restrictions = {}             # {(u,v): (src_map, dst_map)}
        self._sections = {}                 # {node: vector}

    def set_restriction(self, edge: tuple, src_map: np.ndarray, dst_map: np.ndarray) -> None:
        u, v = edge
        if src_map.shape[1] != self.node_dims[u]:
            raise ValueError("src_map column dimension must match dim(node u)")
        if dst_map.shape[1] != self.node_dims[v]:
            raise ValueError("dst_map column dimension must match dim(node v)")
        if src_map.shape[0] != dst_map.shape[0]:
            raise ValueError("src_map and dst_map must have the same row dimension")
        self._restrictions[(u, v)] = (np.asarray(src_map, dtype=float),
                                      np.asarray(dst_map, dtype=float))

    def set_section(self, node, value: np.ndarray) -> None:
        if value.shape[0] != self.node_dims[node]:
            raise ValueError("section vector dimension mismatch")
        self._sections[node] = np.asarray(value, dtype=float)

    def get_section(self, node):
        return self._sections.get(node, np.zeros(self.node_dims[node]))

    def restriction_residual(self, edge: tuple) -> float:
        """|| src_map·s_u - dst_map·s_v ||_2  – a measure of inconsistency."""
        src_map, dst_map = self._restrictions[edge]
        u, v = edge
        su = self.get_section(u)
        sv = self.get_section(v)
        return np.linalg.norm(src_map @ su - dst_map @ sv)


class DenseAssociativeMemory:
    """
    Simple dense associative memory with a symmetric weight matrix W.
    Energy:  E(x) = -0.5 * xᵀ W x + bᵀ x
    Retrieval is performed by gradient descent on the energy.
    """

    def __init__(self, dim: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(scale=0.1, size=(dim, dim))
        self.W = (self.W + self.W.T) / 2.0          # symmetrize
        self.b = rng.normal(scale=0.1, size=dim)

    def energy(self, x: np.ndarray) -> float:
        return -0.5 * x @ self.W @ x + self.b @ x

    def retrieve(self, query: np.ndarray, steps: int = 20, lr: float = 0.1) -> np.ndarray:
        """Iterative gradient descent toward a low‑energy state."""
        x = query.astype(float).copy()
        for _ in range(steps):
            grad = -self.W @ x + self.b               # ∇E = -(W x) + b
            x -= lr * grad
        return x


class RegretWeightedStrategy:
    """
    Adjusts a base learning rate according to observed regret.
    Regret = previous_loss - current_loss.
    """

    def __init__(self, base_lr: float = 0.01):
        self.base_lr = base_lr

    def adjusted_lr(self, prev_loss: float, curr_loss: float) -> float:
        regret = prev_loss - curr_loss
        factor = 1.0 + max(regret, 0.0)               # only positive regret inflates lr
        return self.base_lr * factor


# ----------------------------------------------------------------------
# Hybrid constructs
# ----------------------------------------------------------------------
def hybrid_edge_weight(
    tier_u: ModelTier,
    tier_v: ModelTier,
    prior: float,
    likelihood: float,
    false_pos: float,
) -> float:
    """
    Compute the hybrid scalar that will modulate the restriction maps of edge (u,v).

    w_uv = bayes_marginal(prior, likelihood, false_pos) *
           compute_lsm_similarity(lsm_u, lsm_v)
    """
    lsm_u = tier_u.lsm_vector()
    lsm_v = tier_v.lsm_vector()
    bayes = bayes_marginal(prior, likelihood, false_pos)
    sim = compute_lsm_similarity(lsm_u, lsm_v)
    return bayes * sim


def hybrid_energy(
    sheaf: Sheaf,
    dam: DenseAssociativeMemory,
    edge_weights: dict,
    beta: float = 0.5,
) -> float:
    """
    Total energy = β * Σ_edge w_uv * restriction_residual(edge)  +  (1-β) * DAM energy
    The sheaf part penalises inconsistent sections, scaled by hybrid edge weights.
    The DAM part encourages sections that are also low‑energy memory states.
    """
    # Sheaf contribution
    sheaf_term = 0.0
    for edge in sheaf.edges:
        w = edge_weights.get(edge, 1.0)
        resid = sheaf.restriction_residual(edge)
        sheaf_term += w * resid

    # Build a concatenated state vector from all node sections
    concat = np.concatenate([sheaf.get_section(node) for node in sorted(sheaf.node_dims)])
    dam_term = dam.energy(concat)

    return beta * sheaf_term + (1.0 - beta) * dam_term


def hybrid_update(
    sheaf: Sheaf,
    dam: DenseAssociativeMemory,
    edge_weights: dict,
    tiers: dict,
    prior_map: dict,
    likelihood_map: dict,
    false_pos_map: dict,
    regret_strategy: RegretWeightedStrategy,
    prev_loss: float,
) -> float:
    """
    Perform a single hybrid update:
        1. Retrieve a memory vector using DAM (query = current concatenated sections).
        2. For each node, compute a Bayesian posterior based on edge‑wise priors.
        3. Blend the posterior into the node's section using a regret‑weighted step size.
        4. Return the new total loss (energy) after the update.
    """
    # 1. Current concatenated state and retrieval
    current_state = np.concatenate([sheaf.get_section(node) for node in sorted(sheaf.node_dims)])
    retrieved = dam.retrieve(current_state, steps=10, lr=0.05)

    # 2. Update each node section
    offset = 0
    for node in sorted(sheaf.node_dims):
        dim = sheaf.node_dims[node]
        sec = sheaf.get_section(node)
        # Slice the retrieved vector that corresponds to this node
        retrieved_slice = retrieved[offset: offset + dim]
        offset += dim

        # Aggregate Bayesian information from incident edges
        posterior = 0.0
        weight_sum = 0.0
        for edge in sheaf.edges:
            if node in edge:
                other = edge[1] if edge[0] == node else edge[0]
                tier_node = tiers[node]
                tier_other = tiers[other]
                w = hybrid_edge_weight(
                    tier_node,
                    tier_other,
                    prior_map.get(edge, 0.5),
                    likelihood_map.get(edge, 0.5),
                    false_pos_map.get(edge, 0.5),
                )
                # Simple prior = w, likelihood = similarity, false_pos = 1-w
                prior = w
                likelihood = compute_lsm_similarity(tier_node.lsm_vector(), tier_other.lsm_vector())
                false_pos = 1.0 - w
                post = bayes_update(prior, likelihood, bayes_marginal(prior, likelihood, false_pos))
                posterior += post * w
                weight_sum += w

        if weight_sum > 0:
            posterior /= weight_sum
        else:
            posterior = 0.0

        # 3. Regret‑weighted learning rate
        # Estimate a provisional loss (energy) before the step
        tentative_sec = sec + posterior * retrieved_slice
        sheaf.set_section(node, tentative_sec)
        tentative_loss = hybrid_energy(sheaf, dam, edge_weights)
        lr = regret_strategy.adjusted_lr(prev_loss, tentative_loss)

        # Apply the weighted update
        new_sec = sec + lr * (posterior * retrieved_slice - sec)
        sheaf.set_section(node, new_sec)

    # 4. Compute final loss
    final_loss = hybrid_energy(sheaf, dam, edge_weights)
    return final_loss


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny graph with two nodes and one directed edge
    node_dims = {"A": 3, "B": 3}
    edges = [("A", "B")]

    # Instantiate sheaf
    sheaf = Sheaf(node_dims, edges)
    sheaf.set_section("A", np.array([0.2, 0.1, 0.4]))
    sheaf.set_section("B", np.array([0.5, 0.3, 0.2]))

    # Random restriction maps (identity scaled)
    src_map = np.eye(3)
    dst_map = np.eye(3)
    sheaf.set_restriction(("A", "B"), src_map, dst_map)

    # Model tiers attached to nodes
    tiers = {
        "A": ModelTier("qwen-0.5b", 512, "T1"),
        "B": ModelTier("qwen-7b", 7000, "T3"),
    }

    # Edge‑wise Bayesian parameters (dummy values)
    prior_map = {("A", "B"): 0.6}
    likelihood_map = {("A", "B"): 0.7}
    false_pos_map = {("A", "B"): 0.2}

    # Compute hybrid edge weights
    edge_weights = {
        ("A", "B"): hybrid_edge_weight(
            tiers["A"], tiers["B"],
            prior_map[("A", "B")],
            likelihood_map[("A", "B")],
            false_pos_map[("A", "B")]
        )
    }

    # Initialise DAM with dimension = total node dims
    dam = DenseAssociativeMemory(dim=6, seed=42)

    # Regret‑weighted strategy
    regret_strategy = RegretWeightedStrategy(base_lr=0.05)

    # Initial loss
    loss = hybrid_energy(sheaf, dam, edge_weights)
    print(f"Initial hybrid loss: {loss:.4f}")

    # Perform a few hybrid updates
    for i in range(5):
        loss = hybrid_update(
            sheaf,
            dam,
            edge_weights,
            tiers,
            prior_map,
            likelihood_map,
            false_pos_map,
            regret_strategy,
            prev_loss=loss,
        )
        print(f"Iter {i+1} loss: {loss:.4f}")

    # Final sections
    print("Final sections:")
    for node in sorted(node_dims):
        print(f"  {node}: {sheaf.get_section(node)}")