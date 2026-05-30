# DARWIN HAMMER — match 1630, survivor 4
# gen: 5
# parent_a: hybrid_hybrid_decreasing_pr_hybrid_fisher_locali_m215_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_decisi_hybrid_hybrid_bandit_m14_s0.py (gen4)
# born: 2026-05-29T23:38:04Z

"""HybridFusionAlgorithm
Combines:
- Parent A (hybrid_hybrid_decreasing_pr_hybrid_fisher_locali_m215_s0.py): 
  decreasing-rate pruning, epistemic certainty flags, Euclidean edge lengths.
- Parent B (hybrid_hybrid_hybrid_decisi_hybrid_hybrid_bandit_m14_s0.py):
  3‑dimensional resource vector eᵢ = [dᵢ, pᵢ, sᵢ], virtual VRAM store, and a
  weight‑matrix (TTT) that modulates learning rates in a bandit‑style update.

Mathematical bridge:
For each edge (u,v) we define a *composite edge weight*  

    w_uv = ℓ_uv · C_f(epistemic_uv) · ϕ(e_u, e_v) · W_uv

where ℓ_uv is Euclidean distance, C_f maps an epistemic flag to a certainty
factor, ϕ(e_u,e_v)=1+⟨e_u,e_v⟩/‖e‖² is a similarity term derived from the
resource vectors, and W_uv is the current bandit weight.  
The pruning probability uses a decreasing schedule whose parameters
λ(t) and α(t) are scaled by the mean composite weight of the current graph,
thereby coupling the bandit’s learning dynamics to the structural pruning
process.

The virtual VRAM store S(t) integrates observed rewards and modulates the
learning rate η used to update the weight matrix:

    η_eff = η_base · (1 + S(t))

The following implementation provides:
- `compute_resource_vector` – builds eᵢ from geometry, signature collisions,
  and external scores.
- `compute_composite_weights` – evaluates w_uv for every edge.
- `prune_edges_dynamic` – prunes edges with a rate that adapts to the current
  mean composite weight.
- `HybridFusion` – encapsulates the weight matrix, store, and bandit update
  mechanics.
"""

import math
import random
import sys
from pathlib import Path
from collections.abc import Hashable
from typing import Dict, Tuple, List, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Parent A utilities (pruning, distances, epistemic flags)
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS: Tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")
# Certainty factors (higher = more trustworthy)
_EPISTEMIC_CERTAINTY = {
    "FACT": 1.0,
    "PROBABLE": 0.8,
    "POSSIBLE": 0.6,
    "SURE_MAYBE": 0.4,
    "BULLSHIT": 0.2,
}

def length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def prune_probability(t: float, lam: float = 1.0, alpha: float = 0.2) -> float:
    """Decreasing‑rate pruning probability."""
    if t < 0 or lam < 0 or alpha < 0:
        raise ValueError('t, lam, and alpha must be non‑negative')
    return min(1.0, lam * math.exp(-alpha * t))

def prune_edges(edges: List[Hashable], t: float,
                lam: float = 1.0, alpha: float = 0.2,
                seed: int | str | None = None) -> List[Hashable]:
    """Static pruning using a fixed λ and α."""
    rng = random.Random(seed)
    p = prune_probability(t, lam, alpha)
    return [e for e in edges if rng.random() >= p]

# ----------------------------------------------------------------------
# Parent B utilities (resource vector, bandit store, weight matrix)
# ----------------------------------------------------------------------
def haversine_distance(lat1: float, lon1: float,
                       lat2: float, lon2: float) -> float:
    """Great‑circle distance in metres."""
    R = 6371000.0  # Earth radius in metres
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δφ = math.radians(lat2 - lat1)
    Δλ = math.radians(lon2 - lon1)
    a = math.sin(Δφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(Δλ / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))

# ----------------------------------------------------------------------
# Fusion core functions
# ----------------------------------------------------------------------
def compute_resource_vector(
    nodes: Iterable[Hashable],
    positions: Dict[Hashable, Tuple[float, float]],
    reference: Tuple[float, float],
    signatures: Dict[Hashable, str],
    external_scores: Dict[Hashable, float],
    beta: float = 1.0,
) -> Dict[Hashable, np.ndarray]:
    """
    Build the 3‑D resource vector eᵢ = [dᵢ, pᵢ, sᵢ].

    - dᵢ : Euclidean distance from a reference point.
    - pᵢ : β·σᵢ where σᵢ = 1 if the node's signature collides with any other node,
            otherwise 0.
    - sᵢ : External decision‑hygiene score (default 0 if missing).
    """
    # Pre‑compute signature collision map
    sig_counts = {}
    for sig in signatures.values():
        sig_counts[sig] = sig_counts.get(sig, 0) + 1

    resource = {}
    for n in nodes:
        # distance component
        d = length(positions[n], reference)

        # privacy‑load component
        sigma = 1 if sig_counts.get(signatures.get(n, ""), 0) > 1 else 0
        p = beta * sigma

        # decision hygiene score
        s = float(external_scores.get(n, 0.0))

        resource[n] = np.array([d, p, s], dtype=float)

    return resource

def compute_composite_weights(
    edges: List[Tuple[Hashable, Hashable]],
    positions: Dict[Hashable, Tuple[float, float]],
    epistemic_flags: Dict[Tuple[Hashable, Hashable], str],
    weight_matrix: np.ndarray,
    resource_vec: Dict[Hashable, np.ndarray],
) -> Dict[Tuple[Hashable, Hashable], float]:
    """
    Evaluate the composite weight w_uv = ℓ_uv · C_f · ϕ(e_u,e_v) · W_uv
    for every edge.
    """
    # Normalise resource vectors to avoid exploding similarity terms
    all_vecs = np.stack(list(resource_vec.values()))
    norm = np.linalg.norm(all_vecs, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    normed = {k: v / norm[i] for i, (k, v) in enumerate(resource_vec.items())}

    comp_weights = {}
    for (u, v) in edges:
        # Euclidean length
        ℓ = length(positions[u], positions[v])

        # Epistemic certainty factor
        flag = epistemic_flags.get((u, v), "POSSIBLE")
        C_f = _EPISTEMIC_CERTAINTY.get(flag, 0.6)

        # Resource similarity term
        eu = normed[u]
        ev = normed[v]
        similarity = 1.0 + float(np.dot(eu, ev))  # range (0,2]

        # Bandit weight (matrix is indexed by integer node ids; we map via hash)
        idx_u = int(u) if isinstance(u, (int, np.integer)) else hash(u) % weight_matrix.shape[0]
        idx_v = int(v) if isinstance(v, (int, np.integer)) else hash(v) % weight_matrix.shape[1]
        W_uv = float(weight_matrix[idx_u, idx_v])

        comp_weights[(u, v)] = ℓ * C_f * similarity * W_uv

    return comp_weights

def prune_edges_dynamic(
    edges: List[Tuple[Hashable, Hashable]],
    t: float,
    comp_weights: Dict[Tuple[Hashable, Hashable], float],
    base_lam: float = 1.0,
    base_alpha: float = 0.2,
    seed: int | str | None = None,
) -> List[Tuple[Hashable, Hashable]]:
    """
    Dynamic pruning where λ and α are scaled by the mean composite weight.
    Larger mean weight → more aggressive pruning (higher λ, α).
    """
    if not comp_weights:
        return edges.copy()

    mean_w = float(np.mean(list(comp_weights.values())))

    # Scale parameters (simple linear scaling, clamped to reasonable ranges)
    lam = max(0.1, min(5.0, base_lam * (1 + mean_w)))
    alpha = max(0.01, min(1.0, base_alpha * (1 + mean_w)))

    rng = random.Random(seed)
    p = prune_probability(t, lam, alpha)
    return [e for e in edges if rng.random() >= p]

# ----------------------------------------------------------------------
# HybridFusion class – encapsulates bandit dynamics and VRAM store
# ----------------------------------------------------------------------
class HybridFusion:
    """
    Central object that holds:
    - W : weight matrix (TTT style) modulating edge importance.
    - S : virtual VRAM store (scalar) integrating rewards.
    - η : baseline learning rate.
    The store influences η, creating a feedback loop between pruning outcomes
    (rewards) and future edge weighting.
    """

    def __init__(
        self,
        n_nodes: int,
        seed: int = 0,
        base_eta: float = 0.01,
        alpha: float = 1.0,
        beta: float = 1.0,
        dt: float = 1.0,
        store_decay: float = 0.99,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.W = rng.random((n_nodes, n_nodes))  # initialise uniformly in [0,1)
        self.S = 0.0  # virtual VRAM store
        self.base_eta = base_eta
        self.alpha = alpha
        self.beta = beta
        self.dt = dt
        self.store_decay = store_decay

    def effective_eta(self) -> float:
        """Learning rate modulated by the current store."""
        return self.base_eta * (1.0 + self.S)

    def update_store(self, reward: float) -> None:
        """Exponential moving average integration of reward."""
        self.S = self.store_decay * self.S + (1.0 - self.store_decay) * reward

    def bandit_update(
        self,
        edge: Tuple[Hashable, Hashable],
        reward: float,
        positions: Dict[Hashable, Tuple[float, float]],
        epistemic_flags: Dict[Tuple[Hashable, Hashable], str],
        resource_vec: Dict[Hashable, np.ndarray],
    ) -> None:
        """
        Perform a simple gradient‑like update on the weight matrix entry
        corresponding to *edge* using the reward signal.
        """
        # Map hashable identifiers to matrix indices
        n = self.W.shape[0]
        i = int(edge[0]) if isinstance(edge[0], (int, np.integer)) else hash(edge[0]) % n
        j = int(edge[1]) if isinstance(edge[1], (int, np.integer)) else hash(edge[1]) % n

        # Composite weight before the update (used as a scaling factor)
        comp_w = compute_composite_weights(
            [edge],
            positions,
            epistemic_flags,
            self.W,
            resource_vec,
        )[edge]

        # Gradient step: increase weight if reward positive, decrease otherwise
        delta = self.effective_eta() * reward * comp_w
        self.W[i, j] = max(0.0, self.W[i, j] + delta)  # keep non‑negative

        # Update the VRAM store with the same reward (creates the feedback loop)
        self.update_store(reward)

    def get_weights(self) -> np.ndarray:
        """Return a copy of the current weight matrix."""
        return self.W.copy()

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple synthetic graph with 5 nodes
    nodes = list(range(5))
    positions = {
        0: (0.0, 0.0),
        1: (1.0, 0.0),
        2: (0.0, 1.0),
        3: (1.0, 1.0),
        4: (0.5, 0.5),
    }
    edges = [(i, j) for i in nodes for j in nodes if i < j]

    # Random epistemic flags
    rng = random.Random(42)
    epistemic_flags = {e: rng.choice(EPISTEMIC_FLAGS) for e in edges}

    # Dummy signatures (collision on nodes 0 and 1)
    signatures = {i: "sigA" if i < 2 else f"sig{i}" for i in nodes}
    external_scores = {i: rng.random() for i in nodes}

    # Build resource vectors
    reference_point = (0.0, 0.0)
    resource_vec = compute_resource_vector(
        nodes, positions, reference_point, signatures, external_scores, beta=0.5
    )

    # Initialise fusion object
    fusion = HybridFusion(n_nodes=len(nodes), seed=7)

    # Compute composite weights
    comp_weights = compute_composite_weights(
        edges, positions, epistemic_flags, fusion.get_weights(), resource_vec
    )

    # Perform dynamic pruning at time t=3
    pruned = prune_edges_dynamic(edges, t=3.0, comp_weights=comp_weights, seed=123)

    # Simulate a reward (e.g., proportion of edges retained)
    reward = len(pruned) / len(edges)

    # Update bandit for a randomly chosen surviving edge (if any)
    if pruned:
        chosen_edge = random.choice(pruned)
        fusion.bandit_update(
            chosen_edge,
            reward,
            positions,
            epistemic_flags,
            resource_vec,
        )

    # Print summary (no external dependencies)
    print("Initial edge count:", len(edges))
    print("After dynamic pruning:", len(pruned))
    print("Reward fed to bandit:", reward)
    print("Updated weight matrix (rounded):")
    print(np.round(fusion.get_weights(), 3))
    sys.exit(0)