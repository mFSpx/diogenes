# DARWIN HAMMER — match 4001, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_decrea_hybrid_hybrid_hybrid_m1630_s2.py (gen5)
# parent_b: hybrid_hybrid_hybrid_regret_hybrid_hybrid_hybrid_m1086_s0.py (gen4)
# born: 2026-05-29T23:54:24Z

"""HybridFusionAlgorithm
Integrates:
- Parent A (hybrid_hybrid_hybrid_decrea_hybrid_hybrid_hybrid_m1630_s2.py): resource vector eᵢ = [dᵢ, pᵢ, sᵢ, fᵢ] and pruning probability.
- Parent B (hybrid_hybrid_hybrid_regret_hybrid_hybrid_hybrid_m1086_s0.py): regret‑weighted scores, linear state‑space health computation, and tropical (max‑plus) gain mapping.

Mathematical bridge:
The 4‑dimensional resource vector from Parent A is fed as the state vector **xₜ** of a linear
state‑space model (SSM) used in Parent B.  Regret scores computed from actions act as the
control input **uₜ**.  The SSM produces a scalar health score **hₜ = C·xₜ** which
modulates the pruning probability λ in Parent A and also serves as the argument of a
tropical (max‑plus) network that yields gain candidates for updating the regret term.
Thus the two topologies are fused through the common state **xₜ** / resource vector
and a shared scalar health metric."""

import math
import random
import sys
import pathlib
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Tuple
import hashlib
import numpy as np

# ---------- Parent A utilities ----------
def prune_probability(t: float, lam: float = 1.0, alpha: float = 0.2) -> float:
    if t < 0 or lam < 0 or alpha < 0:
        raise ValueError('t, lam, and alpha must be non-negative')
    return min(1.0, lam * math.exp(-alpha * t))

def prune_edges(edges: List[Tuple[int, int]], t: float,
               lam: float = 1.0, alpha: float = 0.2,
               seed: int | str | None = None) -> List[Tuple[int, int]]:
    rng = random.Random(seed)
    p = prune_probability(t, lam, alpha)
    return [e for e in edges if rng.random() >= p]

def haversine_distance(latlon1: Tuple[float, float],
                       latlon2: Tuple[float, float]) -> float:
    """Return distance in metres between two (lat, lon) pairs using haversine."""
    R = 6371000.0  # Earth radius in metres
    lat1, lon1 = map(math.radians, latlon1)
    lat2, lon2 = map(math.radians, latlon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (math.sin(dlat/2)**2 +
         math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2)
    return 2 * R * math.asin(math.sqrt(a))

# ---------- Parent B utilities ----------
def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")

def signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return []
    return [_hash(i, t) for i, t in enumerate(toks)]

@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0

@dataclass(frozen=True)
class MathCounterfactual:
    action_id: str
    outcome_value: float
    probability: float = 1.0

def hybrid_compute_regret_scores(actions: List[MathAction],
                                 counterfactuals: List[MathCounterfactual]) -> List[float]:
    """Regret = expected_value - best counterfactual outcome."""
    scores = []
    for a in actions:
        cf_vals = [c.outcome_value for c in counterfactuals if c.action_id == a.id]
        best_cf = max(cf_vals) if cf_vals else 0.0
        regret = max(0.0, a.expected_value - best_cf)
        scores.append(regret)
    return scores

def tropical_max_plus(vector: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Tropical (max‑plus) transform: y_i = max_j (vector_j + weights_{ij})
    """
    if vector.ndim != 1 or weights.ndim != 2:
        raise ValueError("vector must be 1‑D and weights 2‑D")
    return np.max(vector[:, None] + weights, axis=0)

# ---------- Fusion core ----------
@dataclass
class Entity:
    """Container for the four components required by Parent A."""
    id: int
    location: Tuple[float, float]          # (lat, lon)
    tokens: List[str]                      # for signature collision detection
    decision_score: float                  # sᵢ from decision hygiene
    fisher_score: float                    # fᵢ ∈ [0,1]

def compute_resource_vectors(entities: List[Entity],
                             reference: Tuple[float, float],
                             beta: float = 1.0) -> np.ndarray:
    """
    Build the 4‑D resource matrix E where each row is eᵢ = [dᵢ, pᵢ, sᵢ, fᵢ].

    - dᵢ: haversine distance from reference.
    - pᵢ: β·σᵢ, σᵢ = 1 if the entity's signature collides with any other entity.
    - sᵢ: decision_score (already normalized).
    - fᵢ: fisher_score (clipped to [0,1]).
    """
    n = len(entities)
    E = np.zeros((n, 4), dtype=float)

    # Pre‑compute all signatures to detect collisions
    sig_map: Dict[int, List[int]] = {}
    for ent in entities:
        sig = signature(ent.tokens)
        sig_map[ent.id] = sig

    # Detect collisions (simple set‑intersection count)
    collision_flags = np.zeros(n, dtype=int)
    for i, ei in enumerate(entities):
        for j, ej in enumerate(entities):
            if i >= j:
                continue
            if set(sig_map[ei.id]) & set(sig_map[ej.id]):
                collision_flags[i] = 1
                collision_flags[j] = 1

    for idx, ent in enumerate(entities):
        d = haversine_distance(ent.location, reference)
        p = beta * collision_flags[idx]
        s = ent.decision_score
        f = max(0.0, min(1.0, ent.fisher_score))
        E[idx] = np.array([d, p, s, f], dtype=float)
    return E

def compute_health_scores(E: np.ndarray,
                          actions: List[MathAction],
                          counterfactuals: List[MathCounterfactual],
                          A: np.ndarray | None = None,
                          B: np.ndarray | None = None,
                          C: np.ndarray | None = None) -> np.ndarray:
    """
    Linear state‑space fusion:
        xₜ₊₁ = A·xₜ + B·uₜ
        hₜ   = C·xₜ₊₁
    where:
        xₜ   = resource vector eᵢ (4‑D)
        uₜ   = regret scores (scalar per entity, broadcast)
        A    = state transition (default: identity)
        B    = control influence (default: column of 0.1)
        C    = health read‑out (default: row of ones → sum)
    Returns a 1‑D array of health scores hᵢ ∈ ℝ.
    """
    n = E.shape[0]

    if A is None:
        A = np.eye(4)
    if B is None:
        B = np.full((4, 1), 0.1)   # each dimension receives 0.1·uₜ
    if C is None:
        C = np.ones((1, 4))

    # Regret scores are computed from the provided actions/counterfactuals
    regret_vec = np.array(hybrid_compute_regret_scores(actions, counterfactuals))
    if regret_vec.size != n:
        # If counts differ, broadcast the mean regret to match entity count
        regret_vec = np.full(n, regret_vec.mean() if regret_vec.size else 0.0)

    # Apply the SSM per entity (vectorised)
    x_next = (A @ E.T).T + (B @ regret_vec.reshape(1, -1)).T
    health = (C @ x_next.T).flatten()
    # Optional non‑linearity to keep health in a bounded range
    health = np.tanh(health)
    return health

def hybrid_prune(edges: List[Tuple[int, int]],
                 entities: List[Entity],
                 reference: Tuple[float, float],
                 actions: List[MathAction],
                 counterfactuals: List[MathCounterfactual],
                 t: float,
                 seed: int | str | None = None) -> List[Tuple[int, int]]:
    """
    End‑to‑end hybrid pruning:
    1. Build resource matrix E.
    2. Compute health scores hᵢ.
    3. Derive a dynamic λ = 1 + mean(h) (maps health ∈ (‑1,1) to λ ∈ (0,2)).
    4. Apply Parent A's prune_edges with the dynamic λ.
    5. Use tropical_max_plus on health to obtain gain candidates (illustrative).
    """
    E = compute_resource_vectors(entities, reference)
    health = compute_health_scores(E, actions, counterfactuals)

    # Dynamic pruning intensity
    lam = 1.0 + health.mean()   # health.mean() ∈ (‑1,1) → lam ∈ (0,2)
    pruned = prune_edges(edges, t, lam=lam, alpha=0.2, seed=seed)

    # Tropical gain (not used for pruning here but demonstrates the bridge)
    weight_matrix = np.random.default_rng(0).uniform(-0.5, 0.5, size=(len(health), len(health)))
    gains = tropical_max_plus(health, weight_matrix)
    # For illustration we could log or return gains; here we just ensure the call succeeds.

    return pruned

# ---------- Smoke test ----------
if __name__ == "__main__":
    # Create a tiny synthetic scenario
    ref_loc = (37.7749, -122.4194)  # San Francisco

    ents = [
        Entity(id=0, location=(37.7750, -122.4180),
               tokens=["alpha", "beta"], decision_score=0.8, fisher_score=0.6),
        Entity(id=1, location=(37.7760, -122.4170),
               tokens=["gamma", "delta"], decision_score=0.3, fisher_score=0.2),
        Entity(id=2, location=(37.7740, -122.4200),
               tokens=["alpha", "epsilon"], decision_score=0.5, fisher_score=0.9),
    ]

    # Simple fully‑connected undirected edge list (by indices)
    edge_list = [(0, 1), (0, 2), (1, 2)]

    # Dummy actions / counterfactuals aligned with entities
    actions = [
        MathAction(id="0", expected_value=1.0),
        MathAction(id="1", expected_value=0.4),
        MathAction(id="2", expected_value=0.7),
    ]
    counterfactuals = [
        MathCounterfactual(action_id="0", outcome_value=0.6),
        MathCounterfactual(action_id="1", outcome_value=0.5),
        MathCounterfactual(action_id="2", outcome_value=0.2),
    ]

    # Perform hybrid pruning at time t=5
    pruned_edges = hybrid_prune(edge_list, ents, ref_loc,
                                actions, counterfactuals,
                                t=5.0, seed=42)

    print("Original edges :", edge_list)
    print("Pruned edges   :", pruned_edges)