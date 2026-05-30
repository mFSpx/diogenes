# DARWIN HAMMER — match 5573, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m767_s2.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_caputo_fracti_m149_s1.py (gen4)
# born: 2026-05-30T00:02:56Z

"""Hybrid Endpoint‑Health‑Fisher + Caputo‑Fractional Pheromone‑Bandit Fusion
Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m767_s2.py (tropical health → hypervector binding/bundling + Hoeffding decision)
- hybrid_hybrid_hybrid_hybrid_hybrid_caputo_fracti_m149_s1.py (Caputo fractional derivative for pheromone decay + bandit confidence updates)

Mathematical Bridge
-------------------
The scalar health score *h_i* of each endpoint (produced by the tropical engine) is
mapped to a bipolar hypervector *z_i*.  Binding *z_i* with the identifier hypervector
*e_i* yields a bound vector *b_i = bind(e_i, z_i)*.  The majority‑vote bundling of all
*b_i* gives a global representation *V*.  The similarity
`s_i = ⟨V, b_i⟩ / D` (where *D* is the dimensionality) is interpreted as a
pheromone strength for endpoint *i*.  This similarity series is fed to a
discrete Caputo fractional derivative of order α, producing a decay term that
modulates a store‑state dynamics.  The same similarity values serve as stochastic
rewards for a multi‑armed bandit; confidence bounds are updated with Hoeffding‑type
inequalities.  Thus health scoring, hyperdimensional encoding, fractional pheromone
decay, and bandit‑driven allocation are mathematically intertwined in a single
feedback loop.
"""

import math
import random
import sys
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

import numpy as np

# ---------------------------------------------------------------------------
# Hyperdimensional primitives
# ---------------------------------------------------------------------------

D = 1024  # dimensionality of bipolar hypervectors


def _hash_to_bits(data: bytes) -> np.ndarray:
    """Deterministic hash → bipolar vector."""
    h = hashlib.sha256(data).digest()
    bits = np.unpackbits(np.frombuffer(h, dtype=np.uint8))
    # repeat/truncate to D
    repeats = (D + len(bits) - 1) // len(bits)
    bits = np.tile(bits, repeats)[:D]
    return np.where(bits, 1, -1).astype(np.int8)


def scalar_to_hv(value: float) -> np.ndarray:
    """Map a float scalar to a bipolar hypervector."""
    b = value.hex().encode()
    return _hash_to_bits(b)


def symbol_vector(symbol: str) -> np.ndarray:
    """Map a string identifier to a bipolar hypervector."""
    return _hash_to_bits(symbol.encode())


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Binding by element‑wise multiplication (XOR for bipolar)."""
    return a * b


def bundle(vectors: List[np.ndarray]) -> np.ndarray:
    """Majority‑vote bundling (component‑wise sign of sum)."""
    if not vectors:
        return np.zeros(D, dtype=np.int8)
    summed = np.sum(vectors, axis=0)
    return np.where(summed >= 0, 1, -1).astype(np.int8)


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine‑like similarity for bipolar vectors (dot / D)."""
    return float(np.dot(a, b) / D)


# ---------------------------------------------------------------------------
# Tropical health computation (simplified placeholder)
# ---------------------------------------------------------------------------

def hybrid_compute_health_scores(endpoints: List["Endpoint"]) -> List[float]:
    """
    Compute a tropical‑style health score for each endpoint.
    The tropical max‑plus algebra is approximated by:
        h_i = max( w1 * failure_rate , w2 * recovery_priority )
    where w1, w2 are positive constants.
    """
    w1, w2 = 1.5, 1.0
    scores = []
    for ep in endpoints:
        h = max(w1 * ep.failure_rate, w2 * ep.recovery_priority)
        ep.health_score = h
        scores.append(h)
    return scores


# ---------------------------------------------------------------------------
# Store dynamics with Caputo fractional derivative
# ---------------------------------------------------------------------------

_LANCZOS_G = 7
_LANCZOS_C = np.array([
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
])


def gamma_lanczos(z: float) -> float:
    """Lanczos approximation of the Gamma function for z > 0."""
    if z < 0.5:
        return math.pi / (math.sin(math.pi * z) * gamma_lanczos(1 - z))
    z -= 1
    x = _LANCZOS_C[0]
    for i in range(1, len(_LANCZOS_C)):
        x += _LANCZOS_C[i] / (z + i)
    t = z + _LANCZOS_G + 0.5
    return math.sqrt(2 * math.pi) * t ** (z + 0.5) * math.exp(-t) * x


def caputo_derivative(series: List[float], order: float) -> float:
    """
    Discrete Caputo fractional derivative of order `order` for the most recent
    time step.  Uses the definition:
        D^α f(t_n) ≈ (1/Γ(1-α)) * Σ_{k=0}^{n-1} (f_{k+1} - f_k) / (t_n - t_k)^{α}
    Assuming uniform spacing Δt = 1.
    """
    if len(series) < 2:
        return 0.0
    coeff = 1.0 / gamma_lanczos(1.0 - order)
    n = len(series) - 1
    total = 0.0
    for k in range(n):
        diff = series[k + 1] - series[k]
        total += diff / ((n - k) ** order)
    return coeff * total


@dataclass
class StoreState:
    level: float = 0.0
    alpha: float = 1.0   # inflow scaling
    beta: float = 1.0    # outflow scaling
    dt: float = 1.0
    history: List[float] = field(default_factory=list)

    def update(self, inflow: float, outflow: float) -> float:
        """Euler update of the store level."""
        delta = self.alpha * inflow - self.beta * outflow
        self.level = max(0.0, self.level + self.dt * delta)
        return self.level


# ---------------------------------------------------------------------------
# Bandit structures and Hoeffding‑type confidence update
# ---------------------------------------------------------------------------

@dataclass
class BanditAction:
    action_id: str
    propensity: float = 0.5
    expected_reward: float = 0.0
    confidence_bound: float = 1.0
    algorithm: str = "hybrid"


def hoeffding_bound(reward_sum: float, n: int, delta: float = 0.05) -> float:
    """Hoeffding bound for bounded rewards in [0,1]."""
    if n == 0:
        return float('inf')
    mean = reward_sum / n
    radius = math.sqrt(math.log(2 / delta) / (2 * n))
    return mean + radius


def bandit_update(actions: List[BanditAction],
                  rewards: List[float],
                  delta: float = 0.05) -> None:
    """Update each action's expected reward and confidence bound."""
    for act, r in zip(actions, rewards):
        act.expected_reward = (act.expected_reward + r) / 2.0
        act.confidence_bound = hoeffding_bound(r, 1, delta)


# ---------------------------------------------------------------------------
# Core hybrid pipeline
# ---------------------------------------------------------------------------

@dataclass
class Endpoint:
    id: str
    failure_rate: float
    recovery_priority: float
    health_score: float = 0.0


def hybrid_encode_and_bundle(endpoints: List[Endpoint],
                             health_scores: List[float]) -> Tuple[np.ndarray,
                                                                  Dict[str, np.ndarray]]:
    """
    Encode each endpoint into a bound hypervector and bundle them.
    Returns the global bundle V and a dict mapping endpoint id → bound vector.
    """
    bound_vectors = {}
    bound_list = []
    for ep, h in zip(endpoints, health_scores):
        e_vec = symbol_vector(ep.id)
        z_vec = scalar_to_hv(h)
        b_vec = bind(e_vec, z_vec)
        bound_vectors[ep.id] = b_vec
        bound_list.append(b_vec)
    V = bundle(bound_list)
    return V, bound_vectors


def hybrid_fractional_bandit_step(store: StoreState,
                                 V: np.ndarray,
                                 bound_vectors: Dict[str, np.ndarray],
                                 actions: List[BanditAction],
                                 frac_order: float = 0.6) -> None:
    """
    1. Compute similarity for each endpoint (pheromone strength).
    2. Update the store using a Caputo fractional derivative of the similarity series.
    3. Use similarities as stochastic rewards to update bandit actions.
    """
    sims = []
    for act in actions:
        b_vec = bound_vectors.get(act.action_id)
        if b_vec is None:
            sims.append(0.0)
            continue
        s = similarity(V, b_vec)
        sims.append(s)

    # Update store history and compute fractional decay term
    store.history.extend(sims)
    decay = caputo_derivative(store.history, frac_order)

    inflow = sum(sims) / len(sims) if sims else 0.0
    store.update(inflow, decay)

    # Bandit confidence update
    bandit_update(actions, sims)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Create a small set of endpoints
    eps = [
        Endpoint(id="ep1", failure_rate=0.1, recovery_priority=0.8),
        Endpoint(id="ep2", failure_rate=0.4, recovery_priority=0.3),
        Endpoint(id="ep3", failure_rate=0.05, recovery_priority=0.9),
    ]

    # Compute health scores (tropical)
    hs = hybrid_compute_health_scores(eps)

    # Encode into hypervectors and bundle
    V_global, bound_dict = hybrid_encode_and_bundle(eps, hs)

    # Initialise store and bandit actions (one per endpoint)
    store = StoreState(alpha=1.2, beta=0.9, dt=1.0)
    actions = [
        BanditAction(action_id="ep1"),
        BanditAction(action_id="ep2"),
        BanditAction(action_id="ep3"),
    ]

    # Run a few hybrid steps
    for step in range(5):
        # recompute health (could change over time; here static)
        hs = hybrid_compute_health_scores(eps)
        V_global, bound_dict = hybrid_encode_and_bundle(eps, hs)

        hybrid_fractional_bandit_step(store, V_global, bound_dict, actions)

        print(f"Step {step+1}: Store level = {store.level:.4f}")
        for act in actions:
            print(f"  Action {act.action_id}: reward≈{act.expected_reward:.3f}, "
                  f"conf_bound={act.confidence_bound:.3f}")
        print("-" * 40)