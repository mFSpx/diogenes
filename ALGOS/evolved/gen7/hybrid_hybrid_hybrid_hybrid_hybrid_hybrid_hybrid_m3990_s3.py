# DARWIN HAMMER — match 3990, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hdc_hy_hybrid_hybrid_nlms_h_m1740_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_krampu_hybrid_hybrid_hybrid_m2003_s2.py (gen6)
# born: 2026-05-29T23:53:05Z

"""Hybrid Hyperdimensional‑Pheromone Bandit Algorithm

Parents:
- hybrid_hybrid_hdc_hybrid_hy_hybrid_hybrid_nlms_h_m1740_s0.py (Algorithm A)
- hybrid_hybrid_hybrid_krampu_hybrid_hybrid_hybrid_m2003_s2.py (Algorithm B)

Mathematical bridge:
Algorithm A provides hyperdimensional primitives **bind** and **bundle** together
with an RBF‑style confidence term `gaussian(r)`.  Algorithm B supplies a
decaying pheromone vector **C** (the “context vector”) and a Hoeffding‑UCB
selection scheme for endpoints with health scores `h_i`.

The fusion treats the decayed pheromone vector as a hyperdimensional binary
vector `p_bin = sign(C)`.  This binary vector is **bound** to a symbolic query
vector `q` (produced by `symbol_vector`) to obtain a context‑aware query
`q' = bind(p_bin, q)`.  The bundle of all endpoint prototype vectors yields a
forecast vector `f`.  The Euclidean distance between `q'` and each endpoint
prototype feeds the RBF confidence `conf_i = gaussian(dist_i)`.  The confidence
modulates the Hoeffding upper‑confidence bound

    U_i = h_i * conf_i + B_i ,

where `B_i` is the Hoeffding bound computed from observed rewards.  The
endpoint with maximal `U_i` is selected, its SSIM similarity to the forecast
updates both the pheromone store and the bandit statistics.

The code below implements this hybrid pipeline with three core functions:
`decay_pheromone`, `contextual_confidence`, and `select_endpoint_ucb`."""
import math
import random
import sys
import threading
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Hyperdimensional primitives (Algorithm A)
# --------------------------------------------------------------------------- #

Vector = List[int]  # binary hyperdimensional vectors (+1 / -1)


def random_vector(dim: int = 10000, seed: str | int | None = None) -> Vector:
    rng = random.Random(seed)
    return [1 if rng.getrandbits(1) else -1 for _ in range(dim)]


def symbol_vector(symbol: str, dim: int = 10000) -> Vector:
    import hashlib

    seed = int.from_bytes(hashlib.sha256(symbol.encode("utf-8")).digest()[:8], "big")
    return random_vector(dim, seed)


def bind(a: Vector, b: Vector) -> Vector:
    """Element‑wise multiplication (binding)."""
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    return [x * y for x, y in zip(a, b)]


def bundle(vectors: List[Vector]) -> Vector:
    """Majority‑vote sum (bundling)."""
    if not vectors:
        raise ValueError("at least one vector is required")
    dim = len(vectors[0])
    if any(len(v) != dim for v in vectors):
        raise ValueError("vectors must have equal length")
    sums = [0] * dim
    for v in vectors:
        for i, x in enumerate(v):
            sums[i] += x
    return [1 if x >= 0 else -1 for x in sums]


def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """RBF‑style confidence decreasing with distance."""
    return math.exp(-((epsilon * r) ** 2))


# --------------------------------------------------------------------------- #
# Pheromone infrastructure and Hoeffding bandit (Algorithm B)
# --------------------------------------------------------------------------- #

FloatVector = List[float]


@dataclass
class PheromoneEntry:
    surface_key: str
    signal_kind: str
    signal_value: float
    half_life_seconds: int
    created_at: datetime = datetime.now(timezone.utc)
    last_decay: datetime = datetime.now(timezone.utc)

    def decay(self, now: datetime | None = None) -> float:
        """Exponential decay according to half‑life."""
        now = now or datetime.now(timezone.utc)
        elapsed = (now - self.last_decay).total_seconds()
        if elapsed <= 0:
            return self.signal_value
        # decay factor = 0.5 ** (elapsed / half_life)
        factor = 0.5 ** (elapsed / self.half_life_seconds)
        self.signal_value *= factor
        self.last_decay = now
        return self.signal_value


def hoeffding_bound(R: float, delta: float, n: int) -> float:
    """Hoeffding bound for bounded reward R ∈ [0, 1]."""
    if n <= 0:
        return float("inf")
    return math.sqrt((R * (1 - R) * math.log(2 / delta)) / n)


def ssim(x: np.ndarray, y: np.ndarray) -> float:
    """Very simple SSIM‑like similarity (range [0,1])."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu_x = x.mean()
    mu_y = y.mean()
    sigma_x = x.var()
    sigma_y = y.var()
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()
    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    return float(numerator / denominator)


# --------------------------------------------------------------------------- #
# Hybrid operations
# --------------------------------------------------------------------------- #


def decay_pheromone(entry: PheromoneEntry) -> FloatVector:
    """
    Decay the pheromone entry and return a normalized float vector that can be
    interpreted as a hyperdimensional context vector C.
    """
    value = entry.decay()
    # Produce a random hyperdimensional vector scaled by the decayed magnitude.
    dim = 10000
    base = random_vector(dim, seed=entry.uuid)
    return [value * x for x in base]  # float‑scaled binary vector


def contextual_confidence(
    query_symbol: str,
    pheromone_vec: FloatVector,
    endpoint_prototypes: List[Vector],
    epsilon: float = 1.0,
) -> Tuple[List[float], Vector]:
    """
    Build a context‑aware query by binding the sign of the pheromone vector with
    a symbolic query vector, then compute RBF confidence for each endpoint
    prototype.  Returns a list of confidences and the bundled forecast vector.
    """
    # Convert pheromone float vector to binary sign vector for binding
    sign_pheromone = [1 if v >= 0 else -1 for v in pheromone_vec]

    # Symbolic query vector
    q = symbol_vector(query_symbol, dim=len(sign_pheromone))

    # Context‑aware query
    q_prime = bind(sign_pheromone, q)

    # Bundle of endpoint prototypes (forecast)
    forecast = bundle(endpoint_prototypes)

    confidences = []
    for proto in endpoint_prototypes:
        # Euclidean distance between bound query and prototype (treated as +/-1)
        dist = euclidean(q_prime, proto)
        conf = gaussian(dist, epsilon=epsilon)
        confidences.append(conf)

    return confidences, forecast


def select_endpoint_ucb(
    endpoint_ids: List[Any],
    health_scores: List[float],
    pheromone_entry: PheromoneEntry,
    endpoint_prototypes: List[Vector],
    rewards: Dict[Any, List[float]],
    delta: float = 0.05,
    epsilon: float = 1.0,
) -> Any:
    """
    Hybrid selection:
    1. Decay pheromone → context vector.
    2. Compute confidence per endpoint via RBF (bound query).
    3. Form Hoeffding upper‑confidence bound using health, confidence and
       observed rewards.
    4. Return the endpoint identifier with maximal UCB.
    """
    # 1. Context vector
    ctx_vec = decay_pheromone(pheromone_entry)

    # 2. Confidence terms
    confidences, _ = contextual_confidence(
        query_symbol="selection_query", pheromone_vec=ctx_vec, endpoint_prototypes=endpoint_prototypes, epsilon=epsilon
    )

    # 3. Compute UCB per endpoint
    ucb_vals = []
    for idx, (eid, h, conf) in enumerate(zip(endpoint_ids, health_scores, confidences)):
        past = rewards.get(eid, [])
        n = len(past)
        avg_reward = sum(past) / n if n > 0 else 0.0
        bound = hoeffding_bound(R=1.0, delta=delta, n=n)  # assume reward in [0,1]
        ucb = h * conf + bound
        ucb_vals.append(ucb)

    # 4. Choose best
    best_idx = int(np.argmax(ucb_vals))
    return endpoint_ids[best_idx]


# --------------------------------------------------------------------------- #
# Simple demo / smoke test
# --------------------------------------------------------------------------- #


def _smoke_test() -> None:
    # Create synthetic endpoints
    num_endpoints = 5
    dim = 10000
    endpoint_ids = [f"ep_{i}" for i in range(num_endpoints)]
    endpoint_prototypes = [random_vector(dim, seed=i) for i in range(num_endpoints)]
    health_scores = [random.uniform(0.5, 1.0) for _ in range(num_endpoints)]

    # Initialise pheromone store
    pheromone = PheromoneEntry(
        surface_key="demo_surface",
        signal_kind="demo_signal",
        signal_value=1.0,
        half_life_seconds=30,
    )

    # Simulated reward history (empty at start)
    rewards: Dict[Any, List[float]] = defaultdict(list)

    # Perform selection
    chosen = select_endpoint_ucb(
        endpoint_ids=endpoint_ids,
        health_scores=health_scores,
        pheromone_entry=pheromone,
        endpoint_prototypes=endpoint_prototypes,
        rewards=rewards,
        delta=0.1,
        epsilon=0.5,
    )
    print(f"Chosen endpoint: {chosen}")

    # Simulate receiving a reward via SSIM similarity to the forecast
    # (use the forecast from the last confidence computation)
    ctx_vec = decay_pheromone(pheromone)
    confidences, forecast = contextual_confidence(
        query_symbol="selection_query", pheromone_vec=ctx_vec, endpoint_prototypes=endpoint_prototypes, epsilon=0.5
    )
    # Compute SSIM between forecast and the chosen prototype
    chosen_idx = endpoint_ids.index(chosen)
    chosen_proto = np.array(endpoint_prototypes[chosen_idx], dtype=float)
    forecast_np = np.array(forecast, dtype=float)
    reward = ssim(chosen_proto, forecast_np)
    rewards[chosen].append(reward)

    # Update pheromone with the new reward (increase signal value proportionally)
    pheromone.signal_value += reward
    print(f"Reward {reward:.4f} added to pheromone; new signal value {pheromone.signal_value:.4f}")


if __name__ == "__main__":
    _smoke_test()